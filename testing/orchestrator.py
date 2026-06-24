#!/usr/bin/env python3
"""
orchestrator.py — run ONE benchmark test end to end.

Modes:
  agg       N aggregated EPD workers on GPU (no disagg)
  epd_gpu   E encode workers on GPU + PD prefill/decode workers on GPU
  epd_xpu   E encode workers on the remote B70 XPU + PD workers on GPU

Lifecycle for one test:
  1. choose idle devices from the candidate lists (hard-enforced); skip if the
     placement (instances x TP) doesn't fit.
  2. start control plane, launch workers (local via lib/add_worker.sh,
     remote XPU via lib/add_encoder_xpu.sh).
  3. readiness gate (endpoints + model card + text smoke). retry once, else skip.
  4. run sglang.bench_serving; save JSON + the result text block.
  5. full teardown (always).

Exit codes: 0 ok, 2 skipped (placement/ready), 3 bench failed.
"""

import argparse
import os
import sys
import time

import bench_lib as B

LIB = B.LIB


def _csv(xs):
    return ",".join(str(x) for x in xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["agg", "epd_gpu", "epd_xpu"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--agg-instances", type=int, default=0)
    ap.add_argument("--e-instances", type=int, default=0)
    ap.add_argument("--pd-instances", type=int, default=0)
    ap.add_argument("--num-prompts", type=int, default=32)
    ap.add_argument("--image-count", type=int, default=8)
    ap.add_argument("--image-resolution", default="1080p")
    ap.add_argument("--output-len", type=int, default=256,
                    help="random-output-len; raise to shift work to decode where "
                         "E/PD disagg can win (e.g. 512,1024,2048)")
    ap.add_argument("--input-len", type=int, default=128)
    ap.add_argument("--request-rate", default="1.0")
    ap.add_argument("--max-concurrency", type=int, default=0,
                    help="optional bench_serving --max-concurrency")
    ap.add_argument("--gpus", default="", help="candidate CUDA indices, csv")
    ap.add_argument("--xpus", default="", help="candidate XPU indices, csv (epd_xpu)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mm-attn-backend", default="", help="optional xpu_attn etc.")
    args = ap.parse_args()

    info = B.model_info(args.model)
    tp = info["tp"]
    os.makedirs(args.out_dir, exist_ok=True)

    gpu_cands = B.parse_devs(args.gpus)
    xpu_cands = B.parse_devs(args.xpus)

    # ---- resolve which GPU server we're running on (mgmt/RoCE IP + UCX NIC) ----
    # The shell pieces default these to dell07; pass the resolved profile so the
    # control plane binds and workers dial THIS host. Auto-detected from the
    # hostname, overridable via $GPU_HOST_PROFILE / $IP_LOCAL / $IP_LOCAL_ROCE /
    # $UCX_NIC (see bench_lib.gpu_host_profile).
    host = B.gpu_host_profile()
    B.log(f"GPU host profile: {host['name']} "
          f"(mgmt={host['mgmt_ip']} roce={host['roce_ip']} nic={host['ucx_nic']})")

    # ---- env shared by the shell pieces ----
    env = dict(os.environ)
    env.update({
        "MODEL_PATH": os.environ.get("MODEL_PATH", info["path"]),
        "TP": str(tp),
        "KV_DTYPE": os.environ.get("KV_DTYPE", info["kv"]),
        "MEM_FRAC": os.environ.get("MEM_FRAC", str(info["mem_frac"])),
        "LOG_DIR": os.path.join(args.out_dir, "logs"),
        "IP_LOCAL": host["mgmt_ip"],
        "IP_LOCAL_ROCE": host["roce_ip"],
        "UCX_NIC": host["ucx_nic"],
        # Keep the shell pieces' frontend port in lockstep with bench_lib's client
        # side, so a $PORT_HTTP override (e.g. to dodge a stale :7001) applies to
        # the control plane and teardown too, not just the bench client.
        "PORT_HTTP": str(B.PORT_HTTP),
    })
    # Optional per-model prefill chunk size (bounds prefill activation peak so big
    # multimodal prompts don't OOM the LLM MLP). Absent -> add_worker.sh default.
    if info.get("chunked") and not os.environ.get("CHUNKED_PREFILL"):
        env["CHUNKED_PREFILL"] = str(info["chunked"])
    for key, env_name in (
        ("chat_template", "CHAT_TEMPLATE"),
        ("max_prefill_tokens", "MAX_PREFILL_TOKENS"),
        ("max_total_tokens", "MAX_TOTAL_TOKENS"),
        ("cuda_graph_max_bs", "CUDA_GRAPH_MAX_BS"),
        ("max_running", "MAX_RUNNING"),
        ("pd_prefill_max", "PD_PREFILL_MAX"),
        ("pd_max_running", "PD_MAX_RUNNING"),
        ("pd_mem_frac", "PD_MEM_FRAC"),
        ("enc_mem_frac", "ENC_MEM_FRAC"),
        ("xpu_apply_patches", "XPU_APPLY_PATCHES"),
        ("use_sglang_tokenizer", "USE_SGLANG_TOKENIZER"),
        ("dyn_chat_processor", "DYN_CHAT_PROCESSOR"),
        ("router_mode", "ROUTER_MODE"),
    ):
        if info.get(key) is not None:
            if os.environ.get(env_name):
                continue
            env[env_name] = str(info[key])
    if args.mm_attn_backend:
        env["MM_ATTN_BACKEND"] = args.mm_attn_backend
    os.makedirs(env["LOG_DIR"], exist_ok=True)

    # ---- build the worker plan + label, with hard device enforcement ----
    if args.mode == "agg":
        n = args.agg_instances
        label = f"{n}AGG"
        idle = B.idle_gpus(gpu_cands)
        plan_gpu = B.place(idle, n, tp)
        if plan_gpu is None:
            B.log(f"SKIP {label}: need {n}x{tp}={n*tp} idle GPUs, have {len(idle)} ({idle})")
            return 2
        plan = [("agg", g) for g in plan_gpu]
        n_workers = n
        xpu_plan = []
    else:
        e, pd = args.e_instances, args.pd_instances
        label = f"{e}E{pd}PD"
        idle = B.idle_gpus(gpu_cands)
        # PD always on GPU.
        pd_plan = B.place(idle, pd, tp)
        if pd_plan is None:
            B.log(f"SKIP {label}: need {pd}x{tp} idle GPUs for PD, have {len(idle)} ({idle})")
            return 2
        used = pd * tp
        if args.mode == "epd_gpu":
            # encoders also on GPU (encoder is TP1), from the GPUs left after PD.
            rest = idle[used:]
            e_plan = B.place(rest, e, 1)
            if e_plan is None:
                B.log(f"SKIP {label}: need {e} more idle GPUs for encoders, "
                      f"have {len(rest)} ({rest})")
                return 2
            plan = [("pd", g) for g in pd_plan] + [("encode", g) for g in e_plan]
            xpu_plan = []
            n_workers = pd + e
        else:  # epd_xpu
            ix = B.idle_xpus(xpu_cands)
            if len(ix) < e:
                B.log(f"SKIP {label}: need {e} idle XPUs, have {len(ix)} ({ix})")
                return 2
            plan = [("pd", g) for g in pd_plan]
            xpu_plan = ix[:e]
            n_workers = pd + e

    # ---- run, with one retry on readiness failure ----
    for attempt in (1, 2):
        B.log(f"=== {label} {args.mode} model={args.model} rate={args.request_rate} "
              f"attempt {attempt} ===")
        _teardown(env, gpu_cands)
        _start_controlplane(env)
        _launch_local(env, plan, args.model)
        if xpu_plan and not _launch_xpu(env, xpu_plan, args.model):
            _teardown(env, gpu_cands)
            return 2

        # disagg (epd_*) routes through the encode worker, which rejects
        # text-only input — the smoke request must carry an image there.
        mm = args.mode in ("epd_gpu", "epd_xpu")
        # Big models load slowly: 235B-FP8 takes ~14 min just for weights
        # (~35s/shard x 24) before CUDA-graph capture, so the default 15-min gate
        # has no margin. Default 30 min; override with $READY_TIMEOUT.
        ready_timeout = int(os.environ.get("READY_TIMEOUT", "1800"))
        if B.wait_ready(args.model, n_workers, timeout_s=ready_timeout,
                        do_smoke=True, multimodal=mm):
            break
        B.log(f"  not ready (attempt {attempt})")
        if attempt == 2:
            _teardown(env, gpu_cands)
            B.log(f"SKIP {label}: workers not ready after retry")
            return 2

    # ---- benchmark ----
    safe = label.replace("/", "_")
    out_json = os.path.join(args.out_dir, f"bench_{safe}_r{args.request_rate}.json")
    ok, block = B.run_bench(
        args.model, args.num_prompts, args.image_count, args.image_resolution,
        args.request_rate, out_json, label,
        output_len=args.output_len, input_len=args.input_len,
        max_concurrency=args.max_concurrency or None)
    txt = os.path.join(args.out_dir, f"result_{safe}_r{args.request_rate}.txt")
    with open(txt, "w") as f:
        f.write(block)
    B.log(f"  result block -> {txt}")

    _teardown(env, gpu_cands)
    return 0 if ok else 3


# ---------------- shell-piece wrappers ----------------

def _start_controlplane(env):
    B.sh(["bash", os.path.join(LIB, "start_controlplane.sh")], env=env)


def _launch_local(env, plan, served):
    # port plan: sys/kv/side bases, offset by a per-worker index
    sys_base = int(env.get("SYS_PORT_BASE", "8100"))
    kv_base = int(env.get("KV_EVENT_BASE", "22100"))
    side_base = int(env.get("SIDE_CHANNEL_BASE", "20100"))
    for i, (role, gpus) in enumerate(plan):
        B.sh(["bash", os.path.join(LIB, "add_worker.sh"),
              role, _csv(gpus),
              str(sys_base + i), str(kv_base + i * 3), str(side_base + i),
              served], env=env)
        time.sleep(3)


def _launch_xpu(env, xpus, served):
    e = dict(env)
    e["SYS_PORT_BASE"] = env.get("XPU_SYS_PORT_BASE", "8091")
    e["KV_EVENT_BASE"] = env.get("XPU_KV_EVENT_BASE", "22090")
    e["SIDE_CHANNEL_BASE"] = env.get("XPU_SIDE_CHANNEL_BASE", "20099")
    # The B70 encoder container sees the shared NFS at its real /home path, not
    # the GPU container's /robin mount. Hand it the host path for the logs dir so
    # encoder logs land alongside the GPU worker logs in this test's logs/.
    e["XPU_LOG_DIR"] = B.host_path(env["LOG_DIR"])
    cp = B.sh(["bash", os.path.join(LIB, "add_encoder_xpu.sh"), _csv(xpus), served], env=e)
    launcher_log = os.path.join(env["LOG_DIR"], "xpu_launcher.log")
    with open(launcher_log, "a") as f:
        f.write(f"$ add_encoder_xpu.sh {_csv(xpus)} {served}\n")
        f.write(f"returncode={cp.returncode}\n")
        if cp.stdout:
            f.write("stdout:\n")
            f.write(cp.stdout)
            if not cp.stdout.endswith("\n"):
                f.write("\n")
        if cp.stderr:
            f.write("stderr:\n")
            f.write(cp.stderr)
            if not cp.stderr.endswith("\n"):
                f.write("\n")
    if cp.returncode != 0:
        B.log(f"  XPU launcher failed rc={cp.returncode}; see {launcher_log}")
        return False
    return True


def _teardown(env, gpu_cands):
    e = dict(env)
    e["GPUS"] = _csv(gpu_cands)
    B.sh(["bash", os.path.join(LIB, "teardown.sh")], env=e)


if __name__ == "__main__":
    sys.exit(main())
