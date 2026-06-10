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
    ap.add_argument("--request-rate", default="1.0")
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

    # ---- env shared by the shell pieces ----
    env = dict(os.environ)
    env.update({
        "MODEL_PATH": info["path"], "TP": str(tp),
        "KV_DTYPE": info["kv"], "MEM_FRAC": str(info["mem_frac"]),
        "LOG_DIR": os.path.join(args.out_dir, "logs"),
    })
    # Optional per-model prefill chunk size (bounds prefill activation peak so big
    # multimodal prompts don't OOM the LLM MLP). Absent -> add_worker.sh default.
    if info.get("chunked"):
        env["CHUNKED_PREFILL"] = str(info["chunked"])
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
        if xpu_plan:
            _launch_xpu(env, xpu_plan, args.model)

        # disagg (epd_*) routes through the encode worker, which rejects
        # text-only input — the smoke request must carry an image there.
        mm = args.mode in ("epd_gpu", "epd_xpu")
        if B.wait_ready(args.model, n_workers, timeout_s=900,
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
        args.request_rate, out_json, label)
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
    sys_base, kv_base, side_base = 8100, 22100, 20100
    for i, (role, gpus) in enumerate(plan):
        B.sh(["bash", os.path.join(LIB, "add_worker.sh"),
              role, _csv(gpus),
              str(sys_base + i), str(kv_base + i * 3), str(side_base + i),
              served], env=env)
        time.sleep(3)


def _launch_xpu(env, xpus, served):
    e = dict(env)
    e["SYS_PORT_BASE"] = "8091"
    e["KV_EVENT_BASE"] = "22090"
    e["SIDE_CHANNEL_BASE"] = "20099"
    # The B70 encoder container sees the shared NFS at its real /home path, not
    # the GPU container's /robin mount. Hand it the host path for the logs dir so
    # encoder logs land alongside the GPU worker logs in this test's logs/.
    e["XPU_LOG_DIR"] = B.host_path(env["LOG_DIR"])
    B.sh(["bash", os.path.join(LIB, "add_encoder_xpu.sh"), _csv(xpus), served], env=e)


def _teardown(env, gpu_cands):
    e = dict(env)
    e["GPUS"] = _csv(gpu_cands)
    B.sh(["bash", os.path.join(LIB, "teardown.sh")], env=e)


if __name__ == "__main__":
    sys.exit(main())
