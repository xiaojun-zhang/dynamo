"""
bench_lib.py — shared helpers for the E/PD benchmark harness.

Runs in a plain python3 INSIDE the dell07 GPU container (where dynamo.sglang and
sglang.bench_serving exist). It orchestrates the composable shell pieces in
lib/ and talks HTTP to the frontend; it has no torch/sglang imports itself.

Covers:
  - model table (path / served / tensor-parallel size / kv dtype / mem-fraction)
  - device-list parsing + idle checks (local nvidia-smi; remote xpu-smi via ssh)
  - placement: instances x TP must fit the idle candidate list, else skip
  - readiness gate: expected 'generate' endpoint count + /v1/models + text smoke
  - bench run + capture of the "==== Serving Benchmark Result ====" text block
"""

import json
import os
import re
import socket
import subprocess
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "lib")

# ---- deployment facts ----
# Frontend HTTP port. Single source of truth: the orchestrator passes this same
# value to the shell pieces (start_controlplane / teardown) so client and server
# always agree. Overridable via $PORT_HTTP -- bump it to dodge a stale frontend
# squatting on the default (an interrupted run can leave :7001 bound).
PORT_HTTP = int(os.environ.get("PORT_HTTP", "7011"))
XPU_HOST = os.environ.get("XPU_HOST", "172.26.46.180")

WEKA = "/mnt/weka/data/llm-d-models-pv"

# ---- GPU host profiles (the ONLY thing that changes between GPU servers) ----
# A worker's network identity comes from three facts about the host it runs on:
#   mgmt_ip  : management IP the control plane (etcd/NATS) binds + advertises and
#              workers dial for NATS/etcd.
#   roce_ip  : RoCE-fabric IP used as the NIXL side-channel host (E<->PD embedding
#              transfer in epd_gpu mode). Must be a local IP on `ucx_nic`.
#   ucx_nic  : UCX RDMA device:port (UCX_NET_DEVICES).
# Everything else (model table, ports, placement, bench) is host-independent, so
# moving the harness to a new GPU server only needs a profile entry here.
#
# Resolution order (see gpu_host_profile): $GPU_HOST_PROFILE -> hostname
# auto-match -> error. Any single field can be overridden by $IP_LOCAL /
# $IP_LOCAL_ROCE / $UCX_NIC (e.g. to run on a host with no profile entry yet).
GPU_HOSTS = {
    "dell07": {
        "match": ("dell07",),          # substrings tested against the hostname
        "mgmt_ip": "172.26.46.178",
        "roce_ip": "192.165.123.65",
        "ucx_nic": "mlx5_0:1",
    },
    "h200": {
        # sc09super21-h200: 8x H200 143 GB. mlx5_0 (enp25s0np0) is the ACTIVE
        # RoCE port carrying 192.165.123.48 on the shared .123/24 fabric.
        "match": ("h200",),
        "mgmt_ip": "172.26.46.133",
        "roce_ip": "192.165.123.48",
        "ucx_nic": "mlx5_0:1",
    },
}


def gpu_host_profile():
    """Resolve this GPU host's network profile (mgmt_ip / roce_ip / ucx_nic).

    Order: $GPU_HOST_PROFILE names an entry in GPU_HOSTS; otherwise the hostname
    is matched against each profile's `match` substrings. Individual fields are
    overridable by $IP_LOCAL / $IP_LOCAL_ROCE / $UCX_NIC -- which is also the way
    to run on a host that has no profile entry yet. Raises if it can't resolve
    all three, rather than silently falling back to the wrong host."""
    name = os.environ.get("GPU_HOST_PROFILE", "").strip()
    prof = None
    if name:
        if name not in GPU_HOSTS:
            raise SystemExit(
                f"Unknown GPU_HOST_PROFILE '{name}'. Known: {', '.join(GPU_HOSTS)}")
        prof = GPU_HOSTS[name]
    else:
        host = socket.gethostname().lower()
        for key, p in GPU_HOSTS.items():
            if any(m in host for m in p["match"]):
                prof, name = p, key
                break
    base = prof or {}
    mgmt = os.environ.get("IP_LOCAL") or base.get("mgmt_ip")
    roce = os.environ.get("IP_LOCAL_ROCE") or base.get("roce_ip")
    nic = os.environ.get("UCX_NIC") or base.get("ucx_nic")
    if not (mgmt and roce and nic):
        raise SystemExit(
            f"Could not resolve a GPU host profile (hostname="
            f"'{socket.gethostname()}'). Set $GPU_HOST_PROFILE to one of "
            f"{{{', '.join(GPU_HOSTS)}}}, or set $IP_LOCAL / $IP_LOCAL_ROCE / "
            f"$UCX_NIC explicitly.")
    return {"name": name or "custom", "mgmt_ip": mgmt,
            "roce_ip": roce, "ucx_nic": nic}

# ---- model table: tp = GPUs per instance, kv = kv-cache dtype ----
MODELS = {
    "Qwen/Qwen3-VL-8B-Instruct": {
        "path": f"{WEKA}/models--Qwen--Qwen3-VL-8B-Instruct",
        "tp": 1, "kv": "auto", "mem_frac": 0.70},
    "Qwen/Qwen3-VL-32B-Instruct-FP8": {
        # On one L40S the 32B-FP8 weights take ~33 GB, leaving only ~11 GB to split
        # between KV cache and the prefill working set -- and those pull opposite
        # ways. The 8-image/1080p workload is ~16.4k prompt tokens:
        #   - mem_frac too low (0.80) -> KV cache only ~14k tokens < prompt ->
        #     every request aborts ("Multimodal prompt is too long").
        #   - mem_frac too high (0.92) -> KV fits but runtime OOMs.
        # mem_frac 0.85 gives KV ~32k tokens (clears the prompt) with ~5.7 GB free.
        # But prefilling all ~16k tokens in ONE chunk then OOMs the LLM MLP
        # activation (~800 MiB short). chunked 8192 halves that prefill peak so it
        # fits, without shrinking KV. (Models w/o "chunked" default to 16384.)
        "path": f"{WEKA}/models--Qwen--Qwen3-VL-32B-Instruct-FP8",
        "tp": 1, "kv": "fp8_e4m3", "mem_frac": 0.85, "chunked": 8192},
    "Qwen/Qwen3-VL-32B-Instruct": {
        # bf16 ~64 GB / TP2 = ~32 GB/GPU weights on 44 GB L40S. mem_frac 0.85 left
        # ~0 free at CUDA-graph capture -> OOM. 0.78 leaves room for the graph +
        # activation working set; chunked 8192 caps the 8-image (~16k tok) prefill.
        "path": f"{WEKA}/models--Qwen--Qwen3-VL-32B-Instruct",
        "tp": 2, "kv": "auto", "mem_frac": 0.78, "chunked": 8192},
    "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8": {
        # Numerically valid TP is {1,2,4} only:
        #   - 16 attention heads -> TP must divide 16 (TP6 aborts "16 not
        #     divisible by 6").
        #   - FP8 block_n=128 quant -> each shard must stay 128-divisible; TP8
        #     gives a gate/up shard of 192 and aborts ("192 not divisible by 128").
        # ~222 GB weights need >= 5 cards on 46 GB L40S, but TP{5,6,7} are invalid
        # and TP8's shard isn't 128-divisible, so it does NOT run on L40S at all.
        # On H200 (143 GB) it fits at TP4: ~55 GB/GPU weights on 4 cards. mem_frac
        # 0.90 leaves ample room for KV + the chunked-8192 prefill working set.
        # chunked 32768: on H200 (~91 GB free/GPU after the ~59 GB TP4 weights)
        # there's no activation-OOM pressure, so the prefill token budget can hold
        # ~2 full 8-image/1080p prompts (~16k tok each) per step -> prefill can
        # batch >1 request instead of splitting a single prompt across chunks.
        "path": f"{WEKA}/models--Qwen--Qwen3-VL-235B-A22B-Instruct-FP8",
        "tp": 4, "kv": "fp8_e4m3", "mem_frac": 0.90, "chunked": 32768},
}


def model_info(served):
    if served not in MODELS:
        raise SystemExit(f"Unknown model '{served}'. Known: {', '.join(MODELS)}")
    return MODELS[served]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# /home is a shared Weka NFS mount visible on BOTH the dell07 GPU host and the
# B70 XPU host. The orchestrator runs inside the GPU container where ~/robin is
# mounted at /robin, but the B70 encoder container sees the same files at their
# real NFS path under /home. host_path() translates a container path to that
# real path so XPU encoders can write logs straight into the shared results dir.
CONTAINER_ROOT = os.environ.get("HARNESS_CONTAINER_ROOT", "/robin")
HOST_ROOT = os.environ.get("HARNESS_HOST_ROOT", "/home/h-zheng/robin")


def host_path(p):
    """Map a container path under CONTAINER_ROOT to its real NFS path under
    HOST_ROOT. If it isn't under CONTAINER_ROOT, return it unchanged (assume it
    is already a shared/absolute path both hosts can see)."""
    ap = os.path.abspath(p)
    if ap == CONTAINER_ROOT or ap.startswith(CONTAINER_ROOT + "/"):
        return HOST_ROOT + ap[len(CONTAINER_ROOT):]
    return ap


def sh(cmd_list, env=None, timeout=None):
    """Run a command list; return CompletedProcess (text captured)."""
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run(cmd_list, capture_output=True, text=True,
                          env=e, timeout=timeout)


# ---------------- device lists + idle checks ----------------

def parse_devs(s):
    if not s:
        return []
    return [int(x) for x in str(s).replace(" ", "").split(",") if x != ""]


def _gpu_used_mib(idx):
    cp = sh(["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", "-i", str(idx)])
    if cp.returncode != 0:
        return None
    try:
        return int(cp.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def idle_gpus(candidates, max_used_mib=1000):
    """Subset of candidate GPU indices that are idle. Unknown -> excluded."""
    free = []
    for g in candidates:
        used = _gpu_used_mib(g)
        if used is None:
            log(f"  WARN gpu {g}: nvidia-smi unreadable -> excluding")
        elif used < max_used_mib:
            free.append(g)
        else:
            log(f"  gpu {g}: {used} MiB used -> busy, excluding")
    return free


def _ssh_opts():
    """SSH options for reaching the XPU host with key auth.
    -F /dev/null skips ~/.ssh/config (in the GPU container that file is mounted
    from the host with non-root owner / group-writable perms, which ssh rejects
    with "Bad owner or permissions"). -i names the key explicitly. XPU_SSH_KEY
    overrides the default key path."""
    key = os.environ.get("XPU_SSH_KEY", "/root/.ssh/id_ed25519")
    return ["-F", "/dev/null", "-i", key,
            "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes"]


def _ssh_xpu(remote_cmd):
    """SSH to the XPU host via key auth (set up the key for XPU_SSH_USER@XPU_HOST)."""
    opts = _ssh_opts()
    user = os.environ.get("XPU_SSH_USER", "h-zheng")
    target = f"{user}@{XPU_HOST}"
    cmd = ["ssh", *opts, target, remote_cmd]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30)


def _xpu_used_mib(idx):
    """Parse 'GPU Memory Used (MiB)' for one remote XPU device via xpu-smi stats.
    Returns int MiB, or None if unreadable. Uses the per-device stats form (the
    `dump -d -1` multi-device form can hang a non-interactive shell)."""
    try:
        cp = _ssh_xpu(
            f"xpu-smi stats -d {idx} 2>/dev/null | grep -i 'GPU Memory Used'")
    except Exception:
        return None
    if cp.returncode != 0 or not cp.stdout.strip():
        return None
    # Lines look like: "| GPU Memory Used (MiB)        | 1234 |" (a card may show
    # multiple tiles, e.g. "1234, 5678"); take the max number on matched lines.
    nums = [int(n) for n in re.findall(r"\d+", cp.stdout)]
    # The "(MiB)" literal contributes no stray digits; if a tile count sneaks in
    # it will be small relative to real memory, so max() is a safe upper bound.
    return max(nums) if nums else None


def idle_xpus(candidates, max_used_mib=1000):
    """Subset of candidate XPU indices that are idle on the remote B70.
    Unreadable / busy -> EXCLUDED (never run on an unverifiable device)."""
    free = []
    for x in candidates:
        used = _xpu_used_mib(x)
        if used is None:
            log(f"  WARN xpu {x}: xpu-smi stats unreadable -> excluding")
        elif used < max_used_mib:
            free.append(x)
        else:
            log(f"  xpu {x}: {used} MiB used -> busy, excluding")
    return free


def place(idle_devices, instances, tp):
    """List of `instances` device-tuples of length `tp`, or None if won't fit."""
    if instances * tp > len(idle_devices):
        return None
    return [idle_devices[i * tp:(i + 1) * tp] for i in range(instances)]


# ---------------- frontend HTTP ----------------

def _get(path, timeout=3):
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{PORT_HTTP}{path}", timeout=timeout) as r:
            return r.read().decode()
    except Exception:
        return None


def n_generate_endpoints():
    body = _get("/health")
    return len(re.findall(r'"endpoint":"generate"', body)) if body else 0


def model_listed(served):
    body = _get("/v1/models")
    return bool(body) and served in body


# A 64x64 white PNG as a data URL — a small but real-sized image to exercise the
# encode->PD path. In disagg (epd) mode a text-only request hits the encode
# worker and 500s with "multi_modal_data is required for the encode worker", so
# the smoke MUST include an image there. (1x1 can trip preprocessor minimums, so
# we use 64x64.)
_TINY_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAS0lEQVR42u3PMQ0AAAwDoPo3"
    "3UrYvQQckD4XAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEB"
    "AYHLAMpT0sIcNbcEAAAAAElFTkSuQmCC"
)


def smoke(served, multimodal, timeout=120):
    """One tiny request to confirm the pipeline generates. If multimodal=True
    (disagg E/PD), include an image so it flows through the encode worker."""
    if multimodal:
        content = [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": _TINY_PNG_DATA_URL}},
        ]
    else:
        content = "hi"
    payload = json.dumps({"model": served,
                          "messages": [{"role": "user", "content": content}],
                          "max_tokens": 4}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT_HTTP}/v1/chat/completions",
        data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception as e:
        log(f"  smoke failed: {e}")
        return False


def wait_ready(served, n_workers, timeout_s=900, do_smoke=True, multimodal=False):
    """Block until n_workers 'generate' endpoints + model card + smoke ok.
    multimodal=True (disagg E/PD) sends an image in the smoke request."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        n = n_generate_endpoints()
        if n >= n_workers and model_listed(served):
            if not do_smoke or smoke(served, multimodal):
                log(f"  READY: {n}/{n_workers} endpoints, model listed"
                    + (", smoke ok" if do_smoke else ""))
                return True
        time.sleep(5)
        log(f"  waiting... {n}/{n_workers} generate endpoints")
    log(f"  TIMEOUT {timeout_s}s ({n_generate_endpoints()}/{n_workers})")
    return False


# ---------------- benchmark ----------------

RESULT_HDR = re.compile(r"=+ Serving Benchmark Result")


def run_bench(served, num_prompts, image_count, image_res, rate,
              out_json, label, output_len=256, input_len=128, timeout_s=2400):
    """Run sglang.bench_serving; return (ok, result_text_block).
    result_text_block is the captured '==== Serving Benchmark Result ===='
    section (with the label injected), or '' if not found.

    output_len/input_len are exposed so the workload can be shifted between
    decode-heavy (long output) and prefill/vision-heavy (many images, short
    output). For E/PD disagg the dominant benefit is queue reduction once the
    aggregated baseline saturates -- not output length per se (the reference
    workloads win at output_len=256)."""
    cmd = [
        "python3", "-m", "sglang.bench_serving",
        "--model", served, "--backend", "sglang-oai-chat",
        "--host", "127.0.0.1", "--port", str(PORT_HTTP),
        "--dataset-name", "image",
        "--num-prompts", str(num_prompts),
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--image-count", str(image_count),
        "--image-resolution", str(image_res),
        "--request-rate", str(rate),
        "--apply-chat-template", "--seed", "0",
        "--disable-tqdm",   # no CR progress bars: keeps captured logs clean
        "--output-file", out_json,
    ]
    log(f"  bench: np={num_prompts} img={image_count} res={image_res} rate={rate}")
    try:
        cp = sh(cmd, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        log(f"  bench TIMEOUT after {timeout_s}s")
        return False, ""
    out = (cp.stdout or "") + "\n" + (cp.stderr or "")
    block = _extract_result_block(out, label)
    ok = cp.returncode == 0 and "Successful requests" in out
    if ok and _all_requests_aborted(out_json):
        # bench_serving exits 0 and prints a result block even when the server
        # aborts every request (e.g. "Multimodal prompt is too long" when the KV
        # cache is smaller than the expanded prompt). The tell is that the
        # RE-tokenized output (actual text returned) is ~1 token/request while the
        # server-reported token count looks normal. Such a run is garbage, not ok.
        log("  bench: all requests aborted (no real output) -> FAIL "
            "(check worker log for 'Multimodal prompt is too long' / aborts)")
        ok = False
    if not ok:
        log(f"  bench rc={cp.returncode} (see captured log)")
    return ok, (block or out[-4000:])


def _all_requests_aborted(out_json):
    """True if the bench JSON shows requests 'completed' but with essentially no
    real generated text (retokenized output ~= request count, i.e. <=1 tok/req).
    That's the signature of server-side aborts, not a genuine benchmark."""
    try:
        with open(out_json) as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False  # no/garbled JSON -> let the existing rc check decide
    completed = d.get("completed") or 0
    retok = d.get("total_output_tokens_retokenized")
    if completed <= 0 or retok is None:
        return False
    # Healthy runs generate many tokens/request; aborted runs return ~1 (just the
    # finish marker). Flag when retokenized output is at most ~1 token per request.
    return retok <= completed


def _extract_result_block(text, label):
    """Pull the '==== Serving Benchmark Result ====' ... '====' block and
    rewrite the header to include our topology label (e.g. 1E4PD)."""
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if RESULT_HDR.search(ln):
            start = i
            break
    if start is None:
        return ""
    end = start + 1
    while end < len(lines) and not re.match(r"^=+$", lines[end].strip()):
        end += 1
    block = lines[start:end + 1]
    if block:
        block[0] = f"============ Serving Benchmark Result: {label} ============"
    return "\n".join(block) + "\n"
