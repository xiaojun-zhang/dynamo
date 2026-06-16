#!/bin/bash
# Launch the dell07 GPU container that the benchmark harness runs inside.
# Run this ON THE dell07 HOST (as h-zheng, not via sudo, not inside a container).
#
# Notes:
#   - Default docker runtime here is runc, so GPUs are injected via --gpus.
#   - The host has 8 L40S boards; one is broken (PCI 0001:8f:00.0 == /dev/nvidia6,
#     reports "Unknown Error"). nvidia-smi/torch hide it via NVML, so the container
#     looks fine (7 GPUs) -- but the moment a worker sets CUDA_VISIBLE_DEVICES, the
#     CUDA runtime scans /dev/nvidia* , trips on the dead node, and aborts every
#     worker with "Error 101: invalid device ordinal / No accelerator available".
#   - `--gpus "device=<UUIDs>"` is NOT enough here: this container is --privileged,
#     which bind-mounts ALL host /dev nodes (including the dead /dev/nvidiaN)
#     regardless of the --gpus selection. So we ALSO delete the dead node's device
#     file inside the container after launch (computed below, not hardcoded). After
#     that, container indices 0-6 map to the 7 good L40S, matching the harness's
#     `--gpus 0,1,2,3,4,5,6`.
#   - ~/.ssh is mounted read-only for key-auth SSH to the B70 XPU host (case 3).
set -e

# Healthy GPU UUIDs only (nvidia-smi lists just the working boards, dropping the
# broken one). Bail if we can't read them rather than fall back to --gpus all.
HEALTHY_GPUS=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd,)
[ -n "$HEALTHY_GPUS" ] || { echo "ERROR: could not read healthy GPU UUIDs from nvidia-smi"; exit 1; }
echo "Pinning container to healthy GPUs: $HEALTHY_GPUS"

# Find the /dev/nvidiaN minor(s) of any board the driver enumerates in
# /proc/driver/nvidia/gpus but nvidia-smi does NOT list (i.e. broken boards).
# These device nodes poison CUDA runtime init and must be removed in-container.
DEAD_MINORS=""
for info in /proc/driver/nvidia/gpus/*/information; do
    # Field values are tab/space-padded; strip ALL whitespace, not just spaces.
    uuid=$(awk -F'GPU UUID:' '/GPU UUID:/{gsub(/[[:space:]]/,"",$2); print $2}' "$info")
    minor=$(awk -F'Device Minor:' '/Device Minor:/{gsub(/[[:space:]]/,"",$2); print $2}' "$info")
    [ -n "$uuid" ] && [ -n "$minor" ] || continue
    if ! printf '%s' "$HEALTHY_GPUS" | grep -qF "$uuid"; then
        echo "Broken board detected: UUID=$uuid -> /dev/nvidia$minor (will remove in container)"
        DEAD_MINORS="$DEAD_MINORS $minor"
    fi
done

docker rm -f robin_sglang_dynamo_l40 2>/dev/null || true

docker run -dit --rm \
  --privileged \
  --gpus "\"device=$HEALTHY_GPUS\"" \
  --network=host --ipc=host \
  $(env | grep -i _proxy | sed 's/^/-e /') \
  --user root --group-add video --cap-add=SYS_ADMIN \
  --mount type=bind,source=/dev/dri/by-path,target=/dev/dri/by-path \
  --mount type=bind,source=/sys,target=/sys \
  --mount type=bind,source=/dev/bus,target=/dev/bus \
  --mount type=bind,source=/dev/char,target=/dev/char \
  --mount type=bind,source=/dev/infiniband,target=/dev/infiniband \
  -v ~/hongming:/hongming \
  -v ~/robin:/robin \
  -v /mnt/weka:/mnt/weka \
  -v ~/.ssh:/root/.ssh:ro \
  --name robin_sglang_dynamo_l40 \
  --entrypoint /bin/bash \
  -w /robin/dynamo/testing \
  amr-registry.caas.intel.com/taas/scalable-deploy-intel/main_dockerfile.dynamo_gpu:422-9e23364

# Remove the broken board's device node INSIDE the container. --privileged leaks
# every host /dev/nvidiaN in regardless of --gpus, and the dead node poisons CUDA
# runtime init (Error 101) for any worker that sets CUDA_VISIBLE_DEVICES.
for m in $DEAD_MINORS; do
    docker exec robin_sglang_dynamo_l40 bash -lc "rm -f /dev/nvidia$m" \
        && echo "Removed dead /dev/nvidia$m inside container"
done

# Install CuPy so NIXL moves embeddings GPU-side. Without it, dynamo.nixl_connect
# logs "Failed to load CuPy ... utilizing numpy" and every E->PD embedding
# transfer is staged through host memory (Device->Host->...->Device), adding
# ~hundreds of ms/request on the critical path -- which masks any disagg benefit.
# This container is --rm (recreated each launch), so the install must live here,
# not be done by hand. Image is CUDA 12.8 -> cupy-cuda12x. Idempotent: skips if
# already importable. Non-fatal so a pip/proxy hiccup doesn't block the container.
echo "Ensuring CuPy (cupy-cuda12x) is present for GPU-side NIXL transfers..."
docker exec robin_sglang_dynamo_l40 bash -lc \
  'python3 -c "import cupy" 2>/dev/null && { echo "  cupy already present"; exit 0; }; \
   pip install --no-cache-dir cupy-cuda12x \
     && python3 -c "import cupy; print(\"  cupy\", cupy.__version__, \"OK\")" \
     || echo "  WARN: cupy install failed -- disagg NIXL will fall back to CPU/numpy"'

# Patch SGLang's Qwen3-VL MoE weight loader so the ENCODE-only worker can load
# Qwen3-VL-235B-A22B (MoE). In encoder-only mode the model has no self.model
# (qwen3_vl.py builds it only when NOT encoder_only), but the MoE load_weights
# does `hasattr(self.model, "start_layer")` -- Python evaluates self.model BEFORE
# hasattr, so nn.Module.__getattr__ raises AttributeError and the encode worker
# dies at load ("'Qwen3VLMoeForConditionalGeneration' object has no attribute
# 'model'"). Dense 8B/32B encoders are unaffected. Fix: guard self.model's
# existence first. Same --rm reasoning as CuPy above -- must re-apply per launch.
# Idempotent (skips if already guarded); non-fatal; only touches the one pattern.
echo "Patching SGLang Qwen3-VL MoE loader for encoder-only 235B (idempotent)..."
# NB: -i is REQUIRED -- without it `docker exec` does not attach stdin, so the
# heredoc never reaches `python3 -`, the script reads empty input, does nothing,
# and exits 0 (so even the `|| WARN` below stays silent). That's exactly how an
# earlier launch printed the header then patched nothing.
docker exec -i robin_sglang_dynamo_l40 python3 - <<'PYEOF' || echo "  WARN: SGLang MoE encoder patch failed -- 235B E/PD encode will crash at load"
old = 'and hasattr(self.model, "start_layer")'
new = 'and hasattr(self, "model") and hasattr(self.model, "start_layer")'
for f in ("/opt/sglang/python/sglang/srt/models/qwen3_vl_moe.py",
          "/opt/sglang/python/sglang/srt/models/qwen3_vl.py"):
    try:
        s = open(f).read()
    except OSError as e:
        print(f"  skip {f}: {e}"); continue
    if new in s:
        print(f"  already patched: {f}")
    elif old in s:
        open(f, "w").write(s.replace(old, new))
        print(f"  patched: {f}")
    else:
        print(f"  WARN pattern absent (sglang version changed?): {f}")
PYEOF

echo ""
echo "Started container 'robin_sglang_dynamo_l40'. Verify GPUs + CUDA:"
echo "  docker exec -it robin_sglang_dynamo_l40 bash -lc \\"
echo "    'nvidia-smi --query-gpu=index,name --format=csv,noheader; \\"
echo "     python3 -c \"import torch; print(torch.cuda.device_count())\"'"
echo ""
echo "Then work inside it:"
echo "  docker exec -it robin_sglang_dynamo_l40 bash"
echo "  cd /robin/dynamo/testing"
