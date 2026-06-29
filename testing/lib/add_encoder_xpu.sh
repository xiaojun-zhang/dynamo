#!/bin/bash
# Composable piece: launch XPU encode worker(s) on the remote B70 host.
# Runs from the dell07 orchestrator; SSHes to the XPU host, docker-runs a FRESH
# container, then docker-execs one encode worker per XPU device.
#
# The encoders register into THIS host's (dell07's) etcd/nats over the network
# and send embeddings to the PD workers via NIXL (ze_copy on the XPU side).
#
# Usage:
#   add_encoder_xpu.sh <xpu_devices_csv> <model_served>
#     xpu_devices_csv : e.g. "0,1"  (one encoder per device)
#     model_served    : Qwen/Qwen3-VL-8B-Instruct
#
# Env:
#   MODEL_PATH (required), XPU_IMAGE, XPU_CONTAINER,
#   IP_LOCAL (dell07 mgmt), PORT_NATS, PORT_ETCD, TRANSFER_MODE,
#   SIDE_CHANNEL_BASE, KV_EVENT_BASE, SYS_PORT_BASE,
#   XPU_LOG_DIR (shared-NFS /home path for encoder logs; falls back to /tmp)
set -e

XPUS_CSV="$1"; SERVED="$2"
[ -z "$SERVED" ] && { echo "usage: add_encoder_xpu.sh <xpu_csv> <served>"; exit 2; }

XPU_HOST="${XPU_HOST:-172.26.46.180}"
XPU_HOST_PROFILE="${XPU_HOST_PROFILE:-}"
XPU_USER="${XPU_SSH_USER:-h-zheng}"
XPU_IMAGE="${XPU_IMAGE:-hm_dynamo_b70_pr26460:latest}"
XPU_CONTAINER="${XPU_CONTAINER:-harness_b70_enc}"
IP_LOCAL="${IP_LOCAL:-172.26.46.178}"
PORT_NATS="${PORT_NATS:-14222}"
PORT_ETCD="${PORT_ETCD:-12379}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be set}"
TRANSFER_MODE="${TRANSFER_MODE:-nixl-read}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-qwen2-vl}"
LOG_LEVEL="${LOG_LEVEL:-debug}"
USE_SGLANG_TOKENIZER="${USE_SGLANG_TOKENIZER:-0}"
ENC_MEM_FRAC="${ENC_MEM_FRAC:-0.5}"
ENC_EXTRA_ARGS="${ENC_EXTRA_ARGS:-}"
XPU_APPLY_PATCHES="${XPU_APPLY_PATCHES:-0}"
XPU_PATCH_SERVER_ARGS="${XPU_PATCH_SERVER_ARGS:-0}"
XPU_PATCH_DIR="${XPU_PATCH_DIR:-/home/h-zheng/robin/dynamo/testing/container_patch_work}"
XPU_INSTALL_SGL_KERNEL_XPU="${XPU_INSTALL_SGL_KERNEL_XPU:-0}"
XPU_SGL_KERNEL_XPU_REPO="${XPU_SGL_KERNEL_XPU_REPO:-https://github.com/sgl-project/sgl-kernel-xpu}"
SIDE_CHANNEL_BASE="${SIDE_CHANNEL_BASE:-20099}"
KV_EVENT_BASE="${KV_EVENT_BASE:-22090}"
SYS_PORT_BASE="${SYS_PORT_BASE:-8091}"
detect_xpu_profile() {
    if [ -n "$XPU_HOST_PROFILE" ]; then
        printf '%s\n' "$XPU_HOST_PROFILE"
        return
    fi
    case "$(printf '%s' "$XPU_HOST" | tr '[:upper:]' '[:lower:]')" in
        *b60*|*intel02*|172.26.46.171) printf '%s\n' "b60" ;;
        *) printf '%s\n' "b70" ;;
    esac
}

XPU_PROFILE="$(detect_xpu_profile)"

# XPU encoders use the image's custom multimodal attention backend by default.
# If MM_ATTN_BACKEND is explicitly set to an empty string, the launcher omits
# --mm-attention-backend and lets SGLang choose its default.
if [ "${MM_ATTN_BACKEND+x}" != "x" ]; then
    MM_ATTN_BACKEND="xpu_attn"
fi

resolve_xpu_fabric() {
    local xpu="$1"
    case "$XPU_PROFILE" in
        b60)
            if [ "$xpu" -gt 3 ]; then
                echo "[xpu] ERROR: B60 profile only has XPU ids 0-3, got $xpu" >&2
                exit 2
            fi
            if [ "$xpu" -lt 2 ]; then
                NIC="mlx5_0:1"; XIP="192.165.123.64"
            else
                NIC="mlx5_1:1"; XIP="192.165.123.70"
            fi
            ;;
        b70)
            if [ "$xpu" -gt 7 ]; then
                echo "[xpu] ERROR: B70 profile only has XPU ids 0-7, got $xpu" >&2
                exit 2
            fi
            # Historical B70 mapping used by this harness.
            if [ "$xpu" -lt 4 ]; then
                NIC="mlx5_0:1"; XIP="192.165.123.40"
            else
                NIC="mlx5_2:1"; XIP="192.165.123.37"
            fi
            ;;
        *)
            echo "[xpu] ERROR: unknown XPU_HOST_PROFILE=$XPU_PROFILE" >&2
            exit 2
            ;;
    esac
}

# Key-based SSH only (set up the key for ${XPU_USER}@${XPU_HOST}). BatchMode
# makes a missing/wrong key fail fast instead of hanging on a password prompt.
# -F /dev/null skips ~/.ssh/config: in the GPU container it's mounted from the
# host with non-root owner / group-writable perms, which ssh rejects ("Bad owner
# or permissions"). -i names the key (override via XPU_SSH_KEY).
XPU_SSH_KEY="${XPU_SSH_KEY:-/root/.ssh/id_ed25519}"
SSH_OPTS=(-F /dev/null -i "$XPU_SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes)
SSH=(ssh "${SSH_OPTS[@]}" "${XPU_USER}@${XPU_HOST}")

echo "[xpu] launching fresh container '$XPU_CONTAINER' on $XPU_HOST ..."
# Remove any stale container with our name, then docker-run a fresh detached one.
"${SSH[@]}" "docker rm -f $XPU_CONTAINER >/dev/null 2>&1 || true"
"${SSH[@]}" "docker run -dit --rm --privileged --name $XPU_CONTAINER \
  --device=/dev/dri --network=host \
  \$(for d in /dev/mei*; do echo --device \$d; done) \
  --group-add video --cap-add=SYS_ADMIN \
  --mount type=bind,source=/dev/dri/by-path,target=/dev/dri/by-path \
  --mount type=bind,source=/sys,target=/sys \
  --mount type=bind,source=/dev/bus,target=/dev/bus \
  --mount type=bind,source=/dev/char,target=/dev/char \
  --mount type=bind,source=/dev/infiniband,target=/dev/infiniband \
  -v /mnt/weka:/mnt/weka \
  -v /home:/home \
  --entrypoint /bin/bash $XPU_IMAGE" \
  || { echo "[xpu] docker run failed"; exit 4; }

# Shared-NFS log dir (real /home path, set by the orchestrator). Encoder logs
# land here next to the GPU worker logs. Falls back to remote /tmp if unset.
XPU_LOG_DIR="${XPU_LOG_DIR:-/tmp}"
"${SSH[@]}" "mkdir -p '$XPU_LOG_DIR' 2>/dev/null || true"
echo "[xpu] encoder logs -> $XPU_LOG_DIR/encode_xpu_<n>.log (shared NFS)"
echo "[xpu] host profile = $XPU_PROFILE ($XPU_HOST)"
echo "[xpu] mm-attention-backend = $MM_ATTN_BACKEND"
echo "[xpu] chat-template = $CHAT_TEMPLATE"
echo "[xpu] use-sglang-tokenizer = $USE_SGLANG_TOKENIZER"

SGLANG_TOKENIZER_ARG=""
[ "$USE_SGLANG_TOKENIZER" = "1" ] && SGLANG_TOKENIZER_ARG="--use-sglang-tokenizer"

if [ "$XPU_INSTALL_SGL_KERNEL_XPU" = "1" ]; then
    echo "[xpu] installing latest sgl-kernel-xpu from $XPU_SGL_KERNEL_XPU_REPO"
    "${SSH[@]}" "docker exec $XPU_CONTAINER bash -lc '\
        set -e
        cd /tmp
        rm -rf sgl-kernel-xpu
        git clone $XPU_SGL_KERNEL_XPU_REPO
        cd sgl-kernel-xpu
        python3 -m pip uninstall -y sgl_kernel || true
        python3 -m pip install .
        python3 - <<PY
import importlib.metadata
print(\"sgl_kernel\", importlib.metadata.version(\"sgl_kernel\"))
PY
    '" || { echo "[xpu] sgl-kernel-xpu install failed"; exit 4; }
fi

if [ "$XPU_APPLY_PATCHES" = "1" ]; then
    echo "[xpu] applying SGLang patches from $XPU_PATCH_DIR"
    if [ "$XPU_PATCH_SERVER_ARGS" = "1" ]; then
        echo "[xpu] patching server_args.py"
        "${SSH[@]}" "docker exec $XPU_CONTAINER bash -lc '\
            set -e
            test -f $XPU_PATCH_DIR/server_args.py
            cp $XPU_PATCH_DIR/server_args.py /opt/sglang/python/sglang/srt/server_args.py
            python3 -m py_compile /opt/sglang/python/sglang/srt/server_args.py
        '" || { echo "[xpu] server_args patch install failed"; exit 4; }
    else
        echo "[xpu] keeping image server_args.py"
    fi
    "${SSH[@]}" "docker exec $XPU_CONTAINER bash -lc '\
        set -e
        test -f $XPU_PATCH_DIR/patch_server_args_xpu.py
        python3 $XPU_PATCH_DIR/patch_server_args_xpu.py
        python3 -m py_compile /opt/sglang/python/sglang/srt/server_args.py
    '" || { echo "[xpu] server_args xpu patch failed"; exit 4; }
    "${SSH[@]}" "docker exec $XPU_CONTAINER bash -lc '\
        set -e
        test -f $XPU_PATCH_DIR/patch_dynamo_encode_worker_xpu.py
        python3 $XPU_PATCH_DIR/patch_dynamo_encode_worker_xpu.py
        python3 -m py_compile /usr/local/lib/python3.12/dist-packages/dynamo/sglang/request_handlers/multimodal/encode_worker_handler.py
    '" || { echo "[xpu] dynamo encode worker patch failed"; exit 4; }
    "${SSH[@]}" "docker exec $XPU_CONTAINER bash -lc '\
        set -e
        test -f $XPU_PATCH_DIR/patch_qwen3_vl_moe_encoder_only.py
        python3 $XPU_PATCH_DIR/patch_qwen3_vl_moe_encoder_only.py
        python3 -m py_compile /opt/sglang/python/sglang/srt/models/qwen3_vl_moe.py
    '" || { echo "[xpu] qwen3_vl_moe patch failed"; exit 4; }
    "${SSH[@]}" "docker exec $XPU_CONTAINER bash -lc '\
        set -e
        test -f $XPU_PATCH_DIR/patch_xpu_vision_head72.py
        python3 $XPU_PATCH_DIR/patch_xpu_vision_head72.py
        python3 -m py_compile /opt/sglang/python/sglang/srt/layers/attention/vision.py
    '" || { echo "[xpu] vision xpu head72 patch failed"; exit 4; }
    "${SSH[@]}" "docker exec $XPU_CONTAINER bash -lc '\
        set -e
        test -f $XPU_PATCH_DIR/encode_server.py
        test -f $XPU_PATCH_DIR/internvl.py
        test -f $XPU_PATCH_DIR/internvl_processor.py
        cp $XPU_PATCH_DIR/encode_server.py /opt/sglang/python/sglang/srt/disaggregation/encode_server.py
        cp $XPU_PATCH_DIR/internvl.py /opt/sglang/python/sglang/srt/models/internvl.py
        cp $XPU_PATCH_DIR/internvl_processor.py /opt/sglang/python/sglang/srt/multimodal/processors/internvl.py
        python3 -m py_compile \
          /opt/sglang/python/sglang/srt/disaggregation/encode_server.py \
          /opt/sglang/python/sglang/srt/models/internvl.py \
          /opt/sglang/python/sglang/srt/multimodal/processors/internvl.py
    '" || { echo "[xpu] patch install failed"; exit 4; }
fi

i=0
IFS=',' read -ra XS <<< "$XPUS_CSV"
for XPU in "${XS[@]}"; do
    sys=$((SYS_PORT_BASE + i)); kv=$((KV_EVENT_BASE + i*3)); side=$((SIDE_CHANNEL_BASE + i))
    resolve_xpu_fabric "$XPU"
    echo "[xpu] exec encoder on XPU $XPU (nic=$NIC ip=$XIP sys=$sys kv=$kv side=$side)"
    "${SSH[@]}" "docker exec -d $XPU_CONTAINER bash -lc '\
        ZE_AFFINITY_MASK=$XPU DYN_SYSTEM_PORT=$sys \
        NATS_SERVER=nats://${IP_LOCAL}:${PORT_NATS} \
        ETCD_ENDPOINTS=http://${IP_LOCAL}:${PORT_ETCD} \
        ETCD_LEASE_TTL=600 ETCD_REQUEST_TIMEOUT=600 \
        DYN_REQUEST_PLANE=tcp TRANSFER_LOCAL=0 PYTHONHASHSEED=0 \
        VLLM_NIXL_SIDE_CHANNEL_HOST=$XIP VLLM_NIXL_SIDE_CHANNEL_PORT=$side \
        DYN_VLLM_KV_EVENT_PORT=$kv \
        UCX_MEMTYPE_CACHE=0 UCX_TLS=ze_copy,rc,tcp UCX_NET_DEVICES=$NIC \
        DYN_SGL_EMBEDDING_TRANSFER_MODE=$TRANSFER_MODE \
        ENABLE_ENCODER_CACHE=0 VISION_ENCODE_SERIALIZE=1 \
        DYN_TCP_MAX_MESSAGE_SIZE=268435456 \
        python3 -m dynamo.sglang \
          --multimodal-encode-worker --model-path $MODEL_PATH \
          --served-model-name $SERVED --enable-multimodal --encoder-only \
          --chat-template $CHAT_TEMPLATE --embedding-transfer-mode $TRANSFER_MODE \
          $SGLANG_TOKENIZER_ARG \
          --skip-tokenizer-init --trust-remote-code --page-size 16 \
          --mem-fraction-static $ENC_MEM_FRAC ${MM_ATTN_BACKEND:+--mm-attention-backend $MM_ATTN_BACKEND} \
          --discovery-backend etcd --event-plane nats \
          --disaggregation-transfer-backend nixl --log-level $LOG_LEVEL \
          $ENC_EXTRA_ARGS \
          > $XPU_LOG_DIR/encode_xpu_${XPU}.log 2>&1' " \
      || echo "[xpu] WARN: exec on XPU $XPU returned nonzero"
    i=$((i+1))
    sleep 5
done
echo "[xpu] launched ${#XS[@]} encoder(s); container=$XPU_CONTAINER"
