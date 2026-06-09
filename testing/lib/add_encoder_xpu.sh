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
#   MODEL_PATH (required), XPU_SSH_PASS (for sshpass), XPU_IMAGE, XPU_CONTAINER,
#   IP_LOCAL (dell07 mgmt), PORT_NATS, PORT_ETCD, TRANSFER_MODE,
#   SIDE_CHANNEL_BASE, KV_EVENT_BASE, SYS_PORT_BASE,
#   XPU_LOG_DIR (shared-NFS /home path for encoder logs; falls back to /tmp)
set -e

XPUS_CSV="$1"; SERVED="$2"
[ -z "$SERVED" ] && { echo "usage: add_encoder_xpu.sh <xpu_csv> <served>"; exit 2; }

XPU_HOST="${XPU_HOST:-172.26.46.180}"
XPU_USER="${XPU_SSH_USER:-h-zheng}"
XPU_IMAGE="${XPU_IMAGE:-hm_dynamo_b70_pr26460:latest}"
XPU_CONTAINER="${XPU_CONTAINER:-harness_b70_enc}"
IP_LOCAL="${IP_LOCAL:-172.26.46.178}"
PORT_NATS="${PORT_NATS:-14222}"
PORT_ETCD="${PORT_ETCD:-12379}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be set}"
TRANSFER_MODE="${TRANSFER_MODE:-nixl-read}"
SIDE_CHANNEL_BASE="${SIDE_CHANNEL_BASE:-20099}"
KV_EVENT_BASE="${KV_EVENT_BASE:-22090}"
SYS_PORT_BASE="${SYS_PORT_BASE:-8091}"
# XPU encoders ALWAYS use the PR26460 xpu_attn vision-attention backend by
# default (this is the B70 image's optimized path). Override only to experiment.
MM_ATTN_BACKEND="${MM_ATTN_BACKEND:-xpu_attn}"

SSH_OPTS=(-o StrictHostKeyChecking=no -o ConnectTimeout=10)
if [ -n "${XPU_SSH_PASS:-}" ]; then
    command -v sshpass >/dev/null || { echo "ERROR: XPU_SSH_PASS set but sshpass not installed"; exit 3; }
    export SSHPASS="$XPU_SSH_PASS"
    SSH=(sshpass -e ssh "${SSH_OPTS[@]}" "${XPU_USER}@${XPU_HOST}")
else
    SSH=(ssh "${SSH_OPTS[@]}" "${XPU_USER}@${XPU_HOST}")
fi

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
echo "[xpu] mm-attention-backend = $MM_ATTN_BACKEND"

# B70 NUMA->NIC mapping: XPU 0-3 -> mlx5_0/.40 ; XPU 4-7 -> mlx5_2/.37
i=0
IFS=',' read -ra XS <<< "$XPUS_CSV"
for XPU in "${XS[@]}"; do
    sys=$((SYS_PORT_BASE + i)); kv=$((KV_EVENT_BASE + i*3)); side=$((SIDE_CHANNEL_BASE + i))
    if [ "$XPU" -lt 4 ]; then NIC="mlx5_0:1"; XIP="192.165.123.40"; else NIC="mlx5_2:1"; XIP="192.165.123.37"; fi
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
          --chat-template qwen2-vl --embedding-transfer-mode $TRANSFER_MODE \
          --skip-tokenizer-init --trust-remote-code --page-size 16 \
          --mem-fraction-static 0.5 ${MM_ATTN_BACKEND:+--mm-attention-backend $MM_ATTN_BACKEND} \
          --discovery-backend etcd --event-plane nats \
          --disaggregation-transfer-backend nixl --log-level debug \
          > $XPU_LOG_DIR/encode_xpu_${XPU}.log 2>&1' " \
      || echo "[xpu] WARN: exec on XPU $XPU returned nonzero"
    i=$((i+1))
    sleep 5
done
echo "[xpu] launched ${#XS[@]} encoder(s); container=$XPU_CONTAINER"
