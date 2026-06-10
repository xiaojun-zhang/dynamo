#!/bin/bash
# Composable piece: launch ONE local CUDA worker and return (does NOT wait for
# registration — the caller polls /health). Assumes the control plane is already
# up (start_controlplane.sh). This is the single source of truth for the CUDA
# worker command; the manual wrappers and the Python orchestrator both use it.
#
# Usage:
#   add_worker.sh <role> <gpus> <sys_port> <kv_port> <side_port> <model_served>
#
#   role      : agg | pd | encode
#               agg    = full Encode+Prefill+Decode in one process (no disagg)
#               pd     = multimodal (prefill+decode) worker, reads NIXL embeddings
#               encode = encode-only worker (vision tower), sends NIXL embeddings
#   gpus      : comma list of CUDA indices for THIS instance (len == TP), e.g. "1" or "4,5"
#   sys_port  : DYN_SYSTEM_PORT (unique per worker on this host)
#   kv_port   : ZMQ KV-event port (unique per worker)
#   side_port : NIXL side-channel port (unique per worker; ignored for agg)
#   model_served : served model name, e.g. Qwen/Qwen3-VL-8B-Instruct
#
# Model facts (path / tp / kv-dtype / mem-fraction) come from env set by the
# caller (the orchestrator / wrapper): MODEL_PATH, TP, KV_DTYPE, MEM_FRAC.
# Transfer mode for disagg is TRANSFER_MODE (default nixl-read).
set -e

ROLE="$1"; GPUS="$2"; SYS_PORT="$3"; KV_PORT="$4"; SIDE_PORT="$5"; SERVED="$6"
[ -z "$SERVED" ] && { echo "usage: add_worker.sh <role> <gpus> <sys> <kv> <side> <served>"; exit 2; }

IP_LOCAL="${IP_LOCAL:-172.26.46.178}"
IP_LOCAL_ROCE="${IP_LOCAL_ROCE:-192.165.123.65}"
PORT_NATS="${PORT_NATS:-14222}"
PORT_ETCD="${PORT_ETCD:-12379}"
UCX_NIC="${UCX_NIC:-mlx5_0:1}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be set by caller}"
TP="${TP:-1}"
KV_DTYPE="${KV_DTYPE:-auto}"
MEM_FRAC="${MEM_FRAC:-0.70}"
TRANSFER_MODE="${TRANSFER_MODE:-nixl-read}"
LOG_DIR="${LOG_DIR:-$(pwd)/logs}"
mkdir -p "$LOG_DIR"

export DYN_TCP_MAX_MESSAGE_SIZE=268435456
export DYN_HTTP_BODY_LIMIT_MB=256
export no_proxy="0.0.0.0,127.0.0.1,localhost,${IP_LOCAL},192.165.123.0/24,172.26.46.180,.intel.com"
export NO_PROXY="$no_proxy"

CVD="$GPUS"                                  # CUDA_VISIBLE_DEVICES (comma list)
TAG="${ROLE}_gpu$(echo "$GPUS" | tr ',' '-')"
LOG="$LOG_DIR/worker_${TAG}.log"

# Common env shared by all roles.
COMMON_ENV=(
  "CUDA_VISIBLE_DEVICES=$CVD"
  "DYN_SYSTEM_PORT=$SYS_PORT"
  "NATS_SERVER=nats://${IP_LOCAL}:${PORT_NATS}"
  "ETCD_ENDPOINTS=http://${IP_LOCAL}:${PORT_ETCD}"
  "ETCD_LEASE_TTL=600"
  "ETCD_REQUEST_TIMEOUT=600"
  "DYN_REQUEST_PLANE=tcp"
  "DYN_LOG=debug"
  "TRANSFER_LOCAL=0"
  "PYTHONHASHSEED=0"
  "DYN_VLLM_KV_EVENT_PORT=$KV_PORT"
  "ENABLE_ENCODER_CACHE=0"
  "NCCL_DEBUG=INFO"
  "NCCL_DEBUG_SUBSYS=INIT,P2P"
)

KV_EVENTS="{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${KV_PORT}\",\"enable_kv_cache_events\":true}"

# Common model args.
COMMON_ARGS=(
  --model-path "$MODEL_PATH"
  --served-model-name "$SERVED"
  --trust-remote-code
  --tp "$TP"
  --page-size 16
  --mem-fraction-static "$MEM_FRAC"
  --discovery-backend etcd
  --event-plane nats
  --log-level debug
  --kv-events-config "$KV_EVENTS"
)

case "$ROLE" in
  agg)
    # Aggregated: full E+P+D in one process. No NIXL, keep in-process vision encode.
    echo "[add_worker] agg  gpus=$GPUS tp=$TP sys=$SYS_PORT -> $LOG"
    env "${COMMON_ENV[@]}" \
      python3 -m dynamo.sglang \
        "${COMMON_ARGS[@]}" \
        --enable-multimodal \
        --chat-template qwen2-vl \
        --dtype auto \
        --kv-cache-dtype "$KV_DTYPE" \
        --max-running-requests "${MAX_RUNNING:-40}" \
        --chunked-prefill-size 16384 \
      > "$LOG" 2>&1 &
    ;;
  pd)
    # Prefill+decode multimodal worker; reads embeddings via NIXL.
    echo "[add_worker] pd   gpus=$GPUS tp=$TP sys=$SYS_PORT side=$SIDE_PORT -> $LOG"
    env "${COMMON_ENV[@]}" \
      VLLM_NIXL_SIDE_CHANNEL_HOST="$IP_LOCAL_ROCE" \
      VLLM_NIXL_SIDE_CHANNEL_PORT="$SIDE_PORT" \
      UCX_TLS="cuda_ipc,ib,rc,ud,rc_verbs,ud_verbs,cuda_copy" \
      UCX_NET_DEVICES="$UCX_NIC" \
      UCX_MEMTYPE_CACHE=0 \
      DYN_SGL_EMBEDDING_TRANSFER_MODE="$TRANSFER_MODE" \
      python3 -m dynamo.sglang \
        "${COMMON_ARGS[@]}" \
        --multimodal-worker \
        --embedding-transfer-mode "$TRANSFER_MODE" \
        --kv-cache-dtype "$KV_DTYPE" \
        --prefill-max-requests 1 \
        --skip-tokenizer-init \
        --disable-radix-cache \
        --disaggregation-transfer-backend nixl \
      > "$LOG" 2>&1 &
    ;;
  encode)
    # Encode-only worker on a local CUDA GPU; sends embeddings via NIXL (cuda_ipc).
    echo "[add_worker] enc  gpus=$GPUS tp=$TP sys=$SYS_PORT side=$SIDE_PORT -> $LOG"
    env "${COMMON_ENV[@]}" \
      VLLM_NIXL_SIDE_CHANNEL_HOST="$IP_LOCAL_ROCE" \
      VLLM_NIXL_SIDE_CHANNEL_PORT="$SIDE_PORT" \
      UCX_TLS="cuda_ipc,ib,rc,ud,rc_verbs,ud_verbs,cuda_copy" \
      UCX_NET_DEVICES="$UCX_NIC" \
      UCX_MEMTYPE_CACHE=0 \
      DYN_SGL_EMBEDDING_TRANSFER_MODE="$TRANSFER_MODE" \
      VISION_ENCODE_SERIALIZE=1 \
      python3 -m dynamo.sglang \
        "${COMMON_ARGS[@]}" \
        --multimodal-encode-worker \
        --enable-multimodal \
        --encoder-only \
        --chat-template qwen2-vl \
        --embedding-transfer-mode "$TRANSFER_MODE" \
        --skip-tokenizer-init \
        --disaggregation-transfer-backend nixl \
      > "$LOG" 2>&1 &
    ;;
  *)
    echo "unknown role: $ROLE (agg|pd|encode)"; exit 2 ;;
esac

WPID=$!
echo "$WPID" >> "$LOG_DIR/harness.pids"   # recorded so teardown kills only ours
echo "  pid=$WPID log=$LOG"
