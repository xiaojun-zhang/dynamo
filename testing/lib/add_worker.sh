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

# GPU-host network identity. The orchestrator resolves these per host (from a
# profile keyed on hostname; see bench_lib.gpu_host_profile) and passes them in;
# the dell07 values are only the standalone fallback for manual invocation.
IP_LOCAL="${IP_LOCAL:-172.26.46.178}"
IP_LOCAL_ROCE="${IP_LOCAL_ROCE:-192.165.123.65}"
PORT_NATS="${PORT_NATS:-14222}"
PORT_ETCD="${PORT_ETCD:-12379}"
UCX_NIC="${UCX_NIC:-mlx5_0:1}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be set by caller}"
TP="${TP:-1}"
KV_DTYPE="${KV_DTYPE:-auto}"
MEM_FRAC="${MEM_FRAC:-0.70}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-qwen2-vl}"
LOG_LEVEL="${LOG_LEVEL:-debug}"
USE_SGLANG_TOKENIZER="${USE_SGLANG_TOKENIZER:-0}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-}"
AGG_EXTRA_ARGS="${AGG_EXTRA_ARGS:-}"
PD_EXTRA_ARGS="${PD_EXTRA_ARGS:-}"
ENC_EXTRA_ARGS="${ENC_EXTRA_ARGS:-}"
# Prefill chunk size: caps how many prompt tokens are prefilled in one forward,
# which bounds the activation working-set peak. Big multimodal prompts (e.g. 8
# images = ~16k tokens) can OOM the LLM MLP on a single GPU if prefilled whole;
# a smaller chunk trades a little prefill speed for fitting in memory.
CHUNKED_PREFILL="${CHUNKED_PREFILL:-16384}"
# nixl-read (PD pulls embeddings via RDMA on demand) is the default: it has no
# pre-provisioned destination-buffer pool, so it does NOT hit "Timeout while
# waiting for available buffer" under high E-worker fan-out (nixl-write exhausted
# its fixed buffer pool with 4 E-workers and dropped ~70% of requests). Still
# GPU-side as long as CuPy is in the container (run_gpu_container.sh installs it);
# without CuPy NIXL silently falls back to a slow host-staged path.
# Override with TRANSFER_MODE=nixl-write to A/B the write path.
TRANSFER_MODE="${TRANSFER_MODE:-nixl-read}"

# ---- PD-worker tunables (disagg). Defaults un-handicap PD vs an agg worker. ----
# Old defaults (prefill_max=1, no max-running, radix off, mem_frac shared 0.70)
# made the PD decode engine run half-empty (batch topped out at ~3 vs agg's ~6),
# so a 2-PD pair lost to 2 agg workers even with an extra encode GPU. These let
# PD batch prefills, fill the decode batch, and use the memory it saves by NOT
# hosting the vision tower for a bigger KV cache.
PD_PREFILL_MAX="${PD_PREFILL_MAX:-8}"          # prefill batch width (was hard 1)
PD_MAX_RUNNING="${PD_MAX_RUNNING:-40}"         # decode batch cap (was unset/null)
PD_MEM_FRAC="${PD_MEM_FRAC:-0.85}"             # PD has no vision tower -> bigger KV
ENC_MEM_FRAC="${ENC_MEM_FRAC:-$MEM_FRAC}"
PD_RADIX="${PD_RADIX:-0}"                      # 1 = enable radix prefix cache on PD
# Encoder: 1 serializes vision encode (safe but no batching); 0 lets it batch.
VISION_ENCODE_SERIALIZE="${VISION_ENCODE_SERIALIZE:-0}"

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

# PD reserves no memory for a vision tower, so it can give more to the KV cache.
EFF_MEM_FRAC="$MEM_FRAC"
[ "$ROLE" = "pd" ] && EFF_MEM_FRAC="$PD_MEM_FRAC"
[ "$ROLE" = "encode" ] && EFF_MEM_FRAC="$ENC_MEM_FRAC"

# Encoders are ALWAYS TP1 (the orchestrator places them on a single GPU). The
# model's TP (e.g. 2 for 32B) applies only to agg/pd. Passing --tp 2 to an
# encoder makes it wait forever for a 2nd TP peer that never joins
# ("DistStoreError: Timed out ... 1/2 clients joined") -> readiness timeout.
EFF_TP="$TP"
[ "$ROLE" = "encode" ] && EFF_TP=1

# Common model args.
COMMON_ARGS=(
  --model-path "$MODEL_PATH"
  --served-model-name "$SERVED"
  --trust-remote-code
  --tp "$EFF_TP"
  --page-size 16
  --mem-fraction-static "$EFF_MEM_FRAC"
  --discovery-backend etcd
  --event-plane nats
  --log-level "$LOG_LEVEL"
  --kv-events-config "$KV_EVENTS"
)

prefill_token_args() {
  if [ -n "$MAX_PREFILL_TOKENS" ]; then
    printf '%s\n' "--max-prefill-tokens" "$MAX_PREFILL_TOKENS"
  fi
}

max_total_token_args() {
  if [ -n "$MAX_TOTAL_TOKENS" ]; then
    printf '%s\n' "--max-total-tokens" "$MAX_TOTAL_TOKENS"
  fi
}

cuda_graph_args() {
  if [ -n "$CUDA_GRAPH_MAX_BS" ]; then
    printf '%s\n' "--cuda-graph-max-bs" "$CUDA_GRAPH_MAX_BS"
  fi
}

sglang_tokenizer_args() {
  if [ "$USE_SGLANG_TOKENIZER" = "1" ]; then
    printf '%s\n' "--use-sglang-tokenizer"
  fi
}

case "$ROLE" in
  agg)
    # Aggregated: full E+P+D in one process. No NIXL, keep in-process vision encode.
    echo "[add_worker] agg  gpus=$GPUS tp=$TP sys=$SYS_PORT -> $LOG"
    # shellcheck disable=SC2086
    env "${COMMON_ENV[@]}" \
      python3 -m dynamo.sglang \
        "${COMMON_ARGS[@]}" \
        --enable-multimodal \
        --chat-template "$CHAT_TEMPLATE" \
        --dtype auto \
        --kv-cache-dtype "$KV_DTYPE" \
        --max-running-requests "${MAX_RUNNING:-40}" \
        --chunked-prefill-size "$CHUNKED_PREFILL" \
        $(prefill_token_args) \
        $(max_total_token_args) \
        $(cuda_graph_args) \
        $(sglang_tokenizer_args) \
        $AGG_EXTRA_ARGS \
      > "$LOG" 2>&1 &
    ;;
  pd)
    # Prefill+decode multimodal worker; reads embeddings via NIXL.
    # PD hosts no vision tower, so let it batch prefills, fill the decode batch,
    # and use a bigger KV cache (PD_* tunables) instead of running half-empty.
    PD_ARGS=(--prefill-max-requests "$PD_PREFILL_MAX"
             --max-running-requests "$PD_MAX_RUNNING"
             --chunked-prefill-size "$CHUNKED_PREFILL")
    [ -n "$MAX_PREFILL_TOKENS" ] && PD_ARGS+=(--max-prefill-tokens "$MAX_PREFILL_TOKENS")
    [ -n "$MAX_TOTAL_TOKENS" ] && PD_ARGS+=(--max-total-tokens "$MAX_TOTAL_TOKENS")
    [ -n "$CUDA_GRAPH_MAX_BS" ] && PD_ARGS+=(--cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS")
    [ "$USE_SGLANG_TOKENIZER" = "1" ] && PD_ARGS+=(--use-sglang-tokenizer)
    [ "$PD_RADIX" = "1" ] || PD_ARGS+=(--disable-radix-cache)
    echo "[add_worker] pd   gpus=$GPUS tp=$TP sys=$SYS_PORT side=$SIDE_PORT "\
"mem_frac=$EFF_MEM_FRAC prefill_max=$PD_PREFILL_MAX max_running=$PD_MAX_RUNNING radix=$PD_RADIX -> $LOG"
    # shellcheck disable=SC2086
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
        "${PD_ARGS[@]}" \
        $PD_EXTRA_ARGS \
        --skip-tokenizer-init \
        --disaggregation-transfer-backend nixl \
      > "$LOG" 2>&1 &
    ;;
  encode)
    # Encode-only worker on a local CUDA GPU; sends embeddings via NIXL (cuda_ipc).
    echo "[add_worker] enc  gpus=$GPUS tp=$TP sys=$SYS_PORT side=$SIDE_PORT "\
"serialize=$VISION_ENCODE_SERIALIZE -> $LOG"
    # shellcheck disable=SC2086
    env "${COMMON_ENV[@]}" \
      VLLM_NIXL_SIDE_CHANNEL_HOST="$IP_LOCAL_ROCE" \
      VLLM_NIXL_SIDE_CHANNEL_PORT="$SIDE_PORT" \
      UCX_TLS="cuda_ipc,ib,rc,ud,rc_verbs,ud_verbs,cuda_copy" \
      UCX_NET_DEVICES="$UCX_NIC" \
      UCX_MEMTYPE_CACHE=0 \
      DYN_SGL_EMBEDDING_TRANSFER_MODE="$TRANSFER_MODE" \
      VISION_ENCODE_SERIALIZE="$VISION_ENCODE_SERIALIZE" \
      python3 -m dynamo.sglang \
        "${COMMON_ARGS[@]}" \
        --multimodal-encode-worker \
        --enable-multimodal \
        --encoder-only \
        --chat-template "$CHAT_TEMPLATE" \
        --embedding-transfer-mode "$TRANSFER_MODE" \
        $(sglang_tokenizer_args) \
        --skip-tokenizer-init \
        --disaggregation-transfer-backend nixl \
        $ENC_EXTRA_ARGS \
      > "$LOG" 2>&1 &
    ;;
  *)
    echo "unknown role: $ROLE (agg|pd|encode)"; exit 2 ;;
esac

WPID=$!
echo "$WPID" >> "$LOG_DIR/harness.pids"   # recorded so teardown kills only ours
echo "  pid=$WPID log=$LOG"
