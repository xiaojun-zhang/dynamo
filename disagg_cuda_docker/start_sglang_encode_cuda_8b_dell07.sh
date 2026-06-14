#!/bin/bash
# Same-host disagg ENCODE-only start script for dell07 (L40S, CUDA).
# Runs the multimodal encode worker on an NVIDIA L40S, paired with the PD
# workers from start_sglang_pd_cuda_8b_dell07.sh on the SAME host.
#
# This host (dell07):  hostname=sc09dell07-rtx, mgmt IP=172.26.46.178
# This script starts:  1 (or more) CUDA encode workers on L40S GPU(s).
# Control plane (NATS/etcd/frontend) + PD workers also run here (dell07).
#
# Model: Qwen/Qwen3-VL-8B-Instruct
# Derived from:
#   - ../disagg_xpu_docker/start_sglang_encode_xpu_8b_b70.sh (encoder role/flags)
#   - start_sglang_pd_cuda_8b_dell07.sh (transport/env — must match the PD side)
#
# RUN ORDER: start the PD side FIRST (it brings up NATS/etcd/frontend + the PD
# workers), then run this encoder. Both register into the same etcd; the PD
# workers read the embeddings this encoder produces.
#
# Transport notes (same-host, all on L40S):
#   - Discovery = etcd, event plane = nats (MUST match the PD side).
#   - Transfer  = nixl-read (MUST match the PD side: DYN_SGL_EMBEDDING_TRANSFER_MODE).
#   - UCX_TLS keeps cuda_ipc (same-host GPU<->GPU P2P) + cuda_copy + verbs.
#     This is the key difference from the cross-host PD launch (which drops
#     cuda_ipc) and from the XPU encoder (which uses ze_copy instead).
#   - GPU selected via CUDA_VISIBLE_DEVICES (not ZE_AFFINITY_MASK — this is NVIDIA).

set -e

# ========================================
# PROXY (bypass for localhost + this host's IPs)
# ========================================
export http_proxy=http://proxy.ims.intel.com:911
export https_proxy=http://proxy.ims.intel.com:911
export ftp_proxy=http://proxy.ims.intel.com:911
export no_proxy=0.0.0.0,127.0.0.1,localhost,172.26.46.178,192.165.123.65,.intel.com
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
export NO_PROXY=$no_proxy

# ========================================
# CONFIGURATION - must match the PD side (start_sglang_pd_cuda_8b_dell07.sh)
# ========================================
export IP_LOCAL_MGMT=172.26.46.178      # control plane (NATS/etcd/frontend)
export IP_LOCAL_ROCE=192.165.123.65     # NIXL side-channel host (matches PD)
export IP_LOCAL=$IP_LOCAL_MGMT

export PORT_NATS=14222
export PORT_ETCD=12379
export PORT_HTTP=7001

# Encoder GPU(s) — one encode worker per L40S listed here. PD uses GPUs 1-4 by
# default, so the encoder defaults to GPU 0. Override: ENC_GPUS="0 5" ./start_...sh
export ENC_GPUS="${ENC_GPUS:-0}"

# RDMA NIC (same as PD). For same-host NIXL the cuda_ipc path carries the data;
# UCX still wants a device for the verbs transports.
export UCX_NIC=mlx5_0:1

# Model + name (must match what the PD workers serve).
export MODEL=/mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-8B-Instruct
export SERVED=Qwen/Qwen3-VL-8B-Instruct

# MUST match the PD side's TRANSFER_MODE (PD = nixl-read).
export TRANSFER_MODE="${TRANSFER_MODE:-nixl-read}"

# Larger TCP message size for multimodal embedding metadata.
export DYN_TCP_MAX_MESSAGE_SIZE=268435456  # 256 MB

# Port bases — kept off the PD ranges to avoid collisions on this host.
# PD uses sys 8082-8085, kv 22081-22084. Encoders start at sys 8091 / kv 22090 /
# side-channel 20099, incremented per encoder.
export SYS_PORT_BASE="${SYS_PORT_BASE:-8091}"
export KV_EVENT_BASE="${KV_EVENT_BASE:-22090}"
export SIDE_CHANNEL_BASE="${SIDE_CHANNEL_BASE:-20099}"

# Stagger between encoder launches (seconds).
export STARTUP_DELAY="${STARTUP_DELAY:-10}"

# ========================================

LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"

# Optional MM attention backend (parity with the XPU script). Leave unset for
# the default. Example: MM_ATTN_BACKEND=flashinfer ./start_...sh
MM_ATTN_FLAG=""
if [ -n "${MM_ATTN_BACKEND:-}" ]; then
    MM_ATTN_FLAG="--mm-attention-backend ${MM_ATTN_BACKEND}"
fi

NUM_ENC=$(echo $ENC_GPUS | wc -w)

echo "=========================================="
echo "Dynamo SGLang Same-Host CUDA Encode Worker(s) (${NUM_ENC}E)"
echo "Model: $SERVED"
echo "Pair : PD workers on this same host (dell07, L40S)"
echo "=========================================="
echo ""
echo "Control plane:   NATS nats://$IP_LOCAL_MGMT:$PORT_NATS  etcd http://$IP_LOCAL_MGMT:$PORT_ETCD"
echo "Encoder GPUs:    $ENC_GPUS  (one encode worker each)"
echo "Transfer mode:   $TRANSFER_MODE   (must match the PD side)"
echo "UCX_TLS:         cuda_ipc,ib,rc,ud,rc_verbs,ud_verbs,cuda_copy  (same-host P2P)"
[ -n "$MM_ATTN_FLAG" ] && echo "MM attn backend: ${MM_ATTN_BACKEND}"
echo ""

# Pre-flight: control plane must be up (PD side started first)
echo "[pre-flight] control plane on $IP_LOCAL_MGMT ..."
for p in $PORT_NATS $PORT_ETCD; do
    if timeout 2 bash -c "cat </dev/null >/dev/tcp/$IP_LOCAL_MGMT/$p" 2>/dev/null; then
        echo "  OK   port $p"
    else
        echo "  FAIL port $p (start the PD side first: ./start_sglang_pd_cuda_8b_dell07.sh)"
    fi
done
if ! timeout 2 bash -c "cat </dev/null >/dev/tcp/$IP_LOCAL_MGMT/$PORT_NATS" 2>/dev/null; then
    read -p "Control plane unreachable. Continue anyway? (y/n) " -n 1 -r
    echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi

# ========================================
# Launch one encode worker per GPU
# ========================================
ENC_PIDS=()
i=0
for gpu in $ENC_GPUS; do
    sys_port=$((SYS_PORT_BASE + i))
    kv_port=$((KV_EVENT_BASE + i * 3))
    side_port=$((SIDE_CHANNEL_BASE + i))
    enc_log="$LOG_DIR/encode_cuda_8b_gpu${gpu}.log"
    echo "Starting CUDA encode worker on GPU $gpu (sys=$sys_port kv=$kv_port side=$side_port) -> $enc_log"

    CUDA_VISIBLE_DEVICES=$gpu \
    DYN_SYSTEM_PORT=$sys_port \
    NATS_SERVER=nats://${IP_LOCAL_MGMT}:${PORT_NATS} \
    ETCD_ENDPOINTS=http://${IP_LOCAL_MGMT}:${PORT_ETCD} \
    ETCD_LEASE_TTL=600 \
    ETCD_REQUEST_TIMEOUT=600 \
    DYN_REQUEST_PLANE=tcp \
    DYN_LOG=debug \
    TRANSFER_LOCAL=0 \
    PYTHONHASHSEED=0 \
    VLLM_NIXL_SIDE_CHANNEL_HOST=${IP_LOCAL_ROCE} \
    VLLM_NIXL_SIDE_CHANNEL_PORT=${side_port} \
    DYN_VLLM_KV_EVENT_PORT=${kv_port} \
    UCX_TLS=cuda_ipc,ib,rc,ud,rc_verbs,ud_verbs,cuda_copy \
    UCX_NET_DEVICES=${UCX_NIC} \
    UCX_MEMTYPE_CACHE=0 \
    DYN_SGL_EMBEDDING_TRANSFER_MODE=${TRANSFER_MODE} \
    ENABLE_ENCODER_CACHE=0 \
    VISION_ENCODE_SERIALIZE=1 \
    NCCL_DEBUG=INFO \
    NCCL_DEBUG_SUBSYS=INIT,P2P \
    python3 -m dynamo.sglang \
        --multimodal-encode-worker \
        --model-path "$MODEL" \
        --served-model-name "$SERVED" \
        --enable-multimodal \
        --encoder-only \
        --chat-template qwen2-vl \
        --embedding-transfer-mode "$TRANSFER_MODE" \
        --skip-tokenizer-init \
        --trust-remote-code \
        --page-size 16 \
        --mem-fraction-static 0.50 \
        ${MM_ATTN_FLAG} \
        --discovery-backend etcd \
        --event-plane nats \
        --disaggregation-transfer-backend nixl \
        --log-level debug \
        --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'$kv_port'","enable_kv_cache_events":true}' \
        > "$enc_log" 2>&1 &
    pid=$!
    ENC_PIDS+=("$pid")
    echo "  [launch] encoder pid=$pid gpu=$gpu -> $enc_log"

    i=$((i + 1))
    if [ "$i" -lt "$NUM_ENC" ]; then
        echo "  ...staggering ${STARTUP_DELAY}s before next encoder"
        sleep "$STARTUP_DELAY"
    fi
done

echo ""
echo "=========================================="
echo "Same-host CUDA encode worker(s) started (${NUM_ENC}E)"
echo "=========================================="
echo "  Model:         $SERVED"
echo "  Encoder GPUs:  $ENC_GPUS"
echo "  PIDs:          ${ENC_PIDS[*]}"
echo "  Transfer mode: $TRANSFER_MODE"
echo ""
echo "Logs:   $LOG_DIR/encode_cuda_8b_gpu<gpu>.log"
echo "Tail:   tail -f $LOG_DIR/encode_cuda_8b_gpu*.log"
echo "Ready:  grep -il 'Model registration succeeded' $LOG_DIR/encode_cuda_8b_gpu*.log"
echo "Stop:   pkill -f 'dynamo.sglang.*multimodal-encode-worker'"
echo ""
echo "Verify the full pipeline:  curl -s http://${IP_LOCAL_MGMT}:${PORT_HTTP}/v1/models"
echo ""
