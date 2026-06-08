#!/bin/bash
# Cross-host disagg ENCODE-only start script for the Intel B70 (Battlemage) XPU host.
# Pairs with: the 4x L40S PD workers on dell07 (start_sglang_pd_cuda_8b_dell07.sh).
#
# This host (encoder): sc09giga01-b70.sc.intel.com
# This script starts:  1 multimodal encode worker on an Intel XPU (Battlemage).
# Control plane (NATS/etcd/frontend) + PD workers run on:  dell07 (172.26.46.178)
#
# Model: Qwen/Qwen3-VL-8B-Instruct
# Derived from: hongming/dynamo/02_xpu_sh/start_sglang_pd_xpu_32b_b70.sh
#
# RUN ORDER: start the dell07 PD side FIRST (it brings up NATS/etcd/frontend +
# the 4 PD workers), then run this script here on the B70 host.
#
# Topology / transport notes:
#   - Discovery = etcd, event plane = nats (REQUIRED cross-host; the file/zmq
#     sample only works when encoder and PD share a host/filesystem).
#   - UCX_TLS = ze_copy,rc,tcp  -> Level-Zero (XPU) copy + RDMA verbs, NO cuda_ipc.
#   - XPU selected via ZE_AFFINITY_MASK (Intel equivalent of CUDA_VISIBLE_DEVICES).
#   - nixl-read: the PD workers read the embeddings this encoder produces.

set -e

# ========================================
# PROXY (bypass for localhost, local RoCE fabric, and the remote PD host)
# ========================================
export http_proxy=http://proxy.ims.intel.com:911
export https_proxy=http://proxy.ims.intel.com:911
export ftp_proxy=http://proxy.ims.intel.com:911
export no_proxy=0.0.0.0,127.0.0.1,localhost,172.26.46.178,192.165.123.0/24,.intel.com
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
export NO_PROXY=$no_proxy

# ========================================
# CONFIGURATION - must match the dell07 PD side
# ========================================

# Remote PD host (NATS / etcd / frontend / PD workers) = dell07 mgmt IP.
export IP_REMOTE=${IP_REMOTE:-172.26.46.178}
export PORT_NATS=${PORT_NATS:-14222}
export PORT_ETCD=${PORT_ETCD:-12379}

# XPU devices (Battlemage) to run encoders on — one encode worker per device.
# Space-separated list of 0..7. Default "0 1" => two encoders on XPU 0 and 1.
# (To spread across both NUMA/NIC fabrics instead, use e.g. XPU_DEVICES="0 4".)
export XPU_DEVICES=${XPU_DEVICES:-"0 1"}

# Base ports — each encoder i gets a distinct port so they don't collide on
# this host. Side-channel = base + i, KV-events = base + i*3, sys = base + i.
export SIDE_CHANNEL_BASE=${SIDE_CHANNEL_BASE:-20099}
export KV_EVENT_BASE=${KV_EVENT_BASE:-22090}
export SYS_PORT_BASE=${SYS_PORT_BASE:-8081}

# Stagger between encoder launches (seconds) — lets each model load settle.
export STARTUP_DELAY=${STARTUP_DELAY:-10}

# Map an XPU index to its NUMA-local RoCE NIC + IP (same mapping as the 32B
# script): XPU 0..3 (NUMA 0) -> mlx5_0 (.40); XPU 4..7 (NUMA 2) -> mlx5_2 (.37).
nic_for_xpu() {  # echoes "UCX_NIC NETDEV IP"
    case "$1" in
        0|1|2|3) echo "mlx5_0:1 ens12np0 192.165.123.40" ;;  # NUMA 0
        4|5|6|7) echo "mlx5_2:1 ens9np0  192.165.123.37" ;;  # NUMA 2
        *)       echo "mlx5_0:1 ens12np0 192.165.123.40" ;;
    esac
}

# Model + name (must match what the PD workers serve).
export MODEL=${MODEL:-/mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-8B-Instruct}
export SERVED=${SERVED:-Qwen/Qwen3-VL-8B-Instruct}
export TRANSFER_MODE=${TRANSFER_MODE:-nixl-read}

# Larger TCP message size for multimodal embedding metadata over the control plane.
export DYN_TCP_MAX_MESSAGE_SIZE=268435456  # 256 MB

# ========================================

LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"

NUM_ENC=$(echo $XPU_DEVICES | wc -w)

echo "=========================================="
echo "Dynamo SGLang Intel B70 XPU Encode Workers (${NUM_ENC}E)"
echo "Model: $SERVED"
echo "Pair : PD workers @ $IP_REMOTE (dell07, 4x L40S)"
echo "=========================================="
echo ""
echo "Remote PD host    : $IP_REMOTE  (NATS:$PORT_NATS  etcd:$PORT_ETCD)"
echo "XPU devices       : $XPU_DEVICES  (one encode worker each)"
echo "Transfer mode     : $TRANSFER_MODE"
echo ""

# Pre-flight: control-plane reachability (NATS / etcd on the PD host)
echo "[pre-flight] control plane on $IP_REMOTE ..."
for p in $PORT_NATS $PORT_ETCD; do
    if timeout 2 bash -c "cat </dev/null >/dev/tcp/$IP_REMOTE/$p" 2>/dev/null; then
        echo "  OK   port $p"
    else
        echo "  FAIL port $p (firewall? PD side not started on dell07?)"
    fi
done

if ! timeout 2 bash -c "cat </dev/null >/dev/tcp/$IP_REMOTE/$PORT_NATS" 2>/dev/null; then
    read -p "Control plane unreachable. Continue anyway? (y/n) " -n 1 -r
    echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi

# Optional: pass --mm-attention-backend if MM_ATTN_BACKEND is set.
# Set MM_ATTN_BACKEND=xpu_attn to use the PR 26460 XPU vision-attention path
# (this image is hm_dynamo_b70_pr26460, which has it). Leave unset to use the
# default triton_attn backend. Example:
#   MM_ATTN_BACKEND=xpu_attn XPU_DEVICES="0 1" ./start_sglang_encode_xpu_8b_b70.sh
MM_ATTN_FLAG=""
if [ -n "${MM_ATTN_BACKEND:-}" ]; then
    MM_ATTN_FLAG="--mm-attention-backend ${MM_ATTN_BACKEND}"
    echo "  MM attention backend: ${MM_ATTN_BACKEND} (PR26460 path)"
else
    echo "  MM attention backend: (default, triton_attn) — set MM_ATTN_BACKEND=xpu_attn to try PR26460 path"
fi
echo ""

# ========================================
# Launch one encode worker per XPU device
# ========================================
ENC_PIDS=()
i=0
for XPU in $XPU_DEVICES; do
    read -r NIC NDEV IP <<<"$(nic_for_xpu $XPU)"
    SCP=$((SIDE_CHANNEL_BASE + i))
    KVP=$((KV_EVENT_BASE + i * 3))
    SYSP=$((SYS_PORT_BASE + i))
    ENC_LOG="$LOG_DIR/encode_xpu_8b_b70_${XPU}.log"

    # Pre-flight: data-plane fabric IP must exist on the chosen netdev
    if ! python3 -c "
import socket, fcntl, struct
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ip = socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, struct.pack('256s', b'$NDEV'[:15]))[20:24])
assert ip == '$IP', f'$NDEV has {ip}, expected $IP'
" 2>/dev/null; then
        echo "WARNING: $NDEV ($NIC) does not have $IP. Verify NIC<->IP mapping for XPU $XPU."
    fi

    echo "Starting encode worker on XPU $XPU (NIC=$NIC IP=$IP sys=$SYSP side=$SCP kv=$KVP) -> $ENC_LOG"

    ZE_AFFINITY_MASK=$XPU \
    DYN_SYSTEM_PORT=$SYSP \
    NATS_SERVER=nats://${IP_REMOTE}:${PORT_NATS} \
    ETCD_ENDPOINTS=http://${IP_REMOTE}:${PORT_ETCD} \
    ETCD_REQUEST_TIMEOUT=600 \
    ETCD_LEASE_TTL=600 \
    DYN_REQUEST_PLANE=tcp \
    TRANSFER_LOCAL=0 \
    PYTHONHASHSEED=0 \
    VLLM_NIXL_SIDE_CHANNEL_HOST=${IP} \
    VLLM_NIXL_SIDE_CHANNEL_PORT=${SCP} \
    DYN_VLLM_KV_EVENT_PORT=${KVP} \
    UCX_MEMTYPE_CACHE=0 \
    UCX_TLS=ze_copy,rc,tcp \
    UCX_NET_DEVICES=${NIC} \
    DYN_SGL_EMBEDDING_TRANSFER_MODE=${TRANSFER_MODE} \
    ENABLE_ENCODER_CACHE=0 \
    VISION_ENCODE_SERIALIZE=1 \
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
        --mem-fraction-static 0.5 \
        ${MM_ATTN_FLAG} \
        --discovery-backend etcd \
        --event-plane nats \
        --disaggregation-transfer-backend nixl \
        --log-level debug \
        --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'$KVP'","enable_kv_cache_events":true}' \
        > "$ENC_LOG" 2>&1 &
    pid=$!
    ENC_PIDS+=("$pid")
    echo "  [launch] encoder pid=$pid xpu=$XPU -> $ENC_LOG"

    i=$((i + 1))
    # Stagger launches so model loads settle (skip the wait after the last one)
    if [ "$i" -lt "$NUM_ENC" ]; then
        echo "  ...staggering ${STARTUP_DELAY}s before next encoder"
        sleep "$STARTUP_DELAY"
    fi
done

echo ""
echo "=========================================="
echo "Intel B70 XPU Encode Workers started (${NUM_ENC}E)"
echo "=========================================="
echo "  Model:           $SERVED"
echo "  Remote PD host:  $IP_REMOTE  (NATS:$PORT_NATS  etcd:$PORT_ETCD)"
echo "  XPU devices:     $XPU_DEVICES"
echo "  PIDs:            ${ENC_PIDS[*]}"
echo "  Transfer mode:   $TRANSFER_MODE   (UCX_TLS=ze_copy,rc,tcp, no cuda_ipc)"
echo ""
echo "Logs:   $LOG_DIR/encode_xpu_8b_b70_<xpu>.log"
echo "Tail:   tail -f $LOG_DIR/encode_xpu_8b_b70_*.log"
echo "Ready:  grep -il 'Model registration succeeded' $LOG_DIR/encode_xpu_8b_b70_*.log"
echo "Stop:   pkill -f 'dynamo.sglang.*multimodal-encode-worker'"
echo ""
echo "Verify from dell07:  curl -s http://${IP_REMOTE}:7001/v1/models"
echo ""
