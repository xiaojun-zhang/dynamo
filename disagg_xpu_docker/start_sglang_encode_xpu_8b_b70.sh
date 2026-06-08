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

# XPU device (Battlemage) to run the single encoder on. 0..7. Default 0 (NUMA 0).
export XPU_DEVICE=${XPU_DEVICE:-0}

# NUMA-local RoCE NIC selection on the B70 host (same mapping as the 32B script):
#   XPU 0..3 (NUMA 0) -> mlx5_0 (192.165.123.40)
#   XPU 4..7 (NUMA 2) -> mlx5_2 (192.165.123.37)
# Override UCX_NIC / IP_LOCAL on the command line to force a specific NIC.
if [ -z "${UCX_NIC:-}" ] || [ -z "${IP_LOCAL:-}" ]; then
    case "$XPU_DEVICE" in
        0|1|2|3)  AUTO_NIC=mlx5_0; AUTO_NDEV=ens12np0; AUTO_IP=192.165.123.40 ;;  # NUMA 0
        4|5|6|7)  AUTO_NIC=mlx5_2; AUTO_NDEV=ens9np0;  AUTO_IP=192.165.123.37 ;;  # NUMA 2
        *)        AUTO_NIC=mlx5_0; AUTO_NDEV=ens12np0; AUTO_IP=192.165.123.40 ;;
    esac
    : "${UCX_NIC:=${AUTO_NIC}:1}"
    : "${IP_LOCAL:=${AUTO_IP}}"
    : "${LOCAL_NDEV:=${AUTO_NDEV}}"
fi
export UCX_NIC IP_LOCAL LOCAL_NDEV

# NIXL side-channel + ZMQ KV-event ports (must be open through any firewall).
export SIDE_CHANNEL_PORT=${SIDE_CHANNEL_PORT:-20099}
export KV_EVENT_PORT=${KV_EVENT_PORT:-22090}

# Local system port for this worker (Dynamo system service).
export SYS_PORT=${SYS_PORT:-8081}

# Model + name (must match what the PD workers serve).
export MODEL=${MODEL:-/mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-8B-Instruct}
export SERVED=${SERVED:-Qwen/Qwen3-VL-8B-Instruct}
export TRANSFER_MODE=${TRANSFER_MODE:-nixl-read}

# Larger TCP message size for multimodal embedding metadata over the control plane.
export DYN_TCP_MAX_MESSAGE_SIZE=268435456  # 256 MB

# ========================================

LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Dynamo SGLang Intel B70 XPU Encode Worker (1E)"
echo "Model: $SERVED"
echo "Pair : PD workers @ $IP_REMOTE (dell07, 4x L40S)"
echo "=========================================="
echo ""
echo "Remote PD host    : $IP_REMOTE  (NATS:$PORT_NATS  etcd:$PORT_ETCD)"
echo "Local B70 fabric  : $UCX_NIC -> $LOCAL_NDEV -> $IP_LOCAL"
echo "XPU device        : $XPU_DEVICE  (NUMA $([ $XPU_DEVICE -lt 4 ] && echo 0 || echo 2))"
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

# Pre-flight: data-plane fabric IP must exist on the chosen netdev
if ! python3 -c "
import socket, fcntl, struct
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ip = socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, struct.pack('256s', b'$LOCAL_NDEV'[:15]))[20:24])
assert ip == '$IP_LOCAL', f'$LOCAL_NDEV has {ip}, expected $IP_LOCAL'
" 2>/dev/null; then
    echo "WARNING: $LOCAL_NDEV ($UCX_NIC) does not have $IP_LOCAL. Verify NIC<->IP mapping."
fi

if ! timeout 2 bash -c "cat </dev/null >/dev/tcp/$IP_REMOTE/$PORT_NATS" 2>/dev/null; then
    read -p "Control plane unreachable. Continue anyway? (y/n) " -n 1 -r
    echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi

# ========================================
# Start Intel B70 XPU Encode Worker
# ========================================
echo ""
echo "Starting Intel B70 XPU Encode Worker (8B)..."
echo "Model load can take a few minutes."

ENC_LOG="$LOG_DIR/encode_xpu_8b_b70.log"

# Optional: pass --mm-attention-backend if MM_ATTN_BACKEND is set.
# Set MM_ATTN_BACKEND=xpu_attn to use the PR 26460 XPU vision-attention path
# (this image is hm_dynamo_b70_pr26460, which has it). Leave unset to use the
# default triton_attn backend. Example:
#   MM_ATTN_BACKEND=xpu_attn ./start_sglang_encode_xpu_8b_b70.sh
MM_ATTN_FLAG=""
if [ -n "${MM_ATTN_BACKEND:-}" ]; then
    MM_ATTN_FLAG="--mm-attention-backend ${MM_ATTN_BACKEND}"
    echo "  MM attention backend: ${MM_ATTN_BACKEND}"
else
    echo "  MM attention backend: (default, triton_attn) — set MM_ATTN_BACKEND=xpu_attn to try PR26460 path"
fi

ZE_AFFINITY_MASK=$XPU_DEVICE \
DYN_SYSTEM_PORT=$SYS_PORT \
NATS_SERVER=nats://${IP_REMOTE}:${PORT_NATS} \
ETCD_ENDPOINTS=http://${IP_REMOTE}:${PORT_ETCD} \
ETCD_REQUEST_TIMEOUT=600 \
DYN_REQUEST_PLANE=tcp \
TRANSFER_LOCAL=0 \
PYTHONHASHSEED=0 \
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP_LOCAL} \
VLLM_NIXL_SIDE_CHANNEL_PORT=${SIDE_CHANNEL_PORT} \
DYN_VLLM_KV_EVENT_PORT=${KV_EVENT_PORT} \
UCX_MEMTYPE_CACHE=0 \
UCX_TLS=ze_copy,rc,tcp \
UCX_NET_DEVICES=${UCX_NIC} \
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
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'$KV_EVENT_PORT'","enable_kv_cache_events":true}' \
    2>&1 | tee -a "$ENC_LOG" &

ENCODE_PID=$!

echo ""
echo "=========================================="
echo "Intel B70 XPU Encode Worker started"
echo "=========================================="
echo "  Model:               $SERVED"
echo "  Remote PD host:      $IP_REMOTE  (NATS:$PORT_NATS  etcd:$PORT_ETCD)"
echo "  Local B70 fabric IP: $IP_LOCAL  (NIC $UCX_NIC)"
echo "  Side-channel:        $IP_LOCAL:$SIDE_CHANNEL_PORT"
echo "  KV events port:      $KV_EVENT_PORT"
echo "  XPU (ZE_AFFINITY):   $XPU_DEVICE"
echo "  System port:         $SYS_PORT"
echo "  PID:                 $ENCODE_PID"
echo ""
echo "  UCX_TLS=ze_copy,rc,tcp   (no cuda_ipc, as required cross-host)"
echo "  DYN_SGL_EMBEDDING_TRANSFER_MODE=$TRANSFER_MODE"
echo ""
echo "Log:    $ENC_LOG"
echo "Tail:   tail -f $ENC_LOG"
echo "Ready:  grep -i 'registered\\|ready' $ENC_LOG"
echo "Stop:   kill $ENCODE_PID   # or: pkill -f 'dynamo.sglang.*multimodal-encode-worker'"
echo ""
echo "Verify from dell07:  curl -s http://${IP_REMOTE}:7001/v1/models"
echo ""
