#!/bin/bash
# Cross-host disagg PD-only start script for dell07 (this L40S host)
# Pairs with: 1 encode worker running on the Intel XPU (B70) host.
#
# This host (dell07):   hostname=sc09dell07-rtx, mgmt IP=172.26.46.178
# This script starts:   NATS, etcd, frontend, and 4 PD workers (4 L40S GPUs)
# Encoder runs on:      sc09giga01-b70.sc.intel.com (172.26.46.180) — start separately
#
# Model: Qwen/Qwen3-VL-8B-Instruct, TP=1 per worker
# Derived from: hongming/dynamo/01_cuda_sh/disagg_h200_32b/start_sglang_pd_cuda_32b_fp8_giga01.sh
#
# Topology notes:
#   - 4 PD (multimodal) workers, one per L40S GPU (GPUs 1,2,3,4 by default).
#   - Each worker registers with etcd; the frontend (kv router) load-balances
#     incoming requests across the 4 workers.
#   - The remote XPU encoder pushes/serves image embeddings to the PD workers
#     over NIXL (RDMA, via mlx5_0 RoCE).
#   - Discovery = etcd, event plane = nats (REQUIRED for cross-host: the file/zmq
#     backends in the sample only work when E and PD share a filesystem/host).
#
# IMPORTANT: Before running, ensure the firewall on dell07 allows from the B70 host:
#   - $PORT_NATS  (NATS, 14222)
#   - $PORT_ETCD  (etcd client, 12379)
#   - 12380       (etcd peer)
#   - RDMA traffic on the RoCE NIC (mlx5_0 / 192.165.123.65)

set -e

# ========================================
# PROXY (corporate proxy intercepts localhost; bypass for local + RoCE + B70)
# ========================================
export http_proxy=http://proxy.ims.intel.com:911
export https_proxy=http://proxy.ims.intel.com:911
export ftp_proxy=http://proxy.ims.intel.com:911
# Bypass proxy for: localhost, this host mgmt+RoCE IPs, the B70 encoder host
export no_proxy=0.0.0.0,127.0.0.1,localhost,172.26.46.178,192.165.123.65,172.26.46.180,.intel.com
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
export NO_PROXY=$no_proxy

# ========================================
# CONFIGURATION
# ========================================

# Two networks on this host:
#  - IP_LOCAL_MGMT: management net (172.26.46.0/22 on eno16895np0).
#    Used for NATS/etcd advertise URLs and frontend HTTP. The B70 encoder
#    reaches this for control-plane registration.
#  - IP_LOCAL_ROCE: RoCE fabric (192.165.123.0/24 on eno17295np0 / mlx5_0).
#    Used as the NIXL side-channel host so the encoder dials this for the
#    embedding data plane. Must match UCX_NIC below.
export IP_LOCAL_MGMT=172.26.46.178
export IP_LOCAL_ROCE=192.165.123.65
export IP_LOCAL=$IP_LOCAL_MGMT  # control plane (NATS/etcd/frontend)

export PORT_NATS=14222
export PORT_ETCD=12379
export PORT_HTTP=7001

# PD worker GPUs — one PD (multimodal) worker per L40S.
# Override on the command line, e.g. PD_GPUS="1 2" ./start_...sh
export PD_GPUS="${PD_GPUS:-1 2 3 4}"

# RDMA NIC. mlx5_0 = eno17295np0 = 192.165.123.65 (RoCE fabric to the B70 host).
export UCX_NIC=mlx5_0:1

# Model + name
export MODEL=/mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-8B-Instruct
export SERVED=Qwen/Qwen3-VL-8B-Instruct

# Cross-host PD reads embeddings produced by the remote encoder.
export TRANSFER_MODE=nixl-read

# Larger TCP message size + HTTP body limit for multimodal payloads.
export DYN_TCP_MAX_MESSAGE_SIZE=268435456  # 256 MB
export DYN_HTTP_BODY_LIMIT_MB=256

# Base port for each worker's DYN_SYSTEM_PORT (worker i -> base + gpu).
export SYS_PORT_BASE=8081
# Base port for each worker's ZMQ KV-event publisher (distinct per worker).
export KV_EVENT_PORT_BASE=22080

# ========================================

LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Dynamo SGLang Cross-Host Disagg — PD side (dell07, 4x L40S)"
echo "Model: $SERVED"
echo "=========================================="
echo ""
echo "This host (dell07):"
echo "  mgmt IP:  $IP_LOCAL_MGMT  (control plane)"
echo "  RoCE IP:  $IP_LOCAL_ROCE  (NIXL data plane, $UCX_NIC)"
echo "Encoder runs on:  sc09giga01-b70.sc.intel.com (172.26.46.180) — start separately"
echo ""
echo "Configuration:"
echo "  - PD Worker GPUs: $PD_GPUS"
echo "  - Frontend HTTP:  http://$IP_LOCAL_MGMT:$PORT_HTTP"
echo "  - NATS:           nats://$IP_LOCAL_MGMT:$PORT_NATS"
echo "  - etcd:           http://$IP_LOCAL_MGMT:$PORT_ETCD"
echo "  - transfer mode:  $TRANSFER_MODE"
echo ""

# ========================================
# Start NATS (bind 0.0.0.0 so the remote encoder can connect)
# ========================================
echo "Starting NATS..."
nats-server -js -a 0.0.0.0 -p $PORT_NATS -m 18222 \
    > "$LOG_DIR/nats_dell07.log" 2>&1 &
NATS_PID=$!
sleep 2

# ========================================
# Start etcd (bind 0.0.0.0 so the remote encoder can register)
# ========================================
echo "Starting etcd..."
rm -rf /tmp/etcd-sglang-pd-8b-dell07-$$
etcd \
  --listen-client-urls=http://0.0.0.0:$PORT_ETCD \
  --advertise-client-urls=http://$IP_LOCAL:$PORT_ETCD \
  --listen-peer-urls=http://0.0.0.0:12380 \
  --initial-advertise-peer-urls=http://0.0.0.0:12380 \
  --initial-cluster=default=http://0.0.0.0:12380 \
  --data-dir=/tmp/etcd-sglang-pd-8b-dell07-$$ \
  > "$LOG_DIR/etcd_dell07.log" 2>&1 &
ETCD_PID=$!
sleep 5

# Verify etcd is reachable
for i in 1 2 3 4 5 6 7 8 9 10; do
    if curl -s -o /dev/null "http://localhost:$PORT_ETCD/version"; then
        echo "  etcd reachable on attempt $i"
        break
    fi
    sleep 2
done

# ========================================
# Start Frontend (kv router, etcd discovery, nats event plane)
# ========================================
echo "Starting Frontend..."
NATS_SERVER=nats://$IP_LOCAL:$PORT_NATS \
ETCD_ENDPOINTS=http://$IP_LOCAL:$PORT_ETCD \
ETCD_LEASE_TTL=600 \
ETCD_REQUEST_TIMEOUT=600 \
DYN_REQUEST_PLANE=tcp \
DYN_LOG=debug \
SGLANG_LOG_LEVEL=debug \
python3 -m dynamo.frontend \
    --http-port $PORT_HTTP \
    --router-mode kv \
    --router-reset-states \
    > "$LOG_DIR/frontend_dell07.log" 2>&1 &
FRONTEND_PID=$!
sleep 5

# ========================================
# Start 4 PD Workers (one per GPU)
# ========================================
# Notes on env vars:
#  - UCX_TLS drops cuda_ipc (no NVLink/shared GPU bus across hosts); keeps
#    IB/RC verbs + cuda_copy so the RDMA path is selected for cross-host NIXL.
#  - DYN_SYSTEM_PORT must be unique per worker on this host.
#  - DYN_SGL_EMBEDDING_TRANSFER_MODE / --embedding-transfer-mode = nixl-read.
declare -a PD_PIDS=()
for gpu in $PD_GPUS; do
    sys_port=$((SYS_PORT_BASE + gpu))        # 8082, 8083, 8084, 8085
    kv_port=$((KV_EVENT_PORT_BASE + gpu))    # 22081, 22082, 22083, 22084
    pd_log="$LOG_DIR/pd_worker_gpu${gpu}.log"
    echo "Starting PD worker on GPU $gpu (sys_port=$sys_port, kv_port=$kv_port)..."

    CUDA_VISIBLE_DEVICES=$gpu \
    DYN_SYSTEM_PORT=$sys_port \
    NATS_SERVER=nats://${IP_LOCAL_MGMT}:${PORT_NATS} \
    ETCD_ENDPOINTS=http://${IP_LOCAL_MGMT}:${PORT_ETCD} \
    ETCD_LEASE_TTL=600 \
    DYN_REQUEST_PLANE=tcp \
    DYN_LOG=debug \
    TRANSFER_LOCAL=0 \
    PYTHONHASHSEED=0 \
    VLLM_NIXL_SIDE_CHANNEL_HOST=${IP_LOCAL_ROCE} \
    UCX_TLS=ib,rc,ud,rc_verbs,ud_verbs,cuda_copy \
    UCX_NET_DEVICES=${UCX_NIC} \
    UCX_MEMTYPE_CACHE=0 \
    DYN_SGL_EMBEDDING_TRANSFER_MODE=${TRANSFER_MODE} \
    ENABLE_ENCODER_CACHE=0 \
    NCCL_DEBUG=INFO \
    NCCL_DEBUG_SUBSYS=INIT,P2P \
    python3 -m dynamo.sglang \
        --multimodal-worker \
        --model-path "$MODEL" \
        --served-model-name "$SERVED" \
        --embedding-transfer-mode "$TRANSFER_MODE" \
        --page-size 16 \
        --tp 1 \
        --prefill-max-requests 1 \
        --log-level debug \
        --trust-remote-code \
        --skip-tokenizer-init \
        --disable-radix-cache \
        --discovery-backend etcd \
        --event-plane nats \
        --disaggregation-transfer-backend nixl \
        > "$pd_log" 2>&1 &
    pid=$!
    PD_PIDS+=("$pid")
    echo "  [launch] pd pid=$pid gpu=$gpu port=$sys_port -> $pd_log"
done

# ========================================
# Status
# ========================================
echo ""
echo "=========================================="
echo "PD-side services started (waiting for model load)"
echo "=========================================="
echo ""
echo "Process IDs:"
echo "  - NATS:      $NATS_PID"
echo "  - etcd:      $ETCD_PID"
echo "  - Frontend:  $FRONTEND_PID"
echo "  - PD Workers: ${PD_PIDS[*]}"
echo ""
echo "Logs in: $LOG_DIR/"
echo ""
echo "=========================================="
echo "Next steps on the B70 encoder host (172.26.46.180):"
echo "=========================================="
echo "  Point the encoder at this host's control plane:"
echo "     NATS_SERVER=nats://${IP_LOCAL_MGMT}:${PORT_NATS}"
echo "     ETCD_ENDPOINTS=http://${IP_LOCAL_MGMT}:${PORT_ETCD}"
echo "     DYN_SGL_EMBEDDING_TRANSFER_MODE=${TRANSFER_MODE}"
echo "     UCX_TLS=ib,rc,ud,rc_verbs,ud_verbs,cuda_copy   # no cuda_ipc (cross-host)"
echo "     (run with --multimodal-encode-worker --discovery-backend etcd --event-plane nats)"
echo ""

# ========================================
# Wait for all 4 PD workers to register
# ========================================
echo "Waiting for $(echo $PD_GPUS | wc -w) PD workers to register (8B loads in ~1-3 min each)..."
NWANT=$(echo $PD_GPUS | wc -w)
for i in {1..60}; do
    sleep 5
    nreg=$(curl -s http://localhost:$PORT_HTTP/v1/models 2>/dev/null | grep -o "Qwen3-VL-8B-Instruct" | head -1 | wc -l)
    # /v1/models shows the model once it is served; check worker health for count
    backends=$(curl -s http://localhost:$PORT_HTTP/health 2>/dev/null | grep -o '"endpoint":"generate"' | wc -l)
    if [ "$backends" -ge "$NWANT" ]; then
        echo ""
        echo "All $NWANT PD workers registered. Now start the B70 encoder."
        echo "  Models: curl -s http://localhost:$PORT_HTTP/v1/models"
        exit 0
    fi
    echo "  Waiting... ($((i*5))s) backends=$backends/$NWANT"
done

echo ""
echo "Timeout waiting for all PD workers."
echo "  Check: tail -100 $LOG_DIR/pd_worker_gpu*.log"
exit 1
