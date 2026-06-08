#!/bin/bash
# Aggregated EPD start script for dell07 (this L40S host).
# No disaggregation: each worker runs the FULL Encode + Prefill + Decode pipeline
# in one process (vision tower + LLM together). No remote encoder, no NIXL transfer.
#
# This host (dell07):  hostname=sc09dell07-rtx, mgmt IP=172.26.46.178
# This script starts:  NATS, etcd, frontend, and 4 aggregated EPD workers
#                       (one per L40S GPU; GPUs 1,2,3,4 by default).
#
# Model: Qwen/Qwen3-VL-8B-Instruct, TP=1 per worker
# Derived from:
#   - ../disagg_cuda_docker/start_sglang_pd_cuda_8b_dell07.sh (structure, control plane)
#   - hongming/dynamo/01_cuda_sh/agg_h200_32b/start_h200_aggregate_epd_server_32b_tp1.sh
#
# Difference vs the disagg PD script:
#   - NO --multimodal-worker / --multimodal-encode-worker / --encoder-only
#     (aggregated = encode + prefill + decode all in one worker)
#   - NO NIXL embedding-transfer env/flags (nothing is transferred between procs)
#   - KEEP --enable-multimodal + --chat-template (the agg worker also does vision encode)
#   - The frontend kv-router load-balances requests across the 4 agg workers.

set -e

# ========================================
# PROXY (corporate proxy intercepts localhost; bypass for local services)
# ========================================
export http_proxy=http://proxy.ims.intel.com:911
export https_proxy=http://proxy.ims.intel.com:911
export ftp_proxy=http://proxy.ims.intel.com:911
export no_proxy=0.0.0.0,127.0.0.1,localhost,172.26.46.178,192.165.123.65,.intel.com
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
export NO_PROXY=$no_proxy

# ========================================
# CONFIGURATION
# ========================================
export IP_LOCAL=172.26.46.178   # dell07 mgmt IP (control plane)
export PORT_NATS=14222
export PORT_ETCD=12379
export PORT_HTTP=7001

# Aggregated EPD worker GPUs — one full EPD worker per L40S.
export AGG_GPUS="1 2 3 4"

# Model + name
export MODEL=/mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-8B-Instruct
export SERVED=Qwen/Qwen3-VL-8B-Instruct

# Worker tuning.
# NOTE: mem-fraction is the static pool (weights + KV). The AGGREGATED worker also
# does vision encode in-process, whose activations for 8x1080p are large, so the
# L40S (44 GiB) needs real headroom. 0.85 (the H200 value) OOMs here — use ~0.70,
# drop to 0.60 if a worker still dies on heavy image workloads.
export MAX_RUNNING=${MAX_RUNNING:-40}
export MEM_FRAC=${MEM_FRAC:-0.70}

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
echo "Dynamo SGLang Aggregated EPD (dell07, 4x L40S)"
echo "Model: $SERVED"
echo "Mode : aggregated (encode+prefill+decode in each worker, no disagg)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Agg EPD GPUs:   $AGG_GPUS  (one full EPD worker each)"
echo "  - Frontend HTTP:  http://$IP_LOCAL:$PORT_HTTP"
echo "  - NATS:           nats://$IP_LOCAL:$PORT_NATS"
echo "  - etcd:           http://$IP_LOCAL:$PORT_ETCD"
echo "  - max-running:    $MAX_RUNNING"
echo "  - mem-fraction:   $MEM_FRAC"
echo ""

# ========================================
# Start NATS
# ========================================
echo "Starting NATS..."
nats-server -js -a 0.0.0.0 -p $PORT_NATS -m 18222 \
    > "$LOG_DIR/nats_dell07.log" 2>&1 &
NATS_PID=$!
sleep 2

# ========================================
# Start etcd
# ========================================
echo "Starting etcd..."
rm -rf /tmp/etcd-sglang-agg-8b-dell07-$$
etcd \
  --listen-client-urls=http://0.0.0.0:$PORT_ETCD \
  --advertise-client-urls=http://$IP_LOCAL:$PORT_ETCD \
  --listen-peer-urls=http://0.0.0.0:12380 \
  --initial-advertise-peer-urls=http://0.0.0.0:12380 \
  --initial-cluster=default=http://0.0.0.0:12380 \
  --data-dir=/tmp/etcd-sglang-agg-8b-dell07-$$ \
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
# Start 4 Aggregated EPD Workers (one per GPU)
# ========================================
# Notes:
#  - Aggregated = encode + prefill + decode in one process; no --multimodal-worker,
#    no --encoder-only, no NIXL transfer (nothing crosses a process boundary).
#  - --enable-multimodal + --chat-template keep the in-process vision encoder.
#  - DYN_SYSTEM_PORT must be unique per worker on this host.
declare -a AGG_PIDS=()
for gpu in $AGG_GPUS; do
    sys_port=$((SYS_PORT_BASE + gpu))        # 8082, 8083, 8084, 8085
    kv_port=$((KV_EVENT_PORT_BASE + gpu))    # 22081, 22082, 22083, 22084
    agg_log="$LOG_DIR/agg_worker_gpu${gpu}.log"
    echo "Starting aggregated EPD worker on GPU $gpu (sys_port=$sys_port, kv_port=$kv_port)..."

    CUDA_VISIBLE_DEVICES=$gpu \
    DYN_SYSTEM_PORT=$sys_port \
    NATS_SERVER=nats://${IP_LOCAL}:${PORT_NATS} \
    ETCD_ENDPOINTS=http://${IP_LOCAL}:${PORT_ETCD} \
    ETCD_LEASE_TTL=600 \
    DYN_REQUEST_PLANE=tcp \
    DYN_LOG=debug \
    TRANSFER_LOCAL=0 \
    PYTHONHASHSEED=0 \
    DYN_VLLM_KV_EVENT_PORT=${kv_port} \
    ENABLE_ENCODER_CACHE=0 \
    NCCL_DEBUG=INFO \
    NCCL_DEBUG_SUBSYS=INIT,P2P \
    python3 -m dynamo.sglang \
        --model-path "$MODEL" \
        --served-model-name "$SERVED" \
        --enable-multimodal \
        --chat-template qwen2-vl \
        --trust-remote-code \
        --dtype auto \
        --max-running-requests $MAX_RUNNING \
        --tp 1 \
        --mem-fraction-static $MEM_FRAC \
        --page-size 16 \
        --chunked-prefill-size 16384 \
        --discovery-backend etcd \
        --event-plane nats \
        --log-level debug \
        --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'$kv_port'","enable_kv_cache_events":true}' \
        > "$agg_log" 2>&1 &
    pid=$!
    AGG_PIDS+=("$pid")
    echo "  [launch] agg pid=$pid gpu=$gpu port=$sys_port -> $agg_log"
done

# ========================================
# Status
# ========================================
echo ""
echo "=========================================="
echo "Aggregated EPD services started (waiting for model load)"
echo "=========================================="
echo ""
echo "Process IDs:"
echo "  - NATS:        $NATS_PID"
echo "  - etcd:        $ETCD_PID"
echo "  - Frontend:    $FRONTEND_PID"
echo "  - Agg Workers: ${AGG_PIDS[*]}"
echo ""
echo "Logs in: $LOG_DIR/"
echo ""

# ========================================
# Wait for all agg workers to register
# ========================================
NWANT=$(echo $AGG_GPUS | wc -w)
echo "Waiting for $NWANT aggregated EPD workers to register (8B loads in ~1-3 min each)..."
for i in {1..60}; do
    sleep 5
    backends=$(curl -s http://localhost:$PORT_HTTP/health 2>/dev/null | grep -o "backend/generate" | wc -l)
    if [ "$backends" -ge "$NWANT" ]; then
        echo ""
        echo "All $NWANT aggregated EPD workers registered."
        echo "  Models: curl -s http://localhost:$PORT_HTTP/v1/models"
        echo "  Bench:  point sglang.bench_serving at http://$IP_LOCAL:$PORT_HTTP"
        exit 0
    fi
    echo "  Waiting... ($((i*5))s) backends=$backends/$NWANT"
done

echo ""
echo "Timeout waiting for all aggregated EPD workers."
echo "  Check: tail -100 $LOG_DIR/agg_worker_gpu*.log"
exit 1
