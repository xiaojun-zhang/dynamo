#!/usr/bin/env bash
# Launch 1 Encode worker + 4 PD workers + frontend for Qwen3-VL EPD disaggregation.
# Encode = GPU 0; PD = GPUs 1,2,3,4. Each PD gets a unique system port + log file.
# Usage: ./launch_1e_4pd.sh
#   Env overrides: MODEL, SERVED, TRANSFER_MODE (default nixl-read), PIN_UCX=1
set -u

cd "$(dirname "$0")/.." || exit 1   # run from the dynamo repo root

export HF_HOME=/home/h-zheng/robin/hf_cache
export DYN_DISCOVERY_BACKEND=file DYN_EVENT_PLANE=zmq DYN_REQUEST_PLANE=tcp
export DYN_TCP_MAX_MESSAGE_SIZE=134217728   # 8x1080p images exceed the 32 MiB default

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
SERVED="${SERVED:-Qwen/Qwen3-VL-8B-Instruct}"
TRANSFER_MODE="${TRANSFER_MODE:-nixl-read}"

# Optional: pin UCX to intra-node transports to avoid the multi-NIC/CNI
# transport-selection problems that caused NIXL_ERR_REMOTE_DISCONNECT.
#   - cuda_ipc/cuda_copy/sm/self = intra-node data movement (same-node GPU<->GPU/host)
#   - tcp = REQUIRED for NIXL's Active-Messages notification/handshake channel
#           (the intra-node TLs provide "no am bcopy"; without tcp UCX fails with
#            "no active messages transport" -> NIXL_ERR_BACKEND)
#   - UCX_NET_DEVICES=lo confines tcp to loopback (same node), avoiding the
#     eno*/cali*/RDMA (mlx5_*) interfaces. NOTE: GPUDirect RDMA is unavailable here
#     (nvidia_peermem not loaded), which is why RDMA lanes disconnected.
# Enable with: PIN_UCX=1 ./launch_1e_4pd.sh
if [[ "${PIN_UCX:-0}" == "1" ]]; then
  export UCX_TLS=cuda_ipc,cuda_copy,sm,self,tcp
  export UCX_NET_DEVICES=lo
  export UCX_LOG_LEVEL=info
  echo "[launch] UCX pinned: UCX_TLS=$UCX_TLS UCX_NET_DEVICES=$UCX_NET_DEVICES"
fi

echo "[launch] MODEL=$MODEL  TRANSFER_MODE=$TRANSFER_MODE"
echo "[launch] logs in: $(pwd)/logfile_*.log"

# --- frontend (default round-robin balances the 4 homogeneous PD workers) ---
python3 -m dynamo.frontend --http-port 8000 \
  --discovery-backend file --event-plane zmq \
  > logfile_frontend_disagg.log 2>&1 &
echo "[launch] frontend pid=$! -> logfile_frontend_disagg.log"

# --- 1 encode worker on GPU 0 ---
CUDA_VISIBLE_DEVICES=0 DYN_SYSTEM_PORT=8081 python3 -m dynamo.sglang \
  --multimodal-encode-worker \
  --model-path "$MODEL" --served-model-name "$SERVED" \
  --chat-template qwen2-vl \
  --embedding-transfer-mode "$TRANSFER_MODE" \
  --skip-tokenizer-init --trust-remote-code \
  --discovery-backend file --event-plane zmq \
  > logfile_encode_disagg.log 2>&1 &
echo "[launch] encode  pid=$! gpu=0 port=8081 -> logfile_encode_disagg.log"

# --- 4 PD workers on GPUs 1..4, unique port + log each ---
for gpu in 1 2 3 4; do
  port=$((8081 + gpu))   # 8082,8083,8084,8085
  CUDA_VISIBLE_DEVICES=$gpu DYN_SYSTEM_PORT=$port python3 -m dynamo.sglang \
    --multimodal-worker \
    --model-path "$MODEL" --served-model-name "$SERVED" \
    --embedding-transfer-mode "$TRANSFER_MODE" \
    --page-size 16 --tp 1 \
    --prefill-max-requests 1 \
    --log-level debug \
    --trust-remote-code --skip-tokenizer-init --disable-radix-cache \
    --discovery-backend file --event-plane zmq \
    --disaggregation-transfer-backend nixl \
    > "logfile_pd_disagg_gpu${gpu}.log" 2>&1 &
  echo "[launch] pd      pid=$! gpu=$gpu port=$port -> logfile_pd_disagg_gpu${gpu}.log"
done

echo
echo "[launch] All processes started in background."
echo "[launch] Wait for models to load (watch: tail -f logfile_pd_disagg_gpu1.log),"
echo "[launch] then run: ./tool_scripts/sanity_check.sh"
