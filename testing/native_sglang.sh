#!/bin/bash
# native_sglang.sh — run ONE native-SGLang test (no Dynamo) end to end.
#
# Narrowing harness: same workload as the Dynamo harness, but against native
# SGLang servers (no NATS/etcd/frontend, no kv-router, no NIXL — embeddings move
# over SGLang's default ZMQ transport on localhost). See NATIVE_SGLANG.md.
#
# Lifecycle: launch server(s) -> poll /health until ready -> bench -> teardown.
#
# Usage (run INSIDE the patched robin_sglang_dynamo_l40 container):
#   ./native_sglang.sh agg      [out_dir]
#   ./native_sglang.sh 1e1pd    [out_dir]
#   ./native_sglang.sh 2e1pd    [out_dir]
#   ./native_sglang.sh 3e1pd    [out_dir]
#   ./native_sglang.sh 4e1pd    [out_dir]
#
# GPU placement (override via env; defaults match the reference run):
#   PD_GPUS   CUDA indices for the agg / PD server (TP defaults to their count).
#             default "0,1,2,3".
#   ENC_GPUS  comma list of CUDA indices, ONE per encoder. 1e1pd uses the first,
#             2e1pd uses the first two, 3e1pd uses the first three, and 4e1pd
#             uses the first four. default "4,5".
#   ENC_GROUPS optional semicolon-separated CUDA_VISIBLE_DEVICES groups for
#             encoders, e.g. ENC_GROUPS="5,6,7" ENC_TP=3 for one TP3 encoder,
#             or ENC_GROUPS="5;6;7" for three TP1 encoders.
#   ENC_TPS   optional semicolon-separated tensor-parallel sizes matching
#             ENC_GROUPS, e.g. ENC_GROUPS="5,6;7" ENC_TPS="2;1".
# e.g.  PD_GPUS=0,1,2,3 ENC_GPUS=4,5 ./native_sglang.sh 2e1pd
#       PD_GPUS=0,1 TP=2 ./native_sglang.sh agg
#
# Override any parity setting via env too, e.g.:
#   NUM_PROMPTS=64 RATE=2.0 ./native_sglang.sh agg
set -uo pipefail

CASE="${1:-}"
[ -z "$CASE" ] && { echo "usage: $0 <agg|1e1pd|2e1pd|3e1pd|4e1pd> [out_dir]"; exit 2; }

# ---- GPU placement (override via env) ----
PD_GPUS="${PD_GPUS:-0,1,2,3}"   # agg / PD server CUDA indices
ENC_GPUS="${ENC_GPUS:-4,5}"     # one encoder per index (1e1pd uses 1st, 2e1pd 1st two)
IFS=',' read -ra PD_GPU_ARR  <<< "$PD_GPUS"
if [ -n "${ENC_GROUPS:-}" ]; then
  IFS=';' read -ra ENC_GPU_ARR <<< "$ENC_GROUPS"
else
  IFS=',' read -ra ENC_GPU_ARR <<< "$ENC_GPUS"
fi
IFS=';' read -ra ENC_TP_ARR <<< "${ENC_TPS:-}"

# ---- model + parity settings (match the model table / harness defaults) ----
MODEL="${MODEL:-/mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-235B-A22B-Instruct-FP8}"
SERVED="${SERVED:-Qwen/Qwen3-VL-235B-A22B-Instruct-FP8}"
# TP defaults to the number of PD GPUs so placement and tensor-parallel agree.
TP="${TP:-${#PD_GPU_ARR[@]}}"
KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"
MEM_FRAC="${MEM_FRAC:-0.90}"
AGG_MEM_FRAC="${AGG_MEM_FRAC:-$MEM_FRAC}"
PD_MEM_FRAC="${PD_MEM_FRAC:-$MEM_FRAC}"
ENC_MEM_FRAC="${ENC_MEM_FRAC:-0.5}"
ENC_TP="${ENC_TP:-1}"
CHUNKED="${CHUNKED:-32768}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-16384}"
PREFILL_MAX="${PREFILL_MAX:-16}"
MAX_RUNNING="${MAX_RUNNING:-40}"
DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-1}"
LOG_LEVEL="${LOG_LEVEL:-info}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-qwen2-vl}"
AGG_EXTRA_ARGS="${AGG_EXTRA_ARGS:-}"
PD_EXTRA_ARGS="${PD_EXTRA_ARGS:-}"
ENC_EXTRA_ARGS="${ENC_EXTRA_ARGS:-}"

# ---- workload (match the reference run) ----
NUM_PROMPTS="${NUM_PROMPTS:-32}"
IMAGE_COUNT="${IMAGE_COUNT:-8}"
IMAGE_RES="${IMAGE_RES:-1080p}"
INPUT_LEN="${INPUT_LEN:-128}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
RATE="${RATE:-1.0}"
OUTPUT_DETAILS="${OUTPUT_DETAILS:-0}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-0}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-}"
BENCH_PYTHONPATH="${BENCH_PYTHONPATH:-}"

# ---- ports ----
PD_PORT="${PD_PORT:-30000}"        # agg / language-only server (bench hits this)
ENC_PORT_BASE="${ENC_PORT_BASE:-30002}"

READY_TIMEOUT="${READY_TIMEOUT:-1800}"   # 235B weight load is ~14 min

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${2:-$HERE/results/native_sglang/$CASE}"
mkdir -p "$OUT_DIR"
PIDS=()   # launched server PIDs, for teardown

log() { echo "[$(date +%H:%M:%S)] $*"; }

radix_args() {
  [ "$DISABLE_RADIX_CACHE" = "1" ] && printf '%s\n' "--disable-radix-cache"
}

bench_detail_args() {
  [ "$OUTPUT_DETAILS" = "1" ] && printf '%s\n' "--output-details"
}

chat_template_args() {
  [ -n "$CHAT_TEMPLATE" ] && printf '%s\n' "--chat-template" "$CHAT_TEMPLATE"
}

# Poll a server's /health until it answers 200, or time out.
wait_ready() {  # <port> <name>
  local port="$1" name="$2" deadline=$(( $(date +%s) + READY_TIMEOUT ))
  log "  waiting for $name on :$port (timeout ${READY_TIMEOUT}s, ~14 min for 235B)..."
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health" 2>/dev/null | grep -q 200; then
      log "  READY: $name on :$port"
      return 0
    fi
    sleep 5
  done
  log "  TIMEOUT: $name on :$port never became ready"
  return 1
}

teardown() {
  log "teardown: stopping launched servers..."
  for p in "${PIDS[@]:-}"; do [ -n "$p" ] && { kill "$p" 2>/dev/null; kill -- "-$p" 2>/dev/null; }; done
  # backstop: port-scoped (only the ports THIS script uses)
  pkill -9 -f "launch_server.*--port $PD_PORT" 2>/dev/null
  local enc_port
  for ((enc_port = ENC_PORT_BASE; enc_port < ENC_PORT_BASE + 10; enc_port++)); do
    pkill -9 -f "launch_server.*--port $enc_port" 2>/dev/null
  done
  sleep 2
}
trap teardown EXIT INT TERM

# ---- server launchers ----
launch_agg() {  # gpus
  log "launch AGG: gpus=$1 tp=$TP -> $OUT_DIR/agg.log"
  CUDA_VISIBLE_DEVICES="$1" setsid python3 -m sglang.launch_server \
    --model-path "$MODEL" --served-model-name "$SERVED" \
    --tensor-parallel-size "$TP" \
    --enable-multimodal $(chat_template_args) \
    --kv-cache-dtype "$KV_DTYPE" --mem-fraction-static "$AGG_MEM_FRAC" \
    --chunked-prefill-size "$CHUNKED" --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
    --prefill-max-requests "$PREFILL_MAX" \
    --max-running-requests "$MAX_RUNNING" $(radix_args) \
    --log-level "$LOG_LEVEL" \
    --trust-remote-code --page-size 16 \
    --host 0.0.0.0 --port "$PD_PORT" \
    $AGG_EXTRA_ARGS \
    > "$OUT_DIR/agg.log" 2>&1 &
  PIDS+=($!)
}

launch_encoder() {  # gpu port name
  local enc_tp="${4:-$ENC_TP}"
  log "launch ENCODER: gpu=$1 tp=$enc_tp port=$2 -> $OUT_DIR/$3.log"
  CUDA_VISIBLE_DEVICES="$1" setsid python3 -m sglang.launch_server \
    --model-path "$MODEL" --served-model-name "$SERVED" \
    --tensor-parallel-size "$enc_tp" \
    --encoder-only --enable-multimodal $(chat_template_args) \
    --enable-prefix-mm-cache --mem-fraction-static "$ENC_MEM_FRAC" \
    --log-level "$LOG_LEVEL" \
    --trust-remote-code --page-size 16 \
    --host 0.0.0.0 --port "$2" \
    $ENC_EXTRA_ARGS \
    > "$OUT_DIR/$3.log" 2>&1 &
  PIDS+=($!)
}

encoder_tp_at() {
  local idx="$1"
  if [ "${#ENC_TP_ARR[@]}" -gt "$idx" ] && [ -n "${ENC_TP_ARR[$idx]}" ]; then
    printf '%s\n' "${ENC_TP_ARR[$idx]}"
  else
    printf '%s\n' "$ENC_TP"
  fi
}

launch_pd() {  # gpus "url1 url2..."
  log "launch PD (language-only): gpus=$1 tp=$TP encoders=[$2] -> $OUT_DIR/pd.log"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$1" setsid python3 -m sglang.launch_server \
    --model-path "$MODEL" --served-model-name "$SERVED" \
    --tensor-parallel-size "$TP" \
    --language-only --encoder-urls $2 \
    $(chat_template_args) \
    --kv-cache-dtype "$KV_DTYPE" --mem-fraction-static "$PD_MEM_FRAC" \
    --chunked-prefill-size "$CHUNKED" --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
    --prefill-max-requests "$PREFILL_MAX" \
    --max-running-requests "$MAX_RUNNING" $(radix_args) \
    --log-level "$LOG_LEVEL" \
    --trust-remote-code --page-size 16 \
    --host 0.0.0.0 --port "$PD_PORT" \
    $PD_EXTRA_ARGS \
    > "$OUT_DIR/pd.log" 2>&1 &
  PIDS+=($!)
}

run_bench() {  # label
  local out_json="$OUT_DIR/bench_native_$1.json"
  local out_txt="$OUT_DIR/bench_native_$1.txt"
  local result_txt="$OUT_DIR/result_${1}_r${RATE}.txt"
  local bench_status
  local -a bench_cmd
  local -a bench_env=()
  log "bench: label=$1 np=$NUM_PROMPTS img=$IMAGE_COUNT res=$IMAGE_RES rate=$RATE -> $out_json"
  if [ -n "$BENCH_PYTHONPATH" ]; then
    bench_env=(env PYTHONPATH="$BENCH_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}")
  fi
  bench_cmd=(
    python3 -m sglang.bench_serving
    --model "$SERVED" --backend sglang-oai-chat \
    --host 127.0.0.1 --port "$PD_PORT" \
    --dataset-name image --num-prompts "$NUM_PROMPTS" \
    --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
    --image-count "$IMAGE_COUNT" --image-resolution "$IMAGE_RES" \
    --request-rate "$RATE" --apply-chat-template --seed 0 \
    --disable-tqdm --output-file "$out_json"
  )
  [ "$OUTPUT_DETAILS" = "1" ] && bench_cmd+=(--output-details)
  [ -n "$MAX_CONCURRENCY" ] && bench_cmd+=(--max-concurrency "$MAX_CONCURRENCY")

  if [ "$BENCH_TIMEOUT" != "0" ]; then
    timeout --foreground "$BENCH_TIMEOUT" "${bench_env[@]}" "${bench_cmd[@]}" 2>&1 | tee "$out_txt" "$result_txt"
    bench_status="${PIPESTATUS[0]}"
    if [ "$bench_status" -eq 124 ]; then
      log "bench timed out: label=$1 timeout=$BENCH_TIMEOUT"
    fi
  else
    "${bench_env[@]}" "${bench_cmd[@]}" 2>&1 | tee "$out_txt" "$result_txt"
    bench_status="${PIPESTATUS[0]}"
  fi
  return "$bench_status"
}

run_bench_or_exit() {  # label
  local status
  run_bench "$1"
  status=$?
  if [ "$status" -ne 0 ]; then
    log "bench failed: label=$1 status=$status"
    exit "$status"
  fi
}

# ---- dispatch ----
# Guard: enough encoder GPUs for the requested case.
need_enc() {  # <n> — require at least n entries in ENC_GPU_ARR
  if [ "${#ENC_GPU_ARR[@]}" -lt "$1" ]; then
    echo "case '$CASE' needs $1 encoder GPU(s) but ENC_GPUS='$ENC_GPUS' has ${#ENC_GPU_ARR[@]}"; exit 2
  fi
}

case "$CASE" in
  agg)
    launch_agg "$PD_GPUS"
    wait_ready "$PD_PORT" "agg" || exit 3
    run_bench_or_exit "1AGG"
    ;;
  1e1pd)
    need_enc 1
    launch_encoder "${ENC_GPU_ARR[0]}" "$ENC_PORT_BASE" "enc0" "$(encoder_tp_at 0)"
    wait_ready "$ENC_PORT_BASE" "enc0" || exit 3
    launch_pd "$PD_GPUS" "http://127.0.0.1:$ENC_PORT_BASE"
    wait_ready "$PD_PORT" "pd" || exit 3
    run_bench_or_exit "1E1PD"
    ;;
  2e1pd)
    need_enc 2
    enc1=$ENC_PORT_BASE; enc2=$((ENC_PORT_BASE + 1))
    launch_encoder "${ENC_GPU_ARR[0]}" "$enc1" "enc0" "$(encoder_tp_at 0)"
    launch_encoder "${ENC_GPU_ARR[1]}" "$enc2" "enc1" "$(encoder_tp_at 1)"
    wait_ready "$enc1" "enc0" || exit 3
    wait_ready "$enc2" "enc1" || exit 3
    launch_pd "$PD_GPUS" "http://127.0.0.1:$enc1 http://127.0.0.1:$enc2"
    wait_ready "$PD_PORT" "pd" || exit 3
    run_bench_or_exit "2E1PD"
    ;;
  3e1pd)
    need_enc 3
    enc1=$ENC_PORT_BASE; enc2=$((ENC_PORT_BASE + 1)); enc3=$((ENC_PORT_BASE + 2))
    launch_encoder "${ENC_GPU_ARR[0]}" "$enc1" "enc0" "$(encoder_tp_at 0)"
    launch_encoder "${ENC_GPU_ARR[1]}" "$enc2" "enc1" "$(encoder_tp_at 1)"
    launch_encoder "${ENC_GPU_ARR[2]}" "$enc3" "enc2" "$(encoder_tp_at 2)"
    wait_ready "$enc1" "enc0" || exit 3
    wait_ready "$enc2" "enc1" || exit 3
    wait_ready "$enc3" "enc2" || exit 3
    launch_pd "$PD_GPUS" "http://127.0.0.1:$enc1 http://127.0.0.1:$enc2 http://127.0.0.1:$enc3"
    wait_ready "$PD_PORT" "pd" || exit 3
    run_bench_or_exit "3E1PD"
    ;;
  4e1pd)
    need_enc 4
    enc1=$ENC_PORT_BASE; enc2=$((ENC_PORT_BASE + 1)); enc3=$((ENC_PORT_BASE + 2)); enc4=$((ENC_PORT_BASE + 3))
    launch_encoder "${ENC_GPU_ARR[0]}" "$enc1" "enc0" "$(encoder_tp_at 0)"
    launch_encoder "${ENC_GPU_ARR[1]}" "$enc2" "enc1" "$(encoder_tp_at 1)"
    launch_encoder "${ENC_GPU_ARR[2]}" "$enc3" "enc2" "$(encoder_tp_at 2)"
    launch_encoder "${ENC_GPU_ARR[3]}" "$enc4" "enc3" "$(encoder_tp_at 3)"
    wait_ready "$enc1" "enc0" || exit 3
    wait_ready "$enc2" "enc1" || exit 3
    wait_ready "$enc3" "enc2" || exit 3
    wait_ready "$enc4" "enc3" || exit 3
    launch_pd "$PD_GPUS" "http://127.0.0.1:$enc1 http://127.0.0.1:$enc2 http://127.0.0.1:$enc3 http://127.0.0.1:$enc4"
    wait_ready "$PD_PORT" "pd" || exit 3
    run_bench_or_exit "4E1PD"
    ;;
  *)
    echo "unknown case '$CASE' (agg|1e1pd|2e1pd|3e1pd|4e1pd)"; exit 2 ;;
esac

log "done. results in $OUT_DIR"
# teardown runs on EXIT
