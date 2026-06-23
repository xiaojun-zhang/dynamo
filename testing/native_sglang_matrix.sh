#!/bin/bash
# Run the recommended native-SGLang AGG vs E/PD matrix.
#
# Delegates server lifecycle and benchmarking to native_sglang.sh; this wrapper
# only sets matrix placement and workload defaults.
set -euo pipefail

usage() {
  cat <<'EOF'
usage: ./native_sglang_matrix.sh [options]

Options:
  --run-root DIR             Output root. Default: results/native_sglang_matrix_<timestamp>
  --rates LIST               Space-separated request rates. Default: "1.0"
  --cases LIST               Space-separated cases. Default: "agg 1e1pd 2e1pd 3e1pd"
  --agg-gpus LIST            AGG CUDA indices. Default: same as --pd-gpus
  --pd-gpus LIST             PD CUDA indices. Default: "1,2,3,6"
  --enc-gpus-1e1pd LIST      Encoder CUDA indices for 1E1PD. Default: "7"
  --enc-gpus-2e1pd LIST      Encoder CUDA indices for 2E1PD. Default: "4,7"
  --enc-gpus-3e1pd LIST      Encoder CUDA indices for 3E1PD. Default: "4,5,7"
  --enc-gpus-4e1pd LIST      Encoder CUDA indices for 4E1PD. Default: "0,4,5,7"
  -h, --help                 Show this help.

Workload defaults match the recommended matrix:
  NUM_PROMPTS=128 IMAGE_COUNT=8 IMAGE_RES=1080p INPUT_LEN=128 OUTPUT_LEN=16
  RATE comes from --rates, OUTPUT_DETAILS=1, PD_PORT=38000, ENC_PORT_BASE=38002.

All native_sglang.sh environment knobs can still be overridden, for example:
  MODEL=... SERVED=... ./native_sglang_matrix.sh --pd-gpus 1,2,3,6
  CHUNKED=65536 MAX_PREFILL_TOKENS=65536 ./native_sglang_matrix.sh --rates "1.0"
EOF
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${RUN_ROOT:-$HERE/results/native_sglang_matrix_$(date +%Y%m%d_%H%M%S)}"
RATES="${RATES:-1.0}"
CASES="${CASES:-agg 1e1pd 2e1pd 3e1pd}"
MATRIX_PD_GPUS="${MATRIX_PD_GPUS:-${PD_GPUS:-1,2,3,6}}"
MATRIX_AGG_GPUS="${MATRIX_AGG_GPUS:-${AGG_GPUS:-}}"
ENC_GPUS_1E1PD="${ENC_GPUS_1E1PD:-7}"
ENC_GPUS_2E1PD="${ENC_GPUS_2E1PD:-4,7}"
ENC_GPUS_3E1PD="${ENC_GPUS_3E1PD:-4,5,7}"
ENC_GPUS_4E1PD="${ENC_GPUS_4E1PD:-0,4,5,7}"

require_value() {
  if [ "$#" -lt 2 ]; then
    echo "$1 requires a value" >&2
    usage >&2
    exit 2
  fi
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --run-root)
      require_value "$@"
      RUN_ROOT="$2"; shift 2 ;;
    --rates)
      require_value "$@"
      RATES="$2"; shift 2 ;;
    --cases)
      require_value "$@"
      CASES="$2"; shift 2 ;;
    --agg-gpus)
      require_value "$@"
      MATRIX_AGG_GPUS="$2"; shift 2 ;;
    --pd-gpus)
      require_value "$@"
      MATRIX_PD_GPUS="$2"; shift 2 ;;
    --enc-gpus-1e1pd)
      require_value "$@"
      ENC_GPUS_1E1PD="$2"; shift 2 ;;
    --enc-gpus-2e1pd)
      require_value "$@"
      ENC_GPUS_2E1PD="$2"; shift 2 ;;
    --enc-gpus-3e1pd)
      require_value "$@"
      ENC_GPUS_3E1PD="$2"; shift 2 ;;
    --enc-gpus-4e1pd)
      require_value "$@"
      ENC_GPUS_4E1PD="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

[ -n "$MATRIX_AGG_GPUS" ] || MATRIX_AGG_GPUS="$MATRIX_PD_GPUS"

mkdir -p "$RUN_ROOT"

COMMON_ENV=(
  PD_PORT="${PD_PORT:-38000}"
  ENC_PORT_BASE="${ENC_PORT_BASE:-38002}"
  NUM_PROMPTS="${NUM_PROMPTS:-128}"
  IMAGE_COUNT="${IMAGE_COUNT:-8}"
  IMAGE_RES="${IMAGE_RES:-1080p}"
  INPUT_LEN="${INPUT_LEN:-128}"
  OUTPUT_LEN="${OUTPUT_LEN:-16}"
  OUTPUT_DETAILS="${OUTPUT_DETAILS:-1}"
  BENCH_TIMEOUT="${BENCH_TIMEOUT:-0}"
  MAX_CONCURRENCY="${MAX_CONCURRENCY:-}"
  BENCH_PYTHONPATH="${BENCH_PYTHONPATH:-}"
  MEM_FRAC="${MEM_FRAC:-0.90}"
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
)

FAILED_CASES=()

run_case() {
  local case_name="$1"
  local label="$2"
  local pd_gpus="$3"
  local enc_gpus="$4"
  local rate="$5"
  local out_dir="$RUN_ROOT/r${rate}/${label}"
  local status=0

  mkdir -p "$out_dir"
  echo "[$(date +%H:%M:%S)] running $label: case=$case_name rate=$rate pd_gpus=$pd_gpus enc_gpus=${enc_gpus:-none} out=$out_dir"

  if [ -n "$enc_gpus" ]; then
    env "${COMMON_ENV[@]}" RATE="$rate" PD_GPUS="$pd_gpus" ENC_GPUS="$enc_gpus" \
      bash "$HERE/native_sglang.sh" "$case_name" "$out_dir" || status=$?
  else
    env "${COMMON_ENV[@]}" RATE="$rate" PD_GPUS="$pd_gpus" \
      bash "$HERE/native_sglang.sh" "$case_name" "$out_dir" || status=$?
  fi

  if [ "$status" -ne 0 ]; then
    FAILED_CASES+=("r${rate}/${label}:status=${status}")
    echo "[$(date +%H:%M:%S)] FAILED $label: status=$status out=$out_dir" >&2
  else
    echo "[$(date +%H:%M:%S)] completed $label: out=$out_dir"
  fi

  return 0
}

echo "native SGLang matrix output: $RUN_ROOT"
echo "Cases: $CASES"
echo "AGG GPUs: $MATRIX_AGG_GPUS"
echo "PD GPUs: $MATRIX_PD_GPUS"
echo "1E1PD encoder GPUs: $ENC_GPUS_1E1PD"
echo "2E1PD encoder GPUs: $ENC_GPUS_2E1PD"
echo "3E1PD encoder GPUs: $ENC_GPUS_3E1PD"
echo "4E1PD encoder GPUs: $ENC_GPUS_4E1PD"

for rate in $RATES; do
  for case_name in $CASES; do
    case "$case_name" in
      agg)
        run_case agg 1AGG "$MATRIX_AGG_GPUS" "" "$rate" ;;
      1e1pd)
        run_case 1e1pd 1E1PD "$MATRIX_PD_GPUS" "$ENC_GPUS_1E1PD" "$rate" ;;
      2e1pd)
        run_case 2e1pd 2E1PD "$MATRIX_PD_GPUS" "$ENC_GPUS_2E1PD" "$rate" ;;
      3e1pd)
        run_case 3e1pd 3E1PD "$MATRIX_PD_GPUS" "$ENC_GPUS_3E1PD" "$rate" ;;
      4e1pd)
        run_case 4e1pd 4E1PD "$MATRIX_PD_GPUS" "$ENC_GPUS_4E1PD" "$rate" ;;
      *)
        echo "unknown matrix case: $case_name" >&2
        exit 2 ;;
    esac
  done
done

if [ "${#FAILED_CASES[@]}" -gt 0 ]; then
  echo "[$(date +%H:%M:%S)] done with failures: $RUN_ROOT" >&2
  printf "  %s\n" "${FAILED_CASES[@]}" >&2
  exit 1
fi

echo "[$(date +%H:%M:%S)] done: $RUN_ROOT"
