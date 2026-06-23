# Native SGLang test (no Dynamo) — narrowing harness

Runs the **same workload** as the Dynamo harness, but against **native SGLang**
servers directly — no NATS/etcd/frontend, no kv-router, no NIXL. Purpose: tell
whether a behavior is a **Dynamo-layer** issue or a **SGLang/model** one.

- If native works but the Dynamo harness fails/regresses → the issue is in the
  Dynamo layer (NIXL transfer, kv-router, registration).
- If native shows the same behavior → it's SGLang or the model.

Reference Dynamo run this mirrors:

```bash
PD_PREFILL_MAX=16 \
python3 run_matrix.py --model Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
    --gpus 0,1,2,3,4,5 --rates 1.0 \
    --num-prompts 32 --image-count 8 --image-resolution 1080p --output-len 256 \
    --case1-agg 1 --case2-epd-gpu "E=1,2;PD=1"
```

Parity settings used below (match the model table + harness defaults): TP4,
`kv=fp8_e4m3`, common language-server `mem_frac=0.90` for both AGG and PD,
`chunked=32768`, `prefill-max-requests=16`, `max-running-requests=40`,
common radix-cache behavior (`--disable-radix-cache` by default), 8 images @1080p,
in=128/out=256, rate=1.0, 32 prompts. Encoder-only workers keep their separate
`mem_frac=0.5` default.

## Prerequisites

- Run **inside the patched GPU container** `robin_sglang_dynamo_l40` (same image
  as the harness, SGLang `0.5.12.dev315`). The encode worker hits a MoE
  weight-loader bug on Qwen3-VL-235B (`qwen3_vl_moe.py:238` dereferences
  `self.model` which doesn't exist in encoder-only mode); `run_gpu_container.sh`
  applies the one-line guard on launch. If you start a *fresh* container, run
  that script first (or the encoders crash at load).
- Topology rules confirmed in `server_args.py` for this version:
  - `--encoder-only` → standalone encoder; cannot also be a prefill/decode node.
  - `--language-only` → combined prefill+decode server; **requires
    `--encoder-urls`**. This is the analog of the harness's `epd_gpu`.
  - `--enable-prefix-mm-cache` requires `--encoder-only`.
  - `Qwen3VLMoeForConditionalGeneration` (235B) is in the supported list.
- Embedding transport is the SGLang default **ZMQ** (`zmq_to_scheduler`) over
  localhost — the bench client hits the language-only/agg server directly, so no
  router is needed.

## 0. Enter the container & set shared vars

GPU placement is parameterized: `PD_GPUS` is the CUDA list for the agg/PD server
(TP = its count), `ENC_GPUS` is one index per encoder. Defaults below match the
reference run (PD on 0–3, encoders on 4,5).

```bash
docker exec -it robin_sglang_dynamo_l40 bash
# --- inside the container ---
MODEL=/mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-235B-A22B-Instruct-FP8
SERVED=Qwen/Qwen3-VL-235B-A22B-Instruct-FP8
PD_GPUS=0,1,2,3      # agg / PD server (TP = number of these)
ENC_GPUS=4,5         # one encoder per index (1E uses 4; 2E uses 4,5)
mkdir -p /robin/dynamo/testing/results/native_sglang
cd /robin/dynamo/testing/results/native_sglang

# bench client args — identical to what bench_lib.run_bench() calls.
# usage: bench <port> <label>
bench() {
  python3 -m sglang.bench_serving \
    --model "$SERVED" --backend sglang-oai-chat \
    --host 127.0.0.1 --port "$1" \
    --dataset-name image --num-prompts 32 \
    --random-input-len 128 --random-output-len 256 \
    --image-count 8 --image-resolution 1080p \
    --request-rate 1.0 --apply-chat-template --seed 0 \
    --disable-tqdm --output-file "bench_native_$2.json"
}

# 235B load is ~14 min; wait for readiness before benching.
# usage: wait_ready <logfile>
wait_ready() { grep -m1 "ready to roll" <(tail -n +1 -f "$1"); echo "  $1: ready"; }
```

## Case 1 — AGG (single combined server, TP = #PD_GPUS, on `$PD_GPUS`)

```bash
TP=$(echo "$PD_GPUS" | tr ',' '\n' | grep -c .)
CUDA_VISIBLE_DEVICES=$PD_GPUS python3 -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name "$SERVED" \
  --tensor-parallel-size "$TP" \
  --enable-multimodal --chat-template qwen2-vl \
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.90 \
  --chunked-prefill-size 32768 --prefill-max-requests 16 \
  --max-running-requests 40 --disable-radix-cache \
  --trust-remote-code --page-size 16 \
  --host 0.0.0.0 --port 30000 \
  > agg.log 2>&1 &

wait_ready agg.log
bench 30000 1AGG
pkill -f "launch_server.*--port 30000"
```

## Case 2 — EPD 1E1PD (encoder TP1 on first `$ENC_GPUS` + language-only PD on `$PD_GPUS`)

```bash
TP=$(echo "$PD_GPUS" | tr ',' '\n' | grep -c .)
ENC0=$(echo "$ENC_GPUS" | cut -d, -f1)

# 1) Encoder server (standalone, TP1)
CUDA_VISIBLE_DEVICES=$ENC0 python3 -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name "$SERVED" \
  --tensor-parallel-size 1 \
  --encoder-only --enable-multimodal --chat-template qwen2-vl \
  --enable-prefix-mm-cache --mem-fraction-static 0.5 \
  --trust-remote-code --page-size 16 \
  --host 0.0.0.0 --port 30002 \
  > enc0.log 2>&1 &

# 2) Language-only PD server (TP = #PD_GPUS), pointing at the encoder URL
CUDA_VISIBLE_DEVICES=$PD_GPUS python3 -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name "$SERVED" \
  --tensor-parallel-size "$TP" \
  --language-only --encoder-urls http://127.0.0.1:30002 \
  --chat-template qwen2-vl \
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.90 \
  --chunked-prefill-size 32768 --prefill-max-requests 16 \
  --max-running-requests 40 --disable-radix-cache \
  --trust-remote-code --page-size 16 \
  --host 0.0.0.0 --port 30000 \
  > pd.log 2>&1 &

wait_ready enc0.log
wait_ready pd.log
bench 30000 1E1PD
pkill -f "launch_server.*--port 3000[02]"
```

## Case 2 — EPD 2E1PD (two encoders on first two `$ENC_GPUS` + same PD on `$PD_GPUS`)

```bash
TP=$(echo "$PD_GPUS" | tr ',' '\n' | grep -c .)
ENC0=$(echo "$ENC_GPUS" | cut -d, -f1); ENC1=$(echo "$ENC_GPUS" | cut -d, -f2)

CUDA_VISIBLE_DEVICES=$ENC0 python3 -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name "$SERVED" --tensor-parallel-size 1 \
  --encoder-only --enable-multimodal --chat-template qwen2-vl \
  --enable-prefix-mm-cache --mem-fraction-static 0.5 \
  --trust-remote-code --page-size 16 --host 0.0.0.0 --port 30002 > enc0.log 2>&1 &

CUDA_VISIBLE_DEVICES=$ENC1 python3 -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name "$SERVED" --tensor-parallel-size 1 \
  --encoder-only --enable-multimodal --chat-template qwen2-vl \
  --enable-prefix-mm-cache --mem-fraction-static 0.5 \
  --trust-remote-code --page-size 16 --host 0.0.0.0 --port 30003 > enc1.log 2>&1 &

CUDA_VISIBLE_DEVICES=$PD_GPUS python3 -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name "$SERVED" --tensor-parallel-size "$TP" \
  --language-only --encoder-urls http://127.0.0.1:30002 http://127.0.0.1:30003 \
  --chat-template qwen2-vl \
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.90 \
  --chunked-prefill-size 32768 --prefill-max-requests 16 \
  --max-running-requests 40 --disable-radix-cache \
  --trust-remote-code --page-size 16 --host 0.0.0.0 --port 30000 > pd.log 2>&1 &

wait_ready enc0.log; wait_ready enc1.log; wait_ready pd.log
bench 30000 2E1PD
pkill -f "launch_server.*--port 3000[023]"
```

## Native SGLang vs the Dynamo harness

| | Dynamo harness (`epd_gpu`) | Native SGLang (here) |
|---|---|---|
| Embedding transport | NIXL (cuda_ipc / RDMA) | ZMQ (`zmq_to_scheduler`), localhost |
| Routing | dynamo frontend + kv-router | none — bench hits PD server directly |
| Control plane | NATS + etcd + frontend | none |
| Encoder↔PD wiring | etcd discovery | static `--encoder-urls` |
| Encoder cmd | `dynamo.sglang --multimodal-encode-worker` | `sglang.launch_server --encoder-only` |
| PD cmd | `dynamo.sglang --multimodal-worker` | `sglang.launch_server --language-only` |

## Notes

- `bench()` does **not** wait — always `wait_ready <log>` first (235B weight load
  is ~14 min: ~35 s/shard × 24 shards).
- Result JSONs land in `results/native_sglang/bench_native_<label>.json`; the
  same metrics `make_perf_csv.py` reads (`request_throughput`, `mean_ttft_ms`,
  `mean_tpot_ms`, `mean_itl_ms`, `mean_e2e_latency_ms`, token counts).
- To **fully** isolate from the Dynamo container, `docker run` the same image
  fresh — but re-apply the MoE patch first (run `run_gpu_container.sh`, or the
  encoders crash at load).
- GPU budget (defaults): AGG = `#PD_GPUS` cards; 1E1PD = `#PD_GPUS + 1`; 2E1PD =
  `#PD_GPUS + 2` — defaults `PD_GPUS=0,1,2,3 ENC_GPUS=4,5` match the
  `--gpus 0,1,2,3,4,5` envelope of the reference Dynamo run. Repoint with e.g.
  `PD_GPUS=4,5,6,7 ENC_GPUS=0,1`.
- The `native_sglang.sh` script takes the **same** `PD_GPUS` / `ENC_GPUS` env
  vars (TP defaults to `#PD_GPUS`). It also uses common language-server defaults
  for AGG and PD: `MEM_FRAC=0.90`, `PREFILL_MAX=16`, and
  `DISABLE_RADIX_CACHE=1`; override them only when intentionally testing that knob.
  ```bash
  PD_GPUS=0,1,2,3 ENC_GPUS=4,5 ./native_sglang.sh 2e1pd
  PD_GPUS=0,1 ./native_sglang.sh agg          # TP2 agg on 2 cards
  ```
