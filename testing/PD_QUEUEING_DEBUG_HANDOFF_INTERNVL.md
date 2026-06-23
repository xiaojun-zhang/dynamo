# InternVL PD Queueing Debug Handoff

This is the focused handoff for InternVL model testing. Use this file for
future Codex sessions unless the pre-InternVL Qwen/Qwen3-VL history is needed.
That earlier history lives in `PD_QUEUEING_DEBUG_HANDOFF_PRE_INTERNVL.md`; the
old `PD_QUEUEING_DEBUG_HANDOFF.md` path is now only a short routing index.

## Current Objective

Determine whether SGLang + Dynamo with disaggregated InternVL vision encoders
can reproduce the native H200-only E/PD advantage, especially for
`OpenGVLab/InternVL3_5-38B`. If not, isolate whether the loss is from Dynamo
routing/frontend overhead, XPU encode speed, NIXL XPU-to-H200 embedding
transfer, or an InternVL/XPU compatibility issue.

## Workspace And Containers

- Repo/test cwd: `/home/h-zheng/robin/dynamo/testing`
- Main GPU container: `robin_sglang_dynamo_l40`
- H200 profile used by the harness: `h200`
- B60 XPU host used for recent tests: `172.26.46.171`
- B60 encoder container name: `harness_b60_enc`
- B60 XPU cards available for encoders: `0,1,2,3`
- SSH to XPU host from inside GPU container commonly uses:
  `ssh -F /dev/null -i /root/.ssh/id_ed25519 -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no h-zheng@172.26.46.171 ...`

Always check H200 and XPU idleness before running. Do not kill unrelated jobs.
Only stop harness processes launched for the current test.

## Models

Use local snapshot paths as served model names so benchmark clients do not pull
in incompatible Hugging Face processor metadata.

```text
InternVL3.5-8B:
/mnt/weka/data/llm-d-models-pv/hub/models--OpenGVLab--InternVL3_5-8B/snapshots/main

InternVL3.5-38B:
/mnt/weka/data/llm-d-models-pv/hub/models--OpenGVLab--InternVL3_5-38B/snapshots/main
```

InternVL runs require `CHAT_TEMPLATE=internvl-2-5`.

## Important Files

```text
bench_lib.py
orchestrator.py
run_matrix.py
bench_patches/sitecustomize.py
lib/add_worker.sh
lib/add_encoder_xpu.sh
lib/teardown.sh
container_patch_work/server_args.py
container_patch_work/encode_server.py
container_patch_work/internvl.py
container_patch_work/internvl_processor.py
container_patch_work/patch_dynamo_encode_worker_xpu.py
container_patch_work/patch_server_args_xpu.py
```

Expected runtime patch destinations inside GPU/XPU containers:

```text
/opt/sglang/python/sglang/srt/server_args.py
/opt/sglang/python/sglang/srt/disaggregation/encode_server.py
/opt/sglang/python/sglang/srt/models/internvl.py
/opt/sglang/python/sglang/srt/multimodal/processors/internvl.py
```

The XPU encoder patcher fixes InternVL encoder disaggregation details,
including `t*h*w` grid splitting, missing image placeholder reinsertion, and
wrapping expanded `<IMG_CONTEXT>` spans as `<img> ... </img>` so the PD worker
finds multimodal pad positions before embedding injection.

The current `container_patch_work/internvl_processor.py` fix decodes list
`input_text` back to text for normal AGG raw-image requests, while preserving
the special precomputed embedding path for E/PD payloads.

## Known Good Recent Result

B60/H200 matrix completed on 2026-06-23 with H200 GPU 0 as AGG/PD and B60 XPU
cards `0,1,2,3` as encoders:

```text
results/dynamo_internvl35_38b_b60_wrapfix_r1_4xpu/20260623_223218__mnt_weka_data_llm-d-models-pv_hub_models--OpenGVLab--InternVL3_5-38B_snapshots_main
```

Summary:

```text
1AGG:  32/32, duration  64.60s, throughput 0.50 req/s, mean TTFT  20464.17ms
1E1PD: 32/32, duration 241.63s, throughput 0.13 req/s, mean TTFT 192526.62ms
2E1PD: 32/32, duration 124.78s, throughput 0.26 req/s, mean TTFT  79331.40ms
3E1PD: 32/32, duration  93.39s, throughput 0.34 req/s, mean TTFT  52743.37ms
4E1PD: 32/32, duration 120.40s, throughput 0.27 req/s, mean TTFT  53707.92ms
```

The summary CSV at that result root has been corrected so all five rows are
`ok`.

## Quick Status Checks

```bash
docker exec robin_sglang_dynamo_l40 \
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits

docker exec -w /robin/dynamo/testing robin_sglang_dynamo_l40 bash -lc \
  'ps -eo pid,ppid,stat,etime,cmd | grep -E "run_matrix|orchestrator|bench_serving|dynamo.sglang|dynamo.frontend" | grep -v grep || true'

docker exec robin_sglang_dynamo_l40 bash -lc \
  'ssh -F /dev/null -i /root/.ssh/id_ed25519 -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no h-zheng@172.26.46.171 "docker ps -a | grep harness_b60_enc || true"'
```

## Canonical B60 Matrix Command

```bash
docker exec -w /robin/dynamo/testing \
  -e XPU_HOST=172.26.46.171 -e XPU_HOST_PROFILE=b60 \
  -e XPU_CONTAINER=harness_b60_enc -e PORT_HTTP=7011 \
  -e READY_TIMEOUT=1800 -e BENCH_TIMEOUT=900 \
  -e BENCH_PYTHONPATH=/robin/dynamo/testing/bench_patches \
  robin_sglang_dynamo_l40 \
  python3 run_matrix.py \
    --model /mnt/weka/data/llm-d-models-pv/hub/models--OpenGVLab--InternVL3_5-38B/snapshots/main \
    --gpus 0 --xpus 0,1,2,3 --rates 1.0 \
    --num-prompts 32 --image-count 4 --image-resolution 1080p \
    --input-len 128 --output-len 16 \
    --case1-agg 1 --case3-epd-xpu 'E=1,2,3,4;PD=1' \
    --results-root results/dynamo_internvl35_38b_b60_wrapfix_r1_4xpu
```

If one leg fails after a fix, rerun only the failed leg with `orchestrator.py`
into the same out-dir, then update `summary.csv` to reflect the retry result.

## Common Failures

- `Unknown model`: add the InternVL 38B snapshot entry to `bench_lib.MODELS`.
- AGG raw-image path error about pre-tokenized `input_ids`: ensure the patched
  `internvl_processor.py` is deployed.
- `shape mismatch ... [0, 5120]`: XPU encode token expansion is missing the
  InternVL `<img> ... </img>` wrapper around `<IMG_CONTEXT>` spans.
- `InternVLChatModel is not supported for encoder disaggregation`: the SGLang
  InternVL patch set was not applied in that container.
- XPU encoder exits immediately: inspect
  `results/.../logs/encode_xpu_<n>.log`; the fresh XPU container may be missing
  patches or a required XPU attention/backend flag.
- NIXL/UCX failure: verify H200/B60 host profiles and selected NIC/IPs.

## Worklog Rule

For each configuration tried, append setup, result path, metrics, failure mode
if any, fix attempted, rerun result, and analysis to this file.

### Try: InternVL3.5-8B TP4, 4 images smoke and E/PD repair

Run roots:

```text
results/native_sglang_matrix_20260623_internvl35_8b_tp4_img4_smoke
results/native_sglang_matrix_20260623_internvl35_8b_tp4_img4_smoke_repair_processor
results/native_sglang_matrix_20260623_internvl35_8b_tp4_img4_3e_repair_epd_patch
results/native_sglang_matrix_20260623_internvl35_8b_tp4_img4_3e_repair_internvl_encoder
results/native_sglang_matrix_20260623_internvl35_8b_tp4_img4_3e_repair_getmmdata
results/native_sglang_matrix_20260623_internvl35_8b_tp4_img4_3e_repair_processor_getmmdata
```

Reason for this try:

- Qwen3-VL-2B E/PD stayed behind AGG after image count, resolution,
  concurrency, output length, encoder count, grouped encoders, and mixed-chunk
  sweeps.
- InternVL3.5-8B has a different vision path and local SGLang model support, so
  it was the next model-family axis to test.

Common workload and knobs:

```bash
MODEL=/mnt/weka/data/llm-d-models-pv/hub/models--OpenGVLab--InternVL3_5-8B/snapshots/main
SERVED=$MODEL
CHAT_TEMPLATE=internvl-2-5
BENCH_PYTHONPATH=/robin/dynamo/testing/bench_patches
NUM_PROMPTS=8 IMAGE_COUNT=4 IMAGE_RES=1080p RATE=0.5
INPUT_LEN=128 OUTPUT_LEN=16
CHUNKED=65536 MAX_PREFILL_TOKENS=65536
MAX_RUNNING=32 PREFILL_MAX=2 MAX_CONCURRENCY=4
MEM_FRAC=0.82 AGG_MEM_FRAC=0.82 PD_MEM_FRAC=0.82 ENC_MEM_FRAC=0.65
AGG_EXTRA_ARGS="--cuda-graph-max-bs 32 --max-total-tokens 500000"
PD_EXTRA_ARGS="--cuda-graph-max-bs 32 --max-total-tokens 500000"
--agg-gpus 1,2,3,4 --pd-gpus 1,2,3,4 --enc-gpus-3e1pd 5,6,7
```

Failures and repairs:

1. Initial AGG benchmark failed before sending requests:

```text
AttributeError: Qwen2Tokenizer has no attribute start_image_token
```

Repair:

- Added `CHAT_TEMPLATE` to `native_sglang.sh` / `native_sglang_matrix.sh`.
- Added `BENCH_PYTHONPATH` so benchmark-only patches do not affect server
  startup.
- Added `bench_patches/sitecustomize.py`, a narrow InternVL fallback processor
  for `sglang.bench_serving` image datasets.
- Set `SERVED=$MODEL` so the benchmark client uses the local snapshot metadata
  instead of pulling incompatible HF processor code from the hub ID.

2. Initial 3E1PD failed at server validation:

```text
ValueError: Model type InternVLChatModel is not supported for encoder disaggregation
```

Repair:

- Patched the container SGLang install to add `InternVLChatModel` to the
  encoder-disaggregation allowlist.
- Patched `InternVLChatModel` so `--encoder-only` skips language weights and
  `--language-only` skips vision weights.

3. Encoder health checks then failed:

```text
Internal encoding error: expected Tensor as element 0 in argument 0, but got list
```

Repair:

- Patched the container encoder server InternVL image path to produce InternVL
  dynamic image tiles and matching grid metadata instead of generic
  `AutoImageProcessor` list output.
- Added InternVL-specific patch/token accounting for encoder-side raw tile
  slicing and embedding token counts.

4. PD reconstruction then failed:

```text
AttributeError: 'InternVLProcessor' object has no attribute 'IM_START_TOKEN_ID'
```

Repair:

- Added an InternVL processor-side `get_mm_data` reconstruction path that
  expands `<image>` placeholders into `<img><IMG_CONTEXT>...</img>`, slices the
  received embeddings, and returns `MultimodalProcessorOutput` using InternVL's
  actual token IDs.

AGG repaired result:

```text
results/native_sglang_matrix_20260623_internvl35_8b_tp4_img4_smoke_repair_processor/r0.5/1AGG
Successful requests: 8
Total images: 32
Request throughput: 0.44 req/s
Input token throughput: 4104.52 tok/s
Mean TTFT: 5390.64 ms
Mean E2E: 6175.90 ms
Mean TPOT: 102.84 ms
Peak concurrent requests: 5
```

3E1PD repaired result:

```text
results/native_sglang_matrix_20260623_internvl35_8b_tp4_img4_3e_repair_processor_getmmdata/r0.5/3E1PD
Successful requests: 8
Total images: 32
Request throughput: 0.51 req/s
Input token throughput: 4711.40 tok/s
Mean TTFT: 1166.82 ms
Mean E2E: 1245.79 ms
Mean TPOT: 11.85 ms
Peak concurrent requests: 3
```

Analysis:

- This is the first confirmed E/PD win in the investigation:

```text
InternVL3.5-8B img4 smoke AGG:    0.44 req/s, mean TTFT 5.39s, mean E2E 6.18s
InternVL3.5-8B img4 smoke 3E1PD:  0.51 req/s, mean TTFT 1.17s, mean E2E 1.25s
```

- Relative to AGG, 3E1PD is about `+16%` request throughput and about `5x`
  lower mean E2E latency on this small smoke workload.
- The model-family switch matters: InternVL's encoder path can be split in a
  way that avoids the large PD-side wait seen in the Qwen3-VL-2B sweeps once
  the missing SGLang InternVL E/PD support is repaired.
- The sample is intentionally small because this began as a repair run. Next
  axis: validate with a larger request count and higher rate on the same
  workload shape before treating this as the target case.

### Try: InternVL3.5-8B TP4, 4 images validation

Run root:

```text
results/native_sglang_matrix_20260623_internvl35_8b_tp4_img4_r1_mc8_validation
```

Reason for this try:

- The repaired smoke run showed the first E/PD win, but it used only 8
  requests.
- This run keeps the same model/workload shape and increases to 32 requests,
  rate 1.0, and max concurrency 8.

Workload and knobs:

```bash
MODEL=/mnt/weka/data/llm-d-models-pv/hub/models--OpenGVLab--InternVL3_5-8B/snapshots/main
SERVED=$MODEL
CHAT_TEMPLATE=internvl-2-5
BENCH_PYTHONPATH=/robin/dynamo/testing/bench_patches
NUM_PROMPTS=32 IMAGE_COUNT=4 IMAGE_RES=1080p RATE=1.0
INPUT_LEN=128 OUTPUT_LEN=16
CHUNKED=65536 MAX_PREFILL_TOKENS=65536
MAX_RUNNING=64 PREFILL_MAX=4 MAX_CONCURRENCY=8
MEM_FRAC=0.82 AGG_MEM_FRAC=0.82 PD_MEM_FRAC=0.82 ENC_MEM_FRAC=0.65
AGG_EXTRA_ARGS="--cuda-graph-max-bs 64 --max-total-tokens 800000"
PD_EXTRA_ARGS="--cuda-graph-max-bs 64 --max-total-tokens 800000"
--agg-gpus 1,2,3,4 --pd-gpus 1,2,3,4 --enc-gpus-3e1pd 5,6,7
```

AGG result:

```text
Successful requests: 32
Total images: 128
Request throughput: 0.60 req/s
Input token throughput: 5587.81 tok/s
Mean TTFT: 9262.27 ms
Mean E2E: 10587.96 ms
Mean TPOT: 284.91 ms
Peak concurrent requests: 13
```

3E1PD result:

```text
Successful requests: 32
Total images: 128
Request throughput: 0.81 req/s
Input token throughput: 7499.66 tok/s
Mean TTFT: 1177.96 ms
Mean E2E: 1373.25 ms
Mean TPOT: 49.35 ms
Peak concurrent requests: 5
```

Analysis:

```text
InternVL3.5-8B img4 validation AGG:    0.60 req/s, mean TTFT 9.26s, mean E2E 10.59s
InternVL3.5-8B img4 validation 3E1PD:  0.81 req/s, mean TTFT 1.18s, mean E2E 1.37s
```

- This validates the win on a larger sample: 3E1PD improves request throughput
  by about `35%` and reduces mean E2E latency by about `87%`.
- The winning allocation is PD/AGG on GPUs `1,2,3,4` and three encoders on
  GPUs `5,6,7`.
- The practical optimization is not Mooncake or grouped encoders; it is
  model/workflow support plus a workload with moderate image count where
  encoder offload keeps the language server responsive without creating the
  very large embedding handoff seen in the Qwen3-VL-2B 32/64/96-image sweeps.
- This is the target case where E/PD disaggregation beats aggregation on the
  recovered H200 server.

### Try: InternVL3.5-38B TP1, 4 images smoke

Run root:

```text
results/native_sglang_matrix_20260623_internvl35_38b_tp1_img4_r05_mc4_smoke
results/native_sglang_matrix_20260623_internvl35_38b_tp1_img4_r05_mc4_fullcases
```

Reason for this try:

- `InternVL3_5-8B` produced a validated E/PD win, but it is small compared with
  the 235B Qwen model.
- `InternVL3_5-38B` is local (`~72G` checkpoint) and is a better next step for
  testing whether the E/PD advantage holds for a larger dense InternVL model.
- User requested TP1 on GPU `0`, with GPUs `1,2,3` available as encoders.
- The first launch was interrupted while running only `agg 3e1pd`; its partial
  processes were cleaned. The restarted matrix includes `agg`, `1e1pd`,
  `2e1pd`, and `3e1pd`.

Planned workload and knobs:

```bash
MODEL=/mnt/weka/data/llm-d-models-pv/hub/models--OpenGVLab--InternVL3_5-38B/snapshots/main
SERVED=$MODEL
CHAT_TEMPLATE=internvl-2-5
BENCH_PYTHONPATH=/robin/dynamo/testing/bench_patches
NUM_PROMPTS=8 IMAGE_COUNT=4 IMAGE_RES=1080p RATE=0.5
INPUT_LEN=128 OUTPUT_LEN=16
CHUNKED=65536 MAX_PREFILL_TOKENS=65536
MAX_RUNNING=16 PREFILL_MAX=2 MAX_CONCURRENCY=4
KV_DTYPE=fp8_e4m3 MEM_FRAC=0.80 AGG_MEM_FRAC=0.80 PD_MEM_FRAC=0.80 ENC_MEM_FRAC=0.70
AGG_EXTRA_ARGS="--cuda-graph-max-bs 16 --max-total-tokens 250000"
PD_EXTRA_ARGS="--cuda-graph-max-bs 16 --max-total-tokens 250000"
--agg-gpus 0 --pd-gpus 0 --enc-gpus-3e1pd 1,2,3
--enc-gpus-1e1pd 1 --enc-gpus-2e1pd 1,2 --enc-gpus-3e1pd 1,2,3
```

Memory note:

- 38B TP1 with the previous 800k-token cap would be too high for one H200:
  weights `~71.5 GiB` plus FP8 KV `~97.7 GiB` before runtime overhead.
- This smoke uses `--max-total-tokens 250000`, which reduces FP8 KV to about
  `30.5 GiB`, leaving enough headroom for weights and runtime overhead on a
  140 GiB H200.

Results:

```text
1AGG:
Successful requests: 8
Total images: 32
Request throughput: 0.43 req/s
Input token throughput: 4012.24 tok/s
Mean TTFT: 4636.42 ms
Mean E2E: 6352.16 ms
Mean TPOT: 416.73 ms
Peak concurrent requests: 4

1E1PD:
Successful requests: 8
Total images: 32
Request throughput: 0.45 req/s
Input token throughput: 4224.14 tok/s
Mean TTFT: 2813.41 ms
Mean E2E: 3666.50 ms
Mean TPOT: 140.67 ms
Peak concurrent requests: 5

2E1PD:
Successful requests: 8
Total images: 32
Request throughput: 0.47 req/s
Input token throughput: 4357.70 tok/s
Mean TTFT: 2269.73 ms
Mean E2E: 2763.34 ms
Mean TPOT: 115.24 ms
Peak concurrent requests: 4

3E1PD:
Successful requests: 8
Total images: 32
Request throughput: 0.47 req/s
Input token throughput: 4373.29 tok/s
Mean TTFT: 2241.29 ms
Mean E2E: 2700.84 ms
Mean TPOT: 105.11 ms
Peak concurrent requests: 3
```

Analysis:

- All four requested 38B TP1 cases completed with the placement requested by
  the user: AGG/PD on GPU `0`, encoders on GPUs `1`, `1,2`, and `1,2,3`.
- Relative to 1AGG, 3E1PD improves request throughput by about `9%`, mean TTFT
  by about `52%`, and mean E2E latency by about `57%`.
- 2E1PD and 3E1PD are effectively tied on throughput at this small sample size;
  3E1PD has slightly better latency. The next useful run is a 32-request
  validation at rate 1.0 and max concurrency 8 with the same placement.

### Try: InternVL3.5-38B TP1, 4 images r1.0 validation

Run root:

```text
results/native_sglang_matrix_20260623_internvl35_38b_tp1_img4_r1_mc8_validation_fullcases
```

Reason for this try:

- The previous 38B smoke used rate `0.5` and only 8 requests. User asked to test
  rate `1.0` instead.
- This run keeps the requested placement: AGG/PD on GPU `0`; encoder GPUs
  `1`, `1,2`, and `1,2,3` for 1E1PD, 2E1PD, and 3E1PD.
- GPUs `4` and `7` were still occupied by unrelated work and were not touched.

Workload and knobs:

```bash
MODEL=/mnt/weka/data/llm-d-models-pv/hub/models--OpenGVLab--InternVL3_5-38B/snapshots/main
SERVED=$MODEL
CHAT_TEMPLATE=internvl-2-5
BENCH_PYTHONPATH=/robin/dynamo/testing/bench_patches
NUM_PROMPTS=32 IMAGE_COUNT=4 IMAGE_RES=1080p RATE=1.0
INPUT_LEN=128 OUTPUT_LEN=16
CHUNKED=65536 MAX_PREFILL_TOKENS=65536
MAX_RUNNING=32 PREFILL_MAX=4 MAX_CONCURRENCY=8
KV_DTYPE=fp8_e4m3 MEM_FRAC=0.80 AGG_MEM_FRAC=0.80 PD_MEM_FRAC=0.80 ENC_MEM_FRAC=0.70
AGG_EXTRA_ARGS="--cuda-graph-max-bs 32 --max-total-tokens 250000"
PD_EXTRA_ARGS="--cuda-graph-max-bs 32 --max-total-tokens 250000"
--agg-gpus 0 --pd-gpus 0
--enc-gpus-1e1pd 1 --enc-gpus-2e1pd 1,2 --enc-gpus-3e1pd 1,2,3
```

Results:

```text
1AGG:
Successful requests: 32
Total images: 128
Request throughput: 0.47 req/s
Input token throughput: 4330.27 tok/s
Mean TTFT: 10702.57 ms
Mean E2E: 14295.13 ms
Mean TPOT: 1338.72 ms
Peak concurrent requests: 13

1E1PD:
Successful requests: 32
Total images: 128
Request throughput: 0.60 req/s
Input token throughput: 5586.14 tok/s
Mean TTFT: 5845.32 ms
Mean E2E: 9944.85 ms
Mean TPOT: 737.34 ms
Peak concurrent requests: 13

2E1PD:
Successful requests: 32
Total images: 128
Request throughput: 0.66 req/s
Input token throughput: 6186.62 tok/s
Mean TTFT: 4763.65 ms
Mean E2E: 7583.27 ms
Mean TPOT: 532.67 ms
Peak concurrent requests: 13

3E1PD:
Successful requests: 32
Total images: 128
Request throughput: 0.69 req/s
Input token throughput: 6461.64 tok/s
Mean TTFT: 4422.47 ms
Mean E2E: 6905.50 ms
Mean TPOT: 455.67 ms
Peak concurrent requests: 11
```

Analysis:

```text
InternVL3.5-38B TP1 img4 r1.0 1AGG:   0.47 req/s, mean TTFT 10.70s, mean E2E 14.30s
InternVL3.5-38B TP1 img4 r1.0 1E1PD:  0.60 req/s, mean TTFT  5.85s, mean E2E  9.94s
InternVL3.5-38B TP1 img4 r1.0 2E1PD:  0.66 req/s, mean TTFT  4.76s, mean E2E  7.58s
InternVL3.5-38B TP1 img4 r1.0 3E1PD:  0.69 req/s, mean TTFT  4.42s, mean E2E  6.91s
```

- This is a validated 38B case where E/PD disaggregation beats aggregation:
  3E1PD improves request throughput by about `47%`, reduces mean TTFT by about
  `59%`, and reduces mean E2E latency by about `52%` relative to 1AGG.
- Scaling from 1 to 3 encoders is monotonic on both throughput and latency for
  this workload. The gain from 2E to 3E is smaller than from 1E to 2E, so 2E1PD
  is the more GPU-efficient point, while 3E1PD is the best absolute result.
- The r1.0 run strengthens the smoke conclusion: with 38B TP1, the single AGG
  server accumulates queueing on mixed vision+language work, while offloading
  image encoding keeps the PD language server more responsive.
- Post-run process check found no stale `native_sglang`, `launch_server`, or
  benchmark processes. GPUs `0,1,2,3` returned to idle after teardown.

## Next session task: SGLang + Dynamo with E on XPU

Status at handoff:

- The validated `OpenGVLab/InternVL3_5-38B` result above is native SGLang only:
  no Dynamo frontend/router, and both E and PD ran on H200 cards.
- The next task is to run the same model/workload through the SGLang+Dynamo
  harness, with PD on H200 and encode workers on the remote B70 XPU host.
- The strict comparison for this next round is `best SGLang+Dynamo aggregation`
  versus `best SGLang+Dynamo E/PD with E on XPU`; do not compare native SGLang
  numbers directly against Dynamo numbers except as sanity/context.

Workspace and branch:

```text
Host workspace: /home/h-zheng/robin/dynamo/testing
Container workspace: /robin/dynamo/testing
Branch: sglang-summit-e-pd-disaggregation-demo
GPU container: robin_sglang_dynamo_l40
XPU host: sc09giga01-b70 / 172.26.46.180
XPU SSH user: h-zheng
Default XPU image in harness: hm_dynamo_b70_pr26460:latest
Default XPU container name: harness_b70_enc
```

Known-good native 38B reference workload:

```bash
MODEL=/mnt/weka/data/llm-d-models-pv/hub/models--OpenGVLab--InternVL3_5-38B/snapshots/main
SERVED=$MODEL
CHAT_TEMPLATE=internvl-2-5
NUM_PROMPTS=32 IMAGE_COUNT=4 IMAGE_RES=1080p RATE=1.0
INPUT_LEN=128 OUTPUT_LEN=16 MAX_CONCURRENCY=8
CHUNKED=65536 MAX_PREFILL_TOKENS=65536
MAX_RUNNING=32 PREFILL_MAX=4
KV_DTYPE=fp8_e4m3 MEM_FRAC=0.80 AGG_MEM_FRAC=0.80 PD_MEM_FRAC=0.80 ENC_MEM_FRAC=0.70
AGG_EXTRA_ARGS="--cuda-graph-max-bs 32 --max-total-tokens 250000"
PD_EXTRA_ARGS="--cuda-graph-max-bs 32 --max-total-tokens 250000"
```

Native reference results to beat within the same topology family:

```text
Native SGLang 38B TP1 img4 r1.0 1AGG:   0.47 req/s, TTFT 10.70s, E2E 14.30s
Native SGLang 38B TP1 img4 r1.0 1E1PD:  0.60 req/s, TTFT  5.85s, E2E  9.94s
Native SGLang 38B TP1 img4 r1.0 2E1PD:  0.66 req/s, TTFT  4.76s, E2E  7.58s
Native SGLang 38B TP1 img4 r1.0 3E1PD:  0.69 req/s, TTFT  4.42s, E2E  6.91s
```

Existing Dynamo harness entry points:

- `orchestrator.py`: one test end to end. Modes are `agg`, `epd_gpu`,
  `epd_xpu`.
- `run_matrix.py`: resumable matrix wrapper. It continues across skipped/failed
  tests and records statuses in `summary.csv`.
- `bench_lib.py`: model table, H200 host profile, GPU/XPU idle checks, placement,
  frontend readiness, and `sglang.bench_serving` invocation.
- `lib/add_worker.sh`: launches local H200 `dynamo.sglang` workers.
- `lib/add_encoder_xpu.sh`: SSHes to the B70 host, starts a fresh XPU container,
  and launches one `dynamo.sglang --multimodal-encode-worker` per XPU.
- `lib/teardown.sh`: kills local harness PIDs and removes the remote XPU
  encoder container.

Important harness gaps before InternVL3.5-38B XPU testing:

1. `bench_lib.MODELS` does not currently contain `InternVL3_5-38B`.
   Add a model-table entry before running:

   ```python
   "/mnt/weka/data/llm-d-models-pv/hub/models--OpenGVLab--InternVL3_5-38B/snapshots/main": {
       "path": "/mnt/weka/data/llm-d-models-pv/hub/models--OpenGVLab--InternVL3_5-38B/snapshots/main",
       "tp": 1,
       "kv": "fp8_e4m3",
       "mem_frac": 0.80,
       "chunked": 65536,
       "chat_template": "internvl-2-5",
       "max_total_tokens": 250000,
       "cuda_graph_max_bs": 32,
   }
   ```

   Using the local snapshot path as the served model name avoids the benchmark
   client pulling incompatible HF processor metadata from the hub ID.

2. `lib/add_worker.sh` and `lib/add_encoder_xpu.sh` currently hardcode
   `--chat-template qwen2-vl`. InternVL needs `--chat-template internvl-2-5`.
   Patch the harness to pass a `CHAT_TEMPLATE` env/model-table value through
   both local H200 workers and remote XPU encoders.

3. Native 38B TP1 used `--max-total-tokens 250000` and
   `--cuda-graph-max-bs 32` to fit the 38B weights plus FP8 KV on one H200.
   The Dynamo worker wrapper does not currently pass those knobs. Add per-role
   extra args or explicit env support before assuming the 38B Dynamo launch will
   have the same memory behavior.

4. `bench_lib.run_bench` does not currently expose `--max-concurrency`, and the
   native validation used `MAX_CONCURRENCY=8`. Add a `--max-concurrency` option
   or document that the first Dynamo run is not an exact workload match.

5. `bench_patches/sitecustomize.py` is required for InternVL image dataset
   generation in `sglang.bench_serving`. The Dynamo harness does not use
   `BENCH_PYTHONPATH`; start it with real `PYTHONPATH`, for example:

   ```bash
   export PYTHONPATH=/robin/dynamo/testing/bench_patches${PYTHONPATH:+:$PYTHONPATH}
   ```

6. The live GPU container has the InternVL SGLang patches applied, but
   `lib/add_encoder_xpu.sh` creates a fresh XPU container for each test. That
   fresh XPU container must also receive the same SGLang patches before launching
   encode workers.

InternVL patch files saved in this repo:

```text
bench_patches/sitecustomize.py
container_patch_work/server_args.py
container_patch_work/encode_server.py
container_patch_work/internvl.py
container_patch_work/internvl_processor.py
```

Expected install destinations inside both GPU and XPU containers:

```text
/opt/sglang/python/sglang/srt/server_args.py
/opt/sglang/python/sglang/srt/disaggregation/encode_server.py
/opt/sglang/python/sglang/srt/models/internvl.py
/opt/sglang/python/sglang/srt/multimodal/processors/internvl.py
```

For XPU, add the copy step to `lib/add_encoder_xpu.sh` after `docker run` and
before `docker exec -d ... python3 -m dynamo.sglang`. Because the XPU container
mounts `/home:/home`, it can read:

```text
/home/h-zheng/robin/dynamo/testing/container_patch_work/
```

Suggested initial checks for the next session:

```bash
docker exec -w /robin/dynamo/testing robin_sglang_dynamo_l40 bash -lc \
  'hostname; python3 - <<EOF
import bench_lib
print(bench_lib.gpu_host_profile())
EOF'

docker exec robin_sglang_dynamo_l40 bash -lc \
  'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits'

docker exec robin_sglang_dynamo_l40 bash -lc \
  'ssh -F /dev/null -i /root/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o BatchMode=yes h-zheng@172.26.46.180 "hostname; for d in 0 1 2 3; do echo XPU \$d; xpu-smi stats -d \$d 2>/dev/null | grep -i \"GPU Memory Used\" || true; done"'
```

Suggested smoke sequence after applying the harness fixes:

```bash
docker exec -w /robin/dynamo/testing robin_sglang_dynamo_l40 bash -lc '
set -e
MODEL=/mnt/weka/data/llm-d-models-pv/hub/models--OpenGVLab--InternVL3_5-38B/snapshots/main
export GPU_HOST_PROFILE=h200
export PYTHONPATH=/robin/dynamo/testing/bench_patches${PYTHONPATH:+:$PYTHONPATH}
export CHAT_TEMPLATE=internvl-2-5
export READY_TIMEOUT=1800
export PORT_HTTP=7011

python3 orchestrator.py --mode agg --model "$MODEL" \
  --agg-instances 1 --gpus 0 \
  --num-prompts 8 --image-count 4 --image-resolution 1080p \
  --input-len 128 --output-len 16 --request-rate 0.5 \
  --out-dir results/dynamo_internvl35_38b_xpu_smoke/1AGG_r0.5

python3 orchestrator.py --mode epd_xpu --model "$MODEL" \
  --e-instances 1 --pd-instances 1 --gpus 0 --xpus 0 \
  --num-prompts 8 --image-count 4 --image-resolution 1080p \
  --input-len 128 --output-len 16 --request-rate 0.5 \
  --out-dir results/dynamo_internvl35_38b_xpu_smoke/1E1PD_r0.5
'
```

Suggested validation matrix once smoke passes:

```bash
docker exec -w /robin/dynamo/testing robin_sglang_dynamo_l40 bash -lc '
set -e
MODEL=/mnt/weka/data/llm-d-models-pv/hub/models--OpenGVLab--InternVL3_5-38B/snapshots/main
export GPU_HOST_PROFILE=h200
export PYTHONPATH=/robin/dynamo/testing/bench_patches${PYTHONPATH:+:$PYTHONPATH}
export CHAT_TEMPLATE=internvl-2-5
export READY_TIMEOUT=1800
export PORT_HTTP=7011

python3 run_matrix.py --model "$MODEL" \
  --gpus 0 --xpus 0,1,2 \
  --rates 1.0 \
  --num-prompts 32 --image-count 4 --image-resolution 1080p \
  --input-len 128 --output-len 16 \
  --case1-agg 1 \
  --case3-epd-xpu "E=1,2,3;PD=1" \
  --results-root results/dynamo_internvl35_38b_xpu_r1
'
```

Operational notes for the next session:

- Check H200 and XPU idleness before every run. Only use devices explicitly
  listed in `--gpus` / `--xpus`; do not kill unrelated processes.
- The prior native 38B tests used H200 GPU `0` for AGG/PD and H200 GPUs
  `1,2,3` for H200 encoders. For XPU testing, use H200 GPU `0` for both 1AGG
  and PD if it is free, so the comparison keeps the PD GPU constant.
- If GPU `0` is busy, pick another free H200 and use the same H200 for both
  Dynamo 1AGG and Dynamo PD.
- If an `epd_xpu` test fails, inspect, fix, and rerun the failed case rather
  than skipping it. Start with:

  ```bash
  docker exec robin_sglang_dynamo_l40 bash -lc \
    'ps -eo pid,ppid,stat,etime,cmd | grep -E "dynamo|sglang|nats|etcd" | grep -v grep || true'
  docker exec robin_sglang_dynamo_l40 bash -lc \
    'find /robin/dynamo/testing/results -path "*logs*" -type f | tail -n 40'
  docker exec robin_sglang_dynamo_l40 bash -lc \
    'ssh -F /dev/null -i /root/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o BatchMode=yes h-zheng@172.26.46.180 "docker ps -a | grep harness_b70_enc || true"'
  ```

- Common likely failure points:
  - `Unknown model`: add the InternVL 38B entry to `bench_lib.MODELS`.
  - tokenizer/processor `start_image_token`: ensure `PYTHONPATH` includes
    `bench_patches`.
  - wrong prompt format or no InternVL image expansion: verify all workers use
    `CHAT_TEMPLATE=internvl-2-5`.
  - `InternVLChatModel is not supported for encoder disaggregation`: the
    SGLang patch set was not applied in that container.
  - XPU encoder exits immediately: inspect
    `results/.../logs/encode_xpu_<n>.log`; the fresh XPU container may be
    missing patches or may need a different `--mm-attention-backend`.
  - NIXL/UCX connection failure: verify H200 profile resolves to
    `mgmt=172.26.46.133`, `roce=192.165.123.48`, `nic=mlx5_0:1`, and that
    `add_encoder_xpu.sh` picked the correct B70 NIC/IP for the selected XPU.

Documentation requirement:

- For each configuration tried, append setup, result path, metrics, failure
  mode if any, fix attempted, rerun result, and analysis to this file.
- The first target is to determine whether SGLang+Dynamo with XPU encoders can
  reproduce the native H200-only E/PD advantage for InternVL3.5-38B. If not,
  isolate whether the loss is from Dynamo routing/frontend overhead, XPU encode
  speed, NIXL XPU-to-H200 embedding transfer, or an InternVL/XPU compatibility
  problem.

### 2026-06-23 15:25 PDT - B60 1E1PD smoke continuation

Inherited run:
`results/dynamo_internvl35_38b_b60_smoke_xpuattn_dynchat_splitfix/1E1PD_r0.5`
with H200 GPU 6 as PD and B60 XPU 0 as encoder. The previous XPU encode patch
fixed the missing placeholder exception: the encoder produced a `(2560, 5120)`
embedding tensor and PD read it successfully over NIXL.

New failure mode: PD scheduler crashed in
`/opt/sglang/python/sglang/srt/managers/mm_utils.py::embed_mm_inputs` with
`shape mismatch: value tensor of shape [2560, 5120] cannot be broadcast to
indexing result of shape [0, 5120]`. PD logs show the input sequence length was
2575 and SGLang warned it found 0 multimodal placeholder positions in the text
for 2560 embedding tokens.

Analysis: InternVL model-side `pad_input_ids()` uses `<img> ... </img>` token
pairs to replace image-token spans with per-item `pad_value`s. The XPU
encode-worker expansion only emitted repeated `<IMG_CONTEXT>` ids, so no
pair-based padding occurred. Later `embed_mm_inputs()` searched for pad values
and got an empty mask.

Fix in progress: updated
`container_patch_work/patch_dynamo_encode_worker_xpu.py` so each expanded
InternVL image placeholder becomes `<img> + <IMG_CONTEXT>*N + </img>` unless it
is already wrapped. The patch keeps the earlier `t*h*w` grid split and missing
placeholder reinsertion fixes. `python3 -m py_compile
container_patch_work/patch_dynamo_encode_worker_xpu.py` passes. Next step is to
rerun the same bounded 1E1PD smoke.

### 2026-06-23 15:33 PDT - B60 1E1PD smoke passed after wrapping fix

Reran the bounded smoke with H200 GPU 6 as PD and B60 XPU 0 as encoder:

```text
results/dynamo_internvl35_38b_b60_smoke_xpuattn_dynchat_wrapfix/1E1PD_r0.5
```

Result: passed, 8/8 successful requests. Key metrics from
`result_1E1PD_r0.5.txt`: duration 61.84s, request throughput 0.13 req/s,
mean TTFT 25428ms. This verifies that the XPU encode path must preserve
InternVL's `<img> ... </img>` wrapper around expanded `<IMG_CONTEXT>` tokens so
PD SGLang can create multimodal pad positions before embedding injection.

### 2026-06-23 15:32-15:53 PDT - B60 1AGG and 1/2/3/4E1PD matrix

User clarified the target matrix: `1AGG` and `1/2/3/4E1PD`, with B60 XPU
cards `0,1,2,3` available as encoders. The B60 host used in this run was
`172.26.46.171`, XPU container `harness_b60_enc`, and the H200 PD/AGG GPU was
GPU 0.

Matrix command:

```bash
docker exec -w /robin/dynamo/testing \
  -e XPU_HOST=172.26.46.171 -e XPU_HOST_PROFILE=b60 \
  -e XPU_CONTAINER=harness_b60_enc -e PORT_HTTP=7011 \
  -e READY_TIMEOUT=1800 -e BENCH_TIMEOUT=900 \
  -e BENCH_PYTHONPATH=/robin/dynamo/testing/bench_patches \
  robin_sglang_dynamo_l40 \
  python3 run_matrix.py \
    --model /mnt/weka/data/llm-d-models-pv/hub/models--OpenGVLab--InternVL3_5-38B/snapshots/main \
    --gpus 0 --xpus 0,1,2,3 --rates 1.0 \
    --num-prompts 32 --image-count 4 --image-resolution 1080p \
    --input-len 128 --output-len 16 \
    --case1-agg 1 --case3-epd-xpu 'E=1,2,3,4;PD=1' \
    --results-root results/dynamo_internvl35_38b_b60_wrapfix_r1_4xpu
```

Result root:

```text
results/dynamo_internvl35_38b_b60_wrapfix_r1_4xpu/20260623_223218__mnt_weka_data_llm-d-models-pv_hub_models--OpenGVLab--InternVL3_5-38B_snapshots_main
```

Initial `1AGG` failed in `bench_serving` before the processor retry with:

```text
[internvl] Cannot process raw images/videos with pre-tokenized input_ids.
Provide multimodal data in 'processor_output' or 'precomputed_embedding' format...
```

Root cause: the patched InternVL processor treated every list `input_text` as
the special precomputed-embedding format. AGG receives pre-tokenized IDs plus
raw images, so it must decode list `input_text` back to text unless the image
or video payload is actually in the special format.

Fix deployed to the running GPU container:

```text
container_patch_work/internvl_processor.py
/opt/sglang/python/sglang/srt/multimodal/processors/internvl.py
```

The source and deployed runtime file both passed `python3 -m py_compile`.

The E/PD sweep completed successfully:

```text
1E1PD: 32/32 requests, duration 241.63s, throughput 0.13 req/s, mean TTFT 192526.62ms
2E1PD: 32/32 requests, duration 124.78s, throughput 0.26 req/s, mean TTFT 79331.40ms
3E1PD: 32/32 requests, duration 93.39s, throughput 0.34 req/s, mean TTFT 52743.37ms
4E1PD: 32/32 requests, duration 120.40s, throughput 0.27 req/s, mean TTFT 53707.92ms
```

After the processor fix, reran `1AGG` only into the same matrix out-dir:

```bash
docker exec -w /robin/dynamo/testing \
  -e PORT_HTTP=7011 -e READY_TIMEOUT=1800 -e BENCH_TIMEOUT=900 \
  -e BENCH_PYTHONPATH=/robin/dynamo/testing/bench_patches \
  robin_sglang_dynamo_l40 \
  python3 orchestrator.py --mode agg \
    --model /mnt/weka/data/llm-d-models-pv/hub/models--OpenGVLab--InternVL3_5-38B/snapshots/main \
    --agg-instances 1 --gpus 0 \
    --num-prompts 32 --image-count 4 --image-resolution 1080p \
    --input-len 128 --output-len 16 --request-rate 1.0 \
    --out-dir results/dynamo_internvl35_38b_b60_wrapfix_r1_4xpu/20260623_223218__mnt_weka_data_llm-d-models-pv_hub_models--OpenGVLab--InternVL3_5-38B_snapshots_main/case1_agg/1AGG_r1.0
```

Retry result: passed, 32/32 requests, duration 64.60s, throughput 0.50 req/s,
mean TTFT 20464.17ms. `summary.csv` was updated from the stale initial
`bench_failed` row to `ok` for `1AGG` after the retry.

Final result files:

```text
case1_agg/1AGG_r1.0/result_1AGG_r1.0.txt
case3_epd_xpu/1E1PD_r1.0/result_1E1PD_r1.0.txt
case3_epd_xpu/2E1PD_r1.0/result_2E1PD_r1.0.txt
case3_epd_xpu/3E1PD_r1.0/result_3E1PD_r1.0.txt
case3_epd_xpu/4E1PD_r1.0/result_4E1PD_r1.0.txt
```

No harness processes from these runs were left running after completion.

### 2026-06-23 23:20 PDT - B60 capped 1AGG and 4E1PD rerun

User asked to rerun only `1AGG` and `4E1PD` with the native-reference client
cap, `--max-concurrency 8`, while keeping H200 GPU 0 for AGG/PD and B60 XPU
cards `0,1,2,3` for encoders.

Command:

```bash
docker exec -w /robin/dynamo/testing \
  -e XPU_HOST=172.26.46.171 -e XPU_HOST_PROFILE=b60 \
  -e XPU_CONTAINER=harness_b60_enc -e PORT_HTTP=7011 \
  -e READY_TIMEOUT=1800 -e BENCH_TIMEOUT=900 \
  -e BENCH_PYTHONPATH=/robin/dynamo/testing/bench_patches \
  robin_sglang_dynamo_l40 \
  python3 run_matrix.py \
    --model /mnt/weka/data/llm-d-models-pv/hub/models--OpenGVLab--InternVL3_5-38B/snapshots/main \
    --gpus 0 --xpus 0,1,2,3 --rates 1.0 \
    --num-prompts 32 --image-count 4 --image-resolution 1080p \
    --input-len 128 --output-len 16 \
    --max-concurrency 8 \
    --case1-agg 1 --case3-epd-xpu 'E=4;PD=1' \
    --results-root results/dynamo_internvl35_38b_b60_wrapfix_r1_mc8_4xpu
```

Result root:

```text
results/dynamo_internvl35_38b_b60_wrapfix_r1_mc8_4xpu/20260623_232027__mnt_weka_data_llm-d-models-pv_hub_models--OpenGVLab--InternVL3_5-38B_snapshots_main
```

Both legs passed:

```text
1AGG:  32/32, duration  69.33s, throughput 0.46 req/s, mean TTFT 10570.59ms, mean E2E 14339.37ms
4E1PD: 32/32, duration 101.82s, throughput 0.31 req/s, mean TTFT 20190.28ms, mean E2E 22164.43ms
```

Comparison to the earlier uncapped B60 run:

```text
1AGG uncapped:  duration  64.60s, throughput 0.50 req/s, mean TTFT 20464.17ms
1AGG mc8:       duration  69.33s, throughput 0.46 req/s, mean TTFT 10570.59ms

4E1PD uncapped: duration 120.40s, throughput 0.27 req/s, mean TTFT 53707.92ms
4E1PD mc8:      duration 101.82s, throughput 0.31 req/s, mean TTFT 20190.28ms
```

The cap makes the Dynamo/XPU result a fairer comparison to the native SGLang
H200 reference and removes a large amount of client-induced queueing. It does
not make XPU E/PD competitive with `1AGG` on this B60 setup: capped `4E1PD`
still has about 1.9x the mean TTFT and 1.55x the mean E2E latency of capped
`1AGG`.

Encoder routing/latency for capped `4E1PD` from `encode_xpu_*.log`:

```text
All routed encoder requests, including readiness smoke:
xpu0:  7 completed, mean encode wall time 12.67s, median 12.17s, max 16.82s
xpu1:  9 completed, mean encode wall time 33.44s, median 38.66s, max 49.85s
xpu2:  6 completed, mean encode wall time 20.74s, median 20.15s, max 31.78s
xpu3: 12 completed, mean encode wall time 16.81s, median 17.75s, max 29.68s
```

For comparison, the previous uncapped `4E1PD` run had a much worse skew and
queue depth:

```text
xpu0:  5 completed, mean 22.76s, max  29.78s
xpu1: 17 completed, mean 93.16s, max 118.34s
xpu2:  9 completed, mean 44.65s, max  59.25s
xpu3:  3 completed, mean 16.16s, max  22.02s
```

Frontend config in `logs/frontend.log` confirms the current harness launched
with `router_mode="round-robin"`. The frontend discovered all four encoders
before benchmark traffic, but the route order was not perfectly cyclic and
still favored XPU 3 and XPU 1. The same log also shows:

```text
KvWorkerMonitor: KV metrics subscriber not available ... skipping load metrics.
```

Answer to "can encoder load be more distributed?": probably yes, but not with
the current hardcoded `round-robin` model-table setting alone. The installed
Dynamo frontend supports `round-robin`, `random`, `power-of-two`, `kv`,
`direct`, `least-loaded`, and `device-aware-weighted`. A controlled next
experiment is to make the harness expose/override `ROUTER_MODE` for this model
and rerun capped `4E1PD` with `least-loaded` or `power-of-two`. If that still
skews, the issue is likely below frontend policy: encoder request accounting,
endpoint set changes during routing, or B60 per-card service-time variance.

Post-run process check found no stale `run_matrix`, `orchestrator`,
`bench_serving`, `dynamo.sglang`, or `dynamo.frontend` processes.
