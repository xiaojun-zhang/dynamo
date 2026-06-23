# PD Queueing Debug Handoff

Date: 2026-06-18
Workspace: `/home/h-zheng/robin/dynamo/testing`

## Goal

Continue debugging why native SGLang E/PD runs have much larger PD queueing and worse TTFT/throughput than AGG, even when AGG and PD use the same TP4 GPU group and similar launch knobs.

Main question:

Why does language-only PD drain prefills much slower / accumulate much larger `#queue-req` and `#pending-token` than AGG for the same image benchmark?

## Model And Benchmark

Model:

```bash
MODEL=/mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-235B-A22B-Instruct-FP8
SERVED=Qwen/Qwen3-VL-235B-A22B-Instruct-FP8
```

Main benchmark shape used for the matrix:

```bash
python3 -m sglang.bench_serving \
  --model "$SERVED" --backend sglang-oai-chat \
  --host 127.0.0.1 --port 38000 \
  --dataset-name image --num-prompts 128 \
  --random-input-len 128 --random-output-len 16 \
  --image-count 8 --image-resolution 1080p \
  --request-rate 1.0 --apply-chat-template --seed 0 \
  --disable-tqdm --output-details \
  --output-file bench.json
```

Each 8-image request is roughly `16.4k` input tokens. With `chunked_prefill_size=32768`, a prefill batch usually holds only about 1-2 such requests.

## Relevant Scripts

Files:

- `native_sglang.sh`: runs one case end to end.
- `native_sglang_matrix.sh`: loops cases/rates and delegates to `native_sglang.sh`.
- `NATIVE_SGLANG.md`: notes and recommended matrix.

Recent script changes:

- `native_sglang.sh` supports cases: `agg`, `1e1pd`, `2e1pd`, `3e1pd`, `4e1pd`.
- `native_sglang_matrix.sh` supports GPU placement args:
  - `--agg-gpus`
  - `--pd-gpus`
  - `--enc-gpus-1e1pd`
  - `--enc-gpus-2e1pd`
  - `--enc-gpus-3e1pd`
  - `--enc-gpus-4e1pd`
- `native_sglang.sh` saves raw `bench_serving` text as both:
  - `bench_native_<CASE>.txt`
  - `result_<CASE>_r<RATE>.txt`
- `native_sglang.sh` now passes `--max-prefill-tokens "$MAX_PREFILL_TOKENS"` to AGG and PD.
- `native_sglang_matrix.sh` now forwards:
  - `CHUNKED`
  - `MAX_PREFILL_TOKENS`
  - `PREFILL_MAX`
  - `MAX_RUNNING`

Syntax check passed:

```bash
bash -n native_sglang.sh
bash -n native_sglang_matrix.sh
```

## Known GPU Notes

Historical note: GPU 4 was previously avoided inside a TP group:

- Old check showed GPU 4 has persistent NVLink issue: only `NV17`, link 14 inactive, while other cards show all 18 links active.
- Using GPU 4 as a standalone encoder may be less risky, but avoid it in AGG/PD TP.

GPU 0 was occupied in one prior check, so a 7-card matrix used:

```bash
--agg-gpus 1,2,3,5
--pd-gpus 1,2,3,5
--enc-gpus-1e1pd 6
--enc-gpus-2e1pd 6,7
```

Check live ports before runs:

```bash
ss -ltnp '( sport = :38000 or sport = :38002 or sport = :38003 or sport = :38004 )'
```

After the 2026-06-22 server recovery, GPU 4 was rechecked and the issue was no
longer present:

```text
nvidia-smi topo -m: GPU4 reports NV18 to GPU0-3 and GPU5-7.
All GPUs 0-7 were free at the time of the recheck.
```

Current placement for clean H200 runs:

```text
AGG/PD TP group: GPUs 1,2,3,4
Encoders: GPUs 5,6, and GPU7 for 3E if available
```

## Key Results Already Observed

Important result directory:

```text
/home/h-zheng/robin/dynamo/testing/results/native_sglang_matrix_20260618_045542/r1.0
```

At rate `1.0`, approximate benchmark results:

```text
AGG:    throughput 0.73 req/s, mean TTFT  35.6s, E2E  71.6s
1E1PD:  throughput 0.26 req/s, mean TTFT 315.1s, E2E 374.7s
2E1PD:  throughput 0.33 req/s, mean TTFT 221.8s, E2E 270.5s
```

2E1PD is better than 1E1PD at `r1.0`, but both are much worse than AGG.

Parsed E/PD request timeline showed embedding transfer is not the main bottleneck:

```text
1E1PD:
  Processing -> encoder+transfer done: ~99.1s mean
  send -> encoder+transfer done:       ~4.1s mean
  transfer-only estimate:              ~1.0s mean
  embeddings ready -> language prefill: ~212.2s mean
  max queue-req: 98
  max pending-token: ~1.61M

2E1PD:
  Processing -> encoder+transfer done: ~26.5s mean
  send -> encoder+transfer done:       ~0.7s mean
  transfer-only estimate:              ~0.07s mean
  embeddings ready -> language prefill: ~191.4s mean
  max queue-req: 113
  max pending-token: ~1.86M
```

AGG vs E/PD image-prefill drain timeline:

```text
AGG:    135 image-prefill seqs over 176s => ~0.77 seq/s
1E1PD:  135 image-prefill seqs over 500s => ~0.27 seq/s
2E1PD:  134 image-prefill seqs over 394s => ~0.34 seq/s
```

This points to the PD side queue/drain path, not encoder transfer alone.

## Important Log Patterns

Memory/startup summary:

```bash
rg -n "Load weight|KV Cache|Memory pool|Capture cuda graph|Required memory|max_total_num_tokens|available_gpu_mem|chunked_prefill_size|max_prefill_tokens|mem usage|avail mem" \
  path/to/pd.log
```

Queue/prefill summary:

```bash
rg -n "Prefill batch|Decode batch|#queue-req|#pending-token|waiting-image-req|input throughput" \
  path/to/pd.log
```

Server args:

```bash
rg -n "server_args=ServerArgs" path/to/pd.log
```

Benchmark result:

```bash
cat path/to/result_1AGG_r1.0.txt
cat path/to/result_1E1PD_r1.0.txt
cat path/to/result_2E1PD_r1.0.txt
```

## Current Hypothesis

The prior `CHUNKED=65536 MAX_RUNNING=64` run was incomplete as a test because SGLang still used:

```text
max_prefill_tokens=16384
```

Example log:

```text
max_total_num_tokens=3014272, chunked_prefill_size=65536, max_prefill_tokens=16384, max_running_requests=64, context_len=262144, available_gpu_mem=13.22 GB
```

Before:

```text
max_total_num_tokens=3014272, chunked_prefill_size=32768, max_prefill_tokens=16384, max_running_requests=40, context_len=262144, available_gpu_mem=13.31 GB
```

Meaning:

- `chunked_prefill_size` controls chunk splitting/admission chunk size.
- `max_prefill_tokens` is SGLang's per-prefill-pass input-token admission budget.
- `prefill_max_requests` caps number of requests admitted into one prefill batch.
- `max_running_requests` caps active concurrent requests.

Because each image request is about `16.4k` tokens, keeping `max_prefill_tokens=16384` means SGLang can still mostly admit only about one full image-heavy request per prefill pass. Raising `chunked_prefill_size` alone may not improve queue drain much.

## Next Test To Run

Run paired `CHUNKED` and `MAX_PREFILL_TOKENS` so the PD can actually batch more image-heavy prefill tokens:

```bash
CHUNKED=65536 MAX_PREFILL_TOKENS=65536 MAX_RUNNING=64 \
./native_sglang_matrix.sh \
  --cases "agg 1e1pd 2e1pd" \
  --rates "1.0" \
  --agg-gpus 1,2,3,5 \
  --pd-gpus 1,2,3,5 \
  --enc-gpus-1e1pd 6 \
  --enc-gpus-2e1pd 6,7
```

Then optionally test:

```bash
CHUNKED=131072 MAX_PREFILL_TOKENS=131072 MAX_RUNNING=64 \
./native_sglang_matrix.sh \
  --cases "agg 1e1pd 2e1pd" \
  --rates "1.0" \
  --agg-gpus 1,2,3,5 \
  --pd-gpus 1,2,3,5 \
  --enc-gpus-1e1pd 6 \
  --enc-gpus-2e1pd 6,7
```

Be careful with memory. Startup should report available GPU memory around 13 GB for the previous configs. Raising prefill token budget increases temporary activation/workspace pressure.

## What To Compare After Next Run

For each AGG / 1E1PD / 2E1PD:

1. Benchmark:
   - request throughput
   - mean/median/p99 TTFT
   - E2E latency
   - TPOT/ITL

2. Startup config:
   - `chunked_prefill_size`
   - `max_prefill_tokens`
   - `max_running_requests`
   - `available_gpu_mem`

3. Prefill drain:
   - count image-prefill `Prefill batch` lines
   - time span from first to last image prefill
   - `#new-seq`
   - `#new-token`
   - `input throughput`

4. Queue:
   - max `#queue-req`
   - max `#pending-token`
   - whether `waiting-image-req` stays nonzero

Expected signal if the hypothesis is correct:

- PD `Prefill batch` lines should show larger `#new-token`, often `32768`, `49152`, `65536`, etc.
- PD image prefill drain rate should improve.
- `#queue-req` and `#pending-token` should peak lower or drain faster.
- E/PD TTFT should drop materially.

If `max_prefill_tokens=65536` still does not help, investigate differences in language-only scheduling path:

- `--language-only` request admission after remote embeddings arrive
- `waiting-image-req`
- overlap scheduling behavior
- whether PD batches remote-embedding requests differently than AGG batches local multimodal prefills
- exact order of request arrival vs first PD prefill

## Useful Source Pointers

SGLang source inside running container was found at:

```text
/opt/sglang/python/sglang
```

Known container names with SGLang installed during prior inspection:

```text
robin_sglang_dynamo_l40
hm_dynamo_s21
```

Useful source files:

```text
/opt/sglang/python/sglang/srt/server_args.py
/opt/sglang/python/sglang/srt/managers/scheduler.py
/opt/sglang/python/sglang/srt/managers/schedule_policy.py
```

Relevant code findings:

- `ServerArgs.max_prefill_tokens` default is `16384`.
- `scheduler.py` passes `self.max_prefill_tokens` and `chunked_prefill_size` into `PrefillAdder`.
- `schedule_policy.py::PrefillAdder` uses:
  - `rem_input_tokens` from `max_prefill_tokens`
  - `rem_chunk_tokens` from `chunked_prefill_size`
  - `prefill_max_requests` as request-count cap

## Caution

Do not run full 235B matrix unless intended; each PD/AGG server load can take many minutes.

Do not kill unrelated SGLang/Docker processes unless explicitly approved.

Do not use the old GPU4 avoidance rule unless a fresh topology check shows the
NVLink issue has returned.

## 2026-06-18 Follow-up: 65k Prefill Run

Observed existing active run:

```bash
CHUNKED=65536 MAX_PREFILL_TOKENS=65536 MAX_RUNNING=64 \
./native_sglang_matrix.sh \
  --cases "agg 1e1pd 2e1pd" --rates "1.0" \
  --agg-gpus 1,2,3,5 --pd-gpus 1,2,3,5 \
  --enc-gpus-1e1pd 6 --enc-gpus-2e1pd 6,7
```

Result directory:

```text
/home/h-zheng/robin/dynamo/testing/results/native_sglang_matrix_20260618_191404/r1.0
```

Completed valid result:

```text
1E1PD: throughput 0.280 req/s, mean TTFT 269.0s, median TTFT 294.2s, mean E2E 353.7s
```

Compared with previous 1E1PD baseline at 16k max-prefill tokens:

```text
throughput: 0.256 -> 0.280 req/s
mean TTFT: 315.1s -> 269.0s
mean E2E: 374.7s -> 353.7s
max #queue-req: 98 -> 81
max #pending-token: 1.61M -> 1.33M
image-prefill lines: 120 -> 40
65k prefill batches admitted: 28
image-prefill span: ~500s -> ~458s
```

Interpretation: raising `max_prefill_tokens` does change admission as expected and improves 1E1PD, but it does not close the gap to AGG. Language-only PD is still spending hundreds of seconds draining prefills.

Invalid/failed results in this run:

- `1AGG` crashed with CUDA OOM in fused MoE after larger multimodal prefill batches (`Tried to allocate 3.00 GiB`, only ~2.36 GiB free). No valid JSON summary.
- `2E1PD` crashed before benchmark completion in the language-only multimodal receive path:

```text
sglang/srt/multimodal/processors/base_processor.py:331
mm_token_num = video_grid_thw[video_idx].prod()
TypeError: 'NoneType' object is not subscriptable
```

The `2E1PD` PD log shows requests assigned as image-only (`OrderedDict({<Modality.IMAGE: 1>: [4, 4]})`), but `build_input_ids` later found a video token in the prompt while only image grid metadata was present. This points at a language-only E/PD multimodal reconstruction bug or stale/mismatched request metadata under two-encoder concurrency, not a benchmark result.

Harness fix applied after observing this: `native_sglang.sh` now preserves the `bench_serving` exit code through the `tee` pipeline, and `native_sglang_matrix.sh` records failed cases while continuing with the rest of the matrix. The matrix exits nonzero only after all requested cases have run.

Recommended next bounded test:

```bash
CHUNKED=65536 MAX_PREFILL_TOKENS=65536 MAX_RUNNING=40 \
./native_sglang_matrix.sh \
  --cases "2e1pd" --rates "1.0" \
  --pd-gpus 1,2,3,5 \
  --enc-gpus-2e1pd 6,7
```

Purpose: separate the 2E crash from the higher `MAX_RUNNING=64` concurrency. If it still crashes, focus on `encode_receiver.py` metadata assembly and `qwen_vl.py`/`base_processor.py` prompt token modality detection for split image requests.

## 2026-06-18 Follow-up: H200 Memory Headroom And 2E Working Point

The lower-concurrency 2E test without changing memory reservation still failed:

```text
results/native_sglang_matrix_20260618_203001_2e65k_mr40/r1.0/2E1PD
CHUNKED=65536 MAX_PREFILL_TOKENS=65536 MAX_RUNNING=40 LOG_LEVEL=warning
```

Failure mode was CUDA OOM in the PD language model fused MoE path, not the earlier
`video_grid_thw` reconstruction crash:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.00 GiB.
GPU 0 ... 3.99 GiB is free ... 134.75 GiB memory in use.
```

Interpretation: 65k prefill admission is useful, but the default
`mem_fraction_static=0.90` over-reserves KV cache on H200 for this workload and
leaves too little dynamic workspace for fused MoE when 2 encoders feed the PD.

Successful H200 memory-headroom run:

```text
results/native_sglang_matrix_20260618_204602_2e65k_pdmem086/r1.0/2E1PD
CHUNKED=65536 MAX_PREFILL_TOKENS=65536 MAX_RUNNING=40 PD_MEM_FRAC=0.86
PD GPUs: 1,2,3,5
Encoder GPUs: 6,7
```

Startup changed PD KV/headroom from roughly:

```text
mem_fraction_static=0.90: max_total_num_tokens=3014272, available_gpu_mem=13.31 GB
mem_fraction_static=0.86: max_total_num_tokens=2769568, available_gpu_mem=18.45 GB
```

Benchmark result:

```text
2E1PD baseline 16k: throughput 0.325 req/s, mean TTFT 221.8s, mean E2E 270.5s
2E1PD 65k/mem086:  throughput 0.331 req/s, mean TTFT 212.6s, mean E2E 255.0s
```

Prefill drain:

```text
2E1PD baseline 16k: 118 image-prefill batches, 134 seqs over 394s => 0.340 seq/s
2E1PD 65k/mem086:   43 image-prefill batches, 149 seqs over 388s => 0.384 seq/s
max #queue-req:     113 -> 100
max #pending-token: 1.86M -> 1.64M
```

Also tested larger 131k admission:

```text
results/native_sglang_matrix_20260618_210250_2e131k_pdmem080/r1.0/2E1PD
CHUNKED=131072 MAX_PREFILL_TOKENS=131072 MAX_RUNNING=40 PD_MEM_FRAC=0.80
```

Startup:

```text
max_total_num_tokens=2402512, available_gpu_mem=27.02 GB
```

Benchmark result regressed versus 65k:

```text
2E1PD 131k/mem080: throughput 0.325 req/s, mean TTFT 218.0s, mean E2E 262.4s
```

131k did reduce queue peaks and prefill batch count, but the larger passes were
not faster end to end:

```text
2E1PD 131k/mem080: 29 image-prefill batches, 135 seqs over 394s => 0.343 seq/s
max #queue-req: 82
max #pending-token: 1.35M
max observed prefill batch: 131072 tokens
```

During the 131k run, live GPU free memory dropped as low as a few hundred MiB on
one PD rank. It completed, but `PD_MEM_FRAC=0.80` is too close to the cliff for a
recommended setting.

Current recommended native-SGLang E/PD working point on this H200 server:

```bash
LOG_LEVEL=info \
CHUNKED=65536 MAX_PREFILL_TOKENS=65536 MAX_RUNNING=40 PD_MEM_FRAC=0.86 \
./native_sglang_matrix.sh \
  --cases "2e1pd" --rates "1.0" \
  --pd-gpus 1,2,3,5 \
  --enc-gpus-2e1pd 6,7
```

This is only a modest improvement. The remaining gap to AGG is still large:

```text
AGG baseline:       throughput 0.730 req/s, mean TTFT 35.6s, mean E2E 71.6s
best 2E1PD so far:  throughput 0.331 req/s, mean TTFT 212.6s, mean E2E 255.0s
```

Next useful experiments:

1. Try intermediate admission, e.g. `CHUNKED=98304 MAX_PREFILL_TOKENS=98304`
   with `PD_MEM_FRAC=0.83` or `0.84`, only if more tuning is worth another long
   run.
2. Test `PD_EXTRA_ARGS="--enable-mixed-chunk"` with the 65k/mem086 working point
   to see whether decode/prefill interleaving helps tail latency.
3. Generate/tune missing H200 Triton 3.6.0 MoE configs for
   `E=128,N=384,dtype=fp8_w8a8,block_shape=[128, 128]`. Logs still fall back to
   Triton 3.2.0 configs, which is a general H200 Qwen3-VL MoE speed issue rather
   than an E/PD-specific queueing issue.

## 2026-06-22/23 E/PD Performance Search Log

User goal for this phase: keep trying workflow, memory/resource placement, and
workload shapes until native SGLang E/PD disaggregation shows a performance win
over aggregation, or until there is no viable path left.

Operational constraints:

- GPU 4 NVLink was rechecked and is healthy again: topology reports `NV18` from
  GPU4 to all peers and all 18 GPU4 NVLink links are active at 26.562 GB/s.
- Use GPUs `1,2,3,4` for AGG/PD TP4.
- Use GPUs `5,6` for encoders, and GPU7 for 3E when available.
- Do not kill unrelated VLLM/specbench jobs. GPU0 has been intermittently used
  by unrelated VLLM jobs during this search.
- Mooncake is installed in `robin_sglang_dynamo_l40`; SGLang imports it and logs
  `Using transfer backend: mooncake`.

### Try: 32 images, low request rate, Mooncake

Run root:

```text
results/native_sglang_matrix_20260623_051903_img32_agg_2e3e_mooncake_mem080
```

Common workload and knobs:

```bash
NUM_PROMPTS=8 IMAGE_COUNT=32 IMAGE_RES=1080p RATE=0.1 OUTPUT_LEN=16
CHUNKED=65536 MAX_PREFILL_TOKENS=65536 MAX_RUNNING=40
AGG_MEM_FRAC=0.80 PD_MEM_FRAC=0.86
PD_EXTRA_ARGS="--encoder-transfer-backend mooncake"
ENC_EXTRA_ARGS="--encoder-transfer-backend mooncake"
--agg-gpus 1,2,3,4 --pd-gpus 1,2,3,4
--enc-gpus-2e1pd 5,6 --enc-gpus-3e1pd 5,6,7
```

Reason for the workload:

- Previous 8-image and 16-image saturated tests showed E/PD losing badly because
  PD queue drain, not raw embedding transfer, dominated.
- A low-rate, very image-heavy, short-output workload should reduce PD queueing
  and give E/PD its best chance to win by moving vision encoding off the TP4
  language workers.
- A prior 32-image AGG run with `AGG_MEM_FRAC=0.86` OOMed during warmup, so this
  retry used `AGG_MEM_FRAC=0.80`.

AGG result:

```text
Successful requests: 8
Total images: 256
Request throughput: 0.08 req/s
Mean TTFT: 15005.65 ms
Mean E2E: 18453.49 ms
Input throughput: 5129.04 tok/s
```

AGG analysis:

- `AGG_MEM_FRAC=0.80` fixed the 32-image warmup OOM. Startup logged
  `max_total_num_tokens=2434640` and about 27.2 GB available GPU memory after
  graph capture.
- Each main request had about 65k input tokens.
- At `RATE=0.1`, AGG did not build queue; this is a latency baseline, not a
  saturated-throughput baseline.
- Baseline to beat for this case: mean TTFT about 15.0s, mean E2E about 18.45s.

2E1PD Mooncake result:

```text
Status: failed/stalled, then manually killed so the matrix could continue.
```

Observed 2E1PD behavior:

- Encoders on GPUs `5,6` became ready quickly.
- PD started with Mooncake and loaded successfully. Startup logged
  `max_total_num_tokens=2804112` and about 18.8 GB available GPU memory after
  graph capture.
- The benchmark started, sent a 32-image request, and the encoders logged
  `/encode` and `/send` success.
- PD logged 32 `Get embedding slice for Modality.IMAGE, num_tokens=2040` lines
  for the request, then stopped making progress.
- No large `Prefill batch` appeared after the embedding slices.
- GPU utilization on PD and encoder GPUs dropped to idle.
- Logs stopped updating for several minutes; the request never completed.

2E1PD analysis:

- This is a Mooncake embedding handoff or receive-side stall after embeddings
  arrive at PD, not a language compute bottleneck.
- The same request size was handled by AGG, so the stall is E/PD transfer path
  specific.
- `Local segment descriptor not found` appeared during Mooncake init/transfer,
  but that also appeared during initialization; the actionable symptom is that
  PD receives all slices and never schedules the language prefill.
- Next action: let 3E1PD Mooncake run to see whether the stall is 2E-specific.
  If it stalls the same way, stop using Mooncake for image E/PD in this search
  and switch back to `zmq_to_scheduler` or `zmq_to_tokenizer`.

3E1PD Mooncake status:

```text
Status: failed/stalled, then manually killed after collecting stack traces.
Encoders: GPUs 5,6,7 ready.
PD: loaded and benchmark started.
```

Observed 3E1PD behavior:

- Same symptom as 2E1PD Mooncake.
- PD received/logged all 32 image embedding slices:

```text
Get embedding slice for Modality.IMAGE, num_tokens=2040
... repeated 32 times ...
```

- No large language `Prefill batch` appeared after the slices.
- Health degraded:

```text
Health check failed. Server couldn't get a response from detokenizer for last 20 seconds.
```

- Captured stack traces under:

```text
results/native_sglang_matrix_20260623_051903_img32_agg_2e3e_mooncake_mem080/r0.1/3E1PD/debug/pyspy_all.txt
```

Important stack evidence:

```text
TP0: get_next_batch_to_run -> _abort_on_waiting_timeout
TP1/TP2/TP3: recv_requests -> broadcast_pyobj -> torch.distributed.broadcast -> Work::wait
detokenizer: waiting in recv_pyobj
bench_serving: waiting for HTTP response
```

3E1PD Mooncake analysis:

- This confirms the 2E stall was not specific to two encoders.
- Mooncake delivers embeddings to the tokenizer-side receiver, and Qwen-VL
  multimodal processing builds the 32 precomputed embedding slices. The failure
  happens after that, before the TP scheduler ranks consistently receive/schedule
  the request.
- The likely failure mechanism is the tokenizer-to-scheduler TP broadcast path
  for a very large multimodal request containing precomputed embeddings. TP
  nonzero ranks wait in `broadcast_pyobj` while rank0 has moved on; no prefill is
  scheduled and the HTTP request hangs.
- This points away from raw encoder throughput and toward a transfer/workflow
  issue: for large image counts, Mooncake-to-tokenizer is the wrong path unless
  we also change how precomputed embeddings reach scheduler ranks.

Next action for failed tests:

- Rerun the failed 32-image E/PD cases with `encoder_transfer_backend` left at
  the default `zmq_to_scheduler`, which sends embeddings directly to scheduler
  ranks and should avoid the huge tokenizer-to-scheduler broadcast.
- If `zmq_to_scheduler` completes but loses, try `zmq_to_tokenizer` only as a
  diagnostic for smaller image counts; it likely has the same large-object risk
  as Mooncake.
- If all ZMQ backends still lose on 32 images, search lower image counts and/or
  lower resolutions where encoder offload helps without creating a giant
  embedding handoff.

Current resource caveat:

- After the failed Mooncake run was stopped, unrelated VLLM/specbench jobs were
  active on GPUs `0`, `5`, and `6`. Only GPUs `1,2,3,4,7` were free for SGLang.
- Do not rerun the requested 2E/3E placement until `5,6` clear again, unless the
  user explicitly authorizes using/killing those unrelated jobs.

Harness update after this failure:

- `native_sglang.sh` now supports `BENCH_TIMEOUT`.
- `native_sglang_matrix.sh` forwards `BENCH_TIMEOUT`.
- Future stalled benchmark cases should use a timeout so failures are recorded
  and the matrix can continue without manual cleanup.

### Try: 32 images, low request rate, ZMQ-to-scheduler rerun

Run root:

```text
results/native_sglang_matrix_20260623_055802_img32_2e3e_zmq_scheduler_mem086
```

Common workload and knobs:

```bash
NUM_PROMPTS=8 IMAGE_COUNT=32 IMAGE_RES=1080p RATE=0.1 OUTPUT_LEN=16
CHUNKED=65536 MAX_PREFILL_TOKENS=65536 MAX_RUNNING=40
PD_MEM_FRAC=0.86 BENCH_TIMEOUT=420
--pd-gpus 1,2,3,4
--enc-gpus-2e1pd 5,6
--enc-gpus-3e1pd 5,6,7
# No encoder-transfer override, so native SGLang uses zmq_to_scheduler.
```

Reason for retry:

- The earlier 2E1PD and 3E1PD Mooncake failures both wedged after all 32 image
  embedding slices were delivered, before language prefill.
- The stack traces pointed at scheduler-rank broadcast/desync after a large
  tokenizer-side multimodal object. The default `zmq_to_scheduler` path should
  send embeddings directly to scheduler ranks and avoid that broadcast path.
- The user explicitly asked not to abandon failed 2E/3E tests; this rerun is the
  concrete fix attempt for the failed 2E1PD setup.

Resource caveat:

- Unrelated VLLM/specbench processes were active on GPUs `5,6` during this run.
  They were not killed. This contaminates encoder performance, so the 2E result
  below is useful as a correctness/stall-resolution result but should not be
  treated as a clean performance comparison.

2E1PD result:

```text
Status: completed
Successful requests: 8
Total images: 256
Request throughput: 0.07 req/s
Mean TTFT: 30291.42 ms
Mean E2E: 48778.05 ms
Input throughput: 4659.43 tok/s
Peak concurrent requests: 6
```

2E1PD analysis:

- `zmq_to_scheduler` resolved the failed-test behavior. Unlike Mooncake, the
  rerun progressed from embedding receive to real language prefill:
  `Prefill batch, #new-token: 65440`.
- The benchmark completed without the previous post-embedding deadlock.
- Performance still loses badly to the 32-image AGG baseline:

```text
AGG baseline:          mean TTFT 15005.65 ms, mean E2E 18453.49 ms
2E1PD ZMQ rerun:       mean TTFT 30291.42 ms, mean E2E 48778.05 ms
```

- The result is additionally penalized by VLLM contention on encoder GPUs 5/6.
  A clean rerun is still required before making a final call on this workload.
- This result suggests the next optimization direction is workload shape and
  encoder/PD queue balance, not Mooncake for 32-image requests. Scheduler-side
  ZMQ is the stable transfer workflow for this large-image-count path.

3E1PD ZMQ status:

```text
Status: completed
```

3E1PD result:

```text
Successful requests: 8
Total images: 256
Request throughput: 0.07 req/s
Mean TTFT: 27238.87 ms
Mean E2E: 34660.37 ms
Input throughput: 4750.85 tok/s
Peak concurrent requests: 4
```

3E1PD analysis:

- `zmq_to_scheduler` also resolved the failed 3E behavior. The run completed
  and did not reproduce the Mooncake post-embedding deadlock.
- 3E was materially better than 2E on this 32-image case:

```text
2E1PD ZMQ:  mean TTFT 30291.42 ms, mean E2E 48778.05 ms
3E1PD ZMQ:  mean TTFT 27238.87 ms, mean E2E 34660.37 ms
```

- It still loses to AGG for the same workload:

```text
AGG baseline: mean TTFT 15005.65 ms, mean E2E 18453.49 ms
3E1PD ZMQ:    mean TTFT 27238.87 ms, mean E2E 34660.37 ms
```

- This result is contaminated: unrelated VLLM workers were active on encoder
  GPUs `5,6` during the run and consumed most of those GPUs' memory, with
  intermittent high SM utilization. GPU7 was clean. A clean 3E rerun is still
  required for a final comparison.
- Even with contamination, 3E reduced E2E by about 29% versus 2E, so adding a
  third encoder helps the large-image case. The next clean-resource experiment
  should keep 3E but reduce request concurrency or image count to avoid the E/PD
  waiting-image queue while preserving enough vision work to matter.

Next action after this try:

- Do not use Mooncake for the 32-image path unless changing the tokenizer to
  scheduler handoff; it has a reproducible post-embedding stall.
- Use `zmq_to_scheduler` for large-image E/PD.
- Wait for clean encoder GPUs before treating new performance numbers as valid.
- Sweep workload shape next: fewer requests at low rate for per-request latency,
  then intermediate image counts (`24`, `16`) with rates that build enough AGG
  pressure without creating a deep E/PD image queue.

### Try: TP2 AGG resource-allocation baseline, 32 images

Run root:

```text
results/native_sglang_matrix_20260623_062018_tp2_img32_agg_baseline
```

Workload and knobs:

```bash
NUM_PROMPTS=8 IMAGE_COUNT=32 IMAGE_RES=1080p RATE=0.1 OUTPUT_LEN=16
CHUNKED=65536 MAX_PREFILL_TOKENS=65536 MAX_RUNNING=24
AGG_MEM_FRAC=0.80
--cases agg --agg-gpus 1,2 --pd-gpus 1,2
```

Reason for this try:

- TP4 AGG is very strong on the 32-image latency workload. A possible
  resource-allocation win for E/PD is TP2 language workers plus external
  encoders, compared against TP2 aggregation on the same language GPU count.
- Reducing TP from 4 to 2 also reduces scheduler-side embedding fanout for E/PD
  from 4 copies to 2 copies, which may materially reduce the ZMQ handoff cost.

Result:

```text
Status: failed before readiness.
```

Failure details:

- Weight load completed on TP2 with about `110.88 GB` used per rank and
  `27.39 GB` available.
- Scheduler then failed during KV memory pool sizing:

```text
RuntimeError: Not enough memory. Please try to increase --mem-fraction-static.
```

Analysis:

- This was not the same as the earlier TP4 32-image warmup OOM. For TP2,
  `AGG_MEM_FRAC=0.80` leaves too little static memory for the model plus minimum
  KV pool after loading the much larger per-rank shard.
- Concrete retry: increase `AGG_MEM_FRAC` to `0.90`, reduce `MAX_RUNNING`, and
  cap CUDA graph capture batch size. If that still fails, use a smaller
  image-count TP2 baseline (`24` or `16`) rather than abandoning TP2.

TP2 retry:

```text
results/native_sglang_matrix_20260623_062754_tp2_img32_agg_mem090_cg8
```

Retry knobs:

```bash
AGG_MEM_FRAC=0.90 MAX_RUNNING=8 AGG_EXTRA_ARGS="--cuda-graph-max-bs 8"
```

Retry result:

```text
Status: failed during benchmark warmup.
```

Retry analysis:

- The higher static memory fraction fixed the KV-pool startup failure.
- The server reached readiness with `max_total_num_tokens=301664` and about
  `13.61 GB` available GPU memory after graph capture.
- It was then killed by the first 32-image benchmark warmup request. This points
  to activation/dynamic-memory pressure for a 65k-token multimodal request on
  TP2, not to weight-load failure.
- Concrete next fix before abandoning TP2 32-image AGG: reduce the KV pool with
  `--max-total-tokens 80000`, lower static fraction to recover dynamic memory,
  disable CUDA graph capture, and run with very low concurrency. If that still
  fails, TP2 cannot support the 32-image case and the TP2 search should move to
  `IMAGE_COUNT=24` or `16`.

TP2 retry 2:

```text
results/native_sglang_matrix_20260623_064228_tp2_img32_agg_mem084_maxtok80k_nograph
```

Retry knobs:

```bash
AGG_MEM_FRAC=0.84
MAX_RUNNING=2 PREFILL_MAX=1
AGG_EXTRA_ARGS="--disable-cuda-graph --max-total-tokens 80000"
```

Retry 2 result:

```text
Status: failed during benchmark warmup.
```

Retry 2 analysis:

- The server reached readiness with `max_total_num_tokens=80000` and about
  `23.58 GB` available GPU memory, so the KV pool and CUDA graph issues were
  addressed.
- It was still killed by the first 32-image warmup request. This makes TP2
  aggregation unsuitable for the 32-image 1080p case under these memory
  constraints.
- Next action: run matching TP2 `3E1PD` with the same low-concurrency language
  settings. If it completes, E/PD has a resource-allocation advantage for this
  case even before measuring a latency win: TP2 aggregation cannot serve the
  32-image workload, while TP2 language-only plus external encoders may be able
  to.

### Try: TP2 3E1PD resource-allocation rerun, 32 images

Run root:

```text
results/native_sglang_matrix_20260623_065641_tp2_img32_3e_mem084_maxtok80k_nograph
```

Workload and knobs:

```bash
NUM_PROMPTS=8 IMAGE_COUNT=32 IMAGE_RES=1080p RATE=0.1 OUTPUT_LEN=16
CHUNKED=65536 MAX_PREFILL_TOKENS=65536
PREFILL_MAX=1 MAX_RUNNING=2
PD_MEM_FRAC=0.84
PD_EXTRA_ARGS="--disable-cuda-graph --max-total-tokens 80000"
BENCH_TIMEOUT=600
--cases 3e1pd --pd-gpus 1,2 --enc-gpus-3e1pd 5,6,7
```

Reason for this retry:

- The matching TP2 aggregation case failed after three memory/resource fixes.
- This rerun tested whether moving vision work to external encoders lets the
  same TP2 language pool serve the 32-image 1080p request shape.
- The failed Mooncake transfer path was avoided; this run used native SGLang's
  stable `zmq_to_scheduler` encoder transfer path.

Result:

```text
Status: completed
Successful requests: 8
Total images: 256
Request throughput: 0.06 req/s
Mean TTFT: 45686.22 ms
Mean E2E: 48366.62 ms
Input throughput: 4061.04 tok/s
Peak concurrent requests: 6
```

Analysis:

- This resolves the TP2 3E1PD failure question: with `zmq_to_scheduler`, low
  PD concurrency, disabled CUDA graph, and `--max-total-tokens 80000`, E/PD can
  complete the 32-image workload on TP2 language GPUs.
- The run passed the point where TP2 aggregation was killed. PD memory rose to
  roughly `142.7 GB` per GPU during benchmark, but it did not OOM.
- This is a resource-allocation/capability advantage over TP2 aggregation:
  the same two language GPUs cannot serve the aggregate 32-image case, while
  language-only TP2 plus external encoders can.
- It is not the requested latency/throughput win versus the stronger TP4 AGG
  baseline:

```text
TP4 AGG 32-image baseline: mean TTFT 15005.65 ms, mean E2E 18453.49 ms, throughput 0.08 req/s
TP2 3E1PD 32-image:        mean TTFT 45686.22 ms, mean E2E 48366.62 ms, throughput 0.06 req/s
```

- The language prefill path is now the bottleneck. During the completed run,
  encoders were mostly idle outside short image bursts, while PD prefilled
  approximately `65k` tokens per request at about `4.8k-6.1k tok/s`.
- Next action: stop treating 32-image TP2 as the target performance win. Use
  clean GPUs `1,2,3,4` for TP4 AGG/PD and GPUs `5,6,7` for encoders, then sweep
  workload shapes where aggregate vision work is heavy enough to matter but the
  E/PD embedding handoff and PD queue do not dominate. Start with `IMAGE_COUNT`
  `16` and `24`, and test higher request rates than `0.1`.

### Try: Qwen3-VL-32B-FP8, 8 images, high request rate

Reason for switching model size:

- The 235B model is dominated by language/MoE prefill. Even when vision is moved
  to encoders, the PD language side drains image-heavy requests much slower than
  TP4 aggregation.
- A smaller VL model should make the vision/offload portion a larger share of
  total work, so it is a reasonable axis to test before concluding E/PD cannot
  beat aggregation on this server.

Model:

```bash
MODEL=/mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-32B-Instruct-FP8
SERVED=Qwen/Qwen3-VL-32B-Instruct-FP8
```

Workload target:

```bash
NUM_PROMPTS=128 IMAGE_COUNT=8 IMAGE_RES=1080p RATE=2.0 OUTPUT_LEN=16
CHUNKED=65536 MAX_PREFILL_TOKENS=65536
--agg-gpus 1,2,3,4 --pd-gpus 1,2,3,4 --enc-gpus-3e1pd 5,6,7
```

Failure/repair chain:

1. Initial run:

```text
results/native_sglang_matrix_20260623_0719_qwen32b_img8_r2_agg_3e
```

The aggregate server spent about 24 minutes in CUDA graph capture and then OOMed
during the benchmark:

```text
CUDA out of memory. Tried to allocate 640 MiB, only about 450 MiB free.
```

Fix attempted: lower static memory and remove CUDA graph capture.

2. Retry with memory headroom and no CUDA graph:

```text
results/native_sglang_matrix_20260623_0752_qwen32b_img8_r2_nograph_mem080
```

Knobs:

```bash
MEM_FRAC=0.80
AGG_EXTRA_ARGS="--disable-cuda-graph --max-total-tokens 1000000"
PD_EXTRA_ARGS="--disable-cuda-graph --max-total-tokens 1000000"
```

This failed before a valid test because port `38000` was still bound from the
previous failed run. The fix was to move to a fresh port block instead of giving
up on the configuration.

3. Retry on fresh ports:

```text
results/native_sglang_matrix_20260623_0757_qwen32b_img8_r2_nograph_mem080_ports381
```

Knobs added:

```bash
PD_PORT=38100 ENC_PORT_BASE=38102
```

This failed in server warmup because DeepGEMM JIT/autotune exceeded the default
300s watchdog. The configuration itself had not reached benchmark execution.

Fix attempted: raise the server watchdog timeout.

4. Retry with longer watchdog:

```text
results/native_sglang_matrix_20260623_0809_qwen32b_img8_r2_nograph_mem080_watchdog
```

Knob added:

```bash
--watchdog-timeout 1800
```

This got past the 300s watchdog, but the internal SGLang warmup HTTP request
timed out at 600s while DeepGEMM was still around 65%. The failure was still
startup/JIT related, not a benchmark result.

Fix attempted: avoid the slow DeepGEMM path for this FP8 model.

5. Valid retry with Triton FP8 GEMM:

```text
results/native_sglang_matrix_20260623_0827_qwen32b_img8_r2_tritonfp8
```

Knobs:

```bash
MEM_FRAC=0.80 AGG_MEM_FRAC=0.80 PD_MEM_FRAC=0.80
AGG_EXTRA_ARGS="--disable-cuda-graph --max-total-tokens 1000000 --fp8-gemm-backend triton"
PD_EXTRA_ARGS="--disable-cuda-graph --max-total-tokens 1000000 --fp8-gemm-backend triton"
ENC_EXTRA_ARGS="--fp8-gemm-backend triton"
PD_PORT=38300 ENC_PORT_BASE=38302
READY_TIMEOUT=1800 BENCH_TIMEOUT=1200
```

AGG result:

```text
Successful requests: 128
Total images: 1024
Request throughput: 0.74 req/s
Input token throughput: 12073.16 tok/s
Mean TTFT: 79834.86 ms
Mean E2E: 118016.08 ms
Peak concurrent requests: 124
```

3E1PD result:

```text
Successful requests: 128
Total images: 1024
Request throughput: 0.28 req/s
Input token throughput: 4524.87 tok/s
Mean TTFT: 308478.42 ms
Mean E2E: 387278.27 ms
Peak concurrent requests: 128
```

Analysis:

- The failed tests were not abandoned: the OOM was addressed with memory
  headroom and no CUDA graph, the stale port was addressed with a fresh port
  block, the watchdog failure was addressed with a longer watchdog, and the
  DeepGEMM warmup timeout was addressed by switching FP8 GEMM to Triton.
- After those repairs, the comparison is valid and E/PD loses badly:

```text
AGG:    0.74 req/s, mean TTFT  79.8s, mean E2E 118.0s
3E1PD:  0.28 req/s, mean TTFT 308.5s, mean E2E 387.3s
```

- Logs show the encoders mostly idle outside bursts, while the language PD side
  is the long pole. For this 32B high-concurrency 8-image workload, E/PD creates
  a deeper queue and does not convert encoder offload into aggregate throughput.
- Next action: move down to `Qwen3-VL-8B-Instruct`. Prior older XPU-style
  results showed a possible throughput win for an 8B E/PD shape, so native GPU
  reproduction with clean resources is the highest-value next search direction.

### Try: Qwen3-VL-8B, 8 images, rate 1.8, native 3E1PD

Run root:

```text
results/native_sglang_matrix_20260623_qwen8b_img8_r18_graph128
```

Reason for this try:

- Reproduce the closest native-GPU version of the older 8B result where E/PD
  had slightly higher throughput than aggregation.
- Use clean H200 placement: AGG/PD on GPUs `1,2,3,4`, encoders on `5,6,7`.
- Keep CUDA graph enabled but cap graph batch capture at 128 so startup remains
  bounded.

Common workload and knobs:

```bash
MODEL=/mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-8B-Instruct
SERVED=Qwen/Qwen3-VL-8B-Instruct
NUM_PROMPTS=128 IMAGE_COUNT=8 IMAGE_RES=1080p RATE=1.8
INPUT_LEN=128 OUTPUT_LEN=64
CHUNKED=65536 MAX_PREFILL_TOKENS=65536
MAX_RUNNING=128 PREFILL_MAX=16
MEM_FRAC=0.90 AGG_MEM_FRAC=0.90 PD_MEM_FRAC=0.90
AGG_EXTRA_ARGS="--cuda-graph-max-bs 128"
PD_EXTRA_ARGS="--cuda-graph-max-bs 128"
BENCH_TIMEOUT=900
```

AGG result:

```text
Successful requests: 128
Total images: 1024
Request throughput: 0.97 req/s
Input token throughput: 15921.66 tok/s
Mean TTFT: 52998.28 ms
Mean E2E: 91586.33 ms
Peak concurrent requests: 125
```

3E1PD result:

```text
Successful requests: 128
Total images: 1024
Request throughput: 0.40 req/s
Input token throughput: 6514.57 tok/s
Mean TTFT: 215480.37 ms
Mean E2E: 279553.41 ms
Peak concurrent requests: 127
```

Analysis:

- This is a clean completed run, not a failure. E/PD loses:

```text
AGG:    0.97 req/s, mean TTFT  53.0s, mean E2E  91.6s
3E1PD:  0.40 req/s, mean TTFT 215.5s, mean E2E 279.6s
```

- The run explains why the earlier possible 8B win did not reproduce here:
  native TP4 aggregation is much stronger than the older aggregate baseline
  (`0.97 req/s` here versus `0.76 req/s` in the old result).
- E/PD eventually completed all requests, but the early run was bottlenecked by
  image readiness. The PD log reached `waiting-image-req: 47`, and language
  prefill did not drain steadily until after most encoder work completed.
- PD prefill throughput after image readiness cleared was only around
  `15.5k tok/s`, while AGG reached repeated `~29k-30k tok/s` prefill passes.
- The E/PD `info` log emitted `4132` `Get embedding slice` lines for this run,
  plus encoder send/receive logs. That logging is asymmetric versus AGG and is
  a concrete workflow overhead to remove before concluding this workload loses.

Next concrete retry:

```bash
LOG_LEVEL=warning
# same model, placement, workload, memory, and graph knobs
```

Purpose: rerun the same 8B comparison with per-image E/PD log overhead removed.
If E/PD still loses, sweep image count/rate rather than retrying the same shape.

### Retry: Qwen3-VL-8B, same workload, warning-level logs

Run root:

```text
results/native_sglang_matrix_20260623_qwen8b_img8_r18_graph128_warning
```

Reason for retry:

- The previous E/PD run emitted `4132` per-image embedding-slice log lines.
- This retry kept the same workload, placement, memory, and graph knobs, but set
  `LOG_LEVEL=warning` for both AGG and E/PD so E/PD logging overhead was not the
  cause of the comparison.

AGG result:

```text
Successful requests: 128
Total images: 1024
Request throughput: 0.97 req/s
Input token throughput: 15994.41 tok/s
Mean TTFT: 52550.08 ms
Mean E2E: 90990.55 ms
Peak concurrent requests: 125
```

3E1PD result:

```text
Successful requests: 128
Total images: 1024
Request throughput: 0.39 req/s
Input token throughput: 6421.89 tok/s
Mean TTFT: 216489.82 ms
Mean E2E: 284286.87 ms
Peak concurrent requests: 127
```

Analysis:

- Lower logging did not help E/PD. It slightly improved AGG and slightly
  worsened/noised E/PD:

```text
AGG info -> warning:    0.97 -> 0.97 req/s, mean E2E 91.6s -> 91.0s
3E info -> warning:     0.40 -> 0.39 req/s, mean E2E 279.6s -> 284.3s
```

- Therefore the native 8B TP4 comparison is not bottlenecked by log overhead.
  The stable signal is that TP4 aggregation drains this fixed 8-image workload
  much faster than TP4 language-only plus three TP1 encoders.
- Likely reason: aggregate TP4 is already using four H200s for the vision path,
  while 3E uses only three single-GPU encoders and adds embedding handoff plus
  readiness scheduling. With only three encoder GPUs available, fixed 8-image
  TP4 E/PD does not beat TP4 aggregation.

Next action:

- Try resource-allocation comparisons where E/PD has a more realistic advantage:
  TP2 aggregation versus TP2 language-only plus three encoders. This uses the
  same aggregation/PD GPU set class but allows E/PD to add clean encoder GPUs.
  Start with the same 8-image workload, then increase image count if both sides
  complete and E/PD is close.

### Try: Qwen3-VL-8B TP2 resource allocation, 8 images

Initial run root:

```text
results/native_sglang_matrix_20260623_qwen8b_tp2_img8_r18_warning
```

Common workload:

```bash
NUM_PROMPTS=128 IMAGE_COUNT=8 IMAGE_RES=1080p RATE=1.8
INPUT_LEN=128 OUTPUT_LEN=64
CHUNKED=65536 MAX_PREFILL_TOKENS=65536
MAX_RUNNING=128 PREFILL_MAX=16
LOG_LEVEL=warning
```

Placement:

```bash
AGG/PD: GPUs 1,2, TP=2
Encoders: GPUs 5,6,7
```

Initial AGG result:

```text
Status: failed during main benchmark.
```

Failure details:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB.
GPU 0 ... 490.06 MiB is free ... process has 136.81 GiB in use.
```

Analysis:

- TP2 AGG reached readiness and passed warmup. It was killed by dynamic-memory
  pressure during the real 8-image request stream.
- This is the same repair class as earlier large-image AGG failures: static KV
  reservation at `AGG_MEM_FRAC=0.90` left too little activation/workspace
  headroom.
- Concrete repair: rerun only the failed AGG baseline with lower static memory.

3E1PD result from the initial TP2 run:

```text
Successful requests: 128
Request throughput: 0.39 req/s
Input token throughput: 6398.05 tok/s
Mean TTFT: 215304.63 ms
Mean E2E: 285242.37 ms
Peak concurrent requests: 127
```

AGG repair run root:

```text
results/native_sglang_matrix_20260623_qwen8b_tp2_img8_r18_agg_mem086
```

Repair knobs:

```bash
AGG_MEM_FRAC=0.86
# Same workload, TP2 placement, max-running, graph cap, and warning logs.
```

Repaired AGG result:

```text
Successful requests: 128
Request throughput: 0.89 req/s
Input token throughput: 14559.06 tok/s
Mean TTFT: 58293.19 ms
Mean E2E: 103501.19 ms
Peak concurrent requests: 125
```

Analysis:

- The failed TP2 AGG test was repaired and rerun successfully by lowering static
  KV reservation from `0.90` to `0.86`.
- TP2 E/PD is not a win:

```text
TP2 AGG mem086:  0.89 req/s, mean TTFT  58.3s, mean E2E 103.5s
TP2 3E1PD:       0.39 req/s, mean TTFT 215.3s, mean E2E 285.2s
```

- Reducing language TP from 4 to 2 does not make this 8-image native E/PD path
  competitive. TP2 aggregation remains much faster once memory headroom is
  configured correctly.

Next action:

- Continue the resource-allocation axis with TP1 aggregation versus TP1 PD plus
  three encoders. Use lower `MAX_RUNNING` and graph cap to stay within TP1 KV
  capacity, and rerun failures with memory fixes if needed.

### Try: Qwen3-VL-8B TP1 resource allocation, 8 images

Run root:

```text
results/native_sglang_matrix_20260623_qwen8b_tp1_img8_r18_warning
```

Reason for this try:

- TP4 and TP2 aggregation both beat E/PD after memory was configured correctly.
- TP1 aggregation has the least aggregate vision/language GPU capacity, while
  TP1 E/PD can still use three external encoders. If E/PD can win through
  resource allocation, this is a plausible place.

Common workload and knobs:

```bash
NUM_PROMPTS=128 IMAGE_COUNT=8 IMAGE_RES=1080p RATE=1.8
INPUT_LEN=128 OUTPUT_LEN=64
CHUNKED=65536 MAX_PREFILL_TOKENS=65536
MAX_RUNNING=64 PREFILL_MAX=16
MEM_FRAC=0.86 AGG_MEM_FRAC=0.86 PD_MEM_FRAC=0.86
LOG_LEVEL=warning
AGG_EXTRA_ARGS="--cuda-graph-max-bs 64"
PD_EXTRA_ARGS="--cuda-graph-max-bs 64"
```

Placement:

```bash
AGG/PD: GPU 1, TP=1
Encoders: GPUs 5,6,7
```

AGG result:

```text
Successful requests: 128
Request throughput: 0.86 req/s
Input token throughput: 14052.69 tok/s
Mean TTFT: 50779.54 ms
Mean E2E: 95213.12 ms
Peak concurrent requests: 125
```

3E1PD result:

```text
Successful requests: 128
Request throughput: 0.38 req/s
Input token throughput: 6244.33 tok/s
Mean TTFT: 212017.71 ms
Mean E2E: 273672.89 ms
Peak concurrent requests: 127
```

Analysis:

- TP1 E/PD is still not a win:

```text
TP1 AGG:    0.86 req/s, mean TTFT  50.8s, mean E2E  95.2s
TP1 3E1PD:  0.38 req/s, mean TTFT 212.0s, mean E2E 273.7s
```

- Across TP4, TP2, and TP1, native 8B 3E1PD stays near `0.38-0.40 req/s`.
  Reducing language TP does not materially change E/PD throughput, so the
  bottleneck is not the PD language GPU count for this shape.
- The aggregate path remains unexpectedly strong even at TP1. This suggests
  the fixed 8-image workload is not sufficiently punishing aggregation's local
  vision path, while E/PD pays encoder dispatch and embedding handoff overhead.

Next action:

- Increase image count and lower the number of prompts so total token volume
  stays bounded. A useful next shape is `IMAGE_COUNT=32`, `NUM_PROMPTS=32`,
  `OUTPUT_LEN=16`, TP1 AGG versus TP1 3E1PD. This should stress local aggregate
  vision processing more directly while keeping the total input token count
  near the 8-image/128-prompt tests.

### Try: Qwen3-VL-8B TP1, 32 images per request

Run root:

```text
results/native_sglang_matrix_20260623_qwen8b_tp1_img32_r04_warning
```

Reason for this try:

- The 8-image TP1 run still favored aggregation. The next axis was to make each
  request more vision-heavy while keeping total images/tokens roughly constant.
- `32 prompts * 32 images = 1024 images`, matching the previous total image
  count, but each request contains about `65k` input tokens.

Common workload and knobs:

```bash
NUM_PROMPTS=32 IMAGE_COUNT=32 IMAGE_RES=1080p RATE=0.4
INPUT_LEN=128 OUTPUT_LEN=16
CHUNKED=65536 MAX_PREFILL_TOKENS=65536
MAX_RUNNING=16 PREFILL_MAX=1
MEM_FRAC=0.80 AGG_MEM_FRAC=0.80 PD_MEM_FRAC=0.80
LOG_LEVEL=warning
AGG_EXTRA_ARGS="--cuda-graph-max-bs 16"
PD_EXTRA_ARGS="--cuda-graph-max-bs 16"
```

Placement:

```bash
AGG/PD: GPU 1, TP=1
Encoders: GPUs 5,6,7
```

AGG result:

```text
Successful requests: 32
Total images: 1024
Request throughput: 0.18 req/s
Input token throughput: 11562.67 tok/s
Mean TTFT: 72930.05 ms
Mean E2E: 118029.57 ms
Peak concurrent requests: 31
```

3E1PD result:

```text
Successful requests: 32
Total images: 1024
Request throughput: 0.10 req/s
Input token throughput: 6534.08 tok/s
Mean TTFT: 186863.25 ms
Mean E2E: 246941.95 ms
Peak concurrent requests: 32
```

Analysis:

- This is closer but still not a win:

```text
TP1 AGG image32:    0.18 req/s, mean TTFT  72.9s, mean E2E 118.0s
TP1 3E1PD image32:  0.10 req/s, mean TTFT 186.9s, mean E2E 246.9s
```

- Aggregate input throughput dropped from `14.1k tok/s` at 8 images to
  `11.6k tok/s` at 32 images.
- E/PD input throughput stayed near the earlier `6.2k-6.5k tok/s` band. This
  reinforces that native E/PD is dominated by encoder dispatch/embedding
  readiness plus a stable PD drain rate, while aggregation only starts to slow
  materially with larger per-request image counts.

Next action:

- Push to `IMAGE_COUNT=64`, `NUM_PROMPTS=16`, `RATE=0.2`, `OUTPUT_LEN=16` with
  TP1. This keeps total images at 1024, makes each request about `130k` input
  tokens, and is the next likely point where aggregation's local multimodal path
  might fall below E/PD's stable throughput band.

### Try: Qwen3-VL-8B TP1, 64 images per request

Initial run root:

```text
results/native_sglang_matrix_20260623_qwen8b_tp1_img64_r02_warning
```

Common workload:

```bash
NUM_PROMPTS=16 IMAGE_COUNT=64 IMAGE_RES=1080p RATE=0.2
INPUT_LEN=128 OUTPUT_LEN=16
CHUNKED=131072 MAX_PREFILL_TOKENS=131072
MAX_RUNNING=8 PREFILL_MAX=1
MEM_FRAC=0.75 AGG_MEM_FRAC=0.75 PD_MEM_FRAC=0.75
LOG_LEVEL=warning
AGG_EXTRA_ARGS="--cuda-graph-max-bs 8"
PD_EXTRA_ARGS="--cuda-graph-max-bs 8"
```

Initial AGG result:

```text
Status: failed during warmup.
```

Failure details:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.12 GiB.
Failure was in qwen3_vl.py visual MLP while embedding the 64-image request.
```

Analysis:

- This was a local aggregate vision-encoder activation OOM, not a server startup
  or language decode issue.
- The failed AGG test was repaired by explicitly shrinking KV reservation with
  `--max-total-tokens` and lowering active concurrency.

3E1PD result from the initial 64-image run:

```text
Successful requests: 16
Request throughput: 0.04 req/s
Input token throughput: 5802.05 tok/s
Mean TTFT: 198897.07 ms
Mean E2E: 278050.30 ms
Peak concurrent requests: 16
```

AGG repair run root:

```text
results/native_sglang_matrix_20260623_qwen8b_tp1_img64_r02_agg_maxtok600k
```

Repair knobs:

```bash
MAX_RUNNING=4
AGG_EXTRA_ARGS="--cuda-graph-max-bs 4 --max-total-tokens 600000"
# Same image count/rate/output shape.
```

Repaired AGG result:

```text
Successful requests: 16
Request throughput: 0.07 req/s
Input token throughput: 8833.11 tok/s
Mean TTFT: 102487.69 ms
Mean E2E: 135536.68 ms
Peak concurrent requests: 16
```

Analysis:

- The failed 64-image AGG case was repaired and rerun successfully.
- E/PD is still not a win:

```text
TP1 AGG image64 repaired:  0.07 req/s, mean TTFT 102.5s, mean E2E 135.5s
TP1 3E1PD image64:         0.04 req/s, mean TTFT 198.9s, mean E2E 278.1s
```

- The gap is narrowing as image count grows:

```text
image8:   AGG 0.86 req/s vs 3E 0.38 req/s
image32:  AGG 0.18 req/s vs 3E 0.10 req/s
image64:  AGG 0.07 req/s vs 3E 0.04 req/s
```

- However, E/PD input throughput also dropped from the `6.2k-6.5k tok/s` band to
  `5.8k tok/s`, so larger requests increase handoff/readiness pressure too.

Next action:

- Try `IMAGE_COUNT=96`, which is still below the model context limit
  (`96 * 2040 image tokens` plus text), with `NUM_PROMPTS=12` and explicit
  `--max-total-tokens` on both AGG and PD. This is the last reasonable
  fixed-image-count step before hitting the Qwen3-VL context ceiling.

### Try: Qwen3-VL-8B TP1, 96 images per request

Run root:

```text
results/native_sglang_matrix_20260623_qwen8b_tp1_img96_r012_maxtok700k
```

Reason for this try:

- The 64-image TP1 run narrowed the AGG/E-PD throughput gap but still favored
  aggregation.
- `IMAGE_COUNT=96` pushes close to the Qwen3-VL context ceiling while keeping
  total requests small. This was expected to penalize aggregate local vision
  processing more than the previous 32/64-image shapes.

Common workload and knobs:

```bash
MODEL=/mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-8B-Instruct
SERVED=Qwen/Qwen3-VL-8B-Instruct
NUM_PROMPTS=12 IMAGE_COUNT=96 IMAGE_RES=1080p RATE=0.12
INPUT_LEN=128 OUTPUT_LEN=16
CHUNKED=196608 MAX_PREFILL_TOKENS=196608
MAX_RUNNING=3 PREFILL_MAX=1
MEM_FRAC=0.75 AGG_MEM_FRAC=0.75 PD_MEM_FRAC=0.75
LOG_LEVEL=warning
AGG_EXTRA_ARGS="--cuda-graph-max-bs 3 --max-total-tokens 700000"
PD_EXTRA_ARGS="--cuda-graph-max-bs 3 --max-total-tokens 700000"
```

Placement:

```text
AGG/PD: GPU 1, TP=1
Encoders: GPUs 5,6,7
```

AGG result:

```text
Successful requests: 12
Total images: 1152
Request throughput: 0.04 req/s
Input token throughput: 7421.84 tok/s
Mean TTFT: 140475.34 ms
Mean E2E: 159651.05 ms
Peak concurrent requests: 11
```

3E1PD result:

```text
Successful requests: 12
Total images: 1152
Request throughput: 0.02 req/s
Input token throughput: 4723.22 tok/s
Mean TTFT: 275534.94 ms
Mean E2E: 301722.73 ms
Peak concurrent requests: 12
```

Analysis:

- This was a clean completed comparison, not a failed/stalled run.
- E/PD still did not beat aggregation:

```text
TP1 AGG image96:    0.04 req/s, mean TTFT 140.5s, mean E2E 159.7s
TP1 3E1PD image96:  0.02 req/s, mean TTFT 275.5s, mean E2E 301.7s
```

- The AGG/E-PD gap continues to narrow as per-request image count grows, but the
  absolute E/PD drain rate also falls. At 96 images, encoder offload does not
  compensate for embedding readiness, scheduler-side handoff, and language
  prefill scheduling overhead.
- This is near the useful upper bound for fixed 1080p image count because each
  request already has about `196k` input tokens. Pushing higher risks exceeding
  context or spending the entire run in memory-management edge cases rather than
  finding a production-relevant win.

Next action:

- Stop increasing fixed 1080p image count. Instead, test whether E/PD can win on
  a mixed/request-shape or lower-resolution workflow where aggregate cannot use
  huge multimodal prefills efficiently but E/PD avoids the largest embedding
  handoff penalty.
- First clean run after server recovery should use the healthy TP4 placement
  (`1,2,3,4`) and clean encoders (`5,6,7`) with `zmq_to_scheduler`.
- Candidate next shapes:
  - `IMAGE_COUNT=16`, `IMAGE_RES=1080p`, `NUM_PROMPTS=64`, sweep rates around
    `0.2`, `0.4`, `0.8`.
  - `IMAGE_COUNT=32`, `IMAGE_RES=720p`, `NUM_PROMPTS=32`, sweep rates around
    `0.2`, `0.4`.
  - If a case fails, inspect logs, apply the smallest resource/workflow fix, and
    rerun that same case before treating it as a data point.

### Try: 235B 32 images, clean post-recovery ZMQ 2E/3E rerun

Run root:

```text
results/native_sglang_matrix_20260623_1142_img32_clean_2e3e_zmq
```

Reason for this try:

- The earlier 32-image 2E/3E run with default `zmq_to_scheduler` resolved the
  Mooncake post-embedding stall, but its encoder GPUs `5,6` were contaminated by
  unrelated VLLM/specbench jobs.
- After server recovery, GPUs `5,6,7` were free and GPU4 topology was healthy.
  This rerun gives a clean answer for the previously failed 2E/3E setup instead
  of abandoning it.

Workload and knobs:

```bash
MODEL=/mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-235B-A22B-Instruct-FP8
SERVED=Qwen/Qwen3-VL-235B-A22B-Instruct-FP8
NUM_PROMPTS=8 IMAGE_COUNT=32 IMAGE_RES=1080p RATE=0.1
INPUT_LEN=128 OUTPUT_LEN=16
CHUNKED=65536 MAX_PREFILL_TOKENS=65536
MAX_RUNNING=40 PREFILL_MAX=16
MEM_FRAC=0.86 PD_MEM_FRAC=0.86
LOG_LEVEL=warning BENCH_TIMEOUT=900 READY_TIMEOUT=2400
PD_PORT=39400 ENC_PORT_BASE=39402
--pd-gpus 1,2,3,4
--enc-gpus-2e1pd 5,6
--enc-gpus-3e1pd 5,6,7
```

2E1PD result:

```text
Status: completed
Successful requests: 8
Total images: 256
Request throughput: 0.07 req/s
Input token throughput: 4732.01 tok/s
Mean TTFT: 37779.44 ms
Mean E2E: 53241.82 ms
Peak concurrent requests: 6
```

3E1PD result:

```text
Status: completed
Successful requests: 8
Total images: 256
Request throughput: 0.07 req/s
Input token throughput: 4680.19 tok/s
Mean TTFT: 31349.56 ms
Mean E2E: 48033.81 ms
Peak concurrent requests: 6
```

Analysis:

- The failed 2E/3E testing path is now resolved on clean GPUs: both cases
  completed with `zmq_to_scheduler` and did not reproduce the Mooncake
  post-embedding deadlock.
- The clean 3E run improved latency versus clean 2E:

```text
2E1PD: mean TTFT 37.8s, mean E2E 53.2s
3E1PD: mean TTFT 31.3s, mean E2E 48.0s
```

- It still loses to the prior clean AGG result for the same workload:

```text
AGG:    throughput 0.08 req/s, mean TTFT 15.0s, mean E2E 18.5s
2E1PD:  throughput 0.07 req/s, mean TTFT 37.8s, mean E2E 53.2s
3E1PD:  throughput 0.07 req/s, mean TTFT 31.3s, mean E2E 48.0s
```

- Adding GPU7 helps latency but not enough. For this 235B low-rate 32-image
  shape, the PD language-prefill/embedding-readiness workflow remains the long
  pole after encoder offload.

Next action:

- Do not spend more time on 235B 32-image low-rate latency with the current
  native workflow. The clean result confirms the failure repair and confirms no
  win.
- Move to smaller model/resource shapes where vision offload can dominate more:
  first `Qwen3-VL-2B-Instruct`, because it should reduce the language-side
  prefill cost that has been masking encoder offload benefits on 235B/32B/8B.

### Try: Qwen3-VL-2B TP4, 8 images, high request rate

Initial run root:

```text
results/native_sglang_matrix_20260623_qwen2b_tp4_img8_r4_agg3e
```

Reason for this try:

- Larger Qwen3-VL models were dominated by PD language prefill/drain. A 2B model
  should make the vision-encoding fraction larger and give encoder offload a
  better chance.
- Started with TP4 AGG/PD on GPUs `1,2,3,4` and three encoders on `5,6,7`.

Common workload:

```bash
MODEL=/mnt/weka/data/llm-d-models-pv/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/main
SERVED=Qwen/Qwen3-VL-2B-Instruct
NUM_PROMPTS=128 IMAGE_COUNT=8 IMAGE_RES=1080p RATE=4.0
INPUT_LEN=128 OUTPUT_LEN=16
CHUNKED=65536 MAX_PREFILL_TOKENS=65536
MAX_RUNNING=128 PREFILL_MAX=16
MEM_FRAC=0.80 AGG_MEM_FRAC=0.80 PD_MEM_FRAC=0.80
LOG_LEVEL=warning
AGG_EXTRA_ARGS="--cuda-graph-max-bs 128"
PD_EXTRA_ARGS="--cuda-graph-max-bs 128"
```

AGG result:

```text
Successful requests: 128
Total images: 1024
Request throughput: 1.12 req/s
Input token throughput: 18434.27 tok/s
Mean TTFT: 67347.78 ms
Mean E2E: 92752.35 ms
Peak concurrent requests: 125
```

Initial 3E1PD result:

```text
Status: failed during main benchmark.
```

Failure details:

```text
TypeError: 'NoneType' object is not subscriptable
base_processor.py:331 mm_token_num = video_grid_thw[video_idx].prod()
```

Analysis:

- This reproduced the earlier language-only receive-path bug at high
  concurrency: the request is image-only, but `build_input_ids` sees a video
  token while only image grid metadata is available.
- This is not OOM. It is a scheduler-side multimodal reconstruction/request
  metadata race under high E/PD image handoff pressure.

Repair attempt 1:

```text
results/native_sglang_matrix_20260623_qwen2b_tp4_img8_r4_3e_zmq_tokenizer_repair
```

Repair knobs:

```bash
PD_EXTRA_ARGS="--cuda-graph-max-bs 128 --encoder-transfer-backend zmq_to_tokenizer"
ENC_EXTRA_ARGS="--encoder-transfer-backend zmq_to_tokenizer"
```

Repair attempt 1 result:

```text
Status: stalled; manually stopped after repeated embedding receive timeouts.
```

Observed failure:

```text
Embedding recv timeout for request ...
```

Analysis:

- `zmq_to_tokenizer` avoided the immediate `video_grid_thw` scheduler exception,
  but it introduced a worse workflow failure under this 128-request image-heavy
  load: embedding receive timeouts with no benchmark progress.
- This matches earlier large-image Mooncake/tokenizer-path behavior: tokenizer
  side transfer is not stable enough for this search workload.

Repair attempt 2:

```text
results/native_sglang_matrix_20260623_qwen2b_tp4_img8_r4_3e_zmq_scheduler_mc32_repair
```

Repair knobs:

```bash
# Back to stable scheduler-side ZMQ, but cap benchmark concurrency.
MAX_CONCURRENCY=32
```

3E1PD repaired result:

```text
Status: completed
Successful requests: 128
Total images: 1024
Request throughput: 0.61 req/s
Input token throughput: 10053.07 tok/s
Mean TTFT: 39716.43 ms
Mean E2E: 47753.22 ms
Peak concurrent requests: 51
```

Matching AGG concurrency-capped run:

```text
results/native_sglang_matrix_20260623_qwen2b_tp4_img8_r4_agg_mc32
MAX_CONCURRENCY=32
```

AGG concurrency-capped result:

```text
Successful requests: 128
Total images: 1024
Request throughput: 1.20 req/s
Input token throughput: 19684.10 tok/s
Mean TTFT: 18189.34 ms
Mean E2E: 23879.78 ms
Peak concurrent requests: 54
```

Analysis:

- The failed 3E test was repaired by capping benchmark concurrency, which
  confirms the `video_grid_thw` crash is pressure/race related in the
  scheduler-side E/PD receive path.
- The repaired E/PD case still loses to aggregation:

```text
AGG mc32:    1.20 req/s, mean TTFT 18.2s, mean E2E 23.9s
3E1PD mc32:  0.61 req/s, mean TTFT 39.7s, mean E2E 47.8s
```

- The E/PD latency improvement versus the uncapped AGG run was due to lower
  concurrency, not disaggregation; the matched AGG run is about 2x faster.

Next action:

- Increase per-request image count while keeping total images near 1024:
  `IMAGE_COUNT=16`, `NUM_PROMPTS=64`, `RATE=2.0`, `MAX_CONCURRENCY=16`.
  This should increase local AGG vision pressure while avoiding the E/PD
  high-concurrency metadata race.

### Try: Qwen3-VL-2B TP4, 16 images, capped concurrency

Run root:

```text
results/native_sglang_matrix_20260623_qwen2b_tp4_img16_r2_mc16_agg3e
```

Reason for this try:

- The previous Qwen3-VL-2B 8-image 3E1PD run failed under uncapped pressure with
  the `video_grid_thw is None` scheduler-side receive bug.
- The failure was repaired by staying on stable `zmq_to_scheduler` and capping
  benchmark concurrency to `32`, but AGG still won by about 2x.
- This run kept total images at `1024`, doubled images per request to `16`, and
  reduced `MAX_CONCURRENCY` to `16` to keep the failed high-pressure metadata
  race out of the comparison.

Common workload and knobs:

```bash
MODEL=/mnt/weka/data/llm-d-models-pv/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/main
SERVED=Qwen/Qwen3-VL-2B-Instruct
NUM_PROMPTS=64 IMAGE_COUNT=16 IMAGE_RES=1080p RATE=2.0
INPUT_LEN=128 OUTPUT_LEN=16
CHUNKED=65536 MAX_PREFILL_TOKENS=65536
MAX_RUNNING=128 PREFILL_MAX=16 MAX_CONCURRENCY=16
MEM_FRAC=0.80 AGG_MEM_FRAC=0.80 PD_MEM_FRAC=0.80
LOG_LEVEL=warning
AGG_EXTRA_ARGS="--cuda-graph-max-bs 128"
PD_EXTRA_ARGS="--cuda-graph-max-bs 128"
--agg-gpus 1,2,3,4 --pd-gpus 1,2,3,4 --enc-gpus-3e1pd 5,6,7
```

AGG result:

```text
Successful requests: 64
Total images: 1024
Request throughput: 0.57 req/s
Input token throughput: 18804.36 tok/s
Mean TTFT: 20272.66 ms
Mean E2E: 25401.62 ms
Peak concurrent requests: 28
```

3E1PD result:

```text
Successful requests: 64
Total images: 1024
Request throughput: 0.31 req/s
Input token throughput: 10216.60 tok/s
Mean TTFT: 40998.09 ms
Mean E2E: 48127.70 ms
Peak concurrent requests: 25
```

Analysis:

- This is a clean completed run, not a failed test.
- E/PD still loses to aggregation:

```text
AGG mc16 image16:    0.57 req/s, mean TTFT 20.3s, mean E2E 25.4s
3E1PD mc16 image16:  0.31 req/s, mean TTFT 41.0s, mean E2E 48.1s
```

- Compared with the 8-image capped run, increasing images per request reduced
  AGG throughput from `1.20` to `0.57 req/s` and E/PD throughput from `0.61` to
  `0.31 req/s`. The ratio remained roughly 2x in favor of AGG.
- The next bounded step is `IMAGE_COUNT=32`, `NUM_PROMPTS=32`, `RATE=1.0`,
  `MAX_CONCURRENCY=8 or 12`. This keeps total images near `1024`, further
  stresses local aggregate vision processing, and keeps E/PD concurrency low
  enough to avoid the known scheduler-side metadata race.

### Active failure-handling rule after user correction

For every failed benchmark configuration:

- Inspect the failing server and benchmark logs.
- Identify whether the failure is resource/memory, stale port/process,
  transfer-workflow, startup/JIT, or request-concurrency pressure.
- Rerun the same intended case with the smallest targeted repair before moving
  on.
- Document both the failure and the repair result here. Do not treat a failed
  2E1PD or 3E1PD setup as a data point until it is either repaired and rerun or
  there is a concrete reason it cannot be repaired further.

### Try: Qwen3-VL-2B TP4, 32 images, capped concurrency

Run root:

```text
results/native_sglang_matrix_20260623_qwen2b_tp4_img32_r1_mc8_agg2e3e
```

Reason for this try:

- The 16-image 2B comparison still favored AGG by roughly 2x.
- Increasing per-request image count from `16` to `32` keeps total images at
  `1024` but should penalize aggregate local vision processing more directly.
- `MAX_CONCURRENCY=8` keeps client pressure below the known E/PD metadata-race
  zone. `CHUNKED=131072` and `MAX_PREFILL_TOKENS=131072` let AGG/PD admit about
  two 32-image requests per prefill pass.
- Included `2E1PD` explicitly so the previous 2E failure pattern would be
  repaired if it reproduced.

Common workload and knobs:

```bash
MODEL=/mnt/weka/data/llm-d-models-pv/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/main
SERVED=Qwen/Qwen3-VL-2B-Instruct
NUM_PROMPTS=32 IMAGE_COUNT=32 IMAGE_RES=1080p RATE=1.0
INPUT_LEN=128 OUTPUT_LEN=16
CHUNKED=131072 MAX_PREFILL_TOKENS=131072
MAX_RUNNING=64 PREFILL_MAX=2 MAX_CONCURRENCY=8
MEM_FRAC=0.80 AGG_MEM_FRAC=0.80 PD_MEM_FRAC=0.80
LOG_LEVEL=warning
AGG_EXTRA_ARGS="--cuda-graph-max-bs 64"
PD_EXTRA_ARGS="--cuda-graph-max-bs 64"
--agg-gpus 1,2,3,4 --pd-gpus 1,2,3,4
--enc-gpus-2e1pd 5,6 --enc-gpus-3e1pd 5,6,7
```

AGG result:

```text
Successful requests: 32
Total images: 1024
Request throughput: 0.29 req/s
Input token throughput: 19059.59 tok/s
Mean TTFT: 20875.37 ms
Mean E2E: 25367.25 ms
Peak concurrent requests: 14
```

2E1PD result:

```text
Successful requests: 32
Total images: 1024
Request throughput: 0.16 req/s
Input token throughput: 10401.98 tok/s
Mean TTFT: 42273.36 ms
Mean E2E: 46720.47 ms
Peak concurrent requests: 12
```

3E1PD result:

```text
Successful requests: 32
Total images: 1024
Request throughput: 0.19 req/s
Input token throughput: 12322.77 tok/s
Mean TTFT: 34859.92 ms
Mean E2E: 39843.99 ms
Peak concurrent requests: 11
```

Analysis:

- This resolved the 2E concern for this shape: with `MAX_CONCURRENCY=8`, 2E1PD
  completed and did not reproduce the `video_grid_thw is None` crash.
- 3E is better than 2E, so the third encoder helps:

```text
2E1PD image32: 0.16 req/s, mean TTFT 42.3s, mean E2E 46.7s
3E1PD image32: 0.19 req/s, mean TTFT 34.9s, mean E2E 39.8s
```

- E/PD still loses to AGG:

```text
AGG image32:    0.29 req/s, mean TTFT 20.9s, mean E2E 25.4s
3E1PD image32:  0.19 req/s, mean TTFT 34.9s, mean E2E 39.8s
```

- The gap continues to narrow as image count grows:

```text
2B TP4 image8:   AGG 1.20 req/s vs 3E 0.61 req/s  (matched concurrency cap)
2B TP4 image16:  AGG 0.57 req/s vs 3E 0.31 req/s
2B TP4 image32:  AGG 0.29 req/s vs 3E 0.19 req/s
```

- The next step is `IMAGE_COUNT=64`, `NUM_PROMPTS=16`, `RATE=0.5`,
  `MAX_CONCURRENCY=4`, `CHUNKED/MAX_PREFILL_TOKENS=262144`. This keeps total
  images at `1024` and is the most likely next point for aggregate local vision
  work to fall below the 3E offload path. If AGG or E/PD fails, repair that case
  before moving on.

### Try: Qwen3-VL-2B TP4, 64 images, capped concurrency

Run root:

```text
results/native_sglang_matrix_20260623_qwen2b_tp4_img64_r05_mc4_agg3e
```

Reason for this try:

- The AGG/E-PD gap narrowed at 32 images but still favored AGG.
- `IMAGE_COUNT=64` keeps total images at `1024` with only `16` requests and
  pushes local aggregate vision work harder while keeping E/PD concurrency low.
- `--max-total-tokens 1000000` was set for both AGG and PD to avoid wasting H200
  memory on an oversized KV pool; this leaves dynamic headroom for 64-image
  vision processing.

Common workload and knobs:

```bash
MODEL=/mnt/weka/data/llm-d-models-pv/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/main
SERVED=Qwen/Qwen3-VL-2B-Instruct
NUM_PROMPTS=16 IMAGE_COUNT=64 IMAGE_RES=1080p RATE=0.5
INPUT_LEN=128 OUTPUT_LEN=16
CHUNKED=262144 MAX_PREFILL_TOKENS=262144
MAX_RUNNING=32 PREFILL_MAX=2 MAX_CONCURRENCY=4
MEM_FRAC=0.75 AGG_MEM_FRAC=0.75 PD_MEM_FRAC=0.75
LOG_LEVEL=warning
AGG_EXTRA_ARGS="--cuda-graph-max-bs 32 --max-total-tokens 1000000"
PD_EXTRA_ARGS="--cuda-graph-max-bs 32 --max-total-tokens 1000000"
--agg-gpus 1,2,3,4 --pd-gpus 1,2,3,4 --enc-gpus-3e1pd 5,6,7
```

AGG result:

```text
Successful requests: 16
Total images: 1024
Request throughput: 0.13 req/s
Input token throughput: 17265.55 tok/s
Mean TTFT: 24594.38 ms
Mean E2E: 28976.28 ms
Peak concurrent requests: 7
```

3E1PD result:

```text
Successful requests: 16
Total images: 1024
Request throughput: 0.09 req/s
Input token throughput: 11541.23 tok/s
Mean TTFT: 41893.72 ms
Mean E2E: 43927.96 ms
Peak concurrent requests: 7
```

Analysis:

- Both cases completed; no repair was needed for this shape.
- E/PD still loses:

```text
AGG image64:    0.13 req/s, mean TTFT 24.6s, mean E2E 29.0s
3E1PD image64:  0.09 req/s, mean TTFT 41.9s, mean E2E 43.9s
```

- The throughput ratio continues to narrow:

```text
2B TP4 image8:   AGG 1.20 req/s vs 3E 0.61 req/s
2B TP4 image16:  AGG 0.57 req/s vs 3E 0.31 req/s
2B TP4 image32:  AGG 0.29 req/s vs 3E 0.19 req/s
2B TP4 image64:  AGG 0.13 req/s vs 3E 0.09 req/s
```

- However, AGG latency only rose to about `29s`, while 3E latency stayed around
  `44s`; E/PD is still paying a large handoff/readiness cost.
- Next step: test `IMAGE_COUNT=96`, `NUM_PROMPTS=12`, `RATE=0.33`,
  `MAX_CONCURRENCY=3`, `CHUNKED/MAX_PREFILL_TOKENS=393216`. This is close to
  the practical 1080p context ceiling and is the last fixed-1080p image-count
  step before switching to lower resolution or mixed request-rate tests.

### Try: Qwen3-VL-2B TP4, 96 images, near context ceiling

Run root:

```text
results/native_sglang_matrix_20260623_qwen2b_tp4_img96_r033_mc3_agg3e
```

Reason for this try:

- The TP4 1080p image-count sweep was the only axis where the E/PD gap kept
  narrowing.
- `96` images per request is near the useful fixed-1080p context limit while
  still keeping requests valid.
- Client concurrency was capped at `3` and both AGG/PD used
  `--max-total-tokens 1000000` for dynamic-memory headroom.

Common workload and knobs:

```bash
MODEL=/mnt/weka/data/llm-d-models-pv/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/main
SERVED=Qwen/Qwen3-VL-2B-Instruct
NUM_PROMPTS=12 IMAGE_COUNT=96 IMAGE_RES=1080p RATE=0.33
INPUT_LEN=128 OUTPUT_LEN=16
CHUNKED=393216 MAX_PREFILL_TOKENS=393216
MAX_RUNNING=24 PREFILL_MAX=2 MAX_CONCURRENCY=3
MEM_FRAC=0.72 AGG_MEM_FRAC=0.72 PD_MEM_FRAC=0.72
LOG_LEVEL=warning
AGG_EXTRA_ARGS="--cuda-graph-max-bs 24 --max-total-tokens 1000000"
PD_EXTRA_ARGS="--cuda-graph-max-bs 24 --max-total-tokens 1000000"
--agg-gpus 1,2,3,4 --pd-gpus 1,2,3,4 --enc-gpus-3e1pd 5,6,7
```

AGG result:

```text
Successful requests: 12
Total images: 1152
Request throughput: 0.08 req/s
Input token throughput: 15959.03 tok/s
Mean TTFT: 31555.87 ms
Mean E2E: 36160.74 ms
Peak concurrent requests: 5
```

3E1PD result:

```text
Successful requests: 12
Total images: 1152
Request throughput: 0.06 req/s
Input token throughput: 11195.24 tok/s
Mean TTFT: 50356.38 ms
Mean E2E: 51781.94 ms
Peak concurrent requests: 5
```

Analysis:

- Both cases completed; no repair was needed.
- E/PD still loses:

```text
AGG image96:    0.08 req/s, mean TTFT 31.6s, mean E2E 36.2s
3E1PD image96:  0.06 req/s, mean TTFT 50.4s, mean E2E 51.8s
```

- The throughput gap is narrowest here, but AGG latency remains much lower.
- Fixed TP4, fixed-1080p image-count scaling is unlikely to produce a clean win
  without exceeding context limits or moving into impractical request shapes.
- Next axis: resource allocation. Test TP1 AGG on GPU1 versus TP1 PD on GPU1
  plus three encoders on GPUs `5,6,7`. This gives E/PD a real vision
  parallelism advantage while keeping the language GPU count matched. Start from
  the same 96-image workload because TP4 showed the narrowest gap there.

### Try: Qwen3-VL-2B TP1 resource allocation, 96 images

Run root:

```text
results/native_sglang_matrix_20260623_qwen2b_tp1_img96_r033_mc3_agg3e
```

Reason for this try:

- TP4 E/PD still lost at the 1080p image-count ceiling.
- This run matched language GPU count instead: AGG on GPU1 versus PD on GPU1
  plus encoders on GPUs `5,6,7`.
- The intent was to give disaggregation a real vision-parallelism advantage
  while keeping language compute capacity equal.

Common workload and knobs:

```bash
MODEL=/mnt/weka/data/llm-d-models-pv/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/main
SERVED=Qwen/Qwen3-VL-2B-Instruct
NUM_PROMPTS=12 IMAGE_COUNT=96 IMAGE_RES=1080p RATE=0.33
INPUT_LEN=128 OUTPUT_LEN=16
CHUNKED=393216 MAX_PREFILL_TOKENS=393216
MAX_RUNNING=24 PREFILL_MAX=2 MAX_CONCURRENCY=3
MEM_FRAC=0.72 AGG_MEM_FRAC=0.72 PD_MEM_FRAC=0.72
LOG_LEVEL=warning
AGG_EXTRA_ARGS="--cuda-graph-max-bs 24 --max-total-tokens 1000000"
PD_EXTRA_ARGS="--cuda-graph-max-bs 24 --max-total-tokens 1000000"
--agg-gpus 1 --pd-gpus 1 --enc-gpus-3e1pd 5,6,7
```

AGG result:

```text
Successful requests: 12
Total images: 1152
Request throughput: 0.07 req/s
Input token throughput: 13670.78 tok/s
Mean TTFT: 34263.39 ms
Mean E2E: 42334.04 ms
Mean TPOT: 1619.49 ms
Peak concurrent requests: 4
```

3E1PD result:

```text
Successful requests: 12
Total images: 1152
Request throughput: 0.06 req/s
Input token throughput: 10867.43 tok/s
Mean TTFT: 48836.68 ms
Mean E2E: 53432.32 ms
Mean TPOT: 644.06 ms
Peak concurrent requests: 5
```

Analysis:

- Both cases completed; no repair was needed.
- TP1 E/PD still loses on short-output E2E latency and throughput:

```text
TP1 AGG image96:    0.07 req/s, mean TTFT 34.3s, mean E2E 42.3s
TP1 3E1PD image96:  0.06 req/s, mean TTFT 48.8s, mean E2E 53.4s
```

- However, this run exposed a useful new direction: E/PD has a much lower
  decode-side TPOT (`644 ms` vs AGG `1619 ms`) but a worse TTFT. With only short
  outputs, the TTFT penalty dominates; with longer outputs, the decode-side
  advantage may offset it.
- Next action: rerun the same image-heavy shape with longer outputs. Use TP4
  first because TP4 image96 had the narrowest throughput gap and very low E/PD
  TPOT (`197 ms` vs AGG `849 ms` in the short-output run). Start with
  `OUTPUT_LEN=128`, `NUM_PROMPTS=12`, `IMAGE_COUNT=96`, `MAX_CONCURRENCY=3`.

### Try: Qwen3-VL-2B TP4, 96 images, longer outputs

Run root:

```text
results/native_sglang_matrix_20260623_qwen2b_tp4_img96_out128_r033_mc3_agg3e
```

Reason for this try:

- Short-output image96 still favored AGG, but E/PD had much lower decode-side
  TPOT in the short-output metrics.
- `OUTPUT_LEN=128` tests whether lower E/PD decode interference can offset the
  worse E/PD TTFT.

Common workload and knobs:

```bash
MODEL=/mnt/weka/data/llm-d-models-pv/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/main
SERVED=Qwen/Qwen3-VL-2B-Instruct
NUM_PROMPTS=12 IMAGE_COUNT=96 IMAGE_RES=1080p RATE=0.33
INPUT_LEN=128 OUTPUT_LEN=128
CHUNKED=393216 MAX_PREFILL_TOKENS=393216
MAX_RUNNING=24 PREFILL_MAX=2 MAX_CONCURRENCY=3
MEM_FRAC=0.72 AGG_MEM_FRAC=0.72 PD_MEM_FRAC=0.72
LOG_LEVEL=warning
AGG_EXTRA_ARGS="--cuda-graph-max-bs 24 --max-total-tokens 1000000"
PD_EXTRA_ARGS="--cuda-graph-max-bs 24 --max-total-tokens 1000000"
--agg-gpus 1,2,3,4 --pd-gpus 1,2,3,4 --enc-gpus-3e1pd 5,6,7
```

AGG result:

```text
Successful requests: 12
Total images: 1152
Total generated tokens: 790
Request throughput: 0.08 req/s
Input token throughput: 15303.07 tok/s
Output token throughput: 5.14 tok/s
Mean TTFT: 32960.34 ms
Mean E2E: 37717.27 ms
Mean TPOT: 91.81 ms
Peak concurrent requests: 5
```

3E1PD result:

```text
Successful requests: 12
Total images: 1152
Total generated tokens: 790
Request throughput: 0.06 req/s
Input token throughput: 11795.35 tok/s
Output token throughput: 3.96 tok/s
Mean TTFT: 47518.22 ms
Mean E2E: 49181.16 ms
Mean TPOT: 28.81 ms
Peak concurrent requests: 5
```

Analysis:

- Both cases completed; no repair was needed.
- E/PD still loses:

```text
TP4 AGG image96 out128:    0.08 req/s, mean TTFT 33.0s, mean E2E 37.7s
TP4 3E1PD image96 out128:  0.06 req/s, mean TTFT 47.5s, mean E2E 49.2s
```

- E/PD does have lower TPOT (`28.8 ms` vs `91.8 ms`), but the TTFT gap is too
  large and AGG still has higher aggregate output throughput.
- Next action: try the same longer-output idea on the TP1 resource-allocation
  shape. TP1 aggregation has less local vision/decode capacity, while TP1 PD can
  still offload vision to three encoders.

### Try: Qwen3-VL-2B TP1 resource allocation, 96 images, longer outputs

Run root:

```text
results/native_sglang_matrix_20260623_qwen2b_tp1_img96_out128_r033_mc3_agg3e
```

Reason for this try:

- TP1 short-output E/PD lost, but TP1 AGG had much worse TPOT than E/PD.
- This rerun tested whether `OUTPUT_LEN=128` lets the decode-side E/PD advantage
  offset worse TTFT.

Common workload and knobs:

```bash
MODEL=/mnt/weka/data/llm-d-models-pv/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/main
SERVED=Qwen/Qwen3-VL-2B-Instruct
NUM_PROMPTS=12 IMAGE_COUNT=96 IMAGE_RES=1080p RATE=0.33
INPUT_LEN=128 OUTPUT_LEN=128
CHUNKED=393216 MAX_PREFILL_TOKENS=393216
MAX_RUNNING=24 PREFILL_MAX=2 MAX_CONCURRENCY=3
MEM_FRAC=0.72 AGG_MEM_FRAC=0.72 PD_MEM_FRAC=0.72
LOG_LEVEL=warning
AGG_EXTRA_ARGS="--cuda-graph-max-bs 24 --max-total-tokens 1000000"
PD_EXTRA_ARGS="--cuda-graph-max-bs 24 --max-total-tokens 1000000"
--agg-gpus 1 --pd-gpus 1 --enc-gpus-3e1pd 5,6,7
```

AGG result:

```text
Successful requests: 12
Total images: 1152
Total generated tokens: 790
Request throughput: 0.07 req/s
Input token throughput: 13203.65 tok/s
Output token throughput: 4.43 tok/s
Mean TTFT: 35186.52 ms
Mean E2E: 43806.85 ms
Mean TPOT: 155.91 ms
Peak concurrent requests: 5
```

3E1PD result:

```text
Successful requests: 12
Total images: 1152
Total generated tokens: 790
Request throughput: 0.06 req/s
Input token throughput: 10989.17 tok/s
Output token throughput: 3.69 tok/s
Mean TTFT: 47686.89 ms
Mean E2E: 52801.37 ms
Mean TPOT: 91.80 ms
Peak concurrent requests: 5
```

Analysis:

- Both cases completed; no repair was needed.
- E/PD still loses:

```text
TP1 AGG image96 out128:    0.07 req/s, mean TTFT 35.2s, mean E2E 43.8s
TP1 3E1PD image96 out128:  0.06 req/s, mean TTFT 47.7s, mean E2E 52.8s
```

- Longer outputs reduce the relative E2E gap, but AGG still has higher output
  throughput and much better TTFT.
- New hypothesis: the 3E setup parallelizes across requests, not within a large
  96-image request. At `MAX_CONCURRENCY=3`, each TP1 encoder handles a huge
  per-request vision batch on one GPU, while AGG encodes that same request with
  TP4. Test a grouped tensor-parallel encoder on GPUs `5,6,7` (`ENC_TP=3`) so
  E/PD can parallelize a single image-heavy request's vision work.

### Harness Update: grouped and ordered encoder startup

Changes made to `native_sglang.sh`:

- Added `ENC_GROUPS`, a semicolon-separated list of CUDA_VISIBLE_DEVICES groups
  for encoders.
- Added `ENC_TP` and `ENC_TPS`, so grouped encoders can run with tensor
  parallelism, e.g. `ENC_GROUPS="5,6;7" ENC_TPS="2;1"`.
- Changed E/PD startup order to wait for encoder health before launching PD.

Reason:

- A grouped encoder can take longer to initialize than the old TP1 encoders.
  Launching PD before the encoder was ready caused PD warmup to call the
  encoder before it was listening, then time out waiting for image embeddings.
- This directly addresses the user's failure-handling instruction: the grouped
  encoder failure was diagnosed and the same case was rerun with a targeted
  harness fix.

Validation:

```bash
bash -n native_sglang.sh
```

### Try: Qwen3-VL-2B grouped encoder attempts, 96 images

Baseline AGG from the grouped-encoder run:

```text
results/native_sglang_matrix_20260623_qwen2b_tp4_img96_r033_mc3_agg_1etp3/r0.33/1AGG
Request throughput: 0.08 req/s
Mean TTFT: 31600.15 ms
Mean E2E: 36293.22 ms
```

Failed attempt: one TP3 encoder on GPUs `5,6,7`.

```text
ENC_GROUPS="5,6,7" ENC_TP=3
```

Failure:

```text
AssertionError: 16 is not divisible by 3
```

Analysis:

- Qwen3-VL 2B vision attention has 16 heads, so encoder TP must divide 16.
- TP3 is not a valid encoder tensor-parallel size for this model.

Repair attempt 1: one TP2 encoder on GPUs `5,6`.

```text
results/native_sglang_matrix_20260623_qwen2b_tp4_img96_r033_mc3_1etp2_repair2_waitenc
ENC_GROUPS="5,6" ENC_TP=2
```

Result:

```text
Successful requests: 12
Request throughput: 0.05 req/s
Input token throughput: 10300.71 tok/s
Mean TTFT: 47704.63 ms
Mean E2E: 56493.42 ms
```

Analysis:

- The encoder-before-PD startup fix repaired the previous connection-timeout
  failure; this run reached benchmark completion.
- One TP2 encoder is worse than three TP1 encoders for this request stream,
  likely because it reduces request-level encoder parallelism too much.

Repair attempt 2: mixed TP2 + TP1 encoders.

```text
results/native_sglang_matrix_20260623_qwen2b_tp4_img96_r033_mc3_2e_mixed_tp2_tp1
ENC_GROUPS="5,6;7" ENC_TPS="2;1"
```

Result:

```text
Successful requests: 12
Request throughput: 0.06 req/s
Input token throughput: 11472.31 tok/s
Mean TTFT: 47743.71 ms
Mean E2E: 50591.84 ms
```

Analysis:

- Mixed TP2+TP1 is better than a single TP2 encoder but still worse than the
  standard three TP1 encoders (`0.06 req/s`, mean E2E `51.8s`) and worse than
  AGG (`0.08 req/s`, mean E2E `36.3s`).
- Grouped encoders are not the winning direction for this Qwen3-VL-2B 96-image
  workload under the available GPUs.
- Next axis: lower image resolution with more request concurrency. This should
  reduce embedding handoff size while preserving enough image work and request
  parallelism for three TP1 encoders to matter.

### Try: Qwen3-VL-2B TP4, 32 images at 720p

Run roots:

```text
results/native_sglang_matrix_20260623_qwen2b_tp4_img32_720p_r2_mc16_agg3e
results/native_sglang_matrix_20260623_qwen2b_tp4_img32_720p_r2_mc16_3e_repair_ports3984
```

Reason for this try:

- Lower image resolution should reduce embedding handoff size.
- `IMAGE_COUNT=32`, `NUM_PROMPTS=64`, and `MAX_CONCURRENCY=16` keep enough
  request-level parallelism for three TP1 encoders to be useful.

Common workload and knobs:

```bash
MODEL=/mnt/weka/data/llm-d-models-pv/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/main
SERVED=Qwen/Qwen3-VL-2B-Instruct
NUM_PROMPTS=64 IMAGE_COUNT=32 IMAGE_RES=720p RATE=2.0
INPUT_LEN=128 OUTPUT_LEN=16
CHUNKED=131072 MAX_PREFILL_TOKENS=131072
MAX_RUNNING=64 PREFILL_MAX=2 MAX_CONCURRENCY=16
MEM_FRAC=0.80 AGG_MEM_FRAC=0.80 PD_MEM_FRAC=0.80
LOG_LEVEL=warning
AGG_EXTRA_ARGS="--cuda-graph-max-bs 64 --max-total-tokens 1000000"
PD_EXTRA_ARGS="--cuda-graph-max-bs 64 --max-total-tokens 1000000"
```

Failure/repair:

- The first 3E attempt failed before PD launch because encoder port `39822`
  was already busy. Encoders on `39823/39824` remained up and the harness waited
  for the failed encoder.
- This was repaired by killing the stale encoder processes and rerunning the
  same 3E case on a fresh port block (`39840/39842-39844`).

AGG result:

```text
Successful requests: 64
Total images: 2048
Total input vision tokens: 1806336
Request throughput: 0.75 req/s
Input token throughput: 21159.23 tok/s
Mean TTFT: 14785.94 ms
Mean E2E: 19522.25 ms
Peak concurrent requests: 27
```

3E1PD repaired result:

```text
Successful requests: 64
Total images: 2048
Total input vision tokens: 1806336
Request throughput: 0.37 req/s
Input token throughput: 10358.14 tok/s
Mean TTFT: 34392.87 ms
Mean E2E: 41068.69 ms
Peak concurrent requests: 30
```

Analysis:

- Lowering resolution helped AGG more than E/PD:

```text
AGG image32 720p:    0.75 req/s, mean TTFT 14.8s, mean E2E 19.5s
3E1PD image32 720p:  0.37 req/s, mean TTFT 34.4s, mean E2E 41.1s
```

- 720p reduces per-image token count, but the E/PD handoff/readiness cost is
  still large and the AGG path remains much stronger.
- Next axis: test whether encoder count parity changes the result. With only
  three TP1 encoders, E/PD has fewer encoder GPUs than TP4 aggregation has
  vision-parallel ranks. GPU0 is currently free, so a bounded 4E test can show
  whether the remaining gap is mostly encoder count or the workflow itself.

### Try: Qwen3-VL-2B TP4, 64 images, 4E using GPU0

Run root:

```text
results/native_sglang_matrix_20260623_qwen2b_tp4_img64_r05_mc4_4e_gpu0
```

Reason for this try:

- 3E remained behind AGG across image counts.
- This bounded test used GPU0 as a fourth encoder while it was free, to check
  whether E/PD was mainly losing because it had fewer encoder GPUs than TP4 AGG
  has vision-parallel ranks.

Workload and knobs:

```bash
MODEL=/mnt/weka/data/llm-d-models-pv/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/main
SERVED=Qwen/Qwen3-VL-2B-Instruct
NUM_PROMPTS=16 IMAGE_COUNT=64 IMAGE_RES=1080p RATE=0.5
INPUT_LEN=128 OUTPUT_LEN=16
CHUNKED=262144 MAX_PREFILL_TOKENS=262144
MAX_RUNNING=32 PREFILL_MAX=2 MAX_CONCURRENCY=4
MEM_FRAC=0.75 PD_MEM_FRAC=0.75
LOG_LEVEL=warning
PD_EXTRA_ARGS="--cuda-graph-max-bs 32 --max-total-tokens 1000000"
--pd-gpus 1,2,3,4 --enc-gpus-4e1pd 0,5,6,7
```

4E1PD result:

```text
Successful requests: 16
Total images: 1024
Request throughput: 0.09 req/s
Input token throughput: 11161.46 tok/s
Mean TTFT: 43518.40 ms
Mean E2E: 45496.83 ms
Peak concurrent requests: 6
```

Analysis:

- 4E completed; no repair was needed.
- It did not improve over the earlier 3E image64 result and still loses to AGG:

```text
AGG image64:    0.13 req/s, mean TTFT 24.6s, mean E2E 29.0s
3E1PD image64:  0.09 req/s, mean TTFT 41.9s, mean E2E 43.9s
4E1PD image64:  0.09 req/s, mean TTFT 43.5s, mean E2E 45.5s
```

- The remaining gap is not simply encoder count. The PD scheduling/handoff path
  remains the dominant issue.
- Next axis: scheduler tuning, starting with `--enable-mixed-chunk` on the PD
  side for a queued image-heavy case.

### Try: Qwen3-VL-2B TP4, 32 images, PD mixed chunk

Run root:

```text
results/native_sglang_matrix_20260623_qwen2b_tp4_img32_r1_mc8_3e_mixedchunk
```

Reason for this try:

- Baseline 3E was still behind AGG on the 32-image queued workload.
- `--enable-mixed-chunk` might allow the PD scheduler to make better progress
  while image-heavy prefills and decode work are interleaved.

Workload and knobs:

```bash
MODEL=/mnt/weka/data/llm-d-models-pv/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/main
SERVED=Qwen/Qwen3-VL-2B-Instruct
NUM_PROMPTS=32 IMAGE_COUNT=32 IMAGE_RES=1080p RATE=1.0
INPUT_LEN=128 OUTPUT_LEN=16
CHUNKED=131072 MAX_PREFILL_TOKENS=131072
MAX_RUNNING=64 PREFILL_MAX=2 MAX_CONCURRENCY=8
MEM_FRAC=0.80 AGG_MEM_FRAC=0.80 PD_MEM_FRAC=0.80
PD_EXTRA_ARGS="--cuda-graph-max-bs 64 --max-total-tokens 1000000 --enable-mixed-chunk"
--pd-gpus 1,2,3,4 --enc-gpus-3e1pd 5,6,7
```

3E1PD mixed-chunk result:

```text
Successful requests: 32
Total images: 1024
Request throughput: 0.17 req/s
Input token throughput: 10909.42 tok/s
Mean TTFT: 38281.70 ms
Mean E2E: 44744.43 ms
Mean TPOT: 1976.07 ms
Peak concurrent requests: 12
```

Comparison:

```text
AGG image32 baseline:          0.29 req/s, mean TTFT 20.9s, mean E2E 25.4s
3E1PD image32 baseline:        0.19 req/s, mean TTFT 34.9s, mean E2E 39.8s
3E1PD image32 mixed chunk:     0.17 req/s, mean TTFT 38.3s, mean E2E 44.7s
```

Analysis:

- The run completed; no repair was needed.
- Mixed chunk worsened both throughput and latency relative to the earlier
  3E1PD baseline.
- For this Qwen3-VL-2B workload, PD-side scheduling tweaks are not enough to
  overcome embedding handoff and E/PD orchestration overhead.
- Next axis: switch to another supported VL model with a different
  vision/language balance. `OpenGVLab/InternVL3_5-8B` is present locally and
  SGLang has an InternVL model implementation, so it is the next candidate.

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
