# SGLang + Dynamo InternVL3.5 PD Disaggregation Reproduction Notes

This file captures the setup used for the successful InternVL3.5-30B matched
`1AGG` versus `2E1PD` rate sweep. It is intended as a handoff for a future code
agent or operator to reproduce the tests without relying on chat history.

## Result To Reproduce

Model:

```text
/mnt/weka/data/llm-d-models-pv/hub/models--OpenGVLab--InternVL3_5-30B-A3B/snapshots/main
```

Final matched workload:

```text
topologies:       1AGG and 2E1PD
rates:            0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0
num-prompts:      128
image-count:      8
image-resolution: 1080p
random-input-len: 128
random-output-len: 16
max-concurrency:  8
seed:             0
backend:          sglang-oai-chat
dataset:          image
```

Final curated artifact folder:

```text
/home/h-zheng/robin/dynamo/testing/results/dynamo_sglang_internvl35_30b_a3b_h200_gpu6_b70_xpu01_np128_img8_1080p_out16_mc8_matched_r0.2_2.0_final
```

Final committed zip:

```text
/home/h-zheng/robin/dynamo/testing/results/dynamo_sglang_internvl35_30b_a3b_h200_gpu6_b70_xpu01_np128_img8_1080p_out16_mc8_matched_r0.2_2.0_final.zip
```

Commit containing the harness changes and zip:

```text
eec423b6b Add InternVL 30B matched sweep artifacts
```

## Machines And Cards

GPU host:

```text
hostname: sc09super21-h200
GPU type: NVIDIA H200, 143771 MiB each
driver:   580.159.03
cards:    0..7 available on host
used:     card 6 for 1AGG and PD
mgmt IP:  172.26.46.130
RoCE IP:  192.165.123.48
UCX NIC:  mlx5_0:1
```

XPU host:

```text
hostname: sc09giga01-b70.sc.intel.com
device:   Intel(R) Graphics [0xe223], B70 server profile
cards:    0..7 available on host
used:     cards 0,1 for encoders
profile:  b70
XPU 0-3:  fabric IP 192.165.123.40, UCX NIC mlx5_0:1
XPU 4-7:  fabric IP 192.165.123.37, UCX NIC mlx5_2:1
```

The harness requires SSH key auth from the GPU container to the XPU host:

```text
ssh user: h-zheng
ssh key:  /root/.ssh/id_ed25519 inside the GPU container
ssh opts: -F /dev/null -o StrictHostKeyChecking=no -o BatchMode=yes
```

## Docker Images

GPU container:

```text
container: robin_sglang_dynamo_l40
image:     amr-registry.caas.intel.com/taas/scalable-deploy-intel/main_dockerfile.dynamo_gpu:422-9e23364
image id:  sha256:4aac4c619953801f215a4c7a8e19e1572bd34727390aada70e7ab3b2b267e4bb
workdir:   /robin/dynamo/testing
```

Important package versions observed inside the GPU container:

```text
ai-dynamo         1.0.0
ai-dynamo-runtime 1.0.0
sglang            0.5.12.dev315+g91907b7b9
torch             2.11.0
transformers      5.6.0
```

XPU encoder container:

```text
container name: harness_b70_enc
image tags:     hm_dynamo_b70:latest, hm_dynamo_b70_pr26460:latest
image id:       sha256:631d255392fe317e31712a33a81b607ec7caa79fc7b5c6900f0bb2f842c41fe8
created:        2026-05-29T22:26:01.207466089Z
```

The XPU container is created fresh for each E/PD run by
`testing/lib/add_encoder_xpu.sh` using `docker run --network=host --privileged`
on `sc09giga01-b70.sc.intel.com`. It mounts:

```text
/mnt/weka:/mnt/weka
/home:/home
/dev/dri, /dev/infiniband, /sys, /dev/bus, /dev/char
```

## Harness Files

Run from:

```text
/home/h-zheng/robin/dynamo/testing
```

Main files:

```text
orchestrator.py                 one test end-to-end
bench_lib.py                    model table, readiness, bench_serving wrapper
bench_patches/sitecustomize.py  bench_serving InternVL processor fallback
lib/start_controlplane.sh       NATS, etcd, Dynamo frontend
lib/add_worker.sh               local CUDA agg/pd/encode workers
lib/add_encoder_xpu.sh          remote B70 XPU encode workers
lib/teardown.sh                 cleanup
container_patch_work/           SGLang/Dynamo XPU patches copied into B70 container
```

Use the current branch/commit or newer:

```bash
cd /home/h-zheng/robin/dynamo
git checkout sglang-summit-e-pd-disaggregation-demo
git rev-parse --short HEAD
```

## Model Configuration Used By The Harness

The InternVL3.5-30B-A3B entry in `bench_lib.py` is the source of truth:

```text
tp:                    1
kv-cache dtype:        fp8_e4m3
mem-fraction-static:   0.80
chunked-prefill-size:  65536
chat-template:         internvl-2-5
max-prefill-tokens:    65536
max-total-tokens:      250000
cuda-graph-max-bs:     32
max-running-requests:  32
pd-prefill-max:        4
pd-max-running:        32
pd-mem-frac:           0.80
enc-mem-frac:          0.70
XPU_APPLY_PATCHES:     1
USE_SGLANG_TOKENIZER:  0
DYN_CHAT_PROCESSOR:    sglang
ROUTER_MODE:           round-robin
```

Additional environment used for the successful runs:

```text
XPU_HOST=sc09giga01-b70.sc.intel.com
XPU_IDLE_MAX_USED_MIB=2000
BENCH_PYTHONPATH=/robin/dynamo/testing/bench_patches
XPU_INSTALL_SGL_KERNEL_XPU=0
```

`XPU_INSTALL_SGL_KERNEL_XPU=0` matters: the final 30B sweep used the old kernel
in the XPU image. Earlier experiments tried reinstalling `sgl-kernel-xpu`, but
that was not part of the final matched run.

The XPU launcher defaults to `--mm-attention-backend xpu_attn` unless
`MM_ATTN_BACKEND` is explicitly set to an empty string. The final runs relied on
that default.

## What The Topologies Mean

`1AGG`:

```text
one local H200 worker on CUDA_VISIBLE_DEVICES=6
role: agg
full encode + prefill + decode in one process
no XPU worker
```

`2E1PD`:

```text
one local H200 PD worker on CUDA_VISIBLE_DEVICES=6
two remote B70 XPU encoder workers on ZE_AFFINITY_MASK=0 and 1
embedding transfer mode: nixl-read
request plane: tcp
event plane: nats
discovery backend: etcd
router mode: round-robin
```

The final tests used H200 card 6 for both `1AGG` and PD. Do not run another
service on that card while testing. For XPU cards 0 and 1, idle baseline memory
can be around 1215 MiB, so the harness uses `XPU_IDLE_MAX_USED_MIB=2000`.

## Exact Runtime Configuration

The key memory and scheduling settings are:

```text
chunked-prefill-size: 65536
max-prefill-tokens:  65536
max-total-tokens:    250000
cuda-graph-max-bs:   32
max-concurrency:     8
```

Changing these makes the result not directly comparable to the final sweep.

### Control Plane

The harness starts one local control plane per test:

```text
frontend HTTP port:       7011
NATS port:                14222
etcd client port:         12379
etcd peer port:           12380
router mode:              round-robin
dyn chat processor:       sglang
request plane:            tcp
frontend body limit:      DYN_HTTP_BODY_LIMIT_MB=1024
frontend TCP msg size:    DYN_TCP_MAX_MESSAGE_SIZE=1073741824
ETCD_LEASE_TTL:           600
ETCD_REQUEST_TIMEOUT:     600
```

The control plane is launched by `lib/start_controlplane.sh`. `PORT_HTTP` is
passed through the orchestrator so `dynamo.frontend` and `sglang.bench_serving`
use the same port.

### Common Worker Environment

These environment settings are shared by local H200 workers:

```text
DYN_REQUEST_PLANE=tcp
DYN_LOG=debug
TRANSFER_LOCAL=0
PYTHONHASHSEED=0
ENABLE_ENCODER_CACHE=0
DYN_TCP_MAX_MESSAGE_SIZE=268435456
DYN_HTTP_BODY_LIMIT_MB=256
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,P2P
```

### 1AGG Worker Runtime Arguments

`1AGG` uses `lib/add_worker.sh agg ...` and runs one local H200 worker:

```text
role:                     agg
CUDA_VISIBLE_DEVICES:      6
tp:                       1
mem-fraction-static:      0.80
dtype:                    auto
kv-cache-dtype:           fp8_e4m3
chat-template:            internvl-2-5
max-running-requests:     32
chunked-prefill-size:     65536
max-prefill-tokens:       65536
max-total-tokens:         250000
cuda-graph-max-bs:        32
page-size:                16
trust-remote-code:        true
enable-multimodal:        true
discovery-backend:        etcd
event-plane:              nats
log-level:                debug
```

The aggregate worker command includes:

```text
--enable-multimodal
--chat-template internvl-2-5
--dtype auto
--kv-cache-dtype fp8_e4m3
--max-running-requests 32
--chunked-prefill-size 65536
--max-prefill-tokens 65536
--max-total-tokens 250000
--cuda-graph-max-bs 32
```

### PD Worker Runtime Arguments

`2E1PD` uses one local H200 PD worker launched by `lib/add_worker.sh pd ...`:

```text
role:                         pd
CUDA_VISIBLE_DEVICES:          6
tp:                           1
mem-fraction-static:          0.80
dtype:                        auto
kv-cache-dtype:               fp8_e4m3
embedding-transfer-mode:      nixl-read
disaggregation-transfer:      nixl
prefill-max-requests:         4
max-running-requests:         32
chunked-prefill-size:         65536
max-prefill-tokens:           65536
max-total-tokens:             250000
cuda-graph-max-bs:            32
disable-radix-cache:          true
skip-tokenizer-init:          true
page-size:                    16
trust-remote-code:            true
discovery-backend:            etcd
event-plane:                  nats
log-level:                    debug
```

The PD worker command includes:

```text
--multimodal-worker
--embedding-transfer-mode nixl-read
--kv-cache-dtype fp8_e4m3
--prefill-max-requests 4
--max-running-requests 32
--chunked-prefill-size 65536
--max-prefill-tokens 65536
--max-total-tokens 250000
--cuda-graph-max-bs 32
--disable-radix-cache
--skip-tokenizer-init
--disaggregation-transfer-backend nixl
```

### XPU Encoder Runtime Arguments

`2E1PD` uses two remote B70 XPU encoder workers launched by
`lib/add_encoder_xpu.sh`:

```text
role:                         multimodal encode worker
XPU host:                     sc09giga01-b70.sc.intel.com
ZE_AFFINITY_MASK:             0 and 1
mem-fraction-static:          0.70
chat-template:                internvl-2-5
embedding-transfer-mode:      nixl-read
disaggregation-transfer:      nixl
mm-attention-backend:         xpu_attn
encoder-only:                 true
enable-multimodal:            true
skip-tokenizer-init:          true
use-sglang-tokenizer:         false
page-size:                    16
trust-remote-code:            true
log-level:                    debug
VISION_ENCODE_SERIALIZE:      1
ENABLE_ENCODER_CACHE:         0
DYN_TCP_MAX_MESSAGE_SIZE:     268435456
XPU_APPLY_PATCHES:            1
XPU_INSTALL_SGL_KERNEL_XPU:    0
```

Each XPU encoder command includes:

```text
--multimodal-encode-worker
--enable-multimodal
--encoder-only
--chat-template internvl-2-5
--embedding-transfer-mode nixl-read
--skip-tokenizer-init
--trust-remote-code
--page-size 16
--mem-fraction-static 0.70
--mm-attention-backend xpu_attn
--discovery-backend etcd
--event-plane nats
--disaggregation-transfer-backend nixl
--log-level debug
```

### XPU Transfer And Fabric Settings

For XPU cards `0,1` on the B70 host:

```text
VLLM_NIXL_SIDE_CHANNEL_HOST:  192.165.123.40
UCX_NET_DEVICES:              mlx5_0:1
UCX_TLS:                      ze_copy,rc,tcp
UCX_MEMTYPE_CACHE:            0
NATS_SERVER:                  nats://172.26.46.130:14222
ETCD_ENDPOINTS:               http://172.26.46.130:12379
ETCD_LEASE_TTL:               600
ETCD_REQUEST_TIMEOUT:         600
```

The PD side uses:

```text
VLLM_NIXL_SIDE_CHANNEL_HOST:  192.165.123.48
UCX_NET_DEVICES:              mlx5_0:1
UCX_TLS:                      cuda_ipc,ib,rc,ud,rc_verbs,ud_verbs,cuda_copy
UCX_MEMTYPE_CACHE:            0
DYN_SGL_EMBEDDING_TRANSFER_MODE: nixl-read
```

### Port Bases

Use fresh high XPU ports for each retry/run. The known successful runs used
patterns like:

```text
XPU_SYS_PORT_BASE:            8611, 8711, or generated high bases
XPU_KV_EVENT_BASE:            26190, 26290, or generated high bases
XPU_SIDE_CHANNEL_BASE:        24199, 24299, or generated high bases
local SYS_PORT_BASE:          8100
local KV_EVENT_BASE:          22100
local SIDE_CHANNEL_BASE:      20100
```

For reproducibility, prefer generated fresh XPU ports per point instead of the
low defaults. Stale remote host-network sockets were a real failure source
during testing.

### Benchmark Client Settings

The benchmark client settings are fixed except for `--request-rate`:

```text
backend:                      sglang-oai-chat
dataset-name:                 image
num-prompts:                  128
random-input-len:             128
random-output-len:            16
image-count:                  8
image-resolution:             1080p
request-rate:                 one of 0.2..2.0
max-concurrency:              8
apply-chat-template:          true
seed:                         0
disable-tqdm:                 true
BENCH_PYTHONPATH:             /robin/dynamo/testing/bench_patches
```

## Single-Rate Reproduction Command

Run this from the repo root on the H200 host. Change `RATE=1.2` as needed.

```bash
cd /home/h-zheng/robin/dynamo

RATE=1.2
RUN_TAG="manual_$(date +%Y%m%d_%H%M)"

docker exec \
  -e RATE="$RATE" \
  -e RUN_TAG="$RUN_TAG" \
  -w /robin/dynamo/testing \
  robin_sglang_dynamo_l40 bash -lc '
set -euo pipefail

export XPU_HOST=sc09giga01-b70.sc.intel.com
export XPU_IDLE_MAX_USED_MIB=2000
export ROUTER_MODE=round-robin
export BENCH_PYTHONPATH=/robin/dynamo/testing/bench_patches
export XPU_INSTALL_SGL_KERNEL_XPU=0

# Use fresh high ports for the remote B70 encoder workers.
export XPU_SYS_PORT_BASE=9011
export XPU_KV_EVENT_BASE=26700
export XPU_SIDE_CHANNEL_BASE=24600

MODEL=/mnt/weka/data/llm-d-models-pv/hub/models--OpenGVLab--InternVL3_5-30B-A3B/snapshots/main
ROOT=results/manual_30b_1agg_2e1pd_rate_${RATE}_${RUN_TAG}

python3 orchestrator.py \
  --mode agg \
  --model "$MODEL" \
  --agg-instances 1 --e-instances 0 --pd-instances 0 \
  --num-prompts 128 \
  --image-count 8 \
  --image-resolution 1080p \
  --input-len 128 \
  --output-len 16 \
  --gpus 6 \
  --xpus "" \
  --max-concurrency 8 \
  --request-rate "$RATE" \
  --out-dir "$ROOT/r${RATE}/1AGG"

python3 orchestrator.py \
  --mode epd_xpu \
  --model "$MODEL" \
  --agg-instances 0 --e-instances 2 --pd-instances 1 \
  --num-prompts 128 \
  --image-count 8 \
  --image-resolution 1080p \
  --input-len 128 \
  --output-len 16 \
  --gpus 6 \
  --xpus 0,1 \
  --max-concurrency 8 \
  --request-rate "$RATE" \
  --out-dir "$ROOT/r${RATE}/2E1PD"

find "$ROOT" -name "bench_*.json" -print | sort | while read f; do
  jq -r --arg f "$f" \
    "[\$f, .completed, .request_rate, .request_throughput, (.mean_ttft_ms/1000), (.mean_tpot_ms/1000), (.mean_e2e_latency_ms/1000)] | @tsv" "$f"
done
'
```

A valid result must have `.completed == 128` in each `bench_*.json`. If a point
returns fewer completions, keep the artifact for debugging but do not use it for
comparison. Re-run the same topology with the same workload and fresh XPU ports.

## Full Sweep Reproduction Command

This reruns the full `0.2..2.0` matched sweep. It validates each point and
retries a failed point once with the same workload and a fresh port range.

```bash
cd /home/h-zheng/robin/dynamo

docker exec -w /robin/dynamo/testing robin_sglang_dynamo_l40 bash -lc '
set -euo pipefail

export XPU_HOST=sc09giga01-b70.sc.intel.com
export XPU_IDLE_MAX_USED_MIB=2000
export ROUTER_MODE=round-robin
export BENCH_PYTHONPATH=/robin/dynamo/testing/bench_patches
export XPU_INSTALL_SGL_KERNEL_XPU=0

MODEL=/mnt/weka/data/llm-d-models-pv/hub/models--OpenGVLab--InternVL3_5-30B-A3B/snapshots/main
ROOT=results/manual_30b_1agg_2e1pd_matched_rate_sweep_$(date +%Y%m%d_%H%M)
mkdir -p "$ROOT"
printf "case,label,rate,attempt,status,completed,out_dir\n" > "$ROOT/summary.csv"

trap "echo interrupted; GPUS=6 bash lib/teardown.sh >/tmp/teardown_30b_manual.log 2>&1 || true; exit 130" INT TERM

run_one() {
  local mode="$1" label="$2" rate="$3" base_out="$4" idx="$5"
  shift 5
  local attempt out rc completed status json

  for attempt in 1 2; do
    if [ "$attempt" = "1" ]; then
      out="$base_out"
    else
      out="${base_out}_attempt${attempt}"
    fi
    mkdir -p "$out"

    # Fresh high ports avoid stale remote host-network sockets.
    export XPU_SYS_PORT_BASE=$((9011 + idx * 20 + attempt * 3))
    export XPU_KV_EVENT_BASE=$((26700 + idx * 40 + attempt * 6))
    export XPU_SIDE_CHANNEL_BASE=$((24600 + idx * 20 + attempt * 3))

    echo "[$(date +%H:%M:%S)] RUN rate=$rate $label attempt=$attempt ports=${XPU_SYS_PORT_BASE}/${XPU_KV_EVENT_BASE}/${XPU_SIDE_CHANNEL_BASE}"
    python3 orchestrator.py "$@" --request-rate "$rate" --out-dir "$out" > "$out/orchestrator.log" 2>&1
    rc=$?
    completed=NA
    status=ok
    json="$out/bench_${label}_r${rate}.json"
    if [ -s "$json" ]; then
      completed=$(jq -r ".completed // \"NA\"" "$json")
    fi
    if [ "$rc" != "0" ]; then
      status="rc=$rc"
    fi
    if [ "$completed" != "128" ]; then
      status="invalid_${status}"
    fi
    printf "%s,%s,%s,%s,%s,%s,%s\n" "$mode" "$label" "$rate" "$attempt" "$status" "$completed" "$out" >> "$ROOT/summary.csv"
    echo "[$(date +%H:%M:%S)] DONE rate=$rate $label attempt=$attempt status=$status completed=$completed"

    if [ "$status" = "ok" ]; then
      return 0
    fi
    echo "[$(date +%H:%M:%S)] RETRY rate=$rate $label"
    GPUS=6 bash lib/teardown.sh >/tmp/teardown_30b_manual_retry.log 2>&1 || true
  done

  echo "[$(date +%H:%M:%S)] STOP invalid result for $label rate=$rate"
  exit 2
}

idx=0
for r in 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0; do
  idx=$((idx + 1))
  run_one case1_agg 1AGG "$r" "$ROOT/r${r}/1AGG" "$idx" \
    --mode agg --model "$MODEL" --agg-instances 1 --e-instances 0 --pd-instances 0 \
    --num-prompts 128 --image-count 8 --image-resolution 1080p --input-len 128 --output-len 16 \
    --gpus 6 --xpus "" --max-concurrency 8

  run_one case3_epd_xpu 2E1PD "$r" "$ROOT/r${r}/2E1PD" "$idx" \
    --mode epd_xpu --model "$MODEL" --agg-instances 0 --e-instances 2 --pd-instances 1 \
    --num-prompts 128 --image-count 8 --image-resolution 1080p --input-len 128 --output-len 16 \
    --gpus 6 --xpus 0,1 --max-concurrency 8
done

echo "SWEEP_DONE $ROOT"
'
```

## Direct `sglang.bench_serving` Parameters

`orchestrator.py` ultimately runs:

```bash
python3 -m sglang.bench_serving \
  --model /mnt/weka/data/llm-d-models-pv/hub/models--OpenGVLab--InternVL3_5-30B-A3B/snapshots/main \
  --backend sglang-oai-chat \
  --host 127.0.0.1 \
  --port 7011 \
  --dataset-name image \
  --num-prompts 128 \
  --random-input-len 128 \
  --random-output-len 16 \
  --image-count 8 \
  --image-resolution 1080p \
  --request-rate <RATE> \
  --apply-chat-template \
  --seed 0 \
  --disable-tqdm \
  --output-file <OUT_JSON> \
  --max-concurrency 8
```

The `BENCH_PYTHONPATH=/robin/dynamo/testing/bench_patches` environment is
needed so `bench_serving` can build InternVL image datasets when the stock
HuggingFace processor metadata is incompatible.

## Validation And Metric Extraction

All selected results must have `completed=128`:

```bash
find results/<RUN_ROOT> -name "bench_*.json" -print | sort | while read f; do
  jq -r --arg f "$f" "[\$f, .completed, .request_rate, .request_throughput, (.mean_ttft_ms/1000), (.mean_tpot_ms/1000), (.mean_e2e_latency_ms/1000), (.p99_e2e_latency_ms/1000)] | @tsv" "$f"
done
```

Check for known E/PD failure signatures:

```bash
rg -n "Number of tokens in multimodal embedding|shape mismatch|Scheduler hit an exception|status=error|Connection refused" \
  results/<RUN_ROOT>/r*/2E1PD/logs
```

Check for leftover processes before or after a run:

```bash
docker exec -w /robin/dynamo/testing robin_sglang_dynamo_l40 bash -lc '
ps -ef | grep -E "orchestrator.py|sglang.bench_serving|dynamo.sglang|etcd|nats" | grep -v grep || true
'
```

Remote XPU container cleanup is handled by the harness, but this checks for a
stale encoder container:

```bash
docker exec -w /robin/dynamo/testing robin_sglang_dynamo_l40 bash -lc '
ssh -F /dev/null -i /root/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes \
  h-zheng@sc09giga01-b70.sc.intel.com "docker ps -a | grep harness_b70_enc || true"
'
```

## Expected Final Metrics

These are the selected valid runs from the final artifact tree. Values are in
seconds for latency columns.

| rate | case | req/s | mean TTFT | mean TPOT | mean E2E | p99 E2E |
|---:|---|---:|---:|---:|---:|---:|
| 0.2 | 1AGG | 0.197 | 6.027 | 0.087 | 6.558 | 14.243 |
| 0.2 | 2E1PD | 0.198 | 2.691 | 0.020 | 2.811 | 5.317 |
| 0.4 | 1AGG | 0.387 | 8.211 | 0.461 | 10.659 | 22.532 |
| 0.4 | 2E1PD | 0.395 | 2.835 | 0.030 | 2.993 | 5.142 |
| 0.6 | 1AGG | 0.524 | 11.185 | 0.469 | 13.720 | 19.994 |
| 0.6 | 2E1PD | 0.589 | 3.029 | 0.069 | 3.421 | 6.686 |
| 0.8 | 1AGG | 0.502 | 13.322 | 0.354 | 15.419 | 26.934 |
| 0.8 | 2E1PD | 0.779 | 3.347 | 0.117 | 4.011 | 8.112 |
| 1.0 | 1AGG | 0.511 | 12.800 | 0.477 | 15.159 | 24.193 |
| 1.0 | 2E1PD | 0.957 | 3.755 | 0.179 | 4.640 | 8.639 |
| 1.2 | 1AGG | 0.510 | 12.551 | 0.423 | 15.199 | 27.199 |
| 1.2 | 2E1PD | 1.118 | 4.101 | 0.271 | 5.567 | 10.467 |
| 1.4 | 1AGG | 0.507 | 13.136 | 0.477 | 15.342 | 24.210 |
| 1.4 | 2E1PD | 1.145 | 5.011 | 0.259 | 6.231 | 11.009 |
| 1.6 | 1AGG | 0.524 | 12.833 | 0.435 | 15.029 | 25.374 |
| 1.6 | 2E1PD | 1.217 | 4.738 | 0.290 | 6.275 | 10.476 |
| 1.8 | 1AGG | 0.493 | 14.076 | 0.322 | 15.840 | 26.958 |
| 1.8 | 2E1PD | 1.216 | 4.806 | 0.310 | 6.323 | 11.275 |
| 2.0 | 1AGG | 0.527 | 12.247 | 0.491 | 14.953 | 26.723 |
| 2.0 | 2E1PD | 1.158 | 5.057 | 0.313 | 6.729 | 11.901 |

## Known Issues And Recovery

Some high-rate attempts failed transiently during the original sweep. The final
artifact set includes only successful retries:

```text
r0.8 2E1PD: first attempt invalid, retry completed 128
r1.0 2E1PD: first attempt invalid, retry completed 128
r2.0 1AGG:  first attempt invalid, retry completed 128
```

The observed E/PD failure signature was:

```text
Number of tokens in multimodal embedding does not match those in the input text.
Got 0 tokens in the text but 18432 tokens from multimodal embeddings.
RuntimeError: shape mismatch: value tensor of shape [18432, 2048] cannot be broadcast to indexing result of shape [0, 2048]
```

Once PD crashes, encoder logs show `Connection refused`. If this happens, do not
use that JSON even if `bench_serving` exits 0. Re-run the same point with fresh
XPU port bases and require `.completed == 128`.

Default XPU ports previously caused stale host-network failures. Use high,
fresh ports:

```text
XPU_SYS_PORT_BASE
XPU_KV_EVENT_BASE
XPU_SIDE_CHANNEL_BASE
```

The control plane HTTP port defaults to `7011` from `bench_lib.py`. If a stale
frontend is using it, set `PORT_HTTP` consistently for the run; the orchestrator
passes it to both the control plane and `bench_serving`.

Manual teardown:

```bash
docker exec -w /robin/dynamo/testing robin_sglang_dynamo_l40 bash -lc '
GPUS=6 bash lib/teardown.sh || true
ssh -F /dev/null -i /root/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes \
  h-zheng@sc09giga01-b70.sc.intel.com "docker rm -f harness_b70_enc >/dev/null 2>&1 || true"
'
```

## Artifact Layout

The final curated artifact tree is normalized as:

```text
r0.2/1AGG/
r0.2/2E1PD/
r0.4/1AGG/
r0.4/2E1PD/
...
r2.0/1AGG/
r2.0/2E1PD/
```

Each case directory contains:

```text
bench_<case>_r<rate>.json
result_<case>_r<rate>.txt
orchestrator.log
logs/frontend.log
logs/worker_*.log
logs/nats.log
logs/etcd.log
logs/harness.pids
logs/xpu_launcher.log          # E/PD only
logs/encode_xpu_0.log          # E/PD only
logs/encode_xpu_1.log          # E/PD only
```

The committed zip contains this tree and is the easiest artifact to move to a
new session.
