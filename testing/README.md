# E/PD benchmark harness

Orchestrates SGLang multimodal serving benchmarks across three topologies on the
dell07 L40S host (+ the remote B70 XPU host for case 3):

1. **agg** — N aggregated EPD workers on GPU (encode+prefill+decode in one process).
2. **epd_gpu** — E encode workers on GPU + PD prefill/decode workers on GPU.
3. **epd_xpu** — E encode workers on the **B70 XPU** + PD workers on GPU.

All three use **one shared control plane** (NATS/etcd/frontend); N worker
instances register into it and the kv-router load-balances. The benchmark client
hits the single frontend at `:7001`.

## Where things run

- **Run the harness INSIDE the dell07 GPU container** (`robin_sglang_dynamo_l40`):
  GPU workers are child processes there, so teardown is `pkill`.
- **XPU encoders (case 3)** are launched on `sc09giga01-b70` (172.26.46.180) by
  SSHing and `docker run`-ing a fresh B70 container per test, then `docker exec`
  one encoder per XPU. Torn down with `docker rm -f`.

```
  [ dell07 GPU container — orchestrator here ]
    NATS + etcd + frontend(:7001)
    + agg/PD/encode workers (GPUs from --gpus)
              │  (case 3 only) ssh + docker
              ▼
  [ giga01-b70 ] fresh container: encode workers (XPUs from --xpus)
```

## Layout

```
testing/
  lib/
    start_controlplane.sh   # NATS + etcd + frontend
    add_worker.sh           # ONE local CUDA worker: role = agg|pd|encode  (source of truth)
    add_encoder_xpu.sh      # ONE-or-more XPU encoders via ssh+docker on the B70 host
    teardown.sh             # kill local workers+plane (pkill) + rm remote container
  bench_lib.py              # model table, device idle-checks, placement, readiness, bench run
  orchestrator.py           # run ONE test end to end (setup -> ready(retry once) -> bench -> teardown)
  run_matrix.py             # OPT-IN sweep over a parameter matrix (resumable)
  results/                  # gitignored: per-round results
```

## Models (TP from a table; instances × TP must fit the device list)

| served name | precision | TP (GPUs/instance) |
|---|---|---|
| `Qwen/Qwen3-VL-8B-Instruct` | bf16 | 1 |
| `Qwen/Qwen3-VL-32B-Instruct-FP8` | fp8 | 1 |
| `Qwen/Qwen3-VL-32B-Instruct` | bf16 | 2 |
| `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8` | fp8 | 6 |

If `instances × TP` exceeds the idle devices in your `--gpus` list, that test is
**skipped and logged** (never silently shrunk). Encoders are always TP1.

## Prerequisites

- Run inside the dell07 GPU container (has `dynamo.sglang`, `sglang.bench_serving`).
- For **case 3** (XPU encoders): `sshpass` installed in the container, and the
  SSH password exported (never committed):
  ```bash
  apt-get update && apt-get install -y sshpass   # if missing
  export XPU_SSH_PASS='********'                 # password for h-zheng@giga01-b70
  ```

## Run ONE test (orchestrator.py)

```bash
# aggregation, 4 instances, 8B, rate 1.0, on GPUs 0-5
python3 orchestrator.py --mode agg --model Qwen/Qwen3-VL-8B-Instruct \
    --agg-instances 4 --gpus 0,1,2,3,4,5 \
    --num-prompts 32 --image-count 8 --image-resolution 1080p \
    --request-rate 1.0 --out-dir results/manual_4agg

# disagg E-on-GPU + PD-on-GPU: 2 encoders + 2 PD
python3 orchestrator.py --mode epd_gpu --model Qwen/Qwen3-VL-8B-Instruct \
    --e-instances 2 --pd-instances 2 --gpus 0,1,2,3,4,5 \
    --request-rate 1.0 --out-dir results/manual_2e2pd

# disagg E-on-XPU + PD-on-GPU: 1 XPU encoder + 4 GPU PD
export XPU_SSH_PASS=...   # required
python3 orchestrator.py --mode epd_xpu --model Qwen/Qwen3-VL-8B-Instruct \
    --e-instances 1 --pd-instances 4 --gpus 1,2,3,4 --xpus 0 \
    --request-rate 1.0 --out-dir results/manual_1expu_4pd
```

Exit codes: `0` ok, `2` skipped (placement or never-ready after one retry),
`3` bench failed. Each run writes `bench_<label>_r<rate>.json` and
`result_<label>_r<rate>.txt` (the parsed `==== Serving Benchmark Result ====`
block, header rewritten to the topology label e.g. `1E4PD`).

## Run a SWEEP (run_matrix.py) — opt-in

**Nothing runs unless you enable a case.** Each enabled case sweeps the full
cross-product of instance counts × rates. **Resumable** — a test whose result
JSON already exists is skipped, so you can Ctrl-C and rerun.

```bash
# case 1 only: agg ∈ {1,2,4}, rates 0.2..1.0
python3 run_matrix.py --model Qwen/Qwen3-VL-8B-Instruct \
    --gpus 0,1,2,3,4,5 --rates 0.2,0.4,0.6,0.8,1.0 \
    --case1-agg 1,2,4

# case 2 + 3: E×PD grids, full rate range 0.2..2.0
export XPU_SSH_PASS=...
python3 run_matrix.py --model Qwen/Qwen3-VL-8B-Instruct \
    --gpus 0,1,2,3,4,5 --xpus 0,1,2,3 \
    --case2-epd-gpu "E=1,2,4;PD=1,2,4" \
    --case3-epd-xpu "E=1,2,4;PD=1,2,4"

# preview the plan without running
python3 run_matrix.py --model Qwen/Qwen3-VL-8B-Instruct --gpus 0,1,2,3,4,5 \
    --case1-agg 1,2,4 --dry-run
```

Flags: `--case1-agg "1,2,4"`, `--case2-epd-gpu "E=...;PD=..."`,
`--case3-epd-xpu "E=...;PD=..."` (absent ⇒ disabled), `--rates`,
`--num-prompts`, `--image-count`, `--image-resolution`, `--gpus`, `--xpus`,
`--results-root`, `--mm-attn-backend`, `--dry-run`, `--max-tests N`.

## Results layout

```
results/<timestamp>_<model>/
  summary.csv                         # one row per test: case,label,rate,status,out_dir
  case1_agg/4AGG_r1.0/
    orchestrator.log                 # per-test narration: device pick, readiness, skip/fail reason, bench
    bench_4AGG_r1.0.json              # raw sglang.bench_serving JSON
    result_4AGG_r1.0.txt             # parsed "==== Serving Benchmark Result: 4AGG ===="
    logs/                            # all worker + control-plane logs for THIS test:
      nats.log  etcd.log  frontend.log
      worker_<role>_gpu<n>.log       # dynamo.sglang GPU workers (agg / pd / encode)
      encode_xpu_<n>.log             # dynamo.sglang XPU encoders (case 3) — written
                                     #   here directly via the shared /home NFS mount
  case2_epd_gpu/1E4PD_r1.0/ ...
  case3_epd_xpu/1E4PD_r1.0/ ...
```

All logs for a test — control plane, GPU workers, **and** the remote B70 XPU
encoders — land in that test's `logs/`. `/home` is a shared Weka NFS mount, so
the B70 encoder container (which mounts `-v /home:/home`) writes its
`dynamo.sglang` logs straight into the same directory; nothing is copied back.
The orchestrator's own narration is in `orchestrator.log` next to the results.

## Device safety (hard-enforced)

The harness **only ever places workers on devices in `--gpus` / `--xpus`**, and
checks each is idle (`nvidia-smi` locally, `xpu-smi` over SSH) before use. A
device not in your list is never touched. If not enough listed devices are idle
for `instances × TP`, the test is skipped and recorded as `skipped` in the CSV.
