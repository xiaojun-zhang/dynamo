# Aggregated EPD — dell07 (4× L40S)

This directory holds the launcher for an **aggregated** multimodal deployment:
**no disaggregation**. Each worker runs the full **Encode + Prefill + Decode**
pipeline in one process (vision tower + LLM together), on its own GPU.

- **Workers** (this host, dell07): 4× L40S, one full EPD worker per GPU.
- **No remote encoder, no NIXL transfer** — everything happens in-process.
- **Model:** `Qwen/Qwen3-VL-8B-Instruct`
- The frontend kv-router load-balances incoming requests across the 4 workers.

```
   [ dell07 — this host ]
   NATS + etcd + frontend(:7001)
        |
        +--> agg EPD worker (GPU 1)   encode+prefill+decode
        +--> agg EPD worker (GPU 2)   encode+prefill+decode
        +--> agg EPD worker (GPU 3)   encode+prefill+decode
        +--> agg EPD worker (GPU 4)   encode+prefill+decode
```

Contrast with `../disagg_cuda_docker/` (PD workers here + a remote B70 encoder
over NIXL). Aggregated keeps the vision encoder on the same GPU as the LLM, so
there is **no cross-host encoder bottleneck** — useful as a baseline and when no
XPU encoder host is available.

---

## Host facts (dell07 = sc09dell07-rtx)

| Item | Value |
|---|---|
| Mgmt IP (control plane) | `172.26.46.178` |
| Worker GPUs | L40S 1, 2, 3, 4 (avoid GPU 6 = HW error, GPU 7 = busy) |
| Frontend HTTP | `:7001` |
| NATS / etcd | `:14222` / `:12379` |
| Per-worker ports | sys `8081+gpu`, KV-events `22080+gpu` |
| Model path | `/mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-8B-Instruct` |
| max-running / mem-fraction | `40` / `0.85` (tunable in the script) |

---

## 1. Launch the container

`robin` is a separate top-level dir, so it **must** be mounted (`-v ~/robin:/robin`)
or the script won't be visible inside the container.

```bash
docker run -dit --rm \
  --privileged \
  --gpus all \
  --network=host \
  --ipc=host \
  $(env | grep -i _proxy | sed 's/^/-e /') \
  --user root \
  --group-add video \
  --cap-add=SYS_ADMIN \
  --mount type=bind,source=/dev/dri/by-path,target=/dev/dri/by-path \
  --mount type=bind,source=/sys,target=/sys \
  --mount type=bind,source=/dev/bus,target=/dev/bus \
  --mount type=bind,source=/dev/char,target=/dev/char \
  --mount type=bind,source=/dev/infiniband,target=/dev/infiniband \
  -v ~/hongming:/hongming \
  -v ~/robin:/robin \
  -v /mnt/weka:/mnt/weka \
  --name robin_sglang_dynamo_l40 \
  --entrypoint /bin/bash \
  -w /robin/dynamo/agg_cuda_docker \
  amr-registry.caas.intel.com/taas/scalable-deploy-intel/main_dockerfile.dynamo_gpu:422-9e23364
```

Run as the `h-zheng` user (not `sudo`) so `~` expands correctly. If a container
named `robin_sglang_dynamo_l40` already exists (e.g. from the disagg setup),
`docker rm -f` it first or pick another `--name`.

## 2. Start the aggregated stack (inside the container)

```bash
docker exec -it robin_sglang_dynamo_l40 bash
cd /robin/dynamo/agg_cuda_docker
./start_sglang_agg_cuda_8b_dell07.sh
```

Starts NATS + etcd + frontend, then 4 aggregated EPD workers (GPUs 1-4). Waits
until all 4 `backend/generate` endpoints register, then exits 0. Logs in
`./logs/` (`nats_dell07.log`, `etcd_dell07.log`, `frontend_dell07.log`,
`agg_worker_gpu<N>.log`).

Common overrides (env vars):
```bash
AGG_GPUS="1 2"  ./start_sglang_agg_cuda_8b_dell07.sh   # fewer workers
MAX_RUNNING=64  ./start_sglang_agg_cuda_8b_dell07.sh   # raise concurrency
MEM_FRAC=0.80   ./start_sglang_agg_cuda_8b_dell07.sh   # lower if a worker OOMs
```

**No B70 encoder is needed for this mode** — aggregated workers do their own
vision encoding.

## 3. Verify the stack

```bash
curl -s http://127.0.0.1:7001/v1/models      # should list Qwen3-VL-8B-Instruct
curl -s http://127.0.0.1:7001/health         # should show 4 backend/generate
```

---

## 4. Benchmark

Run **from dell07** (the frontend lives here). No encoder dependency.

```bash
python3 -m sglang.bench_serving \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --backend sglang-oai-chat \
    --host 127.0.0.1 \
    --port 7001 \
    --dataset-name image \
    --num-prompts 32 \
    --random-input-len 128 \
    --random-output-len 256 \
    --image-count 8 \
    --image-resolution 1080p \
    --request-rate 1 \
    --apply-chat-template \
    --seed 0 \
    --output-file bench_agg_8b_8img_1080p_r1.json
```

Key knobs: `--image-count` × `--image-resolution` (preset `4k/1080p/720p/360p`
or `HxW`), `--num-prompts`, `--request-rate` (sweep 0.5/1/2 to find saturation).

Read the result:
```bash
grep -E "Successful requests|Request throughput|Median TTFT|Median TPOT|Total token throughput" \
  bench_agg_8b_8img_1080p_r1.json
```

> Aggregated runs the vision encoder on the same (fast) L40S GPU as the LLM,
> so it does not hit the cross-host B70 encoder wall the disagg path does — it's
> the natural throughput/latency baseline to compare disagg against.

---

## Teardown

```bash
pkill -9 -f "dynamo.sglang"     # agg workers
pkill -f "dynamo.frontend"; pkill -f nats-server; pkill -f etcd
```
Or just `exit` the container (`--rm` removes it; logs persist on the host via the mount).

---

## Notes

- **Aggregated vs disagg flags:** this worker omits `--multimodal-worker` /
  `--encoder-only` and all NIXL transfer env/flags; it keeps `--enable-multimodal`
  + `--chat-template qwen2-vl` (the agg worker does its own vision encode).
- **Discovery:** etcd + nats (same control plane as the disagg scripts) — so the
  frontend and all 4 workers agree on how to find each other.
- **Per-worker ports** auto-increment by GPU index, so the 4 workers don't
  collide on this host.
- Uses the **same `:7001` frontend** as the disagg setup, so don't run both
  stacks at once on this host (port conflict) — tear one down first.
