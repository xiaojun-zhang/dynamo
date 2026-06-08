# Disagg PD side — dell07 (4× L40S)

This directory holds the launcher for the **prefill+decode (PD) side** of a
cross-host disaggregated multimodal deployment:

- **PD workers** (this host, dell07): 4× L40S, one PD worker per GPU.
- **Encode worker** (remote): 1 encoder on the Intel B70 XPU host
  (`sc09giga01-b70.sc.intel.com`) — see `../disagg_xpu_docker/`.
- **Model:** `Qwen/Qwen3-VL-8B-Instruct`
- **Transport:** NIXL over RoCE (RDMA); embeddings produced on the B70 are read
  by the PD workers (`nixl-read`).

```
   [ B70 XPU host ]                         [ dell07 — this host ]
   1 encode worker   --- NATS/etcd (ctrl) -->  NATS + etcd + frontend(:7001)
   (vision tower)    --- NIXL/RoCE (data) -->  4 PD workers (GPUs 1-4)
```

The control plane (NATS/etcd/frontend) lives **here** on dell07. The encoder
dials back to this host, so **start this side first**.

---

## Host facts (dell07 = sc09dell07-rtx)

| Item | Value |
|---|---|
| Mgmt IP (control plane) | `172.26.46.178` |
| RoCE IP (NIXL data plane) | `192.165.123.65` (NIC `mlx5_0` / `eno17295np0`) |
| PD GPUs | L40S 1, 2, 3, 4 (avoid GPU 6 = HW error, GPU 7 = busy) |
| Frontend HTTP | `:7001` |
| NATS / etcd | `:14222` / `:12379` |
| Model path | `/mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-8B-Instruct` |

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
  -w /robin/dynamo/disagg_cuda_docker \
  amr-registry.caas.intel.com/taas/scalable-deploy-intel/main_dockerfile.dynamo_gpu:422-9e23364
```

Run as the `h-zheng` user (not `sudo`) so `~` expands correctly.

## 2. Start the PD side (inside the container)

```bash
docker exec -it robin_sglang_dynamo_l40 bash
cd /robin/dynamo/disagg_cuda_docker
./start_sglang_pd_cuda_8b_dell07.sh
```

Starts NATS + etcd + frontend, then 4 PD workers (GPUs 1-4). Waits until all 4
`backend/generate` endpoints register, then exits 0. Logs in `./logs/`.

## 3. Start the encoder on the B70 host

See `../disagg_xpu_docker/README.md`. **Do this after the PD side is up.**

## 4. Verify the pipeline

```bash
curl -s http://127.0.0.1:7001/v1/models      # should list Qwen3-VL-8B-Instruct
curl -s http://127.0.0.1:7001/health
```

---

## 5. Benchmark

Run **from dell07** (the frontend lives here), with the encoder already up.

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
    --output-file bench_8b_8img_1080p_r1.json
```

Key knobs: `--image-count` × `--image-resolution` (preset `4k/1080p/720p/360p`
or `HxW`), `--num-prompts`, `--request-rate` (sweep 0.5/1/2 to find saturation).

Read the result:
```bash
grep -E "Successful requests|Request throughput|Median TTFT|Median TPOT|Total token throughput" \
  bench_8b_8img_1080p_r1.json
```

> The encoder **must** be running before benchmarking image workloads — every
> request needs the B70 to produce embeddings. Port is **7001**, not 8000.

---

## Teardown

```bash
pkill -9 -f "dynamo.sglang"     # PD workers
pkill -f "dynamo.frontend"; pkill -f nats-server; pkill -f etcd
```
Or just `exit` the container (`--rm` removes it; logs persist on the host via the mount).
