# Disagg Encode side — Intel B70 XPU host

This directory holds the launcher for the **encode worker** of a cross-host
disaggregated multimodal deployment:

- **Encode worker** (this host, B70): 1 multimodal encoder on an Intel
  Battlemage (B70) XPU — runs the vision tower only (`--encoder-only`).
- **PD workers** (remote): 4× L40S on `dell07` (`172.26.46.178`) — see
  `../disagg_cuda_docker/`.
- **Model:** `Qwen/Qwen3-VL-8B-Instruct`
- **Transport:** NIXL over RoCE (RDMA), `ze_copy` (Level-Zero) on the XPU side,
  no `cuda_ipc` (cross-host). PD workers read embeddings (`nixl-read`).

```
   [ B70 XPU host — this host ]               [ dell07 ]
   1 encode worker  --- NATS/etcd (ctrl) -->  NATS + etcd + frontend(:7001)
   (vision tower)   --- NIXL/RoCE (data) -->  4 PD workers (GPUs 1-4)
```

The control plane lives on **dell07**, not here. **Start the dell07 PD side
first**, then run the encoder here.

---

## Host facts (sc09giga01-b70)

| Item | Value |
|---|---|
| Remote PD host (control plane) | `172.26.46.178` (dell07 mgmt) |
| NATS / etcd (on dell07) | `:14222` / `:12379` |
| XPU device | `ZE_AFFINITY_MASK` 0..7 (default 0) |
| RoCE NIC (auto by XPU NUMA) | XPU 0-3 → `mlx5_0` (192.165.123.40); XPU 4-7 → `mlx5_2` (192.165.123.37) |
| Side-channel / KV-event port | `20099` / `22090` |
| Model path | `/mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-8B-Instruct` |

---

## 1. Launch the container

`robin` must be mounted (`-v ~/robin:/robin`) or the script won't be visible.

```bash
docker run -it --rm \
  --privileged \
  --name robin_sglang_dynamo_b70 \
  --device=/dev/dri \
  --network=host \
  $(env | grep -i proxy | sed 's/^/-e /') \
  $(for dev in /dev/mei*; do echo --device $dev; done) \
  --group-add video \
  --cap-add=SYS_ADMIN \
  --mount type=bind,source=/dev/dri/by-path,target=/dev/dri/by-path \
  --mount type=bind,source=/sys,target=/sys \
  --mount type=bind,source=/dev/bus,target=/dev/bus \
  --mount type=bind,source=/dev/char,target=/dev/char \
  --mount type=bind,source=/dev/infiniband,target=/dev/infiniband \
  -v /software/:/software \
  -v ~/hongming:/hongming \
  -v ~/robin:/robin \
  -v /mnt/weka:/mnt/weka \
  -w /robin/dynamo/disagg_xpu_docker \
  hm_dynamo_b70_pr26460:latest
```

Notes: no `--gpus all` (XPU access is via `--device=/dev/dri` + `/dev/mei*`, not
the NVIDIA runtime); no `--entrypoint` (image default drops to a shell). Run as
`h-zheng` so `~` expands correctly.

## 2. Start the encoder (inside the container)

**Only after the dell07 PD side is up.**

```bash
cd /robin/dynamo/disagg_xpu_docker
./start_sglang_encode_xpu_8b_b70.sh
```

Overrides (env or defaults shown):
```bash
XPU_DEVICE=4 ./start_sglang_encode_xpu_8b_b70.sh    # pick a NUMA-2 XPU + mlx5_2
IP_REMOTE=172.26.46.178 ./start_sglang_encode_xpu_8b_b70.sh   # remote PD host
UCX_NIC=mlx5_2:1 IP_LOCAL=192.165.123.37 ./start_sglang_encode_xpu_8b_b70.sh
```

The script pre-flights NATS/etcd reachability on dell07 and checks the RoCE
NIC↔IP mapping before launching. Log: `./logs/encode_xpu_8b_b70.log`.

## 3. Verify (from dell07)

```bash
curl -s http://127.0.0.1:7001/v1/models      # should now show the served model
```

---

## Benchmark

The benchmark runs **on dell07** against the frontend (`:7001`), not here. See
`../disagg_cuda_docker/README.md` §5.

---

## Notes / gotchas

- **`UCX_TLS=ze_copy,rc,tcp`** — Level-Zero copy + RDMA verbs, no `cuda_ipc`
  (correct for cross-host XPU→CUDA NIXL).
- **`--encoder-only`** runs just the ViT vision tower, so the model's LLM
  attention backend (the PR26460 `xpu_attn` head-size-72 limitation) is not on
  the encode path. If startup fails, check the log for an unrelated cause.
- **`--chat-template qwen2-vl`** is the template used for Qwen3-VL in the
  reference scripts; if the 8B variant needs a different one, point at the
  model's own `chat_template.json`.
- This must use **etcd discovery + nats event plane** (not file/zmq) because
  encoder and PD are on different hosts.

## Teardown

```bash
kill <PID>    # printed at launch
# or:
pkill -f 'dynamo.sglang.*multimodal-encode-worker'
```
