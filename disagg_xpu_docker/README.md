# Disagg Encode side — Intel B70 XPU host

This directory holds the launcher for the **encode worker** of a cross-host
disaggregated multimodal deployment:

- **Encode workers** (this host, B70): multimodal encoders on Intel Battlemage
  (B70) XPUs — vision tower only (`--encoder-only`). **Default = 2 encoders**
  (XPU 0 and 1); configurable via `XPU_DEVICES`.
- **PD workers** (remote): 4× L40S on `dell07` (`172.26.46.178`) — see
  `../disagg_cuda_docker/`.
- **Model:** `Qwen/Qwen3-VL-8B-Instruct`
- **Transport:** NIXL over RoCE (RDMA), `ze_copy` (Level-Zero) on the XPU side,
  no `cuda_ipc` (cross-host). PD workers read embeddings (`nixl-read`).

```
   [ B70 XPU host — this host ]               [ dell07 ]
   N encode workers --- NATS/etcd (ctrl) -->  NATS + etcd + frontend(:7001)
   (vision tower)   --- NIXL/RoCE (data) -->  4 PD workers (GPUs 1-4)
```

The encoders register into etcd; the PD workers pull embeddings from whichever
encoder produced them. More encoders ⇒ more parallel vision-tower throughput
(the B70 ViT is the pipeline bottleneck), so a single PD pool can be fed by
several encoders.

The control plane lives on **dell07**, not here. **Start the dell07 PD side
first**, then run the encoder here.

---

## Host facts (sc09giga01-b70)

| Item | Value |
|---|---|
| Remote PD host (control plane) | `172.26.46.178` (dell07 mgmt) |
| NATS / etcd (on dell07) | `:14222` / `:12379` |
| XPU devices | `XPU_DEVICES` (default `"0 1"` = 2 encoders); each pinned via `ZE_AFFINITY_MASK` |
| RoCE NIC (auto by XPU NUMA) | XPU 0-3 → `mlx5_0` (192.165.123.40); XPU 4-7 → `mlx5_2` (192.165.123.37) |
| Port bases (per encoder i) | side-channel `20099+i`, KV-events `22090+i*3`, sys `8081+i` |
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

## 2. Start the encoders (inside the container)

**Only after the dell07 PD side is up.**

```bash
cd /robin/dynamo/disagg_xpu_docker
./start_sglang_encode_xpu_8b_b70.sh          # default: 2 encoders on XPU 0 and 1
```

The script launches **one encode worker per device in `XPU_DEVICES`** (default
`"0 1"`), staggered by `STARTUP_DELAY` seconds, each with its own ports and a
per-device log `./logs/encode_xpu_8b_b70_<xpu>.log`. It pre-flights NATS/etcd
reachability and the RoCE NIC↔IP mapping for each device.

Common overrides (env vars):
```bash
# How many / which XPUs (one encoder each):
XPU_DEVICES="0 1"     ./start_sglang_encode_xpu_8b_b70.sh   # default, both on NUMA-0 / mlx5_0
XPU_DEVICES="0 4"     ./start_sglang_encode_xpu_8b_b70.sh   # spread across both NICs (mlx5_0 + mlx5_2)
XPU_DEVICES="0 1 2 3" ./start_sglang_encode_xpu_8b_b70.sh   # 4 encoders

# Other knobs:
IP_REMOTE=172.26.46.178 ./start_sglang_encode_xpu_8b_b70.sh   # remote PD host
STARTUP_DELAY=5         ./start_sglang_encode_xpu_8b_b70.sh   # faster stagger
```

> Ports auto-increment per encoder, so workers on the same host don't collide:
> side-channel `20099+i`, KV-events `22090+i*3`, sys `8081+i`. The NIC/IP is
> chosen automatically from each XPU's NUMA node.

### 2a. MM attention backend (`xpu_attn` / PR26460)

By default the vision tower uses the `triton_attn` backend. This image
(`hm_dynamo_b70_pr26460`) also ships the **PR26460 `xpu_attn`** path — an Intel
XPU-specific attention kernel that *may* speed up the ViT. Toggle it with the
`MM_ATTN_BACKEND` env var (composes with `XPU_DEVICES`):

```bash
# Default (triton_attn) — no flag added:
./start_sglang_encode_xpu_8b_b70.sh

# Use the PR26460 xpu_attn path on all encoders:
MM_ATTN_BACKEND=xpu_attn ./start_sglang_encode_xpu_8b_b70.sh

# Combine with device selection:
MM_ATTN_BACKEND=xpu_attn XPU_DEVICES="0 1" ./start_sglang_encode_xpu_8b_b70.sh
```

When set, the script adds `--mm-attention-backend xpu_attn` to each encoder and
logs the active backend. **Treat it as an A/B experiment** — verify in the log:
1. **It loads.** `xpu_attn` is known to reject head_size 72 on the *LLM*
   attention path; `--encoder-only` only runs the *vision tower*, so it should
   be fine, but if you see `Unsupported head size` / a kernel error at startup,
   unset `MM_ATTN_BACKEND` to fall back to `triton_attn`.
2. **It's actually faster.** Compare the per-request `vision_encode` time vs the
   `triton_attn` baseline. Prior analysis predicted only a modest single-digit %
   ViT gain — more encoders / lighter workloads remain the bigger throughput
   levers.

## 3. Verify

On the B70 host, confirm each encoder registered:
```bash
grep -il 'Model registration succeeded' logs/encode_xpu_8b_b70_*.log
```
From dell07, confirm the model is being served:
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
- **Multiple encoders share one model on the weka mount** — each loads its own
  copy into its XPU's memory (~2-4 GB for the encoder-only vision tower), so 2-4
  encoders fit comfortably on the B70.
- **`--chat-template qwen2-vl`** is the template used for Qwen3-VL in the
  reference scripts; if the 8B variant needs a different one, point at the
  model's own `chat_template.json`.
- This must use **etcd discovery + nats event plane** (not file/zmq) because
  encoder and PD are on different hosts.

## Teardown

```bash
# stops all encode workers on this host:
pkill -f 'dynamo.sglang.*multimodal-encode-worker'
```
(Individual PIDs are also printed at launch if you want to stop just one.)
