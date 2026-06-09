#!/bin/bash
# Composable piece: full teardown of one test.
#
# Assumes the orchestrator (and therefore the GPU workers) run INSIDE the dell07
# GPU container, so the GPU-server workers are child processes here and pkill
# reaches them. The XPU encoders run in a separate container on the remote B70
# host, torn down via `docker rm -f` over ssh.
#
# Env: XPU_HOST, XPU_SSH_USER, XPU_SSH_PASS, XPU_CONTAINER, GPUS (csv to watch)
set +e

# ---------------------------------------------------------------
# GPU server (dell07, this container): kill workers + control plane
# ---------------------------------------------------------------
echo "[teardown] GPU server: killing workers + control plane..."
pkill -9 -f "dynamo.sglang"            2>/dev/null   # agg / pd / encode workers
pkill -f    "dynamo.frontend"          2>/dev/null
pkill -f    "nats-server"              2>/dev/null
pkill -f    "etcd.*listen-client-urls" 2>/dev/null
sleep 3

# ---------------------------------------------------------------
# XPU server (giga01-b70): remove the remote encoder container
# ---------------------------------------------------------------
XPU_HOST="${XPU_HOST:-172.26.46.180}"
XPU_USER="${XPU_SSH_USER:-h-zheng}"
XPU_CONTAINER="${XPU_CONTAINER:-harness_b70_enc}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o ConnectTimeout=10)
if [ -n "${XPU_SSH_PASS:-}" ] && command -v sshpass >/dev/null; then
    export SSHPASS="$XPU_SSH_PASS"
    sshpass -e ssh "${SSH_OPTS[@]}" "${XPU_USER}@${XPU_HOST}" \
        "docker rm -f $XPU_CONTAINER >/dev/null 2>&1 || true" 2>/dev/null \
        && echo "[teardown] XPU server: removed remote container $XPU_CONTAINER"
fi

# ---------------------------------------------------------------
# Wait for the watched GPUs to drop below 1 GiB used (memory released)
# ---------------------------------------------------------------
if [ -n "${GPUS:-}" ] && command -v nvidia-smi >/dev/null; then
    for try in $(seq 1 20); do
        busy=0
        IFS=',' read -ra GS <<< "$GPUS"
        for g in "${GS[@]}"; do
            used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
            [ -n "$used" ] && [ "$used" -gt 1000 ] && busy=1
        done
        [ "$busy" -eq 0 ] && { echo "[teardown] GPU server: GPUs $GPUS released"; break; }
        sleep 3
    done
fi
echo "[teardown] done"
