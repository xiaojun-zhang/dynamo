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
# GPU server (dell07, this container): kill OUR workers + control plane only.
# We kill by recorded PID (harness.pids) + their process groups, then a
# PORT-SCOPED backstop. We never blanket-pkill nats/etcd/frontend, because this
# is a shared host with other tenants' stacks on different ports.
# ---------------------------------------------------------------
LOG_DIR="${LOG_DIR:-$(pwd)/logs}"
PIDFILE="$LOG_DIR/harness.pids"
PORT_NATS="${PORT_NATS:-14222}"
PORT_ETCD="${PORT_ETCD:-12379}"
PORT_HTTP="${PORT_HTTP:-7001}"

echo "[teardown] GPU server: killing our recorded PIDs..."
if [ -f "$PIDFILE" ]; then
    # kill children first (workers), parents last; -TERM then -KILL.
    for sig in TERM KILL; do
        while read -r p; do
            [ -n "$p" ] || continue
            kill "-$sig" "$p" 2>/dev/null
            # also the process group, to catch sglang's spawned subprocs
            kill "-$sig" "-$p" 2>/dev/null
        done < <(tac "$PIDFILE")
        sleep 2
    done
fi

# PORT-SCOPED backstop for the control plane only (never a blanket pkill).
# Workers are reliably handled by the PID/process-group kill above (they carry
# the etcd endpoint in env, not argv, so they can't be matched by port here).
pkill -9 -f "dynamo.frontend.*--http-port $PORT_HTTP" 2>/dev/null
pkill -9 -f "nats-server.*-p $PORT_NATS"              2>/dev/null
pkill -9 -f "etcd.*listen-client-urls.*:$PORT_ETCD"   2>/dev/null
sleep 2

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
