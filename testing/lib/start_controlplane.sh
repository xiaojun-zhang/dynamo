#!/bin/bash
# Composable piece: start ONLY the control plane (NATS + etcd + frontend).
# Idempotent-ish: kills any prior control-plane procs first.
#
# Used by the orchestrator and by the manual one-shot wrappers. Workers are
# added separately via add_worker.sh / add_encoder_xpu.sh so that a single
# shared frontend can host E + PD (or N agg) instances together.
#
# Env (all have defaults):
#   IP_LOCAL, PORT_NATS, PORT_ETCD, PORT_HTTP, LOG_DIR
set -e

export IP_LOCAL="${IP_LOCAL:-172.26.46.178}"
export PORT_NATS="${PORT_NATS:-14222}"
export PORT_ETCD="${PORT_ETCD:-12379}"
export PORT_HTTP="${PORT_HTTP:-7001}"
export PORT_ETCD_PEER="${PORT_ETCD_PEER:-12380}"
LOG_DIR="${LOG_DIR:-$(pwd)/logs}"
mkdir -p "$LOG_DIR"

# Proxy: bypass for localhost + this host + the XPU host fabric.
export no_proxy="0.0.0.0,127.0.0.1,localhost,${IP_LOCAL},192.165.123.0/24,172.26.46.180,.intel.com"
export NO_PROXY="$no_proxy"

echo "[controlplane] NATS:$PORT_NATS etcd:$PORT_ETCD frontend:$PORT_HTTP"

# Clean any prior control plane (workers are killed by teardown.sh separately).
pkill -f "dynamo.frontend" 2>/dev/null || true
pkill -f "nats-server.*$PORT_NATS" 2>/dev/null || true
pkill -f "etcd.*listen-client-urls.*$PORT_ETCD" 2>/dev/null || true
sleep 2

echo "[controlplane] starting NATS..."
nats-server -js -a 0.0.0.0 -p "$PORT_NATS" -m 18222 \
    > "$LOG_DIR/nats.log" 2>&1 &
sleep 2

echo "[controlplane] starting etcd..."
rm -rf "/tmp/etcd-harness-$$"
etcd \
  --listen-client-urls="http://0.0.0.0:$PORT_ETCD" \
  --advertise-client-urls="http://$IP_LOCAL:$PORT_ETCD" \
  --listen-peer-urls="http://0.0.0.0:$PORT_ETCD_PEER" \
  --initial-advertise-peer-urls="http://0.0.0.0:$PORT_ETCD_PEER" \
  --initial-cluster="default=http://0.0.0.0:$PORT_ETCD_PEER" \
  --data-dir="/tmp/etcd-harness-$$" \
  > "$LOG_DIR/etcd.log" 2>&1 &
sleep 5

for i in $(seq 1 10); do
    if curl -s -o /dev/null "http://localhost:$PORT_ETCD/version"; then
        echo "[controlplane] etcd reachable (attempt $i)"; break
    fi
    sleep 2
done

echo "[controlplane] starting frontend..."
NATS_SERVER="nats://$IP_LOCAL:$PORT_NATS" \
ETCD_ENDPOINTS="http://$IP_LOCAL:$PORT_ETCD" \
ETCD_LEASE_TTL=600 \
ETCD_REQUEST_TIMEOUT=600 \
DYN_REQUEST_PLANE=tcp \
DYN_LOG=debug \
SGLANG_LOG_LEVEL=info \
python3 -m dynamo.frontend \
    --http-port "$PORT_HTTP" \
    --router-mode kv \
    --router-reset-states \
    > "$LOG_DIR/frontend.log" 2>&1 &
sleep 5

if curl -s -o /dev/null "http://localhost:$PORT_HTTP/health"; then
    echo "[controlplane] frontend up on :$PORT_HTTP"
else
    echo "[controlplane] WARN frontend not answering /health yet (check $LOG_DIR/frontend.log)"
fi
