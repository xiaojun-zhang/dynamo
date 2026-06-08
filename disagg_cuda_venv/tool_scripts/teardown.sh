#!/usr/bin/env bash
# Teardown: kill all Dynamo/SGLang processes and show GPU state.
# Usage: ./teardown.sh
set -u

echo "=== Killing dynamo.sglang workers ==="
pkill -9 -f 'dynamo.sglang' && echo "  killed dynamo.sglang" || echo "  none running"

echo "=== Killing dynamo.frontend ==="
pkill -9 -f 'dynamo.frontend' && echo "  killed dynamo.frontend" || echo "  none running"

# SGLang scheduler/diffusion children sometimes survive the parent kill.
pkill -9 -f 'sglang' 2>/dev/null && echo "  killed stray sglang children" || true

echo "=== Waiting 3s for ports/GPU mem to release ==="
sleep 3

# Clear stale file-discovery registrations so a new frontend doesn't try to
# route to workers that no longer exist (a cause of empty/broken responses).
DISCOVERY_DIR=/tmp/dynamo_store_kv/v1
if [[ -d "$DISCOVERY_DIR" ]]; then
  rm -f "$DISCOVERY_DIR"/instances/* "$DISCOVERY_DIR"/mdc/* 2>/dev/null
  echo "=== Cleared stale discovery state in $DISCOVERY_DIR (instances/, mdc/) ==="
fi

echo "=== Remaining dynamo/sglang processes (should be empty) ==="
ps -eo pid,etime,cmd | grep -E 'dynamo\.(sglang|frontend)|sglang' | grep -v grep || echo "  (none)"

echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
