#!/usr/bin/env bash
# Single-request multimodal sanity check.
# Sends ONE small-image chat request to the frontend and asserts the model
# returns NON-EMPTY text. Exits 0 on healthy output, non-zero otherwise.
#
# This catches the failure mode where requests return HTTP 200 but empty
# generations (the NIXL embedding transfer silently failing), which makes
# bench_serving falsely report "32/32 successful".
#
# Usage: ./sanity_check.sh [host] [port]
#   defaults: localhost 8000
set -u

HOST="${1:-localhost}"
PORT="${2:-8000}"
URL="http://${HOST}:${PORT}/v1/chat/completions"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
PY="$(dirname "$0")/../venv/bin/python"

echo "=== Sanity check -> $URL (model=$MODEL) ==="

"$PY" - "$URL" "$MODEL" <<'PY'
import sys, json, base64, io, urllib.request, urllib.error

url, model = sys.argv[1], sys.argv[2]

# Build a tiny in-memory test image (64x64 solid red JPEG).
from PIL import Image
buf = io.BytesIO()
Image.new("RGB", (64, 64), (200, 30, 30)).save(buf, format="JPEG")
b64 = base64.b64encode(buf.getvalue()).decode()

payload = {
    "model": model,
    "messages": [{"role": "user", "content": [
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text",
         "text": "What color is this image? Answer in one short sentence."},
    ]}],
    "max_completion_tokens": 64,
    "temperature": 0.0,
    "stream": True,   # matches bench_serving (sglang-oai-chat) path
}

req = urllib.request.Request(
    url, data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"})

text = ""
usage = None
try:
    resp = urllib.request.urlopen(req, timeout=180)
except urllib.error.HTTPError as e:
    print(f"FAIL: HTTP {e.code}\n{e.read().decode()[:2000]}")
    sys.exit(2)
except Exception as e:
    print(f"FAIL: request error: {e}")
    sys.exit(2)

if resp.status != 200:
    print(f"FAIL: HTTP {resp.status}")
    sys.exit(2)

for raw in resp:
    line = raw.decode("utf-8").strip()
    if not line:
        continue
    if line.startswith("data: "):
        line = line[6:]
    if line == "[DONE]":
        break
    try:
        d = json.loads(line)
    except Exception:
        continue
    if d.get("usage"):
        usage = d["usage"]
    for ch in (d.get("choices") or []):
        delta = ch.get("delta") or {}
        text += (delta.get("reasoning_content") or "") + (delta.get("content") or "")

text = text.strip()
print(f"  usage: {usage}")
print(f"  generated_text: {text!r}")

if not text:
    print("FAIL: empty generation (HTTP 200 but no text) -- "
          "embedding transfer likely broken (NIXL). Check PD worker logs for "
          "NIXL_ERR_REMOTE_DISCONNECT or 'Timeout while waiting for available buffer'.")
    sys.exit(1)

print("PASS: non-empty generation.")
sys.exit(0)
PY
rc=$?
echo "=== sanity_check exit code: $rc ==="
exit $rc
