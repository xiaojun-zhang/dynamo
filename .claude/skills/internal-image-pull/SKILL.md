---
name: internal-image-pull
description: Pull Dynamo CI images from AWS ECR. Handles nvsec auth, AWS login, image discovery, and docker pull.
user-invocable: true
---

# Pull Dynamo CI Images from AWS ECR

Authenticate to AWS ECR and pull Dynamo container images built by CI. Handles the full auth flow (nvsec + AWS SSO) and auto-discovers the right image tag.

## Constants

- **ECR Registry:** `210086341041.dkr.ecr.us-west-2.amazonaws.com`
- **ECR Repository:** `ai-dynamo/dynamo`
- **AWS Account:** `triton-aws` / `210086341041`
- **AWS Role:** `CS-Engineer-210086341041`
- **Region:** `us-west-2`
- **Speedoflight Dashboard:** `http://speedoflight.nvidia.com/dynamo/commits/`

## Image Tag Format

```
<full_commit_sha>-<framework>-<variant>-cuda<cuda_major>-<arch>
```

Available frameworks: `vllm`, `sglang`, `trtllm`, `dynamo`
Available variants: `runtime`, `dev`, `test`, `local-dev` (built locally from `dev`)
Available CUDA versions: `cuda12`, `cuda13`
Available architectures: `amd64`, `arm64`, or omitted for multi-arch manifest

**Prefer tags without the arch suffix** (e.g., `<sha>-vllm-runtime-cuda13`) — these are multi-arch manifests that auto-resolve to the correct platform. Only use arch-specific tags if the multi-arch tag is unavailable.

## Instructions

### Step 0: Check Prerequisites

Before anything else, verify that `docker`, `aws`, and `nvsec` are available (see "Prerequisites and Auto-Install" section at the bottom). If any are missing, handle installation before proceeding.

### Step 1: Parse User Intent

Determine what the user wants:
- **Framework**: Which framework image? Default: `vllm`. Options: `vllm`, `sglang`, `trtllm`, `dynamo`.
- **Variant**: `runtime` (default), `dev`, or `local-dev`.
- **CUDA version**: Default: `cuda12`. Detect from `nvidia-smi` if available.
- **Architecture**: Auto-detect via `uname -m` (`x86_64` → `amd64`, `aarch64` → `arm64`).
- **Image selection mode**:
  - **"latest"**: Latest main-branch commit that has images available.
  - **"current"** (default): Find the main-branch commit closest to the current repo HEAD (i.e., the base image matching what the user is working on).
  - **Explicit SHA or tag**: User provides a specific commit SHA or full tag.

If the user doesn't specify, default to: current repo's matching main commit, vllm, runtime, auto-detected arch and CUDA.

### Step 2: Find the Right Image Tag

#### For "current" mode (default):
1. Run `git merge-base HEAD origin/main` to find the latest main-branch commit that is an ancestor of the current HEAD.
2. Use the merge-base SHA as the commit for the image tag.
3. Verify the image exists by checking speedoflight (Step 2b) or falling back to ECR list.

#### For "latest" mode:
1. Fetch the speedoflight dashboard to find the most recent main-branch commit with built images.
2. Use this command to extract the latest commit SHA with available images:
   ```bash
   curl -s "http://speedoflight.nvidia.com/dynamo/commits/" | \
     python3 -c "
   import sys, re
   html = sys.stdin.read()
   # docker-row IDs are ordered latest-first on the page
   short_shas = re.findall(r'id=\"docker-row-([a-f0-9]+)\"', html)
   if not short_shas:
       print('ERROR: no commits found'); sys.exit(1)
   # Map first short SHA to full SHA from image tags
   short = short_shas[0]
   full_shas = re.findall(r'([0-9a-f]{40})(?=-)', html)
   match = next((f for f in full_shas if f.startswith(short)), None)
   if match:
       print(match)
   else:
       print('ERROR: could not resolve full SHA for ' + short); sys.exit(1)
   "
   ```

#### Step 2b: Verify image availability
To check which image variants exist for a commit, query speedoflight:
```bash
curl -s "http://speedoflight.nvidia.com/dynamo/commits/" | \
  python3 -c "
import sys, re
html = sys.stdin.read()
sha = '<COMMIT_SHA>'  # substitute the target SHA
short = sha[:9]
# Find all image tags for this commit
tags = re.findall(sha + r'-([a-z0-9-]+)', html)
unique = sorted(set(tags))
for t in unique:
    print(t)
"
```

If the desired variant is not found, inform the user which variants ARE available and suggest alternatives. Note: runtime images are built for every PR commit, dev images are only built from main.

#### For explicit SHA mode:
If the user provides a short SHA (e.g., `14a6122ea`), resolve it to a full 40-character SHA using speedoflight:
```bash
curl -s "http://speedoflight.nvidia.com/dynamo/commits/" | \
  python3 -c "
import sys, re
html = sys.stdin.read()
short = '<SHORT_SHA>'  # substitute user-provided short SHA
full_shas = re.findall(r'([0-9a-f]{40})(?=-)', html)
match = next((f for f in full_shas if f.startswith(short)), None)
if match:
    print(match)
else:
    print('ERROR: no image found for commit ' + short); sys.exit(1)
"
```
Then verify available variants using Step 2b.

#### Fallback: commit not on speedoflight
Speedoflight only shows the last ~100 main-branch commits. If the target commit is not found there, construct the expected tag and verify it exists using `docker manifest inspect` (requires docker login from Step 3d to be done first — reorder if needed):
```bash
docker manifest inspect 210086341041.dkr.ecr.us-west-2.amazonaws.com/ai-dynamo/dynamo:<full_sha>-<framework>-<variant>-<cuda> 2>&1 | head -1
```
- If it returns JSON (starts with `{`): the image exists, proceed to pull.
- If it returns `no such manifest`: the image doesn't exist (never built or expired).

To check which variants exist for a commit not on speedoflight, try the common variants:
```bash
for variant in runtime dev; do
  tag="<full_sha>-<framework>-${variant}-<cuda>"
  result=$(docker manifest inspect 210086341041.dkr.ecr.us-west-2.amazonaws.com/ai-dynamo/dynamo:$tag 2>&1 | head -1)
  if [[ "$result" == "{"* ]]; then echo "EXISTS: $tag"; else echo "MISSING: $tag"; fi
done
```

#### Handling expired images (30-day retention)
ECR images have a **30-day retention policy**. If both speedoflight and manifest inspect fail:
- For "current" mode: walk back through `git log origin/main --oneline` and check via `docker manifest inspect` for the most recent ancestor commit that still has images.
- For explicit SHA: inform the user the image has expired and suggest using "latest" mode instead.

### Step 3: Authenticate to AWS

Run these checks and authentication steps **sequentially**. Diagnose which credential layer is expired and only fix what's needed.

#### 3a: Check if AWS credentials are still valid
```bash
AWS_PROFILE=dynamo-ecr aws sts get-caller-identity 2>&1
```
- **If this succeeds**: credentials are valid, skip to Step 3d (docker login check).
- **If it fails with `ExpiredToken` or `InvalidIdentityToken`**: AWS SSO session expired → go to Step 3c.
- **If it fails with `NoSuchProfileException` or profile not found**: no profile configured yet → go to Step 3b.

#### 3b: Check nvsec access profile
```bash
# Check if access.json exists and its age
python3 -c "
import json, os, time
path = os.path.expanduser('~/.nvsec/access.json')
if not os.path.exists(path):
    print('MISSING'); exit(0)
age_days = (time.time() - os.path.getmtime(path)) / 86400
print(f'AGE:{age_days:.0f}')
with open(path) as f:
    data = json.load(f)
print(f'USER:{data.get(\"user\", {}).get(\"upn\", \"unknown\")}')
print(f'ACCOUNTS:{len(data.get(\"accounts\", []))}')
"
```
- **If MISSING or AGE > 30**: The user needs to run `nvsec aws auth`. This step requires browser interaction and cannot be fully automated:
  - **For SSH users**: Instruct the user to open a **new local terminal** and run:
    ```
    ssh -L 53682:localhost:53682 <user>@<remote-host>
    ```
    Then on the remote session: `nvsec aws auth --no-browser`
    Copy the printed URL and open it in a local browser.
  - **For local terminal users**: Run `nvsec aws auth` directly (opens browser automatically).
- **If profile exists and AGE <= 30**: Proceed to Step 3c.
- **nvsec location**: Check `which nvsec` first. If not on PATH, look in common venv locations in the repo directory (e.g., `venv/bin/nvsec`, `.venv/bin/nvsec`).

#### 3c: Get AWS credentials via device code flow (no SSH tunnel needed)
```bash
# Find the account index
NVSEC=$(which nvsec 2>/dev/null || find . -path '*/bin/nvsec' -type f 2>/dev/null | head -1)

# Configure with device code flow - pipe "n" to skip profile refresh prompt
echo -e "n\ndynamo-ecr" | $NVSEC aws configure 0 --no-browser --no-refresh
```

This will display a URL and device code. Tell the user:
1. Open the URL in any browser (can be on a different machine).
2. Enter the device code shown.
3. Complete SSO login.

Wait for the command to complete (it blocks until auth finishes, typically 30-60 seconds).

#### 3d: Docker login to ECR
First check if docker is already logged in by attempting a quick manifest check:
```bash
docker manifest inspect 210086341041.dkr.ecr.us-west-2.amazonaws.com/ai-dynamo/dynamo:latest 2>&1 | head -1
```
If this fails with "unauthorized" or "no basic auth credentials", re-login:
```bash
AWS_PROFILE=dynamo-ecr aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin 210086341041.dkr.ecr.us-west-2.amazonaws.com
```
If AWS credentials are valid (Step 3a passed), this should always succeed. If it fails, go back to Step 3c.

### Step 4: Pull the Image

Construct and execute the pull command. **Prefer the multi-arch tag (no arch suffix)** — it auto-resolves to the correct platform.

**IMPORTANT: Run the docker pull in the foreground** (do NOT use `run_in_background`) so the user can see download progress streamed in real time. Set a generous timeout (600000ms / 10 minutes) since these images can be very large (10-30GB+):
```bash
# Preferred: multi-arch manifest (auto-selects amd64/arm64)
docker pull 210086341041.dkr.ecr.us-west-2.amazonaws.com/ai-dynamo/dynamo:<sha>-<framework>-<variant>-<cuda>

# Fallback: arch-specific tag (only if multi-arch tag doesn't exist)
docker pull 210086341041.dkr.ecr.us-west-2.amazonaws.com/ai-dynamo/dynamo:<sha>-<framework>-<variant>-<cuda>-<arch>
```

### Step 4b: Build local-dev Image (only if variant is `local-dev`)

The `local-dev` variant is NOT available in ECR — it must be built locally from the `dev` image. It adds a UID/GID remapping layer so bind-mounted volumes don't have permission issues.

1. First, pull the `dev` image (not runtime) using Step 4:
   ```bash
   docker pull 210086341041.dkr.ecr.us-west-2.amazonaws.com/ai-dynamo/dynamo:<sha>-<framework>-dev-<cuda>
   ```

2. Then build the local-dev image using the speedoflight build script. **Run in the foreground** so the user sees build progress:
   ```bash
   ECR=210086341041.dkr.ecr.us-west-2.amazonaws.com/ai-dynamo/dynamo:<sha>-<framework>-dev-<cuda>
   curl -sL http://speedoflight.nvidia.com/dynamo/dynamo-utils.PRODUCTION/container/build_localdev_from_dev.py | python3 - --skip-pull $ECR
   ```

   The `--skip-pull` flag is used because we already pulled the image in step 1.

   The script will:
   - Generate a Dockerfile that remaps UID/GID to match the host user
   - Build and tag the image as `dynamo:<tag>-local-dev` and `dynamo:latest-<framework>-local-dev`

3. After build, show the user the suggested run commands:
   ```bash
   ./container/run.sh --image dynamo:<tag>-local-dev --mount-workspace --hf-home ~/.cache/huggingface -it
   ```

Note: `dev` images are only built from `main` branch commits, not PR commits.

### Step 5: Report Result

After a successful pull, display:
- The full image URI that was pulled
- The commit SHA it corresponds to
- A short `docker run` example if relevant

If the pull fails:
- **Auth errors**: Re-run Step 3
- **Image not found**: Re-check available tags via speedoflight (Step 2b) and suggest alternatives
- **Network errors**: Suggest checking VPN/proxy settings

## Credential Lifetimes

| Credential | Lifetime | Refresh Command |
|---|---|---|
| nvsec access profile | 30 days | `nvsec aws auth` (needs browser) |
| AWS SSO session | ~8 hours | `nvsec aws configure 0 --no-browser --no-refresh` |
| AWS credentials | 45 minutes | Same as above |
| Docker ECR token | 12 hours | `aws ecr get-login-password ...` |

## Prerequisites and Auto-Install

Before starting the auth flow, check that all prerequisites are available. Install what's missing automatically.

### Check and install prerequisites
```bash
# 1. Check Docker
which docker >/dev/null 2>&1 && echo "docker: OK" || echo "docker: MISSING"

# 2. Check AWS CLI
aws --version 2>&1 | head -1 || echo "aws: MISSING"

# 3. Check nvsec
NVSEC=$(which nvsec 2>/dev/null || find . -maxdepth 3 -path '*/bin/nvsec' -type f 2>/dev/null | head -1)
if [ -n "$NVSEC" ]; then echo "nvsec: OK ($NVSEC)"; else echo "nvsec: MISSING"; fi
```

### Install AWS CLI (if missing)
**Ask the user before installing.** They may have it installed elsewhere or prefer a specific installation method. If they confirm, install:
```bash
# Linux x86_64
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "$TMPDIR/awscliv2.zip"
unzip -q "$TMPDIR/awscliv2.zip" -d "$TMPDIR/aws-install"
sudo $TMPDIR/aws-install/aws/install
rm -rf "$TMPDIR/awscliv2.zip" "$TMPDIR/aws-install"

# Linux aarch64
curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "$TMPDIR/awscliv2.zip"
unzip -q "$TMPDIR/awscliv2.zip" -d "$TMPDIR/aws-install"
sudo $TMPDIR/aws-install/aws/install
rm -rf "$TMPDIR/awscliv2.zip" "$TMPDIR/aws-install"
```
Verify: `aws --version` should show v2.24+.

### Install nvsec (if missing)
**Ask the user before installing.** They may have it in a different venv or path. If they confirm, install:
nvsec is an internal NVIDIA package not available on public PyPI. Install it from the internal gitlab:
```bash
pip install nvsec --index-url https://gitlab-master.nvidia.com/api/v4/projects/security%2Fsecurity-portal%2Fnvsec-tool/packages/pypi/simple --extra-index-url https://pypi.org/simple
```
If that URL doesn't work (it may require gitlab auth), direct the user to:
- Visit https://gitlab-master.nvidia.com/security/security-portal/nvsec-tool
- Follow the install instructions in the README
- Or ask a teammate to share the wheel file

Verify: `nvsec version` should print version info.

### Docker (if missing)
Docker installation varies by platform. Direct the user to https://docs.docker.com/engine/install/ rather than attempting to install it automatically.

### Access requirement
Users must be in the DL Access group `access-triton-aws-engineer`. If auth fails with a permissions error, direct them to request access through the NVIDIA access portal.
