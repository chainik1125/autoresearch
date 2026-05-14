#!/usr/bin/env bash
# Container entrypoint. Single image, two modes via $MODE:
#
#   MODE=serve   -> exec `autoresearch serve` (controller path; Railway uses this)
#   MODE=pod     -> bootstrap the persistent volume layout, then exec
#                   `autoresearch run --run-id $RUN_ID` (pod path)
#
# Pod mode expectations:
#   - A persistent network volume is mounted at $WORKSPACE_DIR (default /workspace).
#   - $RUN_ID identifies the Run record in storage.
#   - Optional: $PROJECT_REPO_URL clones the user's project onto the volume on first boot.
#   - Optional: project's requirements.txt installs once per volume (marker-gated).

# NOTE: deliberately NOT using `set -e` globally. Pod boots have been
# silently failing with no externally-visible log surface. We trade strict
# failure semantics for diagnostic visibility: every major checkpoint
# writes a `boot_beacon` finding to R2, so if the pod wedges we can see
# exactly which step ran last via `list_findings(run_id)`. Explicit exits
# (`exit 2`, `exit 3`) still terminate; individual command failures just
# print to stderr and the entrypoint continues.
set -uo pipefail

MODE="${MODE:-pod}"

# Both modes (serve + pod) git-clone autoresearch source. The image only
# carries third-party deps; the package itself is fetched fresh at each
# boot so code changes ship via `git push main` without rebuilding the
# image. AUTORESEARCH_REF (default "main") pins the ref — used by the
# dispatcher to test feature branches.
AUTORESEARCH_REF="${AUTORESEARCH_REF:-main}"
echo "[entrypoint] cloning autoresearch @ ${AUTORESEARCH_REF} into /app"
if [[ ! -d /app/.git ]]; then
    git clone --depth 1 --branch "${AUTORESEARCH_REF}" \
        https://github.com/chainik1125/autoresearch.git /app
else
    cd /app && git fetch --depth 1 origin "${AUTORESEARCH_REF}" \
        && git checkout FETCH_HEAD
fi
cd /app
pip install --no-deps -e .
GIT_SHA=$(git -C /app rev-parse --short HEAD 2>/dev/null || echo "unknown")
echo "[entrypoint] autoresearch source ready @ ${AUTORESEARCH_REF} (${GIT_SHA})"

if [[ "$MODE" == "serve" ]]; then
    # Controller mode (Railway). cwd is /app so Settings.load() finds
    # autoresearch.toml and templates/ + pipelines/.
    echo "[entrypoint] mode=serve; starting controller"
    exec autoresearch serve "$@"
fi

# --- pod mode ---

# Beacon 00 confirms the source actually loaded on the pod (writes to R2).
# Controller mode doesn't write findings (no RUN_ID), so beacons are
# pod-only diagnostics.
python3 -m autoresearch.boot_beacon "00 autoresearch source @ ${AUTORESEARCH_REF} (${GIT_SHA}) ready" || true

# Beacon 01: prove we got past argument-parsing / interpreter startup.
python3 -m autoresearch.boot_beacon "01 entrypoint started; mode=pod, RUN_ID=${RUN_ID:-<unset>}" || true

# Start sshd in the background so the pod is shell-accessible for debugging.
# RunPod's pod-create wiring exposes port 22 + injects the operator's public keys
# into $PUBLIC_KEY. The base image has /usr/sbin/sshd; we just need to give it
# host keys + authorized_keys and let it daemonize. This is a no-op if sshd
# isn't installed (e.g. a future base image change). Cost is ~50ms at boot.
if [[ -x /usr/sbin/sshd ]]; then
    mkdir -p /root/.ssh
    if [[ -n "${PUBLIC_KEY:-}" ]]; then
        echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
        chmod 700 /root/.ssh
        chmod 600 /root/.ssh/authorized_keys
    fi
    ssh-keygen -A 2>/dev/null || true
    /usr/sbin/sshd || echo "[entrypoint] WARN: sshd start failed (continuing)"
    echo "[entrypoint] sshd started; debug with: ssh -p <port> root@<ip>"
fi
python3 -m autoresearch.boot_beacon "02 sshd attempted" || true

WORKSPACE="${WORKSPACE_DIR:-/workspace}"
mkdir -p "${WORKSPACE}/.huggingface" "${WORKSPACE}/.cache/pip"
python3 -m autoresearch.boot_beacon "03 workspace mounted at ${WORKSPACE}" || true

# Disk preflight. Writes a finding with df output, and hard-aborts before any
# heavy work if free space is below the floor. Tuned for "model weights +
# training outputs fit comfortably"; the prep agent can prune the HF cache if
# this trips. Override with AUTORESEARCH_MIN_FREE_GB=<n> (default 25).
export AUTORESEARCH_MIN_FREE_GB="${AUTORESEARCH_MIN_FREE_GB:-25}"
export AUTORESEARCH_DISK_PREFLIGHT_PATH="${WORKSPACE}"
if [[ -n "${RUN_ID:-}" ]]; then
    python3 -m autoresearch.disk_preflight
    DISK_RC=$?
    if [[ $DISK_RC -ne 0 ]]; then
        python3 -m autoresearch.boot_beacon "04 disk preflight FAILED rc=$DISK_RC" || true
        echo "[entrypoint] FATAL: disk preflight failed (rc=$DISK_RC); aborting before heavy work" >&2
        sleep 600  # keep pod alive long enough for SSH inspection
        exit 3
    fi
fi
python3 -m autoresearch.boot_beacon "04 disk preflight passed" || true

# Default HF env if the dispatcher didn't already inject them.
export HF_HOME="${HF_HOME:-${WORKSPACE}/.huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${WORKSPACE}/.cache/pip}"

# Clone the user's project repo on first boot (idempotent across pod restarts because
# the volume persists). After clone, expose pipelines/ at the path PIPELINE_MODULE_PATH
# expects.
#
# Private repos: if PROJECT_REPO_TOKEN is set, inject it as basic auth in the clone URL.
# We rewrite the URL just-in-time and never echo it (the token is sensitive). GitHub
# accepts <token>@github.com as basic auth for both classic and fine-grained PATs.
#
# Branch: PROJECT_REPO_BRANCH (set by make_compatible.md when it creates an adaptation
# branch) overrides the default branch.
if [[ -n "${PROJECT_REPO_URL:-}" ]] && [[ ! -d "${WORKSPACE}/project/.git" ]]; then
    AUTH_URL="${PROJECT_REPO_URL}"
    if [[ -n "${PROJECT_REPO_TOKEN:-}" ]]; then
        # Insert token after the scheme; works for https://github.com/... URLs.
        AUTH_URL="${PROJECT_REPO_URL/https:\/\//https://${PROJECT_REPO_TOKEN}@}"
        echo "[entrypoint] cloning private repo (token redacted) -> ${WORKSPACE}/project"
    else
        echo "[entrypoint] cloning ${PROJECT_REPO_URL} -> ${WORKSPACE}/project"
    fi
    CLONE_ARGS=(--depth 1)
    if [[ -n "${PROJECT_REPO_BRANCH:-}" ]]; then
        CLONE_ARGS+=(--branch "${PROJECT_REPO_BRANCH}")
        echo "[entrypoint] (branch: ${PROJECT_REPO_BRANCH})"
    fi
    GIT_TERMINAL_PROMPT=0 git clone "${CLONE_ARGS[@]}" "${AUTH_URL}" "${WORKSPACE}/project"
    unset AUTH_URL  # don't leave the auth-bearing URL hanging in shell state
fi
if [[ -d "${WORKSPACE}/project/pipelines" ]] && [[ ! -e "${WORKSPACE}/pipelines" ]]; then
    ln -s "${WORKSPACE}/project/pipelines" "${WORKSPACE}/pipelines"
fi
python3 -m autoresearch.boot_beacon "05 project repo + pipelines symlink ready" || true

# Install project requirements once per volume. Marker prevents re-running on
# subsequent pod boots (each volume gets its install cache warmed on the first boot
# that uses it).
REQ_FILE=""
for candidate in "${WORKSPACE}/project/requirements.txt" "${WORKSPACE}/pipelines/requirements.txt"; do
    if [[ -f "$candidate" ]]; then REQ_FILE="$candidate"; break; fi
done
PIP_LOG="${WORKSPACE}/.cache/pip-install.log"
if [[ -n "$REQ_FILE" ]]; then
    echo "[entrypoint] pip install -r $REQ_FILE (cache: $PIP_CACHE_DIR; log -> $PIP_LOG)"
    pip install --cache-dir "$PIP_CACHE_DIR" -r "$REQ_FILE" 2>&1 | tee "$PIP_LOG"
    PIP_RC=${PIPESTATUS[0]}
    if [[ $PIP_RC -ne 0 ]]; then
        echo "[entrypoint] WARN: pip install exit $PIP_RC; continuing so runner can report a clean error"
        python3 -m autoresearch.boot_beacon "06 pip install WARNED rc=$PIP_RC (continuing)" || true
    else
        python3 -m autoresearch.boot_beacon "06 pip install ok" || true
    fi
else
    python3 -m autoresearch.boot_beacon "06 no requirements.txt found (skipped pip install)" || true
fi

if [[ -z "${RUN_ID:-}" ]]; then
    echo "[entrypoint] FATAL: RUN_ID env var is required in pod mode" >&2
    sleep 600
    exit 2
fi

echo "[entrypoint] mode=pod  RUN_ID=$RUN_ID  starting runner"
python3 -m autoresearch.boot_beacon "07 about to invoke 'autoresearch run'" || true

autoresearch run --run-id "$RUN_ID" --heartbeat "$@"
RUNNER_RC=$?
python3 -m autoresearch.boot_beacon "08 'autoresearch run' returned rc=$RUNNER_RC" || true

if [[ $RUNNER_RC -ne 0 ]]; then
    echo "[entrypoint] runner exited $RUNNER_RC; snapshotting recent training logs as findings"
    # Snapshot the tail of any training-like log files modified in the last
    # hour. Tail (not full content) so a giant log doesn't bloat R2.
    find "${WORKSPACE}" -name 'training.log' -mmin -60 -type f -size +0 2>/dev/null | while read -r logf; do
        python3 - "$logf" <<'PY' || echo "[entrypoint] WARN: log snapshot failed for $logf"
import os, sys
from autoresearch.config import Settings, build_storage
from autoresearch.core.run import Run
from autoresearch.core.findings import append, FindingType
path = sys.argv[1]
settings = Settings.load()
storage = build_storage(settings)
run = Run.load(storage, os.environ["RUN_ID"])
with open(path) as f:
    body = f.read()
tail = body[-12000:] if len(body) > 12000 else body
append(storage, run, FindingType.ERROR, f"=== {path} (last {len(tail)} chars) ===\n{tail}")
print(f"[entrypoint] uploaded {path} ({len(tail)} chars) as ERROR finding", file=sys.stderr)
PY
    done
fi

exit $RUNNER_RC
