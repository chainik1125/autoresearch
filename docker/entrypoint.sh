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

set -euo pipefail

MODE="${MODE:-pod}"

if [[ "$MODE" == "serve" ]]; then
    echo "[entrypoint] mode=serve; starting controller"
    exec autoresearch serve "$@"
fi

# --- pod mode ---

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
    /usr/sbin/sshd
    echo "[entrypoint] sshd started; debug with: ssh -p <port> root@<ip>"
fi

WORKSPACE="${WORKSPACE_DIR:-/workspace}"
mkdir -p "${WORKSPACE}/.huggingface" "${WORKSPACE}/.cache/pip"

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

# Install project requirements once per volume. Marker prevents re-running on
# subsequent pod boots (each volume gets its install cache warmed on the first boot
# that uses it).
REQ_FILE=""
for candidate in "${WORKSPACE}/project/requirements.txt" "${WORKSPACE}/pipelines/requirements.txt"; do
    if [[ -f "$candidate" ]]; then REQ_FILE="$candidate"; break; fi
done
PIP_LOG="${WORKSPACE}/.cache/pip-install.log"
if [[ -n "$REQ_FILE" ]]; then
    # We used to gate pip install on a marker file living on the volume, on the
    # theory that "install once per volume" amortized cost across pod boots.
    # That was wrong: the volume persists but the container's site-packages does
    # NOT, so a fresh pod with a stale marker would skip install and crash on
    # ImportError. Always install; the on-volume pip cache at $PIP_CACHE_DIR
    # makes re-runs fast (~30-60s wheels-from-cache vs ~5min cold).
    echo "[entrypoint] pip install -r $REQ_FILE (cache: $PIP_CACHE_DIR; log -> $PIP_LOG)"
    set +e
    pip install --cache-dir "$PIP_CACHE_DIR" -r "$REQ_FILE" 2>&1 | tee "$PIP_LOG"
    PIP_RC=${PIPESTATUS[0]}
    set -e
    if [[ $PIP_RC -ne 0 ]]; then
        echo "[entrypoint] WARN: pip install exit $PIP_RC; continuing so runner can report a clean error"
    fi
fi

if [[ -z "${RUN_ID:-}" ]]; then
    echo "[entrypoint] FATAL: RUN_ID env var is required in pod mode" >&2
    exit 2
fi

echo "[entrypoint] mode=pod  RUN_ID=$RUN_ID  starting runner"
# Don't exec — we want to run a post-failure log-snapshot hook so that
# diagnostic logs from a failing pipeline subprocess (e.g. a SAE training run
# that wrote /workspace/saes/<m>/<h>/training.log) are surfaced via the
# `list_findings` MCP tool. Without this, the only way to see those logs is to
# spin a separate shell pod and SSH the volume — which has cost us a lot of
# debugging time on this branch.
#
# CRITICAL: disable `set -e` for the runner call. The whole point of the
# snapshot below is to handle a non-zero exit; if `set -e` aborts the script
# at the first failure we never reach it.
set +e
autoresearch run --run-id "$RUN_ID" --heartbeat "$@"
RUNNER_RC=$?
set -e

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
