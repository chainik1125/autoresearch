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
MARKER="${WORKSPACE}/.cache/requirements-installed.marker"
if [[ -n "$REQ_FILE" ]] && [[ ! -f "$MARKER" ]]; then
    echo "[entrypoint] one-time pip install -r $REQ_FILE (volume cache)"
    pip install --cache-dir "$PIP_CACHE_DIR" -r "$REQ_FILE"
    echo "$REQ_FILE" > "$MARKER"
fi

if [[ -z "${RUN_ID:-}" ]]; then
    echo "[entrypoint] FATAL: RUN_ID env var is required in pod mode" >&2
    exit 2
fi

echo "[entrypoint] mode=pod  RUN_ID=$RUN_ID  starting runner"
exec autoresearch run --run-id "$RUN_ID" --heartbeat "$@"
