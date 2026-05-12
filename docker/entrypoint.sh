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
if [[ -n "${PROJECT_REPO_URL:-}" ]] && [[ ! -d "${WORKSPACE}/project/.git" ]]; then
    echo "[entrypoint] cloning ${PROJECT_REPO_URL} -> ${WORKSPACE}/project"
    git clone --depth 1 "${PROJECT_REPO_URL}" "${WORKSPACE}/project"
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
