"""Disk-space preflight for pods.

Runs once on pod boot (invoked by docker/entrypoint.sh) BEFORE the pipeline
starts. Two responsibilities:

  1. Always write an OBSERVATION finding with the current `df -h` for the
     persistent volume — so `list_findings(run_id)` shows free space
     without anyone SSHing in.

  2. If free space on the workspace volume is below
     `$AUTORESEARCH_MIN_FREE_GB` (default 25), write an ERROR finding,
     flip the Run to FAILED, and exit non-zero so the entrypoint aborts
     before any heavy work.

Why pre-emptive: ENOSPC mid-pipeline produces tracebacks that don't say
"out of disk" in any obvious way (a tokenizer save failing, a checkpoint
write failing, etc.). We've spent too long diagnosing those. Cheap up-front
check, fail loud.

Override with AUTORESEARCH_MIN_FREE_GB=<n> in the pod env. Set to 0 to
disable the floor (the observation finding still gets written).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys


def _df_h(path: str) -> str:
    """Best-effort `df -h <path>` for the finding body. Plain string."""
    try:
        p = subprocess.run(
            ["df", "-h", path], capture_output=True, text=True, check=False, timeout=5,
        )
        return (p.stdout + p.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return f"(df failed: {e})"


def main() -> int:
    path = os.environ.get("AUTORESEARCH_DISK_PREFLIGHT_PATH", "/workspace")
    floor_gb = int(os.environ.get("AUTORESEARCH_MIN_FREE_GB", "25"))
    run_id = os.environ.get("RUN_ID")

    if not run_id:
        print("[disk_preflight] RUN_ID not set; skipping", file=sys.stderr)
        return 0

    try:
        usage = shutil.disk_usage(path)
    except OSError as e:
        print(f"[disk_preflight] stat({path}) failed: {e}", file=sys.stderr)
        return 0  # treat as advisory — don't abort if we can't measure

    free_gb = usage.free // (1024 ** 3)
    total_gb = usage.total // (1024 ** 3)
    df_body = _df_h(path)
    print(
        f"[disk_preflight] {path}: {free_gb}G free / {total_gb}G total "
        f"(floor {floor_gb}G)",
        file=sys.stderr,
    )

    # Best-effort: write findings. Storage import is deferred so this script
    # can run even if storage config is incomplete (we still want to print).
    try:
        from autoresearch.config import Settings, build_storage
        from autoresearch.core.findings import FindingType, append
        from autoresearch.core.run import Run, RunStatus
    except Exception as e:  # noqa: BLE001
        print(f"[disk_preflight] cannot import storage layer: {e}", file=sys.stderr)
        return 0

    try:
        settings = Settings.load()
        storage = build_storage(settings)
        run = Run.load(storage, run_id)
    except Exception as e:  # noqa: BLE001
        print(f"[disk_preflight] cannot load Run {run_id}: {e}", file=sys.stderr)
        return 0

    body = (
        f"disk preflight @ {path}: free={free_gb}G total={total_gb}G "
        f"floor={floor_gb}G\n\n```\n{df_body}\n```"
    )
    try:
        append(storage, run, FindingType.OBSERVATION, body)
    except Exception as e:  # noqa: BLE001
        print(f"[disk_preflight] could not write OBSERVATION finding: {e}", file=sys.stderr)

    if floor_gb > 0 and free_gb < floor_gb:
        # Prep runs are the *cure* for a full volume — they have explicit
        # authority to prune the HF cache. Don't abort them; just warn.
        if run.workflow == "prepare":
            try:
                append(
                    storage, run, FindingType.OBSERVATION,
                    f"disk floor breached ({free_gb}G < {floor_gb}G) but workflow=prepare "
                    f"is allowed to run so it can prune the volume. The agent will be "
                    f"prompted to clean up before any other work.",
                )
            except Exception:  # noqa: BLE001
                pass
            return 0
        err = (
            f"DISK PREFLIGHT FAILED: {free_gb}G free on {path}, need {floor_gb}G+. "
            f"Aborted before heavy work. Recover by pruning the HF cache "
            f"(prep agent can do this) or resizing the network volume."
        )
        try:
            append(storage, run, FindingType.ERROR, err)
            run.status = RunStatus.FAILED
            run.last_error = (
                f"disk preflight: {free_gb}G free on {path} "
                f"(floor {floor_gb}G)"
            )
            run.save(storage)
        except Exception as e:  # noqa: BLE001
            print(f"[disk_preflight] could not record failure: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
