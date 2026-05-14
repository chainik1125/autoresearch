"""Boot-time diagnostic finding writer.

Called from entrypoint.sh at multiple checkpoints during pod boot. Writes
an OBSERVATION finding to R2 saying "pod reached step <X>". When a pod
boot-stalls, the trail of findings shows the LAST checkpoint that
succeeded — giving us a coordinate on where in the entrypoint the failure
happened.

Usage from entrypoint.sh:
    python3 -m autoresearch.boot_beacon "step name here" || true

Always returns 0; if anything fails (no RUN_ID, can't reach storage, etc.)
we just print to stderr and exit cleanly. Boot-stall diagnostics should
never themselves block boot.
"""

from __future__ import annotations

import os
import sys


def main(argv: list[str]) -> int:
    step = " ".join(argv[1:]) if len(argv) > 1 else "(unspecified)"
    run_id = os.environ.get("RUN_ID")
    if not run_id:
        print(f"[boot_beacon] step={step!r} skipped (no RUN_ID)", file=sys.stderr)
        return 0

    try:
        from autoresearch.config import Settings, build_storage
        from autoresearch.core.findings import FindingType, append
        from autoresearch.core.run import Run
    except Exception as e:  # noqa: BLE001
        print(f"[boot_beacon] step={step!r} import failure: {e}", file=sys.stderr)
        return 0

    try:
        settings = Settings.load()
        storage = build_storage(settings)
        run = Run.load(storage, run_id)
        append(storage, run, FindingType.OBSERVATION, f"[boot_beacon] {step}")
        print(f"[boot_beacon] step={step!r} written for run {run_id}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"[boot_beacon] step={step!r} write failure: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
