"""Example user pipeline — illustrates the Pipeline protocol.

This is a *fake* pipeline used for smoke-testing autoresearch without needing a
GPU or real model weights. A real pipeline would replace `run()`'s body with
actual measurement code. The shape (class with `name`, `required_gpu`,
`estimated_minutes`, and `run(*, params, workspace, storage)`) is the contract.

Copy this file into your project's `pipelines/` directory and adapt it.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from autoresearch.backends.storage import StorageBackend


class FRAExample:
    name = "fra_example"
    required_gpu = "A40"
    estimated_minutes = 1  # fake; real FRA is closer to 90 min

    def run(
        self,
        *,
        params: dict[str, Any],
        workspace: Path,
        storage: StorageBackend,
    ) -> dict[str, Any]:
        target_model = params["target_model"]

        # Real pipeline would: load_model(target_model); run measurement; persist
        # intermediate checkpoints to `workspace`; return final result.
        marker = workspace / "loaded_model.txt"
        if not marker.exists():
            marker.write_text(f"loaded {target_model}\n")
            time.sleep(0.5)

        # Allow tests to force a mid-run failure by setting an env var before
        # the first call; subsequent calls (resume) succeed.
        if os.environ.get("FRA_EXAMPLE_FAIL_ONCE") == "1":
            os.environ["FRA_EXAMPLE_FAIL_ONCE"] = ""
            raise RuntimeError("intentional failure for resume test")

        per_layer = [round(0.5 + (i % 7) / 100, 3) for i in range(16)]
        result = {
            "fra_score": sum(per_layer) / len(per_layer),
            "per_layer": per_layer,
            "metadata": {
                "model": target_model,
                "fake": True,
            },
        }

        # Real pipeline might persist a raw artifact for later inspection.
        storage.write(
            f"artifacts/{self.name}/{target_model.replace('/', '_')}.json",
            json.dumps(result).encode("utf-8"),
        )
        return result
