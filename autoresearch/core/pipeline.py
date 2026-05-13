"""Pipeline — protocol that user-defined research pipelines implement.

A Pipeline is a unit of measurement: it loads a model (or models), runs an
experiment, and returns a result dict. autoresearch dispatches it onto compute,
wraps it with validation, persists findings and checkpoints, and survives pod
death.

The user implements Pipeline in their own project (typically under `pipelines/`).
The protocol is intentionally minimal — autoresearch shouldn't dictate how
measurements are structured beyond "give me a `run()` I can call."

Example:

    class FRAMeasurement:
        name = "fra_measurement"
        required_gpu = "A40"
        estimated_minutes = 90

        def run(self, *, params, workspace, storage):
            model = params["target_model"]
            ... # existing FRA code, instrumented for `workspace` writes
            return {"fra_score": ..., "per_layer": [...]}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from autoresearch.backends.storage import StorageBackend


@runtime_checkable
class Pipeline(Protocol):
    name: str
    # `required_gpu` is a human-readable hint, kept for backwards compat.
    # `required_vram_gb` is what the hardware-selection module (see
    # `core/hardware.py`) actually uses to filter offers. If only one is set,
    # the dispatcher uses whichever it can: `required_vram_gb` first, falling
    # back to a lookup table on `required_gpu`. New pipelines should set
    # `required_vram_gb` so the auto-selector has hard numbers to filter on.
    required_gpu: str
    required_vram_gb: int
    estimated_minutes: int

    def run(
        self,
        *,
        params: dict[str, Any],
        workspace: Path,
        storage: StorageBackend,
    ) -> dict[str, Any]:
        """Execute the pipeline; return a JSON-serializable result dict.

        - `params` is the run's parameters (e.g. {"target_model": "..."}).
        - `workspace` is a path on local disk the pipeline may write to. On the pod
          this is on a persistent volume, so writes survive pod death.
        - `storage` is provided so the pipeline can read shared artifacts (e.g.
          a baseline result from a prior run); the pipeline runner handles
          persisting the *returned* dict to storage automatically.
        """
        ...
