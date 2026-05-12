"""RunTracker protocol — v1 ships only the JSON-in-storage default.

Tracking is separate from the run record / findings forum: it captures
fine-grained metrics suitable for time-series visualization (think loss curves,
per-step throughput). v1's default just appends each metric to storage under
`runs/<id>/metrics/...`; v2 can add a WandB / MLflow adapter for users who
already live in those tools.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RunTracker(Protocol):
    name: str

    def log_metric(self, run_id: str, key: str, value: float, *, step: int | None = None) -> None: ...
    def log_metrics(self, run_id: str, metrics: dict[str, float], *, step: int | None = None) -> None: ...
