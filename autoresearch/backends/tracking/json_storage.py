"""JSONStorageTracker — append metrics as JSON to the run's storage prefix.

Default tracker. Each `log_metric` call appends one JSON object to
`runs/<id>/metrics/<unique-key>.json`. Cheap, no extra service, surfaces in the
findings forum view.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from autoresearch.backends.storage import StorageBackend
from autoresearch.backends.storage.base import append as storage_append


class JSONStorageTracker:
    name = "json_storage"

    def __init__(self, storage: StorageBackend) -> None:
        self._storage = storage

    def log_metric(self, run_id: str, key: str, value: float, *, step: int | None = None) -> None:
        self.log_metrics(run_id, {key: value}, step=step)

    def log_metrics(self, run_id: str, metrics: dict[str, float], *, step: int | None = None) -> None:
        record: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "metrics": metrics,
        }
        if step is not None:
            record["step"] = step
        storage_append(self._storage, f"runs/{run_id}/metrics", json.dumps(record).encode())
