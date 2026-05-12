"""Heartbeat — pod-side liveness signal that the supervisor reads.

The runner spawns a `HeartbeatWriter` thread that writes `runs/<id>/heartbeat.json`
every ~30 seconds. The body carries the current FSM step (read from the checkpoint
on each write) and an `in_long_pipeline_call` flag — the runner flips this to True
around the user `pipeline.run()` call so the supervisor's stale-pod threshold
relaxes from minutes to hours.

Heartbeat errors are swallowed: a transient storage hiccup must never crash the
run. The supervisor will see the staleness if the heartbeat actually breaks.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime

from pydantic import BaseModel

from autoresearch.backends.storage import StorageBackend
from autoresearch.backends.storage.base import KeyNotFound
from autoresearch.core import checkpoint
from autoresearch.core.run import Run


class Heartbeat(BaseModel):
    step: str
    timestamp: datetime
    in_long_pipeline_call: bool


class HeartbeatWriter(threading.Thread):
    def __init__(
        self,
        storage: StorageBackend,
        run: Run,
        *,
        interval_sec: float = 30.0,
    ) -> None:
        super().__init__(daemon=True, name=f"heartbeat-{run.id}")
        self._storage = storage
        self._run = run
        self._interval = interval_sec
        self._in_long_call = False
        self._stop_event = threading.Event()

    def set_long_call(self, value: bool) -> None:
        self._in_long_call = value
        self._write_safe()  # immediate update on flag change

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:  # noqa: D401
        self._write_safe()
        while not self._stop_event.wait(self._interval):
            self._write_safe()

    def _write_safe(self) -> None:
        try:
            self._write()
        except Exception:  # noqa: BLE001 -- never crash the run on heartbeat issues
            pass

    def _write(self) -> None:
        cp = checkpoint.load(self._storage, self._run)
        step = cp.step if cp else "queued"
        beat = Heartbeat(
            step=step,
            timestamp=datetime.now(UTC),
            in_long_pipeline_call=self._in_long_call,
        )
        self._storage.write(self._run.heartbeat_key, beat.model_dump_json().encode())


def load(storage: StorageBackend, run: Run) -> Heartbeat | None:
    """Read the latest heartbeat for a run. None if never written."""
    try:
        data = storage.read(run.heartbeat_key)
    except KeyNotFound:
        return None
    return Heartbeat.model_validate_json(data)
