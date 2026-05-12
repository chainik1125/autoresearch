"""Supervisor — async background loop that restarts pods whose heartbeat went stale.

For every active Run, the supervisor reads `heartbeat.json` and compares its
timestamp to wall-clock now. If the gap exceeds the configured threshold (longer
when the runner is mid-`pipeline.run()`), the pod is assumed dead and a new pod
is spawned on the same network volume. The new runner reads the checkpoint from
storage and resumes from the last completed FSM step.

The supervisor never touches Run state beyond `pod_handle` — the runner owns
status, findings, and checkpoint writes.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from autoresearch.backends.compute import ComputeBackend
from autoresearch.backends.storage import StorageBackend
from autoresearch.config import Settings
from autoresearch.controller import dispatcher
from autoresearch.core import heartbeat
from autoresearch.core.run import Run, RunStatus


_ACTIVE_STATUSES = (RunStatus.QUEUED, RunStatus.LOADING, RunStatus.RUNNING, RunStatus.VALIDATING)
_log = logging.getLogger("autoresearch.supervisor")


class Supervisor:
    def __init__(
        self,
        *,
        settings: Settings,
        storage: StorageBackend,
        compute: ComputeBackend,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.compute = compute
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        if self._task is not None:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._loop(), name="autoresearch-supervisor")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
            self._task = None

    async def _loop(self) -> None:
        interval = self.settings.supervisor_poll_seconds
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self.tick)
            except Exception:  # noqa: BLE001 -- supervisor must never die
                _log.exception("supervisor tick failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    def tick(self) -> list[str]:
        """One pass over all active runs. Returns the list of run ids restarted."""
        restarted: list[str] = []
        now = datetime.now(UTC)
        short_stale = timedelta(minutes=self.settings.supervisor_stale_minutes)
        long_stale = timedelta(hours=self.settings.supervisor_long_call_stale_hours)

        for run in Run.list_all(self.storage):
            if run.status not in _ACTIVE_STATUSES:
                continue
            beat = heartbeat.load(self.storage, run)
            if beat is None:
                continue  # no heartbeat yet — let the new pod write its first one
            threshold = long_stale if beat.in_long_pipeline_call else short_stale
            if now - beat.timestamp <= threshold:
                continue
            _log.warning(
                "run %s heartbeat stale by %s (in_long_call=%s); restarting pod",
                run.id, now - beat.timestamp, beat.in_long_pipeline_call,
            )
            try:
                dispatcher.redispatch(
                    run, settings=self.settings, storage=self.storage, compute=self.compute,
                )
                restarted.append(run.id)
            except Exception:  # noqa: BLE001 -- log and continue
                _log.exception("failed to redispatch run %s", run.id)
        return restarted
