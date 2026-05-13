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
_TERMINAL_STATUSES = (RunStatus.COMPLETED, RunStatus.FAILED)
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
        """One pass over all runs.

        Two things happen on each tick:
          1. **Restart**: any active-status Run with a stale heartbeat gets
             redispatched on the same network volume.
          2. **Cleanup**: any terminal-status Run (COMPLETED/FAILED) whose
             pod is still alive gets its pod terminated. This is what
             solves the "who kills the postflight pod" question — the pod
             marks its own Run as COMPLETED at the end of its workflow,
             and the next supervisor tick (≤30s later) terminates it.
             Applies to *all* workflows (transfer, prepare, mechanic,
             postflight), so we never leak compute on completed work.

        Returns the list of run ids restarted in this tick (for logging /
        observability — cleanups are logged but not returned).
        """
        restarted: list[str] = []
        now = datetime.now(UTC)
        short_stale = timedelta(minutes=self.settings.supervisor_stale_minutes)
        long_stale = timedelta(hours=self.settings.supervisor_long_call_stale_hours)

        for run in Run.list_all(self.storage):
            # --- Auto-postflight: when a transfer Run hits terminal status
            # and the user dispatched it with `auto_postflight=true` in
            # params, fire start_postflight before reaping its pod. This is
            # the "chained agents" path: user disengages after the initial
            # /transfer; transfer pod runs (hours); supervisor sees it
            # complete and spawns postflight. Postflight does its work, then
            # this same cleanup pass reaps its pod too.
            if (
                run.status in _TERMINAL_STATUSES
                and run.workflow == "transfer"
                and (run.params or {}).get("auto_postflight")
                and not (run.params or {}).get("postflight_run_id")
            ):
                try:
                    pf = dispatcher.dispatch_new(
                        workflow="postflight",
                        pipeline_name="postflight",
                        params={
                            "target_run_id": run.id,
                            "project_repo_url": run.params.get("project_repo_url"),
                            "project_repo_branch": run.params.get("project_repo_branch"),
                        },
                        budget_usd=self.settings.default_budget_usd,
                        settings=self.settings,
                        storage=self.storage,
                        compute=self.compute,
                        required_vram_gb=8,  # smallest fitting GPU; postflight is mostly text + git
                        parent_run_id=run.id,
                    )
                    run.params["postflight_run_id"] = pf.id
                    run.save(self.storage)
                    _log.info("run %s terminal; auto-dispatched postflight as %s", run.id, pf.id)
                except Exception:  # noqa: BLE001 -- best-effort; reap pod regardless
                    _log.exception("failed to auto-dispatch postflight for %s", run.id)

            # --- Cleanup: terminate pods of terminal-status Runs --------
            if run.status in _TERMINAL_STATUSES and run.pod_handle:
                try:
                    handle = self.compute.get_session(run.pod_handle)
                    if handle.status in ("running", "queued"):
                        _log.info(
                            "run %s is %s; terminating still-alive pod %s",
                            run.id, run.status.value, run.pod_handle,
                        )
                        self.compute.terminate_session(run.pod_handle)
                except Exception:  # noqa: BLE001 -- best-effort; pod may already be gone
                    pass
                continue

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
