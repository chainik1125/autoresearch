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
from autoresearch.backends.models.base import ModelClient
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
        model_client: ModelClient | None = None,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.compute = compute
        # Optional — when present, auto-dispatched postflight runs route
        # GPU selection through the LLM advisor instead of the deterministic
        # ranker only.
        self.model_client = model_client
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
                and not (run.params or {}).get("postflight_dispatch_failed")
            ):
                # IMPORTANT: write the one-shot sentinel BEFORE attempting the
                # dispatch. Previously this block wrote the sentinel only on
                # success, so a failing dispatch (e.g., RunPod 500) would loop
                # forever — one zombie postflight Run per supervisor tick. The
                # one-shot policy is correct: auto-postflight is best-effort,
                # so a failure is logged as a finding and the user can
                # manually `start_postflight(target_run_id=...)` if they want
                # to retry.
                run.params["postflight_dispatch_failed"] = "in_progress"
                try:
                    run.save(self.storage)
                except Exception:  # noqa: BLE001 -- if we can't even save the sentinel, give up
                    _log.exception("could not write postflight sentinel for %s; skipping", run.id)
                    continue
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
                        model_client=self.model_client,
                        intent="auto-dispatched postflight: text + git push, no GPU compute",
                        compute_type="CPU",                       # ~$0.05/hr; no GPU needed
                        parent_run_id=run.id,
                    )
                    run.params["postflight_run_id"] = pf.id
                    run.params["postflight_dispatch_failed"] = False  # success — clear the sentinel
                    run.save(self.storage)
                    _log.info("run %s terminal; auto-dispatched postflight as %s", run.id, pf.id)
                except Exception as exc:  # noqa: BLE001 -- best-effort; reap pod regardless
                    _log.exception("failed to auto-dispatch postflight for %s", run.id)
                    # Pin the sentinel as "failed" so we never retry. Also write
                    # a finding so the user sees why.
                    run.params["postflight_dispatch_failed"] = True
                    try:
                        run.save(self.storage)
                        from autoresearch.core import findings as findings_mod
                        from autoresearch.core.findings import FindingType
                        findings_mod.append(
                            self.storage, run, FindingType.ERROR,
                            f"auto-postflight dispatch failed: {exc!s}. "
                            f"Manually retry with `start_postflight(target_run_id={run.id!r})`.",
                        )
                    except Exception:  # noqa: BLE001
                        _log.exception("could not pin postflight failure sentinel on %s", run.id)

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
                # No heartbeat ever written. Two cases:
                #   (a) Pod is still booting — that's fine, let it finish.
                #   (b) Pod is wedged (broken image, crashing entrypoint, etc.) —
                #       it'll sit at $X/hr forever unless we time it out.
                # Distinguish by age: a Run that's been QUEUED with no heartbeat
                # for longer than `supervisor_boot_stall_minutes` is presumed
                # wedged. Reap it.
                boot_stall = timedelta(minutes=self.settings.supervisor_boot_stall_minutes)
                age = now - run.created_at
                if run.status == RunStatus.QUEUED and age > boot_stall:
                    _log.warning(
                        "run %s boot-stalled (no heartbeat after %s); failing + terminating pod %s",
                        run.id, age, run.pod_handle,
                    )
                    self._fail_boot_stalled_run(run, age)
                continue
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

    def _fail_boot_stalled_run(self, run: Run, age: timedelta) -> None:
        """Terminate a boot-stalled pod, fetch its container logs as a finding,
        and mark the Run FAILED. Best-effort throughout — supervisor must
        never crash, so each step is wrapped.

        This is the only path that fails a Run from outside the runner. The
        rationale: if the runner can't even write its first heartbeat, the
        container never reached our code, so there's no other actor that can
        record the failure.
        """
        from autoresearch.core import findings as findings_mod
        from autoresearch.core.findings import FindingType

        log_tail: str | None = None
        pod_runpod_status: str | None = None
        pod_image: str | None = None
        if run.pod_handle:
            try:
                handle = self.compute.get_session(run.pod_handle)
                pod_runpod_status = handle.status
                # raw.imageName is the RunPod payload — pull it through if present
                pod_image = (handle.raw or {}).get("imageName")
            except Exception:  # noqa: BLE001
                _log.exception("could not query RunPod status for %s", run.pod_handle)
            try:
                log_tail = self._fetch_pod_logs(run.pod_handle)
            except Exception:  # noqa: BLE001
                _log.exception("could not fetch boot-stall logs for pod %s", run.pod_handle)
            try:
                self.compute.terminate_session(run.pod_handle)
            except Exception:  # noqa: BLE001
                _log.exception("could not terminate boot-stalled pod %s", run.pod_handle)

        try:
            lines = [
                f"BOOT-STALLED: Run sat QUEUED for {age} with no heartbeat.",
                f"Pod handle: {run.pod_handle!r}",
                f"Pod status on RunPod: {pod_runpod_status or 'unknown'}",
                f"Image: {pod_image or '(unknown — could not query)'}",
                "",
                "Likely causes (RunPod doesn't expose container logs over their API, so we can't be more specific):",
                "  - Image pull failed (registry auth, missing tag, broken digest)",
                "  - Entrypoint crashed before the runner could write its first heartbeat",
                "  - Network volume mount failed",
                "",
                "Diagnostics to try:",
                f"  - Verify image is pullable: `docker pull {pod_image or '<image>'}`",
                "  - Check GHCR/ECR/etc. shows the tag",
                "  - Spawn a debug pod with the same image and SSH in manually",
                "  - Inspect entrypoint.sh — most likely culprit on auth or volume setup",
            ]
            if log_tail:
                lines += ["", "Container logs (tail):", "```", log_tail, "```"]
            findings_mod.append(self.storage, run, FindingType.ERROR, "\n".join(lines))
        except Exception:  # noqa: BLE001
            _log.exception("could not write boot-stall finding for %s", run.id)

        try:
            fresh = Run.load(self.storage, run.id)
            fresh.status = RunStatus.FAILED
            fresh.last_error = f"pod boot-stalled (no heartbeat after {age})"
            fresh.save(self.storage)
        except Exception:  # noqa: BLE001
            _log.exception("could not mark %s FAILED after boot-stall", run.id)

    def _fetch_pod_logs(self, pod_id: str) -> str | None:
        """Optional hook to grab container logs from the compute backend.
        Returns the tail of the container's stdout/stderr as a string, or
        None if the backend doesn't expose log retrieval.
        """
        get_logs = getattr(self.compute, "get_session_logs", None)
        if get_logs is None:
            return None
        try:
            return get_logs(pod_id, max_chars=8000)
        except Exception:  # noqa: BLE001
            _log.exception("get_session_logs failed for %s", pod_id)
            return None
