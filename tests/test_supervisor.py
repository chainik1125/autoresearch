"""Supervisor tests — heartbeat-staleness detection and restart."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from autoresearch.backends.storage import LocalStorage
from autoresearch.config import Settings
from autoresearch.controller import dispatcher
from autoresearch.controller.supervisor import Supervisor
from autoresearch.core import checkpoint
from autoresearch.core.heartbeat import Heartbeat
from autoresearch.core.run import Run, RunStatus

from tests._fakes import FakeCompute


def _settings(**overrides) -> Settings:
    base = dict(
        storage="local",
        compute="runpod",
        runpod_api_key="rpa-test",
        runpod_network_volume_id="vol-test",
        runpod_default_image="img",
        supervisor_poll_seconds=0.05,
        supervisor_stale_minutes=5,
        supervisor_long_call_stale_hours=2,
    )
    base.update(overrides)
    return Settings(**base)


def _write_beat(storage, run: Run, *, age: timedelta, in_long_call: bool = False) -> None:
    beat = Heartbeat(
        step="running",
        timestamp=datetime.now(UTC) - age,
        in_long_pipeline_call=in_long_call,
    )
    storage.write(run.heartbeat_key, beat.model_dump_json().encode())


def _seed_active_run(tmp_path: Path, settings: Settings) -> tuple[LocalStorage, FakeCompute, Run]:
    storage = LocalStorage(tmp_path / "store")
    compute = FakeCompute()
    run = dispatcher.dispatch_new(
        workflow="transfer", pipeline_name="x", params={"target_model": "Q"},
        budget_usd=10, settings=settings, storage=storage, compute=compute,
    )
    # Promote past QUEUED so supervisor considers it.
    run.status = RunStatus.RUNNING
    run.save(storage)
    checkpoint.write(storage, run, step="running")
    return storage, compute, run


def test_tick_restarts_stale_pod(tmp_path: Path) -> None:
    settings = _settings()
    storage, compute, run = _seed_active_run(tmp_path, settings)
    _write_beat(storage, run, age=timedelta(minutes=10))  # > 5 min threshold

    sup = Supervisor(settings=settings, storage=storage, compute=compute)
    restarted = sup.tick()
    assert restarted == [run.id]
    assert "pod-1" in compute.terminated
    assert len(compute.created) == 2  # original + restart
    fresh = Run.load(storage, run.id)
    assert fresh.pod_handle == "pod-2"


def test_tick_leaves_fresh_beat_alone(tmp_path: Path) -> None:
    settings = _settings()
    storage, compute, run = _seed_active_run(tmp_path, settings)
    _write_beat(storage, run, age=timedelta(seconds=30))

    sup = Supervisor(settings=settings, storage=storage, compute=compute)
    restarted = sup.tick()
    assert restarted == []
    assert compute.terminated == []
    assert len(compute.created) == 1  # no restart


def test_long_call_relaxes_threshold(tmp_path: Path) -> None:
    """A 30-min-old heartbeat is stale by the short threshold but NOT stale by
    the long-call threshold."""
    settings = _settings()
    storage, compute, run = _seed_active_run(tmp_path, settings)
    _write_beat(storage, run, age=timedelta(minutes=30), in_long_call=True)

    sup = Supervisor(settings=settings, storage=storage, compute=compute)
    assert sup.tick() == []  # still within 2h threshold

    # Push past 2h
    _write_beat(storage, run, age=timedelta(hours=3), in_long_call=True)
    assert sup.tick() == [run.id]


def test_skips_completed_runs(tmp_path: Path) -> None:
    settings = _settings()
    storage, compute, run = _seed_active_run(tmp_path, settings)
    _write_beat(storage, run, age=timedelta(hours=10))
    run.status = RunStatus.COMPLETED
    run.save(storage)

    sup = Supervisor(settings=settings, storage=storage, compute=compute)
    assert sup.tick() == []  # supervisor ignores non-active runs


def test_skips_run_without_heartbeat_yet(tmp_path: Path) -> None:
    """If no heartbeat has been written yet, supervisor leaves the run alone —
    the new pod is still booting."""
    settings = _settings()
    storage, compute, run = _seed_active_run(tmp_path, settings)
    sup = Supervisor(settings=settings, storage=storage, compute=compute)
    assert sup.tick() == []


def test_boot_stalled_run_is_failed_and_pod_terminated(tmp_path: Path) -> None:
    """Regression: a Run that's been QUEUED for longer than
    `supervisor_boot_stall_minutes` with no heartbeat is presumed boot-stalled
    (broken image, crashed entrypoint, failed volume mount). Supervisor must:
      - terminate the pod (so we stop burning $X/hr on a wedged pod),
      - mark the Run FAILED with last_error explaining the stall,
      - write a BOOT-STALLED ERROR finding for diagnostics.

    Backstory: a prior dispatch wedged at uptimeSeconds=0 and burned ~$15 over
    5 hours because the supervisor's "no heartbeat" branch was an
    unconditional skip.
    """
    from autoresearch.core import findings as findings_mod

    settings = _settings(supervisor_boot_stall_minutes=10)
    storage = LocalStorage(tmp_path / "store")
    compute = FakeCompute()
    run = dispatcher.dispatch_new(
        workflow="prepare", pipeline_name="x", params={},
        budget_usd=10, settings=settings, storage=storage, compute=compute,
    )
    # No heartbeat. Backdate created_at to look 15min old.
    run.created_at = datetime.now(UTC) - timedelta(minutes=15)
    run.save(storage)

    sup = Supervisor(settings=settings, storage=storage, compute=compute)
    sup.tick()

    fresh = Run.load(storage, run.id)
    assert fresh.status == RunStatus.FAILED
    assert "boot-stalled" in (fresh.last_error or "").lower()
    assert run.pod_handle in compute.terminated  # pod actually got reaped
    finding_bodies = [f.body for f in findings_mod.list_findings(storage, fresh)]
    assert any("BOOT-STALLED" in b for b in finding_bodies)


def test_recent_queued_run_with_no_heartbeat_is_left_alone(tmp_path: Path) -> None:
    """The boot-stall reaper must NOT fire on a Run that's still within the
    grace window — that's a normal boot, the pod just hasn't written its
    first heartbeat yet."""
    settings = _settings(supervisor_boot_stall_minutes=10)
    storage = LocalStorage(tmp_path / "store")
    compute = FakeCompute()
    run = dispatcher.dispatch_new(
        workflow="prepare", pipeline_name="x", params={},
        budget_usd=10, settings=settings, storage=storage, compute=compute,
    )
    # 2 minutes old — well within the 10-minute grace window.
    run.created_at = datetime.now(UTC) - timedelta(minutes=2)
    run.save(storage)

    sup = Supervisor(settings=settings, storage=storage, compute=compute)
    sup.tick()

    fresh = Run.load(storage, run.id)
    assert fresh.status == RunStatus.QUEUED  # untouched
    assert run.pod_handle not in compute.terminated


@pytest.mark.asyncio
async def test_start_stop_lifecycle(tmp_path: Path) -> None:
    """The async supervisor loop starts and stops cleanly without restart events."""
    settings = _settings(supervisor_poll_seconds=0.02)
    storage = LocalStorage(tmp_path / "store")
    compute = FakeCompute()
    sup = Supervisor(settings=settings, storage=storage, compute=compute)
    sup.start()
    await asyncio.sleep(0.1)  # let it tick a few times
    await sup.stop()
    # No runs created -> no work to do -> no error.
    assert compute.created == []
