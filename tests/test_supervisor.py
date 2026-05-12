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
