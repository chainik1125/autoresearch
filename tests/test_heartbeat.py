"""HeartbeatWriter tests."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from autoresearch.backends.storage import LocalStorage
from autoresearch.core import checkpoint, heartbeat
from autoresearch.core.heartbeat import HeartbeatWriter
from autoresearch.core.run import Run


@pytest.fixture
def storage(tmp_path: Path) -> LocalStorage:
    return LocalStorage(tmp_path / "store")


def test_writes_on_start_then_periodically(storage: LocalStorage) -> None:
    run = Run(workflow="transfer", pipeline_name="stub")
    run.save(storage)
    writer = HeartbeatWriter(storage, run, interval_sec=0.05)
    writer.start()
    try:
        time.sleep(0.2)
        beat = heartbeat.load(storage, run)
        assert beat is not None
        assert beat.in_long_pipeline_call is False
        assert beat.step in ("queued", "loading", "running", "validating", "completed")
    finally:
        writer.stop()
        writer.join(timeout=1)


def test_reflects_checkpoint_step(storage: LocalStorage) -> None:
    run = Run(workflow="transfer", pipeline_name="stub")
    run.save(storage)
    checkpoint.write(storage, run, step="running")
    writer = HeartbeatWriter(storage, run, interval_sec=0.05)
    writer.start()
    try:
        time.sleep(0.15)
        beat = heartbeat.load(storage, run)
        assert beat is not None and beat.step == "running"
    finally:
        writer.stop()
        writer.join(timeout=1)


def test_set_long_call_flips_flag(storage: LocalStorage) -> None:
    run = Run(workflow="transfer", pipeline_name="stub")
    run.save(storage)
    writer = HeartbeatWriter(storage, run, interval_sec=10)  # long interval — set_long_call writes immediately
    writer.start()
    try:
        time.sleep(0.05)
        writer.set_long_call(True)
        time.sleep(0.05)
        beat = heartbeat.load(storage, run)
        assert beat is not None and beat.in_long_pipeline_call is True

        writer.set_long_call(False)
        time.sleep(0.05)
        beat = heartbeat.load(storage, run)
        assert beat is not None and beat.in_long_pipeline_call is False
    finally:
        writer.stop()
        writer.join(timeout=1)


def test_load_returns_none_when_no_heartbeat(storage: LocalStorage) -> None:
    run = Run(workflow="transfer", pipeline_name="stub")
    run.save(storage)
    assert heartbeat.load(storage, run) is None
