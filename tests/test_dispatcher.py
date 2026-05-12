"""Dispatcher tests — verify pod spec construction + run persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoresearch.backends.storage import LocalStorage
from autoresearch.config import Settings
from autoresearch.controller import dispatcher
from autoresearch.core.run import Run, RunStatus

from tests._fakes import FakeCompute


def _settings(**overrides) -> Settings:
    base = dict(
        storage="local",
        storage_root="/tmp/store-not-used",
        compute="runpod",
        runpod_api_key="rpa-test",
        runpod_network_volume_id="vol-test",
        runpod_default_image="my/image:latest",
        runpod_container_disk_gb=75,
        default_gpu="H100 80GB",
        controller_public_url="https://ctrl.example.com",
        project_repo_url="https://github.com/me/proj",
    )
    base.update(overrides)
    return Settings(**base)


def test_dispatch_new_creates_run_and_pod(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "store")
    compute = FakeCompute()
    settings = _settings()

    run = dispatcher.dispatch_new(
        workflow="transfer",
        pipeline_name="fra_example",
        params={"target_model": "Qwen/Qwen2.5-32B"},
        budget_usd=30,
        settings=settings,
        storage=storage,
        compute=compute,
    )

    assert run.pod_handle == "pod-1"
    assert run.workflow == "transfer"
    assert run.pipeline_name == "fra_example"
    assert run.budget_cap_usd == 30
    persisted = Run.load(storage, run.id)
    assert persisted.pod_handle == "pod-1"

    assert len(compute.created) == 1
    spec = compute.created[0]
    assert spec.gpu == "H100 80GB"
    assert spec.image == "my/image:latest"
    assert spec.network_volume_id == "vol-test"
    assert spec.container_disk_gb == 75
    assert spec.env["RUN_ID"] == run.id
    assert spec.env["HF_HOME"] == "/workspace/.huggingface"
    assert spec.env["PROJECT_REPO_URL"] == "https://github.com/me/proj"
    assert spec.env["AUTORESEARCH_STORAGE"] == "local"


def test_dispatch_requires_volume(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "store")
    settings = _settings(runpod_network_volume_id=None)
    with pytest.raises(ValueError, match="runpod_network_volume_id is required"):
        dispatcher.dispatch_new(
            workflow="transfer",
            pipeline_name="x",
            params={"target_model": "y"},
            budget_usd=1,
            settings=settings,
            storage=storage,
            compute=FakeCompute(),
        )


def test_redispatch_terminates_old_pod_and_spawns_new(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "store")
    compute = FakeCompute()
    settings = _settings()

    run = dispatcher.dispatch_new(
        workflow="transfer",
        pipeline_name="x",
        params={"target_model": "Q"},
        budget_usd=10,
        settings=settings,
        storage=storage,
        compute=compute,
    )
    assert run.pod_handle == "pod-1"

    fresh = dispatcher.redispatch(run, settings=settings, storage=storage, compute=compute)
    assert fresh.pod_handle == "pod-2"
    assert "pod-1" in compute.terminated
    # Same network volume reused — the whole point of restart.
    assert compute.created[0].network_volume_id == compute.created[1].network_volume_id


def test_redispatch_swallows_termination_errors(tmp_path: Path) -> None:
    """If the old pod is already gone, redispatch must still succeed."""
    storage = LocalStorage(tmp_path / "store")
    settings = _settings()

    class FlakyCompute(FakeCompute):
        def terminate_session(self, session_id: str) -> None:
            raise RuntimeError("pod already terminated")

    compute = FlakyCompute()
    run = dispatcher.dispatch_new(
        workflow="transfer", pipeline_name="x", params={"target_model": "Q"},
        budget_usd=10, settings=settings, storage=storage, compute=compute,
    )
    fresh = dispatcher.redispatch(run, settings=settings, storage=storage, compute=compute)
    assert fresh.pod_handle == "pod-2"  # didn't crash on terminate failure
