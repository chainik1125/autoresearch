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
        runpod_container_registry_auth_id="reg-auth-xyz",
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
    assert spec.container_registry_auth_id == "reg-auth-xyz"
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


def test_dispatch_passes_repo_token_to_pod_env(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "store")
    compute = FakeCompute()
    settings = _settings()
    dispatcher.dispatch_new(
        workflow="transfer", pipeline_name="x", params={"target_model": "Q"},
        budget_usd=10, settings=settings, storage=storage, compute=compute,
        project_repo_token="ghp_smoke_token",
        project_repo_branch="autoresearch/feature-x",
    )
    spec = compute.created[0]
    assert spec.env["PROJECT_REPO_TOKEN"] == "ghp_smoke_token"
    assert spec.env["PROJECT_REPO_BRANCH"] == "autoresearch/feature-x"


def test_dispatch_project_repo_url_per_call_wins(tmp_path: Path) -> None:
    """Per-dispatch project_repo_url overrides the controller's settings.project_repo_url.

    This is what lets `/transfer` dispatch fra_proj's pipelines without
    the controller's autoresearch.toml needing to know about fra_proj."""
    storage = LocalStorage(tmp_path / "store")
    compute = FakeCompute()
    settings = _settings()  # has no project_repo_url
    dispatcher.dispatch_new(
        workflow="transfer", pipeline_name="x", params={"target_model": "Q"},
        budget_usd=10, settings=settings, storage=storage, compute=compute,
        project_repo_url="https://github.com/me/fra_proj.git",
    )
    assert compute.created[0].env["PROJECT_REPO_URL"] == "https://github.com/me/fra_proj.git"


def test_dispatch_project_repo_url_falls_back_to_settings(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "store")
    compute = FakeCompute()
    settings = _settings(project_repo_url="https://github.com/me/default.git")
    dispatcher.dispatch_new(
        workflow="transfer", pipeline_name="x", params={"target_model": "Q"},
        budget_usd=10, settings=settings, storage=storage, compute=compute,
    )
    assert compute.created[0].env["PROJECT_REPO_URL"] == "https://github.com/me/default.git"


def test_redispatch_preserves_repo_token(tmp_path: Path) -> None:
    """A pod restart must reuse the same token; otherwise the new pod fails
    to clone the private repo and the run is dead."""
    storage = LocalStorage(tmp_path / "store")
    compute = FakeCompute()
    settings = _settings(project_repo_token="controller-default-token")
    run = dispatcher.dispatch_new(
        workflow="transfer", pipeline_name="x", params={"target_model": "Q"},
        budget_usd=10, settings=settings, storage=storage, compute=compute,
    )
    fresh = dispatcher.redispatch(run, settings=settings, storage=storage, compute=compute)
    # Both pods got the token from settings.
    assert compute.created[0].env["PROJECT_REPO_TOKEN"] == "controller-default-token"
    assert compute.created[1].env["PROJECT_REPO_TOKEN"] == "controller-default-token"
    assert fresh.pod_handle == "pod-2"


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
