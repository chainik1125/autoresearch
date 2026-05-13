"""Dispatcher tests — verify pod spec construction + run persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoresearch.backends.storage import LocalStorage
from autoresearch.config import Settings
from autoresearch.controller import dispatcher
from autoresearch.core.hardware import GpuOffer
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


def test_dispatch_explicit_gpu_list_passes_through(tmp_path: Path) -> None:
    """Caller's explicit gpu list bypasses auto-selection."""
    storage = LocalStorage(tmp_path / "store")
    compute = FakeCompute()
    settings = _settings()
    dispatcher.dispatch_new(
        workflow="transfer", pipeline_name="x", params={"target_model": "Q"},
        budget_usd=10, settings=settings, storage=storage, compute=compute,
        gpu=["NVIDIA L40S", "NVIDIA RTX A6000"],
    )
    assert compute.created[0].gpu == ["NVIDIA L40S", "NVIDIA RTX A6000"]


def test_dispatch_required_vram_uses_hardware_selector(tmp_path: Path) -> None:
    """When the caller passes only `required_vram_gb`, the dispatcher should
    query the compute backend's offers and pick from there. Uses the default
    fastest_least_complicated heuristic: smallest VRAM bucket with headroom,
    newest gen within that bucket."""
    offers = [
        GpuOffer(id="NVIDIA RTX A6000", memory_gb=48, price_per_hour=0.33, available_in_dc=True),
        GpuOffer(id="NVIDIA L40S", memory_gb=48, price_per_hour=0.79, available_in_dc=True),
        GpuOffer(id="NVIDIA H100 80GB HBM3", memory_gb=80, price_per_hour=2.69, available_in_dc=True),
        # Out-of-DC for the volume: should be filtered out
        GpuOffer(id="NVIDIA A40", memory_gb=48, price_per_hour=None, available_in_dc=False),
    ]
    storage = LocalStorage(tmp_path / "store")
    compute = FakeCompute(offers=offers)
    settings = _settings(runpod_data_center="US-CA-2")

    dispatcher.dispatch_new(
        workflow="transfer", pipeline_name="x", params={"target_model": "Q"},
        budget_usd=10, settings=settings, storage=storage, compute=compute,
        required_vram_gb=30,
    )
    picked = compute.created[0].gpu
    # 48GB bucket (smallest with headroom for 30GB requirement). Newest in
    # bucket = highest price = L40S.
    assert isinstance(picked, list)
    assert picked[0] == "NVIDIA L40S"
    assert "NVIDIA A40" not in picked  # out-of-DC, filtered


def test_dispatch_explicit_gpu_beats_required_vram(tmp_path: Path) -> None:
    """If both `gpu` and `required_vram_gb` are passed, `gpu` wins —
    the explicit pin is an override the user wants."""
    storage = LocalStorage(tmp_path / "store")
    compute = FakeCompute(offers=[
        GpuOffer(id="NVIDIA RTX A6000", memory_gb=48, price_per_hour=0.33, available_in_dc=True),
    ])
    settings = _settings()
    dispatcher.dispatch_new(
        workflow="transfer", pipeline_name="x", params={"target_model": "Q"},
        budget_usd=10, settings=settings, storage=storage, compute=compute,
        gpu="NVIDIA B200",
        required_vram_gb=30,
    )
    assert compute.created[0].gpu == "NVIDIA B200"  # explicit wins


def test_dispatch_falls_back_to_default_gpu_when_no_offers(tmp_path: Path) -> None:
    """If `required_vram_gb` is set but the backend has no offers (or none
    match), the dispatcher should fall back to settings.default_gpu rather
    than fail."""
    storage = LocalStorage(tmp_path / "store")
    compute = FakeCompute(offers=[])  # empty catalog
    settings = _settings()
    dispatcher.dispatch_new(
        workflow="transfer", pipeline_name="x", params={"target_model": "Q"},
        budget_usd=10, settings=settings, storage=storage, compute=compute,
        required_vram_gb=30,
    )
    assert compute.created[0].gpu == settings.default_gpu


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
