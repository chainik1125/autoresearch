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


def test_dispatch_records_parent_run_id(tmp_path: Path) -> None:
    """A dispatched-by-agent Run records its parent so the run-tree can be
    reconstructed. Root Runs have parent_run_id=None; agent-spawned ones
    record the spawning Run's id."""
    storage = LocalStorage(tmp_path / "store")
    compute = FakeCompute()
    settings = _settings()

    # Root dispatch: no parent.
    root = dispatcher.dispatch_new(
        workflow="transfer", pipeline_name="x", params={"target_model": "Q"},
        budget_usd=10, settings=settings, storage=storage, compute=compute,
    )
    assert root.parent_run_id is None
    persisted_root = Run.load(storage, root.id)
    assert persisted_root.parent_run_id is None

    # Child dispatch (simulates a prep agent spawning a compute run).
    child = dispatcher.dispatch_new(
        workflow="transfer", pipeline_name="y", params={"target_model": "Q"},
        budget_usd=10, settings=settings, storage=storage, compute=compute,
        parent_run_id=root.id,
    )
    assert child.parent_run_id == root.id
    persisted_child = Run.load(storage, child.id)
    assert persisted_child.parent_run_id == root.id


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


def test_agent_workflow_picks_cheapest_not_default_gpu(tmp_path: Path) -> None:
    """Regression: agent workflows (prepare/mechanic/postflight) must use the
    cheapest sufficient GPU, not silently fall through to settings.default_gpu.

    Backstory: a prep dispatch landed on H200 ($5.49/hr) for a 30-second LLM
    call because (a) prep's required_vram_gb resolution returned no picks,
    (b) the dispatcher fell back to default_gpu="H100 80GB", which RunPod
    fuzzy-matched to whatever 80GB+ SKU had stock. ~5000x cost overrun.
    """
    offers = [
        GpuOffer(id="NVIDIA RTX A4000", memory_gb=16, price_per_hour=0.17, available_in_dc=True),
        GpuOffer(id="NVIDIA RTX A5000", memory_gb=24, price_per_hour=0.26, available_in_dc=True),
        GpuOffer(id="NVIDIA H100 80GB HBM3", memory_gb=80, price_per_hour=2.69, available_in_dc=True),
    ]
    storage = LocalStorage(tmp_path / "store")
    compute = FakeCompute(offers=offers)
    settings = _settings(runpod_data_center="US-CA-2")

    dispatcher.dispatch_new(
        workflow="prepare", pipeline_name="x", params={},
        budget_usd=10, settings=settings, storage=storage, compute=compute,
        required_vram_gb=8,
        prefer="cheapest",
        fallback_to_default=False,
    )
    picked = compute.created[0].gpu
    assert isinstance(picked, list)
    # Cheapest sufficient: A4000 at $0.17/hr, then A5000, then (way) H100.
    assert picked[0] == "NVIDIA RTX A4000"
    assert "NVIDIA H100 80GB HBM3" in picked  # still available as final fallback


def test_cpu_pod_container_disk_clamped_to_40(tmp_path: Path) -> None:
    """Regression: RunPod's CPU pods cap `containerDiskInGb` at 40. Our
    GPU-tuned default is 50; the dispatcher must clamp on the CPU path or
    RunPod returns 500 'Container Disk must be less than or equal to 40'.

    Backstory: verified against RunPod's REST API on 2026-05-13 — the first
    CPU pod rollout 500'd on this exact error before the clamp was added.
    """
    storage = LocalStorage(tmp_path / "store")
    compute = FakeCompute()
    settings = _settings(runpod_container_disk_gb=50)  # the GPU-tuned default

    dispatcher.dispatch_new(
        workflow="prepare", pipeline_name="x", params={},
        budget_usd=10, settings=settings, storage=storage, compute=compute,
        compute_type="CPU",
    )
    spec = compute.created[0]
    assert spec.compute_type == "CPU"
    assert spec.container_disk_gb == 40, (
        f"CPU pod disk must be clamped to 40, got {spec.container_disk_gb}"
    )


def test_dispatched_hardware_recorded_in_params(tmp_path: Path) -> None:
    """Regression: a dispatched Run's params must record what hardware was
    actually requested (compute_type, image, cpu_flavors OR gpu). Otherwise
    `get_run` can't tell whether `compute_type="CPU"` was honored or silently
    dropped — exactly the auditability gap the local Claude hit."""
    storage = LocalStorage(tmp_path / "store")
    compute = FakeCompute()
    settings = _settings()

    # CPU dispatch
    cpu_run = dispatcher.dispatch_new(
        workflow="prepare", pipeline_name="x", params={"user_key": "user_value"},
        budget_usd=10, settings=settings, storage=storage, compute=compute,
        compute_type="CPU",
    )
    hw = cpu_run.params["_dispatched_hardware"]
    assert hw["compute_type"] == "CPU"
    assert hw["cpu_flavors"] == ["cpu3c", "cpu3g", "cpu5c", "cpu5g"]
    assert hw["vcpu_count"] == 4
    assert "gpu" not in hw   # no GPU fields on a CPU dispatch
    # User-supplied params must be preserved alongside the audit field.
    assert cpu_run.params["user_key"] == "user_value"

    # GPU dispatch
    gpu_run = dispatcher.dispatch_new(
        workflow="transfer", pipeline_name="x", params={"target_model": "Q"},
        budget_usd=10, settings=settings, storage=storage, compute=compute,
        gpu="H100 80GB",
    )
    hw = gpu_run.params["_dispatched_hardware"]
    assert hw["compute_type"] == "GPU"
    assert hw["gpu"] == ["H100 80GB"]


def test_agent_workflow_dispatches_cpu_pod_by_default(tmp_path: Path) -> None:
    """Regression: when compute_type='CPU', the SessionSpec must have CPU
    fields populated and the GPU path must be skipped entirely.

    Backstory: agent workflows (prep / mechanic / postflight) are pure
    LLM+shell+git workloads with no GPU compute. Dispatching them on the
    cheapest GPU pod (~$0.17/hr) is still ~4x what a CPU pod costs
    (~$0.05/hr). This test pins the CPU default.
    """
    storage = LocalStorage(tmp_path / "store")
    compute = FakeCompute()  # no offers — CPU path doesn't consult catalog
    settings = _settings()

    dispatcher.dispatch_new(
        workflow="prepare", pipeline_name="x", params={},
        budget_usd=10, settings=settings, storage=storage, compute=compute,
        compute_type="CPU",
    )
    spec = compute.created[0]
    assert spec.compute_type == "CPU"
    # Defaults are cheapest-first with multiple fallbacks (verified against
    # US-CA-2 stock 2026-05-13). RunPod picks any in-stock entry.
    assert spec.cpu_flavors == ["cpu3c", "cpu3g", "cpu5c", "cpu5g"]
    assert spec.vcpu_count == 4
    assert spec.gpu == []  # explicitly empty


def test_cpu_dispatch_falls_back_no_volume_when_primary_dc_out_of_stock(tmp_path: Path) -> None:
    """Regression: when the primary DC has no CPU stock (RunPod 500
    'no instances available'), CPU dispatches retry without the network
    volume so RunPod can place the pod in any DC.

    Backstory: a real US-CA-2 outage on 2026-05-14 made all GPU + CPU SKUs
    return 500. Agent workflows don't actually need the volume contents
    (they're shell + LLM + git, no model downloads), so we drop the volume
    on retry to unblock.
    """
    storage = LocalStorage(tmp_path / "store")
    settings = _settings()

    class FlakyThenOK(FakeCompute):
        def __init__(self):
            super().__init__()
            self._attempts = 0

        def create_session(self, spec):
            self._attempts += 1
            if self._attempts == 1:
                # Simulate the RunPod 500 we see in real life
                raise RuntimeError(
                    "create pod: There are no longer any instances available "
                    "with the requested specifications. Please refresh and try again."
                )
            return super().create_session(spec)

    compute = FlakyThenOK()
    run = dispatcher.dispatch_new(
        workflow="prepare", pipeline_name="x", params={},
        budget_usd=10, settings=settings, storage=storage, compute=compute,
        compute_type="CPU",
    )

    # First attempt was with volume; second was without
    assert compute._attempts == 2
    assert compute.created[0].network_volume_id == ""  # the retry spec
    # The original spec (before retry) had the volume; the retry stripped it.
    # The Run record reflects the volume_dropped fallback.
    hw = run.params["_dispatched_hardware"]
    assert hw.get("volume_dropped") is True
    assert "fallback DC" in hw.get("note", "")


def test_gpu_dispatch_does_not_fall_back_no_volume(tmp_path: Path) -> None:
    """The volume-drop fallback must NOT fire for compute_type='GPU' — the
    transfer workflow needs the volume's HF cache + outputs. Better to fail
    loud than silently dispatch a GPU pod that'll re-download everything.
    """
    storage = LocalStorage(tmp_path / "store")
    settings = _settings()

    class AlwaysNoStock(FakeCompute):
        def create_session(self, spec):
            raise RuntimeError("create pod: There are no instances currently available")

    compute = AlwaysNoStock()
    with pytest.raises(RuntimeError, match="no instances"):
        dispatcher.dispatch_new(
            workflow="transfer", pipeline_name="x", params={"target_model": "Q"},
            budget_usd=10, settings=settings, storage=storage, compute=compute,
            gpu="H100 80GB",
        )


def test_agent_workflow_no_fallback_raises_when_catalog_empty(tmp_path: Path) -> None:
    """With fallback_to_default=False, a missing catalog must fail loudly
    rather than silently dispatch on settings.default_gpu (which is "H100
    80GB" — an order-of-magnitude cost mistake for an LLM-only workload).
    """
    storage = LocalStorage(tmp_path / "store")
    compute = FakeCompute(offers=[])  # empty catalog
    settings = _settings()

    with pytest.raises(ValueError, match="fallback_to_default=False"):
        dispatcher.dispatch_new(
            workflow="prepare", pipeline_name="x", params={},
            budget_usd=10, settings=settings, storage=storage, compute=compute,
            required_vram_gb=8,
            prefer="cheapest",
            fallback_to_default=False,
        )
