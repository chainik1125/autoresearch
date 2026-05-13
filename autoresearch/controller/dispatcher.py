"""Dispatcher — spawn a pod for a Run.

Responsible for:
  - Building the SessionSpec (env, image, GPU, network volume) from settings + Run.
  - Calling ComputeBackend.create_session and persisting the pod handle on the Run.

Shared between the MCP `start_*` tools (initial dispatch) and the supervisor
(restart after pod death).
"""

from __future__ import annotations

from autoresearch.backends.compute import ComputeBackend, SessionSpec
from autoresearch.backends.models.base import ModelClient
from autoresearch.backends.storage import StorageBackend
from autoresearch.config import Settings
from autoresearch.core import secrets
from autoresearch.core.hardware import recommend as recommend_hardware
from autoresearch.core.run import Run, RunStatus


def _resolve_gpu(
    *,
    explicit_gpu: str | list[str] | None,
    required_vram_gb: int | None,
    settings: Settings,
    compute: ComputeBackend,
    model_client: ModelClient | None = None,
    intent: str | None = None,
    pipeline_name: str = "<unknown>",
    estimated_minutes: int = 0,
) -> str | list[str]:
    """Pick the GPU spec for a dispatch.

    Precedence:
      1. Caller's explicit `gpu` (string or list) — bypasses auto-selection.
      2. Pipeline's `required_vram_gb` → query backend's offers + run
         `core/hardware.py:recommend()` (LLM-advised if `model_client` and
         `intent` are supplied, deterministic otherwise).
      3. `settings.default_gpu` — last-resort literal.

    Thin dispatcher-side glue; the actual selection logic lives in
    `core/hardware.py` so heuristics and prompts evolve independently.
    """
    if explicit_gpu is not None:
        return explicit_gpu
    if required_vram_gb is not None:
        try:
            offers = compute.list_gpu_offers(settings.runpod_data_center)
        except Exception:  # noqa: BLE001 -- never block dispatch on advisor failure
            offers = []
        rec = recommend_hardware(
            offers,
            required_vram_gb=required_vram_gb,
            data_center_id=settings.runpod_data_center,
            intent=intent,
            pipeline_name=pipeline_name,
            estimated_minutes=estimated_minutes,
            client=model_client,
        )
        if rec.picks:
            return rec.picks
    return settings.default_gpu


def _build_spec(
    run: Run,
    settings: Settings,
    *,
    gpu: str | list[str] | None = None,
    required_vram_gb: int | None = None,
    compute: ComputeBackend | None = None,
    model_client: ModelClient | None = None,
    intent: str | None = None,
    pipeline_name: str = "<unknown>",
    estimated_minutes: int = 0,
    project_repo_url: str | None = None,
    project_repo_token: str | None = None,
    project_repo_branch: str | None = None,
) -> SessionSpec:
    if not settings.runpod_network_volume_id:
        raise ValueError(
            "runpod_network_volume_id is required for dispatch — set it in autoresearch.toml"
        )
    env = secrets.env_for_run(
        run,
        settings,
        project_repo_token=project_repo_token,
        project_repo_branch=project_repo_branch,
    )
    # Per-dispatch project_repo_url override wins over the controller's setting.
    repo_url = project_repo_url or settings.project_repo_url
    if repo_url:
        env["PROJECT_REPO_URL"] = repo_url

    # Hardware selection — gpu may be a string, a preference-ordered list, or
    # auto-resolved from required_vram_gb against the backend's catalog.
    resolved_gpu: str | list[str]
    if compute is not None:
        resolved_gpu = _resolve_gpu(
            explicit_gpu=gpu,
            required_vram_gb=required_vram_gb,
            settings=settings,
            compute=compute,
            model_client=model_client,
            intent=intent,
            pipeline_name=pipeline_name,
            estimated_minutes=estimated_minutes,
        )
    else:
        resolved_gpu = gpu or settings.default_gpu

    return SessionSpec(
        gpu=resolved_gpu,
        image=settings.runpod_default_image,
        network_volume_id=settings.runpod_network_volume_id,
        env=env,
        name=f"autoresearch-{run.id}",
        container_disk_gb=settings.runpod_container_disk_gb,
        container_registry_auth_id=settings.runpod_container_registry_auth_id,
    )


def dispatch_new(
    *,
    workflow: str,
    pipeline_name: str,
    params: dict,
    budget_usd: float,
    settings: Settings,
    storage: StorageBackend,
    compute: ComputeBackend,
    gpu: str | list[str] | None = None,
    required_vram_gb: int | None = None,
    model_client: ModelClient | None = None,
    intent: str | None = None,
    estimated_minutes: int = 0,
    project_repo_url: str | None = None,
    project_repo_token: str | None = None,
    project_repo_branch: str | None = None,
) -> Run:
    """Create a fresh Run and launch its pod. Returns the persisted Run.

    GPU selection (in order):
      - `gpu` (string or list) — caller's explicit choice.
      - `required_vram_gb` — auto-select via `core/hardware.py:recommend()`.
        If `model_client` and `intent` are supplied, the LLM advisor is used
        and its recommendation is honored; otherwise the deterministic
        fastest-least-complicated heuristic.
      - `settings.default_gpu` — last-resort literal.
    """
    run = Run(
        workflow=workflow,
        pipeline_name=pipeline_name,
        params=params,
        budget_cap_usd=budget_usd,
    )
    run.save(storage)
    spec = _build_spec(
        run, settings,
        gpu=gpu,
        required_vram_gb=required_vram_gb,
        compute=compute,
        model_client=model_client,
        intent=intent,
        pipeline_name=pipeline_name,
        estimated_minutes=estimated_minutes,
        project_repo_url=project_repo_url,
        project_repo_token=project_repo_token,
        project_repo_branch=project_repo_branch,
    )
    handle = compute.create_session(spec)
    run.pod_handle = handle.id
    run.status = RunStatus.QUEUED
    run.save(storage)
    return run


def redispatch(
    run: Run,
    *,
    settings: Settings,
    storage: StorageBackend,
    compute: ComputeBackend,
) -> Run:
    """Re-launch a pod for an existing Run after death/preemption.

    Best-effort terminates the old pod, then creates a new one attached to the
    same network volume. The runner on the new pod reads the checkpoint from
    storage and resumes from where the previous pod crashed.
    """
    if run.pod_handle:
        try:
            compute.terminate_session(run.pod_handle)
        except Exception:  # noqa: BLE001 -- old pod may already be gone
            pass
    # On redispatch we use the deterministic selector (no advisor LLM in the
    # supervisor's restart loop). `compute` is passed so the selector can
    # query inventory; without it, we fall back to `settings.default_gpu`.
    spec = _build_spec(run, settings, compute=compute)
    handle = compute.create_session(spec)
    run.pod_handle = handle.id
    run.save(storage)
    return run
