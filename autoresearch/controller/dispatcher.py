"""Dispatcher — spawn a pod for a Run.

Responsible for:
  - Building the SessionSpec (env, image, GPU, network volume) from settings + Run.
  - Calling ComputeBackend.create_session and persisting the pod handle on the Run.

Shared between the MCP `start_*` tools (initial dispatch) and the supervisor
(restart after pod death).
"""

from __future__ import annotations

from autoresearch.backends.compute import ComputeBackend, SessionSpec
from autoresearch.backends.storage import StorageBackend
from autoresearch.config import Settings
from autoresearch.core import secrets
from autoresearch.core.run import Run, RunStatus


def _build_spec(
    run: Run,
    settings: Settings,
    *,
    gpu: str | None = None,
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
    return SessionSpec(
        gpu=gpu or settings.default_gpu,
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
    gpu: str | None = None,
    project_repo_url: str | None = None,
    project_repo_token: str | None = None,
    project_repo_branch: str | None = None,
) -> Run:
    """Create a fresh Run and launch its pod. Returns the persisted Run."""
    run = Run(
        workflow=workflow,
        pipeline_name=pipeline_name,
        params=params,
        budget_cap_usd=budget_usd,
    )
    run.save(storage)
    spec = _build_spec(
        run, settings, gpu=gpu,
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
    spec = _build_spec(run, settings)
    handle = compute.create_session(spec)
    run.pod_handle = handle.id
    run.save(storage)
    return run
