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
    workflow: str = "transfer",
    estimated_minutes: int = 0,
    prefer: str = "fastest_least_complicated",
    fallback_to_default: bool = True,
) -> str | list[str]:
    """Pick the GPU spec for a dispatch.

    Precedence:
      1. Caller's explicit `gpu` (string or list) — bypasses auto-selection.
      2. Pipeline's `required_vram_gb` → query backend's offers + run
         `core/hardware.py:recommend()` (LLM-advised if `model_client` and
         `intent` are supplied, deterministic otherwise).
      3. `settings.default_gpu` — last-resort literal. Suppressed via
         `fallback_to_default=False` (used by agent workflows, where the
         "default" H100 would be a 1000x cost overrun for a 30-second LLM
         call — better to fail loudly than silently dispatch on a $5/hr GPU).

    `prefer` selects the ranking heuristic. Pipeline runs use
    "fastest_least_complicated" (the default); agent workflows
    (prepare/mechanic/postflight) pass "cheapest" since they're LLM/network
    calls, not GPU compute.

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
            workflow=workflow,
            estimated_minutes=estimated_minutes,
            client=model_client,
            prefer=prefer,  # type: ignore[arg-type]
        )
        if rec.picks:
            return rec.picks
    if not fallback_to_default:
        raise ValueError(
            "GPU resolution failed and fallback_to_default=False — refusing to "
            "silently dispatch on settings.default_gpu. Catalog query may have "
            "errored, or no in-stock GPU met required_vram_gb. Pass `gpu=` "
            "explicitly, or check RunPod inventory."
        )
    return settings.default_gpu


_DEFAULT_AGENT_CPU_FLAVORS = ("cpu3c", "cpu5c")
_DEFAULT_AGENT_VCPU_COUNT = 4


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
    workflow: str = "transfer",
    estimated_minutes: int = 0,
    project_repo_url: str | None = None,
    project_repo_token: str | None = None,
    project_repo_branch: str | None = None,
    prefer: str = "fastest_least_complicated",
    fallback_to_default: bool = True,
    compute_type: str = "GPU",
    cpu_flavors: list[str] | None = None,
    vcpu_count: int | None = None,
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

    # CPU pod path — agent workflows (prepare/mechanic/postflight) default
    # here. RunPod CPU pods are ~$0.05/hr; the cheapest GPU pod is ~$0.17/hr,
    # and the LLM-only workloads have no use for the GPU at all.
    if compute_type == "CPU":
        flavors = list(cpu_flavors) if cpu_flavors else list(_DEFAULT_AGENT_CPU_FLAVORS)
        return SessionSpec(
            compute_type="CPU",
            cpu_flavors=flavors,
            vcpu_count=vcpu_count or _DEFAULT_AGENT_VCPU_COUNT,
            gpu=[],
            image=settings.runpod_default_image,
            network_volume_id=settings.runpod_network_volume_id,
            env=env,
            name=f"autoresearch-{run.id}",
            container_disk_gb=settings.runpod_container_disk_gb,
            container_registry_auth_id=settings.runpod_container_registry_auth_id,
        )

    # GPU pod path — pipeline runs, plus any agent workflow that explicitly
    # asked for a GPU via compute_type='GPU' override.
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
            workflow=workflow,
            estimated_minutes=estimated_minutes,
            prefer=prefer,
            fallback_to_default=fallback_to_default,
        )
    else:
        if gpu is None and not fallback_to_default:
            raise ValueError(
                "GPU not specified and fallback_to_default=False (no compute "
                "backend to query for catalog either). Pass `gpu=` explicitly."
            )
        resolved_gpu = gpu or settings.default_gpu

    return SessionSpec(
        compute_type="GPU",
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
    parent_run_id: str | None = None,
    prefer: str = "fastest_least_complicated",
    fallback_to_default: bool = True,
    compute_type: str = "GPU",
    cpu_flavors: list[str] | None = None,
    vcpu_count: int | None = None,
) -> Run:
    """Create a fresh Run and launch its pod. Returns the persisted Run.

    GPU selection (in order):
      - `gpu` (string or list) — caller's explicit choice.
      - `required_vram_gb` — auto-select via `core/hardware.py:recommend()`.
        If `model_client` and `intent` are supplied, the LLM advisor is used
        and its recommendation is honored; otherwise the deterministic
        fastest-least-complicated heuristic.
      - `settings.default_gpu` — last-resort literal.

    `parent_run_id` is recorded on the new Run when an agent (prep /
    mechanic / postflight, or a recursive /transfer call) is spawning
    downstream work. Lets the run-tree be reconstructed for introspection.
    """
    run = Run(
        workflow=workflow,
        pipeline_name=pipeline_name,
        params=params,
        budget_cap_usd=budget_usd,
        parent_run_id=parent_run_id,
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
        workflow=workflow,
        estimated_minutes=estimated_minutes,
        project_repo_url=project_repo_url,
        project_repo_token=project_repo_token,
        project_repo_branch=project_repo_branch,
        prefer=prefer,
        fallback_to_default=fallback_to_default,
        compute_type=compute_type,
        cpu_flavors=cpu_flavors,
        vcpu_count=vcpu_count,
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
