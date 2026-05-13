"""Settings — env + autoresearch.toml + defaults.

Precedence (highest first): explicit constructor args > env vars (prefixed
AUTORESEARCH_) > autoresearch.toml in cwd > field defaults.

Secrets (API keys, AWS credentials) come from env only. Toml is for project shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AUTORESEARCH_",
        env_file=".env",
        toml_file="autoresearch.toml",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

    # --- Storage ---
    storage: Literal["local", "s3"] = "local"
    storage_root: str = "./local-storage"
    storage_bucket: str | None = None
    storage_endpoint_url: str | None = None
    storage_region: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    # --- Compute ---
    compute: Literal["runpod", "modal", "local"] = "local"
    default_gpu: str = "A40"
    runpod_api_key: str | None = None
    runpod_template_id: str | None = None
    runpod_network_volume_id: str | None = None
    runpod_data_center: str | None = None
    runpod_container_disk_gb: int = 50
    runpod_default_image: str = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
    runpod_container_registry_auth_id: str | None = None

    # --- Models / validation ---
    anthropic_api_key: str | None = None
    validation_model: str = "claude-haiku-4-5-20251001"
    preflight: bool = True
    postflight: bool = True
    summarize_errors: bool = True

    # --- Workflow defaults ---
    # Budget cap per dispatched Run, in USD. Override per-call via the
    # `budget_usd` arg to start_transfer / CLI `--budget`. The cap covers
    # all Anthropic spend accumulated via `core.budget.add_spend` for the
    # life of the run — see budget.py for the spend-tracking semantics
    # (what IS and isn't tracked yet).
    default_budget_usd: float = 30.0
    pipeline_module_path: str = "pipelines"

    # --- Controller ---
    controller_url: str | None = None
    controller_public_url: str | None = None
    controller_host: str = "0.0.0.0"
    controller_port: int = 8000
    base_image_tag: str | None = None
    project_repo_url: str | None = None       # cloned onto the pod's persistent volume on first boot
    project_repo_token: str | None = None     # GitHub PAT for private-repo cloning (optional)
    project_repo_branch: str | None = None    # specific branch to clone; default = remote's HEAD
    supervisor_poll_seconds: float = 30.0
    supervisor_stale_minutes: int = 5         # heartbeat staleness threshold (short tools)
    supervisor_long_call_stale_hours: int = 2 # threshold when in_long_pipeline_call=true

    @classmethod
    def load(cls) -> Settings:
        """Construct Settings from env + autoresearch.toml + defaults."""
        return cls()


def build_storage(settings: Settings):
    """Construct the StorageBackend the settings select."""
    from autoresearch.backends.storage import LocalStorage, S3Storage

    if settings.storage == "local":
        return LocalStorage(settings.storage_root)
    if settings.storage == "s3":
        if not settings.storage_bucket:
            raise ValueError("storage=s3 requires storage_bucket")
        return S3Storage(
            bucket=settings.storage_bucket,
            endpoint_url=settings.storage_endpoint_url,
            region_name=settings.storage_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
        )
    raise ValueError(f"unknown storage backend: {settings.storage}")


def build_compute(settings: Settings):
    """Construct the ComputeBackend the settings select."""
    if settings.compute == "local":
        return None  # local dev path runs the pipeline in-process; no compute backend needed
    if settings.compute == "runpod":
        if not settings.runpod_api_key:
            raise ValueError("compute=runpod requires runpod_api_key (env: AUTORESEARCH_RUNPOD_API_KEY)")
        from autoresearch.backends.compute.runpod import RunPodCompute

        return RunPodCompute(api_key=settings.runpod_api_key)
    if settings.compute == "modal":
        from autoresearch.backends.compute.modal import ModalCompute

        return ModalCompute()
    raise ValueError(f"unknown compute backend: {settings.compute}")


def build_model_client(settings: Settings):
    """Construct the validation ModelClient, or None if validators are all off."""
    if not (settings.preflight or settings.postflight or settings.summarize_errors):
        return None
    if not settings.anthropic_api_key:
        # No key — caller may still pass model_client=None and disable validators.
        return None
    from autoresearch.backends.models.anthropic import AnthropicClient

    return AnthropicClient(
        api_key=settings.anthropic_api_key,
        model=settings.validation_model,
    )
