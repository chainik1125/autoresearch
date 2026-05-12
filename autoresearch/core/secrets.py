"""Secrets / pod env contract.

Given a Run + Settings, build the env dict to inject into the pod at create time.
Tokens (AWS_*, ANTHROPIC_API_KEY, HF_TOKEN) come from the controller's process env,
NOT from the Run record or autoresearch.toml — they should never sit in S3.

The pod side reads these env vars and configures itself: storage backend, HF cache
root, model API access, etc.
"""

from __future__ import annotations

import os

from autoresearch.config import Settings
from autoresearch.core.run import Run


_PASSTHROUGH_SECRETS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "ANTHROPIC_API_KEY",
    "HF_TOKEN",
)


def env_for_run(
    run: Run,
    settings: Settings,
    *,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build the env dict injected into the pod for this run.

    Layout (entries with empty values are dropped):
      - Run identity: RUN_ID, CONTROLLER_PUBLIC_URL.
      - Pipeline location: PIPELINE_MODULE_PATH (path on the persistent volume).
      - Storage tier 1 (R2): AUTORESEARCH_STORAGE_*, AWS_*.
      - Storage tier 2 (HF cache): HF_HOME, HF_HUB_ENABLE_HF_TRANSFER, HF_TOKEN.
      - Validation: ANTHROPIC_API_KEY.
      - Workflow flags: AUTORESEARCH_PREFLIGHT, etc., so the pod's CLI run matches
        the controller's config.
    """
    env: dict[str, str] = {
        "RUN_ID": run.id,
        "CONTROLLER_PUBLIC_URL": settings.controller_public_url or settings.controller_url or "",
        "PIPELINE_MODULE_PATH": "/workspace/pipelines",

        "AUTORESEARCH_STORAGE": settings.storage,
        "AUTORESEARCH_STORAGE_BUCKET": settings.storage_bucket or "",
        "AUTORESEARCH_STORAGE_ENDPOINT_URL": settings.storage_endpoint_url or "",
        "AUTORESEARCH_STORAGE_REGION": settings.storage_region or "",

        "HF_HOME": "/workspace/.huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",

        "AUTORESEARCH_PREFLIGHT": "true" if settings.preflight else "false",
        "AUTORESEARCH_POSTFLIGHT": "true" if settings.postflight else "false",
        "AUTORESEARCH_SUMMARIZE_ERRORS": "true" if settings.summarize_errors else "false",
        "AUTORESEARCH_VALIDATION_MODEL": settings.validation_model,
    }

    for name in _PASSTHROUGH_SECRETS:
        if val := os.environ.get(name):
            env[name] = val

    if extra:
        env.update(extra)

    return {k: v for k, v in env.items() if v}
