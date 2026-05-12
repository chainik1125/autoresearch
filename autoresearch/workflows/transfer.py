"""TRANSFER workflow — run an existing pipeline against a different model.

Use case: the user has a measurement that works on model A; they want the same
measurement on model B. The pipeline code is unchanged; only `params["target_model"]`
(and optionally `params["source_model"]` for comparison) differs across runs.

The workflow wraps the pipeline call with two bounded LLM validation steps:
  - preflight: "does this model name look right? is the GPU choice sensible?"
  - postflight: "given the source result (if any) and the target result, comment
    on plausibility."

If `ModelClient` is None, both validators are skipped silently — useful for local
testing without an API key.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

from autoresearch.backends.models.base import ModelClient
from autoresearch.backends.storage import StorageBackend
from autoresearch.backends.storage.base import KeyNotFound
from autoresearch.core import validation
from autoresearch.core.heartbeat import HeartbeatWriter
from autoresearch.core.pipeline import Pipeline
from autoresearch.core.pipeline_runner import RunnerContext, WorkflowHooks, run_pipeline
from autoresearch.core.run import Run


PREFLIGHT_SYSTEM = """You are a research-engineering assistant validating a measurement run.
Be terse. If everything looks fine, say "looks good" in one line. Otherwise list
specific concerns in <=3 short bullets. Do not speculate about results — you are
checking the request, not predicting the outcome."""

PREFLIGHT_USER_TEMPLATE = """About to run pipeline `{pipeline_name}` on model `{target_model}`.
{source_clause}

Pipeline metadata:
  - required_gpu: {required_gpu}
  - estimated_minutes: {estimated_minutes}

Other params: {other_params}

Flag anything that looks wrong: implausible model name, GPU/model size mismatch,
missing required params, etc."""


POSTFLIGHT_SYSTEM = """You are a research-engineering assistant validating that a
measurement produced a plausible result. Be terse. Focus on: order-of-magnitude
sanity, consistency with the source-model result (if provided), and any obvious
red flags (NaNs, zero variance, suspicious uniformity). Do not over-interpret —
the researcher will draw conclusions; you are just flagging things to check."""

POSTFLIGHT_USER_TEMPLATE = """Pipeline `{pipeline_name}` finished.

Target model: {target_model}
Target result:
{target_result}

{source_block}

Comment on plausibility in <=5 short bullets. End with one line: "PLAUSIBLE",
"SUSPICIOUS", or "INVALID"."""


ERROR_SUMMARY_SYSTEM = """You summarize Python tracebacks for a research engineer.
Output 2-3 sentences max: what failed, the likely cause, and what to check first.
Use plain language, not jargon."""


def _preflight_hook(client: ModelClient, storage: StorageBackend):
    def hook(run: Run, pipeline: Pipeline) -> str:
        params = dict(run.params)
        target = params.pop("target_model", "<missing>")
        source = params.pop("source_model", None)
        source_clause = (
            f"Comparing against source model `{source}` (must already have a stored result)."
            if source
            else "No source model for comparison."
        )
        user = PREFLIGHT_USER_TEMPLATE.format(
            pipeline_name=pipeline.name,
            target_model=target,
            source_clause=source_clause,
            required_gpu=pipeline.required_gpu,
            estimated_minutes=pipeline.estimated_minutes,
            other_params=json.dumps(params, default=str) or "(none)",
        )
        return validation.run_validator(
            storage, run, client, system=PREFLIGHT_SYSTEM, user=user, max_tokens=400
        )

    return hook


def _postflight_hook(client: ModelClient, storage: StorageBackend):
    def hook(run: Run, pipeline: Pipeline, result: dict[str, Any]) -> str:
        target = run.params.get("target_model", "<missing>")
        source = run.params.get("source_model")
        source_block = "No source-model result available for comparison."
        if source:
            source_key = f"baselines/{pipeline.name}/{source}.json"
            try:
                source_result = json.loads(storage.read(source_key).decode("utf-8"))
                source_block = (
                    f"Source model: {source}\nSource result:\n{json.dumps(source_result, indent=2)}"
                )
            except KeyNotFound:
                source_block = (
                    f"Source model `{source}` was named but no baseline found at"
                    f" `{source_key}`. Compare against target alone."
                )
        user = POSTFLIGHT_USER_TEMPLATE.format(
            pipeline_name=pipeline.name,
            target_model=target,
            target_result=json.dumps(result, indent=2),
            source_block=source_block,
        )
        return validation.run_validator(
            storage, run, client, system=POSTFLIGHT_SYSTEM, user=user, max_tokens=600
        )

    return hook


def _error_hook(client: ModelClient, storage: StorageBackend):
    def hook(run: Run, exc: BaseException) -> str:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return validation.run_validator(
            storage,
            run,
            client,
            system=ERROR_SUMMARY_SYSTEM,
            user=f"Pipeline `{run.pipeline_name}` failed.\n\nTraceback:\n```\n{tb}\n```",
            max_tokens=400,
        )

    return hook


def transfer(
    run: Run,
    pipeline: Pipeline,
    *,
    storage: StorageBackend,
    workspace: Path,
    model_client: ModelClient | None = None,
    preflight: bool = True,
    postflight: bool = True,
    summarize_errors: bool = True,
    heartbeat: bool = False,
) -> dict[str, Any]:
    """Run the TRANSFER workflow for `run` against `pipeline`.

    `model_client` is required for any enabled validator; pass None and set the
    `preflight`/`postflight`/`summarize_errors` flags to False for an LLM-free
    smoke test.

    `heartbeat=True` spawns a `HeartbeatWriter` thread that writes the run's
    heartbeat key every 30s and flips an "in long call" flag around the user
    pipeline's `run()`. Enable for pod-side execution; leave off for local.
    """
    if model_client is None and (preflight or postflight or summarize_errors):
        raise ValueError(
            "model_client is required when any validator is enabled; "
            "pass model_client=AnthropicClient(...) or disable all three flags."
        )

    hb: HeartbeatWriter | None = None
    long_call_start = None
    long_call_end = None
    if heartbeat:
        hb = HeartbeatWriter(storage, run)
        hb.start()
        long_call_start = lambda _r: hb.set_long_call(True)  # noqa: E731
        long_call_end = lambda _r: hb.set_long_call(False)   # noqa: E731

    try:
        hooks = WorkflowHooks(
            preflight=_preflight_hook(model_client, storage) if preflight else None,
            postflight=_postflight_hook(model_client, storage) if postflight else None,
            summarize_error=_error_hook(model_client, storage) if summarize_errors else None,
            on_long_call_start=long_call_start,
            on_long_call_end=long_call_end,
        )
        ctx = RunnerContext(storage=storage, workspace=workspace, hooks=hooks)
        return run_pipeline(run, pipeline, ctx)
    finally:
        if hb is not None:
            hb.stop()
            hb.join(timeout=2.0)
