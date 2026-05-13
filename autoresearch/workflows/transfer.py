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
from autoresearch.core import findings as findings_mod
from autoresearch.core import validation
from autoresearch.core.findings import FindingType
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


def _build_experiment_report(
    run: Run, pipeline_result: dict[str, Any] | None
) -> str:
    """Markdown experiment-report body. Headline carries the spend total.

    Designed to be readable both as a finding (via `list_findings`) and as a
    standalone `experiment_summary.md` file written into the workspace.
    """
    llm_spent = run.budget_spent_usd
    cap = run.budget_cap_usd
    under = (cap <= 0) or (llm_spent <= cap)

    compute_estimate: float | None = None
    elapsed_s: float | None = None
    if isinstance(pipeline_result, dict):
        elapsed_s = pipeline_result.get("elapsed_seconds")
        if "actual_run_cost_usd" in pipeline_result:
            compute_estimate = pipeline_result.get("actual_run_cost_usd")
        elif elapsed_s and "cost_per_hour" in (run.params or {}):
            compute_estimate = (elapsed_s / 3600.0) * float(run.params["cost_per_hour"])

    total_est = llm_spent + (compute_estimate or 0.0)
    compute_str = (
        f"${compute_estimate:.4f} (elapsed {elapsed_s:.0f}s × cost_per_hour)"
        if compute_estimate is not None else "not tracked"
    )
    budget_state = "under" if under else "**EXCEEDED**"

    target = (run.params or {}).get("target_model", "—")

    lines: list[str] = [
        f"# /transfer {run.id} — {run.status.value.upper()}",
        "",
        f"**Total spend (est.): ${total_est:.4f}** "
        f"— LLM ${llm_spent:.4f} + compute {compute_str}. "
        f"Budget cap ${cap:.2f} ({budget_state}).",
        "",
        "## Pipeline",
        f"- name: `{run.pipeline_name}`",
        f"- target_model: `{target}`",
        f"- params: `{json.dumps(run.params, default=str)}`",
    ]
    if run.parent_run_id:
        lines.append(f"- parent_run_id: `{run.parent_run_id}`")
    if run.pod_handle:
        lines.append(f"- pod_handle: `{run.pod_handle}`")

    if isinstance(pipeline_result, dict):
        lines.extend(["", "## Result", "```json", json.dumps(pipeline_result, indent=2, default=str), "```"])
    elif run.last_error:
        lines.extend(["", "## Error", "```", run.last_error, "```"])

    lines.extend([
        "",
        "## Spend accounting",
        f"- Tracked LLM spend: `${llm_spent:.4f}` (validators + summarize_run)",
        f"- Compute spend est: `{compute_str}`",
        f"- **Total est: `${total_est:.4f}`** vs cap `${cap:.2f}` ({budget_state})",
        "",
        "_Compute pod-hours are not yet tracked in-flight — the estimate above is",
        "post-hoc from the pipeline's `elapsed_seconds` × `cost_per_hour`, not from",
        "RunPod billing. Hardware-advisor + future agent spend isn't aggregated yet._",
        "See `notes/ideas.md` → v2 TODOs (Budget accounting).",
    ])
    return "\n".join(lines)


def _write_spend_summary(
    storage: StorageBackend,
    run_id: str,
    pipeline_result: dict[str, Any] | None,
    workspace: Path | None = None,
) -> None:
    """Emit the markdown experiment report at terminal state (COMPLETED/FAILED).

    Two artifacts:
      1. An OBSERVATION finding with the full markdown body — readable via
         `list_findings(run_id)` from anywhere.
      2. `<workspace>/experiment_summary.md` on the persistent volume —
         co-located with training.log + SAE outputs for the user to grab
         when they SSH in or pull artifacts. Best-effort; absence of
         workspace just skips the file write.

    Both are best-effort — a failure here must never mask the real run result.
    """
    try:
        fresh = Run.load(storage, run_id)
    except Exception:  # noqa: BLE001
        return
    body = _build_experiment_report(fresh, pipeline_result)
    try:
        findings_mod.append(storage, fresh, FindingType.OBSERVATION, body)
    except Exception:  # noqa: BLE001
        pass
    if workspace is not None:
        try:
            (Path(workspace) / "experiment_summary.md").write_text(body)
        except Exception:  # noqa: BLE001
            pass


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

    result: dict[str, Any] | None = None
    try:
        hooks = WorkflowHooks(
            preflight=_preflight_hook(model_client, storage) if preflight else None,
            postflight=_postflight_hook(model_client, storage) if postflight else None,
            summarize_error=_error_hook(model_client, storage) if summarize_errors else None,
            on_long_call_start=long_call_start,
            on_long_call_end=long_call_end,
        )
        ctx = RunnerContext(storage=storage, workspace=workspace, hooks=hooks)
        result = run_pipeline(run, pipeline, ctx)
        return result
    finally:
        # Lightweight bolt-on: always emit a spend_summary finding at the
        # terminal state of a /transfer (success OR failure). Wrapped in
        # try/except so a bad spend write never masks the real result/error.
        try:
            _write_spend_summary(storage, run.id, result, workspace=workspace)
        except Exception:  # noqa: BLE001 -- spend summary is best-effort
            pass
        if hb is not None:
            hb.stop()
            hb.join(timeout=2.0)
