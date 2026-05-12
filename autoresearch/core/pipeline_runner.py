"""Pipeline runner — the deterministic FSM that executes a workflow.

State machine:

    queued ─┬─▶ loading ──▶ running ──▶ validating ──▶ completed
            │                                              │
            └────────────────────────────────────────▶ failed

Each transition writes a checkpoint to storage. On pod death, the supervisor
spawns a new pod with the same persistent volume; the new runner reads the
checkpoint and resumes from the recorded step.

Resume semantics:
  - queued / no checkpoint:  run preflight, then pipeline, then postflight.
  - loading:                 (preflight already ran) run pipeline, then postflight.
  - running:                 we crashed mid-pipeline. Re-run pipeline; the
                             pipeline is responsible for resuming from its own
                             on-disk state in `workspace`. Then postflight.
  - validating:              pipeline completed, result is in storage. Re-run
                             postflight (idempotent — duplicate findings are
                             cheap and ordered).
  - completed:               nothing to do; return stored result.

The runner itself contains no LLM calls. Preflight/postflight prompts come from
the workflow via `WorkflowHooks`; the runner just calls them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from autoresearch.backends.storage import StorageBackend
from autoresearch.core import checkpoint, findings
from autoresearch.core.findings import FindingType
from autoresearch.core.pipeline import Pipeline
from autoresearch.core.run import Run, RunStatus


@dataclass
class WorkflowHooks:
    """Workflow-specific hooks invoked by the runner.

    `preflight` and `postflight` receive (run, pipeline, result_or_none) and
    return text to write as an observation finding. If the function is None,
    the step is skipped.

    `on_long_call_start` / `on_long_call_end` let the heartbeat module flip its
    stale-threshold flag. Optional.
    """

    preflight: Callable[[Run, Pipeline], str] | None = None
    postflight: Callable[[Run, Pipeline, dict[str, Any]], str] | None = None
    on_long_call_start: Callable[[Run], None] | None = None
    on_long_call_end: Callable[[Run], None] | None = None
    # Steps captured by Run.last_error on failure are also written as a finding.
    summarize_error: Callable[[Run, BaseException], str] | None = None


@dataclass
class RunnerContext:
    storage: StorageBackend
    workspace: Path
    hooks: WorkflowHooks = field(default_factory=WorkflowHooks)


def _set_status(storage: StorageBackend, run: Run, status: RunStatus) -> Run:
    fresh = Run.load(storage, run.id)
    fresh.status = status
    fresh.save(storage)
    return fresh


def _write_result(storage: StorageBackend, run: Run, result: dict[str, Any]) -> None:
    storage.write(run.result_key, json.dumps(result).encode("utf-8"))


def _read_result(storage: StorageBackend, run: Run) -> dict[str, Any]:
    return json.loads(storage.read(run.result_key).decode("utf-8"))


def run_pipeline(run: Run, pipeline: Pipeline, ctx: RunnerContext) -> dict[str, Any]:
    """Execute the pipeline through the FSM. Returns the final result dict.

    Idempotent: calling twice on a `completed` run is a no-op (returns stored result).
    Resumable: crashing at any point and restarting picks up from the latest checkpoint.
    """
    cp = checkpoint.load(ctx.storage, run)
    step = cp.step if cp else "queued"

    try:
        # Phase 1: preflight.
        if step == "queued":
            run = _set_status(ctx.storage, run, RunStatus.LOADING)
            if ctx.hooks.preflight is not None:
                text = ctx.hooks.preflight(run, pipeline)
                findings.append(ctx.storage, run, FindingType.OBSERVATION, text)
            checkpoint.write(ctx.storage, run, step="loading")
            step = "loading"

        # Phase 2: pipeline run.
        if step in ("loading", "running"):
            run = _set_status(ctx.storage, run, RunStatus.RUNNING)
            checkpoint.write(ctx.storage, run, step="running")
            if ctx.hooks.on_long_call_start is not None:
                ctx.hooks.on_long_call_start(run)
            try:
                result = pipeline.run(
                    params=run.params,
                    workspace=ctx.workspace,
                    storage=ctx.storage,
                )
            finally:
                if ctx.hooks.on_long_call_end is not None:
                    ctx.hooks.on_long_call_end(run)
            _write_result(ctx.storage, run, result)
            findings.append(
                ctx.storage, run, FindingType.RESULT, json.dumps(result, indent=2)
            )
            checkpoint.write(ctx.storage, run, step="validating", partial_result=result)
            step = "validating"

        # Phase 3: postflight.
        if step == "validating":
            run = _set_status(ctx.storage, run, RunStatus.VALIDATING)
            result = _read_result(ctx.storage, run)
            if ctx.hooks.postflight is not None:
                text = ctx.hooks.postflight(run, pipeline, result)
                findings.append(ctx.storage, run, FindingType.OBSERVATION, text)
            checkpoint.write(ctx.storage, run, step="completed")
            step = "completed"

        # Phase 4: done.
        run = _set_status(ctx.storage, run, RunStatus.COMPLETED)
        return _read_result(ctx.storage, run)

    except BaseException as exc:
        # Best-effort error capture; do not swallow.
        # Critically: do NOT overwrite the checkpoint. Its `step` field marks where
        # the runner was when the error happened — leaving it there is what makes
        # resume work (the supervisor/CLI just calls run_pipeline again and the FSM
        # re-enters the same phase).
        try:
            summary = (
                ctx.hooks.summarize_error(run, exc)
                if ctx.hooks.summarize_error is not None
                else f"{type(exc).__name__}: {exc}"
            )
        except Exception:  # noqa: BLE001 -- summarize is best-effort
            summary = f"{type(exc).__name__}: {exc}"
        try:
            findings.append(ctx.storage, run, FindingType.ERROR, summary)
            fresh = Run.load(ctx.storage, run.id)
            fresh.status = RunStatus.FAILED
            fresh.last_error = summary
            fresh.save(ctx.storage)
        except Exception:  # noqa: BLE001 -- never let cleanup hide the real error
            pass
        raise
