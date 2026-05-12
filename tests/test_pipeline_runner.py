"""Pipeline runner FSM tests.

Uses a fake Pipeline and fake hook functions — no Anthropic API, no real GPUs.
Exercises the resume-from-checkpoint semantics that are the most important
correctness property of the runner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from autoresearch.backends.storage import LocalStorage, StorageBackend
from autoresearch.core import checkpoint, findings
from autoresearch.core.findings import FindingType
from autoresearch.core.pipeline_runner import RunnerContext, WorkflowHooks, run_pipeline
from autoresearch.core.run import Run, RunStatus


class StubPipeline:
    name = "stub"
    required_gpu = "A40"
    estimated_minutes = 1

    def __init__(self, *, result: dict[str, Any] | None = None, fail_times: int = 0) -> None:
        self.result = result or {"score": 0.5}
        self.fail_times = fail_times
        self.calls = 0

    def run(self, *, params, workspace, storage):
        self.calls += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError(f"intentional failure (calls={self.calls})")
        return self.result


@pytest.fixture
def storage(tmp_path: Path) -> StorageBackend:
    return LocalStorage(tmp_path / "store")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


def make_run(storage: StorageBackend) -> Run:
    run = Run(workflow="transfer", pipeline_name="stub", params={"target_model": "Q-32B"})
    run.save(storage)
    return run


def collect_hook_calls() -> tuple[list[tuple[str, tuple[Any, ...]]], WorkflowHooks]:
    calls: list[tuple[str, tuple[Any, ...]]] = []

    def pre(run, pipeline):
        calls.append(("pre", (run.id, pipeline.name)))
        return "preflight ok"

    def post(run, pipeline, result):
        calls.append(("post", (run.id, pipeline.name, result)))
        return "postflight ok"

    def err(run, exc):
        calls.append(("err", (run.id, type(exc).__name__)))
        return f"{type(exc).__name__}: {exc}"

    return calls, WorkflowHooks(preflight=pre, postflight=post, summarize_error=err)


def test_happy_path(storage: StorageBackend, workspace: Path) -> None:
    run = make_run(storage)
    pipeline = StubPipeline(result={"score": 0.7})
    calls, hooks = collect_hook_calls()
    ctx = RunnerContext(storage=storage, workspace=workspace, hooks=hooks)

    result = run_pipeline(run, pipeline, ctx)

    assert result == {"score": 0.7}
    assert pipeline.calls == 1

    final = Run.load(storage, run.id)
    assert final.status == RunStatus.COMPLETED

    cp = checkpoint.load(storage, run)
    assert cp is not None and cp.step == "completed"

    fs = findings.list_findings(storage, run)
    types = [f.type for f in fs]
    assert types == [FindingType.OBSERVATION, FindingType.RESULT, FindingType.OBSERVATION]
    assert fs[0].body == "preflight ok"
    assert fs[2].body == "postflight ok"

    assert [name for name, _ in calls] == ["pre", "post"]


def test_resume_from_loading_skips_preflight(storage: StorageBackend, workspace: Path) -> None:
    run = make_run(storage)
    pipeline = StubPipeline()
    calls, hooks = collect_hook_calls()
    ctx = RunnerContext(storage=storage, workspace=workspace, hooks=hooks)

    # Simulate: preflight ran in a previous incarnation.
    checkpoint.write(storage, run, step="loading")

    run_pipeline(run, pipeline, ctx)

    # Preflight should NOT be called again on resume from loading.
    assert [name for name, _ in calls] == ["post"]
    assert pipeline.calls == 1


def test_resume_from_running_reruns_pipeline(storage: StorageBackend, workspace: Path) -> None:
    """If we crashed mid-pipeline, resume means: call pipeline.run() again.

    The pipeline is responsible for resuming from its own workspace state."""
    run = make_run(storage)
    pipeline = StubPipeline()
    calls, hooks = collect_hook_calls()
    ctx = RunnerContext(storage=storage, workspace=workspace, hooks=hooks)

    checkpoint.write(storage, run, step="running")

    run_pipeline(run, pipeline, ctx)

    assert pipeline.calls == 1  # called once on this incarnation
    assert [name for name, _ in calls] == ["post"]  # no preflight on resume


def test_resume_from_validating_skips_pipeline(storage: StorageBackend, workspace: Path) -> None:
    """If we crashed during postflight, resume reads the stored result and re-runs postflight."""
    run = make_run(storage)
    pipeline = StubPipeline(result={"score": 0.42})
    calls, hooks = collect_hook_calls()
    ctx = RunnerContext(storage=storage, workspace=workspace, hooks=hooks)

    import json as _json

    storage.write(run.result_key, _json.dumps({"score": 0.42}).encode())
    checkpoint.write(storage, run, step="validating", partial_result={"score": 0.42})

    result = run_pipeline(run, pipeline, ctx)

    assert result == {"score": 0.42}
    assert pipeline.calls == 0  # NOT re-run
    assert [name for name, _ in calls] == ["post"]


def test_resume_from_completed_is_noop(storage: StorageBackend, workspace: Path) -> None:
    run = make_run(storage)
    pipeline = StubPipeline(result={"final": True})
    calls, hooks = collect_hook_calls()
    ctx = RunnerContext(storage=storage, workspace=workspace, hooks=hooks)

    import json as _json

    storage.write(run.result_key, _json.dumps({"final": True}).encode())
    checkpoint.write(storage, run, step="completed")

    result = run_pipeline(run, pipeline, ctx)

    assert result == {"final": True}
    assert pipeline.calls == 0
    assert calls == []  # no hooks invoked


def test_failure_records_error_and_reraises(storage: StorageBackend, workspace: Path) -> None:
    run = make_run(storage)
    pipeline = StubPipeline(fail_times=99)  # always fail
    calls, hooks = collect_hook_calls()
    ctx = RunnerContext(storage=storage, workspace=workspace, hooks=hooks)

    with pytest.raises(RuntimeError, match="intentional failure"):
        run_pipeline(run, pipeline, ctx)

    final = Run.load(storage, run.id)
    assert final.status == RunStatus.FAILED
    assert final.last_error is not None
    assert "intentional failure" in final.last_error

    # Critically: the checkpoint is NOT overwritten to "failed". It stays at
    # the step that was in progress when the error happened, so the next
    # run_pipeline call resumes at the same phase rather than getting stuck.
    cp = checkpoint.load(storage, run)
    assert cp is not None and cp.step == "running"

    fs = findings.list_findings(storage, run)
    assert any(f.type == FindingType.ERROR for f in fs)
    assert [name for name, _ in calls] == ["pre", "err"]


def test_failure_then_resume_succeeds(storage: StorageBackend, workspace: Path) -> None:
    """Cycle: run fails mid-pipeline, supervisor 'restarts', second attempt succeeds."""
    run = make_run(storage)
    pipeline = StubPipeline(fail_times=1)
    _calls, hooks = collect_hook_calls()
    ctx = RunnerContext(storage=storage, workspace=workspace, hooks=hooks)

    with pytest.raises(RuntimeError):
        run_pipeline(run, pipeline, ctx)

    failed = Run.load(storage, run.id)
    assert failed.status == RunStatus.FAILED
    # Checkpoint is preserved at "running" — no supervisor intervention needed.
    cp = checkpoint.load(storage, run)
    assert cp is not None and cp.step == "running"

    # Just call run_pipeline again; the FSM re-enters the running phase.
    result = run_pipeline(failed, pipeline, ctx)
    assert result == {"score": 0.5}
    assert Run.load(storage, run.id).status == RunStatus.COMPLETED


def test_skip_all_validators(storage: StorageBackend, workspace: Path) -> None:
    """No hooks at all — runner still drives the FSM, just no findings from hooks."""
    run = make_run(storage)
    pipeline = StubPipeline(result={"score": 0.1})
    ctx = RunnerContext(storage=storage, workspace=workspace, hooks=WorkflowHooks())

    result = run_pipeline(run, pipeline, ctx)
    assert result == {"score": 0.1}

    fs = findings.list_findings(storage, run)
    types = [f.type for f in fs]
    assert types == [FindingType.RESULT]  # only the auto-written result finding


def test_long_call_hooks_fire(storage: StorageBackend, workspace: Path) -> None:
    """on_long_call_start/end bracket the pipeline.run() call."""
    run = make_run(storage)
    pipeline = StubPipeline()
    started: list[str] = []
    ended: list[str] = []

    def start(r):
        started.append(r.id)

    def end(r):
        ended.append(r.id)

    ctx = RunnerContext(
        storage=storage,
        workspace=workspace,
        hooks=WorkflowHooks(on_long_call_start=start, on_long_call_end=end),
    )
    run_pipeline(run, pipeline, ctx)

    assert started == [run.id]
    assert ended == [run.id]


def test_long_call_end_fires_even_on_failure(storage: StorageBackend, workspace: Path) -> None:
    run = make_run(storage)
    pipeline = StubPipeline(fail_times=99)
    started: list[str] = []
    ended: list[str] = []
    ctx = RunnerContext(
        storage=storage,
        workspace=workspace,
        hooks=WorkflowHooks(
            on_long_call_start=lambda r: started.append(r.id),
            on_long_call_end=lambda r: ended.append(r.id),
        ),
    )

    with pytest.raises(RuntimeError):
        run_pipeline(run, pipeline, ctx)

    assert started == [run.id]
    assert ended == [run.id]
