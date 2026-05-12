"""State layer round-trip tests: Run, findings, logs, checkpoint, budget through
both LocalStorage and S3Storage backends."""

from __future__ import annotations

import pytest

from autoresearch.backends.storage import StorageBackend
from autoresearch.core import budget, checkpoint, findings, logs
from autoresearch.core.findings import FindingType
from autoresearch.core.run import Run, RunStatus


def make_run(workflow: str = "transfer", pipeline_name: str = "fra") -> Run:
    return Run(
        workflow=workflow,
        pipeline_name=pipeline_name,
        params={"target_model": "Qwen/Qwen2.5-32B"},
        budget_cap_usd=50.0,
    )


def test_run_save_load_roundtrip(storage: StorageBackend) -> None:
    run = make_run()
    run.save(storage)

    loaded = Run.load(storage, run.id)
    assert loaded.id == run.id
    assert loaded.workflow == "transfer"
    assert loaded.params == {"target_model": "Qwen/Qwen2.5-32B"}
    assert loaded.status == RunStatus.QUEUED
    assert loaded.budget_cap_usd == 50.0


def test_list_all_runs(storage: StorageBackend) -> None:
    r1 = make_run()
    r2 = make_run()
    r1.save(storage)
    r2.save(storage)
    listed = Run.list_all(storage)
    assert {r.id for r in listed} == {r1.id, r2.id}


def test_findings_append_and_list(storage: StorageBackend) -> None:
    run = make_run()
    run.save(storage)

    for i in range(10):
        findings.append(
            storage,
            run,
            FindingType.OBSERVATION,
            f"step {i}",
            metadata={"step_num": i},
        )

    listed = findings.list_findings(storage, run)
    assert len(listed) == 10
    # Ordering should match the order appended (chronological).
    for i, f in enumerate(listed):
        assert f.type == FindingType.OBSERVATION
        assert f.body == f"step {i}"
        assert f.metadata == {"step_num": i}


def test_findings_since_cursor(storage: StorageBackend) -> None:
    run = make_run()
    run.save(storage)
    keys = [findings.append(storage, run, FindingType.OBSERVATION, f"f{i}") for i in range(5)]
    after = findings.list_findings(storage, run, since_cursor=keys[2])
    assert [f.body for f in after] == ["f3", "f4"]


def test_logs_append_and_tail(storage: StorageBackend) -> None:
    run = make_run()
    run.save(storage)

    # Many chunks, multiple lines each.
    for i in range(20):
        logs.append(storage, run, f"chunk-{i}-line-1\nchunk-{i}-line-2\n")

    tailed = logs.tail(storage, run, lines=5)
    tail_lines = tailed.splitlines()
    assert len(tail_lines) == 5
    assert tail_lines[-1] == "chunk-19-line-2"
    assert tail_lines[0] == "chunk-17-line-2"


def test_logs_tail_empty(storage: StorageBackend) -> None:
    run = make_run()
    run.save(storage)
    assert logs.tail(storage, run) == ""


def test_checkpoint_write_load(storage: StorageBackend) -> None:
    run = make_run()
    run.save(storage)
    assert checkpoint.load(storage, run) is None

    checkpoint.write(storage, run, step="loading")
    cp = checkpoint.load(storage, run)
    assert cp is not None
    assert cp.step == "loading"
    assert cp.partial_result is None

    checkpoint.write(storage, run, step="validating", partial_result={"score": 0.42})
    cp = checkpoint.load(storage, run)
    assert cp is not None
    assert cp.step == "validating"
    assert cp.partial_result == {"score": 0.42}


def test_budget_add_spend_persists(storage: StorageBackend) -> None:
    run = make_run()
    run.save(storage)
    updated = budget.add_spend(storage, run, 1.25)
    assert updated.budget_spent_usd == 1.25
    reloaded = Run.load(storage, run.id)
    assert reloaded.budget_spent_usd == 1.25


def test_budget_check_raises_at_cap(storage: StorageBackend) -> None:
    run = make_run()
    run.budget_cap_usd = 5.0
    run.save(storage)
    budget.add_spend(storage, run, 4.99)
    fresh = Run.load(storage, run.id)
    budget.check(fresh)  # below cap, no raise
    budget.add_spend(storage, fresh, 0.02)
    fresh = Run.load(storage, run.id)
    with pytest.raises(budget.BudgetExceeded):
        budget.check(fresh)


def test_budget_remaining(storage: StorageBackend) -> None:
    run = make_run()
    run.budget_cap_usd = 10.0
    run.save(storage)
    assert budget.remaining(run) == 10.0
    updated = budget.add_spend(storage, run, 3.5)
    assert budget.remaining(updated) == 6.5


def test_bulk_roundtrip(storage: StorageBackend) -> None:
    """Run + 10 findings + 100 log lines + 1 checkpoint, end-to-end."""
    run = make_run()
    run.save(storage)

    for i in range(10):
        findings.append(storage, run, FindingType.RESULT, f"result-{i}")
    logs.append(storage, run, "\n".join(f"log line {i}" for i in range(100)) + "\n")
    checkpoint.write(storage, run, step="completed", partial_result={"final": True})

    assert len(findings.list_findings(storage, run)) == 10
    tailed_all = logs.tail(storage, run, lines=200)
    assert "log line 0" in tailed_all
    assert "log line 99" in tailed_all
    cp = checkpoint.load(storage, run)
    assert cp is not None and cp.step == "completed"
