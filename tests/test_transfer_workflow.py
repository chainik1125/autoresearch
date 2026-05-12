"""Transfer workflow assembly test — no real LLM, no real GPU.

Verifies that `workflows.transfer.transfer(...)` wires the runner + hooks
correctly. The LLM validators are exercised in tests that use a fake ModelClient,
but for the no-LLM smoke path we just confirm the workflow runs cleanly with all
validators disabled.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autoresearch.backends.models.base import ModelClient, ModelResponse
from autoresearch.backends.storage import LocalStorage
from autoresearch.core import findings
from autoresearch.core.findings import FindingType
from autoresearch.core.run import Run, RunStatus
from autoresearch.workflows.transfer import transfer
from tests.test_pipeline_runner import StubPipeline


class FakeModelClient:
    model = "fake-model"

    def __init__(self, *, response: str = "fake validator response") -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, system: str, user: str, max_tokens: int = 1024) -> ModelResponse:
        self.calls.append((system, user))
        return ModelResponse(text=self.response, input_tokens=10, output_tokens=20, cost_usd=0.0)


def test_transfer_no_validators(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "store")
    workspace = tmp_path / "ws"
    workspace.mkdir()

    run = Run(
        workflow="transfer",
        pipeline_name="stub",
        params={"target_model": "Qwen/Qwen2.5-32B"},
    )
    run.save(storage)
    pipeline = StubPipeline(result={"score": 0.9})

    result = transfer(
        run,
        pipeline,
        storage=storage,
        workspace=workspace,
        model_client=None,
        preflight=False,
        postflight=False,
        summarize_errors=False,
    )

    assert result == {"score": 0.9}
    assert Run.load(storage, run.id).status == RunStatus.COMPLETED

    fs = findings.list_findings(storage, run)
    # Only the auto-written result finding; no LLM-driven observations.
    assert [f.type for f in fs] == [FindingType.RESULT]


def test_transfer_with_fake_validators(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "store")
    workspace = tmp_path / "ws"
    workspace.mkdir()

    run = Run(
        workflow="transfer",
        pipeline_name="stub",
        params={"target_model": "Qwen/Qwen2.5-32B"},
    )
    run.save(storage)
    pipeline = StubPipeline(result={"score": 0.42})
    client = FakeModelClient(response="looks good")

    result = transfer(
        run,
        pipeline,
        storage=storage,
        workspace=workspace,
        model_client=client,
        preflight=True,
        postflight=True,
        summarize_errors=True,
    )

    assert result == {"score": 0.42}
    assert len(client.calls) == 2  # preflight + postflight; no error.
    assert "Qwen/Qwen2.5-32B" in client.calls[0][1]  # target model appears in preflight user prompt
    assert "0.42" in client.calls[1][1]  # result appears in postflight user prompt

    fs = findings.list_findings(storage, run)
    types = [f.type for f in fs]
    assert types == [FindingType.OBSERVATION, FindingType.RESULT, FindingType.OBSERVATION]
    assert fs[0].body == "looks good"
    assert fs[2].body == "looks good"


def test_transfer_requires_client_when_validators_enabled(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "store")
    run = Run(workflow="transfer", pipeline_name="stub", params={"target_model": "X"})
    run.save(storage)
    with pytest.raises(ValueError, match="model_client is required"):
        transfer(
            run,
            StubPipeline(),
            storage=storage,
            workspace=tmp_path,
            model_client=None,
            preflight=True,
        )
