"""MCP surface integration tests — invoke tools via FastMCP's call_tool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoresearch.backends.models.base import ModelResponse
from autoresearch.backends.storage import LocalStorage
from autoresearch.config import Settings
from autoresearch.controller.mcp_surface import build_mcp
from autoresearch.core import findings as findings_mod
from autoresearch.core import logs as logs_mod
from autoresearch.core.findings import FindingType
from autoresearch.core.run import Run

from tests._fakes import FakeCompute


class FakeModel:
    model = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, system: str, user: str, max_tokens: int = 1024) -> ModelResponse:
        self.calls.append((system, user))
        return ModelResponse(text="summary text", input_tokens=10, output_tokens=20, cost_usd=0.0)


def _settings(**overrides) -> Settings:
    base = dict(
        storage="local",
        compute="runpod",
        runpod_api_key="rpa-test",
        runpod_network_volume_id="vol-test",
        runpod_default_image="img",
        pipeline_module_path="templates/pipelines",
        default_gpu="H100 80GB",
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def setup(tmp_path: Path):
    storage = LocalStorage(tmp_path / "store")
    compute = FakeCompute()
    model = FakeModel()
    mcp = build_mcp(
        settings=_settings(),
        storage=storage,
        compute=compute,
        model_client=model,
    )
    return mcp, storage, compute, model


async def call(mcp, name: str, args: dict):
    """Invoke an MCP tool, returning its payload normalized.

    FastMCP wraps non-dict returns (str / list[primitive]) in {"result": value}
    but passes dict-typed returns through directly. We unwrap the single-key
    "result" shape so tests can treat all return values uniformly.
    """
    _blocks, structured = await mcp.call_tool(name, args)
    if isinstance(structured, dict) and list(structured.keys()) == ["result"]:
        return structured["result"]
    return structured


@pytest.mark.asyncio
async def test_start_transfer_creates_run_and_pod(setup) -> None:
    mcp, storage, compute, _ = setup
    out = await call(mcp, "start_transfer", {
        "pipeline_name": "fra_example",
        "target_model": "Qwen/Qwen2.5-32B",
        "budget_usd": 5,
    })
    assert "run_id" in out
    run_id = out["run_id"]
    assert out["pod_handle"] == "pod-1"

    run = Run.load(storage, run_id)
    assert run.pipeline_name == "fra_example"
    assert run.params["target_model"] == "Qwen/Qwen2.5-32B"
    assert len(compute.created) == 1


@pytest.mark.asyncio
async def test_list_and_get_run(setup) -> None:
    mcp, storage, _, _ = setup
    run = Run(workflow="transfer", pipeline_name="fra_example",
              params={"target_model": "X"}, budget_cap_usd=5)
    run.save(storage)

    listed = await call(mcp, "list_runs", {})
    assert len(listed) == 1
    assert listed[0]["id"] == run.id

    got = await call(mcp, "get_run", {"run_id": run.id})
    assert got["id"] == run.id
    assert got["workflow"] == "transfer"


@pytest.mark.asyncio
async def test_findings_and_logs(setup) -> None:
    mcp, storage, _, _ = setup
    run = Run(workflow="transfer", pipeline_name="x", params={"target_model": "Q"})
    run.save(storage)
    findings_mod.append(storage, run, FindingType.OBSERVATION, "first")
    findings_mod.append(storage, run, FindingType.RESULT, '{"score": 0.5}')
    logs_mod.append(storage, run, "line one\nline two\n")

    fs = await call(mcp, "list_findings", {"run_id": run.id})
    assert [f["body"] for f in fs] == ["first", '{"score": 0.5}']

    tail = await call(mcp, "tail_log", {"run_id": run.id, "lines": 10})
    assert "line one" in tail


@pytest.mark.asyncio
async def test_budget_get_set(setup) -> None:
    mcp, storage, _, _ = setup
    run = Run(workflow="transfer", pipeline_name="x", params={"target_model": "Q"},
              budget_cap_usd=5)
    run.save(storage)

    info = await call(mcp, "get_budget", {"run_id": run.id})
    assert info["cap_usd"] == 5
    assert info["remaining_usd"] == 5

    upd = await call(mcp, "set_budget", {"run_id": run.id, "new_cap_usd": 12.5})
    assert upd["cap_usd"] == 12.5
    assert Run.load(storage, run.id).budget_cap_usd == 12.5


@pytest.mark.asyncio
async def test_summarize_run_calls_model(setup) -> None:
    mcp, storage, _, model = setup
    run = Run(workflow="transfer", pipeline_name="x", params={"target_model": "Q"})
    run.save(storage)
    findings_mod.append(storage, run, FindingType.RESULT, json.dumps({"score": 0.42}))

    out = await call(mcp, "summarize_run", {"run_id": run.id, "max_tokens": 100})
    assert out == "summary text"
    assert len(model.calls) == 1
    assert run.id in model.calls[0][1]  # run id appears in the user prompt


@pytest.mark.asyncio
async def test_takeover_release_cancel(setup) -> None:
    mcp, storage, compute, _ = setup
    started = await call(mcp, "start_transfer", {
        "pipeline_name": "fra_example",
        "target_model": "Q",
        "budget_usd": 1,
    })
    run_id = started["run_id"]

    take = await call(mcp, "takeover", {"run_id": run_id})
    assert "ssh root@10.0.0.1" in take["ssh_command"]
    assert take["status"] == "paused"

    rel = await call(mcp, "release", {"run_id": run_id})
    assert rel["status"] == "running"

    can = await call(mcp, "cancel", {"run_id": run_id})
    assert can["status"] == "failed"
    assert "pod-1" in compute.terminated


@pytest.mark.asyncio
async def test_list_pipelines_finds_example(setup) -> None:
    mcp, _, _, _ = setup
    names = await call(mcp, "list_pipelines", {})
    assert "fra_example" in names
