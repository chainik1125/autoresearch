"""Controller server smoke tests — construct app, hit /healthz, hit MCP endpoint."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build a controller app using local storage + no compute backend.

    The supervisor is disabled (no compute), so this smoke test doesn't need to
    deal with background asyncio tasks.
    """
    monkeypatch.setenv("AUTORESEARCH_STORAGE", "local")
    monkeypatch.setenv("AUTORESEARCH_STORAGE_ROOT", str(tmp_path / "store"))
    monkeypatch.setenv("AUTORESEARCH_COMPUTE", "local")
    monkeypatch.setenv("AUTORESEARCH_PREFLIGHT", "false")
    monkeypatch.setenv("AUTORESEARCH_POSTFLIGHT", "false")
    monkeypatch.setenv("AUTORESEARCH_SUMMARIZE_ERRORS", "false")
    monkeypatch.setenv("AUTORESEARCH_PIPELINE_MODULE_PATH", "pipelines")
    # No autoresearch.toml in cwd would still get picked up; ensure clean by
    # chdir'ing tmp_path so the cwd toml lookup misses.
    monkeypatch.chdir(tmp_path)

    from autoresearch.controller.server import create_app

    return create_app()


def test_healthz(app) -> None:
    with TestClient(app) as client:
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_version(app) -> None:
    with TestClient(app) as client:
        r = client.get("/version")
        assert r.status_code == 200
        assert "version" in r.json()


def test_mcp_endpoint_mounted(app) -> None:
    """Verify the MCP sub-app is mounted under /mcp (full URL: /mcp/).

    We don't actually send an MCP JSON-RPC request here — that requires a
    session init dance covered by the in-process call_tool tests in
    test_mcp_surface.py. End-to-end HTTP-level MCP gets exercised in task 12."""
    from starlette.routing import Mount

    mounts = [r for r in app.routes if isinstance(r, Mount) and r.path == "/mcp"]
    assert len(mounts) == 1
    # We set FastMCP's streamable_http_path to "/" so the route inside the
    # mounted app is "/" — the full URL is just /mcp/ (no /mcp/mcp redirect).
    sub_paths = [getattr(r, "path", None) for r in mounts[0].app.routes]
    assert "/" in sub_paths
