"""CLI smoke tests — exercise `autoresearch run` via subprocess.

These tests are slower than unit tests but they catch packaging/entrypoint bugs
that unit tests can miss (e.g. broken module imports under the installed
console-script entrypoint).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def run_cli(*args: str, env: dict[str, str | None] | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run autoresearch CLI in a subprocess.

    `env` values may be `None` to UNSET an inherited variable (useful when a
    test needs to assert behavior with a particular var missing — e.g.
    ANTHROPIC_API_KEY — even when the developer's shell happens to have it
    set).
    """
    base_env = os.environ.copy()
    for k, v in (env or {}).items():
        if v is None:
            base_env.pop(k, None)
        else:
            base_env[k] = v
    return subprocess.run(
        [sys.executable, "-m", "autoresearch.cli", *args],
        cwd=cwd or REPO_ROOT,
        env=base_env,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def cli_env(tmp_path: Path) -> dict[str, str]:
    return {
        "AUTORESEARCH_STORAGE": "local",
        "AUTORESEARCH_STORAGE_ROOT": str(tmp_path / "store"),
        "AUTORESEARCH_PIPELINE_MODULE_PATH": str(REPO_ROOT / "pipelines"),
        "AUTORESEARCH_PREFLIGHT": "false",
        "AUTORESEARCH_POSTFLIGHT": "false",
        "AUTORESEARCH_SUMMARIZE_ERRORS": "false",
    }


def test_cli_run_completes_successfully(cli_env: dict[str, str], tmp_path: Path) -> None:
    proc = run_cli(
        "run",
        "--workflow", "transfer",
        "--pipeline", "fra_example",
        "--target-model", "Qwen/Qwen2.5-32B",
        "--budget", "5",
        "--workspace", str(tmp_path / "ws"),
        env=cli_env,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["metadata"]["model"] == "Qwen/Qwen2.5-32B"
    assert "fra_score" in result


def test_cli_run_resume_after_failure(cli_env: dict[str, str], tmp_path: Path) -> None:
    """First attempt fails; second attempt with --run-id resumes and succeeds."""
    fail_env = {**cli_env, "FRA_EXAMPLE_FAIL_ONCE": "1"}

    attempt1 = run_cli(
        "run",
        "--workflow", "transfer",
        "--pipeline", "fra_example",
        "--target-model", "Qwen/Qwen2.5-32B",
        "--budget", "5",
        "--workspace", str(tmp_path / "ws"),
        env=fail_env,
    )
    assert attempt1.returncode != 0
    assert "intentional failure" in attempt1.stderr

    # Grab the run id from stderr ("created run XXX")
    run_id = None
    for line in attempt1.stderr.splitlines():
        if line.startswith("created run "):
            run_id = line.split()[-1]
    assert run_id, f"no run id in stderr: {attempt1.stderr}"

    attempt2 = run_cli(
        "run",
        "--workflow", "transfer",
        "--pipeline", "fra_example",
        "--target-model", "Qwen/Qwen2.5-32B",
        "--budget", "5",
        "--run-id", run_id,
        "--workspace", str(tmp_path / "ws"),
        env=cli_env,
    )
    assert attempt2.returncode == 0, attempt2.stderr
    result = json.loads(attempt2.stdout)
    assert "fra_score" in result


def test_cli_agent_workflow_without_api_key_exits_zero(cli_env: dict[str, str], tmp_path: Path) -> None:
    """Regression: a prep/mechanic/postflight dispatch on a pod with no
    ANTHROPIC_API_KEY must exit 0 (not 2). Otherwise RunPod's
    restart-on-non-zero-exit will loop the pod forever, each iteration
    failing the same way. We trade a "cleaner" non-zero exit for not
    burning compute on a structural config gap.

    Backstory: a prep dispatch ran rc=2 every 30s for hours because
    ANTHROPIC_API_KEY was missing from the controller env. Beacons in
    findings showed the same 01→08 trail per restart.
    """
    from autoresearch.backends.storage import LocalStorage
    from autoresearch.core.run import Run, RunStatus

    store_root = tmp_path / "store"
    storage = LocalStorage(store_root)
    run = Run(workflow="prepare", pipeline_name="x", params={})
    run.save(storage)

    # Run with NO ANTHROPIC_API_KEY anywhere (including inherited env from
    # the developer's shell — the run_cli helper treats None as "unset").
    env: dict[str, str | None] = {
        **cli_env,
        "AUTORESEARCH_STORAGE_ROOT": str(store_root),
        "ANTHROPIC_API_KEY": None,
        "AUTORESEARCH_ANTHROPIC_API_KEY": None,
    }
    proc = run_cli("run", "--run-id", run.id, "--heartbeat", env=env)

    assert proc.returncode == 0, f"expected exit 0 to stop RunPod restart loop, got {proc.returncode}\nstderr: {proc.stderr}"
    assert "ANTHROPIC_API_KEY" in proc.stderr

    # Run status must be FAILED with a clear last_error
    fresh = Run.load(storage, run.id)
    assert fresh.status == RunStatus.FAILED
    assert "ANTHROPIC_API_KEY" in (fresh.last_error or "")
