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


def run_cli(*args: str, env: dict[str, str] | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    base_env = os.environ.copy()
    base_env.update(env or {})
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
        "AUTORESEARCH_PIPELINE_MODULE_PATH": str(REPO_ROOT / "templates" / "pipelines"),
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
