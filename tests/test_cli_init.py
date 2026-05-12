"""`autoresearch init` smoke test — verifies template copy + idempotency."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def run_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "autoresearch.cli", *args],
        cwd=cwd,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )


def test_init_copies_expected_files(tmp_path: Path) -> None:
    proc = run_cli("init", "--target", str(tmp_path), cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stderr

    assert (tmp_path / "autoresearch.toml").exists()
    assert (tmp_path / ".claude" / "skills" / "transfer.md").exists()
    assert (tmp_path / "pipelines" / "fra_example.py").exists()

    # Content sanity
    toml = (tmp_path / "autoresearch.toml").read_text()
    assert "controller_url" in toml
    skill = (tmp_path / ".claude" / "skills" / "transfer.md").read_text()
    assert "/transfer" in skill


def test_init_is_idempotent_without_force(tmp_path: Path) -> None:
    (tmp_path / "autoresearch.toml").write_text("# pre-existing\n")
    proc = run_cli("init", "--target", str(tmp_path), cwd=REPO_ROOT)
    assert proc.returncode == 0
    # Pre-existing file should be untouched.
    assert (tmp_path / "autoresearch.toml").read_text() == "# pre-existing\n"
    assert "skip (exists)" in proc.stderr


def test_init_force_overwrites(tmp_path: Path) -> None:
    (tmp_path / "autoresearch.toml").write_text("# stale\n")
    proc = run_cli("init", "--target", str(tmp_path), "--force", cwd=REPO_ROOT)
    assert proc.returncode == 0
    assert "controller_url" in (tmp_path / "autoresearch.toml").read_text()
