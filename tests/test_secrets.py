"""secrets.env_for_run: pure function over (Run, Settings) + process env."""

from __future__ import annotations

from typing import Iterator

import pytest

from autoresearch.config import Settings
from autoresearch.core.run import Run
from autoresearch.core.secrets import env_for_run


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    """Strip secret env vars so each test starts from a known state."""
    for name in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "ANTHROPIC_API_KEY",
        "HF_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    yield monkeypatch


def _settings(**overrides) -> Settings:
    base = dict(
        storage="s3",
        storage_bucket="fra-proj",
        storage_endpoint_url="https://acct.r2.cloudflarestorage.com",
        storage_region="auto",
        controller_public_url="https://controller.example.com",
        preflight=True,
        postflight=False,
        summarize_errors=True,
    )
    base.update(overrides)
    return Settings(**base)


def _run() -> Run:
    return Run(workflow="transfer", pipeline_name="stub", params={"target_model": "X"})


def test_basic_env_contains_run_id_and_storage(clean_env) -> None:
    env = env_for_run(_run(), _settings())
    assert env["RUN_ID"]
    assert env["AUTORESEARCH_STORAGE"] == "s3"
    assert env["AUTORESEARCH_STORAGE_BUCKET"] == "fra-proj"
    assert env["AUTORESEARCH_STORAGE_ENDPOINT_URL"].endswith("cloudflarestorage.com")
    assert env["AUTORESEARCH_STORAGE_REGION"] == "auto"
    assert env["CONTROLLER_PUBLIC_URL"] == "https://controller.example.com"


def test_hf_defaults(clean_env) -> None:
    env = env_for_run(_run(), _settings())
    assert env["HF_HOME"] == "/workspace/.huggingface"
    assert env["HF_HUB_ENABLE_HF_TRANSFER"] == "1"


def test_passthrough_secrets_only_when_present(clean_env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ak-test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    env = env_for_run(_run(), _settings())
    assert env["AWS_ACCESS_KEY_ID"] == "ak-test"
    assert env["AWS_SECRET_ACCESS_KEY"] == "sk-test"
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-test"
    assert "HF_TOKEN" not in env  # not set in env, so not passed through


def test_empty_optionals_are_stripped(clean_env) -> None:
    env = env_for_run(_run(), _settings(controller_public_url=None, controller_url=None))
    assert "CONTROLLER_PUBLIC_URL" not in env  # was empty


def test_workflow_flags_propagate(clean_env) -> None:
    env = env_for_run(_run(), _settings(preflight=True, postflight=False, summarize_errors=False))
    assert env["AUTORESEARCH_PREFLIGHT"] == "true"
    assert env["AUTORESEARCH_POSTFLIGHT"] == "false"
    assert env["AUTORESEARCH_SUMMARIZE_ERRORS"] == "false"


def test_extra_overrides_existing(clean_env) -> None:
    env = env_for_run(_run(), _settings(), extra={"HF_HOME": "/other/path", "CUSTOM": "x"})
    assert env["HF_HOME"] == "/other/path"
    assert env["CUSTOM"] == "x"


def test_pipeline_module_path_pod_default(clean_env) -> None:
    env = env_for_run(_run(), _settings())
    assert env["AUTORESEARCH_PIPELINE_MODULE_PATH"] == "/workspace/pipelines"
