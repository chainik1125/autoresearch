"""RunPodCompute tests using a respx mock against the REST API.

These don't hit RunPod's real API — they verify our request shape and our
parsing of typical responses.
"""

from __future__ import annotations

import httpx
import pytest

# pytest-httpx or respx aren't deps; we install httpx's MockTransport directly.

from autoresearch.backends.compute.base import SessionSpec
from autoresearch.backends.compute.runpod import RunPodCompute, _resolve_gpu


def _mock_compute(handler) -> RunPodCompute:
    transport = httpx.MockTransport(handler)
    compute = RunPodCompute(api_key="test")
    compute._client = httpx.Client(
        base_url="https://rest.runpod.io/v1",
        headers={"Authorization": "Bearer test", "Content-Type": "application/json"},
        transport=transport,
    )
    return compute


def test_gpu_alias_resolution() -> None:
    assert _resolve_gpu("H100 80GB") == "NVIDIA H100 80GB HBM3"
    assert _resolve_gpu("A100") == "NVIDIA A100-SXM4-80GB"
    assert _resolve_gpu("A40") == "NVIDIA A40"
    # Unknown short names pass through.
    assert _resolve_gpu("NVIDIA RTX 4090") == "NVIDIA RTX 4090"


def test_create_session_request_shape() -> None:
    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["method"] = req.method
        captured["url"] = str(req.url)
        captured["headers"] = dict(req.headers)
        import json as _json
        captured["body"] = _json.loads(req.content)
        return httpx.Response(
            201,
            json={
                "id": "pod-abc",
                "desiredStatus": "RUNNING",
                "publicIp": "1.2.3.4",
                "ports": [{"privatePort": 22, "publicPort": 12345, "ip": "1.2.3.4"}],
            },
        )

    compute = _mock_compute(handler)
    spec = SessionSpec(
        gpu="H100 80GB",
        image="my/image:latest",
        network_volume_id="vol-xyz",
        env={"RUN_ID": "abc123", "HF_HOME": "/workspace/.huggingface"},
        name="autoresearch-run-abc123",
    )
    handle = compute.create_session(spec)

    assert captured["method"] == "POST"
    assert captured["url"].endswith("/pods")
    assert captured["headers"]["authorization"] == "Bearer test"
    body = captured["body"]
    assert body["name"] == "autoresearch-run-abc123"
    assert body["imageName"] == "my/image:latest"
    assert body["gpuTypeIds"] == ["NVIDIA H100 80GB HBM3"]
    assert body["networkVolumeId"] == "vol-xyz"
    assert body["env"]["RUN_ID"] == "abc123"
    assert body["ports"] == ["22/tcp"]
    assert body["containerDiskInGb"] == 50
    assert "containerRegistryAuthId" not in body  # spec didn't set it

    assert handle.id == "pod-abc"
    assert handle.status == "running"
    assert handle.public_ip == "1.2.3.4"
    assert handle.ssh_port == 12345
    assert handle.ssh_command() == "ssh root@1.2.3.4 -p 12345"


def test_get_session_maps_status() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "GET"
        return httpx.Response(200, json={"id": "pod-1", "desiredStatus": "EXITED"})

    compute = _mock_compute(handler)
    handle = compute.get_session("pod-1")
    assert handle.status == "exited"


def test_terminate_session_idempotent_on_404() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not found"})

    compute = _mock_compute(handler)
    # 404 must not raise — termination is idempotent.
    compute.terminate_session("pod-does-not-exist")


def test_create_session_passes_extra_ports_and_disables_ssh() -> None:
    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        import json as _json
        captured["body"] = _json.loads(req.content)
        return httpx.Response(201, json={"id": "p", "desiredStatus": "QUEUED"})

    compute = _mock_compute(handler)
    spec = SessionSpec(
        gpu="A40",
        image="img",
        network_volume_id="v",
        expose_ssh=False,
        extra_ports=["8000/http"],
    )
    compute.create_session(spec)
    body = captured["body"]
    assert "22/tcp" not in body["ports"]
    assert body["ports"] == ["8000/http"]


def test_create_session_includes_registry_auth_when_set() -> None:
    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        import json as _json
        captured["body"] = _json.loads(req.content)
        return httpx.Response(201, json={"id": "p", "desiredStatus": "QUEUED"})

    compute = _mock_compute(handler)
    spec = SessionSpec(
        gpu="A40",
        image="ghcr.io/me/private:latest",
        network_volume_id="v",
        container_registry_auth_id="my-cred-id",
    )
    compute.create_session(spec)
    assert captured["body"]["containerRegistryAuthId"] == "my-cred-id"


def test_status_mapping() -> None:
    """Verify the status normalization handles RunPod's various string values."""
    from autoresearch.backends.compute.runpod import _interpret_status

    assert _interpret_status("RUNNING") == "running"
    assert _interpret_status("READY") == "running"
    assert _interpret_status("CREATED") == "queued"
    assert _interpret_status("PROVISIONING") == "queued"
    assert _interpret_status("EXITED") == "exited"
    assert _interpret_status("TERMINATED") == "terminated"
    assert _interpret_status("FAILED") == "failed"
    assert _interpret_status(None) == "unknown"
    assert _interpret_status("WHATEVER") == "unknown"
