"""Reusable test fakes."""

from __future__ import annotations

from autoresearch.backends.compute.base import SessionHandle, SessionSpec
from autoresearch.core.hardware import GpuOffer


class FakeCompute:
    """In-memory ComputeBackend stand-in. Records calls; assigns deterministic pod ids."""

    name = "fake"

    def __init__(self, offers: list[GpuOffer] | None = None) -> None:
        self.created: list[SessionSpec] = []
        self.terminated: list[str] = []
        self._counter = 0
        self._sessions: dict[str, SessionHandle] = {}
        # Optional offers list for hardware-selection tests. Empty by default
        # so existing tests that don't care about offers still work.
        self._offers: list[GpuOffer] = offers or []

    def create_session(self, spec: SessionSpec) -> SessionHandle:
        self._counter += 1
        pod_id = f"pod-{self._counter}"
        handle = SessionHandle(
            id=pod_id,
            status="queued",
            public_ip="10.0.0.1",
            ssh_port=22222 + self._counter,
        )
        self._sessions[pod_id] = handle
        self.created.append(spec)
        return handle

    def get_session(self, session_id: str) -> SessionHandle:
        return self._sessions[session_id]

    def terminate_session(self, session_id: str) -> None:
        self.terminated.append(session_id)
        if session_id in self._sessions:
            self._sessions[session_id] = self._sessions[session_id].model_copy(
                update={"status": "terminated"}
            )

    def list_gpu_offers(self, data_center_id: str | None = None) -> list[GpuOffer]:
        # data_center_id is ignored by the fake; offers come from the
        # constructor argument so tests can inject exactly what they need.
        return list(self._offers)
