"""Reusable test fakes."""

from __future__ import annotations

from autoresearch.backends.compute.base import SessionHandle, SessionSpec


class FakeCompute:
    """In-memory ComputeBackend stand-in. Records calls; assigns deterministic pod ids."""

    name = "fake"

    def __init__(self) -> None:
        self.created: list[SessionSpec] = []
        self.terminated: list[str] = []
        self._counter = 0
        self._sessions: dict[str, SessionHandle] = {}

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
