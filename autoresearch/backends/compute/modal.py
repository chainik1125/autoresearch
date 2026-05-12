"""ModalCompute — v2 stub.

The plan was to use Modal for *ephemeral parallel* agents (multi-model debates,
short LLM-only sweeps) where pay-per-second + sub-second cold start matter. The
protocol exists so workflows that need this shape can be written against it; the
implementation is intentionally absent in v1 because TRANSFER is sequential and
doesn't need the ephemeral primitive yet.

When this gets implemented in v2:
  - `create_session` should map to a long-lived `modal.Sandbox.create(...)` for
    parity with RunPod's session shape (used by SWEEP per-config workers).
  - `run_ephemeral` (not yet in the protocol) should map to `@app.function().spawn(...)`
    for true fire-and-forget parallel work (multi-model debate, abstract review).
  - Image build: Modal builds images from a Python recipe rather than a Dockerfile,
    so we'd ship a `modal.Image.from_dockerfile(Dockerfile)` shim that points
    at the same Dockerfile RunPod uses.
"""

from __future__ import annotations

from autoresearch.backends.compute.base import SessionHandle, SessionSpec


_NOT_IMPLEMENTED = (
    "ModalCompute is a v1 stub. The protocol exists so future workflows "
    "(SWEEP, MULTI_MODEL_REVIEW) can target Modal's ephemeral primitives, but "
    "TRANSFER doesn't need them. Implement when the second workflow lands."
)


class ModalCompute:
    name = "modal"

    def __init__(self, *_args, **_kwargs) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def create_session(self, spec: SessionSpec) -> SessionHandle:  # pragma: no cover
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def get_session(self, session_id: str) -> SessionHandle:  # pragma: no cover
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def terminate_session(self, session_id: str) -> None:  # pragma: no cover
        raise NotImplementedError(_NOT_IMPLEMENTED)
