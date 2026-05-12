"""Checkpoint — single-writer atomic-overwrite state snapshot for a run.

Each runner step writes a checkpoint capturing where in the FSM it is and (for
the `validating` step) the result it computed. On pod death the supervisor spawns
a new pod; the new runner reads the checkpoint and resumes from the recorded step.

The checkpoint is intentionally small — just enough to drive the FSM. The user
pipeline's own intermediate state lives on the pod's persistent volume; it's the
pipeline author's job to checkpoint that if they care to.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from autoresearch.backends.storage import StorageBackend
from autoresearch.backends.storage.base import KeyNotFound
from autoresearch.core.run import Run


class Checkpoint(BaseModel):
    step: str
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    partial_result: dict[str, Any] | None = None
    error: str | None = None


def write(
    storage: StorageBackend,
    run: Run,
    step: str,
    *,
    partial_result: dict[str, Any] | None = None,
    error: str | None = None,
) -> Checkpoint:
    cp = Checkpoint(step=step, partial_result=partial_result, error=error)
    storage.write(run.checkpoint_key, cp.model_dump_json().encode("utf-8"))
    return cp


def load(storage: StorageBackend, run: Run) -> Checkpoint | None:
    try:
        data = storage.read(run.checkpoint_key)
    except KeyNotFound:
        return None
    return Checkpoint.model_validate_json(data)
