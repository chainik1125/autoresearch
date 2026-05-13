"""Run — the central record describing a single workflow execution.

A Run is created by the controller's dispatcher (or by `autoresearch run` for local
development). It is persisted to `runs/{run_id}/run.json` and read by everything
that needs to know what's happening: the supervisor, the pipeline runner on the pod,
the MCP surface answering queries.

The Run is single-writer: only the active pod's runner mutates it (with the dispatcher
performing the initial write at creation). Concurrent writers would be a bug.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from autoresearch.backends.storage.base import StorageBackend


class RunStatus(StrEnum):
    QUEUED = "queued"
    LOADING = "loading"
    RUNNING = "running"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class Run(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    workflow: str
    pipeline_name: str
    params: dict[str, Any] = Field(default_factory=dict)
    status: RunStatus = RunStatus.QUEUED
    pod_handle: str | None = None
    budget_cap_usd: float = 0.0
    budget_spent_usd: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_heartbeat_at: datetime | None = None
    last_error: str | None = None
    # The Run that spawned this one — set when a prep / mechanic / postflight
    # agent (or a recursive /transfer call) dispatches downstream work. Lets
    # the agent tree be reconstructed for introspection without inventing a
    # separate edges table. None at the root (the user's initial dispatch).
    parent_run_id: str | None = None

    @property
    def key(self) -> str:
        return f"runs/{self.id}/run.json"

    @property
    def findings_prefix(self) -> str:
        return f"runs/{self.id}/findings"

    @property
    def logs_prefix(self) -> str:
        return f"runs/{self.id}/logs"

    @property
    def checkpoint_key(self) -> str:
        return f"runs/{self.id}/checkpoint.json"

    @property
    def heartbeat_key(self) -> str:
        return f"runs/{self.id}/heartbeat.json"

    @property
    def result_key(self) -> str:
        return f"runs/{self.id}/result.json"

    def save(self, storage: StorageBackend) -> None:
        storage.write(self.key, self.model_dump_json().encode("utf-8"))

    @classmethod
    def load(cls, storage: StorageBackend, run_id: str) -> Run:
        data = storage.read(f"runs/{run_id}/run.json")
        return cls.model_validate_json(data)

    @classmethod
    def list_all(cls, storage: StorageBackend) -> list[Run]:
        keys = storage.list("runs/")
        runs: list[Run] = []
        for key in keys:
            if key.endswith("/run.json"):
                runs.append(cls.model_validate_json(storage.read(key)))
        return runs
