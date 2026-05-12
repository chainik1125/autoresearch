"""Findings forum — append-only structured records of what an agent observed.

Findings are the durable artifact of a research run. Code is means; findings are
the cross-agent communication channel. v1 only has single-agent transfers, but the
data model is set up for cross-run sharing later.

Each finding is its own S3 object with a unique sortable key — no concurrent
mutation of a single file, no merge conflicts. Sort order is chronological.

Finding types follow W2S's vocabulary:
  - result:      a measurement / numeric output
  - observation: something the runner or agent noticed
  - hypothesis:  a conjecture worth testing
  - insight:     an interpretation or conclusion
  - error:       something went wrong
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from autoresearch.backends.storage import StorageBackend
from autoresearch.backends.storage.base import append as storage_append
from autoresearch.core.run import Run


class FindingType(StrEnum):
    RESULT = "result"
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    INSIGHT = "insight"
    ERROR = "error"


class Finding(BaseModel):
    type: FindingType
    body: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


def append(
    storage: StorageBackend,
    run: Run,
    type: FindingType,
    body: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Append a finding to the run's findings prefix; return the storage key."""
    finding = Finding(type=type, body=body, metadata=metadata or {})
    return storage_append(storage, run.findings_prefix, finding.model_dump_json().encode("utf-8"))


def list_findings(
    storage: StorageBackend,
    run: Run,
    *,
    since_cursor: str | None = None,
) -> list[Finding]:
    """List findings for a run, oldest first. If `since_cursor` is given (a key),
    only return findings strictly after it."""
    keys = storage.list(run.findings_prefix)
    if since_cursor is not None:
        keys = [k for k in keys if k > since_cursor]
    return [Finding.model_validate_json(storage.read(k)) for k in keys]


def latest_key(storage: StorageBackend, run: Run) -> str | None:
    """Return the most recent finding key, or None if there are no findings."""
    keys = storage.list(run.findings_prefix)
    return keys[-1] if keys else None
