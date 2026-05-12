"""Logs — append-only text log streams per run.

Logs are coarse-grained: each `append` writes a chunk (a multi-line block of text
captured during one runner step, or one tool call's stdout). Chunks land at unique
sortable keys so concurrent writers don't race. `tail` returns the last N lines
across all chunks, oldest-to-newest.
"""

from __future__ import annotations

from autoresearch.backends.storage import StorageBackend
from autoresearch.backends.storage.base import append as storage_append
from autoresearch.core.run import Run


def append(storage: StorageBackend, run: Run, chunk: str) -> str:
    """Append a log chunk; return the storage key."""
    return storage_append(storage, run.logs_prefix, chunk.encode("utf-8"), suffix=".txt")


def tail(storage: StorageBackend, run: Run, lines: int = 200) -> str:
    """Return the last `lines` lines across all chunks."""
    keys = storage.list(run.logs_prefix)
    if not keys:
        return ""
    accumulated: list[str] = []
    for key in reversed(keys):
        chunk = storage.read(key).decode("utf-8", errors="replace")
        accumulated.append(chunk)
        joined = "".join(reversed(accumulated))
        if joined.count("\n") >= lines:
            break
    text = "".join(reversed(accumulated))
    split = text.splitlines()
    return "\n".join(split[-lines:])
