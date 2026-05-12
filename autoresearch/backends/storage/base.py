"""StorageBackend protocol.

Storage is a flat key-value store. Keys are URL-style paths (e.g. "runs/abc/findings/0001.json").
All implementations are synchronous; wrap in asyncio.to_thread when calling from async code.

Keys never start with "/". Prefixes never end with "/". The append() helper enforces this.
"""

from __future__ import annotations

import secrets
import time
from typing import Protocol, runtime_checkable


class KeyNotFound(KeyError):
    """Raised when a key is read that does not exist."""


@runtime_checkable
class StorageBackend(Protocol):
    def read(self, key: str) -> bytes:
        """Return the bytes at `key`. Raises KeyNotFound if missing."""
        ...

    def write(self, key: str, data: bytes) -> None:
        """Write `data` to `key`, overwriting any existing value."""
        ...

    def list(self, prefix: str) -> list[str]:
        """Return all keys under `prefix`, sorted lexicographically."""
        ...

    def delete(self, key: str) -> None:
        """Delete `key`. No-op if missing."""
        ...

    def exists(self, key: str) -> bool:
        """Return True if `key` exists."""
        ...


def make_append_key(prefix: str, suffix: str = ".json") -> str:
    """Generate a unique sortable key under `prefix` suitable for append-only writes.

    Format: "{prefix}/{ns_timestamp}-{random_hex}{suffix}". The timestamp ensures
    chronological sort order; the random suffix avoids collisions on concurrent
    writers.
    """
    prefix = prefix.rstrip("/")
    ts_ns = time.time_ns()
    nonce = secrets.token_hex(4)
    return f"{prefix}/{ts_ns}-{nonce}{suffix}"


def append(backend: StorageBackend, prefix: str, data: bytes, suffix: str = ".json") -> str:
    """Append `data` under `prefix` at a unique sortable key; return the key.

    This is a free function rather than a Protocol method because every backend
    implements it identically — generate a unique key, then write.
    """
    key = make_append_key(prefix, suffix)
    backend.write(key, data)
    return key
