"""Storage backend round-trip tests. Exercises LocalStorage and S3Storage identically."""

from __future__ import annotations

import pytest

from autoresearch.backends.storage import StorageBackend
from autoresearch.backends.storage.base import KeyNotFound, append


def test_write_then_read(storage: StorageBackend) -> None:
    storage.write("a/b/c.json", b'{"hello":"world"}')
    assert storage.read("a/b/c.json") == b'{"hello":"world"}'


def test_read_missing_raises(storage: StorageBackend) -> None:
    with pytest.raises(KeyNotFound):
        storage.read("does/not/exist.json")


def test_exists(storage: StorageBackend) -> None:
    assert not storage.exists("x.txt")
    storage.write("x.txt", b"hi")
    assert storage.exists("x.txt")


def test_delete(storage: StorageBackend) -> None:
    storage.write("x.txt", b"hi")
    storage.delete("x.txt")
    assert not storage.exists("x.txt")
    storage.delete("x.txt")  # idempotent


def test_list_returns_sorted(storage: StorageBackend) -> None:
    storage.write("p/2.json", b"2")
    storage.write("p/1.json", b"1")
    storage.write("p/3.json", b"3")
    storage.write("other/x.json", b"x")
    assert storage.list("p/") == ["p/1.json", "p/2.json", "p/3.json"]


def test_append_generates_unique_sorted_keys(storage: StorageBackend) -> None:
    keys = [append(storage, "stream", f"item-{i}".encode()) for i in range(50)]
    assert len(set(keys)) == 50  # all unique
    assert keys == sorted(keys)  # chronological order
    for key, expected in zip(keys, range(50), strict=True):
        assert storage.read(key) == f"item-{expected}".encode()
