"""LocalStorage — filesystem-backed StorageBackend for local development."""

from __future__ import annotations

from pathlib import Path

from autoresearch.backends.storage.base import KeyNotFound


class LocalStorage:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        if key.startswith("/"):
            raise ValueError(f"key must not start with '/': {key!r}")
        return self.root / key

    def read(self, key: str) -> bytes:
        path = self._path(key)
        try:
            return path.read_bytes()
        except FileNotFoundError as e:
            raise KeyNotFound(key) from e

    def write(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)

    def list(self, prefix: str) -> list[str]:
        base = self._path(prefix) if prefix else self.root
        if not base.exists():
            return []
        keys: list[str] = []
        for p in base.rglob("*"):
            if p.is_file():
                keys.append(str(p.relative_to(self.root)).replace("\\", "/"))
        return sorted(keys)

    def delete(self, key: str) -> None:
        path = self._path(key)
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()
