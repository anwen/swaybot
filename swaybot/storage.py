"""Pluggable storage backends for sessions, memory, runs, and violations."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class StorageError(ValueError):
    """Raised when a storage operation fails."""


@runtime_checkable
class StorageBackend(Protocol):
    """Abstract storage for keyed records.

    Implementations may be file-based (JSON, JSONL) or remote (SQLite, S3,
    Redis). The protocol intentionally avoids SQL specifics so the core
    agent code stays backend-agnostic.
    """

    def create(self, key: str) -> None:
        """Ensure the storage location for ``key`` exists."""
        ...

    def exists(self, key: str) -> bool:
        ...

    def delete(self, key: str) -> None:
        ...

    def list_keys(self, prefix: str = "") -> list[str]:
        """Return all keys starting with ``prefix`` (without extension)."""
        ...

    def load(self, key: str) -> Any:
        """Load the entire value stored at ``key``."""
        ...

    def save(self, key: str, data: Any) -> None:
        """Overwrite the value at ``key``."""
        ...

    def append(self, key: str, record: dict[str, Any]) -> None:
        """Append ``record`` to a stream or list at ``key``."""
        ...

    def load_stream(
        self, key: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Load a list of records from ``key`` in order."""
        ...

    def query(self, key: str, filters: dict[str, Any]) -> list[dict[str, Any]]:
        """Return records at ``key`` matching simple equality filters."""
        ...


class _BaseFileBackend:
    """Shared filename sanitization and directory handling."""

    EXTENSION: str = ""

    def __init__(self, base_dir: Path | str) -> None:
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _safe_key(cls, key: str) -> str:
        if not key:
            raise StorageError("key must not be empty")
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", key)
        if safe != key:
            raise StorageError(f"invalid key: {key!r}")
        return safe

    def _path(self, key: str) -> Path:
        safe = self._safe_key(key)
        path = (self.base_dir / f"{safe}{self.EXTENSION}").resolve()
        if not str(path).startswith(str(self.base_dir)):
            raise StorageError(f"key path escapes base_dir: {key}")
        return path

    def create(self, key: str) -> None:
        self._path(key).touch(exist_ok=True)

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def list_keys(self, prefix: str = "") -> list[str]:
        keys: list[str] = []
        for path in sorted(self.base_dir.glob(f"*{self.EXTENSION}")):
            name = path.stem
            if name.startswith(prefix):
                keys.append(name)
        return keys


class JSONBackend(_BaseFileBackend):
    """Store each key as a single JSON file."""

    EXTENSION = ".json"

    def load(self, key: str) -> Any:
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, key: str, data: Any) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def append(self, key: str, record: dict[str, Any]) -> None:
        data = self.load(key)
        if data is None:
            data = [record]
        elif isinstance(data, list):
            data.append(record)
        else:
            raise StorageError(
                f"cannot append to non-list JSON at key {key!r}"
            )
        self.save(key, data)

    def load_stream(
        self, key: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        data = self.load(key)
        if data is None:
            return []
        if isinstance(data, dict):
            records = [data]
        elif isinstance(data, list):
            records = data
        else:
            return []
        if limit is not None:
            records = records[-limit:]
        return records

    def query(self, key: str, filters: dict[str, Any]) -> list[dict[str, Any]]:
        records = self.load_stream(key)
        return [
            r
            for r in records
            if isinstance(r, dict)
            and all(r.get(k) == v for k, v in filters.items())
        ]


class JSONLBackend(_BaseFileBackend):
    """Store each key as an append-only JSON-lines file."""

    EXTENSION = ".jsonl"

    def load(self, key: str) -> list[dict[str, Any]]:
        return self.load_stream(key)

    def save(self, key: str, data: Any) -> None:
        records = data if isinstance(data, list) else [data]
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def append(self, key: str, record: dict[str, Any]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load_stream(
        self, key: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        path = self._path(key)
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
        if limit is not None:
            records = records[-limit:]
        return records

    def query(self, key: str, filters: dict[str, Any]) -> list[dict[str, Any]]:
        records = self.load_stream(key)
        return [
            r
            for r in records
            if all(r.get(k) == v for k, v in filters.items())
        ]


@dataclass
class InMemoryBackend:
    """In-memory backend for tests and transient use."""

    _data: dict[str, Any] = field(default_factory=dict)

    def create(self, key: str) -> None:
        self._safe_key(key)
        self._data.setdefault(key, None)

    @staticmethod
    def _safe_key(key: str) -> str:
        if not key:
            raise StorageError("key must not be empty")
        return key

    def exists(self, key: str) -> bool:
        self._safe_key(key)
        return key in self._data and self._data[key] is not None

    def delete(self, key: str) -> None:
        self._safe_key(key)
        self._data.pop(key, None)

    def list_keys(self, prefix: str = "") -> list[str]:
        return sorted(k for k in self._data if k.startswith(prefix))

    def load(self, key: str) -> Any:
        self._safe_key(key)
        return self._data.get(key)

    def save(self, key: str, data: Any) -> None:
        self._safe_key(key)
        self._data[key] = data

    def append(self, key: str, record: dict[str, Any]) -> None:
        self._safe_key(key)
        current = self._data.get(key)
        if current is None:
            current = []
        if not isinstance(current, list):
            raise StorageError(f"cannot append to non-list at key {key!r}")
        current.append(record)
        self._data[key] = current

    def load_stream(
        self, key: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        data = self.load(key)
        if data is None:
            return []
        if isinstance(data, dict):
            records = [data]
        elif isinstance(data, list):
            records = list(data)
        else:
            return []
        if limit is not None:
            records = records[-limit:]
        return records

    def query(self, key: str, filters: dict[str, Any]) -> list[dict[str, Any]]:
        records = self.load_stream(key)
        return [
            r
            for r in records
            if isinstance(r, dict)
            and all(r.get(k) == v for k, v in filters.items())
        ]


def default_session_backend(base_dir: Path | str | None = None) -> JSONLBackend:
    if base_dir is None:
        base_dir = Path.home() / ".swaybot" / "sessions"
    return JSONLBackend(base_dir)


def default_memory_backend(path: Path | str | None = None) -> JSONBackend:
    if path is None:
        path = Path.home() / ".swaybot" / "memory.json"
    return JSONBackend(Path(path).parent)
