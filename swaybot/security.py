"""Security hardening: workspace scope, path traversal guard, escalation."""

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .storage import JSONBackend, StorageBackend


class SecurityError(ValueError):
    """Raised when a security policy is violated."""


class SecurityEscalation(SecurityError):
    """Raised after repeated violations within a session."""


class PathGuard:
    """Confine filesystem paths to a workspace root."""

    def __init__(self, root: Path | str | None = None) -> None:
        if root is None:
            import os

            root = os.environ.get("SWAYBOT_WORKSPACE", ".swaybot_workspace")
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, path: str) -> Path:
        """Resolve ``path`` inside the workspace or raise SecurityError."""
        target = (self.root / path).resolve()
        if not str(target).startswith(str(self.root)):
            raise SecurityError(
                f"path '{path}' escapes workspace '{self.root}'"
            )
        return target


class CommandGuard:
    """Restrict shell commands to an allowlist and safe characters."""

    ALLOWED = {"ls", "cat", "grep", "find", "pwd", "echo", "wc", "head", "tail"}
    FORBIDDEN = re.compile(r"[;|&$`\n\r]")

    def validate(self, command: str) -> list[str]:
        if self.FORBIDDEN.search(command):
            raise SecurityError("shell metacharacters are not allowed")
        try:
            parts = shlex.split(command)
        except ValueError as exc:
            raise SecurityError(str(exc)) from exc
        if not parts:
            raise SecurityError("empty command")
        if parts[0] not in self.ALLOWED:
            raise SecurityError(
                f"command '{parts[0]}' is not in the allowed list"
            )
        return parts


@runtime_checkable
class ViolationStore(Protocol):
    """Persists security violation counts per principal/session."""

    def get(self, session_id: str) -> int:
        ...

    def set(self, session_id: str, count: int) -> None:
        ...

    def reset(self, session_id: str) -> None:
        ...


class JSONViolationStore:
    """Store violation counts in a single JSON file."""

    def __init__(
        self,
        path: Path | str | None = None,
        backend: StorageBackend | None = None,
    ) -> None:
        if backend is not None:
            self.backend = backend
            self._key = "violations"
        elif path is not None:
            p = Path(path)
            self.backend = JSONBackend(p.parent)
            self._key = p.stem
        else:
            self.backend = JSONBackend(Path.home() / ".swaybot")
            self._key = "violations"

    def _load(self) -> dict[str, int]:
        data = self.backend.load(self._key)
        if isinstance(data, dict):
            return {k: int(v) for k, v in data.items() if isinstance(v, int)}
        return {}

    def _save(self, data: dict[str, int]) -> None:
        self.backend.save(self._key, data)

    def get(self, session_id: str) -> int:
        return self._load().get(session_id, 0)

    def set(self, session_id: str, count: int) -> None:
        data = self._load()
        data[session_id] = count
        self._save(data)

    def reset(self, session_id: str) -> None:
        data = self._load()
        data.pop(session_id, None)
        self._save(data)


@dataclass
class InMemoryViolationStore:
    """In-memory violation store for tests and transient use."""

    _data: dict[str, int] = field(default_factory=dict)

    def get(self, session_id: str) -> int:
        return self._data.get(session_id, 0)

    def set(self, session_id: str, count: int) -> None:
        self._data[session_id] = count

    def reset(self, session_id: str) -> None:
        self._data.pop(session_id, None)


class SecurityManager:
    """Track violations per session and escalate after repeated abuse."""

    DEFAULT_THRESHOLD = 3

    def __init__(
        self,
        path_guard: PathGuard | None = None,
        command_guard: CommandGuard | None = None,
        threshold: int = DEFAULT_THRESHOLD,
        store: ViolationStore | None = None,
    ) -> None:
        self.path_guard = path_guard or PathGuard()
        self.command_guard = command_guard or CommandGuard()
        self.threshold = threshold
        # Default to in-memory store to keep tests isolated and avoid
        # accidental file pollution.
        self.store = store or InMemoryViolationStore()

    def check_path(self, session_id: str, path: str) -> Path:
        """Validate a path, recording any violation for ``session_id``."""
        try:
            return self.path_guard.resolve(path)
        except SecurityError as exc:
            self.record_violation(session_id)
            raise

    def check_command(self, session_id: str, command: str) -> list[str]:
        """Validate a command, recording any violation for ``session_id``."""
        try:
            return self.command_guard.validate(command)
        except SecurityError as exc:
            self.record_violation(session_id)
            raise

    def record_violation(self, session_id: str) -> None:
        """Increment the violation count for ``session_id``."""
        self.store.set(
            session_id, self.store.get(session_id) + 1
        )

    def is_escalated(self, session_id: str) -> bool:
        return self.store.get(session_id) >= self.threshold

    def reset(self, session_id: str) -> None:
        self.store.reset(session_id)

    def status(self, session_id: str) -> dict[str, Any]:
        count = self.store.get(session_id)
        return {
            "violations": count,
            "threshold": self.threshold,
            "escalated": count >= self.threshold,
        }
