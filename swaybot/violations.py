"""Violation tracking and security exceptions.

This module holds the exception hierarchy (``SecurityError`` and
subclasses) and the violation-counting layer: ``ViolationStore`` persists
per-session violation counts, and ``SecurityManager`` records violations,
logs them, and decides when a session has escalated. The permission
policy lives in ``permission`` and the workspace/command guards in
``guards``.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .logging import log_event
from .request_context import RequestContext, current_context
from .storage import StorageBackend, _resolve_file_backend


class SecurityError(ValueError):
    """Raised when a security policy is violated."""


class SecurityEscalation(SecurityError):
    """Raised after repeated violations within a session."""


class ToolPermissionError(SecurityError):
    """Raised when a tool is blocked by the active permission policy."""


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
        self.backend, self._key = _resolve_file_backend(
            backend,
            path,
            Path.home() / ".swaybot",
            "violations",
        )

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
        path_guard: Any | None = None,
        command_guard: Any | None = None,
        threshold: int = DEFAULT_THRESHOLD,
        store: ViolationStore | None = None,
    ) -> None:
        # Imported lazily to avoid a guard -> violations -> guard import
        # cycle; guards import SecurityError from this module.
        from .guards import CommandGuard, PathGuard

        self.path_guard = path_guard or PathGuard()
        self.command_guard = command_guard or CommandGuard()
        self.threshold = threshold
        # Default to in-memory store to keep tests isolated and avoid
        # accidental file pollution.
        self.store = store or InMemoryViolationStore()

    def check_path(
        self,
        session_id: str,
        path: str,
        request_context: RequestContext | None = None,
    ) -> Any:
        """Validate a path, recording any violation for ``session_id``."""
        ctx = request_context or current_context()
        try:
            return self.path_guard.resolve(path)
        except SecurityError:
            self.record_violation(session_id, ctx)
            raise

    def check_command(
        self,
        session_id: str,
        command: str,
        request_context: RequestContext | None = None,
    ) -> list[str]:
        """Validate a command, recording any violation for ``session_id``."""
        ctx = request_context or current_context()
        try:
            return self.command_guard.validate(command)
        except SecurityError:
            self.record_violation(session_id, ctx)
            raise

    def record_violation(
        self,
        session_id: str,
        request_context: RequestContext | None = None,
    ) -> None:
        """Increment the violation count for ``session_id``."""
        ctx = request_context or current_context()
        self.store.set(
            session_id, self.store.get(session_id) + 1
        )
        log_event(
            "security_violation",
            session_id=session_id,
            principal=ctx.principal,
            request_id=ctx.request_id,
            violations=self.store.get(session_id),
            threshold=self.threshold,
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
