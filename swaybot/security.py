"""Security hardening: workspace scope, path traversal guard, escalation."""

import re
import shlex
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable

from .logging import log_event
from .request_context import RequestContext, current_context
from .storage import StorageBackend, _resolve_file_backend


class SecurityError(ValueError):
    """Raised when a security policy is violated."""


class SecurityEscalation(SecurityError):
    """Raised after repeated violations within a session."""


class ToolPermissionError(SecurityError):
    """Raised when a tool is blocked by the active permission policy."""


class PermissionMode(str, Enum):
    """Coarse permission modes for agent execution.

    - ``read-only``: only reads and safe queries.
    - ``workspace-write``: read-only plus file writes within the workspace.
    - ``danger``: arbitrary shell execution and other high-risk tools.
    """

    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"
    DANGER = "danger"


_PERMISSION_RANK = {
    PermissionMode.READ_ONLY: 0,
    PermissionMode.WORKSPACE_WRITE: 1,
    PermissionMode.DANGER: 2,
}

_TOOL_PERMISSION_RANK = {"read": 0, "write": 1, "danger": 2}


class PermissionPolicy:
    """Named permission mode with optional tool allow/deny lists."""

    def __init__(
        self,
        mode: PermissionMode | str = PermissionMode.WORKSPACE_WRITE,
        denied_tools: Iterable[str] | None = None,
        allowed_tools: Iterable[str] | None = None,
    ) -> None:
        self.mode = PermissionMode(mode)
        self.denied_tools: set[str] = set(denied_tools or ())
        self.allowed_tools: set[str] | None = (
            set(allowed_tools) if allowed_tools is not None else None
        )

    def allows(self, tool_name: str, required_permission: str) -> tuple[bool, str]:
        """Return (allowed, reason) for the given tool."""
        if tool_name in self.denied_tools:
            return False, f"tool '{tool_name}' is denied by policy"
        if self.allowed_tools is not None and tool_name not in self.allowed_tools:
            return False, f"tool '{tool_name}' is not in the allowed list"
        required_rank = _TOOL_PERMISSION_RANK.get(required_permission)
        if required_rank is None:
            return False, f"unknown required permission '{required_permission}'"
        if required_rank > _PERMISSION_RANK[self.mode]:
            return False, (
                f"tool '{tool_name}' requires {required_permission} permission "
                f"(current mode: {self.mode.value})"
            )
        return True, ""

    def __repr__(self) -> str:
        return (
            f"PermissionPolicy(mode={self.mode.value!r}, "
            f"denied_tools={sorted(self.denied_tools)!r}, "
            f"allowed_tools={sorted(self.allowed_tools) if self.allowed_tools is not None else None!r})"
        )


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
        if not target.is_relative_to(self.root):
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

    def check_path(
        self,
        session_id: str,
        path: str,
        request_context: RequestContext | None = None,
    ) -> Path:
        """Validate a path, recording any violation for ``session_id``."""
        ctx = request_context or current_context()
        try:
            return self.path_guard.resolve(path)
        except SecurityError as exc:
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
        except SecurityError as exc:
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
