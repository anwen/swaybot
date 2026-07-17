"""Security hardening: workspace scope, path traversal guard, escalation."""

import re
import shlex
from pathlib import Path
from typing import Any


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


class SecurityManager:
    """Track violations per session and escalate after repeated abuse."""

    DEFAULT_THRESHOLD = 3

    def __init__(
        self,
        path_guard: PathGuard | None = None,
        command_guard: CommandGuard | None = None,
        threshold: int = DEFAULT_THRESHOLD,
    ) -> None:
        self.path_guard = path_guard or PathGuard()
        self.command_guard = command_guard or CommandGuard()
        self.threshold = threshold
        self._violations: dict[str, int] = {}

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
        self._violations[session_id] = self._violations.get(session_id, 0) + 1

    def is_escalated(self, session_id: str) -> bool:
        return self._violations.get(session_id, 0) >= self.threshold

    def reset(self, session_id: str) -> None:
        self._violations.pop(session_id, None)

    def status(self, session_id: str) -> dict[str, Any]:
        count = self._violations.get(session_id, 0)
        return {
            "violations": count,
            "threshold": self.threshold,
            "escalated": count >= self.threshold,
        }
