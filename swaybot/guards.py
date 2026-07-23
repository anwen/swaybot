"""Workspace scope guards: path confinement and command allowlisting.

This module holds the guard layer of security. ``PathGuard`` confines
filesystem paths to a workspace root; ``CommandGuard`` restricts shell
commands to an allowlist of safe characters/commands. Both raise
``SecurityError`` (defined in ``violations``) on violation.
"""

import os
import re
import shlex
from pathlib import Path

from .violations import SecurityError


class PathGuard:
    """Confine filesystem paths to a workspace root."""

    def __init__(self, root: Path | str | None = None) -> None:
        if root is None:
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
