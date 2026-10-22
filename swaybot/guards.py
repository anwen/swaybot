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
    """Tiered shell command policy, mirroring the permission modes.

    - ``read-only``: a small allowlist of read-only inspection commands;
      no metacharacters. Agents can look, not touch.
    - ``workspace-write``: the read-only set plus interpreters/test runners
      (python/python3/pytest), still one command per call — agents can run
      the code they wrote, but not chain commands or open pipes.
    - ``danger``: full shell (pipes, redirects, chaining). Parity with a
      yolo harness; the workspace cwd still applies.

    The allowlist tiers prevent casual damage, not determined escape —
    interpreters run with the process owner's privileges. Real isolation
    is a future sandbox concern, not a command list.
    """

    READ_ONLY_COMMANDS = frozenset(
        {"ls", "cat", "grep", "find", "pwd", "echo", "wc", "head", "tail"}
    )
    EXTENDED_COMMANDS = READ_ONLY_COMMANDS | frozenset(
        {"python", "python3", "pytest"}
    )
    MODES = ("read-only", "workspace-write", "danger")
    FORBIDDEN = re.compile(r"[;|&$`\n\r]")

    def __init__(self, mode: str = "read-only") -> None:
        self.set_mode(mode)

    def set_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unknown guard mode {mode!r}; use one of {self.MODES}")
        self.mode = mode

    @property
    def full_shell(self) -> bool:
        return self.mode == "danger"

    def validate(self, command: str) -> list[str]:
        if self.full_shell:
            if not command.strip():
                raise SecurityError("empty command")
            # Full shell executes via shell=True; no allowlist applies.
            return [command]
        if self.FORBIDDEN.search(command):
            raise SecurityError(
                f"shell metacharacters are not allowed in {self.mode} mode "
                "(they enable command chaining; danger mode allows them)"
            )
        try:
            parts = shlex.split(command)
        except ValueError as exc:
            raise SecurityError(str(exc)) from exc
        if not parts:
            raise SecurityError("empty command")
        allowed = (
            self.READ_ONLY_COMMANDS
            if self.mode == "read-only"
            else self.EXTENDED_COMMANDS
        )
        if parts[0] not in allowed:
            raise SecurityError(
                f"command '{parts[0]}' is not allowed in {self.mode} mode "
                f"(allowed: {', '.join(sorted(allowed))}). This denial is "
                "final for this command — adjust your approach or finish."
            )
        return parts
