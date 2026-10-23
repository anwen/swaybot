"""Sandboxed shell tool confined to the workspace."""

import re
import subprocess
from pathlib import Path

from ..guards import CommandGuard, PathGuard
from ..violations import SecurityError
from .output_limits import truncate_text
from . import tool


class ShellError(SecurityError):
    """Raised for disallowed shell commands or workspace escapes."""


class Shell:
    """Run commands inside the workspace under a tiered CommandGuard."""

    def __init__(self, root: Path | str | None = None, mode: str = "read-only") -> None:
        self.path_guard = PathGuard(root)
        self.command_guard = CommandGuard(mode)

    @property
    def root(self) -> Path:
        return self.path_guard.root

    def set_guard_mode(self, mode: str) -> None:
        self.command_guard.set_mode(mode)

    def resolve(self, cwd: str) -> Path:
        try:
            return self.path_guard.resolve(cwd)
        except SecurityError as exc:
            raise ShellError(str(exc)) from exc

    def run(self, command: str, cwd: str = ".", timeout: int = 60) -> str:
        try:
            parts = self.command_guard.validate(command)
        except SecurityError as exc:
            raise ShellError(str(exc)) from exc
        workdir = self.resolve(cwd)
        full_shell = self.command_guard.full_shell
        if not full_shell:
            self._validate_arguments(parts)
        try:
            result = subprocess.run(
                command if full_shell else parts,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=max(1, min(timeout, 600)),
                shell=full_shell,
            )
            output = result.stdout
            if result.stderr:
                output += "\n" + result.stderr
            if result.returncode != 0 and not output.strip():
                return f"Error: exit code {result.returncode}"
            return output.strip() or f"exit code {result.returncode}"
        except subprocess.TimeoutExpired:
            return "Error: command timed out"
        except Exception as exc:
            return f"Error: {exc}"

    @staticmethod
    def _validate_arguments(parts: list[str]) -> None:
        """Reject arguments that escape the workspace or invoke sub-commands."""
        if not parts:
            return
        for arg in parts[1:]:
            if arg.startswith("/"):
                raise ShellError("absolute paths are not allowed")
            if ".." in arg:
                raise ShellError("path traversal is not allowed")
        if parts[0] == "find":
            forbidden = {"-exec", "-execdir", "-ok", "-okdir", "-delete"}
            for arg in parts[1:]:
                if arg in forbidden:
                    raise ShellError(
                        f"find action '{arg}' is not allowed"
                    )


# 环境探测（env/env 命令）会把 API key 一并带进工具输出，随后进入
# 观察、记忆、rollout——生命实验 run9 实际发生了。在源头抹掉敏感值。
_SECRET_ENV_RE = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:API_)?(?:KEY|TOKEN|SECRET|PASSWORD)[A-Z0-9_]*)=(\S+)"
)


def _scrub_secrets(output: str) -> str:
    return _SECRET_ENV_RE.sub(r"\1=[REDACTED]", output)


def create_shell_tool(root: Path | str | None = None):
    """Build a run_shell_command tool bound to its own Shell instance.

    One per registry so each Agent's permission policy can set the guard
    mode independently (the old module-level instance shared state across
    agents). The Shell is exposed on the Tool as ``_shell`` for that wiring.
    """

    shell = Shell(root)

    @tool(read_only=False, risk_level="medium")
    def run_shell_command(command: str, cwd: str = ".", timeout: int = 60) -> str:
        """Run a shell command inside the workspace.

        What is allowed depends on the permission mode: read-only allows a
        few inspection commands; workspace-write additionally runs
        python/python3/pytest (one command per call, no pipes); danger
        allows the full shell including pipes and redirection. Prefer
        ``python3 <script>`` over inline code; give long-running commands a
        larger ``timeout`` (seconds, max 600).
        """
        output = _scrub_secrets(shell.run(command, cwd, timeout))
        return truncate_text(output, shell.path_guard.root)

    run_shell_command._shell = shell  # type: ignore[attr-defined]
    return run_shell_command
