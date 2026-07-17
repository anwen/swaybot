"""Sandboxed shell tool confined to the workspace."""

import subprocess
from pathlib import Path

from ..output_limits import truncate_text
from ..security import CommandGuard, PathGuard, SecurityError
from . import tool


class ShellError(SecurityError):
    """Raised for disallowed shell commands or workspace escapes."""


class Shell:
    """Run a restricted set of commands inside the workspace."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.path_guard = PathGuard(root)
        self.command_guard = CommandGuard()

    @property
    def root(self) -> Path:
        return self.path_guard.root

    def resolve(self, cwd: str) -> Path:
        try:
            return self.path_guard.resolve(cwd)
        except SecurityError as exc:
            raise ShellError(str(exc)) from exc

    def run(self, command: str, cwd: str = ".") -> str:
        try:
            parts = self.command_guard.validate(command)
        except SecurityError as exc:
            raise ShellError(str(exc)) from exc
        workdir = self.resolve(cwd)
        try:
            result = subprocess.run(
                parts,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=30,
                shell=False,
            )
            output = result.stdout
            if result.stderr:
                output += "\n" + result.stderr
            return output.strip() or f"exit code {result.returncode}"
        except subprocess.TimeoutExpired:
            return "Error: command timed out"
        except Exception as exc:
            return f"Error: {exc}"


_shell = Shell()


@tool(read_only=True, risk_level="high")
def run_shell_command(command: str, cwd: str = ".") -> str:
    """Run an allowed shell command inside the workspace."""
    output = _shell.run(command, cwd)
    return truncate_text(output, _shell.path_guard.root)
