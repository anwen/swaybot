"""Sandboxed filesystem tools scoped to a workspace root."""

import fnmatch
import os
from pathlib import Path

from ..security import PathGuard, SecurityError
from . import tool


class WorkspaceError(SecurityError):
    """Raised when a path escapes the allowed workspace."""


class FileSystem(PathGuard):
    """Filesystem operations confined to a workspace root."""

    def resolve(self, path: str) -> Path:
        try:
            return super().resolve(path)
        except SecurityError as exc:
            raise WorkspaceError(str(exc)) from exc


_fs = FileSystem()


@tool(read_only=True)
def read_file(path: str, max_length: int = 10000) -> str:
    """Read the contents of a file inside the workspace."""
    target = _fs.resolve(path)
    if not target.is_file():
        return f"Error: {path} is not a file"
    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Error: {exc}"
    if len(content) > max_length:
        content = content[:max_length] + "\n... [truncated]"
    return content


@tool(read_only=False, risk_level="medium")
def write_file(path: str, content: str) -> str:
    """Write ``content`` to a file inside the workspace."""
    target = _fs.resolve(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {path}"
    except Exception as exc:
        return f"Error: {exc}"


@tool(read_only=True)
def list_directory(path: str = ".") -> list[str]:
    """List files and directories inside ``path``."""
    target = _fs.resolve(path)
    if not target.is_dir():
        return [f"Error: {path} is not a directory"]
    try:
        return sorted(
            f"{p.name}/" if p.is_dir() else p.name
            for p in target.iterdir()
        )
    except Exception as exc:
        return [f"Error: {exc}"]


@tool(read_only=True)
def search_files(query: str, path: str = ".") -> list[str]:
    """Search file names matching ``query`` (glob) under ``path``."""
    target = _fs.resolve(path)
    if not target.is_dir():
        return [f"Error: {path} is not a directory"]
    matches: list[str] = []
    for root, _dirs, files in os.walk(target):
        rel_root = Path(root).relative_to(_fs.root)
        for name in files:
            if fnmatch.fnmatch(name.lower(), query.lower()):
                matches.append(str(rel_root / name))
    return sorted(matches)
