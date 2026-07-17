"""Sandboxed filesystem tools scoped to a workspace root."""

import fnmatch
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from ..code_quality import run_quality_hooks
from ..output_limits import truncate_lines, truncate_text
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
    return truncate_text(content, _fs.root, max_chars=max_length, max_lines=200)


@tool(read_only=False, risk_level="medium")
def write_file(path: str, content: str) -> str:
    """Write ``content`` to a file inside the workspace."""
    target = _fs.resolve(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except Exception as exc:
        return f"Error: {exc}"
    result = f"Wrote {len(content)} characters to {path}"
    quality = run_quality_hooks(target)
    if quality:
        result += f"\n\n{quality}"
    return result


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


@tool(read_only=False, risk_level="medium")
def edit_file(
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """Replace ``old_string`` with ``new_string`` in ``path``.

    If ``old_string`` is not found, the file has changed since it was read
    (stale content). If it appears more than once and ``replace_all`` is
    False, the edit is rejected to avoid ambiguous replacements.
    """
    target = _fs.resolve(path)
    try:
        current = target.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Error: {exc}"
    occurrences = current.count(old_string)
    if occurrences == 0:
        return f"StaleContentError: old_string not found in {path}"
    if not replace_all and occurrences > 1:
        return (
            f"Error: old_string appears {occurrences} times in {path}; "
            "set replace_all=true to replace all occurrences"
        )
    new_content = current.replace(old_string, new_string) if replace_all else current.replace(old_string, new_string, 1)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(new_content, encoding="utf-8")
    except Exception as exc:
        return f"Error: {exc}"
    result = f"Edited {path}: replaced {occurrences} occurrence(s)"
    quality = run_quality_hooks(target)
    if quality:
        result += f"\n\n{quality}"
    return result


@tool(read_only=True)
def grep(pattern: str, path: str = ".", max_results: int = 50) -> list[str]:
    """Search file contents for ``pattern`` (regex) under ``path``.

    Uses ripgrep (``rg --json``) when available, otherwise falls back to a
    stdlib recursive search. Results are formatted as ``file:line: text``.
    Large result sets are truncated and the full set is saved to an overflow
    file under ``.swaybot/overflows``.
    """
    target = _fs.resolve(path)
    if not target.exists():
        return [f"Error: {path} does not exist"]

    collect_limit = max(max_results * 10, 500)

    if shutil.which("rg"):
        try:
            cmd = [
                "rg",
                "--json",
                "-n",
                "--max-count",
                str(collect_limit),
                "--",
                pattern,
            ]
            if target.is_dir():
                cmd.append(".")
            else:
                cmd.append(str(target.relative_to(_fs.root)))
            proc = subprocess.run(
                cmd,
                cwd=str(_fs.root),
                capture_output=True,
                text=True,
                timeout=30,
                shell=False,
            )
            results: list[str] = []
            for line in proc.stdout.splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") != "match":
                    continue
                data = obj.get("data", {})
                file_text = data.get("path", {}).get("text", "")
                line_no = data.get("line_number", 0)
                text = data.get("lines", {}).get("text", "")
                results.append(f"{file_text}:{line_no}: {text.rstrip(chr(10))}")
                if len(results) >= collect_limit:
                    break
            return truncate_lines(results, _fs.root, max_results)
        except Exception:
            pass

    regex = re.compile(pattern)
    results = []
    files = [target] if target.is_file() else [p for p in target.rglob("*") if p.is_file()]
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        rel = f.relative_to(_fs.root)
        for i, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                results.append(f"{rel}:{i}: {line}")
                if len(results) >= collect_limit:
                    break
        if len(results) >= collect_limit:
            break
    return truncate_lines(results, _fs.root, max_results)
