"""Helpers for capping large tool outputs and offloading overflow to files."""

import uuid
from pathlib import Path


DEFAULT_MAX_CHARS = 50000
DEFAULT_MAX_LINES = 200


def _overflow_dir(root: Path) -> Path:
    path = root / ".swaybot" / "overflows"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_overflow(content: str | list[str], root: Path) -> Path:
    """Write ``content`` to a unique overflow file under ``root/.swaybot/overflows``.

    Returns the path relative to ``root``.
    """
    overflow = _overflow_dir(root)
    uid = uuid.uuid4().hex
    file_path = overflow / f"{uid}.txt"
    if isinstance(content, list):
        text = "\n".join(content)
    else:
        text = content
    file_path.write_text(text, encoding="utf-8")
    return file_path.relative_to(root)


def truncate_text(
    text: str,
    root: Path,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_lines: int = DEFAULT_MAX_LINES,
) -> str:
    """Return ``text`` if it fits; otherwise return a preview + overflow path."""
    lines = text.splitlines()
    if len(text) <= max_chars and len(lines) <= max_lines:
        return text
    overflow = write_overflow(text, root)
    preview_lines = lines[:max_lines]
    preview = "\n".join(preview_lines)[:max_chars]
    return f"{preview}\n\n... [truncated; full content saved to {overflow}]"


def truncate_lines(
    lines: list[str],
    root: Path,
    max_lines: int = DEFAULT_MAX_LINES,
) -> list[str]:
    """Return ``lines`` if they fit; otherwise return a preview + overflow marker."""
    if len(lines) <= max_lines:
        return lines
    overflow = write_overflow(lines, root)
    return lines[:max_lines] + [
        f"... [truncated; {len(lines)} total lines saved to {overflow}]"
    ]
