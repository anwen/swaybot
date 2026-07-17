"""Load project-level instruction files (AGENTS.md, .swaybot/rules.md, etc.)."""

import subprocess
from pathlib import Path


INSTRUCTION_FILES = ["AGENTS.md"]
SWAYBOT_RULES_DIR = Path(".swaybot")
SWAYBOT_RULES_FILES = ["rules.md"]


def _find_git_root(start: Path) -> Path | None:
    """Return the git root for ``start`` if inside a git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(start),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return Path(result.stdout.strip()).resolve()
    except Exception:
        pass
    return None


def _walk_up(start: Path) -> list[Path]:
    """Return directories from ``start`` up to the filesystem root.

    If ``start`` is inside a git repository, stop at the git root.
    """
    start = start.resolve()
    directories = [start]
    git_root = _find_git_root(start)
    for parent in start.parents:
        directories.append(parent)
        if git_root is not None and parent == git_root:
            break
    return directories


def _collect_instruction_paths(root: Path) -> list[Path]:
    """Return instruction markdown files under ``root`` in load order.

    Root-level files are loaded first; deeper (more specific) files later.
    """
    paths: list[Path] = []
    for name in INSTRUCTION_FILES:
        candidate = root / name
        if candidate.is_file():
            paths.append(candidate)

    rules_dir = root / SWAYBOT_RULES_DIR
    if rules_dir.is_dir():
        for name in SWAYBOT_RULES_FILES:
            candidate = rules_dir / name
            if candidate.is_file():
                paths.append(candidate)
        # Also collect .swaybot/*.md in alphabetical order.
        for candidate in sorted(rules_dir.glob("*.md")):
            if candidate.name not in SWAYBOT_RULES_FILES and candidate.is_file():
                paths.append(candidate)
    return paths


def load_project_instructions(
    start_dir: str | Path | None = None,
    max_chars_per_file: int = 50000,
) -> str:
    """Collect project instruction markdown from ``start_dir`` up to repo root.

    Looks for ``AGENTS.md`` and ``.swaybot/rules.md`` / ``.swaybot/*.md``.
    Returns the concatenated markdown text, or an empty string if none found.
    """
    if start_dir is None:
        start_dir = Path.cwd()
    else:
        start_dir = Path(start_dir)

    if not start_dir.exists():
        return ""

    directories = _walk_up(start_dir)
    # Collect from root down to start_dir so more specific instructions come later.
    collected: list[tuple[Path, Path]] = []
    for directory in reversed(directories):
        for path in _collect_instruction_paths(directory):
            collected.append((directory, path))

    if not collected:
        return ""

    sections: list[str] = []
    for directory, path in collected:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if len(text) > max_chars_per_file:
            text = text[:max_chars_per_file] + "\n... [truncated]"
        rel = path.relative_to(directory) if path.is_relative_to(directory) else path
        sections.append(f"\n<!-- instructions from {rel} -->\n{text}\n")
    return "\n".join(sections).strip()
