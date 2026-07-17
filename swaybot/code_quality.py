"""Optional post-write code quality checks for file edits."""

import shutil
import subprocess
from pathlib import Path


DEFAULT_PYTHON_HOOKS = [
    ("format", ["ruff", "format", "{path}"]),
    ("lint", ["ruff", "check", "{path}"]),
    ("typecheck", ["pyright", "{path}"]),
]


def _command_available(parts: list[str]) -> bool:
    return shutil.which(parts[0]) is not None


def run_quality_hooks(
    path: Path,
    hooks: list[tuple[str, list[str]]] | None = None,
    timeout: float = 30.0,
) -> str:
    """Run optional formatter/linter/typechecker on ``path``.

    Only runs commands that are installed. Returns a human-readable report
    with command output, or an empty string if no hooks ran or all passed
    silently.
    """
    if hooks is None:
        suffix = path.suffix.lower()
        if suffix != ".py":
            return ""
        hooks = DEFAULT_PYTHON_HOOKS

    reports: list[str] = []
    for name, template in hooks:
        command = [part.format(path=str(path)) for part in template]
        if not _command_available(command):
            continue
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,
            )
        except (subprocess.TimeoutExpired, Exception) as exc:
            reports.append(f"{name}: failed ({exc})")
            continue
        output = (proc.stdout or "").strip()
        errors = (proc.stderr or "").strip()
        if proc.returncode == 0 and not output and not errors:
            continue
        header = f"{name}: {'passed' if proc.returncode == 0 else 'failed'}"
        parts = [header]
        if output:
            parts.append(output)
        if errors:
            parts.append(errors)
        reports.append("\n".join(parts))

    if not reports:
        return ""
    return "\n\n".join(reports)
