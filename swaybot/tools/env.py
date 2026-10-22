"""On-demand environment info — the model's choice, not injected context.

The agent runs with filesystem and shell tools yet, by default, knows
nothing about where it is. Rather than paying prompt tokens to inject
environment context on every turn (and busting the stable cache prefix),
this tool lets the model ask when it actually needs to know.
"""

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import tool

_GIT_TIMEOUT_SECONDS = 3.0
_MAX_STATUS_CHARS = 2000
_MAX_RECENT_COMMITS = 5


def _git(args: list[str], cwd: Path) -> str | None:
    """Run a read-only git query; None on failure or timeout (git optional)."""
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _git_snapshot(cwd: Path) -> dict:
    inside = _git(["rev-parse", "--is-inside-work-tree"], cwd)
    if inside != "true":
        return {"is_git_repository": False}
    status = _git(["status", "--short"], cwd) or ""
    if len(status) > _MAX_STATUS_CHARS:
        status = status[:_MAX_STATUS_CHARS] + "…"
    commits = [
        line for line in (_git(["log", "--oneline", f"-n{_MAX_RECENT_COMMITS}"], cwd) or "").splitlines() if line
    ]
    return {
        "is_git_repository": True,
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd) or "HEAD",
        "git_status": "dirty" if status else "clean",
        "git_status_short": status,
        "git_recent_commits": commits,
    }


@tool(read_only=True)
def env_info() -> str:
    """Inspect the current working environment.

    Returns the working directory, platform and OS version, current UTC
    date/time, and — when inside a git repository — the branch, working-tree
    status, and the most recent commits. Call this when you need to know
    where you are working or what changed recently; the information is NOT
    provided automatically.
    """
    cwd = Path.cwd()
    info: dict = {
        "cwd": str(cwd),
        "platform": platform.system().lower(),
        "os_version": f"{platform.system()} {platform.release()} {platform.machine()}",
        "python": platform.python_version(),
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    shell = os.path.basename(os.environ.get("SHELL") or "")
    if shell:
        info["shell"] = shell
    info.update(_git_snapshot(cwd))
    return json.dumps(info, ensure_ascii=False, indent=2)
