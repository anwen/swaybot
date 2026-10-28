"""Shared .env loader for experiment scripts (no secrets echoed)."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_env() -> None:
    for line in (REPO_ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())
    os.environ["TAU2_DATA_DIR"] = os.path.expanduser(
        "~/.cache/modelscope/datasets/evalscope--tau3-bench-data/snapshots/master"
    )
    os.environ.setdefault("SWAYBOT_LITELLM_TIMEOUT", "180")
