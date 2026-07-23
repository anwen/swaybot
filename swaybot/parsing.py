"""Small parsing helpers shared across brain/explorer modules.

These live in a leaf module (no internal dependencies) so that low-level
components like ``LLMBrain`` can reuse them without pulling in the agent
stack via ``explorer`` (which imports ``Agent``).
"""

import json
from typing import Any


def strip_markdown_fences(raw: str) -> str:
    """Remove leading/trailing ``` markdown fences if present."""
    raw = raw.strip()
    if not raw.startswith("```"):
        return raw
    lines = raw.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_exploration_response(raw: str) -> dict[str, Any]:
    """Parse a JSON response expected to contain task and hypothesis."""
    raw = strip_markdown_fences(raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    if not isinstance(data, dict):
        return {}

    return {
        "task": data.get("task", data.get("explore", "")),
        "hypothesis": data.get("hypothesis", ""),
    }
