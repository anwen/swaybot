"""Golden snapshot tests: prompts and tool schemas are locked byte-level.

Fixture-as-spec methodology (borrowed from zcode-cli): any change to the
system prompts or the default toolset must show up as an intentional
golden diff, never slip in silently. Regenerate with:

    UPDATE_GOLDEN=1 python -m pytest tests/test_prompt_snapshots.py
"""

import json
import os
from pathlib import Path

import pytest

from swaybot.prompts import render_prompt
from swaybot.tools import build_default_registry

GOLDEN_DIR = Path(__file__).parent / "golden"

_BASE = dict(
    behavior_guidance="",
    project_instructions="",
    skill_context="",
    available_tools=["echo", "final_answer"],
    tool_descriptions=[],
)

PROMPT_CASES = {
    "system_fc-bare": ("system_fc", dict(_BASE)),
    "system_fc-guidance": (
        "system_fc",
        dict(_BASE, behavior_guidance="- You already searched twice; answer now."),
    ),
    "system_fc-full": (
        "system_fc",
        dict(
            _BASE,
            project_instructions="Be terse.",
            skill_context="- skill: kb-lookup",
        ),
    ),
    "system-bare": ("system", dict(_BASE)),
    "system-schemas": (
        "system",
        dict(
            _BASE,
            tool_descriptions=[
                {
                    "name": "final_answer",
                    "description": "Provide the final answer.",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        ),
    ),
    "user-default": (
        "user",
        dict(task="count to three", step=1, max_steps=5, history=[]),
    ),
}


def _write_golden(name: str, content: str) -> None:
    GOLDEN_DIR.mkdir(exist_ok=True)
    (GOLDEN_DIR / f"{name}.txt").write_text(content, encoding="utf-8")


@pytest.mark.parametrize("case", sorted(PROMPT_CASES))
def test_prompt_snapshot(case):
    template, kwargs = PROMPT_CASES[case]
    text = render_prompt(template, **kwargs)
    golden = GOLDEN_DIR / f"prompt-{case}.txt"
    if os.environ.get("UPDATE_GOLDEN"):
        _write_golden(f"prompt-{case}", text)
    assert golden.exists(), (
        f"missing golden {golden.name}; run UPDATE_GOLDEN=1 pytest "
        "tests/test_prompt_snapshots.py to create it, then review the diff"
    )
    assert text == golden.read_text(encoding="utf-8"), (
        f"prompt '{template}' drifted from golden {golden.name}; if the change "
        "is intentional, regenerate goldens and record the rationale in the "
        "commit message"
    )


def test_tool_schema_snapshot():
    schemas = build_default_registry().schemas()
    text = json.dumps(schemas, ensure_ascii=False, indent=2, sort_keys=True)
    golden = GOLDEN_DIR / "tool-schemas.txt"
    if os.environ.get("UPDATE_GOLDEN"):
        _write_golden("tool-schemas", text)
    assert golden.exists(), (
        "missing golden tool-schemas.txt; run UPDATE_GOLDEN=1 pytest "
        "tests/test_prompt_snapshots.py"
    )
    assert text == golden.read_text(encoding="utf-8"), (
        "default toolset schemas drifted; if intentional (tool added/"
        "removed/renamed), regenerate goldens and say why in the commit"
    )
