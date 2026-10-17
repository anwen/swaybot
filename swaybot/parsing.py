"""Small parsing helpers shared across brain/explorer modules.

These live in a leaf module (no internal dependencies) so that low-level
components like ``LLMBrain`` can reuse them without pulling in the agent
stack via ``explorer`` (which imports ``Agent``).
"""

import json
import re
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


# Matches prose-embedded tool tags the model emits instead of native tool
# calls, e.g. ``<KB_search_bm25 query="gold card">``. Kept in sync with
# Variant 3 in extract_tool_call: any ``<Name ...>`` / ``<Name .../>`` /
# ``</Name>`` tag.
PROSE_TOOL_TAG_RE = re.compile(
    r"</?[A-Za-z_][\w-]*(?:\s+[^<>]*?)?\s*/?>", re.DOTALL
)


def extract_tool_call(
    raw: str, available_tools: list[str] | None = None
) -> dict[str, Any] | None:
    """Normalize Qwen/DeepSeek-style XML tool calls into JSON actions.

    Models fine-tuned for ``<tool_call>`` formatting sometimes emit either
    ``<tool_call>{"name": ..., "args": ...}</tool_call>`` or a nested
    ``<function=name><parameter=key>value</parameter>...</function>`` block.
    We convert both into SwayBot's standard ``{"name": ..., "args": ...}``.

    A third variant is ``<ToolName key="value">`` prose tags: some models
    (notably glm-5.1-fp8 on retrieval tools) emit a known tool name directly
    as an XML-like tag instead of a native tool_call. When ``available_tools``
    is provided we recognize such tags ONLY for names that are actually
    registered tools, so ordinary HTML/markdown prose is not misparsed.
    """
    text = raw.strip()
    if not text.startswith("<"):
        return None

    # Variant 1: <tool_call> { "name": "...", "args": {...} } </tool_call>
    if "tool_call" in text:
        stripped = text
        for tag in ("<tool_call>", "</tool_call>", "<tool_call>", "</tool_call>"):
            stripped = stripped.replace(tag, "")
        stripped = stripped.strip()
        try:
            data = json.loads(stripped)
            if isinstance(data, dict) and "name" in data:
                return {"name": data["name"], "args": data.get("args", {})}
        except json.JSONDecodeError:
            pass

    # Variant 2: <function=name><parameter=key>value</parameter>...</function>
    func_match = re.search(r"<function=([^>\s]+)>", text)
    if func_match:
        return _parse_function_tag(text, func_match.group(1))

    # Variant 3: <ToolName key="value">  (prose-embedded tool tag). Only
    # recognized when the tag name is a registered tool, so arbitrary HTML is
    # left untouched. The model often emits several such tags per turn; we
    # take the first (the brain processes one action per turn).
    if available_tools:
        tool_set = set(available_tools)
        for m in re.finditer(r"<([A-Za-z_][\w-]*)((\s+[^<>]*?)?)\s*/?>", text):
            name = m.group(1)
            if name in tool_set:
                return {"name": name, "args": _parse_tag_attrs(m.group(2))}
    return None


def _parse_function_tag(text: str, name: str) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for param_match in re.finditer(
        r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", text, re.DOTALL
    ):
        key = param_match.group(1)
        value_text = param_match.group(2).strip()
        # XML parameter values are plain text. Only interpret values that are
        # clearly JSON literals (objects, arrays, booleans, null). Plain
        # numbers are kept as strings so ID fields such as product_id/zip
        # are not accidentally coerced to integers.
        lowered = value_text.lower()
        if value_text.startswith(("{", "[")) or lowered in (
            "true",
            "false",
            "null",
        ):
            try:
                value = json.loads(value_text)
            except json.JSONDecodeError:
                value = value_text
        else:
            value = value_text
        args[key] = value
    return {"name": name, "args": args}


def _parse_tag_attrs(attr_text: str) -> dict[str, Any]:
    """Parse ``key="value"`` / ``key='value'`` / ``key=value`` attributes from
    a prose tool tag like ``<KB_search query="gold card">``. Bare (quoteless)
    values run to the next whitespace; quoted values may contain spaces."""
    args: dict[str, Any] = {}
    pattern = r'''(\w[\w-]*)\s*=\s*("([^"]*)"|'([^']*)'|(\S+))'''
    for m in re.finditer(pattern, attr_text):
        value = m.group(3) if m.group(3) is not None else (
            m.group(4) if m.group(4) is not None else m.group(5)
        )
        args[m.group(1)] = value if value is not None else ""
    return args


def strip_prose_tool_tags(prose: str) -> str:
    """Remove prose-embedded tool tags (``<ToolName ...>`` / ``</ToolName>``)
    from a string, collapsing the gaps they leave so the surrounding text
    stays readable. Used when prose carrying such tags is about to be sent to
    the user as a final_answer (the tags are tool-call syntax, not an answer).

    Leaves any text the model wrote around the tags intact.
    """
    cleaned = PROSE_TOOL_TAG_RE.sub(" ", prose)
    # Collapse runs of whitespace left by the removal.
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    return cleaned.strip()


def parse_action(
    raw: str, perception: dict, available_tools: list[str] | None = None
) -> dict[str, Any]:
    raw = strip_markdown_fences(raw)

    # Some Qwen/DeepSeek-style models emit tool calls in XML tags even in
    # text mode. Try to normalize those into the standard JSON action shape.
    tool_action = extract_tool_call(raw, available_tools)
    if tool_action is not None:
        return tool_action

    try:
        action = json.loads(raw)
    except json.JSONDecodeError:
        return fallback_action(perception, raw)

    if not isinstance(action, dict) or "name" not in action:
        return fallback_action(perception, raw)

    return action


def parse_plan(raw: str) -> dict[str, Any]:
    raw = strip_markdown_fences(raw)

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        return {"name": "plan", "args": {"steps": []}}

    if isinstance(plan, list) and all(isinstance(s, str) for s in plan):
        return {"name": "plan", "args": {"steps": plan}}
    if isinstance(plan, dict):
        steps = plan.get("steps", plan.get("plan", []))
        if isinstance(steps, list) and all(isinstance(s, str) for s in steps):
            return {"name": "plan", "args": {"steps": steps}}

    return {"name": "plan", "args": {"steps": []}}


# Safety valve for pathological model output; shared by fallback_action
# and the brain's prose->final_answer recovery.
PROSE_CAP_CHARS = 4000


def fallback_action(perception: dict, raw: str) -> dict[str, Any]:
    if perception["step"] + 1 >= perception["max_steps"]:
        return {"name": "done", "args": {}}
    # The model emitted prose instead of a parseable tool call. This is common
    # for reasoning models whose visible-content budget is eaten by reasoning
    # tokens, so the final_answer tool call is never emitted. Rather than
    # truncate to 200 chars (which causes a "you got cut off" -> re-emit loop),
    # pass the full prose through as the user-facing message so the turn
    # completes. A generous cap only guards against pathological output.
    message = (raw or "").strip()
    if len(message) > PROSE_CAP_CHARS:
        message = message[:PROSE_CAP_CHARS]
    return {"name": "echo", "args": {"message": message}}


def is_explicit_tool_call(
    raw: str, available_tools: list[str] | None = None
) -> bool:
    """True when ``raw`` itself encodes a tool call (JSON action object or
    XML tool tag) — as opposed to prose that ``fallback_action`` wrapped
    into an echo. Used to keep the prose->final_answer recovery from
    hijacking genuine echo tool calls."""
    if extract_tool_call(raw, available_tools) is not None:
        return True
    try:
        data = json.loads(strip_markdown_fences(raw))
    except json.JSONDecodeError:
        return False
    return isinstance(data, dict) and "name" in data
