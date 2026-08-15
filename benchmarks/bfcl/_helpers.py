"""Standalone helpers for BFCL server.

These functions have no dependency on bfcl-eval and can be tested
without installing it.
"""

from __future__ import annotations

import json

from swaybot.tools import Tool, ToolRegistry


def build_registry(tools: list[dict]) -> ToolRegistry:
    """Map OpenAI-format tool definitions to a SwayBot ToolRegistry."""
    registry = ToolRegistry()
    for tool_def in tools:
        func = tool_def.get("function", {})
        name = func.get("name", "unknown")
        description = func.get("description", "")
        parameters = func.get("parameters", {})

        swaybot_tool = Tool(
            name=name,
            description=description,
            inputs=parameters if isinstance(parameters, dict) else {},
            output_type="string",
            fn=make_noop_fn(name),
        )
        registry.register(name, swaybot_tool)
    return registry


def normalize_tool_name(name: str) -> str:
    """Return a canonical form of a tool name for fuzzy matching.

    Some model backends replace dots/ dashes with underscores in tool names.
    """
    return name.replace(".", "_").replace("-", "_")


def resolve_tool_name(registry: ToolRegistry, returned_name: str) -> str:
    """Map a model-returned tool name back to the original registered name.

    BFCL uses names like ``math.factorial``; the model may emit
    ``math_factorial``.  If *returned_name* is not an exact match, try the
    normalized form against the registry.
    """
    if returned_name in registry.names():
        return returned_name
    normalized = normalize_tool_name(returned_name)
    for original in registry.names():
        if normalize_tool_name(original) == normalized:
            return original
    return returned_name


def make_noop_fn(name: str):
    """No-op function — BFCL executes tools itself."""
    def _noop(**kwargs):
        return ""
    _noop.__name__ = name
    _noop.__doc__ = f"No-op for BFCL tool: {name}"
    return _noop


def extract_task(messages: list[dict]) -> str:
    """Extract the last user message as the task description."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                return " ".join(
                    part.get("text", str(part))
                    for part in content
                    if isinstance(part, dict)
                )
            return str(content)
    return ""


def messages_to_history(messages: list[dict]) -> list[dict]:
    """Convert OpenAI messages to SwayBot history format."""
    history: list[dict] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {})
                history.append({
                    "role": "assistant",
                    "action": {
                        "name": fn.get("name", ""),
                        "args": json.loads(fn.get("arguments", "{}")),
                    },
                })
        elif msg.get("role") == "tool":
            history.append({
                "role": "tool",
                "tool_call_id": msg.get("tool_call_id", ""),
                "content": str(content),
            })
        else:
            history.append({"role": role, "content": str(content)})
    return history
