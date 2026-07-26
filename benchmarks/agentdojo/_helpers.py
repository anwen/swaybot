"""Standalone helpers for AgentDojo adapter.

These functions have no dependency on agentdojo and can be tested
without installing it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from swaybot.tools import Tool, ToolRegistry


def _parameters_to_schema(parameters: Any) -> dict[str, Any]:
    """Convert an AgentDojo function parameter spec into a JSON schema dict.

    AgentDojo stores parameters as a Pydantic model class.  We need the
    JSON schema so SwayBot can pass correct tool signatures to the LLM.
    """
    if parameters is None:
        return {}
    if isinstance(parameters, dict):
        return parameters
    # Pydantic v2
    to_json_schema = getattr(parameters, "model_json_schema", None)
    if callable(to_json_schema):
        return cast(dict[str, Any], to_json_schema())
    # Pydantic v1 fallback
    to_schema = getattr(parameters, "schema", None)
    if callable(to_schema):
        return cast(dict[str, Any], to_schema())
    return {}


def build_registry_from_functions(functions: dict[str, Any]) -> ToolRegistry:
    """Map AgentDojo's FunctionsRuntime.functions dict into a SwayBot ToolRegistry.

    The dict maps function names → function-like objects.  Each object
    should have ``name``, ``description``, and optionally ``parameters``
    (a JSON schema dict).

    The registry is used only for schema/metadata — actual execution
    goes through AgentDojo's runtime to preserve state tracking.
    """
    registry = ToolRegistry()
    for func_name, func in functions.items():
        name = getattr(func, "name", func_name)
        description = getattr(func, "description", "") or ""
        parameters = _parameters_to_schema(getattr(func, "parameters", None))

        swaybot_tool = Tool(
            name=name,
            description=description,
            inputs=parameters if isinstance(parameters, dict) else {},
            output_type="string",
            fn=make_placeholder_fn(name),
        )
        registry.register(name, swaybot_tool)
    return registry


def make_placeholder_fn(tool_name: str):
    """No-op placeholder. Actual execution via AgentDojo runtime."""
    def _placeholder(**_kwargs):
        return f"[{tool_name} executed via AgentDojo runtime]"
    _placeholder.__name__ = tool_name
    _placeholder.__doc__ = f"Placeholder for AgentDojo tool: {tool_name}"
    return _placeholder


def _content_to_str(content: Any) -> str:
    """Flatten a message content value (string or list of blocks) to text."""
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", str(block)))
            else:
                parts.append(getattr(block, "text", str(block)))
        return " ".join(parts)
    return str(content)


def messages_to_history(messages: Sequence[dict]) -> list[dict]:
    """Convert chat message dicts to SwayBot history format.

    Handles both plain string content and list-of-blocks content.
    """
    history: list[dict] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = _content_to_str(msg.get("content"))
        history.append({"role": role, "content": content})
    return history


def messages_to_run_history(messages: Sequence[dict]) -> list[dict]:
    """Convert AgentDojo chat messages into SwayBot run-history entries.

    SwayBot's user prompt expects each entry to have ``step``, ``action``,
    and ``result``.  AgentDojo messages alternate assistant tool-call
    messages with tool result messages, so we pair them up here.
    """
    run_history: list[dict] = []
    pending_action: dict | None = None

    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                tc = tool_calls[0]
                if isinstance(tc, dict):
                    fn = tc.get("function", {}) or {}
                    name = fn.get("name") if isinstance(fn, dict) else tc.get("function")
                    args = tc.get("args", {})
                else:
                    name = getattr(tc, "function", None)
                    args = getattr(tc, "args", None) or {}
                pending_action = {"name": name, "args": dict(args)}
            else:
                content = _content_to_str(msg.get("content"))
                pending_action = {"name": "final_answer", "args": {"answer": content}}
        elif role == "tool" and pending_action is not None:
            result = _content_to_str(msg.get("content"))
            run_history.append(
                {"step": len(run_history) + 1, "action": pending_action, "result": result}
            )
            pending_action = None

    # Record a trailing assistant message that did not produce a tool result
    # (e.g. a final_answer/done text response).
    if pending_action is not None:
        run_history.append(
            {"step": len(run_history) + 1, "action": pending_action, "result": ""}
        )

    return run_history
