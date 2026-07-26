"""Standalone helpers for tau3-bench adapter.

These functions have no dependency on tau2-bench and can be tested
without installing it.
"""

from __future__ import annotations


def make_placeholder_fn(tool_name: str):
    """Create a no-op placeholder function for a tau2 tool name.

    The actual execution goes through ``env.execute_action()``, so
    the SwayBot registry only needs the function for schema metadata.
    """
    def _placeholder(**kwargs):
        return f"[{tool_name} executed via tau2 environment]"
    _placeholder.__name__ = tool_name
    _placeholder.__doc__ = f"Placeholder for tau2 tool: {tool_name}"
    return _placeholder


def extract_user_content(message: object) -> str:
    """Extract text content from a tau2 message object.

    Handles duck-typed objects that look like tau2 messages
    (AssistantMessage, ToolMessage, MultiToolMessage) without
    requiring the actual tau2 imports.
    """
    # MultiToolMessage-like: has tool_calls with function.arguments
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls and isinstance(tool_calls, list):
        parts = []
        for tc in tool_calls:
            fn = getattr(tc, "function", None)
            if fn and hasattr(fn, "arguments"):
                parts.append(str(fn.arguments))
        if parts:
            return " ".join(parts)

    # Simple content-based message
    content = getattr(message, "content", "")
    if content:
        return str(content)

    return ""
