"""SwayBot adapter for AgentDojo's BasePipelineElement interface.

This module provides ``SwayBotPipelineElement`` which wraps SwayBot's
``Brain`` into the pipeline interface AgentDojo expects:

- ``query(query, runtime, env, messages, extra_args)`` → 5-tuple

Key design decisions
--------------------
- **Tool execution is delegated to AgentDojo's FunctionsRuntime** so
  that the ``TaskEnvironment`` state (banking, workspace, etc.) is
  tracked correctly for utility/security evaluation.
- The SwayBot brain *decides* which tool to call; AgentDojo's runtime
  *executes* it and tracks side effects.
- A ``ToolsExecutor`` is placed after this element in the pipeline to
  handle actual tool invocations.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from swaybot.brain import Brain, EchoBrain

from ._helpers import build_registry_from_functions, messages_to_run_history

try:
    from agentdojo.agent_pipeline import BasePipelineElement
    from agentdojo.functions_runtime import FunctionsRuntime
    from agentdojo.types import (
        ChatMessage,
        ChatAssistantMessage,
        FunctionCall,
        text_content_block_from_string,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "agentdojo is required for this adapter. "
        "Install it with: pip install agentdojo"
    ) from exc


class SwayBotPipelineElement(BasePipelineElement):
    """Adapt a SwayBot Brain to AgentDojo's pipeline interface.

    Parameters
    ----------
    brain :
        A SwayBot ``Brain`` instance.  Used to decide which tool to
        call each turn.
    name :
        Optional name for this pipeline element (used in logging).
    """

    def __init__(
        self,
        brain: Brain | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.name = name or "swaybot"
        self._brain = brain or EchoBrain()
        self._step = 0
        self._max_steps = 50

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Any = None,
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Any, Sequence[ChatMessage], dict]:
        """Process one turn: decide which tool to call.

        The SwayBot brain inspects available tools from the runtime,
        decides an action, and returns a ChatAssistantMessage with
        either text or tool calls for downstream ``ToolsExecutor``.
        """
        self._step += 1

        # Build SwayBot tool registry from AgentDojo's runtime
        functions = runtime.functions
        registry = build_registry_from_functions(functions)

        # Build perception from conversation history
        task = query
        perception = {
            "task": task,
            "step": self._step,
            "max_steps": self._max_steps,
            "history": messages_to_run_history(list(messages)),
            "tool_descriptions": registry.schemas(),
            "available_tools": registry.names(),
        }

        # Ask the brain
        action = self._brain.think(perception, registry.names())

        # Build AgentDojo response message
        tool_name = action.get("name", "")
        tool_args = action.get("args", {})

        if tool_name and tool_name not in ("done", "final_answer", "echo"):
            # Convert tool_args dict to the mapping FunctionCall expects
            fc_args = {str(k): v for k, v in tool_args.items()} if isinstance(tool_args, dict) else {}
            assistant_msg = ChatAssistantMessage(
                role="assistant",
                content=[text_content_block_from_string("")],
                tool_calls=[FunctionCall(
                    function=tool_name,
                    args=fc_args,
                    id=f"call_{self._step}",
                )],
            )
        else:
            text = ""
            if tool_name == "final_answer":
                text = tool_args.get("answer", "")
            elif tool_name == "echo":
                text = tool_args.get("message", "")
            else:
                text = action.get("text", "")

            assistant_msg = ChatAssistantMessage(
                role="assistant",
                content=[text_content_block_from_string(text or "I'm done.")],
                tool_calls=None,
            )

        new_messages = list(messages) + [assistant_msg]
        return query, runtime, env, new_messages, extra_args
