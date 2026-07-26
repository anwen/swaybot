"""SwayBot adapter for tau2-bench's HalfDuplexAgent interface.

This module provides ``SwayBotHalfDuplexAgent`` which wraps SwayBot's
``Brain`` + ``ToolRegistry`` into the interface tau2-bench expects:

- ``get_init_state()`` → empty ``AgentState``
- ``generate_next_message(message, state)`` → one turn of agent logic

Key design decisions
-------------------
- **Tool execution is delegated to tau2's environment** (``env.execute_action``)
  so that the domain's DB/state tracking works correctly.
- The SwayBot brain *decides* which tool to call; the tau2 environment
  *executes* it and tracks side effects.
- Domain tools are mapped into a fresh ``ToolRegistry`` per task so that
  the brain can reason about available tools via SwayBot's normal
  ``perception["tool_descriptions"]`` mechanism.
"""

from __future__ import annotations

from typing import Any

from swaybot.brain import Brain, EchoBrain
from swaybot.tools import Tool, ToolRegistry

from ._helpers import extract_user_content, make_placeholder_fn

try:
    from tau2.agent.base.agent_base import AgentState, HalfDuplexAgent
    from tau2.data_model.message import (
        AssistantMessage,
        MultiToolMessage,
        ToolMessage,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "tau2-bench is required for this adapter. "
        "Install it with: pip install tau2-bench  (requires Python >= 3.12)"
    ) from exc


class SwayBotHalfDuplexAgent(HalfDuplexAgent):
    """Adapt a SwayBot Brain to tau2's HalfDuplexAgent interface.

    Parameters
    ----------
    brain :
        A SwayBot ``Brain`` instance (e.g. ``LLMBrain``).  Used to decide
        which tool to call each turn.
    instructions :
        Optional domain policy string injected into the system prompt.
    tools :
        Optional list of tau2 ``Tool`` objects provided by the environment.
        If *None*, tools will be set later via ``set_tools()``.
    """

    def __init__(
        self,
        brain: Brain | None = None,
        instructions: str | None = None,
        tools: list[Any] | None = None,
    ) -> None:
        self._brain = brain or EchoBrain()
        self._instructions = instructions or ""
        self._tau2_tools: list[Any] = tools or []
        self._registry = ToolRegistry()
        self._history: list[dict] = []
        self._step = 0
        self._max_steps = 100
        self._task = ""
        self._env: Any = None  # set via set_environment()

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def set_tools(self, tools: list[Any]) -> None:
        """Accept tau2 environment tools and map into SwayBot ToolRegistry."""
        self._tau2_tools = tools
        self._registry = ToolRegistry()
        for t in tools:
            swaybot_tool = Tool(
                name=t.name,
                description=t.description or "",
                inputs=t.schema if hasattr(t, "schema") and isinstance(t.schema, dict) else {},
                output_type="string",
                fn=make_placeholder_fn(t.name),
            )
            self._registry.register(t.name, swaybot_tool)

    def set_environment(self, env: Any) -> None:
        """Store a reference to the tau2 environment for tool execution."""
        self._env = env

    def set_task(self, task: str, max_steps: int = 100) -> None:
        """Set the current task description and step budget."""
        self._task = task
        self._max_steps = max_steps
        self._step = 0
        self._history = []

    # ------------------------------------------------------------------
    # HalfDuplexAgent interface
    # ------------------------------------------------------------------

    def get_init_state(self) -> AgentState:
        """Return an empty initial state."""
        return AgentState(messages=[])

    def generate_next_message(
        self,
        message: AssistantMessage | ToolMessage | MultiToolMessage,
        state: AgentState,
    ) -> tuple[AssistantMessage, AgentState]:
        """One turn: decide which tool to call, delegate execution to tau2 env.

        Flow:
        1. Convert incoming tau2 message → SwayBot perception dict
        2. Call ``brain.think()`` to get an action dict
        3. Execute the action via ``env.execute_action()``
        4. Convert result → tau2 AssistantMessage
        """
        self._step += 1

        # Build perception for SwayBot brain
        user_content = extract_user_content(message)
        if user_content and not self._task:
            self._task = user_content

        perception = {
            "task": self._task,
            "step": self._step,
            "max_steps": self._max_steps,
            "history": self._history,
            "tool_descriptions": self._registry.schemas(),
            "available_tools": self._registry.names(),
        }
        if self._instructions:
            perception["behavior_guidance"] = self._instructions

        # Ask the brain what to do
        action = self._brain.think(perception, self._registry.names())

        # Execute via tau2 environment (preserves state tracking)
        result_str = ""
        if self._env is not None and action.get("name"):
            try:
                result = self._env.execute_action(action)
                result_str = str(result)
            except Exception as exc:
                result_str = f"Error: {exc}"
        elif action.get("name"):
            # Fallback: try SwayBot's own registry (won't track tau2 state)
            try:
                result = self._registry.execute(action)
                result_str = str(result)
            except Exception as exc:
                result_str = f"Error: {exc}"

        # Record in local history
        self._history.append({
            "step": self._step,
            "action": action,
            "result": result_str,
        })

        # Build tau2 response
        tool_name = action.get("name", "")
        tool_args = action.get("args", {})

        if tool_name and tool_name not in ("done", "final_answer"):
            assistant_msg = AssistantMessage(
                role="assistant",
                content=action.get("text") or result_str or "",
            )
        else:
            text = tool_args.get("answer", result_str) if tool_name == "final_answer" else result_str
            assistant_msg = AssistantMessage(
                role="assistant",
                content=text,
            )

        # Update state
        new_state = AgentState(
            messages=list(state.messages) + [assistant_msg],
        )
        return assistant_msg, new_state
