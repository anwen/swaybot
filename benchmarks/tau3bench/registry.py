"""Factory registration for SwayBot agent in tau2-bench.

Registers a ``SwayBotHalfDuplexAgent`` factory with tau2's global
agent registry so it can be selected via ``--agent swaybot`` on the CLI.
"""

from __future__ import annotations

from typing import Any

from swaybot.brain import Brain, EchoBrain
from swaybot.llm_brain import LLMBrain

from .adapter import SwayBotHalfDuplexAgent

try:
    from tau2.registry import registry
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "tau2-bench is required. Install it with: pip install tau2-bench"
    ) from exc


def create_swaybot_agent(
    *,
    instructions: str | None = None,
    tools: list[Any] | None = None,
    llm_agent: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> SwayBotHalfDuplexAgent:
    """Factory function that creates a SwayBotHalfDuplexAgent.

    If *llm_agent* is provided (e.g. ``"gpt-4o"``), an ``LLMBrain``
    is constructed; otherwise an ``EchoBrain`` is used.
    """
    brain: Brain
    if llm_agent:
        brain = LLMBrain(
            api_key=api_key,
            base_url=base_url,
            model=llm_agent,
        )
    else:
        brain = EchoBrain()

    return SwayBotHalfDuplexAgent(
        brain=brain,
        instructions=instructions,
        tools=tools,
    )


# Register with tau2's global registry
registry.register_agent_factory(create_swaybot_agent, "swaybot")
