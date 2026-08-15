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
        "tau2-bench is required. Install it with: pip install 'tau2[knowledge] @ git+https://github.com/sierra-research/tau2-bench'"
    ) from exc


def create_swaybot_agent(
    tools: list[Any],
    domain_policy: str,
    *,
    llm: str | None = None,
    llm_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> SwayBotHalfDuplexAgent:
    """Factory function that creates a SwayBotHalfDuplexAgent.

    tau2's ``build_agent`` calls registered factories with
    ``tools``, ``domain_policy``, ``llm``, ``llm_args``, and ``task``.
    If *llm* is provided, an ``LLMBrain`` is constructed; otherwise an
    ``EchoBrain`` is used.

    LLMBrain picks up the API key / base URL from the environment
    (``SWAYBOT_API_KEY``, ``SWAYBOT_API_BASE``) when they are not passed
    explicitly.
    """
    return SwayBotHalfDuplexAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=llm,
        llm_args=llm_args,
    )


# Register with tau2's global registry
registry.register_agent_factory(create_swaybot_agent, "swaybot")
