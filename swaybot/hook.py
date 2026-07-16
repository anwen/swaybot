"""Lifecycle hooks for observing and extending agent runs."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AgentHook(Protocol):
    """Observe key moments of an agent run without changing core logic."""

    def before_iteration(
        self, task: str, step: int, perception: dict
    ) -> None:
        ...

    def after_iteration(
        self,
        task: str,
        step: int,
        action: dict,
        result: Any,
        metadata: dict | None = None,
    ) -> None:
        ...

    def after_run(
        self, task: str, env: Any, reflections: list[str]
    ) -> None:
        ...


class CompositeHook:
    """Run multiple hooks as one."""

    def __init__(self, hooks: list[AgentHook] | None = None) -> None:
        self.hooks = hooks or []

    def before_iteration(
        self, task: str, step: int, perception: dict
    ) -> None:
        for hook in self.hooks:
            hook.before_iteration(task, step, perception)

    def after_iteration(
        self,
        task: str,
        step: int,
        action: dict,
        result: Any,
        metadata: dict | None = None,
    ) -> None:
        for hook in self.hooks:
            hook.after_iteration(task, step, action, result, metadata)

    def after_run(
        self, task: str, env: Any, reflections: list[str]
    ) -> None:
        for hook in self.hooks:
            hook.after_run(task, env, reflections)
