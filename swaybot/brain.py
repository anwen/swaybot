from typing import Any, Callable, Protocol, runtime_checkable


StreamCallback = Callable[[str], None]
ReasoningCallback = Callable[[str], None]


DEFAULT_EXPLORATION_TASKS = [
    {
        "task": "Test whether the echo tool preserves punctuation.",
        "hypothesis": "echo returns the exact message including punctuation.",
    },
    {
        "task": "Verify that add works with negative numbers.",
        "hypothesis": "add(a=-1, b=1) returns 0.",
    },
    {
        "task": "Check what happens when done is called multiple times.",
        "hypothesis": "done always returns 'finished'.",
    },
]


@runtime_checkable
class Brain(Protocol):
    """Decision-making core of the agent."""

    def think(
        self,
        perception: dict,
        available_tools: list[str],
        metadata: dict | None = None,
        stream_callback: StreamCallback | None = None,
        reasoning_callback: ReasoningCallback | None = None,
    ) -> dict[str, Any]:
        """Return the next action to execute."""
        ...


class EchoBrain:
    """Deterministic brain for bootstrapping. Requires no external API."""

    def __init__(self, echo_text: str = "thinking...") -> None:
        self.echo_text = echo_text
        self._explore_index = 0

    def think(
        self,
        perception: dict,
        available_tools: list[str],
        metadata: dict | None = None,
        stream_callback: StreamCallback | None = None,
        reasoning_callback: ReasoningCallback | None = None,
    ) -> dict[str, Any]:
        if perception.get("planning"):
            return {
                "name": "plan",
                "args": {
                    "steps": [
                        "Understand the task",
                        "Select the best tool",
                        "Execute and observe",
                        "Finish",
                    ]
                },
            }
        if perception.get("exploring"):
            task = DEFAULT_EXPLORATION_TASKS[self._explore_index]
            self._explore_index = (self._explore_index + 1) % len(
                DEFAULT_EXPLORATION_TASKS
            )
            return {"name": "explore", "args": task}
        if perception["step"] + 1 >= perception["max_steps"]:
            return {"name": "done", "args": {}}
        return {"name": "echo", "args": {"message": self.echo_text}}
