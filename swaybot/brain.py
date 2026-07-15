from typing import Protocol, runtime_checkable


@runtime_checkable
class Brain(Protocol):
    """Decision-making core of the agent."""

    def think(self, perception: dict, available_tools: list[str]) -> dict:
        """Return the next action to execute."""
        ...


class EchoBrain:
    """Deterministic brain for bootstrapping. Requires no external API."""

    def __init__(self, echo_text: str = "thinking...") -> None:
        self.echo_text = echo_text

    def think(self, perception: dict, available_tools: list[str]) -> dict:
        if perception["step"] + 1 >= perception["max_steps"]:
            return {"name": "done", "args": {}}
        return {"name": "echo", "args": {"message": self.echo_text}}
