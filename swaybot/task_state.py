from dataclasses import dataclass, field


@dataclass
class Environment:
    """The agent's view of the world for a single task."""

    task: str
    max_steps: int = 10
    step: int = 0
    done: bool = False
    # "tool" — the agent signalled done/final_answer; "max_steps" — the
    # step budget ran out while it was still asking for tools.
    stop_reason: str | None = None
    history: list[dict] = field(default_factory=list)

    def perceive(self) -> dict:
        """Return a snapshot of the current state."""
        return {
            "task": self.task,
            "step": self.step,
            "max_steps": self.max_steps,
            "done": self.done,
            "history": self.history,
        }

    def observe(self, action: dict, result: object) -> None:
        """Record an action and its result, then update termination state."""
        self.step += 1
        self.history.append({"step": self.step, "action": action, "result": result})
        if action.get("name") in {"done", "final_answer"}:
            self.done = True
            self.stop_reason = "tool"
        elif self.step >= self.max_steps:
            self.done = True
            self.stop_reason = "max_steps"
