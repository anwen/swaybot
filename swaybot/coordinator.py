"""Sustained-goal / long-task coordinator."""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .agent import Agent


@dataclass
class Subtask:
    """A single step toward a larger goal."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    description: str = ""
    status: str = "pending"  # pending, running, done, failed
    result: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class Goal:
    """A sustained goal broken into subtasks."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    description: str = ""
    context: str = ""
    subtasks: list[Subtask] = field(default_factory=list)
    status: str = "pending"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class GoalCoordinator:
    """Break a long goal into subtasks and execute them sequentially.

    ``agent_factory`` should return an ``Agent`` instance for each subtask.
    """

    def __init__(
        self,
        agent_factory: Callable[[], Agent],
        state_path: Path | str | None = None,
        max_subtask_steps: int = 10,
    ) -> None:
        self.agent_factory = agent_factory
        if state_path is None:
            state_path = Path.home() / ".swaybot" / "goals.jsonl"
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_subtask_steps = max_subtask_steps

    def run(self, description: str, context: str = "") -> Goal:
        """Plan, execute, and record a goal."""
        goal = Goal(description=description, context=context)
        goal.subtasks = self._plan(goal)
        if not goal.subtasks:
            goal.subtasks.append(Subtask(description=description))

        goal.status = "running"
        for subtask in goal.subtasks:
            subtask.status = "running"
            agent = self.agent_factory()
            try:
                env = agent.run(
                    subtask.description,
                    max_steps=self.max_subtask_steps,
                    reflect=False,
                )
                final = env.history[-1]["result"] if env.history else ""
                subtask.result = str(final)
                if any("error" in h for h in env.history) or str(final).startswith("Error:"):
                    subtask.status = "failed"
                else:
                    subtask.status = "done"
            except Exception as exc:
                subtask.result = f"Error: {exc}"
                subtask.status = "failed"

        failed = any(s.status == "failed" for s in goal.subtasks)
        goal.status = "failed" if failed else "done"
        self._save(goal)
        return goal

    def _plan(self, goal: Goal) -> list[Subtask]:
        """Ask the agent's brain for a list of subtasks."""
        agent = self.agent_factory()
        perception = {
            "task": goal.description,
            "context": goal.context,
            "step": 0,
            "max_steps": 3,
            "planning": True,
            "tool_descriptions": agent.tools.schemas(),
            "available_tools": agent.tools.names(),
        }
        action = agent.brain.think(perception, agent.tools.names())
        if action.get("name") == "plan":
            steps = action.get("args", {}).get("steps", [])
            if steps:
                return [Subtask(description=str(s)) for s in steps]
        return []

    def _save(self, goal: Goal) -> None:
        data = asdict(goal)
        with self.state_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(data, ensure_ascii=False) + "\n")

    def load_history(self) -> list[dict[str, Any]]:
        if not self.state_path.exists():
            return []
        entries = []
        with self.state_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries
