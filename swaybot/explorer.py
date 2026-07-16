import json
from dataclasses import dataclass

from .agent import Agent
from .brain import EchoBrain
from .environment import Environment
from .reflection import Reflector


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


@dataclass
class ExplorationTask:
    """A self-generated task for the agent to run."""

    task: str
    hypothesis: str = ""

    def to_dict(self) -> dict:
        return {"task": self.task, "hypothesis": self.hypothesis}


class Explorer:
    """Generate and run small exploratory tasks when no user task is given."""

    def __init__(self, agent: Agent, max_steps: int = 5):
        self.agent = agent
        self.max_steps = max_steps
        self._default_index = 0

    def generate_task(self) -> ExplorationTask:
        """Ask the brain for a curiosity-driven task, or pick a default."""
        memory_context = self._memory_summary()
        candidate_hypotheses = self._candidate_hypotheses_from_memory()
        perception = {
            "task": "Generate an exploratory task",
            "exploring": True,
            "memory_context": memory_context,
            "candidate_hypotheses": candidate_hypotheses,
            "tool_descriptions": self.agent.tools.schemas(),
            "available_tools": self.agent.tools.names(),
        }

        # EchoBrain ignores the prompt, so short-circuit if we already have
        # concrete questions or contradictions to investigate.
        if candidate_hypotheses and isinstance(self.agent.brain, EchoBrain):
            return ExplorationTask(
                task=str(candidate_hypotheses[0]),
                hypothesis="Investigate this open question or contradiction.",
            )

        action = self.agent.brain.think(perception, self.agent.tools.names())
        args = action.get("args", {}) if isinstance(action, dict) else {}
        task_text = args.get("task") if isinstance(args, dict) else None
        hypothesis = args.get("hypothesis", "") if isinstance(args, dict) else ""
        if task_text:
            return ExplorationTask(task=str(task_text), hypothesis=str(hypothesis))

        if candidate_hypotheses:
            return ExplorationTask(
                task=str(candidate_hypotheses[0]),
                hypothesis="Investigate this open question or contradiction.",
            )
        return self._default_task()

    def run(self, max_steps: int | None = None) -> tuple[ExplorationTask, Environment]:
        """Generate a task and run the agent on it."""
        if self.agent.memory is not None and self.agent.reflector is None:
            self.agent.reflector = Reflector(self.agent.memory)
        task = self.generate_task()
        env = self.agent.run(
            task.task,
            max_steps=max_steps or self.max_steps,
            reflect=True,
            plan=False,
            hypothesis=task.hypothesis,
        )
        return task, env

    def _default_task(self) -> ExplorationTask:
        data = DEFAULT_EXPLORATION_TASKS[self._default_index]
        self._default_index = (self._default_index + 1) % len(DEFAULT_EXPLORATION_TASKS)
        return ExplorationTask(**data)

    def _memory_summary(self) -> str:
        if self.agent.memory is None:
            return ""
        long_term = self.agent.memory.query(scope="long_term", limit=10)
        if not long_term:
            return ""
        lines = []
        for mem in long_term:
            content = getattr(mem, "content", "")
            if content:
                lines.append(f"- {content}")
        return "\n".join(lines)

    def _candidate_hypotheses_from_memory(self) -> list[str]:
        """Pull unresolved questions and contradictions from long-term memory."""
        if self.agent.memory is None:
            return []
        seen: set[str] = set()
        candidates: list[str] = []
        for kind in ("question", "contradiction"):
            for mem in self.agent.memory.query(kind=kind, scope="long_term", limit=20):
                content = getattr(mem, "content", "")
                if content and content not in seen:
                    seen.add(content)
                    candidates.append(content)
        return candidates


def parse_exploration_response(raw: str) -> dict:
    """Parse a JSON response expected to contain task and hypothesis."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    if not isinstance(data, dict):
        return {}

    return {
        "task": data.get("task", data.get("explore", "")),
        "hypothesis": data.get("hypothesis", ""),
    }
