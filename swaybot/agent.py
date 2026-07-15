from .brain import Brain, EchoBrain
from .environment import Environment
from .memory import (
    ActionStep,
    MemoryStore,
    ObservationStep,
    PlanningStep,
    TaskStep,
)
from .reflection import Reflector, reflection_to_memory
from .tools import ToolRegistry, build_default_registry


class Agent:
    """Minimal agent: perceive, think, act, observe, loop, reflect."""

    def __init__(
        self,
        brain: Brain | None = None,
        tools: ToolRegistry | None = None,
        memory: MemoryStore | None = None,
        reflector: Reflector | None = None,
    ):
        self.brain = brain or EchoBrain()
        self.tools = tools or build_default_registry()
        self.memory = memory
        self.reflector = reflector

    def run(
        self,
        task: str,
        max_steps: int = 10,
        reflect: bool = True,
        plan: bool = False,
        hypothesis: str | None = None,
    ) -> Environment:
        env = Environment(task=task, max_steps=max_steps)
        if self.memory is not None:
            self.memory.add(
                TaskStep(task=task, max_steps=max_steps, tags=[task])
            )
        if plan and self.memory is not None:
            self._create_plan(task, max_steps)
        while not env.done:
            perception = env.perceive()
            if self.memory is not None:
                perception["memory_context"] = self._memory_context(task)
                perception["behavior_guidance"] = self._behavior_guidance(task)
                perception["messages"] = self._build_messages(task)
            perception["tool_descriptions"] = self.tools.schemas()
            action = self.brain.think(perception, self.tools.names())
            result = self.tools.execute(action)
            env.observe(action, result)
            if self.memory is not None:
                self.memory.add(
                    ActionStep(
                        step=env.step,
                        action=action,
                        tags=[task],
                    )
                )
                self.memory.add(
                    ObservationStep(
                        step=env.step,
                        observation=str(result),
                        tags=[task],
                    )
                )

        if reflect and self.memory is not None and self.reflector is not None:
            for reflection in self.reflector.reflect_on_run(
                task, env.history, hypothesis=hypothesis
            ):
                self.memory.add(reflection_to_memory(reflection))
            self.memory.prune(scope="short_term", tag=task)

        return env

    def _create_plan(self, task: str, max_steps: int) -> None:
        """Ask the brain for a plan and store it as a PlanningStep."""
        assert self.memory is not None
        perception = {
            "task": task,
            "max_steps": max_steps,
            "step": 0,
            "planning": True,
            "tool_descriptions": self.tools.schemas(),
        }
        action = self.brain.think(perception, self.tools.names())
        if action.get("name") == "plan":
            steps = action.get("args", {}).get("steps", [])
            if steps:
                self.memory.add(PlanningStep(plan=steps, tags=[task]))

    def _memory_context(self, task: str) -> str:
        if self.memory is None:
            return ""
        relevant = self.memory.query_relevant(task, scope="long_term", limit=5)
        if not relevant:
            return ""
        return "\n".join(
            f"- {getattr(m, 'content', str(m))}" for m in relevant
        )

    def _behavior_guidance(self, task: str) -> str:
        if self.memory is None:
            return ""
        theories = self.memory.query_relevant(task, scope="long_term", limit=5)
        if not theories:
            return ""
        lines: list[str] = []
        for theory in theories:
            if getattr(theory, "credibility", 0.0) < 0.5:
                continue
            content = getattr(theory, "content", str(theory))
            if content:
                lines.append(f"- {content}")
        return "\n".join(lines)

    def _build_messages(self, task: str) -> list[dict]:
        if self.memory is None:
            return []
        messages: list[dict] = []
        context = self._memory_context(task)
        if context:
            messages.append(
                {"role": "user", "content": f"Relevant memories:\n{context}"}
            )
        for step in self.memory.memories:
            if task not in step.tags or step.scope != "short_term":
                continue
            messages.extend(step.to_messages())
        return messages
