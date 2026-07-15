from .brain import Brain, EchoBrain
from .environment import Environment
from .memory import Memory, MemoryStore
from .tools import ToolRegistry, build_default_registry


class Agent:
    """Minimal agent: perceive, think, act, observe, loop."""

    def __init__(
        self,
        brain: Brain | None = None,
        tools: ToolRegistry | None = None,
        memory: MemoryStore | None = None,
    ):
        self.brain = brain or EchoBrain()
        self.tools = tools or build_default_registry()
        self.memory = memory

    def run(self, task: str, max_steps: int = 10) -> Environment:
        env = Environment(task=task, max_steps=max_steps)
        while not env.done:
            perception = env.perceive()
            if self.memory is not None:
                perception["memory_context"] = self._memory_context(task)
            action = self.brain.think(perception, self.tools.names())
            result = self.tools.execute(action)
            env.observe(action, result)
            if self.memory is not None:
                self.memory.add(
                    Memory(
                        content=f"Step {env.step}: {action} -> {result}",
                        kind="experience",
                        source="agent.run",
                        evidence=str(result),
                        tags=[task],
                    )
                )
        return env

    def _memory_context(self, task: str) -> str:
        if self.memory is None:
            return ""
        relevant = self.memory.query(tag=task, limit=5)
        if not relevant:
            return ""
        return "\n".join(f"- {m.content}" for m in relevant)
