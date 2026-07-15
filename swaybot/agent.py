from .brain import Brain, EchoBrain
from .environment import Environment
from .tools import ToolRegistry, build_default_registry


class Agent:
    """Minimal agent: perceive, think, act, observe, loop."""

    def __init__(self, brain: Brain | None = None, tools: ToolRegistry | None = None):
        self.brain = brain or EchoBrain()
        self.tools = tools or build_default_registry()

    def run(self, task: str, max_steps: int = 10) -> Environment:
        env = Environment(task=task, max_steps=max_steps)
        while not env.done:
            perception = env.perceive()
            action = self.brain.think(perception, self.tools.names())
            result = self.tools.execute(action)
            env.observe(action, result)
        return env
