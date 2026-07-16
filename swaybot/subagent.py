"""Subagent manager for running parallel agent tasks."""

from concurrent.futures import ThreadPoolExecutor

from .agent import Agent


class SubagentManager:
    """Run multiple ``Agent`` instances in parallel."""

    def __init__(self, agent_factory):
        """``agent_factory`` should return a fresh ``Agent`` instance."""
        self.agent_factory = agent_factory

    def _run_one(self, spec: dict) -> dict:
        agent: Agent = self.agent_factory()
        env = agent.run(
            spec["task"],
            max_steps=spec.get("max_steps", 10),
            reflect=spec.get("reflect", True),
            plan=spec.get("plan", False),
            hypothesis=spec.get("hypothesis"),
        )
        return {
            "task": spec["task"],
            "done": env.done,
            "history": env.history,
        }

    def run_tasks(self, tasks: list[dict]) -> list[dict]:
        """Execute a list of task specs concurrently and return results in order."""
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._run_one, spec) for spec in tasks]
        return [f.result() for f in futures]
