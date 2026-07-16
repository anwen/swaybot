import pytest

from swaybot.agent import Agent
from swaybot.environment import Environment
from swaybot.subagent import SubagentManager


def test_subagent_manager_runs_tasks_concurrently(monkeypatch):
    call_count = {"value": 0}

    def factory():
        return Agent()

    def fast_run(self, task, max_steps=10, **kwargs):
        call_count["value"] += 1
        env = Environment(task=task, max_steps=max_steps)
        env.done = True
        return env

    monkeypatch.setattr(Agent, "run", fast_run)

    manager = SubagentManager(factory)
    results = manager.run_tasks(
        [{"task": "one"}, {"task": "two"}, {"task": "three"}]
    )
    assert len(results) == 3
    assert all(r["done"] for r in results)
    assert call_count["value"] == 3
