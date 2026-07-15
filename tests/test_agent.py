import pytest

from swaybot.agent import Agent
from swaybot.environment import Environment
from swaybot.tools import ToolRegistry, build_default_registry


def test_environment_perceive():
    env = Environment(task="test", max_steps=3)
    p = env.perceive()
    assert p["task"] == "test"
    assert p["step"] == 0
    assert not p["done"]


def test_environment_done_after_done_tool():
    env = Environment(task="test", max_steps=10)
    env.observe({"name": "done", "args": {}}, "finished")
    assert env.done


def test_environment_done_after_max_steps():
    env = Environment(task="test", max_steps=2)
    env.observe({"name": "echo", "args": {"message": "x"}}, "x")
    env.observe({"name": "echo", "args": {"message": "x"}}, "x")
    assert env.done


def test_tool_registry_execute():
    registry = build_default_registry()
    assert registry.execute({"name": "add", "args": {"a": 1, "b": 2}}) == 3
    assert registry.execute({"name": "echo", "args": {"message": "hi"}}) == "hi"
    assert registry.execute({"name": "done", "args": {}}) == "finished"


def test_tool_registry_unknown_tool():
    registry = ToolRegistry()
    with pytest.raises(ValueError):
        registry.execute({"name": "missing", "args": {}})


def test_agent_run():
    agent = Agent()
    env = agent.run("hello", max_steps=3)
    assert env.done
    assert len(env.history) == 3
    assert env.history[-1]["action"]["name"] == "done"


def test_agent_run_ends_early_with_done_brain():
    class DoneBrain:
        def think(self, perception, available_tools):
            return {"name": "done", "args": {}}

    agent = Agent(brain=DoneBrain())
    env = agent.run("hello", max_steps=10)
    assert env.done
    assert len(env.history) == 1
