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


def test_format_action():
    from swaybot.tools import format_action

    assert format_action({"name": "add", "args": {"a": 2, "b": 2}}) == "add(a=2, b=2)"
    assert format_action({"name": "echo", "args": {"message": "hi"}}) == "echo(message='hi')"
    assert format_action({"name": "done", "args": {}}) == "done()"
    assert format_action({"name": "unknown"}) == "unknown()"


def test_tool_registry_descriptions():
    registry = build_default_registry()
    descriptions = registry.descriptions()
    assert set(descriptions) == {"add(a, b)", "echo(message='')", "done()"}


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


def test_agent_build_messages_includes_memory_context_and_short_term_steps():
    from swaybot.memory import Memory, MemoryStore

    store = MemoryStore()
    store.add(Memory(content="demo context long", scope="long_term", tags=["demo"]))
    store.add(Memory(content="demo short note", scope="short_term", tags=["demo"]))
    agent = Agent(memory=store)
    messages = agent._build_messages("demo")
    assert any("Relevant memories" in msg["content"] and "demo context long" in msg["content"] for msg in messages)
    assert any(msg["content"] == "demo short note" for msg in messages)


def test_agent_run_with_plan_creates_planning_step():
    from swaybot.memory import MemoryStore, PlanningStep

    store = MemoryStore()
    agent = Agent(memory=store)
    env = agent.run("demo", max_steps=2, reflect=False, plan=True)
    assert env.done
    assert any(isinstance(m, PlanningStep) for m in store.memories)
    plan_step = next(m for m in store.memories if isinstance(m, PlanningStep))
    assert len(plan_step.plan) >= 1
    assert "demo" in plan_step.tags


def test_agent_behavior_guidance_pulls_high_credibility_theories():
    from swaybot.memory import MemoryStore, ReflectionStep

    store = MemoryStore()
    store.add(
        ReflectionStep(
            content="avoid recursion in demo tasks",
            scope="long_term",
            tags=["demo"],
            credibility=0.8,
        )
    )
    store.add(
        ReflectionStep(
            content="low confidence tip for demo tasks",
            scope="long_term",
            tags=["demo"],
            credibility=0.3,
        )
    )
    agent = Agent(memory=store)
    guidance = agent._behavior_guidance("demo")
    assert "avoid recursion" in guidance
    assert "low confidence" not in guidance
    from swaybot.memory import Memory, MemoryStore

    store = MemoryStore()
    store.add(Memory(content="short demo note", scope="short_term", tags=["demo"]))
    store.add(Memory(content="long demo fact", scope="long_term", tags=["demo"]))
    store.add(Memory(content="unrelated", scope="long_term", tags=["other"]))
    agent = Agent(memory=store)
    context = agent._memory_context("demo")
    assert "long demo fact" in context
    assert "short demo note" not in context
    assert "unrelated" not in context
