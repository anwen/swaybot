import json
from pathlib import Path

import pytest

from swaybot.agent import Agent
from swaybot.memory import (
    ActionStep,
    Memory,
    MemoryStore,
    ObservationStep,
    PlanningStep,
    ReflectionStep,
    TaskStep,
)


def test_memory_defaults():
    m = Memory(content="test")
    assert m.kind == "experience"
    assert m.scope == "long_term"
    assert 0.0 <= m.credibility <= 1.0
    assert m.created_at


def test_memory_store_scope_filter():
    store = MemoryStore()
    store.add(Memory(content="short", scope="short_term", tags=["task-a"]))
    store.add(Memory(content="long", scope="long_term", tags=["task-a"]))
    assert len(store.query(tag="task-a", scope="short_term")) == 1
    assert store.query(tag="task-a", scope="short_term")[0].content == "short"
    assert len(store.query(tag="task-a", scope="long_term")) == 1
    assert store.query(tag="task-a", scope="long_term")[0].content == "long"


def test_memory_store_add_and_query():
    store = MemoryStore()
    store.add(Memory(content="alpha", tags=["task-a"], surprise=0.8))
    store.add(Memory(content="beta", tags=["task-b"], surprise=0.2))
    store.add(Memory(content="gamma", tags=["task-a"], kind="fact"))

    assert len(store.query(tag="task-a")) == 2
    assert len(store.query(kind="fact")) == 1
    assert len(store.query(min_surprise=0.5)) == 1
    assert store.query(tag="task-a", limit=1)[0].content == "gamma"


def test_memory_store_persistence(tmp_path: Path):
    path = tmp_path / "memory.json"
    store = MemoryStore(path=path)
    store.add(Memory(content="persistent", tags=["x"]))

    store2 = MemoryStore(path=path)
    assert len(store2.memories) == 1
    assert store2.memories[0].content == "persistent"
    assert json.loads(path.read_text(encoding="utf-8"))[0]["content"] == "persistent"


def test_find_counterexamples():
    store = MemoryStore()
    store.add(Memory(content="The sky is blue", kind="fact", credibility=0.9, tags=["sky"]))
    store.add(Memory(content="At sunset the sky appears red", kind="experience", surprise=0.6, tags=["sky"]))
    store.add(Memory(content="Cats are mammals", kind="fact", credibility=0.9, tags=["animals"]))

    results = store.find_counterexamples("The sky is always blue")
    assert any("red" in m.content for m in results)


def test_find_counterexamples_no_overlap():
    store = MemoryStore()
    store.add(Memory(content=" unrelated ", credibility=0.9, surprise=0.9))
    assert store.find_counterexamples("completely different topic") == []


def test_agent_records_short_term_memories():
    store = MemoryStore()
    agent = Agent(memory=store)
    env = agent.run("demo", max_steps=3)
    assert env.done
    assert len(store.memories) == 7  # task + 3 actions + 3 observations
    assert all(m.scope == "short_term" for m in store.memories)
    assert all("demo" in m.tags for m in store.memories)
    assert any(m.source == "user" for m in store.memories)


def test_agent_records_memories():
    store = MemoryStore()
    agent = Agent(memory=store)
    env = agent.run("demo", max_steps=3)
    assert env.done
    assert len(store.memories) == 7  # task + 3 actions + 3 observations
    assert all("demo" in m.tags for m in store.memories)


def test_agent_without_memory_unchanged():
    agent = Agent()
    env = agent.run("demo", max_steps=2)
    assert env.done
    assert len(env.history) == 2


def test_typed_steps_to_messages():
    task = TaskStep(task="demo", max_steps=3, tags=["demo"])
    assert task.to_messages()[0]["role"] == "user"
    assert "Task: demo" in task.to_messages()[0]["content"]

    action = ActionStep(step=1, action={"name": "echo"}, tags=["demo"])
    assert action.to_messages()[0]["role"] == "assistant"
    assert '"name": "echo"' in action.to_messages()[0]["content"]

    obs = ObservationStep(step=1, observation="result", tags=["demo"])
    assert obs.to_messages()[0]["role"] == "user"
    assert "result" in obs.to_messages()[0]["content"]

    plan = PlanningStep(plan=["search", "summarize"], tags=["demo"])
    assert plan.to_messages()[0]["role"] == "user"
    assert "1. search" in plan.to_messages()[0]["content"]

    reflection = ReflectionStep(content="theory", credibility=0.9, tags=["demo"])
    assert reflection.to_messages()[0]["role"] == "user"
    assert reflection.scope == "long_term"
    assert reflection.kind == "theory"


def test_memory_store_persists_typed_steps(tmp_path: Path):
    path = tmp_path / "memory.json"
    store = MemoryStore(path=path)
    store.add(TaskStep(task="demo", max_steps=3, tags=["demo"]))
    store.add(ActionStep(step=1, action={"name": "echo"}, tags=["demo"]))
    store.add(ObservationStep(step=1, observation="ok", tags=["demo"]))

    store2 = MemoryStore(path=path)
    assert len(store2.memories) == 3
    assert isinstance(store2.memories[0], TaskStep)
    assert isinstance(store2.memories[1], ActionStep)
    assert isinstance(store2.memories[2], ObservationStep)
