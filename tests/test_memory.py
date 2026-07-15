import json
from pathlib import Path

import pytest

from swaybot.agent import Agent
from swaybot.memory import Memory, MemoryStore


def test_memory_defaults():
    m = Memory(content="test")
    assert m.kind == "experience"
    assert 0.0 <= m.credibility <= 1.0
    assert m.created_at


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


def test_agent_records_memories():
    store = MemoryStore()
    agent = Agent(memory=store)
    env = agent.run("demo", max_steps=3)
    assert env.done
    assert len(store.memories) == 3
    assert all("demo" in m.tags for m in store.memories)


def test_agent_without_memory_unchanged():
    agent = Agent()
    env = agent.run("demo", max_steps=2)
    assert env.done
    assert len(env.history) == 2
