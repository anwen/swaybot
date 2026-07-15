import pytest

from swaybot.agent import Agent
from swaybot.memory import Memory, MemoryStore
from swaybot.reflection import Reflector, Reflection, reflection_to_memory


def test_reflection_to_memory():
    r = Reflection(content="test", kind="summary", confidence=0.9)
    m = reflection_to_memory(r)
    assert m.kind == "theory"
    assert m.scope == "long_term"
    assert m.source == "reflector"
    assert m.credibility == 0.9


def test_reflect_on_run_summary():
    store = MemoryStore()
    reflector = Reflector(store)
    reflections = reflector.reflect_on_run("demo", [{}, {}])
    assert any(r.kind == "summary" for r in reflections)


def test_reflect_on_run_detects_surprises():
    store = MemoryStore()
    store.add(Memory(content="odd result", tags=["demo"], surprise=0.9))
    reflector = Reflector(store)
    reflections = reflector.reflect_on_run("demo", [{}])
    assert any(r.kind == "question" for r in reflections)


def test_reflect_on_run_detects_contradictions():
    store = MemoryStore()
    store.add(Memory(content="sky is blue", tags=["demo"], credibility=0.9))
    store.add(
        Memory(
            content="sunset sky is red",
            tags=["demo"],
            surprise=0.6,
            credibility=0.8,
        )
    )
    reflector = Reflector(store)
    reflections = reflector.reflect_on_run("demo", [{}])
    assert any(r.kind == "contradiction" for r in reflections)


def test_verify_claim_with_counterexample():
    store = MemoryStore()
    store.add(
        Memory(
            content="sky appears red at sunset",
            kind="fact",
            credibility=0.9,
            surprise=0.6,
        )
    )
    reflector = Reflector(store)
    result = reflector.verify_claim("The sky is always blue", tag="demo")
    assert result.kind == "verification"
    assert "counterexamples" in result.content


def test_verify_claim_with_support():
    store = MemoryStore()
    store.add(Memory(content="sky is blue", kind="fact", credibility=0.9, tags=["demo"]))
    reflector = Reflector(store)
    result = reflector.verify_claim("The sky is blue", tag="demo")
    assert result.kind == "verification"
    assert "supported" in result.content


def test_verify_claim_insufficient_evidence():
    store = MemoryStore()
    reflector = Reflector(store)
    result = reflector.verify_claim("Mars is made of cheese", tag="demo")
    assert result.kind == "verification"
    assert "insufficient evidence" in result.content


def test_agent_records_reflection_memories():
    store = MemoryStore()
    reflector = Reflector(store)
    agent = Agent(memory=store, reflector=reflector)
    agent.run("demo", max_steps=2)
    theories = store.query(kind="theory")
    assert len(theories) >= 1
    assert any(t.source == "reflector" for t in theories)


def test_agent_reflect_flag_disabled():
    store = MemoryStore()
    reflector = Reflector(store)
    agent = Agent(memory=store, reflector=reflector)
    agent.run("demo", max_steps=2, reflect=False)
    assert len(store.query(kind="theory")) == 0
