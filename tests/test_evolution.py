"""Tests for the evidence-gated self-evolution director."""

from swaybot.agent import Agent
from swaybot.agent_events import (
    ExplorationBegin,
    ExplorationEnd,
    ExperimentRecorded,
    RunBegin,
    RunEnd,
)
from swaybot.brain import EchoBrain
from swaybot.evolution import SelfEvolutionDirector
from swaybot.memory import MemoryStore, ReflectionStep
from swaybot.reflection import Reflector


def _agent_with_questions(store, questions, kinds=None, on_event=None):
    kinds = kinds or ["question"] * len(questions)
    for content, kind in zip(questions, kinds):
        store.add(ReflectionStep(content=content, kind=kind, scope="long_term"))
    return Agent(
        brain=EchoBrain(),
        memory=store,
        reflector=Reflector(store),
        on_event=on_event,
    )


def test_cycle_emits_full_event_chain(tmp_path):
    store = MemoryStore(path=tmp_path / "memory.json")
    events: list = []
    # CLI 同款接线：agent 与 director 共享同一事件 sink → 完整链路按序交错
    agent = _agent_with_questions(
        store,
        ["does add commute?", "does echo keep punctuation?"],
        on_event=events.append,
    )
    director = SelfEvolutionDirector(agent, min_open=2, max_runs=1, emit=events.append)
    emitted = director.cycle()

    # 事件顺序：ExplorationBegin → (agent 的 RunBegin..RunEnd 经共享 sink) →
    # ExperimentRecorded → ExplorationEnd
    kinds = [type(e) for e in events]
    assert kinds[0] is ExplorationBegin
    assert RunBegin in kinds and RunEnd in kinds
    assert kinds[-2] is ExperimentRecorded
    assert kinds[-1] is ExplorationEnd
    assert len(emitted) == 3

    # 实验记录入库并带 relation 回链
    experiments = [
        m for m in store.memories if getattr(m, "kind", "") == "experiment"
    ]
    assert experiments and experiments[0].relation == "does add commute?"


def test_cycle_converges_on_resolved_questions(tmp_path):
    store = MemoryStore(path=tmp_path / "memory.json")
    agent = _agent_with_questions(store, ["does add commute?", "is echo stable?"])
    director = SelfEvolutionDirector(agent, min_open=2, max_runs=5)
    first = director.cycle()
    assert first  # 有探索发生

    # 反思已把验证过的假设标记 resolved → 开放集缩小 → 门关上
    open_after = director.open_candidates()
    assert len(open_after) < 2
    second = director.cycle()
    assert second == []


def test_cycle_prefers_contradictions(tmp_path):
    store = MemoryStore(path=tmp_path / "memory.json")
    agent = _agent_with_questions(
        store,
        ["open question two?", "the contradiction first?"]
        if False else ["plain question?", "a conflicts with b"],
        kinds=["question", "contradiction"],
    )
    director = SelfEvolutionDirector(agent, min_open=2, max_runs=1)
    emitted = director.cycle()
    assert isinstance(emitted[0], ExplorationBegin)
    assert emitted[0].hypothesis == "a conflicts with b"


def test_gate_closed_below_min_open(tmp_path):
    store = MemoryStore(path=tmp_path / "memory.json")
    agent = _agent_with_questions(store, ["only one question?"])
    director = SelfEvolutionDirector(agent, min_open=2, max_runs=3)
    assert director.cycle() == []


def test_testing_status_blocks_duplicate_pickup(tmp_path):
    store = MemoryStore(path=tmp_path / "memory.json")
    agent = _agent_with_questions(store, ["q one?", "q two?"])
    director = SelfEvolutionDirector(agent, min_open=2, max_runs=1)
    director.cycle()
    statuses = {
        m.content: getattr(m, "status", "open")
        for m in store.memories
        if getattr(m, "kind", "") in ("question", "contradiction")
    }
    # 被测的进入 testing（验证后 resolved），其余保持 open
    assert set(statuses.values()) <= {"open", "testing", "resolved"}
    assert "resolved" in statuses.values()
