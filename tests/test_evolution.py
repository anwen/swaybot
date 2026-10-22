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


# ---------- 灵魂慢通道 ----------

def _aged_theory(content, credibility=0.9, days_old=10):
    from datetime import datetime, timedelta, timezone

    from swaybot.memory import ReflectionStep

    return ReflectionStep(
        content=content,
        kind="theory",
        scope="long_term",
        credibility=credibility,
        created_at=(
            datetime.now(timezone.utc) - timedelta(days=days_old)
        ).isoformat(),
    )


def _soul_store(tmp_path, theories, contradictions=()):
    store = MemoryStore(path=tmp_path / "memory.json")
    for theory in theories:
        store.add(theory)
    for content in contradictions:
        store.add(
            ReflectionStep(
                content=content, kind="contradiction", scope="long_term"
            )
        )
    return store


def test_soul_channel_proposal_gates(tmp_path):
    from swaybot.evolution import SoulChannel

    # 不足 5 条高可信理论 → 不提案
    store = _soul_store(tmp_path, [_aged_theory(f"t{i}") for i in range(3)])
    assert SoulChannel(store, tmp_path).propose() is None

    # 5 条达标 → 提案（diff 含新增行）
    store = _soul_store(tmp_path, [_aged_theory(f"stable theory {i}") for i in range(5)])
    proposal = SoulChannel(store, tmp_path).propose()
    assert proposal is not None
    digest, diff = proposal
    assert "stable theory 0" in digest
    assert diff.startswith("---") and "+- (credibility" in diff


def test_soul_channel_age_and_credibility_gates(tmp_path):
    from swaybot.evolution import SoulChannel

    store = _soul_store(
        tmp_path,
        [_aged_theory(f"old hi {i}", credibility=0.9, days_old=10) for i in range(5)]
        + [
            _aged_theory("young one", credibility=0.95, days_old=0),
            _aged_theory("weak old one", credibility=0.5, days_old=30),
        ],
    )
    proposal = SoulChannel(store, tmp_path).propose()
    assert proposal is not None
    digest = proposal[0]
    assert "old hi" in digest
    assert "young one" not in digest and "weak old one" not in digest


def test_soul_channel_quiet_brake(tmp_path):
    from swaybot.evolution import SoulChannel

    store = _soul_store(
        tmp_path,
        [_aged_theory(f"stable {i}") for i in range(5)],
        contradictions=["a conflicts with b"],  # created NOW
    )
    assert SoulChannel(store, tmp_path, quiet_days=1.0).propose() is None


def test_soul_channel_writes_only_with_approval(tmp_path):
    from swaybot.evolution import SoulChannel

    store = _soul_store(tmp_path, [_aged_theory(f"stable {i}") for i in range(5)])
    channel = SoulChannel(store, tmp_path)
    digest, _diff = channel.propose()
    target = tmp_path / ".swaybot" / "soul-digest.md"
    channel.apply(digest, approve=False)
    assert not target.exists()  # 未批准：零写入
    outcome = channel.apply(digest, approve=True)
    assert "written" in outcome
    assert target.exists() and "stable 0" in target.read_text(encoding="utf-8")


def test_soul_digest_loads_into_instructions(tmp_path):
    from swaybot.instructions import _collect_instruction_paths
    from pathlib import Path as P

    root = tmp_path
    (root / ".swaybot").mkdir()
    (root / ".swaybot" / "soul-digest.md").write_text("# Soul digest\n- x", encoding="utf-8")
    paths = _collect_instruction_paths(root)
    assert any(p.name == "soul-digest.md" for p in paths)
