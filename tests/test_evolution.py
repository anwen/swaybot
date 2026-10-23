"""Tests for the evidence-gated self-evolution director."""

from swaybot.agent import Agent
from swaybot.agent_events import (
    ExperimentDesigned,
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


def _protocol_steps(rounds, run_answer="checked", verdict="supported"):
    """每轮 3 次模型调用：设计(JSON) → 执行(final_answer) → 裁决(JSON)。"""
    from swaybot.models import ScriptedModel  # noqa: F401

    steps = []
    for _ in range(rounds):
        steps.append(
            {
                "action": {
                    "name": "final_answer",
                    "args": {
                        "answer": '{"conjecture": "c1", "method": "m1", '
                        '"success_criteria": "s1"}'
                    },
                }
            }
        )
        steps.append({"action": {"name": "final_answer", "args": {"answer": run_answer}}})
        steps.append(
            {
                "action": {
                    "name": "final_answer",
                    "args": {"answer": f'{{"verdict": "{verdict}", "evidence": "e"}}'},
                }
            }
        )
    return steps


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
    from swaybot.llm_brain import LLMBrain
    from swaybot.models import ScriptedModel

    store = MemoryStore(path=tmp_path / "memory.json")
    events: list = []
    # CLI 同款接线：agent 与 director 共享同一事件 sink → 完整链路按序交错
    agent = _agent_with_questions(
        store,
        ["does add commute?", "does echo keep punctuation?"],
        on_event=events.append,
    )
    agent.brain = LLMBrain(backend=ScriptedModel(_protocol_steps(1)))
    director = SelfEvolutionDirector(agent, min_open=2, max_runs=1, emit=events.append)
    emitted = director.cycle()

    # 事件顺序：Begin → Designed → (RunBegin..RunEnd 经共享 sink) → Recorded → End
    kinds = [type(e) for e in events]
    assert kinds[0] is ExplorationBegin
    assert kinds[1] is ExperimentDesigned
    assert RunBegin in kinds and RunEnd in kinds
    assert kinds[-2] is ExperimentRecorded
    assert kinds[-1] is ExplorationEnd
    assert len(emitted) == 4

    # 实验记录入库并带 relation 回链 + 协议裁决
    experiments = [
        m for m in store.memories if getattr(m, "kind", "") == "experiment"
    ]
    assert experiments and experiments[0].relation == "does add commute?"
    assert "verdict=supported" in experiments[0].content
    assert "method=m1" in experiments[0].content


def test_cycle_converges_on_resolved_questions(tmp_path):
    store = MemoryStore(path=tmp_path / "memory.json")
    agent = _agent_with_questions(store, ["does add commute?", "is echo stable?"])
    from swaybot.llm_brain import LLMBrain
    from swaybot.models import ScriptedModel

    agent.brain = LLMBrain(backend=ScriptedModel(_protocol_steps(6)))
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
    from swaybot.llm_brain import LLMBrain
    from swaybot.models import ScriptedModel

    agent.brain = LLMBrain(backend=ScriptedModel(_protocol_steps(1)))
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


def test_conjecture_lifecycle_and_decomposition(tmp_path):
    """猜想入库 → inconclusive 裁决 → 置信度不动但触发子问题分解 → 子问题带 relation/depth。"""
    from swaybot.agent_events import ExperimentDesigned, SubQuestionsAdded
    from swaybot.llm_brain import LLMBrain
    from swaybot.models import ScriptedModel

    store = MemoryStore(path=tmp_path / "memory.json")
    agent = _agent_with_questions(store, ["what am I?", "another open q"])
    steps = [
        # 第 1 轮：设计 → 执行 → 裁决(inconclusive) → 分解(2 子问题)
        {"action": {"name": "final_answer", "args": {"answer": '{"conjecture": "I am my policy", "method": "m", "success_criteria": "s"}'}}},
        {"action": {"name": "final_answer", "args": {"answer": "did stuff"}}},
        {"action": {"name": "final_answer", "args": {"answer": '{"verdict": "inconclusive", "evidence": "e"}'}}},
        {"action": {"name": "final_answer", "args": {"answer": '{"sub_questions": ["can I run python here?", "does my answer survive context reset?"]}'}}},
        # 第 2 轮（子问题）：设计 → 执行 → 裁决(supported)
        {"action": {"name": "final_answer", "args": {"answer": '{"conjecture": "python runs", "method": "m2", "success_criteria": "s2"}'}}},
        {"action": {"name": "final_answer", "args": {"answer": "ran python"}}},
        {"action": {"name": "final_answer", "args": {"answer": '{"verdict": "supported", "evidence": "exit 0"}'}}},
    ]
    agent.brain = LLMBrain(backend=ScriptedModel(steps))
    director = SelfEvolutionDirector(agent, min_open=1, max_runs=2, max_depth=2)
    emitted = director.cycle()

    kinds = [type(e).__name__ for e in emitted]
    assert "SubQuestionsAdded" in kinds and "ExperimentDesigned" in kinds

    # 猜想作为一等公民入库
    conjectures = [m for m in store.memories if getattr(m, "kind", "") == "conjecture"]
    assert any(m.relation == "what am I?" and m.content == "I am my policy" for m in conjectures)
    # supported 裁决把对应猜想置信度 0.3 → 0.5
    python_conj = next(m for m in conjectures if m.content == "python runs")
    assert abs(python_conj.credibility - 0.5) < 1e-9

    # 子问题带 relation 与 depth=1；父问题状态 decomposed
    subs = [m for m in store.memories if getattr(m, "kind", "") == "question" and getattr(m, "relation", "")]
    assert subs and all(getattr(m, "depth", 0) == 1 for m in subs)
    parent = next(m for m in store.memories if getattr(m, "content", "") == "what am I?")
    assert parent.status == "decomposed"


def test_decomposition_depth_cap(tmp_path):
    """超过 max_depth 的 inconclusive 不再分解。"""
    from swaybot.llm_brain import LLMBrain
    from swaybot.models import ScriptedModel

    store = MemoryStore(path=tmp_path / "memory.json")
    agent = _agent_with_questions(store, ["deep q?", "another q"])
    # 手动把两个问题都设为深度上限（浅层优先排序下任一都可能被选中）
    for m in store.memories:
        m.depth = 2
    steps = [
        {"action": {"name": "final_answer", "args": {"answer": '{"conjecture": "c", "method": "m", "success_criteria": "s"}'}}},
        {"action": {"name": "final_answer", "args": {"answer": "did"}}},
        {"action": {"name": "final_answer", "args": {"answer": '{"verdict": "inconclusive", "evidence": "e"}'}}},
    ]
    agent.brain = LLMBrain(backend=ScriptedModel(steps))
    director = SelfEvolutionDirector(agent, min_open=1, max_runs=1, max_depth=2)
    emitted = director.cycle()
    assert not any(type(e).__name__ == "SubQuestionsAdded" for e in emitted)


def test_sediment_views_render_knowledge_and_curiosity(tmp_path):
    """沉淀视图：知识按置信度排序带证据链；好奇按调度优先级排序。"""
    from swaybot.evolution import write_sediment
    from swaybot.memory import Memory, MemoryStore

    store = MemoryStore(path=tmp_path / "memory.json")
    # 三条猜想不同置信度 + 一条理论
    store.add(Memory(content="I am my policy", kind="conjecture", scope="long_term",
                     relation="who am I?", credibility=0.8))
    store.add(Memory(content="world is sandboxed", kind="conjecture", scope="long_term",
                     relation="what world?", credibility=0.3))
    store.add(Memory(content="weak guess", kind="conjecture", scope="long_term",
                     relation="x?", credibility=0.1))
    # 证据链：两条实验（supported + inconclusive）挂在 who am I?
    store.add(Memory(content="Experiment on 'who am I?': verdict=supported", kind="experiment",
                     scope="long_term", relation="who am I?", credibility=0.5))
    store.add(Memory(content="Experiment on 'who am I?': verdict=inconclusive", kind="experiment",
                     scope="long_term", relation="who am I?", credibility=0.5))
    # 好奇列表：一个矛盾（应排最前）+ 深浅两个问题
    from swaybot.memory import ReflectionStep

    store.add(ReflectionStep(content="deep sub q", kind="question", scope="long_term",
                             depth=1, relation="parent"))
    store.add(ReflectionStep(content="shallow q", kind="question", scope="long_term", depth=0))
    store.add(ReflectionStep(content="a vs b conflict", kind="contradiction", scope="long_term"))
    store.add(ReflectionStep(content="resolved q", kind="question", scope="long_term",
                             status="resolved"))

    knowledge, curiosity = write_sediment(store, tmp_path)
    k = knowledge.read_text(encoding="utf-8")
    assert "I am my policy" in k and "★" in k  # ≥0.8 标星
    assert k.index("I am my policy") < k.index("weak guess")  # 置信度降序
    assert "2 experiment(s): supported=1" in k  # 证据链

    c = curiosity.read_text(encoding="utf-8")
    numbered = [l for l in c.splitlines() if l.startswith("1. ")]
    first_line = numbered[0]  # 排序后的第一条
    assert "a vs b conflict" in first_line  # 矛盾优先
    assert "resolved q" not in c  # 已解决不出现
    assert c.index("shallow q") < c.index("deep sub q")  # 浅层在前


def test_belief_counts_and_decay(tmp_path):
    """裁决累加计数 + last_verified_at；有效置信度随未验证时间半衰。"""
    from datetime import datetime, timedelta, timezone

    from swaybot.evolution import SelfEvolutionDirector, _parse_json_object
    from swaybot.llm_brain import LLMBrain
    from swaybot.memory import Memory, MemoryStore, effective_credibility
    from swaybot.models import ScriptedModel

    store = MemoryStore(path=tmp_path / "m.json")
    store.add(Memory(content="conj", kind="conjecture", scope="long_term",
                     relation="q?", credibility=0.5, supported_count=0, refuted_count=0))
    agent = _agent_with_questions(store, ["q?", "another"])
    agent.brain = LLMBrain(backend=ScriptedModel([
        {"action": {"name": "final_answer", "args": {"answer":
            '{"conjecture": "c", "method": "m", "success_criteria": "s", "applicability": "macOS sandbox"}'}}},
        {"action": {"name": "final_answer", "args": {"answer": "ran"}}},
        {"action": {"name": "final_answer", "args": {"answer": '{"verdict": "supported", "evidence": "e"}'}}},
    ]))
    SelfEvolutionDirector(agent, min_open=1, max_runs=1).cycle()
    conj = next(m for m in store.memories if getattr(m, "kind", "") == "conjecture" and m.relation == "q?")
    assert conj.supported_count == 1 and conj.last_verified_at
    # 设计带出的适用条件入库存（新猜想）
    designed = [m for m in store.memories if getattr(m, "kind", "") == "conjecture"
                and getattr(m, "applicability", "")]
    assert designed and designed[0].applicability == "macOS sandbox"

    # 半衰：30 天前验证过的 0.8 → 有效 ~0.4（一个半衰期）
    old = Memory(content="old", kind="theory", scope="long_term", credibility=0.8,
                 last_verified_at=(datetime.now(timezone.utc) - timedelta(days=30)).isoformat())
    assert abs(effective_credibility(old, half_life_days=30) - 0.4) < 0.01
    fresh = Memory(content="new", kind="theory", scope="long_term", credibility=0.6,
                   last_verified_at=datetime.now(timezone.utc).isoformat())
    assert effective_credibility(fresh) > effective_credibility(old)
