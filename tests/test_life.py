"""Tests for the continuous LifeRunner (explore -> wonder -> dream)."""

from swaybot.agent import Agent
from swaybot.agent_events import DreamBegin, DreamEnd, Wondered
from swaybot.brain import EchoBrain
from swaybot.evolution import LifeRunner
from swaybot.llm_brain import LLMBrain
from swaybot.memory import MemoryStore, ReflectionStep
from swaybot.models import ScriptedModel
from swaybot.reflection import Reflector

SEED = "探索：我是谁？这个世界是怎样的？我做什么是有价值的？"


def test_seed_prompt_splits_into_questions(tmp_path):
    store = MemoryStore(path=tmp_path / "memory.json")
    agent = Agent(brain=EchoBrain(), memory=store, reflector=Reflector(store))
    runner = LifeRunner(agent, SEED, stop_check=lambda: True)
    runner.live()  # 立即停止，但种子已入库
    questions = [m for m in store.memories if getattr(m, "kind", "") == "question"]
    assert len(questions) == 3


def _protocol_step(kind):
    """一轮 = 设计(JSON) → 执行(final_answer) → 裁决(JSON)。"""
    if kind == "design":
        return {
            "action": {
                "name": "final_answer",
                "args": {
                    "answer": '{"conjecture": "c", "method": "m", '
                    '"success_criteria": "s"}'
                },
            }
        }
    if kind == "adjudicate":
        return {
            "action": {
                "name": "final_answer",
                "args": {"answer": '{"verdict": "supported", "evidence": "e"}'},
            }
        }
    return {"action": {"name": "final_answer", "args": {"answer": "checked"}}}


def _scripted_life_brain():
    # 3 轮完整协议 → wonder(3 问文本) → dream(理论+新问) → 余量
    steps = []
    for _ in range(3):
        steps.extend([_protocol_step("design"), _protocol_step("run"), _protocol_step("adjudicate")])
    steps.append({"text": "what is my purpose?\nhow do tools shape me?\ncan i trust memory?"})
    steps.append({"text": "- I learn through verified tool use\n? what deserves long-term belief?"})
    steps.extend(_protocol_step("run") for _ in range(9))
    return LLMBrain(backend=ScriptedModel(steps))


def test_life_runner_full_cycle_and_graceful_stop(tmp_path):
    store = MemoryStore(path=tmp_path / "memory.json")
    events: list = []
    agent = Agent(
        brain=_scripted_life_brain(),
        memory=store,
        reflector=Reflector(store),
        on_event=events.append,
    )
    state = {"rounds_seen": 0}
    import itertools

    counter = itertools.count()

    def stop_check():
        # 第二次评估时停——首轮已覆盖 探索→wonder→dream 全相位
        return next(counter) >= 1

    runner = LifeRunner(
        agent,
        SEED,
        emit=events.append,
        workspace=tmp_path,
        min_open=1,
        max_rounds_per_cycle=3,
        rounds_between_dreams=1,
        stop_check=stop_check,
    )
    summary = runner.live()

    kinds = [type(e) for e in events]
    assert Wondered in kinds
    assert DreamBegin in kinds and DreamEnd in kinds
    assert summary["rounds"] >= 1
    theories = [m for m in store.memories if getattr(m, "kind", "") == "theory"]
    assert theories  # dream 沉淀了理论
    experiments = [m for m in store.memories if getattr(m, "kind", "") == "experiment"]
    assert experiments  # 探索留下了实验记录


def test_life_runner_dies_gracefully_without_curiosity(tmp_path):
    """确定性大脑（EchoBrain）无法 wonder → 干涸三连后自然止息，不死循环。"""
    store = MemoryStore(path=tmp_path / "memory.json")
    agent = Agent(brain=EchoBrain(), memory=store, reflector=Reflector(store))
    runner = LifeRunner(agent, "one seed question?", min_open=1, max_rounds_per_cycle=1)
    summary = runner.live()
    assert summary["rounds"] >= 1
    assert summary["wonders"] == 0  # EchoBrain 每次都答不出问题
