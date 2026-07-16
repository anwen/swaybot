import json
from pathlib import Path

import pytest

from swaybot.agent import Agent
from swaybot.memory import (
    ActionStep,
    Consolidator,
    Dream,
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


def test_action_step_stores_metadata():
    store = MemoryStore()
    store.add(
        ActionStep(
            step=1,
            action={"name": "add", "args": {"a": 1, "b": 2}},
            tags=["demo"],
            model_input_messages=[{"role": "user", "content": "hi"}],
            raw_output='{"name": "add", "args": {"a": 1, "b": 2}}',
            token_usage={"prompt_tokens": 3, "completion_tokens": 5},
            duration_ms=12.5,
        )
    )
    step = store.memories[0]
    assert step.raw_output is not None
    assert step.token_usage["prompt_tokens"] == 3
    assert step.duration_ms == 12.5
    data = step.to_dict()
    assert data["raw_output"] == step.raw_output
    assert data["token_usage"] == step.token_usage


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


def test_memory_store_prune_removes_matching_steps():
    store = MemoryStore()
    store.add(Memory(content="long", scope="long_term", tags=["demo"]))
    store.add(TaskStep(task="demo", tags=["demo"]))
    store.add(ActionStep(step=1, action={"name": "echo"}, tags=["demo"]))
    store.add(ObservationStep(step=1, observation="ok", tags=["demo"]))

    removed = store.prune(scope="short_term", tag="demo")
    assert removed == 3
    assert len(store.memories) == 1
    assert store.memories[0].scope == "long_term"


def test_memory_store_query_relevant_ranks_by_overlap():
    store = MemoryStore()
    store.add(Memory(content="sky is blue", scope="long_term", tags=["sky"]))
    store.add(Memory(content="cats are mammals", scope="long_term", tags=["animals"]))
    store.add(Memory(content="ocean is deep blue", scope="long_term", tags=["ocean"]))

    results = store.query_relevant("blue sky", limit=2)
    assert len(results) == 2
    assert results[0].content == "sky is blue"
    assert results[1].content == "ocean is deep blue"


def test_agent_prunes_short_term_after_reflection():
    from swaybot.reflection import Reflector

    store = MemoryStore()
    reflector = Reflector(store)
    agent = Agent(memory=store, reflector=reflector)
    agent.run("demo", max_steps=2)
    theories = store.query(kind="theory")
    short_term = store.query(scope="short_term")
    assert len(theories) >= 1
    assert len(short_term) == 0


def test_consolidator_archives_short_term_to_history(tmp_path: Path):
    history_path = tmp_path / "history.jsonl"
    store = MemoryStore()
    store.add(TaskStep(task="demo", tags=["demo"]))
    store.add(ActionStep(step=1, action={"name": "echo"}, tags=["demo"]))
    store.add(ObservationStep(step=1, observation="ok", tags=["demo"]))

    consolidator = Consolidator(history_path=history_path)
    archived = consolidator.archive(store, tag="demo")

    assert len(archived) == 3
    assert all(m.scope == "short_term" for m in archived)
    assert len(store.query(scope="short_term", tag="demo")) == 0

    lines = history_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3
    assert json.loads(lines[0])["step_kind"] == "task"


def test_consolidator_keeps_last_n(tmp_path: Path):
    history_path = tmp_path / "history.jsonl"
    store = MemoryStore()
    for i in range(3):
        store.add(ActionStep(step=i, action={"name": "echo"}, tags=["demo"]))

    consolidator = Consolidator(history_path=history_path)
    archived = consolidator.archive(store, tag="demo", keep_last=1)

    assert len(archived) == 2
    assert len(store.query(scope="short_term", tag="demo")) == 1
    assert store.query(scope="short_term", tag="demo")[0].step == 2


def test_consolidator_summarizes_with_brain(tmp_path: Path):
    history_path = tmp_path / "history.jsonl"
    store = MemoryStore()
    store.add(ActionStep(step=1, action={"name": "echo"}, tags=["demo"]))

    def summarize(steps):
        return "summarized"

    consolidator = Consolidator(history_path=history_path, brain=summarize)
    consolidator.archive(store, tag="demo", summarize=True)

    long_term = store.query(scope="long_term", tag="consolidated")
    assert len(long_term) == 1
    assert long_term[0].content == "summarized"


def test_dream_appends_insights_to_durable_memory(tmp_path: Path):
    durable_path = tmp_path / "SOUL.md"
    store = MemoryStore()
    store.add(ReflectionStep(content="always verify assumptions", tags=["demo"]))

    dream = Dream(durable_path=durable_path)
    content = dream.edit(store)

    assert "Consolidated insights" in content
    assert "always verify assumptions" in content
    assert "always verify assumptions" in durable_path.read_text(encoding="utf-8")


def test_dream_brain_can_rewrite_durable_memory(tmp_path: Path):
    durable_path = tmp_path / "SOUL.md"
    durable_path.write_text("# Old\n", encoding="utf-8")
    store = MemoryStore()
    store.add(ReflectionStep(content="new insight", tags=["demo"]))

    def rewrite(content, insights):
        return f"{content.strip()}\n\nnew: {insights[0].content}"

    dream = Dream(durable_path=durable_path, brain=rewrite)
    content = dream.edit(store)

    assert content.startswith("# Old")
    assert "new: new insight" in content
