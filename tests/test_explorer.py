from swaybot.agent import Agent
from swaybot.brain import EchoBrain
from swaybot.explorer import Explorer, parse_exploration_response
from swaybot.memory import MemoryStore, ReflectionStep


def test_explorer_generates_default_task_with_echo_brain():
    agent = Agent(brain=EchoBrain(), memory=MemoryStore())
    explorer = Explorer(agent)
    task = explorer.generate_task()
    assert task.task
    assert "hypothesis" in task.to_dict()


def test_explorer_runs_generated_task():
    agent = Agent(brain=EchoBrain(), memory=MemoryStore())
    explorer = Explorer(agent, max_steps=2)
    task, env = explorer.run()
    assert env.done
    assert len(env.history) == 2
    assert env.history[-1]["action"]["name"] == "done"


def test_explorer_uses_memory_summary_for_task_generation():
    store = MemoryStore()
    store.add(
        ReflectionStep(
            content="echo returns the exact input for simple strings",
            scope="long_term",
            tags=["explore"],
            credibility=0.8,
        )
    )
    agent = Agent(brain=EchoBrain(), memory=store)
    explorer = Explorer(agent)
    summary = explorer._memory_summary()
    assert "echo returns" in summary


def test_explorer_produces_reflection_after_run():
    store = MemoryStore()
    agent = Agent(brain=EchoBrain(), memory=store)
    explorer = Explorer(agent, max_steps=2)
    task, env = explorer.run()
    long_term = [m for m in store.memories if m.scope == "long_term"]
    assert long_term
    contents = " ".join(getattr(m, "content", "") for m in long_term)
    assert task.hypothesis in contents or "Run" in contents


def test_parse_exploration_response_strips_fences():
    raw = '```json\n{"task": "check add", "hypothesis": "add works"}\n```'
    result = parse_exploration_response(raw)
    assert result == {"task": "check add", "hypothesis": "add works"}


def test_parse_exploration_response_returns_empty_on_invalid_json():
    assert parse_exploration_response("not json") == {}
