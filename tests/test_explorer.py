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


def test_explorer_prefers_question_from_memory():
    store = MemoryStore()
    store.add(
        ReflectionStep(
            content="Why does echo return an empty string for None?",
            kind="question",
            scope="long_term",
            tags=["explore"],
            credibility=0.6,
        )
    )
    agent = Agent(brain=EchoBrain(), memory=store)
    explorer = Explorer(agent)
    task = explorer.generate_task()
    assert "empty string for None" in task.task


def test_explorer_prefers_contradiction_from_memory():
    store = MemoryStore()
    store.add(
        ReflectionStep(
            content="Possible contradiction to: the sky is always blue",
            kind="contradiction",
            scope="long_term",
            tags=["explore"],
            credibility=0.5,
        )
    )
    agent = Agent(brain=EchoBrain(), memory=store)
    explorer = Explorer(agent)
    task = explorer.generate_task()
    assert "contradiction" in task.task.lower()


def test_explorer_passes_candidate_hypotheses_to_brain():
    class CapturingBrain:
        def __init__(self):
            self.perception = None

        def think(self, perception, available_tools, metadata=None):
            self.perception = perception
            return {
                "name": "explore",
                "args": {"task": "investigate", "hypothesis": "probe"},
            }

    store = MemoryStore()
    store.add(
        ReflectionStep(
            content="Why does add fail with strings?",
            kind="question",
            scope="long_term",
            tags=["explore"],
        )
    )
    brain = CapturingBrain()
    agent = Agent(brain=brain, memory=store)
    explorer = Explorer(agent)
    explorer.generate_task()
    assert brain.perception is not None
    assert "Why does add fail with strings?" in brain.perception.get(
        "candidate_hypotheses", []
    )
