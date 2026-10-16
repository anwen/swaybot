"""Tests for the agent loop's typed event contract."""

from swaybot.agent import Agent
from swaybot.agent_events import (
    ReasoningDelta,
    RunBegin,
    RunEnd,
    StepBegin,
    StepEnd,
    TokenDelta,
    to_ndjson,
)
from swaybot.memory import MemoryStore
from swaybot.permission import PermissionPolicy
from swaybot.models import ScriptedModel


def _collect(agent_kwargs: dict, run_kwargs: dict) -> list:
    events: list = []
    agent = Agent(on_event=events.append, **agent_kwargs)
    agent.run(**run_kwargs)
    return events


def test_event_sequence_for_echo_run():
    events = _collect({}, {"task": "count", "max_steps": 2})
    assert isinstance(events[0], RunBegin)
    assert events[0].task == "count"
    assert events[0].max_steps == 2
    assert events[-1] == RunEnd(steps=2, done=True, reflections=[])
    body = events[1:-1]
    # Each step is a StepBegin/StepEnd pair with matching, 1-based numbers.
    assert [type(e) for e in body] == [StepBegin, StepEnd, StepBegin, StepEnd]
    assert [e.step for e in body] == [1, 1, 2, 2]


def test_step_end_carries_denied_permission_error():
    from swaybot.llm_brain import LLMBrain

    model = ScriptedModel(
        [{"action": {"name": "write_file", "args": {"path": "x.txt", "content": "x"}}}]
    )
    events = _collect(
        {
            "brain": LLMBrain(backend=model),
            "permission_policy": PermissionPolicy(mode="read-only"),
        },
        {"task": "write a file", "max_steps": 1, "reflect": False},
    )
    step_end = next(e for e in events if isinstance(e, StepEnd))
    assert step_end.error is not None
    assert "Permission denied" in step_end.error
    assert to_ndjson(step_end)["error"] == step_end.error


def test_hypothesis_flows_through_run_begin():
    events = _collect({}, {"task": "verify colors", "max_steps": 1, "hypothesis": "h"})
    assert events[0].hypothesis == "h"
    assert to_ndjson(events[0])["hypothesis"] == "h"


def test_token_deltas_stream_with_scripted_model():
    from swaybot.llm_brain import LLMBrain

    model = ScriptedModel(
        [
            {"text": "Thinking about it."},
            {"action": {"name": "final_answer", "args": {"answer": "42"}}},
        ]
    )
    events: list = []
    agent = Agent(brain=LLMBrain(backend=model), on_event=events.append)
    agent.run("meaning of life", max_steps=3, reflect=False, stream=True)
    deltas = [e for e in events if isinstance(e, TokenDelta)]
    assert deltas and "".join(d.text for d in deltas) == "Thinking about it."
    # The scripted tool call reached the environment as a real action.
    step_ends = [e for e in events if isinstance(e, StepEnd)]
    assert step_ends[-1].action["name"] == "final_answer"
    # No reasoning without a reasoning-capable backend.
    assert not any(isinstance(e, ReasoningDelta) for e in events)


def test_run_log_written_from_event_stream(tmp_path):
    """The run-log recorder consumes the same events and writes runs.jsonl."""
    from swaybot.run_log import load_runs, run_log_path_for_memory

    memory = MemoryStore(path=tmp_path / "memory.json")
    agent = Agent(memory=memory)
    agent.run("count", max_steps=2, reflect=False)
    runs = load_runs(run_log_path_for_memory(memory.path))
    assert len(runs) == 1
    record = runs[0]
    assert record["task"] == "count"
    assert len(record["steps"]) == 2
    assert record["steps"][0]["step"] == 1


def test_to_ndjson_skips_streaming_and_begin_events():
    assert to_ndjson(StepBegin(step=1, action={})) is None
    assert to_ndjson(TokenDelta(text="x")) is None
    assert to_ndjson(ReasoningDelta(text="y")) is None
