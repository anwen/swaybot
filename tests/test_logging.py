import io
import json
import logging

import pytest

from swaybot.agent_events import RunBegin, RunEnd, StepEnd
from swaybot.logging import JSONFormatter, StructuredLoggingHook, log_event
from swaybot.request_context import RequestContext, set_context


@pytest.fixture
def capture_event_logs():
    """Capture JSON-formatted event log output in a StringIO stream."""
    logger = logging.getLogger("swaybot.events")
    original_handlers = list(logger.handlers)
    original_propagate = logger.propagate
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    yield stream
    logger.handlers = original_handlers
    logger.propagate = original_propagate


def _last_event(stream: io.StringIO) -> dict:
    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    assert lines, "no log lines captured"
    return json.loads(lines[-1])


def test_log_event_includes_context(capture_event_logs):
    ctx = RequestContext(
        principal="alice", request_id="req-1", session_id="s-1"
    )
    with set_context(ctx):
        log_event("tool_call", tool="echo", result="hi")

    payload = _last_event(capture_event_logs)
    assert payload["event"]["type"] == "tool_call"
    assert payload["event"]["principal"] == "alice"
    assert payload["event"]["request_id"] == "req-1"
    assert payload["event"]["tool"] == "echo"


def test_structured_logging_hook_step_end(capture_event_logs):
    hook = StructuredLoggingHook()
    hook(RunBegin(task="demo", max_steps=3))
    hook(
        StepEnd(
            step=1,
            action={"name": "echo", "args": {"message": "hi"}},
            result="hi",
        )
    )
    payload = _last_event(capture_event_logs)
    assert payload["event"]["type"] == "tool_call"
    assert payload["event"]["action"]["name"] == "echo"
    assert payload["event"]["task"] == "demo"


def test_structured_logging_hook_run_end(capture_event_logs):
    hook = StructuredLoggingHook()
    hook(RunBegin(task="demo", max_steps=2))
    hook(RunEnd(steps=1, done=True, reflections=["reflection"]))
    payload = _last_event(capture_event_logs)
    assert payload["event"]["type"] == "agent.run_end"
    assert payload["event"]["reflections"] == ["reflection"]
