import json

from swaybot.agent_events import RunBegin, RunEnd, StepEnd
from swaybot.logging import JSONFormatter
from swaybot.metrics import (
    InMemoryMetricSink,
    LoggingMetricSink,
    MetricsHook,
)


def test_in_memory_metric_sink_records():
    sink = InMemoryMetricSink()
    sink.record("tool.success", 1.0, {"tool": "echo"})
    assert len(sink.records) == 1
    assert sink.records[0]["name"] == "tool.success"
    assert sink.records[0]["labels"]["tool"] == "echo"


def _feed(hook, *events):
    for event in events:
        hook(event)


def test_metrics_hook_records_success():
    sink = InMemoryMetricSink()
    hook = MetricsHook(sink)
    _feed(
        hook,
        RunBegin(task="demo", max_steps=3),
        StepEnd(
            step=1,
            action={"name": "echo", "args": {"message": "hi"}},
            result="hi",
            metadata={},
        ),
    )
    names = [r["name"] for r in sink.records]
    assert "tool.success" in names
    success = next(r for r in sink.records if r["name"] == "tool.success")
    assert success["labels"] == {"task": "demo", "tool": "echo"}


def test_metrics_hook_records_error():
    sink = InMemoryMetricSink()
    hook = MetricsHook(sink)
    _feed(
        hook,
        RunBegin(task="demo", max_steps=3),
        StepEnd(
            step=1,
            action={"name": "echo", "args": {"message": "hi"}},
            result="Error: something went wrong",
        ),
    )
    names = [r["name"] for r in sink.records]
    assert "tool.error" in names


def test_metrics_hook_records_token_usage():
    sink = InMemoryMetricSink()
    hook = MetricsHook(sink)
    _feed(
        hook,
        RunBegin(task="demo", max_steps=3),
        StepEnd(
            step=1,
            action={"name": "echo"},
            result="ok",
            metadata={
                "duration_ms": 120.0,
                "token_usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        ),
    )
    names = [r["name"] for r in sink.records]
    assert "model.duration_ms" in names
    assert "model.tokens.prompt_tokens" in names
    assert "model.tokens.completion_tokens" in names


def test_metrics_hook_run_end_records_steps():
    sink = InMemoryMetricSink()
    hook = MetricsHook(sink)
    _feed(hook, RunBegin(task="demo", max_steps=3), RunEnd(steps=2, done=True, status="finished"))
    assert sink.records[-1]["name"] == "agent.run.steps"
    assert sink.records[-1]["value"] == 2.0


def test_logging_metric_sink_emits_structured_event():
    import io
    import logging

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JSONFormatter())
    logger = logging.getLogger("swaybot.logging.test_metrics")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.addHandler(handler)
    sink = LoggingMetricSink(logger=logger)
    sink.record("tool.success", 1.0, {"tool": "echo"})
    handler.flush()
    event = json.loads(stream.getvalue().strip())
    assert event["event"]["type"] == "metric"
    assert event["event"]["name"] == "tool.success"
    assert event["event"]["value"] == 1.0
    assert event["event"]["labels"]["tool"] == "echo"
