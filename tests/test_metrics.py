import json

from swaybot.environment import Environment
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


def test_metrics_hook_after_iteration_records_success():
    sink = InMemoryMetricSink()
    hook = MetricsHook(sink)
    hook.after_iteration(
        task="demo",
        step=1,
        action={"name": "echo", "args": {"message": "hi"}},
        result="hi",
        metadata={},
    )
    names = [r["name"] for r in sink.records]
    assert "tool.success" in names


def test_metrics_hook_after_iteration_records_error():
    sink = InMemoryMetricSink()
    hook = MetricsHook(sink)
    hook.after_iteration(
        task="demo",
        step=1,
        action={"name": "echo", "args": {"message": "hi"}},
        result="Error: something went wrong",
        metadata={},
    )
    names = [r["name"] for r in sink.records]
    assert "tool.error" in names


def test_metrics_hook_records_token_usage():
    sink = InMemoryMetricSink()
    hook = MetricsHook(sink)
    hook.after_iteration(
        task="demo",
        step=1,
        action={"name": "echo"},
        result="ok",
        metadata={
            "duration_ms": 120.0,
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )
    names = [r["name"] for r in sink.records]
    assert "model.duration_ms" in names
    assert "model.tokens.prompt_tokens" in names
    assert "model.tokens.completion_tokens" in names


import logging
import io

def test_logging_metric_sink_emits_structured_event():
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


def test_metrics_hook_after_run_records_steps():
    sink = InMemoryMetricSink()
    hook = MetricsHook(sink)
    env = Environment(task="demo", max_steps=3)
    env.step = 2
    hook.after_run("demo", env, [])
    assert sink.records[-1]["name"] == "agent.run.steps"
    assert sink.records[-1]["value"] == 2.0
