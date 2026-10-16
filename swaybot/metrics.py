"""Metrics sink for agent/tool/model observations."""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .agent_events import RunBegin, RunEnd, StepEnd
from .logging import log_event


@runtime_checkable
class MetricSink(Protocol):
    """Destination for metric observations."""

    def record(
        self,
        name: str,
        value: float,
        labels: dict[str, Any] | None = None,
    ) -> None:
        ...


@dataclass
class InMemoryMetricSink:
    """Keeps metric records in memory for tests and introspection."""

    records: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        name: str,
        value: float,
        labels: dict[str, Any] | None = None,
    ) -> None:
        self.records.append(
            {
                "name": name,
                "value": value,
                "labels": labels or {},
            }
        )


class LoggingMetricSink:
    """Emit metrics as structured log events."""

    def __init__(self, logger: Any | None = None) -> None:
        self._logger = logger

    def record(
        self,
        name: str,
        value: float,
        labels: dict[str, Any] | None = None,
    ) -> None:
        if self._logger is not None:
            self._logger.info(
                "metric",
                extra={"event_payload": {"type": "metric", "name": name, "value": value, "labels": labels or {}}},
            )
        else:
            log_event("metric", name=name, value=value, labels=labels or {})


class MetricsHook:
    """Agent-event consumer that forwards observations to a MetricSink.

    Feed it to ``Agent(on_event=...)``, optionally composed with other
    sinks; the agent itself attaches nothing by default.
    """

    def __init__(self, sink: MetricSink) -> None:
        self.sink = sink
        self._task = ""

    def __call__(self, event: object) -> None:
        if isinstance(event, RunBegin):
            self._task = event.task
        elif isinstance(event, StepEnd):
            self._record_step(event)
        elif isinstance(event, RunEnd):
            self.sink.record(
                "agent.run.steps",
                float(event.steps),
                {"task": self._task},
            )

    def _record_step(self, event: StepEnd) -> None:
        labels = {"task": self._task, "tool": event.action.get("name", "")}
        has_error = bool(event.error) or (
            isinstance(event.result, str) and event.result.startswith("Error:")
        )
        self.sink.record(
            "tool.error" if has_error else "tool.success",
            1.0,
            labels,
        )
        metadata = event.metadata or {}
        if metadata.get("duration_ms") is not None:
            self.sink.record(
                "model.duration_ms",
                float(metadata["duration_ms"]),
                labels,
            )
        token_usage = metadata.get("token_usage")
        if isinstance(token_usage, dict):
            for key, val in token_usage.items():
                if isinstance(val, (int, float)):
                    self.sink.record(
                        f"model.tokens.{key}",
                        float(val),
                        labels,
                    )
