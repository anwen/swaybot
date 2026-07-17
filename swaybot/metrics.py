"""Metrics sink for agent/tool/model observations."""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

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
    """AgentHook that forwards observations to a MetricSink."""

    def __init__(self, sink: MetricSink) -> None:
        self.sink = sink

    def on_metric(
        self, name: str, value: float, labels: dict[str, Any] | None = None
    ) -> None:
        self.sink.record(name, value, labels)

    def after_iteration(
        self,
        task: str,
        step: int,
        action: dict,
        result: Any,
        metadata: dict | None = None,
    ) -> None:
        labels = {"task": task, "tool": action.get("name", "")}
        has_error = isinstance(result, str) and result.startswith("Error:")
        self.sink.record(
            "tool.error" if has_error else "tool.success",
            1.0,
            labels,
        )
        if metadata:
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

    def after_run(
        self, task: str, env: Any, reflections: list[str]
    ) -> None:
        self.sink.record(
            "agent.run.steps",
            float(getattr(env, "step", 0)),
            {"task": task},
        )
