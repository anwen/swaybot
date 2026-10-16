"""Structured logging helpers and an agent-event consumer for audit logs."""

import json
import logging
from typing import Any

from .agent_events import RunBegin, RunEnd, StepEnd
from .request_context import current_context


class JSONFormatter(logging.Formatter):
    """Format log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        event_payload = getattr(record, "event_payload", None)
        if isinstance(event_payload, dict):
            payload["event"] = event_payload
        return json.dumps(payload, ensure_ascii=False, default=str)


def get_event_logger(name: str = "swaybot.events") -> logging.Logger:
    """Return a logger suitable for emitting structured events."""
    return logging.getLogger(name)


def log_event(event_type: str, **kwargs: Any) -> None:
    """Emit a structured event including the current request context."""
    ctx = current_context()
    payload = {
        "type": event_type,
        "principal": ctx.principal,
        "session_id": ctx.session_id,
        "request_id": ctx.request_id,
        "permission_level": ctx.permission_level,
    }
    payload.update(kwargs)
    logger = get_event_logger()
    logger.info(
        event_type,
        extra={"event_payload": payload},
    )


class StructuredLoggingHook:
    """Agent-event consumer that emits structured audit logs.

    Feed it to ``Agent(on_event=...)`` alongside other sinks. Streaming
    deltas (TokenDelta/ReasoningDelta) are intentionally not logged —
    per-token audit lines are noise; step and run boundaries suffice.
    """

    def __init__(self) -> None:
        self._task = ""

    def __call__(self, event: object) -> None:
        if isinstance(event, RunBegin):
            self._task = event.task
            log_event("agent.run_begin", task=event.task, max_steps=event.max_steps)
        elif isinstance(event, StepEnd):
            log_event(
                "tool_call",
                task=self._task,
                step=event.step,
                action=event.action,
                result=str(event.result),
                error=event.error,
            )
        elif isinstance(event, RunEnd):
            log_event(
                "agent.run_end",
                task=self._task,
                steps=event.steps,
                done=event.done,
                reflections=list(event.reflections),
            )
