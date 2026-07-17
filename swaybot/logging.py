"""Structured logging helpers and an AgentHook that emits audit events."""

import json
import logging
from typing import Any

from .hook import AgentHook
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


class StructuredLoggingHook(AgentHook):
    """Emit structured events for agent lifecycle moments."""

    def before_iteration(
        self, task: str, step: int, perception: dict
    ) -> None:
        log_event(
            "agent.before_iteration",
            task=task,
            step=step,
        )

    def after_iteration(
        self,
        task: str,
        step: int,
        action: dict,
        result: Any,
        metadata: dict | None = None,
    ) -> None:
        event_type = "tool_call"
        if metadata and metadata.get("model"):
            event_type = "model_call"
        log_event(
            event_type,
            task=task,
            step=step,
            action=action,
            result=str(result),
            metadata=metadata,
        )

    def after_run(
        self, task: str, env: Any, reflections: list[str]
    ) -> None:
        log_event(
            "agent.after_run",
            task=task,
            steps=getattr(env, "step", None),
            done=getattr(env, "done", None),
            reflections=reflections,
        )

    def on_token(self, token: str) -> None:
        log_event("agent.token", token=token)

    def on_reasoning(self, reasoning: str) -> None:
        log_event("agent.reasoning", reasoning=reasoning)
