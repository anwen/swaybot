"""Typed event contract for the agent loop.

The loop's only outward-facing output stream: CLI rendering and run-log
recording are consumers of the same events, and future wire/UI consumers
subscribe the same way. Events are immutable snapshots; consuming them must
not affect run state. (Design borrowed from zcode-cli's loop events.)

The step index follows ``Environment.observe`` semantics: StepBegin carries
the step number the action is about to get, StepEnd the number it got.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union


@dataclass(frozen=True)
class RunBegin:
    task: str
    max_steps: int
    hypothesis: Optional[str] = None


@dataclass(frozen=True)
class StepBegin:
    """The brain chose an action; the tool is about to run."""

    step: int
    action: dict


@dataclass(frozen=True)
class StepEnd:
    """The tool ran (or was denied); the environment recorded the result."""

    step: int
    action: dict
    result: Any
    error: Optional[str] = None
    metadata: Optional[dict] = None


@dataclass(frozen=True)
class TokenDelta:
    """Streaming model output text."""

    text: str


@dataclass(frozen=True)
class ReasoningDelta:
    """Streaming model reasoning content."""

    text: str


@dataclass(frozen=True)
class RunEnd:
    steps: int
    done: bool
    # "finished" — done/final_answer; "steps_exhausted" — the step budget
    # ran out while the model was still requesting tools.
    status: Literal["finished", "steps_exhausted"]
    reflections: list[str] = field(default_factory=list)


Event = Union[RunBegin, StepBegin, StepEnd, TokenDelta, ReasoningDelta, RunEnd]


def to_ndjson(event: Event) -> Optional[dict]:
    """Map an event to the CLI NDJSON wire shape.

    Returns None for events with no NDJSON line (StepBegin and the streaming
    deltas) — callers skip them. Callers may enrich the run_start payload with
    their own context (brain name, permission mode, run type).
    """
    if isinstance(event, RunBegin):
        payload = {
            "event": "run_start",
            "task": event.task,
            "max_steps": event.max_steps,
        }
        if event.hypothesis:
            payload["hypothesis"] = event.hypothesis
        return payload
    if isinstance(event, StepEnd):
        payload = {
            "event": "step",
            "step": event.step,
            "action": event.action,
            "result": str(event.result),
        }
        if event.error:
            payload["error"] = event.error
        return payload
    if isinstance(event, RunEnd):
        return {
            "event": "run_end",
            "steps": event.steps,
            "done": event.done,
            "status": event.status,
        }
    return None
