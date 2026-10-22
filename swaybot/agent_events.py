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


@dataclass(frozen=True)
class ExplorationBegin:
    """The self-evolution director picked a conjecture to test."""

    hypothesis: str
    task: str


@dataclass(frozen=True)
class ExplorationEnd:
    hypothesis: str
    status: str


@dataclass(frozen=True)
class ExperimentDesigned:
    """A falsifiable protocol was designed for an open question."""

    hypothesis: str
    method: str
    success_criteria: str
    conjecture: str = ""


@dataclass(frozen=True)
class SubQuestionsAdded:
    """An inconclusive question was decomposed into sharper sub-questions."""

    parent: str
    questions: tuple


@dataclass(frozen=True)
class ExperimentRecorded:
    """An exploration run was written into memory as an experiment record."""

    hypothesis: str
    summary: str


@dataclass(frozen=True)
class Wondered:
    """New open questions were generated from memory state."""

    questions: tuple


@dataclass(frozen=True)
class DreamBegin:
    """Reflection-only phase: no new experiments, consolidate memory."""


@dataclass(frozen=True)
class DreamEnd:
    summary: str


@dataclass(frozen=True)
class SoulProposed:
    """The slow lane proposes a digest of stable knowledge (diff only —
    writing requires the explicit approval gate)."""

    excerpt: str


Event = Union[
    RunBegin,
    StepBegin,
    StepEnd,
    TokenDelta,
    ReasoningDelta,
    RunEnd,
    ExplorationBegin,
    ExplorationEnd,
    ExperimentDesigned,
    ExperimentRecorded,
    SubQuestionsAdded,
    Wondered,
    DreamBegin,
    DreamEnd,
    SoulProposed,
]


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
    if isinstance(event, ExplorationBegin):
        return {
            "event": "exploration_begin",
            "hypothesis": event.hypothesis,
            "task": event.task,
        }
    if isinstance(event, ExplorationEnd):
        return {
            "event": "exploration_end",
            "hypothesis": event.hypothesis,
            "status": event.status,
        }
    if isinstance(event, ExperimentDesigned):
        payload = {
            "event": "experiment_designed",
            "hypothesis": event.hypothesis,
            "method": event.method[:200],
            "success_criteria": event.success_criteria[:200],
        }
        if event.conjecture:
            payload["conjecture"] = event.conjecture[:300]
        return payload
    if isinstance(event, SubQuestionsAdded):
        return {
            "event": "sub_questions_added",
            "parent": event.parent,
            "questions": list(event.questions),
        }
    if isinstance(event, ExperimentRecorded):
        return {
            "event": "experiment_recorded",
            "hypothesis": event.hypothesis,
            "summary": event.summary[:200],
        }
    if isinstance(event, Wondered):
        return {"event": "wondered", "questions": list(event.questions)}
    if isinstance(event, DreamBegin):
        return {"event": "dream_begin"}
    if isinstance(event, DreamEnd):
        return {"event": "dream_end", "summary": event.summary[:300]}
    if isinstance(event, SoulProposed):
        return {"event": "soul_proposed", "excerpt": event.excerpt[:300]}
    return None
