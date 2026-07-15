import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

_STEP_REGISTRY: dict[str, type["MemoryStep"]] = {}


@dataclass
class MemoryStep(ABC):
    """A single step in the agent's memory stream.

    Subclasses know how to render themselves into LLM messages via
    ``to_messages()``.  The base class provides common metadata and a
    registry so ``MemoryStore`` can serialize and deserialize steps.
    """

    kind: str = "experience"
    scope: str = "short_term"
    source: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    step_kind: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.step_kind:
            _STEP_REGISTRY[cls.step_kind] = cls

    @abstractmethod
    def to_messages(self) -> list[dict]:
        """Return OpenAI-style messages representing this step."""
        ...

    def to_dict(self) -> dict:
        data = asdict(self)
        data["step_kind"] = self.step_kind
        return data


@dataclass
class Memory(MemoryStep):
    """A long-lived record of something the agent observed or learned."""

    content: str = ""
    evidence: str = ""
    credibility: float = 0.5
    surprise: float = 0.0
    relation: str = ""
    kind: str = "experience"
    scope: str = "long_term"
    step_kind: ClassVar[str] = "memory"

    def to_messages(self) -> list[dict]:
        return [{"role": "user", "content": self.content}]


@dataclass
class TaskStep(MemoryStep):
    """The initial task given to the agent."""

    task: str = ""
    max_steps: int = 10
    source: str = "user"
    step_kind: ClassVar[str] = "task"

    def to_messages(self) -> list[dict]:
        return [
            {
                "role": "user",
                "content": f"Task: {self.task}\nYou have up to {self.max_steps} steps.",
            }
        ]


@dataclass
class ActionStep(MemoryStep):
    """A tool call chosen by the brain."""

    step: int = 0
    action: dict = field(default_factory=dict)
    source: str = "brain"
    step_kind: ClassVar[str] = "action"

    def to_messages(self) -> list[dict]:
        return [
            {
                "role": "assistant",
                "content": json.dumps(self.action, ensure_ascii=False),
            }
        ]


@dataclass
class ObservationStep(MemoryStep):
    """The result returned by the environment after a tool call."""

    step: int = 0
    observation: str = ""
    source: str = "agent.run"
    step_kind: ClassVar[str] = "observation"

    def to_messages(self) -> list[dict]:
        return [
            {
                "role": "user",
                "content": f"Observation (step {self.step}): {self.observation}",
            }
        ]


@dataclass
class PlanningStep(MemoryStep):
    """A plan produced by the brain before acting."""

    plan: list[str] = field(default_factory=list)
    step_kind: ClassVar[str] = "planning"

    def to_messages(self) -> list[dict]:
        lines = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(self.plan))
        return [{"role": "user", "content": f"Plan:\n{lines}"}]


@dataclass
class ReflectionStep(MemoryStep):
    """A higher-order insight produced by examining memories."""

    content: str = ""
    credibility: float = 0.5
    evidence: list[str] = field(default_factory=list)
    kind: str = "theory"
    scope: str = "long_term"
    source: str = "reflector"
    step_kind: ClassVar[str] = "reflection"

    def to_messages(self) -> list[dict]:
        return [{"role": "user", "content": self.content}]


class MemoryStore:
    """Simple store for memory steps with optional JSON persistence."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else None
        self.memories: list[MemoryStep] = []
        if self.path and self.path.exists():
            self.load()

    def add(self, memory: MemoryStep) -> None:
        """Append a step and persist if a path is configured."""
        self.memories.append(memory)
        if self.path:
            self.save()

    def query(
        self,
        tag: str | None = None,
        kind: str | None = None,
        scope: str | None = None,
        min_surprise: float = 0.0,
        limit: int = 10,
    ) -> list[MemoryStep]:
        """Return steps matching the given filters."""
        results = self.memories
        if kind:
            results = [m for m in results if m.kind == kind]
        if scope:
            results = [m for m in results if m.scope == scope]
        if tag:
            results = [m for m in results if tag in m.tags]
        if min_surprise:
            results = [
                m for m in results if getattr(m, "surprise", 0.0) >= min_surprise
            ]
        return results[-limit:]

    def find_counterexamples(self, claim: str) -> list[MemoryStep]:
        """Return memories that may contradict the claim.

        A memory counts as a counterexample when it shares topical keywords
        with the claim and is either explicitly marked as surprising or
        contains negation/contradiction markers. This is a deliberately
        simple heuristic; later versions may use embeddings or an LLM judge.
        """
        claim_words = set(_tokenize(claim))
        if not claim_words:
            return []

        negation_markers = {
            "not", "no", "never", "none", "nobody", "nothing", "nowhere",
            "neither", "nor", "hardly", "barely", "scarcely",
        }
        contradiction_markers = {
            "but", "however", "yet", "although", "though", "except", "unless",
            "instead", "rather", "contradict", "disprove", "refute", "deny",
            "false", "wrong",
        }

        results = []
        for m in self.memories:
            content = getattr(m, "content", "")
            memory_words = set(_tokenize(content))
            if not (claim_words & memory_words):
                continue
            if getattr(m, "surprise", 0.0) > 0.5:
                results.append(m)
                continue
            if getattr(m, "credibility", 1.0) <= 0.7:
                continue
            markers = memory_words & (negation_markers | contradiction_markers)
            if markers:
                results.append(m)
        return results

    def to_dicts(self) -> list[dict]:
        return [m.to_dict() for m in self.memories]

    def load(self) -> None:
        if not self.path:
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.memories = [_load_step(item) for item in data]

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.to_dicts(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def prune(
        self,
        scope: str | None = None,
        tag: str | None = None,
        keep_last: int = 0,
    ) -> int:
        """Remove matching steps, optionally preserving the last ``keep_last``.

        Returns the number of removed steps.
        """

        def matches(step: MemoryStep) -> bool:
            if scope and step.scope != scope:
                return False
            if tag and tag not in step.tags:
                return False
            return True

        indices = [i for i, m in enumerate(self.memories) if matches(m)]
        if keep_last > 0:
            indices = indices[:-keep_last]
        for i in reversed(indices):
            del self.memories[i]
        if self.path:
            self.save()
        return len(indices)


def _load_step(item: dict) -> MemoryStep:
    data = dict(item)
    step_kind = data.pop("step_kind", None)
    cls = _STEP_REGISTRY.get(step_kind, Memory)
    try:
        return cls(**data)
    except TypeError:
        # Legacy or corrupted entry: keep what we can as a plain Memory.
        return Memory(content=json.dumps(data, ensure_ascii=False))


def _tokenize(text: str) -> list[str]:
    """Very basic tokenizer for counterexample search."""
    return [word.lower().strip(".,;:!?()[]{}\"'") for word in text.split() if word]
