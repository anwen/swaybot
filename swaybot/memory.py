import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, ClassVar

from .storage import (
    JSONLBackend,
    StorageBackend,
    default_memory_backend,
    InMemoryBackend,
)

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
    model_input_messages: list[dict] | None = None
    raw_output: str | None = None
    token_usage: dict | None = None
    duration_ms: float | None = None

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

    def __init__(
        self,
        path: Path | str | None = None,
        backend: StorageBackend | None = None,
        key: str | None = None,
    ) -> None:
        if backend is not None:
            self.backend = backend
            self._key = key or "memory"
        elif path is not None:
            p = Path(path)
            self.backend = default_memory_backend(p)
            self._key = p.stem
        else:
            # Default to in-memory storage for backward compatibility and
            # to keep tests isolated.
            self.backend = InMemoryBackend()
            self._key = "memory"
        self.path = (
            getattr(self.backend, "_path", lambda _k: None)(self._key)
            if hasattr(self.backend, "_path")
            else None
        )
        self.memories: list[MemoryStep] = []
        if self.backend.exists(self._key):
            self.load()

    def add(self, memory: MemoryStep) -> None:
        """Append a step and persist if a backend is configured."""
        self.memories.append(memory)
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

    def query_relevant(
        self,
        query: str,
        scope: str | None = "long_term",
        limit: int = 5,
    ) -> list[MemoryStep]:
        """Return the most relevant steps for ``query`` using keyword overlap.

        Scoring is a simple Jaccard-style overlap over content words, with a
        boost when the query matches one of the step's tags. This keeps the
        core stdlib-only; embeddings can be swapped in later.
        """
        query_tokens = set(_tokenize(query)) - _STOP_WORDS
        if not query_tokens:
            return []

        scored: list[tuple[float, MemoryStep]] = []
        for m in self.memories:
            if scope and m.scope != scope:
                continue
            content = getattr(m, "content", "")
            memory_tokens = set(_tokenize(content)) - _STOP_WORDS
            if not memory_tokens:
                continue
            overlap = query_tokens & memory_tokens
            if not overlap:
                continue
            score = len(overlap) / len(query_tokens | memory_tokens)
            if query in m.tags:
                score += 1.0
            scored.append((score, m))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [m for _, m in scored[:limit]]

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
        data = self.backend.load(self._key)
        if data is None:
            return
        if isinstance(data, list):
            self.memories = [_load_step(item) for item in data]
        else:
            self.memories = [_load_step(data)]

    def save(self) -> None:
        self.backend.save(self._key, self.to_dicts())

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
        self.save()
        return len(indices)


class Consolidator:
    """Archive short-term steps to a jsonl history file.

    Optionally compresses the archived batch into a long-term summary memory
    using a brain/model callable.
    """

    def __init__(
        self,
        history_path: Path | str,
        brain: Callable[[list[MemoryStep]], str] | None = None,
    ) -> None:
        self.history_path = Path(history_path)
        self.brain = brain
        self._backend = JSONLBackend(self.history_path.parent)
        self._key = self.history_path.stem

    def archive(
        self,
        store: MemoryStore,
        tag: str | None = None,
        keep_last: int = 0,
        summarize: bool = False,
    ) -> list[MemoryStep]:
        """Move short-term steps to ``history.jsonl`` and optionally summarize.

        Returns the archived steps.
        """
        candidates = [
            m
            for m in store.memories
            if m.scope == "short_term" and (tag is None or tag in m.tags)
        ]
        if keep_last > 0:
            archived = candidates[:-keep_last]
        else:
            archived = candidates
        if not archived:
            return []

        for step in archived:
            self._backend.append(self._key, step.to_dict())

        for step in archived:
            store.memories.remove(step)
        store.save()

        if summarize and self.brain:
            summary = self.brain(archived)
            if summary:
                store.add(
                    Memory(
                        content=summary,
                        scope="long_term",
                        tags=[tag, "consolidated"] if tag else ["consolidated"],
                    )
                )

        return archived


class Dream:
    """Edit a durable memory file (e.g. SOUL.md) using long-term insights.

    With a ``brain`` callable the file can be rewritten; otherwise new
    insights are appended as a markdown list.
    """

    def __init__(
        self,
        durable_path: Path | str,
        brain: Callable[[str, list[MemoryStep]], str] | None = None,
    ) -> None:
        self.durable_path = Path(durable_path)
        self.brain = brain

    def edit(self, store: MemoryStore) -> str:
        """Read durable memory, merge in new long-term insights, and write back."""
        content = self._read()
        insights = [
            m
            for m in store.memories
            if m.scope == "long_term" and getattr(m, "kind", "") == "theory"
        ]
        if not content and not insights:
            return ""

        if self.brain:
            new_content = self.brain(content, insights)
        else:
            new_content = self._default_edit(content, insights)

        self.durable_path.parent.mkdir(parents=True, exist_ok=True)
        self.durable_path.write_text(new_content, encoding="utf-8")
        return new_content

    def _read(self) -> str:
        if not self.durable_path.exists():
            return ""
        return self.durable_path.read_text(encoding="utf-8")

    def _default_edit(self, content: str, insights: list[MemoryStep]) -> str:
        lines = []
        if content:
            lines.append(content.rstrip())
        if insights:
            lines.append("")
            lines.append("## Consolidated insights")
            lines.append("")
            for m in insights:
                text = getattr(m, "content", "")
                if text:
                    lines.append(f"- {text}")
        return "\n".join(lines)


class AutoCompact:
    """Compress excess short-term steps into a summary memory.

    When the number of short-term steps for a tag exceeds ``max_steps``,
    the oldest excess steps are summarized by a brain/model callable and
    replaced with a single long-term memory.
    """

    def __init__(
        self,
        brain: Callable[[list[dict]], str] | None = None,
        max_steps: int = 6,
    ) -> None:
        self.brain = brain
        self.max_steps = max_steps

    def compact(self, store: MemoryStore, tag: str | None = None) -> bool:
        """Compact short-term steps if they exceed the threshold.

        Returns ``True`` if compaction happened.
        """
        candidates = [
            m
            for m in store.memories
            if m.scope == "short_term" and (tag is None or tag in m.tags)
        ]
        if len(candidates) <= self.max_steps:
            return False

        to_compact = candidates[: -self.max_steps]
        if not to_compact:
            return False

        if self.brain:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Summarize the following agent steps concisely. "
                        "Preserve key decisions, surprises, and final outcomes."
                    ),
                }
            ]
            for step in to_compact:
                messages.extend(step.to_messages())
            summary = self.brain(messages)
        else:
            summary = self._fallback_summary(to_compact)

        for step in to_compact:
            store.memories.remove(step)
        if summary:
            store.add(
                Memory(
                    content=summary,
                    scope="long_term",
                    tags=[tag, "compact"] if tag else ["compact"],
                )
            )
        else:
            store.save()
        return True

    def _fallback_summary(self, steps: list[MemoryStep]) -> str:
        snippets = []
        for step in steps:
            text = getattr(step, "content", "") or str(
                getattr(step, "action", getattr(step, "observation", ""))
            )
            if text:
                snippets.append(text)
        joined = "; ".join(snippets)
        if len(joined) > 200:
            joined = joined[:197] + "..."
        return f"Compacted: {joined}"


def _load_step(item: dict) -> MemoryStep:
    data = dict(item)
    step_kind = data.pop("step_kind", None)
    cls = _STEP_REGISTRY.get(step_kind, Memory)
    try:
        return cls(**data)
    except TypeError:
        # Legacy or corrupted entry: keep what we can as a plain Memory.
        return Memory(content=json.dumps(data, ensure_ascii=False))


_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "and", "or", "for", "in", "on", "at", "by", "with", "from",
    "as", "it", "its", "this", "that", "these", "those", "i", "you", "he",
    "she", "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "his", "her", "our", "their", "what", "which", "who", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "not", "only", "own", "same", "so", "than", "too",
    "very", "can", "will", "just", "should", "now", "do", "does", "did",
}


def _tokenize(text: str) -> list[str]:
    """Very basic tokenizer for counterexample search."""
    return [word.lower().strip(".,;:!?()[]{}\"'") for word in text.split() if word]
