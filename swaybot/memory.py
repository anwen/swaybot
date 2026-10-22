import json
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, ClassVar

from .storage import (
    InMemoryBackend,
    StorageBackend,
    _resolve_file_backend,
    default_memory_backend,
)
from .utils import utc_now_iso

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
    created_at: str = field(default_factory=utc_now_iso)
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
    # Lifecycle for question/contradiction entries: open -> testing (an
    # exploration run picked it up) -> resolved (verified). Prevents the
    # self-evolution loop from re-testing settled conjectures.
    status: str = "open"
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
            self.backend, self._key = _resolve_file_backend(
                None,
                path,
                Path.home() / ".swaybot",
                "memory",
            )
        else:
            # Default to in-memory storage for backward compatibility and
            # to keep tests isolated.
            self.backend = InMemoryBackend()
            self._key = "memory"
        _path_for = getattr(self.backend, "path_for", None)
        self.path = _path_for(self._key) if callable(_path_for) else None
        self.memories: list[MemoryStep] = []
        self._batch_depth = 0
        if self.backend.exists(self._key):
            self.load()

    # Long-term kinds eligible for corroboration-merge on add(): similar
    # content bumps the existing entry's credibility instead of stacking
    # near-duplicates that crowd out query_relevant's top slots. Questions
    # and contradictions merge only on exact content (distinct-but-similar
    # wording is common and must not be collapsed).
    _MERGEABLE_KINDS = {"theory", "verification", "experiment"}

    def add(self, memory: MemoryStep) -> None:
        """Append a step and persist if a backend is configured.

        Long-term insights that restate an existing entry merge into it
        (credibility +0.1, "corroborated" tag) instead of duplicating.
        """
        duplicate = self._find_mergeable(memory)
        if duplicate is not None:
            duplicate.credibility = min(1.0, float(duplicate.credibility) + 0.1)
            if "corroborated" not in duplicate.tags:
                duplicate.tags = list(duplicate.tags) + ["corroborated"]
            self.save()
            return
        self.memories.append(memory)
        self.save()

    def _find_mergeable(self, memory: MemoryStep) -> MemoryStep | None:
        if getattr(memory, "scope", "") != "long_term":
            return None
        kind = getattr(memory, "kind", "")
        content = getattr(memory, "content", "")
        if not content or kind not in self._MERGEABLE_KINDS:
            return None
        tokens = set(_tokenize(content)) - _STOP_WORDS
        if not tokens:
            return None
        for existing in self.memories:
            if (
                getattr(existing, "scope", "") != "long_term"
                or getattr(existing, "kind", "") != kind
            ):
                continue
            existing_content = getattr(existing, "content", "")
            if existing_content == content:
                return existing
            existing_tokens = set(_tokenize(existing_content)) - _STOP_WORDS
            if not existing_tokens:
                continue
            overlap = len(tokens & existing_tokens) / len(tokens | existing_tokens)
            if overlap >= 0.75:
                return existing
        return None

    def add_steps(self, steps: Iterable[MemoryStep]) -> None:
        """Append multiple steps and persist."""
        self.memories.extend(steps)
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

    def iter_short_term(self, tag: str | None = None) -> Iterator[MemoryStep]:
        """Yield short-term steps, optionally filtered to a tag."""
        for m in self.memories:
            if m.scope != "short_term":
                continue
            if tag is not None and tag not in m.tags:
                continue
            yield m

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
        if self._batch_depth:
            return
        self.backend.save(self._key, self.to_dicts())

    @contextmanager
    def batch(self):
        """Defer saves until the outermost batch exits.

        A run persists two steps per iteration; rewriting the whole file
        each time is quadratic I/O. Inside a batch, add/add_steps/prune
        mutate in memory and a single save happens on exit — including
        on exception, so a crashed run still leaves its steps on disk.
        """
        self._batch_depth += 1
        try:
            yield
        finally:
            self._batch_depth -= 1
            if self._batch_depth == 0:
                self.save()

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


class AutoCompact:
    """Compress excess short-term steps into a summary memory.

    Two triggers, whichever fires first: more than ``keep_recent`` steps,
    or an estimated context size over ``max_context_chars`` (a long
    observation can blow the window in a few steps — step count alone is
    a poor proxy). The oldest steps beyond the recent window are
    summarized — by the brain callable when it produces one, otherwise by
    a deterministic digest — and replaced with a single long-term memory.
    """

    def __init__(
        self,
        brain: Callable[[list[dict]], str] | None = None,
        max_steps: int = 6,
        max_context_chars: int = 32_000,
    ) -> None:
        self.brain = brain
        self.max_steps = max_steps
        self.max_context_chars = max_context_chars

    def compact(self, store: MemoryStore, tag: str | None = None) -> bool:
        """Compact short-term steps if a threshold is exceeded.

        Returns ``True`` if compaction happened.
        """
        # TaskStep 是任务陈述锚点，永不参与压缩
        candidates = [s for s in store.iter_short_term(tag) if not isinstance(s, TaskStep)]
        total_chars = sum(self._step_chars(step) for step in candidates)
        if len(candidates) > self.max_steps:
            keep = self.max_steps  # 步数触发：保留最近 keep 步
        elif total_chars > self.max_context_chars and len(candidates) > 1:
            # 字符触发（步数不多但单条很大）：保留最近一半，至少留 1 步
            keep = max(1, len(candidates) // 2)
        else:
            return False

        to_compact = candidates[: -keep] if keep else candidates
        if not to_compact:
            return False

        summary = ""
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
            try:
                summary = (self.brain(messages) or "").strip()
            except Exception:
                # 默认开启后，摘要失败绝不能炸掉运行中的 agent —— 回退确定性摘要
                summary = ""
        if not summary:
            # No summarizer, or the brain produced nothing useful
            # (e.g. EchoBrain) — deterministic digest instead.
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

    @staticmethod
    def _step_chars(step: MemoryStep) -> int:
        data = step.to_dict()
        content = data.get("content") or data.get("action") or data.get("observation")
        return len(str(content))

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


def format_memory_bullets(memories: list[MemoryStep]) -> str:
    """Render a list of memory steps as bullet points of their content."""
    lines = []
    for m in memories:
        content = getattr(m, "content", "")
        if content:
            lines.append(f"- {content}")
    return "\n".join(lines)


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
