import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class Memory:
    """A single record of something the agent has observed or learned."""

    content: str
    kind: str = "experience"  # fact, experience, theory, conjecture, inspiration
    scope: str = "long_term"  # short_term (raw experience) or long_term (validated knowledge)
    source: str = ""
    evidence: str = ""
    credibility: float = 0.5
    surprise: float = 0.0
    relation: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class MemoryStore:
    """Simple store for memories with optional JSON persistence."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else None
        self.memories: list[Memory] = []
        if self.path and self.path.exists():
            self.load()

    def add(self, memory: Memory) -> None:
        """Append a memory and persist if a path is configured."""
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
    ) -> list[Memory]:
        """Return memories matching the given filters."""
        results = self.memories
        if kind:
            results = [m for m in results if m.kind == kind]
        if scope:
            results = [m for m in results if m.scope == scope]
        if tag:
            results = [m for m in results if tag in m.tags]
        if min_surprise:
            results = [m for m in results if m.surprise >= min_surprise]
        return results[-limit:]

    def find_counterexamples(self, claim: str) -> list[Memory]:
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
            memory_words = set(_tokenize(m.content))
            if not (claim_words & memory_words):
                continue
            if m.surprise > 0.5:
                results.append(m)
                continue
            if m.credibility <= 0.7:
                continue
            markers = memory_words & (negation_markers | contradiction_markers)
            if markers:
                results.append(m)
        return results

    def to_dicts(self) -> list[dict]:
        return [asdict(m) for m in self.memories]

    def load(self) -> None:
        if not self.path:
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.memories = [Memory(**item) for item in data]

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.to_dicts(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _tokenize(text: str) -> list[str]:
    """Very basic tokenizer for counterexample search."""
    return [word.lower().strip(".,;:!?()[]{}\"'") for word in text.split() if word]
