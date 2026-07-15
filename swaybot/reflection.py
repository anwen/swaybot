from dataclasses import dataclass, field
from datetime import datetime, timezone

from .memory import MemoryStore, ReflectionStep


@dataclass
class Reflection:
    """A higher-order insight produced by examining memories."""

    content: str
    kind: str = "summary"  # summary, belief_update, contradiction, question, verification
    confidence: float = 0.5
    evidence: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class Reflector:
    """Turn raw experience memories into structured reflections."""

    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory

    def reflect_on_run(
        self,
        task: str,
        history: list[dict],
        hypothesis: str | None = None,
    ) -> list[Reflection]:
        """Generate reflections after a completed run."""
        reflections: list[Reflection] = []

        reflections.append(
            Reflection(
                content=f"Run '{task}' completed in {len(history)} steps.",
                kind="summary",
                confidence=1.0,
                tags=[task],
            )
        )

        if hypothesis:
            evidence = [
                f"{entry['action']} -> {entry['result']}" for entry in history
            ]
            verdict = self._verify_hypothesis(history)
            reflections.append(
                Reflection(
                    content=(
                        f"Hypothesis for '{task}': {hypothesis} "
                        f"-> verdict: {verdict}."
                    ),
                    kind="verification",
                    confidence=0.7,
                    evidence=evidence,
                    tags=[task],
                )
            )

        surprising = self.memory.query(tag=task, min_surprise=0.5, limit=5)
        if surprising:
            reflections.append(
                Reflection(
                    content=(
                        f"Encountered {len(surprising)} surprising event(s) "
                        f"during '{task}' that deserve deeper examination."
                    ),
                    kind="question",
                    evidence=[
                        getattr(m, "content", str(m)) for m in surprising
                    ],
                    tags=[task],
                )
            )

        recent = self.memory.query(tag=task, limit=20)
        seen: set[str] = set()
        for mem in recent:
            content = getattr(mem, "content", "")
            if not content or content in seen:
                continue
            seen.add(content)
            counters = self.memory.find_counterexamples(content)
            if counters:
                reflections.append(
                    Reflection(
                        content=f"Possible contradiction to: {content}",
                        kind="contradiction",
                        confidence=0.5,
                        evidence=[getattr(c, "content", str(c)) for c in counters],
                        tags=[task],
                    )
                )

        return reflections

    def _verify_hypothesis(self, history: list[dict]) -> str:
        """Simple heuristic: did any step report an error or fallback?"""
        for entry in history:
            result = str(entry.get("result", "")).lower()
            action = entry.get("action", {})
            if action.get("name") == "echo" and "failed" in result:
                return "refuted"
            if "error" in result or "exception" in result:
                return "refuted"
        return "supported"

    def verify_claim(self, claim: str, tag: str | None = None) -> Reflection:
        """Check a claim against stored memories and return a verdict."""
        counters = self.memory.find_counterexamples(claim)
        if counters:
            return Reflection(
                content=f"Claim '{claim}' has possible counterexamples.",
                kind="verification",
                confidence=0.3,
                evidence=[getattr(c, "content", str(c)) for c in counters],
                tags=[tag] if tag else [],
            )

        support = self.memory.query(tag=tag, kind="fact", limit=5) if tag else []
        if support:
            return Reflection(
                content=f"Claim '{claim}' is supported by existing facts.",
                kind="verification",
                confidence=0.8,
                evidence=[getattr(s, "content", str(s)) for s in support],
                tags=[tag] if tag else [],
            )

        return Reflection(
            content=f"Claim '{claim}' has insufficient evidence in memory.",
            kind="verification",
            confidence=0.5,
            tags=[tag] if tag else [],
        )


def reflection_to_memory(reflection: Reflection) -> ReflectionStep:
    """Convert a reflection into a memory suitable for long-term storage."""
    return ReflectionStep(
        content=reflection.content,
        kind="theory",
        scope="long_term",
        source="reflector",
        evidence=reflection.evidence,
        credibility=reflection.confidence,
        tags=reflection.tags,
    )
