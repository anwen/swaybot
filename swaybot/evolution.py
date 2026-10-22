"""Self-evolution director: evidence-gated orchestration of the growth loop.

Consumes nothing from inside the agent loop and touches no loop internals —
it schedules whole runs from OUTSIDE, driven purely by memory state and
emitting its own events into the same stream the loop uses. That boundary
(an orchestrator speaking events, not hooks) is what keeps the mechanism
observable, and what would let it ship as a skill for other agents later.

Phases and their evidence gates (no timers anywhere):

- reflection: always runs — each exploration run executes with reflect=True
- exploration: fires while >= ``min_open`` open questions/contradictions
  exist; contradictions are picked before questions (a contradiction is
  closer to a knowledge boundary than an open question)
- experiment recording: every exploration run lands in memory as a
  kind="experiment" record linked back to its hypothesis (relation field)
- belief update: performed by the Reflector inside each run (verified
  hypotheses mark their question resolved — see reflection.py)

The loop converges by construction: resolved questions leave the open set,
so the gate closes when there is nothing left worth testing.
"""

import difflib
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .agent_events import Event, ExplorationBegin, ExplorationEnd, ExperimentRecorded
from .memory import Memory, MemoryStore


class SelfEvolutionDirector:
    """Schedule exploration runs from memory state, emitting events."""

    def __init__(
        self,
        agent,
        *,
        min_open: int = 2,
        max_runs: int = 3,
        max_steps: int = 5,
        emit=None,
    ) -> None:
        self.agent = agent
        self.min_open = min_open
        self.max_runs = max_runs
        self.max_steps = max_steps
        self._emit_sink = emit

    # -- public ---------------------------------------------------------

    def cycle(self) -> list[Event]:
        """Run exploration rounds while the evidence gate stays open.

        Returns the director-emitted events (the agent's own run events go
        through the shared sink as usual).
        """
        emitted: list[Event] = []
        for _ in range(self.max_runs):
            if self.agent.memory is None:
                break
            candidates = self.open_candidates()
            if len(candidates) < self.min_open:
                break
            hypothesis = candidates[0].content
            self._set_status(hypothesis, "testing")
            task = f"Verify this conjecture: {hypothesis}"
            event = ExplorationBegin(hypothesis=hypothesis, task=task)
            emitted.append(event)
            self._emit(event)
            env = self.agent.run(
                task, max_steps=self.max_steps, reflect=True, hypothesis=hypothesis
            )
            record_event = self._record_experiment(hypothesis, env.history, env.stop_reason)
            emitted.append(record_event)
            self._emit(record_event)
            end = ExplorationEnd(
                hypothesis=hypothesis,
                status=env.stop_reason or "unknown",
            )
            emitted.append(end)
            self._emit(end)
        return emitted

    def open_candidates(self) -> list:
        """Open questions/contradictions, contradictions first."""
        memory = self.agent.memory
        if memory is None:
            return []
        questions = [
            m
            for m in memory.query(kind="question", scope="long_term", limit=50)
            if getattr(m, "status", "open") == "open"
        ]
        contradictions = [
            m
            for m in memory.query(kind="contradiction", scope="long_term", limit=50)
            if getattr(m, "status", "open") == "open"
        ]
        return contradictions + questions

    # -- internals ------------------------------------------------------

    def _record_experiment(
        self, hypothesis: str, history: list[dict], stop_reason: str | None
    ) -> ExperimentRecorded:
        final = ""
        for entry in reversed(history):
            action = entry.get("action") or {}
            if action.get("name") == "final_answer":
                final = str((action.get("args") or {}).get("answer", ""))
                break
        summary = (
            f"outcome={stop_reason}, steps={len(history)}, "
            f"final_answer={final[:200] or 'none'}"
        )
        memory = self.agent.memory
        if memory is not None:
            memory.add(
                Memory(
                    content=f"Experiment on '{hypothesis}': {summary}",
                    kind="experiment",
                    scope="long_term",
                    relation=hypothesis,
                    source="director",
                    credibility=0.5,
                    tags=["experiment"],
                )
            )
        return ExperimentRecorded(hypothesis=hypothesis, summary=summary)

    def _set_status(self, hypothesis: str, status: str) -> None:
        memory = self.agent.memory
        if memory is None:
            return
        for mem in memory.memories:
            if (
                getattr(mem, "step_kind", "") == "reflection"
                and getattr(mem, "kind", "") in ("question", "contradiction")
                and getattr(mem, "content", "") == hypothesis
            ):
                mem.status = status
        memory.save()

    def _emit(self, event: Event) -> None:
        if self._emit_sink is not None:
            self._emit_sink(event)


class SoulChannel:
    """The slow, cautious lane: digests of stable knowledge.

    Four brakes make "slow" structural rather than prompted:
    1. evidence  — at least ``min_theories`` theories at credibility
       >= ``min_credibility`` (a stable set, never a single result);
    2. age       — each included theory is at least ``min_age_days`` old;
    3. quiet     — no contradiction newer than ``quiet_days`` (active
       conflicts anywhere in memory pause the whole channel);
    4. approval  — propose() only ever produces a DIFF; writing happens
       in apply() and only with an explicit approve=True (CLI
       --allow-soul-edit), followed by a best-effort git commit so every
       change to the digest is versioned and revertable.

    The digest lands at ``<workspace>/.swaybot/soul-digest.md``, which
    instructions.py loads into future system prompts — the loop from
    verified experience back into disposition.
    """

    def __init__(
        self,
        memory: MemoryStore,
        workspace,
        *,
        min_theories: int = 5,
        min_credibility: float = 0.8,
        min_age_days: float = 7.0,
        quiet_days: float = 1.0,
        digest_path: str = ".swaybot/soul-digest.md",
    ) -> None:
        self.memory = memory
        self.workspace = workspace
        self.min_theories = min_theories
        self.min_credibility = min_credibility
        self.min_age_days = min_age_days
        self.quiet_days = quiet_days
        self.digest_path = digest_path

    def stable_theories(self) -> list:
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.min_age_days)
        theories = [
            m
            for m in self.memory.query(kind="theory", scope="long_term", limit=200)
            if float(getattr(m, "credibility", 0.0)) >= self.min_credibility
            and _parse_iso(getattr(m, "created_at", "")) is not None
            and _parse_iso(getattr(m, "created_at", "")) <= cutoff
        ]
        return theories

    def propose(self):
        """Return (digest_text, unified_diff) when all gates open, else None."""
        theories = self.stable_theories()
        if len(theories) < self.min_theories:
            return None
        quiet_cutoff = datetime.now(timezone.utc) - timedelta(days=self.quiet_days)
        for m in self.memory.query(kind="contradiction", scope="long_term", limit=200):
            created = _parse_iso(getattr(m, "created_at", ""))
            if created is not None and created > quiet_cutoff:
                return None  # active conflict: the channel pauses
        digest = self._render(theories)
        target = Path(self.workspace) / self.digest_path
        existing = target.read_text(encoding="utf-8") if target.exists() else ""
        if digest == existing:
            return None  # nothing new to say
        diff = "\n".join(
            difflib.unified_diff(
                existing.splitlines(),
                digest.splitlines(),
                fromfile=str(target),
                tofile=str(target),
                lineterm="",
            )
        )
        return digest, diff

    def apply(self, digest: str, *, approve: bool) -> str:
        """Write the digest when approved; returns what happened."""
        target = Path(self.workspace) / self.digest_path
        if not approve:
            return "not approved; diff only"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(digest, encoding="utf-8")
        committed = self._git_commit(target)
        return "written" + (" + committed" if committed else " (no git repo; unversioned)")

    def _render(self, theories: list) -> str:
        lines = [
            "# Soul digest",
            "",
            "> Auto-proposed digest of stable, high-credibility theories.",
            "> Generated by the soul channel; every change is gated and diffed.",
            "",
        ]
        for m in sorted(
            theories, key=lambda m: float(getattr(m, "credibility", 0.0)), reverse=True
        ):
            cred = float(getattr(m, "credibility", 0.0))
            lines.append(f"- (credibility {cred:.2f}) {getattr(m, 'content', '')}")
        return "\n".join(lines) + "\n"

    def _git_commit(self, target: Path) -> bool:
        import subprocess

        try:
            subprocess.run(
                ["git", "add", str(target)],
                cwd=self.workspace,
                capture_output=True,
                timeout=10,
                check=False,
            )
            result = subprocess.run(
                [
                    "git",
                    "commit",
                    "-m",
                    "soul channel: update stable-knowledge digest",
                    "--no-verify",
                ],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return result.returncode == 0
        except Exception:
            return False


def _parse_iso(value: str):
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
