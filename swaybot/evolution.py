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
