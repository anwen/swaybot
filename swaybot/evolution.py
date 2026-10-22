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

from .agent_events import (
    DreamBegin,
    ExperimentDesigned,
    SubQuestionsAdded,
    DreamEnd,
    Event,
    ExplorationBegin,
    ExplorationEnd,
    ExperimentRecorded,
    Wondered,
)
from .memory import Memory, MemoryStore, ReflectionStep


def _parse_json_object(text: str) -> dict | None:
    """Leniently extract a JSON object from a model answer."""
    import json as _json

    from .parsing import strip_markdown_fences

    cleaned = strip_markdown_fences(text.strip())
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = _json.loads(cleaned[start : end + 1])
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _ask_brain(agent, prompt: str) -> str:
    """Ask a brain via the final_answer channel ('' when it cannot answer)."""
    perception = {
        "task": prompt[:200],
        "step": 0,
        # max_steps=2: 见 LifeRunner._ask_brain 注释——1 会把散文兜底变 done
        "max_steps": 2,
        "messages": [{"role": "user", "content": prompt}],
        "tool_descriptions": [],
    }
    action = agent.brain.think(perception, ["final_answer"])
    if action.get("name") == "final_answer":
        return str(action.get("args", {}).get("answer", ""))
    return ""


class SelfEvolutionDirector:
    """Schedule exploration runs from memory state, emitting events."""

    def __init__(
        self,
        agent,
        *,
        min_open: int = 2,
        max_runs: int = 3,
        max_steps: int = 5,
        max_depth: int = 2,
        emit=None,
    ) -> None:
        self.agent = agent
        self.min_open = min_open
        self.max_runs = max_runs
        self.max_steps = max_steps
        self.max_depth = max_depth
        self._emit_sink = emit

    # -- public ---------------------------------------------------------

    def cycle(self) -> list[Event]:
        """Run exploration rounds while the evidence gate stays open.

        Each round is a full experiment protocol: design (falsifiable
        conjecture + method + success criteria, informed by prior
        experiments on the question) -> execute -> adjudicate against the
        criteria -> record. Returns the director-emitted events.
        """
        emitted: list[Event] = []
        for _ in range(self.max_runs):
            if self.agent.memory is None:
                break
            candidates = self.open_candidates()
            if len(candidates) < self.min_open:
                break
            entry = candidates[0]
            hypothesis = entry.content
            protocol = self._design_experiment(hypothesis)
            self._record_conjecture(hypothesis, protocol["conjecture"])
            self._set_status(hypothesis, "testing")
            begin = ExplorationBegin(hypothesis=hypothesis, task=protocol["task"])
            designed = ExperimentDesigned(
                hypothesis=hypothesis,
                method=protocol["method"],
                success_criteria=protocol["criteria"],
                conjecture=protocol["conjecture"],
            )
            emitted.extend((begin, designed))
            self._emit(begin)
            self._emit(designed)
            env = self.agent.run(
                protocol["task"],
                max_steps=self.max_steps,
                reflect=True,
                hypothesis=hypothesis,
            )
            verdict = self._adjudicate(protocol, env.history)
            self._update_conjecture(hypothesis, verdict.get("verdict", ""))
            record_event = self._record_experiment(
                hypothesis, env.history, env.stop_reason, protocol=protocol, verdict=verdict
            )
            emitted.append(record_event)
            self._emit(record_event)
            if verdict.get("verdict") == "inconclusive":
                sub_event = self._decompose(entry)
                if sub_event is not None:
                    emitted.append(sub_event)
                    self._emit(sub_event)
            end = ExplorationEnd(
                hypothesis=hypothesis,
                status=f"{env.stop_reason or 'unknown'}/{verdict.get('verdict', 'unknown')}",
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
        # 浅层优先：先把父问题域的近层解决掉，再深入
        pool = contradictions + questions
        return sorted(pool, key=lambda m: int(getattr(m, "depth", 0) or 0))

    # -- internals ------------------------------------------------------

    def _design_experiment(self, hypothesis: str) -> dict:
        """Turn an open question into a falsifiable protocol.

        The designer sees prior experiments on the same question so
        successive rounds go deeper instead of repeating. Falls back to a
        generic protocol when the brain yields nothing parseable.
        """
        prior = ""
        memory = self.agent.memory
        if memory is not None:
            prior_records = [
                m.content[:150]
                for m in memory.query(kind="experiment", scope="long_term", limit=10)
                if getattr(m, "relation", "") == hypothesis
            ]
            if prior_records:
                prior = "\nPrior experiments on this question:\n" + "\n".join(
                    f"- {r}" for r in prior_records
                )
        tools = sorted(getattr(self.agent, "tools", None).names()) if getattr(
            self.agent, "tools", None
        ) else ["final_answer"]
        prompt = (
            "Design ONE experiment to make progress on this open question:\n"
            f"  {hypothesis}\n{prior}\n"
            "Constraints: the method MUST be executable in a single run inside a "
            "sandboxed workspace with EXACTLY these tools:\n"
            f"  {', '.join(tools)}\n"
            "(e.g. write and run small Python programs with run_shell_command, "
            "probe the environment with env_info, keep files in the workspace). "
            "Do NOT design experiments needing other minds, hardware, or long "
            "waiting. If the question is too broad, sharpen it into a concrete "
            "falsifiable conjecture this sandbox can actually test.\n"
            "Reply ONLY with a JSON object:\n"
            '{"conjecture": "<falsifiable statement>", "method": "<what to do, '
            'concretely, with the available tools>", "success_criteria": '
            '"<what observation would count as support, and what as refutation>"}'
        )
        # thinking 模型偶发空返回/非 JSON——重试一次再落兜底协议
        parsed = _parse_json_object(_ask_brain(self.agent, prompt))
        if not (
            parsed
            and parsed.get("conjecture")
            and parsed.get("method")
            and parsed.get("success_criteria")
        ):
            parsed = _parse_json_object(_ask_brain(self.agent, prompt))
        if (
            parsed
            and parsed.get("conjecture")
            and parsed.get("method")
            and parsed.get("success_criteria")
        ):
            conjecture = str(parsed["conjecture"])
            method = str(parsed["method"])
            criteria = str(parsed["success_criteria"])
        else:
            conjecture = hypothesis
            method = "Investigate directly with the available tools."
            criteria = "The run ends with a concrete final_answer addressing the question."
        task = (
            "Execute this experiment.\n"
            f"Conjecture: {conjecture}\n"
            f"Method: {method}\n"
            f"Success criteria: {criteria}\n"
            "Follow the method with the available tools, check the outcome "
            "against the criteria, then finish with final_answer stating "
            "supported / refuted / inconclusive plus the evidence you saw."
        )
        return {"conjecture": conjecture, "method": method, "criteria": criteria, "task": task}

    def _record_conjecture(self, question: str, conjecture: str) -> None:
        """Store the designed conjecture as a first-class belief (kind=conjecture,
        linked to its question) so verdicts can move ITS credibility precisely
        instead of by keyword relevance."""
        memory = self.agent.memory
        if memory is None or not conjecture or conjecture == question:
            return
        memory.add(
            Memory(
                content=conjecture,
                kind="conjecture",
                scope="long_term",
                relation=question,
                source="director",
                credibility=0.3,
                tags=["conjecture"],
            )
        )

    def _update_conjecture(self, question: str, verdict: str) -> None:
        """Evidence moves the conjecture's credibility via the relation link."""
        memory = self.agent.memory
        if memory is None:
            return
        delta = {"supported": 0.2, "refuted": -0.3}.get(verdict, 0.0)
        if delta == 0.0:
            return
        for mem in memory.memories:
            if (
                getattr(mem, "kind", "") == "conjecture"
                and getattr(mem, "relation", "") == question
                and hasattr(mem, "credibility")
            ):
                mem.credibility = max(0.0, min(1.0, float(mem.credibility) + delta))
        memory.save()

    def _decompose(self, entry) -> SubQuestionsAdded | None:
        """Split an inconclusive question into sharper sub-questions.

        Sub-questions link back to the parent (relation) and carry depth+1;
        a depth cap keeps the tree from running away. The parent moves to
        status 'decomposed' so it is not re-picked while its children are
        explored.
        """
        parent = entry.content
        depth = int(getattr(entry, "depth", 0) or 0)
        if depth >= self.max_depth:
            return None
        answer = _ask_brain(
            self.agent,
            "This question could not be settled by one experiment:\n"
            f"  {parent}\n"
            "Split it into 2-3 SHARPER sub-questions, each answerable by a "
            "small sandbox experiment with file/shell/python tools. Reply "
            'ONLY with JSON: {"sub_questions": ["...", "..."]}',
        )
        parsed = _parse_json_object(answer)
        subs = [
            str(q).strip()
            for q in (parsed.get("sub_questions") or [])
            if str(q).strip() and len(str(q).strip()) > 4
        ][:3] if parsed else []
        if not subs:
            return None
        memory = self.agent.memory
        if memory is not None:
            for question in subs:
                memory.add(
                    ReflectionStep(
                        content=question,
                        kind="question",
                        scope="long_term",
                        status="open",
                        relation=parent,
                        depth=depth + 1,
                    )
                )
            self._set_status(parent, "decomposed")
        return SubQuestionsAdded(parent=parent, questions=tuple(subs))

    def _adjudicate(self, protocol: dict, history: list[dict]) -> dict:
        """Judge the run against the protocol's success criteria (LLM, with
        deterministic fallback)."""
        trace = "\n".join(
            f"- {entry.get('action')} -> {str(entry.get('result'))[:120]}"
            for entry in history
        )
        answer = _ask_brain(
            self.agent,
            "Adjudicate this experiment run against its success criteria.\n"
            f"Criteria: {protocol['criteria']}\n"
            f"Run trace:\n{trace}\n"
            'Reply ONLY with JSON: {"verdict": "supported|refuted|inconclusive", '
            '"evidence": "<one sentence citing the trace>"}'
        )
        parsed = _parse_json_object(answer)
        if parsed and parsed.get("verdict") in ("supported", "refuted", "inconclusive"):
            return {"verdict": parsed["verdict"], "evidence": str(parsed.get("evidence", ""))[:300]}
        final = ""
        for entry in reversed(history):
            action = entry.get("action") or {}
            if action.get("name") == "final_answer":
                final = str((action.get("args") or {}).get("answer", "")).lower()
                break
        for verdict in ("refuted", "inconclusive", "supported"):
            if verdict in final:
                return {"verdict": verdict, "evidence": "from final_answer text"}
        return {"verdict": "inconclusive", "evidence": ""}

    def _record_experiment(
        self,
        hypothesis: str,
        history: list[dict],
        stop_reason: str | None,
        *,
        protocol: dict | None = None,
        verdict: dict | None = None,
    ) -> ExperimentRecorded:
        protocol = protocol or {}
        verdict = verdict or {}
        summary = (
            f"outcome={stop_reason}, verdict={verdict.get('verdict', 'n/a')}, "
            f"steps={len(history)}"
        )
        if protocol.get("method"):
            summary += f", method={protocol['method'][:120]}"
        if verdict.get("evidence"):
            summary += f", evidence={verdict['evidence'][:160]}"
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


class LifeRunner:
    """Continuous self-directed operation until externally stopped.

    The loop a person would recognize: explore what you are uncertain
    about (director rounds), wonder — when the open set runs dry, look at
    your own memory and let new questions form — and periodically dream:
    run NO new experiments, only reflect over recent ones into durable
    theories (duplicates corroborate instead of stacking) and propose a
    soul digest as a diff. Stops only by external force: the
    ``stop_check`` callback or KeyboardInterrupt/SIGTERM; memory persists
    at every run boundary, so an interrupted life loses nothing.
    """

    def __init__(
        self,
        agent,
        seed: str,
        *,
        emit=None,
        workspace=None,
        min_open: int = 1,
        max_rounds_per_cycle: int = 2,
        rounds_between_dreams: int = 3,
        stop_check=None,
    ) -> None:
        self.agent = agent
        self.seed = seed
        self._emit_sink = emit
        self.workspace = workspace
        self.min_open = min_open
        self.max_rounds_per_cycle = max_rounds_per_cycle
        self.rounds_between_dreams = rounds_between_dreams
        self.stop_check = stop_check

    def live(self) -> dict:
        """Run until external stop; returns a small life summary."""
        self._seed_questions()
        rounds = 0
        wonders = 0
        dreams = 0
        dry_streak = 0
        try:
            while True:
                if self.stop_check is not None and self.stop_check():
                    break
                director = SelfEvolutionDirector(
                    self.agent,
                    min_open=self.min_open,
                    max_runs=self.max_rounds_per_cycle,
                    emit=self._emit_sink,
                )
                explored = director.cycle()
                rounds += len(explored) // 3  # 每轮 3 个 director 事件
                if not director.open_candidates():
                    added = self._wonder()
                    wonders += len(added)
                    dry_streak = dry_streak + 1 if not added else 0
                    if dry_streak >= 3:
                        # 无法再产生好奇心（如确定性大脑）——生命自然止息
                        break
                else:
                    dry_streak = 0
                rounds_total = rounds
                if rounds_total and rounds_total % self.rounds_between_dreams == 0:
                    self._dream()
                    dreams += 1
        except KeyboardInterrupt:
            pass  # 外力停止：记忆已在每个 run 边界持久化
        return {"rounds": rounds, "wonders": wonders, "dreams": dreams}

    # -- phases ---------------------------------------------------------

    def _seed_questions(self) -> None:
        """Split the seed prompt into initial open questions."""
        if self.agent.memory is None:
            return
        import re

        parts = [p.strip() for p in re.split(r"[？！?;；\n]", self.seed) if p.strip()]
        for content in parts:
            self.agent.memory.add(
                ReflectionStep(
                    content=content,
                    kind="question",
                    scope="long_term",
                    status="open",
                )
            )

    def _wonder(self) -> list[str]:
        """Generate new open questions from memory state (curiosity)."""
        if self.agent.memory is None:
            return []
        memory_digest = "\n".join(
            getattr(m, "content", "")[:120]
            for m in self.agent.memory.query(scope="long_term", limit=15)
        )
        answer = self._ask_brain(
            "You just finished what you could verify. Here is your memory:\n"
            f"{memory_digest}\n\n"
            "What are you now MOST uncertain about? Reply with up to 3 "
            "questions, one per line, no numbering."
        )
        questions = [
            line.strip(" -•\t")
            for line in answer.splitlines()
            if line.strip() and len(line.strip()) > 4
        ][:3]
        for content in questions:
            self.agent.memory.add(
                ReflectionStep(
                    content=content, kind="question", scope="long_term", status="open"
                )
            )
        if questions:
            self._emit(Wondered(questions=tuple(questions)))
        return questions

    def _dream(self) -> None:
        """Reflection-only consolidation; no new experiments."""
        self._emit(DreamBegin())
        summary = "nothing to consolidate"
        memory = self.agent.memory
        if memory is not None:
            experiments = memory.query(kind="experiment", scope="long_term", limit=10)
            digest = "\n".join(
                f"- {getattr(m, 'content', '')[:150]}" for m in experiments
            )
            answer = self._ask_brain(
                "Dream phase: no new experiments. Reflect ONLY over these "
                f"experiment records:\n{digest or '(none yet)'}\n\n"
                "State up to 3 durable theories you now hold (one per line, "
                "each starting with '- '), then at most 1 new open question "
                "on a line starting with '? '."
            )
            theories = 0
            new_questions = 0
            for line in answer.splitlines():
                stripped = line.strip()
                if stripped.startswith("- ") and len(stripped) > 6:
                    memory.add(
                        ReflectionStep(
                            content=stripped[2:].strip(),
                            kind="theory",
                            scope="long_term",
                            credibility=0.6,
                        )
                    )
                    theories += 1
                elif stripped.startswith("?") and len(stripped) > 6:
                    memory.add(
                        ReflectionStep(
                            content=stripped[1:].strip(),
                            kind="question",
                            scope="long_term",
                            status="open",
                        )
                    )
                    new_questions += 1
            proposal = ""
            if self.workspace is not None:
                channel = SoulChannel(memory, self.workspace)
                if channel.propose() is not None:
                    proposal = " (soul digest proposed as diff)"
            summary = (
                f"consolidated {theories} theories, {new_questions} new "
                f"questions from {len(experiments)} experiments{proposal}"
            )
        self._emit(DreamEnd(summary=summary))

    # -- helpers --------------------------------------------------------

    def _ask_brain(self, prompt: str) -> str:
        return _ask_brain(self.agent, prompt)

    def _emit(self, event: Event) -> None:
        if self._emit_sink is not None:
            self._emit_sink(event)
