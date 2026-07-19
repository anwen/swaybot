# pyright: reportMissingImports=false
"""Validation experiments for SwayBot's self-improvement design.

These experiments exercise the current implementation with deterministic and
optional LLM brains, then report whether the behaviors required by README.md
and SOUL.md are observable.
"""

import argparse
import copy
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from swaybot.agent import Agent
from swaybot.brain import Brain, EchoBrain
from swaybot.explorer import Explorer
from swaybot.llm_brain import LLMBrain
from swaybot.memory import Memory, MemoryStore, ReflectionStep
from swaybot.reflection import Reflector
from swaybot.run_log import load_runs, run_log_path_for_memory


class MonitoringBrain(Brain):
    """A deterministic brain that also populates monitoring metadata."""

    def __init__(self, actions: list[dict]) -> None:
        self.actions = actions
        self.index = 0

    def think(
        self,
        perception: dict,
        _: list[str],
        metadata: dict | None = None,
    ) -> dict:
        if metadata is not None:
            metadata["model_input_messages"] = [
                {"role": "user", "content": json.dumps(perception, default=str)}
            ]
            metadata["raw_output"] = f"raw-{self.index}"
            metadata["token_usage"] = {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
            }
            metadata["duration_ms"] = 10.0 + self.index
        action = copy.deepcopy(self.actions[self.index % len(self.actions)])
        self.index += 1
        return action


class CapturingBrain(Brain):
    """Record the perception so we can inspect what the agent saw."""

    def __init__(self) -> None:
        self.perceptions: list[dict] = []
        self.index = 0

    def think(
        self,
        perception: dict,
        _available_tools: list[str],
        _metadata: dict | None = None,
    ) -> dict:
        _ = (_available_tools, _metadata)
        self.perceptions.append(copy.deepcopy(perception))
        self.index += 1
        if perception["step"] + 1 >= perception["max_steps"]:
            return {"name": "done", "args": {}}
        return {"name": "echo", "args": {"message": "captured"}}


def _make_memory(path: Path) -> MemoryStore:
    path.parent.mkdir(parents=True, exist_ok=True)
    return MemoryStore(path=path)


def experiment_memory_lifecycle_and_inspectability() -> dict:
    """Verify that short-term steps are archived and runs remain inspectable."""
    with tempfile.TemporaryDirectory() as tmp:
        memory_path = Path(tmp) / "memory.json"
        store = _make_memory(memory_path)
        agent = Agent(
            brain=MonitoringBrain([
                {"name": "echo", "args": {"message": "hello"}},
                {"name": "done", "args": {}},
            ]),
            memory=store,
            reflector=Reflector(store),
        )
        env = agent.run("lifecycle test", max_steps=3, reflect=True)

        runs = load_runs(run_log_path_for_memory(memory_path))
        short_term = [m for m in store.memories if m.scope == "short_term"]
        long_term = [m for m in store.memories if m.scope == "long_term"]

        checks = {
            "run_finished": env.done,
            "run_log_written": len(runs) == 1,
            "short_term_pruned": len(short_term) == 0,
            "long_term_reflections": len(long_term) >= 1,
            "steps_have_monitoring": all(
                step.get("raw_output")
                and step.get("token_usage")
                and step.get("duration_ms") is not None
                and step.get("model_input_messages")
                for step in runs[0]["steps"]
            ) if runs else False,
        }
        return {
            "name": "memory lifecycle & inspectability",
            "checks": checks,
            "details": {
                "steps": len(env.history),
                "long_term_kinds": Counter(m.kind for m in long_term),
                "totals": runs[0].get("totals") if runs else None,
            },
        }


def experiment_contradiction_and_belief_update() -> dict:
    """Verify that refuted hypotheses lower credibility of related beliefs."""
    with tempfile.TemporaryDirectory() as tmp:
        memory_path = Path(tmp) / "memory.json"
        store = _make_memory(memory_path)
        store.add(
            Memory(
                content="add always succeeds",
                kind="fact",
                credibility=0.8,
                tags=["math"],
            )
        )
        agent = Agent(
            brain=MonitoringBrain([
                {"name": "add", "args": {"a": 1, "b": "x"}},
                {"name": "done", "args": {}},
            ]),
            memory=store,
            reflector=Reflector(store),
        )
        agent.run(
            "test add with invalid input",
            max_steps=2,
            reflect=True,
            hypothesis="add always succeeds",
        )

        fact = store.query(kind="fact")[0]
        belief_updates = store.query(kind="belief_update")
        verifications = store.query(kind="verification")

        checks = {
            "fact_credibility_lowered": fact.credibility < 0.8,
            "belief_update_recorded": len(belief_updates) >= 1,
            "verification_recorded": len(verifications) >= 1,
            "verdict_refuted": "refuted" in verifications[0].content.lower(),
        }
        return {
            "name": "contradiction & belief update",
            "checks": checks,
            "details": {
                "final_credibility": fact.credibility,
                "belief_update_content": belief_updates[0].content if belief_updates else None,
            },
        }


def experiment_idle_self_reflection_convergence() -> dict:
    """Verify idle exploration is driven by open questions and avoids repeats."""
    with tempfile.TemporaryDirectory() as tmp:
        memory_path = Path(tmp) / "memory.json"
        store = _make_memory(memory_path)
        question = "Why does echo return an empty string for None?"
        contradiction = "Possible contradiction: echo changes whitespace"
        store.add(
            ReflectionStep(
                content=question,
                kind="question",
                scope="long_term",
                tags=["explore"],
                credibility=0.6,
            )
        )
        store.add(
            ReflectionStep(
                content=contradiction,
                kind="contradiction",
                scope="long_term",
                tags=["explore"],
                credibility=0.5,
            )
        )
        agent = Agent(brain=EchoBrain(), memory=store)
        explorer = Explorer(agent, max_steps=2)

        tasks: list[str] = []
        for _ in range(4):
            task, _ = explorer.run()
            tasks.append(task.task)

        candidates = explorer._candidate_hypotheses_from_memory()
        verifications = store.query(kind="verification", scope="long_term")
        verified_tags = set()
        for mem in verifications:
            verified_tags.update(mem.tags)

        checks = {
            "question_picked_first": question in tasks[0],
            "contradiction_picked_second": contradiction in tasks[1],
            "verified_question_skipped": question in verified_tags,
            "verified_contradiction_skipped": contradiction in verified_tags,
            "candidates_exhausted": len(candidates) == 0,
            "fallback_after_exhaustion": "Test whether" in tasks[2] or "Verify that" in tasks[2],
        }
        return {
            "name": "idle self-reflection convergence",
            "checks": checks,
            "details": {
                "tasks": tasks,
                "verification_count": len(verifications),
                "long_term_kinds": Counter(m.kind for m in store.memories),
            },
        }


def experiment_multi_turn_behavior_reuse() -> dict:
    """Verify long-term reflections from a first run guide a second run."""
    with tempfile.TemporaryDirectory() as tmp:
        memory_path = Path(tmp) / "memory.json"
        store = _make_memory(memory_path)

        first_agent = Agent(
            brain=MonitoringBrain([
                {"name": "echo", "args": {"message": "hi!"}},
                {"name": "final_answer", "args": {"answer": "hi!"}},
            ]),
            memory=store,
            reflector=Reflector(store),
        )
        first_agent.run(
            "echo punctuation",
            max_steps=3,
            reflect=True,
            hypothesis="echo preserves punctuation",
        )

        capturing = CapturingBrain()
        second_agent = Agent(brain=capturing, memory=store)
        second_agent.run("echo longer sentence", max_steps=2, reflect=False)

        first_perception = capturing.perceptions[0]
        guidance = first_perception.get("behavior_guidance", "")
        context = first_perception.get("memory_context", "")

        checks = {
            "behavior_guidance_present": bool(guidance),
            "first_reflection_in_guidance": "echo preserves punctuation" in guidance,
            "first_reflection_in_context": "echo preserves punctuation" in context,
        }
        return {
            "name": "multi-turn behavior reuse",
            "checks": checks,
            "details": {
                "behavior_guidance": guidance,
                "memory_context": context,
            },
        }


def experiment_llm_multi_turn(run_llm: bool = False) -> dict:
    """Optional: run the same multi-turn sequence with an LLM brain."""
    if not run_llm:
        return {
            "name": "LLM multi-turn dialogue",
            "skipped": True,
            "checks": {},
            "details": {"reason": "pass --run-llm or set SWAYBOT_API_KEY to enable"},
        }

    with tempfile.TemporaryDirectory() as tmp:
        memory_path = Path(tmp) / "memory.json"
        store = _make_memory(memory_path)
        brain = LLMBrain()
        agent = Agent(brain=brain, memory=store, reflector=Reflector(store))
        env1 = agent.run(
            "What is 3 + 4? Use a tool and then final_answer.",
            max_steps=4,
            reflect=True,
        )
        env2 = agent.run(
            "Now what is 5 + 6?",
            max_steps=4,
            reflect=True,
        )
        runs = load_runs(run_log_path_for_memory(memory_path))
        checks = {
            "ran": True,
            "first_run_finished": env1.done,
            "second_run_finished": env2.done,
            "second_run_uses_memory": len(store.query(scope="long_term")) > 0,
            "runs_inspectable": len(runs) == 2,
        }
        return {
            "name": "LLM multi-turn dialogue",
            "checks": checks,
            "details": {
                "first_final_answer": runs[0].get("final_answer") if runs else None,
                "second_final_answer": runs[1].get("final_answer") if len(runs) > 1 else None,
            },
        }


def _print_result(result: dict) -> None:
    print(f"\n## {result['name']}")
    for name, ok in result["checks"].items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
    print(f"  details: {json.dumps(result['details'], indent=2, ensure_ascii=False)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate SwayBot against README.md and SOUL.md goals"
    )
    parser.add_argument(
        "--run-llm",
        action="store_true",
        help="Also run the optional LLM multi-turn experiment",
    )
    args = parser.parse_args()

    results = [
        experiment_memory_lifecycle_and_inspectability(),
        experiment_contradiction_and_belief_update(),
        experiment_idle_self_reflection_convergence(),
        experiment_multi_turn_behavior_reuse(),
        experiment_llm_multi_turn(run_llm=args.run_llm),
    ]

    for result in results:
        _print_result(result)

    all_pass = all(
        all(r["checks"].values()) for r in results if not r.get("skipped")
    )
    print("\n" + "=" * 50)
    print(f"Overall: {'PASS' if all_pass else 'FAIL'}")

    print("\nMapping to README/SOUL goals:")
    mapping = [
        ("memory lifecycle & inspectability", "Experience becomes inspectable structure"),
        ("contradiction & belief update", "Detect contradictions and update credibility"),
        ("idle self-reflection convergence", "Idle curiosity reduces uncertainty, avoids loops"),
        ("multi-turn behavior reuse", "Past reflections guide future decisions"),
        ("LLM multi-turn dialogue", "End-to-end learning in real language tasks"),
    ]
    for exp, goal in mapping:
        print(f"  - {exp}: {goal}")

    print("\nKnown gaps vs SOUL.md:")
    gaps = [
        "No active external search / counterexample gathering beyond existing memory",
        "No explicit experimental-design step; Explorer picks questions but does not plan controls",
        "No self-modification of code/tools (body maintenance) yet",
        "No principle-review stage before decisions",
        "No training of next-generation weights",
    ]
    for gap in gaps:
        print(f"  - {gap}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
