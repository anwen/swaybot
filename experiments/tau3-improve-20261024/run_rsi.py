#!/usr/bin/env python3
"""RSI minimal demo: 4-generation auto-evolution on telecom cold start.

Per generation g (1..G):
  test batch  : N tasks, WITH the memory-projected guidance (gen 1: none)
  control batch: M tasks, NEVER with guidance (same sampling strata)
  after running: failures in the test batch -> extract <=2 rules each ->
  MemoryStore (corroboration merge) -> project top-K live rules to guidance

Judgement: test-batch score rising across generations while the control
batch stays flat attributes gains to accumulated rules, not to re-runs or
endpoint drift. All tasks are fresh ids every generation (2285-task pool).

Usage: python run_rsi.py --generations 4 --test 12 --control 6
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

RUN_DIR = Path(__file__).parent
sys.path.insert(0, str(RUN_DIR))
sys.path.insert(0, str(RUN_DIR.parents[1]))

from _env import load_env

load_env()

DOMAIN = "telecom"
GUIDANCE_FILE = RUN_DIR / "rsi-guidance.md"
MEMORY_FILE = RUN_DIR / "rsi-memory.json"
RESULTS = RUN_DIR / "results-rsi.jsonl"
MAX_LIVE_RULES = 24
TOP_K_RULES_PER_GEN = 12


def _persona(t) -> str:
    return t.id.split("[PERSONA:")[1].rstrip("]")


def _workflow(t) -> str:
    return t.id.split("]")[0].strip("[")


def _stratified_sample(tasks, n: int, rng: random.Random, exclude: set[str]) -> list:
    """Sample n tasks stratified by workflow x persona."""
    pools: dict[tuple, list] = collections.defaultdict(list)
    for t in tasks:
        if t.id in exclude:
            continue
        pools[(_workflow(t), _persona(t))].append(t)
    keys = sorted(pools)
    picked: list = []
    i = 0
    while len(picked) < n:
        key = keys[i % len(keys)]
        i += 1
        if pools[key]:
            picked.append(pools[key].pop(rng.randrange(len(pools[key]))))
        if all(not pools[k] for k in keys):
            break
    return picked[:n]


def _extract_rules(transcript_path: Path, reward: float) -> list[str]:
    from swaybot.llm_brain import LLMBrain
    from swaybot.parsing import strip_markdown_fences

    d = json.loads(transcript_path.read_text())
    lines = []
    for m in d["messages"]:
        if m.get("tool_calls"):
            for tc in m["tool_calls"]:
                lines.append(f'agent: [{tc["name"]}] {json.dumps(tc["arguments"], ensure_ascii=False)[:140]}')
        else:
            c = m.get("content") or ""
            if isinstance(c, str) and c:
                lines.append(f'{m["role"]}: {c[:200]}')
    transcript = "\n".join(lines)[:9000]
    verdict = "PASS" if reward >= 0.99 else "FAIL"
    brain = LLMBrain(temperature=0.0)
    perception = {
        "task": "learn from telecom support task",
        "step": 0,
        "max_steps": 2,
        "messages": [
            {
                "role": "user",
                "content": (
                    "You just ran a telecom customer-support task (troubleshooting "
                    "a mobile device with diagnostic tools). Learn from it.\n"
                    f"Final reward: {reward:.2f} ({verdict}).\n"
                    f"Transcript (truncated):\n{transcript}\n\n"
                    "State up to 2 DURABLE rules for FUTURE telecom tasks (tool "
                    "usage patterns, diagnostic ordering, what to confirm with "
                    "the customer). One rule per line starting with '- '. Only "
                    "rules with direct transcript evidence."
                ),
            }
        ],
        "tool_descriptions": [],
    }
    action = brain.think(perception, ["final_answer"])
    answer = ""
    if action.get("name") == "final_answer":
        answer = strip_markdown_fences(str(action.get("args", {}).get("answer", "")))
    return [
        line[2:].strip()
        for line in answer.splitlines()
        if line.strip().startswith("- ") and len(line.strip()) > 12
    ][:2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--test", type=int, default=12)
    ap.add_argument("--control", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import os

    from benchmarks.tau3bench import registry  # noqa: F401
    from tau2.data_model.simulation import TextRunConfig
    from tau2.run import get_tasks, run_single_task

    model = os.environ["SWAYBOT_MODEL"]
    os.environ["OPENAI_API_KEY"] = os.environ["SWAYBOT_API_KEY"]
    os.environ["OPENAI_API_BASE"] = os.environ["SWAYBOT_API_BASE"]

    all_tasks = get_tasks(DOMAIN)
    rng = random.Random(args.seed)
    used: set[str] = set()

    from swaybot.memory import Memory, MemoryStore

    store = MemoryStore(path=MEMORY_FILE)

    def make_config():
        return TextRunConfig(
            domain=DOMAIN,
            agent="swaybot",
            user="user_simulator",
            llm_agent=model,
            llm_user=f"openai/{model}",
            max_steps=100,
            num_trials=1,
            seed=0,
        )

    def run_batch(label: str, batch, guidance: bool, gen: int) -> list[float]:
        rewards = []
        tdir = RUN_DIR / f"transcripts/rsi-g{gen}"
        tdir.mkdir(parents=True, exist_ok=True)
        for t in batch:
            os.environ.pop("SWAYBOT_EXTRA_GUIDANCE_FILE", None)
            if guidance:
                os.environ["SWAYBOT_EXTRA_GUIDANCE_FILE"] = str(GUIDANCE_FILE)
            try:
                result = run_single_task(make_config(), t)
                reward = float(getattr(getattr(result, "reward_info", None), "reward", 0.0))
                msgs = []
                for m in getattr(result, "messages", None) or []:
                    entry = {"role": getattr(m, "role", "?")}
                    tc = getattr(m, "tool_calls", None) or []
                    if tc:
                        entry["tool_calls"] = [
                            {"name": getattr(c, "name", ""), "arguments": getattr(c, "arguments", {})}
                            for c in tc
                        ]
                    c = getattr(m, "content", "")
                    if isinstance(c, str) and c:
                        entry["content"] = c
                    msgs.append(entry)
                (tdir / f"{t.id}.json").write_text(
                    json.dumps({"task": t.id, "reward": reward, "messages": msgs},
                               ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[{label}] ERROR {type(exc).__name__}: {exc}", flush=True)
                reward = 0.0
            rewards.append(reward)
            used.add(t.id)
            with RESULTS.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "gen": gen, "arm": label, "task": t.id,
                    "reward": reward, "persona": _persona(t),
                }, ensure_ascii=False) + "\n")
            print(f"[g{gen} {label}] {_persona(t):>4} {_workflow(t)[:22]:<22} reward={reward:.2f}", flush=True)
        return rewards

    for gen in range(1, args.generations + 1):
        test_batch = _stratified_sample(all_tasks, args.test, rng, exclude=used)
        control_batch = _stratified_sample(all_tasks, args.control, rng, exclude=used)
        ctrl = run_batch("control", control_batch, guidance=False, gen=gen)
        test = run_batch("test", test_batch, guidance=gen > 1, gen=gen)
        print(f"== gen{gen}: test={sum(test)/len(test):.3f} control={sum(ctrl)/len(ctrl):.3f} ==", flush=True)

        # learn from THIS generation's test failures
        new_rules = 0
        for t, reward in zip(test_batch, test):
            if reward >= 0.99:
                continue
            tp = RUN_DIR / f"transcripts/rsi-g{gen}" / f"{t.id}.json"
            if not tp.exists():
                continue
            for rule in _extract_rules(tp, reward):
                store.add(Memory(
                    content=rule, kind="theory", scope="long_term",
                    credibility=0.6, tags=[DOMAIN], source=f"rsi-gen{gen}",
                ))
                new_rules += 1
        print(f"[g{gen}] extracted {new_rules} rules; memory={len(store.query(kind='theory', scope='long_term', limit=999))}", flush=True)

        # project top live rules to guidance (credibility desc, capped)
        theories = store.query(kind="theory", scope="long_term", limit=999)
        theories = sorted(theories, key=lambda m: m.credibility, reverse=True)[:MAX_LIVE_RULES]
        GUIDANCE_FILE.write_text(
            "\n".join(f"- {m.content}" for m in theories) + "\n", encoding="utf-8"
        )

    # summary
    print("\n==== RSI SUMMARY ====")
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for line in RESULTS.read_text().splitlines():
        d = json.loads(line)
        by[d["gen"]][d["arm"]].append(d["reward"])
    for gen in sorted(by):
        t = by[gen].get("test", [])
        c = by[gen].get("control", [])
        print(f"gen{gen}: test={sum(t)/len(t):.3f}(n={len(t)}) control={sum(c)/len(c):.3f}(n={len(c)})")


if __name__ == "__main__":
    main()
