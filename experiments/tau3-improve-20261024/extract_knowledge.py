#!/usr/bin/env python3
"""Extract durable banking rules from archived failure transcripts into the
MemoryStore, then project them into a guidance file for the transfer test.

Learning set (16 tasks) is DISJOINT from the held-out test set (20 tasks);
both come from the v9 (max_steps=100, no guidance) baseline where every task
scored 0, so any treatment-arm flip is attributable to the injected rules.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RUN_DIR = Path(__file__).parent
sys.path.insert(0, str(RUN_DIR.parents[1]))

from _env import load_env

load_env()

LEARN_IDS = [
    "task_018", "task_019", "task_022", "task_026", "task_028", "task_031",
    "task_036", "task_038", "task_039", "task_040", "task_041", "task_044",
    "task_045", "task_052", "task_053", "task_054",
]
MEMORY_FILE = RUN_DIR / "knowledge-memory.json"
GUIDANCE_FILE = RUN_DIR / "fresh-guidance.md"


def _find_transcript(tid: str) -> Path | None:
    for pat in ("transcripts/v9-lane*", "transcripts/v9-rest*", "transcripts/full-lane[12]"):
        for f in sorted(RUN_DIR.glob(pat)):
            p = f / f"{tid}.json"
            if p.exists():
                return p
    return None


def _transcript_text(d: dict, limit: int = 9000) -> str:
    out = []
    for m in d["messages"]:
        role = m["role"]
        if m.get("tool_calls"):
            for tc in m["tool_calls"]:
                out.append(f'{role}: [{tc["name"]}] {json.dumps(tc["arguments"], ensure_ascii=False)[:160]}')
        else:
            c = m.get("content") or ""
            if isinstance(c, str) and c:
                out.append(f"{role}: {c[:220]}")
    return "\n".join(out)[:limit]


def _ask(prompt: str) -> str:
    from swaybot.llm_brain import LLMBrain
    from swaybot.parsing import strip_markdown_fences

    brain = LLMBrain(temperature=0.0)
    perception = {
        "task": prompt[:200],
        "step": 0,
        "max_steps": 2,
        "messages": [{"role": "user", "content": prompt}],
        "tool_descriptions": [],
    }
    action = brain.think(perception, ["final_answer"])
    if action.get("name") == "final_answer":
        return strip_markdown_fences(str(action.get("args", {}).get("answer", "")))
    return ""


def main() -> None:
    from swaybot.memory import Memory, MemoryStore

    store = MemoryStore(path=MEMORY_FILE)
    for tid in LEARN_IDS:
        tp = _find_transcript(tid)
        if tp is None:
            print(f"skip {tid}: no transcript")
            continue
        d = json.loads(tp.read_text())
        reward = d["reward"]
        verdict = "PASS" if reward >= 0.99 else "FAIL"
        answer = _ask(
            "You just ran a Rho-Bank customer-service task. Learn from it.\n"
            f"Task: {tid}. Final reward: {reward:.2f} ({verdict}).\n"
            f"Transcript (truncated):\n{_transcript_text(d)}\n\n"
            "State up to 2 DURABLE domain rules that would help on FUTURE tasks "
            "in this domain (procedure/tool-usage patterns, policy facts, "
            "verification requirements, common failure modes). One rule per "
            "line, each starting with '- '. Only rules with direct evidence "
            "in the transcript; if the task failed, state what to do "
            "differently. Do not mention task ids in the rules."
        )
        rules = [
            line[2:].strip()
            for line in answer.splitlines()
            if line.strip().startswith("- ") and len(line.strip()) > 12
        ][:2]
        for rule in rules:
            store.add(
                Memory(
                    content=rule,
                    kind="theory",
                    scope="long_term",
                    credibility=0.6,
                    tags=["banking_knowledge"],
                    source="v7-failure-learning",
                )
            )
        print(f"{tid} ({verdict}): {len(rules)} rules", flush=True)

    theories = store.query(kind="theory", scope="long_term", limit=60)
    lines = [f"- {m.content}" for m in theories if not str(m.content).startswith("Run '")]
    GUIDANCE_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nguidance file: {GUIDANCE_FILE} ({len(lines)} rules)")


if __name__ == "__main__":
    main()
