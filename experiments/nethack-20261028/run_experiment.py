#!/usr/bin/env python3
"""NetHack experience-learning experiment for the swaybot architecture.

Design (disjoint learn/test split per task, memory ON vs OFF):
  per task:
    learn : N episodes with experience extraction + injection (arm=learn)
    test  : N episodes on a DIFFERENT env id of the same task family
            arm=test_memory   (with learned guidance)
            arm=test_baseline (no guidance)

Tasks are MiniHack mini-missions (a few hundred steps); the metric is
success rate (reward>0 at episode end) and steps-to-success.

Usage: python run_experiment.py --tasks MiniHack-Room-5x5-v0 --episodes 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

RUN_DIR = Path(__file__).parent
sys.path.insert(0, str(RUN_DIR.parents[1]))

from benchmarks.nethack.adapter import (
    COMMAND_WORDS,
    MOVE_NAMES,
    build_action_map,
    make_env,
    parse_action,
    render_observation,
)

from _env import load_env


def _episode(env_id: str, brain, guidance: str, max_steps: int) -> dict:
    env = make_env(env_id)
    action_map = build_action_map(env)
    obs, _ = env.reset()
    transcript: list[str] = []
    total_reward = 0.0
    done = False
    truncated = False
    step = 0
    while step < max_steps and not (done or truncated):
        r = render_observation(obs)
        goal = (
            "Walk onto the food (%), pick it up (action 'pickup'), then eat it "
            "(action 'eat') to win. If the game asks for confirmation (e.g. "
            "'eat it? [ynq]'), answer 'yes'. "
            if "Eat" in env_id else
            "Reach the staircase (>) to descend and win. "
        )
        actions_hint = ", ".join(MOVE_NAMES) + (
            ", eat, pickup" if "Eat" in env_id else ""
        )
        prompt = (
            f"You are playing NetHack. {goal}"
            "The map is ASCII: '@' is you, '.' floor, '#' wall, '+' door, "
            "'%' food. You see the map every step.\n"
            f"Current map:\n{r['map']}\n"
            f"Last message: {r['message']}\nHP: {r['hp']} | position: ({r['x']},{r['y']})\n"
            f"Actions: {actions_hint}.\n"
            "Choose ONE action. Reply with exactly one action word "
            "(e.g. 'east' or 'eat'), or a single letter if the game is "
            "asking you to choose an item (e.g. 'f' or 'y'). Do not explain."
        )
        if guidance:
            prompt += (
                "\n\nLessons from previous episodes (apply when relevant):\n"
                + guidance
            )
        perception = {
            "task": "nethack",
            "step": step,
            "max_steps": max_steps,
            "messages": [{"role": "user", "content": prompt}],
            "tool_descriptions": [],
        }
        action = brain.think(perception, ["final_answer"])
        answer = ""
        if action.get("name") == "final_answer":
            answer = str(action.get("args", {}).get("answer", ""))
        act = parse_action(answer, action_map, last_message=r["message"])
        if act is None:
            act = env.action_space.sample()
        obs, reward, done, truncated, _ = env.step(act)
        total_reward += reward
        r2 = render_observation(obs)
        transcript.append(
            f"step {step}: action={answer.strip() or 'random'} -> reward={reward:.2f}, "
            f"msg={r2['message'][:80]}"
        )
        step += 1
    env.close()
    return {
        "env": env_id,
        "success": bool(done and total_reward > 0),
        "steps": step,
        "total_reward": total_reward,
        "transcript": "\n".join(transcript[-60:]),
    }


def _extract_lesson(result: dict) -> list[str]:
    from swaybot.llm_brain import LLMBrain
    from swaybot.parsing import strip_markdown_fences

    verdict = "WIN" if result["success"] else "DEATH/TIMEOUT"
    brain = LLMBrain(temperature=0.0)
    perception = {
        "task": "nethack lesson",
        "step": 0,
        "max_steps": 2,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"You just played a NetHack episode ({result['env']}, {verdict}, "
                    f"{result['steps']} steps). Learn from it.\n"
                    f"Transcript tail:\n{result['transcript']}\n\n"
                    "State up to 2 DURABLE lessons for future NetHack episodes "
                    "(movement strategy, map reading, common death causes). One "
                    "lesson per line starting with '- '. Only lessons with direct "
                    "evidence in the transcript."
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
    ap.add_argument("--tasks", nargs="+", default=["MiniHack-Room-5x5-v0"])
    ap.add_argument("--episodes", type=int, default=4,
                    help="episodes per arm per task")
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--out", default="results.jsonl")
    args = ap.parse_args()

    load_env()
    from swaybot.llm_brain import LLMBrain

    results_path = RUN_DIR / args.out
    from swaybot.memory import Memory, MemoryStore

    store = MemoryStore(path=RUN_DIR / "nethack-memory.json")

    def guidance_text() -> str:
        theories = store.query(kind="theory", scope="long_term", limit=30)
        return "\n".join(f"- {m.content}" for m in theories)

    for task in args.tasks:
        # LEARN arm: with accumulation + injection
        for ep in range(args.episodes):
            res = _episode(task, LLMBrain(temperature=0.0, max_tokens=1024),
                           guidance_text(), args.max_steps)
            print(f"[{task} learn ep{ep}] success={res['success']} steps={res['steps']}", flush=True)
            for rule in _extract_lesson(res):
                store.add(Memory(content=rule, kind="theory", scope="long_term",
                                 credibility=0.6, tags=["nethack"], source="minihack"))
            with results_path.open("a") as fh:
                fh.write(json.dumps({"task": task, "arm": "learn", "ep": ep,
                                     "success": res["success"], "steps": res["steps"]}) + "\n")
        # TEST arm: baseline (no guidance)
        for ep in range(args.episodes):
            res = _episode(task, LLMBrain(temperature=0.0, max_tokens=1024), "", args.max_steps)
            print(f"[{task} test-base ep{ep}] success={res['success']} steps={res['steps']}", flush=True)
            with results_path.open("a") as fh:
                fh.write(json.dumps({"task": task, "arm": "test_base", "ep": ep,
                                     "success": res["success"], "steps": res["steps"]}) + "\n")
        # TEST arm: memory (full guidance)
        for ep in range(args.episodes):
            res = _episode(task, LLMBrain(temperature=0.0, max_tokens=1024),
                           guidance_text(), args.max_steps)
            print(f"[{task} test-mem ep{ep}] success={res['success']} steps={res['steps']}", flush=True)
            with results_path.open("a") as fh:
                fh.write(json.dumps({"task": task, "arm": "test_mem", "ep": ep,
                                     "success": res["success"], "steps": res["steps"]}) + "\n")

    # summary
    import collections
    by = collections.defaultdict(list)
    for line in results_path.read_text().splitlines():
        d = json.loads(line)
        by[(d["task"], d["arm"])].append(d)
    print("\n==== SUMMARY ====")
    for (task, arm), rows in sorted(by.items()):
        succ = sum(r["success"] for r in rows)
        avg_steps = sum(r["steps"] for r in rows) / len(rows)
        print(f"{task} [{arm}]: success {succ}/{len(rows)} avg_steps={avg_steps:.1f}")


if __name__ == "__main__":
    main()
