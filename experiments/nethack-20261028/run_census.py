#!/usr/bin/env python3
"""MiniHack mechanism-family census: one representative per family, one
60-step baseline episode each. Produces a feasibility matrix answering
"which mechanics fall inside the model's capability window".

Families with multiple variants get their SIMPLEST representative
(fixed layout, smallest size, no Hard/Dark/Pro suffix).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

RUN_DIR = Path(__file__).parent
sys.path.insert(0, str(RUN_DIR.parents[1]))

from benchmarks.nethack.adapter import (
    MOVE_NAMES, build_action_map, parse_action, render_observation,
)
from _env import load_env

# family -> simplest representative
REPRESENTATIVES = {
    "MultiRoom": "MiniHack-MultiRoom-N2-Lava-v0",
    "LavaCrossing": "MiniHack-LavaCrossingS9N1-v0",
    "SimpleCrossing": "MiniHack-SimpleCrossingS9N1-v0",
    "KeyRoom": "MiniHack-KeyRoom-Fixed-v0",
    "ClosedDoor": "MiniHack-ClosedDoor-v0",
    "Boxoban": "MiniHack-Boxoban-v0",
    "HideNSeek": "MiniHack-HideNSeek-v0",
    "WoD": "MiniHack-WoD--Restricted-v0",
    "Freeze": "MiniHack-Freeze-Wand-Fixed-v0",
    "Zap": "MiniHack-Zap-Fixed-v0",
    "Wield": "MiniHack-Wield-Fixed-v0",
    "Wear": "MiniHack-Wear-Fixed-v0",
    "PutOn": "MiniHack-PutOn-Fixed-v0",
    "Read": "MiniHack-Read-Fixed-v0",
    "Sink": "MiniHack-Sink-Fixed-v0",
    "River": "MiniHack-River-Narrow-v0",
    "CorridorBattle": "MiniHack-CorridorBattle-v0",
    "Room-Trap": "MiniHack-Room-Trap-v0",
    "MazeWalk": "MiniHack-MazeWalk-9x9-v0",
    "Quest": "MiniHack-Quest-v0",
}

GOAL_HINTS = {
    "Eat": "Walk onto the food (%), then 'eat' (confirm with 'yes' if asked).",
    "Boxoban": "Push the boulder onto the target square.",
    "KeyRoom": "Find the key, pick it up, then open the locked door and reach >.",
}


def _episode(env, brain, max_steps: int = 60) -> dict:
    action_map = build_action_map(env)
    obs, _ = env.reset()
    history: list[str] = []
    total_reward = 0.0
    done = truncated = False
    step = 0
    env_id = env.spec.id
    goal = next((h for k, h in GOAL_HINTS.items() if k in env_id),
                "Reach the staircase (>) to descend and win.")
    while step < max_steps and not (done or truncated):
        r = render_observation(obs)
        prompt = (
            f"You are playing NetHack. {goal} "
            "The map is ASCII: '@' is you, '.' floor, '#' wall, '+' door, "
            "'%' food, '>' staircase down. You see the map every step.\n"
            f"Current map:\n{r['map']}\n"
            f"Last message: {r['message']}\nHP: {r['hp']} | position: ({r['x']},{r['y']})\n"
            f"Actions: {', '.join(MOVE_NAMES)}, eat, pickup.\n"
            "READ the game message: if it asks a question ([ynq] or item choice), "
            "answer it (e.g. 'yes' or the item letter) instead of moving.\n"
            "Choose ONE action. Reply with exactly one action word, or a single "
            "letter if the game asks for an item choice. Do not explain."
        )
        if history:
            prompt += "\nRecent actions (do not repeat failing ones):\n" + "\n".join(history[-8:])
        perception = {
            "task": "nethack", "step": step, "max_steps": max_steps,
            "messages": [{"role": "user", "content": prompt}],
            "tool_descriptions": [],
        }
        action = brain.think(perception, ["final_answer"])
        answer = str(action.get("args", {}).get("answer", "")) if action.get("name") == "final_answer" else ""
        act = parse_action(answer, action_map, last_message=r["message"])
        if act is None:
            act = env.action_space.sample()
            answer = f"random({act})"
        obs, reward, done, truncated, _ = env.step(act)
        msg = bytes(obs["message"]).decode("ascii", errors="replace").strip("\x00")[:60]
        history.append(f"step{step}: {answer} -> {msg or 'moved'}")
        total_reward += reward
        step += 1
    return {"success": bool(done and total_reward > 0), "steps": step, "reward": total_reward}


def main() -> None:
    load_env()
    import minihack, gymnasium as gym
    from swaybot.llm_brain import LLMBrain

    brain = LLMBrain(temperature=0.0, max_tokens=1024)
    out = RUN_DIR / "results-census.jsonl"
    for fam, task_id in REPRESENTATIVES.items():
        try:
            env = gym.make(task_id)
        except Exception as exc:
            print(f"[{fam}] {task_id}: UNAVAILABLE ({str(exc)[:60]})", flush=True)
            continue
        t0 = time.time()
        try:
            res = _episode(env, brain, 60)
            verdict = "PASS" if res["success"] else "FAIL"
        except Exception as exc:
            print(f"[{fam}] {task_id}: ERROR {type(exc).__name__}: {str(exc)[:80]}", flush=True)
            env.close()
            continue
        env.close()
        print(f"[{fam}] {task_id}: {verdict} steps={res['steps']} ({time.time()-t0:.0f}s)", flush=True)
        with out.open("a") as fh:
            fh.write(json.dumps({"family": fam, "task": task_id, **res}) + "\n")

    print("\n==== CENSUS MATRIX ====")
    rows = [json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []
    for r in rows:
        print(f"{'PASS' if r['success'] else 'FAIL'}  {r['family']:<16} {r['task']} ({r['steps']} steps)")


if __name__ == "__main__":
    main()
