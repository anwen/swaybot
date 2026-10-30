#!/usr/bin/env python3
"""Scratchpad memory for MazeExplore: inject action history + accumulated
explored map into every step's prompt (the agent's notebook).

Arm A (none):  current-observation only (reproduces the wall-bump loop)
Arm B (notes): + action history (last 10) + accumulated local map
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RUN_DIR = Path(__file__).parent
sys.path.insert(0, str(RUN_DIR.parents[1]))

from benchmarks.nethack.adapter import (
    MOVE_NAMES, build_action_map, parse_action, render_observation,
)
from _env import load_env


def _episode(env, brain, max_steps: int, notes: bool) -> dict:
    action_map = build_action_map(env)
    obs, _ = env.reset()
    history: list[str] = []
    known: dict[tuple[int, int], str] = {}
    total_reward = 0.0
    done = truncated = False
    step = 0
    while step < max_steps and not (done or truncated):
        r = render_observation(obs)
        chars = obs["chars"]
        for y in range(chars.shape[0]):
            for x in range(chars.shape[1]):
                c = chr(chars[y, x])
                if c != " ":
                    known[(x, y)] = c
        prompt = (
            "You are playing NetHack. Reach the staircase (>) to descend and win. "
            "The map is ASCII: '@' is you, '.' floor, '#' wall, '+' door. "
            "You see the map every step.\n"
            f"Current map:\n{r['map']}\n"
            f"Last message: {r['message']}\nHP: {r['hp']} | position: ({r['x']},{r['y']})\n"
            f"Actions: {', '.join(MOVE_NAMES)}.\n"
        )
        if notes and history:
            prompt += (
                "\nYour notebook (what you already tried):\n"
                + "\n".join(history[-10:])
                + "\nDo NOT repeat actions that hit walls. Explore new directions.\n"
            )
        prompt += (
            "Choose ONE action. Reply with exactly one direction word "
            "(e.g. 'east'). Do not explain."
        )
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
    return {"success": bool(done and total_reward > 0), "steps": step,
            "reward": total_reward, "visited_cells": len(known)}


def main() -> None:
    load_env()
    import minihack, gymnasium as gym
    from swaybot.llm_brain import LLMBrain

    for arm, notes in (("none", False), ("notes", True)):
        for ep in range(3):
            env = gym.make("MiniHack-ExploreMaze-Easy-v0")
            res = _episode(env, LLMBrain(temperature=0.0, max_tokens=1024), 60, notes)
            env.close()
            print(f"[{arm} ep{ep}] success={res['success']} steps={res['steps']} cells={res['visited_cells']}", flush=True)
            with (RUN_DIR / "results-scratchpad.jsonl").open("a") as fh:
                fh.write(json.dumps({"arm": arm, "ep": ep, **res}) + "\n")

    for arm in ("none", "notes"):
        rows = [json.loads(l) for l in (RUN_DIR / "results-scratchpad.jsonl").read_text().splitlines()
                if json.loads(l)["arm"] == arm]
        succ = sum(r["success"] for r in rows)
        avg = sum(r["steps"] for r in rows) / len(rows)
        cells = sum(r["visited_cells"] for r in rows) / len(rows)
        print(f"{arm}: success {succ}/{len(rows)} avg_steps={avg:.1f} avg_cells={cells:.1f}")


if __name__ == "__main__":
    main()
