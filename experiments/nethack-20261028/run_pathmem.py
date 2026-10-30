#!/usr/bin/env python3
"""In-episode path memory: does telling the agent WHERE IT HAS ALREADY BEEN
reduce aimless re-exploration in MazeExplore?

Arm A (none):  no path info (baseline)
Arm B (path):  prompt includes the list of visited (x,y) positions so far
Each arm 4 episodes on MiniHack-ExploreMaze-Easy-v0.
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


def _episode(env, brain, max_steps: int, path_memory: bool) -> dict:
    action_map = build_action_map(env)
    obs, _ = env.reset()
    visited: list[tuple[int, int]] = []
    total_reward = 0.0
    done = truncated = False
    step = 0
    while step < max_steps and not (done or truncated):
        r = render_observation(obs)
        pos = (r["x"], r["y"])
        if pos not in visited:
            visited.append(pos)
        prompt = (
            "You are playing NetHack. Reach the staircase (>) to descend and win. "
            "The map is ASCII: '@' is you, '.' floor, '#' wall, '+' door. "
            "You see the map every step.\n"
            f"Current map:\n{r['map']}\n"
            f"Last message: {r['message']}\nHP: {r['hp']} | position: ({r['x']},{r['y']})\n"
            f"Actions: {', '.join(MOVE_NAMES)}.\n"
        )
        if path_memory:
            trail = " ".join(f"({x},{y})" for x, y in visited[-20:])
            prompt += f"You have already visited: {trail}\nAvoid re-visiting old cells unless necessary.\n"
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
        obs, reward, done, truncated, _ = env.step(act)
        total_reward += reward
        step += 1
    return {"success": bool(done and total_reward > 0), "steps": step,
            "reward": total_reward, "visited": len(visited)}


def main() -> None:
    load_env()
    import minihack, gymnasium as gym
    from swaybot.llm_brain import LLMBrain

    for arm, pm in (("none", False), ("path", True)):
        for ep in range(4):
            env = gym.make("MiniHack-ExploreMaze-Easy-v0")
            res = _episode(env, LLMBrain(temperature=0.0, max_tokens=1024), 60, pm)
            env.close()
            print(f"[{arm} ep{ep}] success={res['success']} steps={res['steps']} visited={res['visited']}", flush=True)
            with (RUN_DIR / "results-pathmem.jsonl").open("a") as fh:
                fh.write(json.dumps({"arm": arm, "ep": ep, **res}) + "\n")

    for arm in ("none", "path"):
        rows = [json.loads(l) for l in (RUN_DIR / "results-pathmem.jsonl").read_text().splitlines()
                if json.loads(l)["arm"] == arm]
        succ = sum(r["success"] for r in rows)
        avg = sum(r["steps"] for r in rows) / len(rows)
        avg_v = sum(r["visited"] for r in rows) / len(rows)
        print(f"{arm}: success {succ}/{len(rows)} avg_steps={avg:.1f} avg_visited={avg_v:.1f}")


if __name__ == "__main__":
    main()
