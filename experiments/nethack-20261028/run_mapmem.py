#!/usr/bin/env python3
"""Map-state memory experiment: does accumulating EXPLORED MAP across
episodes help on MazeExplore (vs behavior-rule memory alone)?

Arm A (rule):  guidance = behavior rules only (current MemoryStore content)
Arm B (map):   guidance = behavior rules + accumulated explored map
Each arm plays 4 episodes on MiniHack-ExploreMaze-Easy-v0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RUN_DIR = Path(__file__).parent
sys.path.insert(0, str(RUN_DIR.parents[1]))

from benchmarks.nethack.adapter import (
    COMMAND_WORDS, MOVE_NAMES, build_action_map, parse_action, render_observation,
)
from _env import load_env


def _render_accumulated_map(known: dict[tuple[int, int], str], width: int = 79, height: int = 21) -> str:
    """Render explored cells as an ASCII map; unexplored = space."""
    if not known:
        return "(no explored map yet)"
    rows = []
    for y in range(height):
        row = []
        for x in range(width):
            row.append(known.get((x, y), " "))
        line = "".join(row).rstrip()
        if line.strip():
            rows.append(line)
    return "\n".join(rows) or "(empty)"


def _episode(env, brain, guidance: str, known: dict, max_steps: int = 60) -> dict:
    action_map = build_action_map(env)
    obs, _ = env.reset()
    total_reward = 0.0
    done = truncated = False
    step = 0
    while step < max_steps and not (done or truncated):
        r = render_observation(obs)
        # accumulate explored cells
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
            "Choose ONE action. Reply with exactly one direction word "
            "(e.g. 'east'). Do not explain."
        )
        if guidance:
            prompt += "\n\n" + guidance
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
    return {"success": bool(done and total_reward > 0), "steps": step, "reward": total_reward}


def main() -> None:
    load_env()
    import minihack, gymnasium as gym
    from swaybot.llm_brain import LLMBrain

    for arm in ("rule", "map"):
        known: dict[tuple[int, int], str] = {}
        for ep in range(4):
            env = gym.make("MiniHack-ExploreMaze-Easy-v0")
            guidance = ""
            if arm == "map" and known:
                guidance = (
                    "Previously explored map (from earlier episodes; use it to "
                    "navigate toward unexplored areas and the staircase '>'):\n"
                    + _render_accumulated_map(known)
                )
            res = _episode(env, LLMBrain(temperature=0.0, max_tokens=1024), guidance, known, 60)
            env.close()
            print(f"[{arm} ep{ep}] success={res['success']} steps={res['steps']} known={len(known)}", flush=True)
            with (RUN_DIR / "results-mapmem.jsonl").open("a") as fh:
                fh.write(json.dumps({"arm": arm, "ep": ep, **res, "known": len(known)}) + "\n")

    for arm in ("rule", "map"):
        rows = [json.loads(l) for l in (RUN_DIR / "results-mapmem.jsonl").read_text().splitlines()
                if json.loads(l)["arm"] == arm]
        succ = sum(r["success"] for r in rows)
        avg = sum(r["steps"] for r in rows) / len(rows)
        print(f"{arm}: success {succ}/{len(rows)} avg_steps={avg:.1f}")


if __name__ == "__main__":
    main()
