import sys
sys.path.insert(0, ".")
from run_census import _episode
from _env import load_env
load_env()
import minihack, gymnasium as gym
from swaybot.llm_brain import LLMBrain

FIX = {
    "KeyRoom": "MiniHack-KeyRoom-Fixed-S5-v0",
    "Boxoban": "MiniHack-Boxoban-Medium-v0",
    "WoD": "MiniHack-WoD-Easy-Restricted-v0",
    "Room-Trap": "MiniHack-Room-Trap-5x5-v0",
    "Quest": "MiniHack-Quest-Easy-v0",
    "MazeWalk-Mapped": "MiniHack-MazeWalk-Mapped-9x9-v0",
}
brain = LLMBrain(temperature=0.0, max_tokens=1024)
for fam, task_id in FIX.items():
    try:
        env = gym.make(task_id)
    except Exception as exc:
        print(f"[{fam}] {task_id}: UNAVAILABLE ({str(exc)[:50]})", flush=True)
        continue
    try:
        res = _episode(env, brain, 60)
        print(f"[{fam}] {task_id}: {'PASS' if res['success'] else 'FAIL'} steps={res['steps']}", flush=True)
        with open("results-census.jsonl", "a") as fh:
            fh.write(json.dumps({"family": fam, "task": task_id, **res}) + "\n")
    except Exception as exc:
        print(f"[{fam}] ERROR {type(exc).__name__}: {str(exc)[:60]}", flush=True)
    env.close()
