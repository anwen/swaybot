"""MiniHack adapter for swaybot's self-evolution architecture.

Text rendering of NLE observations -> LLMBrain single-action-per-step ->
experience extraction after each episode -> MemoryStore -> guidance
injection into the next episode. Same loop as tau3-knowledge, different
modality: long-horizon trial-and-error learning instead of procedure
following.
"""

from __future__ import annotations

import json
from pathlib import Path

import minihack  # noqa: F401  (registers envs)
import gymnasium as gym

# NLE movement actions (ASCII): k=j? no - see ACTIONS: 0='k'(107) north,
# 1='j'(106) south, 2='l'(108) east, 3='h'(104) west, 4='y'(121) NW,
# 5='u'(117) NE, 6='b'(98) SW, 7='n'(110) SE.
MOVE_NAMES = ["north", "south", "east", "west", "northwest", "northeast",
              "southwest", "southeast"]

DIRECTION_WORDS = {
    "north": "k", "south": "j", "east": "l", "west": "h",
    "northwest": "y", "northeast": "u", "southwest": "b", "southeast": "n",
    "up": ">", "down": "<", "wait": ".", "eat": "e", "pickup": ",",
    "zap": "z", "read": "r", "open": "o", "close": "c",
    "kick": "^D", "search": "s",
}


def render_observation(obs) -> dict:
    chars = obs["chars"]
    lines = ["".join(chr(c) for c in row).rstrip() for row in chars]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    width = max((len(l) for l in lines), default=0)
    msg = bytes(obs["message"]).decode("ascii", errors="replace").rstrip("\x00")
    bl = obs["blstats"]
    # blstats layout (NLE): [x,y,STR,DEX,CON,INT,WIS,CHA,score,HP,maxHP,...]
    return {
        "map": "\n".join(f"{l:<{width}}" for l in lines),
        "message": msg,
        "hp": f"{bl[9]}/{bl[10]}",
        "x": int(bl[0]), "y": int(bl[1]),
        "hunger": int(bl[21]) if len(bl) > 21 else -1,
    }


def parse_action(answer: str, n_actions: int) -> int | None:
    """Extract an action index from the brain's answer.

    The brain answers with a single direction word (or action name);
    unknown words map to None (the driver treats them as a wait step).
    """
    text = (answer or "").strip().lower()
    for word in ("northwest", "northeast", "southwest", "southeast",
                 "north", "south", "east", "west", "up", "down",
                 "wait", "eat", "pickup", "search", "open", "close", "zap", "read"):
        if word in text:
            idx = MOVE_NAMES.index(word) if word in MOVE_NAMES else None
            if idx is not None and idx < n_actions:
                return idx
            return None
    return None


def make_env(task_id: str, max_steps: int = 300):
    env = gym.make(task_id)
    return env
