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

# NLE CompassDirection order (unwrapped.actions): N,E,S,W,NE,SE,SW,NW
# (earlier draft had the diagonal order wrong - NW was mapped to NE).
MOVE_NAMES = ["north", "east", "south", "west", "northeast", "southeast",
              "southwest", "northwest"]

# direction word -> NLE character
DIRECTION_WORDS = {
    "north": "k", "south": "j", "east": "l", "west": "h",
    "northwest": "y", "northeast": "u", "southwest": "b", "southeast": "n",
}

# non-movement command words -> NLE character
COMMAND_WORDS = {
    "up": "<", "down": ">", "wait": ".", "eat": "e", "pickup": ",",
    "zap": "z", "read": "r", "open": "o", "close": "c", "search": "s",
    "yes": "y",
}


def build_action_map(env) -> dict[str, int]:
    """Map NLE character -> action index for THIS env's action subset."""
    mapping: dict[str, int] = {}
    for idx, act in enumerate(env.unwrapped.actions):
        val = act.value
        if isinstance(val, int) and 32 <= val < 127:
            mapping[chr(val)] = idx
    return mapping


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


def parse_action(answer: str, action_map: dict[str, int], last_message: str = "") -> int | None:
    """Extract an action index from the brain's answer.

    Accepts direction words, command words, and single-letter responses to
    in-game prompts (e.g. 'f' for the apple, 'y' for yes at a [ynq] prompt).
    Unknown words map to None (the driver treats them as a wait step).
    """
    text = (answer or "").strip().lower()
    # Single-letter response to an in-game prompt: prefer the letter's action
    # over direction-word matching when the game is asking a question.
    if len(text) == 1 and text in action_map and any(
        k in last_message for k in ("[ynq]", "what do you want", "eat it?", "eat?")
    ):
        return action_map[text]
    words = {**DIRECTION_WORDS, **COMMAND_WORDS}
    # longest-first so "northeast" wins over "east"
    for word in sorted(words, key=len, reverse=True):
        if word in text:
            return action_map.get(words[word])
    # Fallback: bare single letter with no prompt context (e.g. 'f', 'y')
    if len(text) == 1 and text in action_map:
        return action_map[text]
    return None


def make_env(task_id: str, max_steps: int = 300):
    env = gym.make(task_id)
    return env
