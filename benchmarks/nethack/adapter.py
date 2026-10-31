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

    Prompt-state aware: NetHack prompts change what a letter means.
      - "[ynq]" prompts -> y/n/q
      - "What do you want to X? [a-f or ?*]" -> inventory slot letter
      - "In what direction?" -> a direction word (NOT a slot letter)
    Falls back to direction/command words; unknown -> None (caller waits).
    """
    text = (answer or "").strip().lower()
    msg = (last_message or "").lower()

    # 1. Direction prompt: only direction words are valid.
    if "in what direction" in msg:
        for word in sorted(DIRECTION_WORDS, key=len, reverse=True):
            if word in text:
                return action_map.get(DIRECTION_WORDS[word])
        return None

    # 2. Item-selection prompts: a bare slot letter (a-z) picks the item.
    if "what do you want" in msg or "eat it?" in msg:
        if len(text) == 1 and text in action_map:
            return action_map[text]
        # also accept "<letter> - <item>" style or embedded letter
        for ch in text:
            if ch.isalpha() and ch in action_map:
                return action_map[ch]
        return None

    # 3. Yes/no prompt: map yes/no words and letters.
    if "[ynq]" in msg:
        if text.startswith("y") or "yes" in text:
            return action_map.get("y")
        if text.startswith("n"):
            return action_map.get("n")
        return None

    # 4. Normal state: command/direction words; a bare single letter is
    #    accepted only if it is a known command char (not a risky misfire).
    words = {**DIRECTION_WORDS, **COMMAND_WORDS}
    for word in sorted(words, key=len, reverse=True):
        if word in text:
            return action_map.get(words[word])
    if len(text) == 1 and text in action_map:
        return action_map[text]
    return None


def make_env(task_id: str, max_steps: int = 300):
    env = gym.make(task_id)
    return env
