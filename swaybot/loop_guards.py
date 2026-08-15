"""Loop-level guards that observe an agent's action stream and inject advisory
context when the agent appears stuck.

These are the SwayBot analogue of a harness's repeat-call reminder: they do
NOT veto or rewrite actions (the ``AgentHook`` protocol is observe-only).
Instead they expose ``RepeatCallGuard.observe`` which the agent loop calls
after each iteration; the guard returns an optional notice string that the
agent folds into the next perception so the model sees a "you are repeating
yourself" nudge before its next call.

Design (mirrors dsh ``repeat-tool-reminder`` at a fraction of the size):
- Track consecutive identical calls per agent run. Identity = tool name +
  canonical arguments (deep key-sort then JSON stringify) so argument key
  order does not split a real repeat into distinct calls.
- Escalating thresholds (default ``[3, 5, 8]``): the first hit is a gentle
  reminder, later hits are detailed (name + count + argument preview).
- A new user/task turn resets the chain: repetition across a user message is
  not a loop. The agent loop resets by calling ``reset`` when it sees a fresh
  task input.
"""

from __future__ import annotations

import json
from typing import Any


def _sort_value(value: Any) -> Any:
    """Deep key-sort a JSON value so two argument objects that differ only in
    property order canonicalize identically."""
    if isinstance(value, list):
        return [_sort_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _sort_value(value[k]) for k in sorted(value, key=str)}
    return value


def _canonicalize(arguments: Any) -> str:
    """Canonical string form of a call's arguments: deep key-sort, then stringify."""
    return json.dumps(_sort_value(arguments), sort_keys=False, default=str)


_GENTLE_REMINDER = (
    "You are repeating the exact same tool call with identical arguments. "
    "Carefully analyze the previous result before calling again: if the task is "
    "not complete, try a different approach or different arguments instead of "
    "repeating the call."
)


def _detailed_reminder(tool_name: str, count: int, canonical: str, preview_chars: int) -> str:
    preview = canonical if len(canonical) <= preview_chars else (
        canonical[:preview_chars] + f"… (+{len(canonical) - preview_chars} more chars)"
    )
    return (
        "Repeated tool call detected:\n"
        f"- tool: {tool_name}\n"
        f"- consecutive_calls: {count}\n"
        f"- arguments: {preview}\n"
        "The repeated calls are not making progress. Do not call this tool with "
        "these exact arguments again. Inspect the latest result and choose a "
        "different action, different arguments, or finish the task if enough "
        "evidence has been gathered."
    )


class RepeatCallGuard:
    """Advisory per-run repeat-call detector.

    ``observe`` advances the consecutive-repeat chain for one call and returns
    a reminder string when the run length hits a configured threshold, else
    ``None``. The agent loop injects the non-None result into the next
    perception. Counting happens after execution (including denied/errored
    calls), because a model hammering a failing call is exactly the loop worth
    breaking.
    """

    def __init__(
        self,
        thresholds: list[int] | None = None,
        arguments_preview_chars: int = 500,
    ) -> None:
        if thresholds is None:
            thresholds = [3, 5, 8]
        if not thresholds:
            raise ValueError("RepeatCallGuard: thresholds must not be empty")
        for value in thresholds:
            if not isinstance(value, int) or value < 2:
                raise ValueError(
                    f"RepeatCallGuard: invalid threshold {value} — every threshold must be an integer >= 2"
                )
        if len(set(thresholds)) != len(thresholds):
            raise ValueError("RepeatCallGuard: thresholds must not contain duplicates")
        self._thresholds = sorted(thresholds)
        self._threshold_set = set(self._thresholds)
        self._preview_chars = arguments_preview_chars
        if not isinstance(arguments_preview_chars, int) or arguments_preview_chars < 1:
            raise ValueError(
                "RepeatCallGuard: arguments_preview_chars must be an integer >= 1"
            )
        self._key: str | None = None
        self._count: int = 0

    def reset(self) -> None:
        """Clear the chain; a new user/task turn is not a continuation of a loop."""
        self._key = None
        self._count = 0

    def observe(self, action: dict) -> str | None:
        """Advance the chain for one call; return a reminder at threshold hits."""
        name = action.get("name", "")
        canonical = _canonicalize(action.get("args", {}))
        key = json.dumps([name, canonical])
        if key == self._key:
            self._count += 1
        else:
            self._key = key
            self._count = 1
        if self._count not in self._threshold_set:
            return None
        if self._count == self._thresholds[0]:
            return _GENTLE_REMINDER
        return _detailed_reminder(name, self._count, canonical, self._preview_chars)
