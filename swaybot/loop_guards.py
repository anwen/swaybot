"""Loop-level guards that observe an agent's action stream and inject advisory
context when the agent appears stuck.

These are the SwayBot analogue of a harness's repeat-call reminder: they do
NOT veto or rewrite actions (the ``AgentHook`` protocol is observe-only).
Instead they expose ``RepeatCallGuard.observe`` which the agent loop calls
after each iteration; the guard returns an optional notice string that the
agent folds into the next perception so the model sees a "you are repeating
yourself" nudge before its next call.

Design (mirrors dsh ``repeat-tool-reminder`` at a fraction of the size):

Two detection channels run in parallel:

1. **Exact repeat** (``thresholds``, default ``[3, 5, 8]``): the *same* tool
   name *and* canonical arguments (deep key-sort then JSON stringify) called
   consecutively. Argument key order does not split a real repeat. Catches a
   model hammering a failing call verbatim.

2. **Name-only repeat** (``name_only_thresholds``, default ``[5, 8]``): the
   *same* tool name called consecutively **regardless of arguments**. Catches
   the subtler loop where the model rotates arguments (e.g. a different
   reservation_id each turn) but never converges on a decision. Observed on
   airline task 7: 8 ``get_reservation_details`` calls with varying ids and no
   ``final_answer``. Exact-repeat detection misses this because every call
   canonicalizes differently; name-only detection catches it.

Escalation: the first threshold hit is a gentle reminder; later hits are
detailed (name + count + argument preview). A new user/task turn resets both
chains: repetition across a user message is not a loop. The agent loop resets
by calling ``reset`` when it sees a fresh task input.
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


def _validate_thresholds(thresholds: list[int]) -> tuple[list[int], set[int]]:
    """Validate an escalating-threshold list; return (sorted list, set)."""
    if not thresholds:
        raise ValueError("RepeatCallGuard: thresholds must not be empty")
    for value in thresholds:
        if not isinstance(value, int) or value < 2:
            raise ValueError(
                f"RepeatCallGuard: invalid threshold {value} — every threshold must be an integer >= 2"
            )
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("RepeatCallGuard: thresholds must not contain duplicates")
    sorted_thresholds = sorted(thresholds)
    return sorted_thresholds, set(sorted_thresholds)


_GENTLE_REMINDER = (
    "You are repeating the exact same tool call with identical arguments. "
    "Carefully analyze the previous result before calling again: if the task is "
    "not complete, try a different approach or different arguments instead of "
    "repeating the call."
)

_GENTLE_NAME_REMINDER = (
    "You are calling the same tool repeatedly with different arguments. "
    "Re-querying is not making progress. Review what you have already learned, "
    "decide on a concrete action, and move toward final_answer. Do not keep "
    "calling this tool to explore further."
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
        name_only_thresholds: list[int] | None = None,
    ) -> None:
        if thresholds is None:
            thresholds = [3, 5, 8]
        if name_only_thresholds is None:
            name_only_thresholds = [5, 8]
        self._thresholds, self._threshold_set = _validate_thresholds(thresholds)
        self._name_thresholds, self._name_threshold_set = _validate_thresholds(
            name_only_thresholds
        )
        self._preview_chars = arguments_preview_chars
        if not isinstance(arguments_preview_chars, int) or arguments_preview_chars < 1:
            raise ValueError(
                "RepeatCallGuard: arguments_preview_chars must be an integer >= 1"
            )
        self._key: str | None = None
        self._count: int = 0
        self._name: str | None = None
        self._name_count: int = 0

    def reset(self) -> None:
        """Clear both chains; a new user/task turn is not a continuation of a loop."""
        self._key = None
        self._count = 0
        self._name = None
        self._name_count = 0

    def observe(self, action: dict) -> str | None:
        """Advance both chains for one call; return a reminder at threshold hits.

        Returns the *most actionable* reminder if both channels fire on the
        same call: exact-repeat takes precedence (it carries argument detail),
        else name-only. Counting happens after execution (including
        denied/errored calls), because a model hammering a failing call is
        exactly the loop worth breaking.
        """
        name = action.get("name", "")
        canonical = _canonicalize(action.get("args", {}))
        key = json.dumps([name, canonical])

        # --- exact-repeat chain (name + canonical args) ---
        exact_notice: str | None = None
        if key == self._key:
            self._count += 1
        else:
            self._key = key
            self._count = 1
        if self._count in self._threshold_set:
            if self._count == self._thresholds[0]:
                exact_notice = _GENTLE_REMINDER
            else:
                exact_notice = _detailed_reminder(
                    name, self._count, canonical, self._preview_chars
                )

        # --- name-only chain (ignores args; catches argument rotation) ---
        name_notice: str | None = None
        if name == self._name:
            self._name_count += 1
        else:
            self._name = name
            self._name_count = 1
        if self._name_count in self._name_threshold_set:
            if self._name_count == self._name_thresholds[0]:
                name_notice = _GENTLE_NAME_REMINDER
            else:
                name_notice = _detailed_reminder(
                    name, self._name_count, canonical, self._preview_chars
                )

        # Exact-repeat notice is more specific (it knows the args match); prefer it.
        return exact_notice or name_notice
