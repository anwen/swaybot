"""Session-scoped task list tools (TodoRead / TodoWrite).

The agent's own to-do list: an externalized plan it can consult and tick
off. This is the cheapest intervention against non-convergent loops — the
airline task-7 failure mode of rotating tools for 50 steps without ever
signalling completion — and gives long runs an observable progress trail.

State is per-store (one per default registry, i.e. per Agent); statuses
follow the pending / in_progress / completed convention.
"""

import json

from . import tool

_STATUSES = ("pending", "in_progress", "completed")


class TodoStore:
    """In-memory task list; lives as long as the owning Agent."""

    def __init__(self) -> None:
        self.todos: list[dict] = []


def _validate(todos: object) -> str | None:
    if not isinstance(todos, list):
        return "Error: todos must be a list"
    for item in todos:
        if not isinstance(item, dict):
            return "Error: each todo must be an object with content and status"
        if not str(item.get("content", "")).strip():
            return "Error: each todo needs non-empty content"
        if item.get("status") not in _STATUSES:
            return (
                "Error: status must be one of "
                + ", ".join(_STATUSES)
                + f" (got {item.get('status')!r})"
            )
    return None


def _summary(todos: list[dict]) -> str:
    counts = {status: 0 for status in _STATUSES}
    for item in todos:
        counts[item["status"]] += 1
    parts = [f"{status}={counts[status]}" for status in _STATUSES]
    return f"{len(todos)} todos ({', '.join(parts)})"


def create_todo_tools() -> tuple:
    """Build a todo_read/todo_write pair bound to one fresh TodoStore."""

    store = TodoStore()

    @tool(read_only=True)
    def todo_read() -> str:
        """Read your current task list with each item's status."""
        return json.dumps({"todos": store.todos}, ensure_ascii=False)

    @tool(read_only=True)
    def todo_write(todos: list) -> str:
        """Replace the task list.

        Pass the FULL list every time (not just changes). Each item is
        ``{"content": str, "status": "pending" | "in_progress" | "completed"}``.
        Mark items completed as soon as they are done, keep exactly one
        in_progress, and add new items as they arise. An empty list clears it.
        """
        problem = _validate(todos)
        if problem:
            return problem
        store.todos = [
            {"content": str(item["content"]), "status": item["status"]}
            for item in todos
        ]
        return f"Task list updated: {_summary(store.todos)}"

    return todo_read, todo_write
