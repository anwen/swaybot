"""Tests for the session-scoped todo tools."""

import json

from swaybot.tools import build_default_registry


def _registry():
    return build_default_registry()


def test_todo_write_read_roundtrip():
    registry = _registry()
    out = registry.execute(
        {
            "name": "todo_write",
            "args": {
                "todos": [
                    {"content": "search the KB", "status": "completed"},
                    {"content": "compose the answer", "status": "in_progress"},
                    {"content": "verify amounts", "status": "pending"},
                ]
            },
        }
    )
    assert "3 todos" in out and "completed=1" in out and "in_progress=1" in out
    state = json.loads(registry.execute({"name": "todo_read", "args": {}}))
    assert [t["status"] for t in state["todos"]] == [
        "completed",
        "in_progress",
        "pending",
    ]


def test_todo_write_rejects_bad_status_and_structure():
    registry = _registry()
    out = registry.execute(
        {"name": "todo_write", "args": {"todos": [{"content": "x", "status": "done"}]}}
    )
    assert out.startswith("Error:")
    assert "pending, in_progress, completed" in out
    # 非 list 类型在 schema 校验层即被拒绝（registry 抛 ValueError，Agent 层兜为错误观察）
    import pytest

    with pytest.raises(ValueError):
        registry.execute({"name": "todo_write", "args": {"todos": "nope"}})


def test_todo_store_is_per_registry():
    first, second = _registry(), _registry()
    first.execute(
        {"name": "todo_write", "args": {"todos": [{"content": "mine", "status": "pending"}]}}
    )
    assert json.loads(second.execute({"name": "todo_read", "args": {}}))["todos"] == []


def test_todo_clears_with_empty_list():
    registry = _registry()
    registry.execute(
        {"name": "todo_write", "args": {"todos": [{"content": "a", "status": "pending"}]}}
    )
    out = registry.execute({"name": "todo_write", "args": {"todos": []}})
    assert "0 todos" in out
