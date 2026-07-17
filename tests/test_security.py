"""Tests for security primitives."""

import pytest

from swaybot.security import (
    CommandGuard,
    PathGuard,
    SecurityError,
    SecurityEscalation,
    SecurityManager,
)


def test_path_guard_resolves_workspace_path(tmp_path):
    guard = PathGuard(tmp_path)
    resolved = guard.resolve("foo.txt")
    assert resolved == tmp_path / "foo.txt"


def test_path_guard_rejects_traversal(tmp_path):
    guard = PathGuard(tmp_path)
    with pytest.raises(SecurityError):
        guard.resolve("../outside.txt")


def test_path_guard_rejects_absolute_escape(tmp_path):
    guard = PathGuard(tmp_path)
    with pytest.raises(SecurityError):
        guard.resolve("/etc/passwd")


def test_command_guard_allows_whitelisted_command():
    guard = CommandGuard()
    assert guard.validate("ls -la") == ["ls", "-la"]


def test_command_guard_rejects_forbidden_metacharacters():
    guard = CommandGuard()
    with pytest.raises(SecurityError):
        guard.validate("ls; cat /etc/passwd")


def test_command_guard_rejects_unknown_command():
    guard = CommandGuard()
    with pytest.raises(SecurityError):
        guard.validate("rm -rf /")


def test_command_guard_rejects_empty_command():
    guard = CommandGuard()
    with pytest.raises(SecurityError):
        guard.validate("")


def test_security_manager_escalates_after_threshold():
    manager = SecurityManager(threshold=2)
    session_id = "sess-1"
    manager.record_violation(session_id)
    assert not manager.is_escalated(session_id)
    manager.record_violation(session_id)
    assert manager.is_escalated(session_id)


def test_security_manager_tracks_violations_per_session():
    manager = SecurityManager(threshold=3)
    manager.record_violation("a")
    manager.record_violation("a")
    manager.record_violation("b")
    assert manager.status("a")["violations"] == 2
    assert manager.status("b")["violations"] == 1


def test_security_manager_reset_clears_violations():
    manager = SecurityManager(threshold=1)
    manager.record_violation("a")
    assert manager.is_escalated("a")
    manager.reset("a")
    assert not manager.is_escalated("a")


def test_security_manager_check_path_records_violation():
    manager = SecurityManager(threshold=1)
    with pytest.raises(SecurityError):
        manager.check_path("x", "/etc/passwd")
    assert manager.is_escalated("x")


def test_security_manager_check_command_records_violation():
    manager = SecurityManager(threshold=1)
    with pytest.raises(SecurityError):
        manager.check_command("x", "rm -rf /")
    assert manager.is_escalated("x")


def test_security_manager_status_includes_threshold():
    manager = SecurityManager(threshold=5)
    assert manager.status("new") == {
        "violations": 0,
        "threshold": 5,
        "escalated": False,
    }
