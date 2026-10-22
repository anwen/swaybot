"""Tests for security primitives."""

import pytest

from swaybot.guards import CommandGuard, PathGuard
from swaybot.violations import SecurityError, SecurityEscalation, SecurityManager


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


def test_path_guard_rejects_sibling_prefix_traversal(tmp_path):
    """A directory whose name starts with the workspace name must not pass."""
    sibling = tmp_path.parent / (tmp_path.name + "2")
    sibling.mkdir(exist_ok=True)
    (sibling / "secret.txt").write_text("leak")
    guard = PathGuard(tmp_path)
    with pytest.raises(SecurityError):
        guard.resolve("../" + sibling.name + "/secret.txt")


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


def test_command_guard_mode_tiers():
    from swaybot.guards import CommandGuard
    from swaybot.violations import SecurityError
    import pytest as _pytest

    read_only = CommandGuard("read-only")
    with _pytest.raises(SecurityError, match="not allowed in read-only mode"):
        read_only.validate("python3 x.py")

    extended = CommandGuard("workspace-write")
    assert extended.validate("python3 eight_queens.py")[0] == "python3"
    assert extended.validate("pytest -q")[0] == "pytest"
    with _pytest.raises(SecurityError, match="metacharacters"):
        extended.validate("python3 x.py | tail -3")

    full = CommandGuard("danger")
    assert full.full_shell is True
    assert full.validate("python3 x.py | tail -3")  # 管道允许
    with _pytest.raises(ValueError):
        CommandGuard("bogus-mode")


def test_agent_wires_shell_guard_from_policy(tmp_path):
    from swaybot.agent import Agent
    from swaybot.permission import PermissionMode, PermissionPolicy
    from swaybot.tools import build_default_registry

    danger_agent = Agent(
        tools=build_default_registry(),
        permission_policy=PermissionPolicy(mode=PermissionMode.DANGER),
    )
    mode = getattr(danger_agent.tools.get("run_shell_command"), "_shell").command_guard.mode
    assert mode == "danger"

    ws_agent = Agent(
        tools=build_default_registry(),
        permission_policy=PermissionPolicy(mode=PermissionMode.WORKSPACE_WRITE),
    )
    mode = getattr(ws_agent.tools.get("run_shell_command"), "_shell").command_guard.mode
    assert mode == "workspace-write"
