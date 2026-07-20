"""Tests for the permission-policy engine."""

import pytest

from swaybot.agent import Agent
from swaybot.security import (
    PermissionMode,
    PermissionPolicy,
    ToolPermissionError,
)
from swaybot.tools import ToolRegistry, build_default_registry, tool


def test_read_only_policy_allows_read_tools():
    policy = PermissionPolicy(mode=PermissionMode.READ_ONLY)
    allowed, _ = policy.allows("read_file", "read")
    assert allowed


def test_read_only_policy_denies_write_tools():
    policy = PermissionPolicy(mode=PermissionMode.READ_ONLY)
    allowed, reason = policy.allows("write_file", "write")
    assert not allowed
    assert "write permission" in reason


def test_read_only_policy_denies_danger_tools():
    policy = PermissionPolicy(mode=PermissionMode.READ_ONLY)
    allowed, reason = policy.allows("run_shell_command", "danger")
    assert not allowed
    assert "danger permission" in reason


def test_workspace_write_policy_allows_writes():
    policy = PermissionPolicy(mode=PermissionMode.WORKSPACE_WRITE)
    assert policy.allows("read_file", "read")[0]
    assert policy.allows("write_file", "write")[0]


def test_workspace_write_policy_denies_danger():
    policy = PermissionPolicy(mode=PermissionMode.WORKSPACE_WRITE)
    allowed, reason = policy.allows("run_shell_command", "danger")
    assert not allowed
    assert "danger permission" in reason


def test_danger_policy_allows_everything():
    policy = PermissionPolicy(mode=PermissionMode.DANGER)
    assert policy.allows("read_file", "read")[0]
    assert policy.allows("write_file", "write")[0]
    assert policy.allows("run_shell_command", "danger")[0]


def test_policy_denied_tools_override_mode():
    policy = PermissionPolicy(
        mode=PermissionMode.DANGER, denied_tools={"run_shell_command"}
    )
    allowed, reason = policy.allows("run_shell_command", "danger")
    assert not allowed
    assert "denied by policy" in reason


def test_policy_allowed_tools_whitelist():
    policy = PermissionPolicy(
        mode=PermissionMode.DANGER, allowed_tools={"read_file"}
    )
    assert policy.allows("read_file", "read")[0]
    allowed, reason = policy.allows("write_file", "write")
    assert not allowed
    assert "not in the allowed list" in reason


def test_tool_required_permission_mapping():
    registry = build_default_registry()
    assert registry.get("read_file").required_permission == "read"
    assert registry.get("write_file").required_permission == "write"
    assert registry.get("edit_file").required_permission == "write"
    assert registry.get("run_shell_command").required_permission == "danger"


def test_agent_permission_policy_blocks_unauthorized_tools():
    registry = ToolRegistry()

    @tool(read_only=False, risk_level="medium")
    def mutate() -> str:
        """Mutate something."""
        return "mutated"

    registry.register("mutate", mutate)

    class MutateBrain:
        def think(
            self, perception, available_tools, metadata=None,
            stream_callback=None, reasoning_callback=None
        ):
            return {"name": "mutate", "args": {}}

    agent = Agent(
        brain=MutateBrain(),
        tools=registry,
        permission_policy=PermissionPolicy(mode=PermissionMode.READ_ONLY),
    )
    env = agent.run("test", max_steps=2)
    assert any(
        "requires write permission" in str(h.get("result", "")) for h in env.history
    )


def test_agent_permission_policy_allows_authorized_tools():
    registry = ToolRegistry()

    @tool(read_only=False, risk_level="medium")
    def mutate() -> str:
        """Mutate something."""
        return "mutated"

    registry.register("mutate", mutate)

    class MutateBrain:
        def think(
            self, perception, available_tools, metadata=None,
            stream_callback=None, reasoning_callback=None
        ):
            return {"name": "mutate", "args": {}}

    agent = Agent(
        brain=MutateBrain(),
        tools=registry,
        permission_policy=PermissionPolicy(mode=PermissionMode.WORKSPACE_WRITE),
    )
    env = agent.run("test", max_steps=2)
    assert any(h.get("result") == "mutated" for h in env.history)


def test_agent_falls_back_to_permission_level_without_policy():
    registry = ToolRegistry()

    @tool(risk_level="high")
    def dangerous() -> str:
        """Do something risky."""
        return "done"

    registry.register("dangerous", dangerous)

    class DangerBrain:
        def think(
            self, perception, available_tools, metadata=None,
            stream_callback=None, reasoning_callback=None
        ):
            return {"name": "dangerous", "args": {}}

    agent = Agent(brain=DangerBrain(), tools=registry, permission_level="low")
    env = agent.run("test", max_steps=2)
    assert any("Permission denied" in str(h.get("result", "")) for h in env.history)
