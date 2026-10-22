"""Permission policy: coarse modes and per-tool allow/deny enforcement.

This module holds the policy layer of security. ``PermissionMode`` is the
coarse execution mode; ``PermissionPolicy`` maps a tool's required
permission against the active mode. Errors live in ``violations`` and the
workspace/command guards live in ``guards``.
"""

from enum import Enum
from typing import Iterable

from .violations import SecurityError, ToolPermissionError


class PermissionMode(str, Enum):
    """Coarse permission modes for agent execution.

    - ``read-only``: only reads and safe queries.
    - ``workspace-write``: file writes within the workspace, plus running
      the code the agent wrote (python/python3/pytest, single command).
    - ``danger``: arbitrary shell execution and other high-risk tools.
    """

    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"
    DANGER = "danger"


_PERMISSION_RANK = {
    PermissionMode.READ_ONLY: 0,
    PermissionMode.WORKSPACE_WRITE: 1,
    PermissionMode.DANGER: 2,
}

_TOOL_PERMISSION_RANK = {"read": 0, "write": 1, "danger": 2}


def permission_level_to_mode(level: str) -> PermissionMode:
    """Map coarse permission level strings to a PermissionMode."""
    mapping = {
        "low": PermissionMode.READ_ONLY,
        "medium": PermissionMode.WORKSPACE_WRITE,
        "high": PermissionMode.DANGER,
    }
    return mapping.get(level, PermissionMode.WORKSPACE_WRITE)


class PermissionPolicy:
    """Named permission mode with optional tool allow/deny lists."""

    def __init__(
        self,
        mode: PermissionMode | str = PermissionMode.WORKSPACE_WRITE,
        denied_tools: Iterable[str] | None = None,
        allowed_tools: Iterable[str] | None = None,
    ) -> None:
        self.mode = PermissionMode(mode)
        self.denied_tools: set[str] = set(denied_tools or ())
        self.allowed_tools: set[str] | None = (
            set(allowed_tools) if allowed_tools is not None else None
        )

    def allows(self, tool_name: str, required_permission: str) -> tuple[bool, str]:
        """Return (allowed, reason) for the given tool."""
        if tool_name in self.denied_tools:
            return False, f"tool '{tool_name}' is denied by policy"
        if self.allowed_tools is not None and tool_name not in self.allowed_tools:
            return False, f"tool '{tool_name}' is not in the allowed list"
        required_rank = _TOOL_PERMISSION_RANK.get(required_permission)
        if required_rank is None:
            return False, f"unknown required permission '{required_permission}'"
        if required_rank > _PERMISSION_RANK[self.mode]:
            return False, (
                f"tool '{tool_name}' requires {required_permission} permission "
                f"(current mode: {self.mode.value})"
            )
        return True, ""

    def __repr__(self) -> str:
        return (
            f"PermissionPolicy(mode={self.mode.value!r}, "
            f"denied_tools={sorted(self.denied_tools)!r}, "
            f"allowed_tools={sorted(self.allowed_tools) if self.allowed_tools is not None else None!r})"
        )
