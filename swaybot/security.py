"""Security hardening: workspace scope, path traversal guard, escalation.

This module re-exports the security surface that was previously defined
here. The implementation is split into focused submodules:

- :mod:`swaybot.violations` - exceptions, violation stores, ``SecurityManager``
- :mod:`swaybot.permission` - ``PermissionMode`` / ``PermissionPolicy``
- :mod:`swaybot.guards` - ``PathGuard`` / ``CommandGuard``

``swaybot.security`` remains the canonical import path for all of them.
"""

from .guards import CommandGuard, PathGuard
from .permission import (
    PermissionMode,
    PermissionPolicy,
    permission_level_to_mode,
)
from .violations import (
    InMemoryViolationStore,
    JSONViolationStore,
    SecurityError,
    SecurityEscalation,
    SecurityManager,
    ToolPermissionError,
    ViolationStore,
)

__all__ = [
    "CommandGuard",
    "InMemoryViolationStore",
    "JSONViolationStore",
    "PathGuard",
    "PermissionMode",
    "PermissionPolicy",
    "SecurityError",
    "SecurityEscalation",
    "SecurityManager",
    "ToolPermissionError",
    "ViolationStore",
    "permission_level_to_mode",
]
