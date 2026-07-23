"""Pluggable tool provider abstraction.

``ToolProvider`` is the contract for sources of callable tools that live
outside the built-in registry (custom tools, shell adapters, future MCP
support). ``ToolRegistry.register_provider`` stores providers under a
namespace and dispatches ``provider/name`` actions to them.

This is an enabled-but-extension-point abstraction: the built-in tools are
registered directly via :func:`~swaybot.tools.tool`, while providers are
opt-in for integrations that surface external tool surfaces.
"""

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..request_context import RequestContext

if TYPE_CHECKING:
    from ..tools import Tool


@runtime_checkable
class ToolProvider(Protocol):
    """A source of callable tools."""

    def list_tools(self) -> list["Tool"]:
        """Return the tools provided by this provider."""
        ...

    def execute(
        self,
        name: str,
        args: dict[str, Any],
        context: RequestContext | None = None,
    ) -> Any:
        """Execute the named tool with the given arguments."""
        ...

    def close(self) -> None:
        """Release any provider resources."""
        ...
