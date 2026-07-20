"""Pluggable tool provider abstraction."""

from typing import Any, Protocol, runtime_checkable

from ..request_context import RequestContext
from ..tools import Tool


@runtime_checkable
class ToolProvider(Protocol):
    """A source of callable tools."""

    def list_tools(self) -> list[Tool]:
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
