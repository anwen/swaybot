"""Pluggable tool provider abstraction."""

from typing import Any, Protocol, runtime_checkable

from ..request_context import RequestContext
from ..tools import Tool, ToolRegistry


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


class LocalToolProvider:
    """Wraps the built-in ``ToolRegistry`` as a provider."""

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or ToolRegistry()

    def list_tools(self) -> list[Tool]:
        return list(self.registry._tools.values())

    def execute(
        self,
        name: str,
        args: dict[str, Any],
        context: RequestContext | None = None,
    ) -> Any:
        return self.registry.execute(
            {"name": name, "args": args},
            request_context=context,
        )

    def close(self) -> None:
        pass


class ShellToolProvider:
    """Provider that exposes the sandboxed shell command tool."""

    def __init__(self, command_guard=None) -> None:
        from .shell import run_shell_command
        from ..security import CommandGuard

        self.fn = run_shell_command
        self.guard = command_guard or CommandGuard()

    def list_tools(self) -> list[Tool]:
        return [
            Tool(
                name="run_shell_command",
                description="Run an allowed shell command.",
                inputs={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                    },
                    "required": ["command"],
                },
                output_type="string",
                fn=self.fn,
                read_only=True,
                risk_level="high",
            )
        ]

    def execute(
        self,
        name: str,
        args: dict[str, Any],
        context: RequestContext | None = None,
    ) -> Any:
        if name != "run_shell_command":
            raise ValueError(f"Unknown tool: {name}")
        command = args.get("command", "")
        self.guard.validate(command)
        return self.fn(command)

    def close(self) -> None:
        pass
