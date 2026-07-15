from typing import Callable


class ToolRegistry:
    """Simple registry for named tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Callable] = {}

    def register(self, name: str, fn: Callable) -> None:
        self._tools[name] = fn

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def execute(self, action: dict) -> object:
        name = action.get("name")
        args = action.get("args", {})
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        return self._tools[name](**args)


def echo(message: str = "") -> str:
    return message


def add(a: float, b: float) -> float:
    return a + b


def done() -> str:
    return "finished"


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("echo", echo)
    registry.register("add", add)
    registry.register("done", done)
    return registry
