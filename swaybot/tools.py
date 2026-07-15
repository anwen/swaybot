import inspect
from typing import Callable


class ToolRegistry:
    """Simple registry for named tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Callable] = {}

    def register(self, name: str, fn: Callable) -> None:
        self._tools[name] = fn

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def describe(self, name: str) -> str:
        """Return a short signature description for a registered tool."""
        if name not in self._tools:
            return f"{name}: unknown tool"
        fn = self._tools[name]
        sig = inspect.signature(fn)
        params = []
        for param in sig.parameters.values():
            if param.default is inspect.Parameter.empty:
                params.append(f"{param.name}")
            else:
                params.append(f"{param.name}={param.default!r}")
        return f"{name}({', '.join(params)})"

    def execute(self, action: dict) -> object:
        name = action.get("name")
        args = action.get("args", {})
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        fn = self._tools[name]
        sig = inspect.signature(fn)
        valid_args = {
            k: v for k, v in args.items() if k in sig.parameters
        }
        return fn(**valid_args)


def echo(message: str = "") -> str:
    return message


def add(a: float, b: float) -> float:
    return a + b


def done() -> str:
    return "finished"


def format_action(action: dict) -> str:
    """Render an action dict as a readable tool call."""
    name = action.get("name", "unknown")
    args = action.get("args", {})
    args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return f"{name}({args_str})"


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("echo", echo)
    registry.register("add", add)
    registry.register("done", done)
    return registry
