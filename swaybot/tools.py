import inspect
from dataclasses import dataclass
from typing import Callable, Union, get_origin, get_args


def _type_to_json_schema(tp: type) -> dict:
    """Map a Python type to a minimal JSON schema fragment."""
    if tp is str:
        return {"type": "string"}
    if tp is int:
        return {"type": "integer"}
    if tp is float:
        return {"type": "number"}
    if tp is bool:
        return {"type": "boolean"}
    if tp in (list, set, tuple):
        return {"type": "array"}
    if tp is dict:
        return {"type": "object"}

    origin = get_origin(tp)
    if origin is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return _type_to_json_schema(args[0])

    return {"type": "string"}


@dataclass
class Tool:
    """A tool with inferred name, description, and JSON schema."""

    name: str
    description: str
    inputs: dict
    output_type: str
    fn: Callable

    def __call__(self, **kwargs):
        return self.fn(**kwargs)


def tool(fn: Callable | None = None, *, name: str | None = None) -> Tool:
    """Decorator that turns a plain function into a schema-aware Tool."""
    if fn is None:
        return lambda f: tool(f, name=name)  # type: ignore[misc]

    tool_name = name or fn.__name__
    description = (fn.__doc__ or "").strip()
    sig = inspect.signature(fn)

    properties: dict[str, dict] = {}
    required: list[str] = []
    for param_name, param in sig.parameters.items():
        schema = _type_to_json_schema(param.annotation)
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
        else:
            schema["default"] = param.default
        properties[param_name] = schema

    inputs = {"type": "object", "properties": properties}
    if required:
        inputs["required"] = required

    return_annotation = sig.return_annotation
    if return_annotation is inspect.Parameter.empty:
        output_type = "string"
    else:
        output_type = _type_to_json_schema(return_annotation).get("type", "string")

    return Tool(
        name=tool_name,
        description=description,
        inputs=inputs,
        output_type=output_type,
        fn=fn,
    )


class ToolRegistry:
    """Simple registry for named tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, fn: Callable | Tool) -> None:
        """Register a callable or an existing Tool under the given name."""
        if isinstance(fn, Tool):
            self._tools[name] = fn
        else:
            self._tools[name] = tool(fn)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def describe(self, name: str) -> str:
        """Return a short signature description for a registered tool."""
        t = self._tools.get(name)
        if t is None:
            return f"{name}: unknown tool"
        props = t.inputs.get("properties", {})
        params = []
        for param_name, schema in props.items():
            if "default" in schema:
                params.append(f"{param_name}={schema['default']!r}")
            else:
                params.append(param_name)
        return f"{name}({', '.join(params)})"

    def descriptions(self) -> list[str]:
        """Return signature descriptions for all registered tools."""
        return [self.describe(name) for name in self.names()]

    def schemas(self) -> list[dict]:
        """Return JSON-schema descriptions for all registered tools."""
        return [
            {
                "name": name,
                "description": t.description,
                "parameters": t.inputs,
            }
            for name, t in self._tools.items()
        ]

    def execute(self, action: dict) -> object:
        name = action.get("name")
        args = action.get("args", {})
        t = self._tools.get(name)
        if t is None:
            raise ValueError(f"Unknown tool: {name}")
        sig = inspect.signature(t.fn)
        valid_args = {k: v for k, v in args.items() if k in sig.parameters}
        return t.fn(**valid_args)


def echo(message: str = "") -> str:
    """Echo a message back unchanged."""
    return message


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def done() -> str:
    """Signal that the task is complete."""
    return "finished"


def final_answer(answer: str) -> str:
    """Provide the final answer and end the task."""
    return answer


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
    registry.register("final_answer", final_answer)
    return registry
