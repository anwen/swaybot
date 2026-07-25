import contextvars
import inspect
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Union, get_args, get_origin

from ..logging import log_event
from ..request_context import RequestContext, current_context

if TYPE_CHECKING:
    from .provider import ToolProvider


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
    args = get_args(tp)
    if origin in (list, set, tuple):
        schema: dict = {"type": "array"}
        if args:
            schema["items"] = _type_to_json_schema(args[0])
        return schema
    if origin is dict:
        schema = {"type": "object"}
        if len(args) >= 2:
            schema["additionalProperties"] = _type_to_json_schema(args[1])
        return schema
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _type_to_json_schema(non_none[0])

    return {"type": "string"}


def _check_type(value: object, schema_type: str) -> bool:
    """Check whether a value matches a JSON schema type."""
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "object":
        return isinstance(value, dict)
    return True


@dataclass
class Tool:
    """A tool with inferred name, description, and JSON schema."""

    name: str
    description: str
    inputs: dict
    output_type: str
    fn: Callable
    read_only: bool = False
    concurrency_safe: bool = True
    exclusive: bool = False
    risk_level: str = "low"

    def __call__(self, **kwargs):
        return self.fn(**kwargs)

    def validate_arguments(self, args: dict) -> None:
        """Validate args against the tool's JSON schema.

        Raises ValueError with a descriptive message on mismatch.
        """
        properties = self.inputs.get("properties", {})
        required = self.inputs.get("required", [])

        for param_name in required:
            if param_name not in args:
                raise ValueError(
                    f"Tool '{self.name}' is missing required argument '{param_name}'."
                )

        for param_name, value in args.items():
            if param_name not in properties:
                continue
            schema = properties[param_name]
            schema_type = schema.get("type")
            if schema_type and not _check_type(value, schema_type):
                raise ValueError(
                    f"Tool '{self.name}' argument '{param_name}' must be {schema_type}, "
                    f"got {type(value).__name__}."
                )

    @property
    def metadata(self) -> dict:
        """Return execution metadata including risk level."""
        return {
            "read_only": self.read_only,
            "concurrency_safe": self.concurrency_safe,
            "exclusive": self.exclusive,
            "risk_level": self.risk_level,
        }

    @property
    def required_permission(self) -> str:
        """Map tool flags to the coarse permission mode they require."""
        if self.risk_level == "high":
            return "danger"
        if not self.read_only:
            return "write"
        return "read"


def tool(
    fn: Callable | None = None,
    *,
    name: str | None = None,
    read_only: bool = False,
    concurrency_safe: bool = True,
    exclusive: bool = False,
    risk_level: str = "low",
) -> Tool:
    """Decorator that turns a plain function into a schema-aware Tool."""
    if fn is None:
        return lambda f: tool(  # type: ignore[misc]
            f,
            name=name,
            read_only=read_only,
            concurrency_safe=concurrency_safe,
            exclusive=exclusive,
            risk_level=risk_level,
        )

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
        read_only=read_only,
        concurrency_safe=concurrency_safe,
        exclusive=exclusive,
        risk_level=risk_level,
    )


class ToolRegistry:
    """Registry of tools, with optional pluggable providers and namespaced names."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._providers: dict[str, "ToolProvider"] = {}

    def register_provider(self, name: str, provider: "ToolProvider") -> None:
        """Add a :class:`~swaybot.tools.provider.ToolProvider` under the namespace ``name``."""
        self._providers[name] = provider

    def register(
        self,
        name: str,
        fn: Callable | Tool,
        *,
        provider: str = "local",
        read_only: bool = False,
        concurrency_safe: bool = True,
        exclusive: bool = False,
        risk_level: str = "low",
    ) -> None:
        """Register a callable or an existing Tool under the given name.

        If ``provider`` is not ``"local"`` the tool is stored under
        ``provider/name``.
        """
        key = name if provider == "local" else f"{provider}/{name}"
        if isinstance(fn, Tool):
            self._tools[key] = fn
        else:
            self._tools[key] = tool(
                fn,
                read_only=read_only,
                concurrency_safe=concurrency_safe,
                exclusive=exclusive,
                risk_level=risk_level,
            )

    def _iter_registered(self) -> Iterator[tuple[str, Tool]]:
        """Yield (qualified_name, tool) for local and provider tools."""
        yield from self._tools.items()
        for provider_name, provider in self._providers.items():
            for tool in provider.list_tools():
                yield f"{provider_name}/{tool.name}", tool

    def names(self) -> list[str]:
        return [name for name, _ in self._iter_registered()]

    def get(self, name: str) -> Tool | None:
        for n, tool in self._iter_registered():
            if n == name:
                return tool
        return None

    def describe(self, name: str) -> str:
        """Return a short signature description for a registered tool."""
        t = self.get(name)
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
            for name, t in self._iter_registered()
        ]

    def execute(
        self,
        action: dict,
        request_context: RequestContext | None = None,
    ) -> object:
        name = action.get("name") or ""
        args = action.get("args", {})
        if not name:
            raise ValueError("action is missing a tool name")
        ctx = request_context or current_context()
        if "/" in name:
            provider_name, tool_name = name.split("/", 1)
            provider = self._providers.get(provider_name)
            if provider is not None:
                return provider.execute(tool_name, args, ctx)
        t = self._tools.get(name)
        if t is None:
            raise ValueError(f"Unknown tool: {name}")
        t.validate_arguments(args)
        sig = inspect.signature(t.fn)
        if any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        ):
            result = t.fn(**args)
        else:
            unexpected = set(args) - set(sig.parameters)
            if unexpected:
                raise ValueError(
                    f"Tool '{name}' received unexpected arguments: {sorted(unexpected)}"
                )
            valid_args = {k: v for k, v in args.items() if k in sig.parameters}
            result = t.fn(**valid_args)
        log_event(
            "tool_call",
            tool=name,
            args=args,
            result=str(result),
        )
        return result

    def execute_batch(self, actions: list[dict]) -> list[object]:
        """Execute a list of actions respecting concurrency metadata.

        - Exclusive tools run alone.
        - Non-concurrency-safe tools run sequentially.
        - Otherwise tools run concurrently in a thread pool.
        """
        results: list[object] = [None] * len(actions)
        i = 0
        while i < len(actions):
            name = actions[i].get("name") or ""
            if not name:
                raise ValueError("action is missing a tool name")
            t = self.get(name)
            if t is None:
                raise ValueError(f"Unknown tool: {name}")

            if t.exclusive or not t.concurrency_safe:
                results[i] = self.execute(actions[i])
                i += 1
                continue

            group: list[tuple[int, dict]] = []
            while i < len(actions):
                action = actions[i]
                a_name = action.get("name") or ""
                if not a_name:
                    raise ValueError("action is missing a tool name")
                a_tool = self.get(a_name)
                if a_tool is None:
                    raise ValueError(f"Unknown tool: {a_name}")
                if a_tool.exclusive or not a_tool.concurrency_safe:
                    break
                group.append((i, action))
                i += 1

            with ThreadPoolExecutor() as executor:
                futures = []
                for _, action in group:
                    ctx = contextvars.copy_context()
                    futures.append(
                        executor.submit(ctx.run, self.execute, action)
                    )
            for (idx, _), fut in zip(group, futures):
                results[idx] = fut.result()

        return results


@tool(read_only=True)
def echo(message: str = "") -> str:
    """Echo a message back unchanged."""
    return message


@tool(read_only=True)
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool(read_only=True, exclusive=True)
def done() -> str:
    """Signal that the task is complete."""
    return "finished"


@tool(read_only=True, exclusive=True)
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
    # Imported lazily so that ``swaybot.tools`` does not load the fs/shell/web
    # submodules during its own initialization. fs.py and shell.py do
    # ``from . import tool``, which previously relied on the ordering of
    # bottom-of-file imports in this module (a partial-init cycle).
    from .fs import edit_file, grep, list_directory, read_file, search_files, write_file
    from .shell import run_shell_command
    from .web import web_fetch, web_search

    registry = ToolRegistry()
    registry.register("echo", echo)
    registry.register("add", add)
    registry.register("done", done)
    registry.register("final_answer", final_answer)
    registry.register("web_fetch", web_fetch, read_only=True, risk_level="medium")
    registry.register("web_search", web_search, read_only=True, risk_level="medium")
    registry.register("read_file", read_file, read_only=True, risk_level="low")
    registry.register("write_file", write_file, read_only=False, risk_level="medium")
    registry.register("edit_file", edit_file, read_only=False, risk_level="medium")
    registry.register("list_directory", list_directory, read_only=True, risk_level="low")
    registry.register("search_files", search_files, read_only=True, risk_level="low")
    registry.register("grep", grep, read_only=True, risk_level="low")
    registry.register("run_shell_command", run_shell_command, read_only=False, risk_level="high")
    return registry
