import pytest

from swaybot.tools import Tool, ToolRegistry, build_default_registry, tool


def test_tool_decorator_infers_schema():
    @tool
    def get_weather(location: str, celsius: bool = False) -> str:
        """Get the weather for a location."""
        return "sunny"

    assert isinstance(get_weather, Tool)
    assert get_weather.name == "get_weather"
    assert "Get the weather" in get_weather.description
    assert get_weather.output_type == "string"
    props = get_weather.inputs["properties"]
    assert props["location"] == {"type": "string"}
    assert props["celsius"] == {"type": "boolean", "default": False}
    assert get_weather.inputs["required"] == ["location"]


def test_tool_decorator_with_name_override():
    @tool(name="sum_two")
    def add(a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    assert add.name == "sum_two"


def test_tool_registry_accepts_callable_and_tool():
    registry = ToolRegistry()

    @tool
    def multiply(x: float, y: float) -> float:
        """Multiply."""
        return x * y

    registry.register("multiply", multiply)
    registry.register("add", lambda a, b: a + b)

    assert "multiply" in registry.names()
    assert "add" in registry.names()
    assert isinstance(registry.get("multiply"), Tool)
    assert isinstance(registry.get("add"), Tool)


def test_tool_registry_schemas():
    registry = build_default_registry()
    schemas = registry.schemas()
    names = {s["name"] for s in schemas}
    assert names == {"echo", "add", "done", "final_answer"}
    echo_schema = next(s for s in schemas if s["name"] == "echo")
    assert echo_schema["description"] == "Echo a message back unchanged."
    assert echo_schema["parameters"]["properties"]["message"] == {
        "type": "string",
        "default": "",
    }


def test_tool_registry_execute():
    registry = build_default_registry()
    assert registry.execute({"name": "add", "args": {"a": 1, "b": 2}}) == 3
    assert registry.execute({"name": "echo", "args": {"message": "hi"}}) == "hi"
    assert registry.execute({"name": "done", "args": {}}) == "finished"
    assert (
        registry.execute({"name": "final_answer", "args": {"answer": "42"}}) == "42"
    )


def test_tool_registry_execute_rejects_missing_required_argument():
    registry = build_default_registry()
    with pytest.raises(ValueError, match="missing required argument 'a'"):
        registry.execute({"name": "add", "args": {"b": 2}})


def test_tool_registry_execute_rejects_wrong_type():
    registry = build_default_registry()
    with pytest.raises(ValueError, match="must be number"):
        registry.execute({"name": "add", "args": {"a": "one", "b": 2}})
