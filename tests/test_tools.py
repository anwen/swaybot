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
    assert names == {
        "echo",
        "add",
        "done",
        "final_answer",
        "web_fetch",
        "web_search",
        "read_file",
        "write_file",
        "edit_file",
        "list_directory",
        "search_files",
        "grep",
        "run_shell_command",
    }
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


def test_tool_metadata_flags():
    registry = build_default_registry()
    assert registry.get("add").read_only
    assert registry.get("add").concurrency_safe
    assert registry.get("done").exclusive
    assert registry.get("final_answer").exclusive


def test_execute_batch_runs_read_only_tools_concurrently():
    registry = ToolRegistry()
    calls = []

    def slow(x: int) -> int:
        import time

        time.sleep(0.05)
        calls.append(x)
        return x * 2

    registry.register("slow", slow, read_only=True, concurrency_safe=True)
    import time

    start = time.time()
    results = registry.execute_batch(
        [{"name": "slow", "args": {"x": 1}}, {"name": "slow", "args": {"x": 2}}]
    )
    elapsed = time.time() - start
    assert set(results) == {2, 4}
    assert elapsed < 0.09


def test_execute_batch_runs_exclusive_tools_sequentially():
    registry = ToolRegistry()
    state = []

    @tool(exclusive=True)
    def append(value: int) -> int:
        state.append(value)
        return value

    registry.register("append", append)
    results = registry.execute_batch(
        [{"name": "append", "args": {"value": 1}}, {"name": "append", "args": {"value": 2}}]
    )
    assert results == [1, 2]
    assert state == [1, 2]


def test_execute_batch_mixed_grouping():
    registry = ToolRegistry()

    @tool(read_only=True, concurrency_safe=True)
    def double(x: int) -> int:
        return x * 2

    @tool(exclusive=True)
    def finalize() -> str:
        return "done"

    registry.register("double", double)
    registry.register("finalize", finalize)

    actions = [
        {"name": "double", "args": {"x": 1}},
        {"name": "finalize", "args": {}},
        {"name": "double", "args": {"x": 2}},
    ]
    assert registry.execute_batch(actions) == [2, "done", 4]
