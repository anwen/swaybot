from swaybot.agent import Agent
from swaybot.context import ContextBuilder
from swaybot.memory import Memory, MemoryStore
from swaybot.tools import build_default_registry


def test_context_builder_includes_tool_context():
    tools = build_default_registry()
    builder = ContextBuilder(memory=None, tools=tools)
    perception = builder.build("demo", 0, 3, [])
    assert "echo" in perception["available_tools"]
    assert any(item["name"] == "echo" for item in perception["tool_descriptions"])


def test_context_builder_includes_memory_context_and_messages():
    store = MemoryStore()
    store.add(
        Memory(
            content="demo long term fact",
            scope="long_term",
            tags=["demo"],
        )
    )
    store.add(
        Memory(
            content="short term note",
            scope="short_term",
            tags=["demo"],
        )
    )
    tools = build_default_registry()
    builder = ContextBuilder(memory=store, tools=tools)
    perception = builder.build("demo", 1, 3, [])
    assert "long term fact" in perception["memory_context"]
    assert any(
        "short term note" in msg["content"] for msg in perception["messages"]
    )


def test_agent_uses_context_builder():
    agent = Agent()
    assert isinstance(agent.context_builder, ContextBuilder)
