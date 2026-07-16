import sys
from pathlib import Path

import pytest

from swaybot.mcp_client import McpClient, register_mcp_tools
from swaybot.tools import ToolRegistry


FAKE_SERVER = [sys.executable, str(Path(__file__).with_name("fake_mcp_server.py"))]


@pytest.fixture
def mcp_client():
    client = McpClient(FAKE_SERVER).start()
    yield client
    client.close()


def test_mcp_client_lists_tools(mcp_client):
    tools = mcp_client.list_tools()
    assert len(tools) == 1
    assert tools[0]["name"] == "greet"


def test_mcp_client_calls_tool(mcp_client):
    result = mcp_client.call_tool("greet", {"name": "world"})
    assert result == "Hello, world!"


def test_register_mcp_tools_adds_tool_to_registry():
    registry = ToolRegistry()
    client = register_mcp_tools(registry, FAKE_SERVER)
    try:
        assert "greet" in registry.names()
        result = registry.execute({"name": "greet", "args": {"name": "test"}})
        assert result == "Hello, test!"
    finally:
        client.close()
