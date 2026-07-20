from swaybot.tools import ToolRegistry, build_default_registry, tool
from swaybot.tools.provider import ToolProvider


class UpperProvider(ToolProvider):
    """Tiny provider for testing namespaced dispatch."""

    def list_tools(self):
        @tool
        def shout(message: str) -> str:
            """Shout a message."""
            return message.upper()

        return [shout]

    def execute(self, name, args, context=None):
        if name == "shout":
            return args.get("message", "").upper()
        raise ValueError(f"Unknown tool: {name}")

    def close(self):
        pass


def test_registry_registers_provider_namespace():
    registry = ToolRegistry()
    registry.register_provider("upper", UpperProvider())
    assert "upper/shout" in registry.names()
    assert registry.execute({"name": "upper/shout", "args": {"message": "hello"}}) == "HELLO"


def test_registry_register_with_provider_namespace():
    registry = ToolRegistry()

    @tool
    def greet(name: str) -> str:
        """Greet someone."""
        return f"hello {name}"

    registry.register("greet", greet, provider="hello")
    assert "hello/greet" in registry.names()
    assert registry.execute({"name": "hello/greet", "args": {"name": "world"}}) == "hello world"
