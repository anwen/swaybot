from swaybot.plugin_loader import PluginLoader, load_plugins
from swaybot.tools import ToolRegistry


def test_load_directory_registers_tool(tmp_path):
    plugin = tmp_path / "my_plugin.py"
    plugin.write_text(
        "from swaybot.tools import tool\n"
        "@tool\n"
        "def greet(name: str) -> str:\n"
        "    '''Greet someone.'''\n"
        "    return f'Hello, {name}'\n"
    )
    loader = PluginLoader()
    loaded = loader.load_directory(tmp_path)
    assert "greet" in loaded
    assert loader.registry.execute({"name": "greet", "args": {"name": "Ada"}}) == "Hello, Ada"


def test_load_directory_skips_private_files(tmp_path):
    private = tmp_path / "_hidden.py"
    private.write_text(
        "from swaybot.tools import tool\n"
        "@tool\n"
        "def hidden() -> str:\n"
        "    return 'hidden'\n"
    )
    loader = PluginLoader()
    loaded = loader.load_directory(tmp_path)
    assert "hidden" not in loaded


def test_load_module_registers_tool():
    loader = PluginLoader()
    loaded = loader.load_module("swaybot.tools.web")
    assert "web_fetch" in loaded or "web_search" in loaded


def test_load_plugins_returns_registry(tmp_path):
    plugin = tmp_path / "p.py"
    plugin.write_text(
        "from swaybot.tools import tool\n"
        "@tool\n"
        "def ping() -> str:\n"
        "    return 'pong'\n"
    )
    registry = load_plugins(directories=[tmp_path])
    assert "ping" in registry.names()


def test_load_module_plain_function_wrapped():
    loader = PluginLoader()
    loaded = loader.load_module("swaybot.tools")
    assert "echo" in loaded
    assert "add" in loaded
