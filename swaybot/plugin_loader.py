"""Load tools from plugin directories or modules at runtime."""

import importlib
import importlib.util
import inspect
from pathlib import Path
from types import ModuleType

from .tools import Tool, ToolRegistry, tool


class PluginLoader:
    """Discover and register tools from external plugins.

    Loading plugins from arbitrary directories executes the plugin code in the
    current process. To prevent accidental code execution, directory loading is
    only allowed for directories listed in ``trusted_directories``.
    """

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        trusted_directories: list[Path | str] | None = None,
    ) -> None:
        self.registry = registry or ToolRegistry()
        self.trusted_directories = [
            Path(d).expanduser().resolve()
            for d in (trusted_directories or [])
        ]

    def load_directory(self, path: Path | str) -> list[str]:
        """Import every ``*.py`` file in ``path`` and register its tools."""
        directory = Path(path).expanduser().resolve()
        if not any(directory.is_relative_to(trusted) for trusted in self.trusted_directories):
            raise ValueError(
                f"refusing to load plugins from untrusted directory: {directory}"
            )
        loaded: list[str] = []
        if not directory.exists():
            return loaded
        for file_path in sorted(directory.glob("*.py")):
            if file_path.name.startswith("_"):
                continue
            module = self._load_module_from_path(file_path)
            loaded.extend(self._register_module_tools(module))
        return loaded

    def load_module(self, module_name: str) -> list[str]:
        """Import ``module_name`` and register its tools."""
        module = importlib.import_module(module_name)
        return self._register_module_tools(module)

    def _load_module_from_path(self, file_path: Path) -> ModuleType:
        module_name = f"_swaybot_plugin_{file_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load plugin {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _register_module_tools(self, module: ModuleType) -> list[str]:
        loaded: list[str] = []
        for _name, value in inspect.getmembers(module):
            if isinstance(value, Tool):
                self.registry.register(value.name, value)
                loaded.append(value.name)
            elif inspect.isfunction(value) and value.__module__ == module.__name__:
                wrapped = tool(value)
                if wrapped.description or wrapped.inputs.get("properties"):
                    self.registry.register(wrapped.name, wrapped)
                    loaded.append(wrapped.name)
        return loaded


def load_plugins(
    directories: list[Path | str] | None = None,
    modules: list[str] | None = None,
    registry: ToolRegistry | None = None,
    trusted_directories: list[Path | str] | None = None,
) -> ToolRegistry:
    """Convenience helper to load tools from directories and/or modules.

    ``directories`` must be covered by ``trusted_directories``; if the latter
    is omitted, directory loading is refused.
    """
    loader = PluginLoader(
        registry=registry, trusted_directories=trusted_directories
    )
    for directory in directories or []:
        loader.load_directory(directory)
    for module in modules or []:
        loader.load_module(module)
    return loader.registry
