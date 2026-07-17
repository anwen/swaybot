"""Model abstraction for text generation backends."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class ToolCall:
    """A native tool call returned by a model."""

    id: str
    name: str
    arguments: dict


@dataclass
class ModelResponse:
    """Structured response from a text generation backend."""

    content: str = ""
    tool_calls: list[ToolCall] | None = None


@runtime_checkable
class Model(Protocol):
    """A text generation backend that turns messages into text or tool calls."""

    def generate(
        self,
        messages: list[dict],
        metadata: dict | None = None,
        tools: list[dict] | None = None,
    ) -> str | None:
        """Return the model's text output, or None on failure.

        Implementations that support native tool calls may return a JSON
        string encoding ``{"name": ..., "args": ...}`` when a tool is
        chosen, or expose ``supports_tools = True`` and return a richer
        representation.
        """
        ...


class FallbackModel:
    """Try a chain of models until one succeeds."""

    def __init__(self, models: list[Model]) -> None:
        self.models = models

    def generate(
        self,
        messages: list[dict],
        metadata: dict | None = None,
        tools: list[dict] | None = None,
    ) -> str | None:
        last_error = ""
        for model in self.models:
            try:
                result = model.generate(messages, metadata=metadata, tools=tools)
                if result is not None:
                    return result
            except Exception as exc:  # pragma: no cover - defensive
                last_error = str(exc)
        if metadata is not None and last_error:
            metadata["error"] = f"all fallback models failed: {last_error}"
        return None
