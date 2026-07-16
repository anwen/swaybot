"""Model abstraction for text generation backends."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Model(Protocol):
    """A text generation backend that turns messages into text."""

    def generate(
        self,
        messages: list[dict],
        metadata: dict | None = None,
    ) -> str | None:
        """Return the model's text output, or None on failure."""
        ...


class FallbackModel:
    """Try a chain of models until one succeeds."""

    def __init__(self, models: list[Model]) -> None:
        self.models = models

    def generate(
        self,
        messages: list[dict],
        metadata: dict | None = None,
    ) -> str | None:
        last_error = ""
        for model in self.models:
            try:
                result = model.generate(messages, metadata=metadata)
                if result is not None:
                    return result
            except Exception as exc:  # pragma: no cover - defensive
                last_error = str(exc)
        if metadata is not None and last_error:
            metadata["error"] = f"all fallback models failed: {last_error}"
        return None
