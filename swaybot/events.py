"""Event bus for decoupled, observable agent interactions."""

import inspect
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Protocol, runtime_checkable


@dataclass
class Event:
    """A typed event on the bus."""

    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    context: dict[str, Any] | None = None


@runtime_checkable
class EventBus(Protocol):
    """Publish/subscribe event bus abstraction."""

    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribers."""
        ...

    def subscribe(
        self,
        event_type: str,
        handler: Callable[[Event], Any],
    ) -> None:
        """Subscribe ``handler`` to events of ``event_type``.

        Use ``event_type="*"`` to receive all events.
        """
        ...

    async def start(self) -> None:
        """Optional lifecycle hook."""
        ...

    async def stop(self) -> None:
        """Optional lifecycle hook."""
        ...


class InMemoryEventBus:
    """Process-local async event bus."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[[Event], Any]]] = {}

    def subscribe(
        self,
        event_type: str,
        handler: Callable[[Event], Any],
    ) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    async def publish(self, event: Event) -> None:
        handlers = (
            self._handlers.get(event.type, [])
            + self._handlers.get("*", [])
        )
        for handler in handlers:
            try:
                if inspect.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception:  # pragma: no cover
                # Subscribers must not break the bus.
                pass

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass
