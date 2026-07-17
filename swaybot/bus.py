"""Async message bus for per-session agent events."""

import asyncio
import inspect
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable


@dataclass
class InboundMessage:
    """A message posted into an agent session."""

    role: str = "user"
    content: str = ""
    session_id: str = ""
    posted_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class OutboundMessage:
    """A message produced by the agent for a session."""

    role: str = "assistant"
    content: str = ""
    session_id: str = ""
    posted_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class MessageBus:
    """In-memory async message bus with per-session queues and locks."""

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[InboundMessage]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._subscribers: list[Callable[[OutboundMessage], None]] = []

    def create_session(self, session_id: str) -> None:
        """Allocate a queue and lock for ``session_id``."""
        if session_id not in self._queues:
            self._queues[session_id] = asyncio.Queue()
            self._locks[session_id] = asyncio.Lock()

    def session_exists(self, session_id: str) -> bool:
        return session_id in self._queues

    def lock(self, session_id: str) -> asyncio.Lock:
        """Return the per-session lock, creating the session if needed."""
        self.create_session(session_id)
        return self._locks[session_id]

    async def post(self, message: InboundMessage) -> None:
        """Post a message to a session queue."""
        self.create_session(message.session_id)
        await self._queues[message.session_id].put(message)

    async def get(self, session_id: str, timeout: float | None = None) -> InboundMessage | None:
        """Get the next message for ``session_id``; return None on timeout."""
        self.create_session(session_id)
        try:
            return await asyncio.wait_for(
                self._queues[session_id].get(), timeout=timeout
            )
        except asyncio.TimeoutError:
            return None

    def subscribe(self, callback: Callable[[OutboundMessage], None]) -> None:
        """Register a callback for outbound messages."""
        self._subscribers.append(callback)

    async def emit(self, message: OutboundMessage) -> None:
        """Notify all subscribers of an outbound message."""
        for callback in self._subscribers:
            try:
                if inspect.iscoroutinefunction(callback):
                    await callback(message)
                else:
                    callback(message)
            except Exception:  # pragma: no cover
                pass

    def close_session(self, session_id: str) -> None:
        """Remove a session's queue and lock."""
        self._queues.pop(session_id, None)
        self._locks.pop(session_id, None)

    def pending_count(self, session_id: str) -> int:
        """Return the number of queued messages for ``session_id``."""
        self.create_session(session_id)
        return self._queues[session_id].qsize()
