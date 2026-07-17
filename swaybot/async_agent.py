"""Async agent loop consuming the message bus."""

import asyncio
from typing import Callable

from .agent import Agent
from .bus import InboundMessage, MessageBus, OutboundMessage
from .session import SessionManager


class AsyncAgent:
    """Run a sync Agent inside an async, per-session message bus consumer.

    Each session has its own memory/agent instance (created by ``agent_factory``)
    and is protected by a per-session lock so messages posted during a turn are
    queued and processed in order. If ``session_manager`` is provided, all
    inbound and outbound messages are persisted as JSONL.
    """

    def __init__(
        self,
        bus: MessageBus,
        agent_factory: Callable[[str], Agent],
        max_steps: int = 10,
        max_wait_between_messages: float = 0.05,
        session_manager: SessionManager | None = None,
    ) -> None:
        self.bus = bus
        self.agent_factory = agent_factory
        self.max_steps = max_steps
        self.max_wait_between_messages = max_wait_between_messages
        self.session_manager = session_manager
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._running = False

    async def start(self) -> None:
        """Begin accepting messages for all known sessions."""
        self._running = True

    async def stop(self) -> None:
        """Stop accepting new messages and cancel active session loops."""
        self._running = False
        for task in self._tasks.values():
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()

    async def run_session(self, session_id: str) -> None:
        """Consume inbound messages for ``session_id`` until the bus closes."""
        agent = self.agent_factory(session_id)
        self.bus.create_session(session_id)
        while self._running:
            message = await self.bus.get(session_id, timeout=0.5)
            if message is None:
                continue

            async with self.bus.lock(session_id):
                parts = [message.content]
                if self.session_manager is not None:
                    self.session_manager.append(session_id, message)
                while True:
                    extra = await self.bus.get(
                        session_id, timeout=self.max_wait_between_messages
                    )
                    if extra is None:
                        break
                    parts.append(extra.content)
                    if self.session_manager is not None:
                        self.session_manager.append(session_id, extra)
                task_text = "\n".join(parts).strip()
                if not task_text:
                    continue

                loop = asyncio.get_running_loop()
                env = await loop.run_in_executor(
                    None,
                    lambda: agent.run(
                        task_text,
                        max_steps=self.max_steps,
                        reflect=False,
                    ),
                )
                for record in env.history:
                    outbound = OutboundMessage(
                        role="assistant",
                        content=str(record["result"]),
                        session_id=session_id,
                    )
                    if self.session_manager is not None:
                        self.session_manager.append(session_id, outbound)
                    await self.bus.emit(outbound)

    def ensure_session(self, session_id: str) -> None:
        """Start a background consumer for ``session_id`` if none exists."""
        if session_id not in self._tasks or self._tasks[session_id].done():
            self.bus.create_session(session_id)
            self._tasks[session_id] = asyncio.create_task(
                self.run_session(session_id)
            )

    async def post(self, message: InboundMessage) -> None:
        """Post a message and make sure its session consumer is running."""
        self.ensure_session(message.session_id)
        await self.bus.post(message)
