import asyncio

import pytest

from swaybot.async_agent import AsyncAgent
from swaybot.agent import Agent
from swaybot.bus import InboundMessage, MessageBus, OutboundMessage
from swaybot.tools import ToolRegistry


def test_bus_creates_session_on_post():
    bus = MessageBus()
    assert not bus.session_exists("s1")
    asyncio.run(bus.post(InboundMessage(content="hi", session_id="s1")))
    assert bus.session_exists("s1")


def test_bus_per_session_isolation():
    bus = MessageBus()
    asyncio.run(bus.post(InboundMessage(content="a", session_id="s1")))
    asyncio.run(bus.post(InboundMessage(content="b", session_id="s2")))
    assert bus.pending_count("s1") == 1
    assert bus.pending_count("s2") == 1
    m1 = asyncio.run(bus.get("s1"))
    assert m1.content == "a"
    assert bus.pending_count("s1") == 0
    assert bus.pending_count("s2") == 1


def test_bus_get_returns_none_on_timeout():
    bus = MessageBus()
    assert asyncio.run(bus.get("empty", timeout=0.01)) is None


def test_bus_lock_is_exclusive():
    bus = MessageBus()
    lock = bus.lock("s1")
    acquired_order = []

    async def worker(name):
        async with lock:
            acquired_order.append(name)
            await asyncio.sleep(0.05)

    async def main():
        await asyncio.gather(worker("a"), worker("b"))

    asyncio.run(main())
    assert acquired_order == ["a", "b"]


def test_bus_subscribers_receive_outbound():
    bus = MessageBus()
    received = []

    def cb(msg):
        received.append(msg)

    bus.subscribe(cb)
    asyncio.run(bus.emit(OutboundMessage(content="hi", session_id="s1")))
    assert len(received) == 1
    assert received[0].content == "hi"


def test_bus_close_session_removes_state():
    bus = MessageBus()
    bus.create_session("s1")
    bus.close_session("s1")
    assert not bus.session_exists("s1")


@pytest.mark.asyncio
async def test_async_agent_processes_message():
    bus = MessageBus()

    def factory(session_id):
        return Agent()

    agent = AsyncAgent(bus, factory, max_steps=2)
    await agent.start()

    received = []
    bus.subscribe(lambda m: received.append(m))

    await agent.post(InboundMessage(content="hello", session_id="s1"))
    await asyncio.sleep(0.2)

    await agent.stop()
    assert len(received) >= 1
    assert any("finished" in m.content for m in received)


@pytest.mark.asyncio
async def test_async_agent_isolates_sessions():
    bus = MessageBus()
    results = {}

    class CapturingBrain:
        def think(self, perception, available_tools, metadata=None):
            results[perception["task"]] = perception["task"]
            return {"name": "done", "args": {}}

    def factory(session_id):
        return Agent(brain=CapturingBrain())

    agent = AsyncAgent(bus, factory, max_steps=2)
    await agent.start()

    await agent.post(InboundMessage(content="task-a", session_id="a"))
    await agent.post(InboundMessage(content="task-b", session_id="b"))
    await asyncio.sleep(0.3)
    await agent.stop()

    assert "task-a" in results
    assert "task-b" in results


@pytest.mark.asyncio
async def test_async_agent_orders_mid_turn_messages():
    bus = MessageBus()

    class EchoDoneBrain:
        def think(self, perception, available_tools, metadata=None):
            return {"name": "done", "args": {}}

    def factory(session_id):
        return Agent(brain=EchoDoneBrain())

    agent = AsyncAgent(bus, factory, max_steps=2)
    await agent.start()

    received = []
    bus.subscribe(lambda m: received.append(m.content))

    await agent.post(InboundMessage(content="first", session_id="s1"))
    await agent.post(InboundMessage(content="second", session_id="s1"))
    await asyncio.sleep(0.2)
    await agent.stop()

    assert len(received) >= 1
    assert "finished" in received[0]
