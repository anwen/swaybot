import pytest

from swaybot.events import Event, InMemoryEventBus


@pytest.mark.asyncio
async def test_event_bus_delivers_to_subscribers():
    bus = InMemoryEventBus()
    received = []

    def handler(event: Event) -> None:
        received.append(event)

    bus.subscribe("tool_call", handler)
    await bus.publish(Event(type="tool_call", payload={"name": "add"}))
    assert len(received) == 1
    assert received[0].payload["name"] == "add"


@pytest.mark.asyncio
async def test_event_bus_delivers_to_async_handlers():
    bus = InMemoryEventBus()
    received = []

    async def handler(event: Event) -> None:
        received.append(event)

    bus.subscribe("model_call", handler)
    await bus.publish(Event(type="model_call", payload={"model": "gpt"}))
    assert len(received) == 1


@pytest.mark.asyncio
async def test_event_bus_wildcard_receives_all_events():
    bus = InMemoryEventBus()
    received = []

    def handler(event: Event) -> None:
        received.append(event.type)

    bus.subscribe("*", handler)
    await bus.publish(Event(type="a", payload={}))
    await bus.publish(Event(type="b", payload={}))
    assert received == ["a", "b"]


@pytest.mark.asyncio
async def test_event_bus_unsubscribe_removes_handler():
    bus = InMemoryEventBus()
    received = []

    def handler(event: Event) -> None:
        received.append(event)

    unsubscribe = bus.subscribe("x", handler)
    unsubscribe()
    await bus.publish(Event(type="x", payload={}))
    assert len(received) == 0
