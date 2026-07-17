import time

from fastapi.testclient import TestClient

from swaybot.agent import Agent
from swaybot.api import create_app
from swaybot.bus import MessageBus
from swaybot.session import SessionManager


def test_health():
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_webui_root():
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "SwayBot WebUI" in resp.text


def test_post_message_and_history(tmp_path):
    manager = SessionManager(base_dir=tmp_path)
    bus = MessageBus()

    class DoneBrain:
        def think(self, perception, available_tools, metadata=None):
            return {"name": "done", "args": {}}

    app = create_app(
        bus=bus,
        session_manager=manager,
        agent_factory=lambda session_id: Agent(brain=DoneBrain()),
        max_steps=2,
    )
    with TestClient(app) as client:
        resp = client.post(
            "/v1/sessions/s1/messages",
            json={"role": "user", "content": "hello"},
        )
        assert resp.status_code == 200

        time.sleep(0.3)
        for _ in range(20):
            hist = client.get("/v1/sessions/s1/history").json()
            if any(m["role"] == "assistant" for m in hist["messages"]):
                break
            time.sleep(0.05)
        assert any(m["role"] == "user" and m["content"] == "hello" for m in hist["messages"])
        assert any(m["role"] == "assistant" for m in hist["messages"])


def test_chat_completions(tmp_path):
    manager = SessionManager(base_dir=tmp_path)
    bus = MessageBus()

    class DoneBrain:
        def think(self, perception, available_tools, metadata=None):
            return {"name": "done", "args": {}}

    app = create_app(
        bus=bus,
        session_manager=manager,
        agent_factory=lambda session_id: Agent(brain=DoneBrain()),
        max_steps=2,
    )
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "swaybot",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"


def test_chat_completions_streaming(tmp_path):
    manager = SessionManager(base_dir=tmp_path)
    bus = MessageBus()

    class DoneBrain:
        def think(self, perception, available_tools, metadata=None):
            return {"name": "done", "args": {}}

    app = create_app(
        bus=bus,
        session_manager=manager,
        agent_factory=lambda session_id: Agent(brain=DoneBrain()),
        max_steps=2,
    )
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "swaybot",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        assert b"data:" in resp.content
