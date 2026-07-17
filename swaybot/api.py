"""OpenAI-compatible HTTP API, WebUI, and session endpoints."""

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .async_agent import AsyncAgent
from .bus import InboundMessage, MessageBus, OutboundMessage
from .scheduler import Scheduler
from .session import SessionManager


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "swaybot"
    messages: list[ChatMessage]
    stream: bool = False


class PostMessageRequest(BaseModel):
    role: str = "user"
    content: str


def create_app(
    bus: MessageBus | None = None,
    session_manager: SessionManager | None = None,
    agent_factory=None,
    max_steps: int = 10,
    scheduler: Scheduler | None = None,
) -> FastAPI:
    bus = bus or MessageBus()
    manager = session_manager or SessionManager()
    scheduler = scheduler or Scheduler()

    if agent_factory is None:
        from .agent import Agent

        def _default_agent_factory(session_id: str) -> Agent:
            return Agent()

        agent_factory = _default_agent_factory

    async_agent = AsyncAgent(
        bus=bus,
        agent_factory=agent_factory,
        max_steps=max_steps,
        session_manager=manager,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await scheduler.start()
        await async_agent.start()
        yield
        await async_agent.stop()
        await scheduler.stop()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest):
        session_id = f"chatcmpl-{uuid.uuid4().hex}"
        content = body.messages[-1].content if body.messages else ""
        if body.stream:
            return StreamingResponse(
                _stream_chat(bus, async_agent, session_id, content, body.model),
                media_type="text/event-stream",
            )
        answer = await _run_one_turn(
            bus, async_agent, session_id, content, timeout=30.0
        )
        created = int(time.time())
        return {
            "id": session_id,
            "object": "chat.completion",
            "created": created,
            "model": body.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    @app.post("/v1/sessions/{session_id}/messages")
    async def post_message(session_id: str, body: PostMessageRequest):
        await async_agent.post(
            InboundMessage(
                role=body.role, content=body.content, session_id=session_id
            )
        )
        return {"status": "accepted", "session_id": session_id}

    @app.get("/v1/sessions/{session_id}/history")
    async def get_history(session_id: str, limit: int | None = None):
        return {
            "session_id": session_id,
            "messages": manager.load(session_id, limit=limit),
        }

    @app.get("/", response_class=HTMLResponse)
    async def webui():
        return HTMLResponse(content=_WEBUI_HTML)

    return app


async def _stream_chat(
    bus: MessageBus,
    async_agent: AsyncAgent,
    session_id: str,
    content: str,
    model: str,
):
    """Stream assistant output chunks as SSE."""
    queue: asyncio.Queue[OutboundMessage] = asyncio.Queue()

    def on_message(msg: OutboundMessage) -> None:
        if msg.session_id == session_id:
            queue.put_nowait(msg)

    bus.subscribe(on_message)
    await async_agent.post(
        InboundMessage(role="user", content=content, session_id=session_id)
    )

    created = int(time.time())
    sent_role = False
    while True:
        try:
            msg = await asyncio.wait_for(queue.get(), timeout=10.0)
        except asyncio.TimeoutError:
            break
        if not sent_role:
            chunk = {
                "id": session_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            sent_role = True
        chunk = {
            "id": session_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": msg.content},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    final = {
        "id": session_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


async def _run_one_turn(
    bus: MessageBus,
    async_agent: AsyncAgent,
    session_id: str,
    content: str,
    timeout: float = 30.0,
) -> str:
    """Post a message and wait for the first outbound response."""
    answer_event = asyncio.Event()
    last_answer: list[str] = [""]

    def on_message(msg: OutboundMessage) -> None:
        if msg.session_id == session_id:
            last_answer[0] = msg.content
            answer_event.set()

    bus.subscribe(on_message)
    await async_agent.post(
        InboundMessage(role="user", content=content, session_id=session_id)
    )
    try:
        await asyncio.wait_for(answer_event.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        pass
    return last_answer[0]


_WEBUI_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SwayBot WebUI</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; }
    #history { border: 1px solid #ccc; border-radius: 8px; min-height: 200px; padding: 1rem; margin-bottom: 1rem; }
    .msg { margin: 0.5rem 0; }
    .user { color: #0066cc; }
    .assistant { color: #228822; }
    textarea { width: 100%; height: 80px; }
    button { padding: 0.5rem 1rem; }
  </style>
</head>
<body>
  <h1>SwayBot WebUI</h1>
  <label>Session ID: <input id="sessionId" value="demo" /></label>
  <div id="history"></div>
  <textarea id="input" placeholder="Type a message..."></textarea><br/>
  <button id="send">Send</button>
  <script>
    const sessionId = () => document.getElementById('sessionId').value || 'demo';
    async function loadHistory() {
      const res = await fetch(`/v1/sessions/${sessionId()}/history`);
      const data = await res.json();
      const box = document.getElementById('history');
      box.innerHTML = data.messages.map(m =>
        `<div class="msg ${m.role}"><b>${m.role}:</b> ${escapeHtml(m.content)}</div>`
      ).join('');
    }
    function escapeHtml(text) {
      return text.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;"'"'"'":'&#39;'}[c]));
    }
    document.getElementById('send').onclick = async () => {
      const input = document.getElementById('input');
      await fetch(`/v1/sessions/${sessionId()}/messages`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({role: 'user', content: input.value})
      });
      input.value = '';
      setTimeout(loadHistory, 500);
    };
    setInterval(loadHistory, 2000);
    loadHistory();
  </script>
</body>
</html>
"""


def main():
    """CLI entrypoint: uvicorn swaybot.api:app"""
    import uvicorn

    app = create_app()
    uvicorn.run(app, host="127.0.0.1", port=8000)
