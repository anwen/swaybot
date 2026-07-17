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
from .coordinator import GoalCoordinator
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


class CreateGoalRequest(BaseModel):
    description: str
    context: str = ""


class GoalResponse(BaseModel):
    id: str
    description: str
    context: str
    state: str
    subtasks: list[dict[str, Any]]
    created_at: str
    updated_at: str


def create_app(
    bus: MessageBus | None = None,
    session_manager: SessionManager | None = None,
    agent_factory=None,
    max_steps: int = 10,
    scheduler: Scheduler | None = None,
    goal_coordinator: GoalCoordinator | None = None,
) -> FastAPI:
    bus = bus or MessageBus()
    manager = session_manager or SessionManager()
    if scheduler is None:
        scheduler = Scheduler()
    assert scheduler is not None

    if agent_factory is None:
        from .agent import Agent

        def _default_agent_factory(session_id: str = "") -> Agent:
            return Agent()

        agent_factory = _default_agent_factory

    coordinator = goal_coordinator or GoalCoordinator(
        agent_factory=agent_factory,
        max_subtask_steps=max_steps,
    )

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

    @app.post("/v1/goals")
    async def create_goal(body: CreateGoalRequest):
        goal = await coordinator.arun(body.description, body.context)
        return GoalResponse(
            id=goal.id,
            description=goal.description,
            context=goal.context,
            state=goal.state,
            subtasks=[{"id": s.id, "description": s.description, "status": s.status, "result": s.result} for s in goal.subtasks],
            created_at=goal.created_at,
            updated_at=goal.updated_at,
        )

    @app.get("/v1/goals")
    async def list_goals():
        return [
            GoalResponse(
                id=g.id,
                description=g.description,
                context=g.context,
                state=g.state,
                subtasks=[{"id": s.id, "description": s.description, "status": s.status, "result": s.result} for s in g.subtasks],
                created_at=g.created_at,
                updated_at=g.updated_at,
            )
            for g in coordinator.state_machine.list_goals()
        ]

    @app.get("/v1/goals/{goal_id}")
    async def get_goal(goal_id: str):
        try:
            goal = coordinator.state_machine.load(goal_id)
        except KeyError:
            return JSONResponse({"error": "not found"}, status_code=404)
        return GoalResponse(
            id=goal.id,
            description=goal.description,
            context=goal.context,
            state=goal.state,
            subtasks=[{"id": s.id, "description": s.description, "status": s.status, "result": s.result} for s in goal.subtasks],
            created_at=goal.created_at,
            updated_at=goal.updated_at,
        )

    @app.post("/v1/goals/{goal_id}/retry")
    async def retry_goal(goal_id: str):
        goal = await coordinator.retry(goal_id)
        return GoalResponse(
            id=goal.id,
            description=goal.description,
            context=goal.context,
            state=goal.state,
            subtasks=[{"id": s.id, "description": s.description, "status": s.status, "result": s.result} for s in goal.subtasks],
            created_at=goal.created_at,
            updated_at=goal.updated_at,
        )

    @app.delete("/v1/goals/{goal_id}")
    async def cancel_goal(goal_id: str):
        goal = await coordinator.cancel(goal_id)
        return GoalResponse(
            id=goal.id,
            description=goal.description,
            context=goal.context,
            state=goal.state,
            subtasks=[{"id": s.id, "description": s.description, "status": s.status, "result": s.result} for s in goal.subtasks],
            created_at=goal.created_at,
            updated_at=goal.updated_at,
        )

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
