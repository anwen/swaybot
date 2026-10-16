"""OpenAI-compatible HTTP API, WebUI, and session endpoints."""

import asyncio
from functools import lru_cache
import json
import logging
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Header, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .agent import Agent
from .async_agent import AsyncAgent
from .bus import InboundMessage, MessageBus, OutboundMessage
from .request_context import RequestContext, current_context, set_context
from .goals import Goal, GoalCoordinator, Scheduler
from .dotenv import load_dotenv
from .logging import JSONFormatter, get_event_logger

from .permission import PermissionPolicy, permission_level_to_mode
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
    agent_factory: Callable[[str], Agent] | None = None,
    max_steps: int = 10,
    scheduler: Scheduler | None = None,
    goal_coordinator: GoalCoordinator | None = None,
) -> FastAPI:
    bus = bus or MessageBus()
    manager = session_manager or SessionManager()
    if scheduler is None:
        scheduler = Scheduler()

    if agent_factory is None:
        def _default_agent_factory(session_id: str = "") -> Agent:
            return Agent(
                permission_policy=PermissionPolicy(
                    mode=permission_level_to_mode(current_context().permission_level)
                )
            )

        agent_factory = _default_agent_factory

    coordinator = goal_coordinator or GoalCoordinator(
        agent_factory=agent_factory,
        max_subtask_steps=max_steps,
        scheduler=scheduler,
        event_bus=bus.event_bus,
    )

    async_agent = AsyncAgent(
        bus=bus,
        agent_factory=agent_factory,
        max_steps=max_steps,
        session_manager=manager,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await bus.event_bus.start()
        await scheduler.start()
        await async_agent.start()
        yield
        await async_agent.stop()
        await scheduler.stop()
        await bus.event_bus.stop()

    app = FastAPI(lifespan=lifespan)

    # Configure structured event logging to stderr by default.
    event_logger = get_event_logger()
    if not event_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        event_logger.addHandler(handler)
        event_logger.setLevel(logging.INFO)

    @app.middleware("http")
    async def request_context_middleware(
        request: Request, call_next: Callable[[Request], Any]
    ):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        api_key = request.headers.get("X-API-Key", "")
        admin_key = os.environ.get("SWAYBOT_ADMIN_API_KEY")
        if admin_key and secrets.compare_digest(api_key, admin_key):
            principal = "admin"
            permission_level = "high"
        else:
            principal = "anonymous"
            permission_level = "medium"
        ctx = RequestContext(
            request_id=request_id,
            principal=principal,
            permission_level=permission_level,
        )
        with set_context(ctx):
            response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    def _session_context(session_id: str) -> RequestContext:
        base = current_context()
        return RequestContext(
            session_id=session_id,
            request_id=base.request_id,
            principal=base.principal,
            permission_level=base.permission_level,
            metadata=base.metadata.copy(),
        )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest):
        session_id = f"chatcmpl-{uuid.uuid4().hex}"
        content = _conversation_text(body.messages)
        with set_context(_session_context(session_id)):
            if body.stream:
                return StreamingResponse(
                    _stream_chat(bus, async_agent, session_id, content, body.model),
                    media_type="text/event-stream",
                )
            answer, timed_out = await _run_one_turn(
                bus, async_agent, session_id, content, timeout=30.0
            )
        if timed_out:
            return JSONResponse(
                {"error": {"message": "upstream agent timed out", "type": "timeout"}},
                status_code=504,
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
        with set_context(_session_context(session_id)):
            await async_agent.post(
                InboundMessage(
                    role=body.role, content=body.content, session_id=session_id
                )
            )
        return {"status": "accepted", "session_id": session_id}

    @app.get("/v1/sessions/{session_id}/history")
    async def get_history(session_id: str, limit: int | None = None):
        with set_context(_session_context(session_id)):
            messages = await manager.aload(session_id, limit=limit)
        return {
            "session_id": session_id,
            "messages": messages,
        }

    @app.get("/", response_class=HTMLResponse)
    async def webui():
        return HTMLResponse(content=_webui_html())

    def _goal_to_response(goal: Goal) -> GoalResponse:
        return GoalResponse(
            id=goal.id,
            description=goal.description,
            context=goal.context,
            state=goal.state,
            subtasks=[
                {
                    "id": s.id,
                    "description": s.description,
                    "status": s.status,
                    "result": s.result,
                }
                for s in goal.subtasks
            ],
            created_at=goal.created_at,
            updated_at=goal.updated_at,
        )

    @app.post("/v1/goals")
    async def create_goal(body: CreateGoalRequest):
        with set_context(_session_context(f"goal-{uuid.uuid4().hex[:8]}")):
            goal = await coordinator.arun(body.description, body.context)
        return _goal_to_response(goal)

    @app.get("/v1/goals")
    async def list_goals():
        goals = await coordinator.state_machine.alist_goals()
        return [_goal_to_response(g) for g in goals]

    @app.get("/v1/goals/{goal_id}")
    async def get_goal(goal_id: str):
        try:
            goal = await coordinator.state_machine.aload(goal_id)
        except KeyError:
            return JSONResponse({"error": "not found"}, status_code=404)
        return _goal_to_response(goal)

    @app.post("/v1/goals/{goal_id}/retry")
    async def retry_goal(goal_id: str):
        goal = await coordinator.retry(goal_id)
        return _goal_to_response(goal)

    @app.delete("/v1/goals/{goal_id}")
    async def cancel_goal(goal_id: str):
        goal = await coordinator.cancel(goal_id)
        return _goal_to_response(goal)

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

    unsubscribe = bus.subscribe(on_message)
    try:
        await async_agent.post(
            InboundMessage(role="user", content=content, session_id=session_id)
        )

        created = int(time.time())
        sent_role = False
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                break
            if not sent_role:
                yield _sse_chunk(
                    session_id, created, model, {"role": "assistant"}
                )
                sent_role = True
            yield _sse_chunk(
                session_id, created, model, {"content": msg.content}
            )
            if msg.final:
                break

        yield _sse_chunk(session_id, created, model, {}, finish_reason="stop")
        yield "data: [DONE]\n\n"
    finally:
        unsubscribe()


def _sse_chunk(
    session_id: str,
    created: int,
    model: str,
    delta: dict,
    finish_reason: str | None = None,
) -> str:
    """Format one server-sent event chunk."""
    chunk = {
        "id": session_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"


async def _run_one_turn(
    bus: MessageBus,
    async_agent: AsyncAgent,
    session_id: str,
    content: str,
    timeout: float = 30.0,
) -> tuple[str, bool]:
    """Post a message and wait for the final outbound response.

    Returns (answer, timed_out); a timed-out turn yields an empty answer
    rather than pretending a normal stop.
    """
    answer_event = asyncio.Event()
    last_answer: list[str] = [""]

    def on_message(msg: OutboundMessage) -> None:
        if msg.session_id == session_id and msg.final:
            last_answer[0] = msg.content
            answer_event.set()

    unsubscribe = bus.subscribe(on_message)
    await async_agent.post(
        InboundMessage(role="user", content=content, session_id=session_id)
    )
    timed_out = False
    try:
        await asyncio.wait_for(answer_event.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        timed_out = True
    finally:
        unsubscribe()
    return last_answer[0], timed_out


def _conversation_text(messages: list[ChatMessage]) -> str:
    """Render the full message history as the agent's task prompt.

    Single-message requests pass through verbatim; multi-turn conversations
    become a role-prefixed transcript so the model sees the prior context
    (each completion request runs a fresh session, so history must travel
    with the prompt).
    """
    if len(messages) == 1:
        return messages[0].content
    return "\n".join(f"{m.role}: {m.content}" for m in messages)


@lru_cache(maxsize=1)
def _webui_html() -> str:
    return (
        Path(__file__).resolve().parent / "templates" / "webui.html"
    ).read_text(encoding="utf-8")


def main():
    """CLI entrypoint: uvicorn swaybot.api:app"""
    import uvicorn

    load_dotenv()
    app = create_app()
    uvicorn.run(app, host="127.0.0.1", port=8000)
