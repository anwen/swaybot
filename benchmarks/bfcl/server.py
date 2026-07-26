"""OpenAI-compatible API server wrapping SwayBot's LLMBrain.

This server exposes ``/v1/chat/completions`` that accepts OpenAI-format
messages + tools and routes them through SwayBot's ``LLMBrain``.
BFCL's evaluation harness calls this endpoint as a remote model.

The server does **not** execute tools — BFCL runs its own tool execution
loop against live Python backends. The server only translates
SwayBot's brain output into OpenAI-format function-calling responses.
"""

from __future__ import annotations

import json
import os
import time
import uuid

from swaybot.llm_brain import LLMBrain

from ._helpers import build_registry, extract_task, messages_to_history


def create_app():
    """Create a FastAPI app wrapping SwayBot's LLMBrain.

    Requires ``fastapi`` and ``uvicorn`` (``pip install 'swaybot[api]'``).
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "FastAPI is required for the BFCL server. "
            "Install it with: pip install 'swaybot[api]'"
        ) from exc

    app = FastAPI(title="SwayBot BFCL Server")

    # Per-request brain instances (configured from environment or defaults)
    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    def chat_completions(request: dict):
        """OpenAI-compatible chat completions endpoint.

        Accepts the standard OpenAI Chat Completions request body and
        returns a response with optional tool_calls for function calling.
        """
        model_name = request.get("model", "swaybot")
        messages = request.get("messages", [])
        tools = request.get("tools", [])
        temperature = request.get("temperature", 0.7)
        max_tokens = request.get("max_tokens", 1024)

        if not messages:
            raise HTTPException(status_code=400, detail="messages is required")

        # Build a fresh ToolRegistry from the request's tool definitions
        registry = build_registry(tools)

        # Build SwayBot perception from messages
        task = extract_task(messages)
        history = messages_to_history(messages)

        perception = {
            "task": task,
            "step": len(history),
            "max_steps": 50,
            "history": history,
            "tool_descriptions": registry.schemas(),
            "available_tools": registry.names(),
        }

        # Create a brain for this request. Prefer the underlying model from the
        # environment so that benchmark harnesses can pass any model name while
        # still routing to the configured backend.
        brain_model = os.environ.get("SWAYBOT_MODEL", model_name)
        brain = LLMBrain(
            model=brain_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Ask the brain
        action = brain.think(perception, registry.names())

        # Build OpenAI-format response
        tool_name = action.get("name", "")
        tool_args = action.get("args", {})

        if tool_name and tool_name not in ("done", "final_answer", "echo"):
            # Return a function-calling response
            response = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_args),
                            },
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        else:
            # Text-only response
            text = ""
            if tool_name == "final_answer":
                text = tool_args.get("answer", "")
            elif tool_name == "echo":
                text = tool_args.get("message", "")
            else:
                text = action.get("text", "")

            response = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text or "I'm done.",
                    },
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }

        return JSONResponse(content=response)

    return app
