import json
from typing import Any

from .brain import ReasoningCallback, StreamCallback
from .models import FallbackModel, Model, OpenAIModel
from .parsing import parse_exploration_response, strip_markdown_fences
from .prompts import render_prompt


# Schemas for the control tools used to finish a task.  These are injected into
# native function-calling requests when the backend supports ``tools`` so that
# the model can explicitly signal completion instead of emitting JSON text.
FINAL_ANSWER_SCHEMA: dict[str, Any] = {
    "name": "final_answer",
    "description": "Provide the final answer to the task.",
    "parameters": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The final answer text to return to the user.",
            },
        },
        "required": ["answer"],
    },
}

DONE_SCHEMA: dict[str, Any] = {
    "name": "done",
    "description": "Signal that the task is complete or cannot proceed.",
    "parameters": {
        "type": "object",
        "properties": {},
    },
}


class LLMBrain:
    """Brain backed by a Model text-generation backend."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        max_retries: int = 3,
        backoff: float = 1.0,
        timeout: float = 60.0,
        backend: Model | None = None,
        fallback_models: list[Model] | None = None,
    ):
        if backend is None:
            backend = OpenAIModel(
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
                backoff=backoff,
                timeout=timeout,
            )
        if fallback_models:
            backend = FallbackModel([backend] + fallback_models)
        self._model = backend
        self.api_key = getattr(backend, "api_key", None)
        self.base_url = getattr(backend, "base_url", None)
        self.model = getattr(backend, "model", None)

    def think(
        self,
        perception: dict,
        available_tools: list[str],
        metadata: dict | None = None,
        stream_callback: StreamCallback | None = None,
        reasoning_callback: ReasoningCallback | None = None,
    ) -> dict[str, Any]:
        call_info: dict = {}
        if perception.get("planning"):
            plan = self._plan(perception, available_tools, metadata=call_info)
            _merge_metadata(metadata, call_info)
            return plan
        if perception.get("exploring"):
            exploration = self._explore(
                perception, available_tools, metadata=call_info
            )
            _merge_metadata(metadata, call_info)
            return exploration

        messages = [self._system_message(perception, available_tools)]
        if perception.get("messages"):
            messages.extend(perception["messages"])
        else:
            messages.append(
                {"role": "user", "content": render_prompt("user", **perception)}
            )

        # When using a backend that supports native tool calling, augment the
        # provided tool schemas with the control tools so the model can finish.
        supports_tools = getattr(self._model, "supports_tools", False)
        tool_descriptions = perception.get("tool_descriptions")
        tools_for_api = tool_descriptions
        if supports_tools and tool_descriptions:
            tools_for_api = list(tool_descriptions)
            existing = {t.get("name") for t in tools_for_api if isinstance(t, dict)}
            if "final_answer" not in existing:
                tools_for_api.append(FINAL_ANSWER_SCHEMA)
            if "done" not in existing:
                tools_for_api.append(DONE_SCHEMA)

        raw = self._model.generate(
            messages,
            metadata=call_info,
            tools=tools_for_api if supports_tools else None,
            stream_callback=stream_callback,
            reasoning_callback=reasoning_callback,
        )
        _merge_metadata(metadata, call_info)
        if raw is None:
            error_detail = call_info.get("error", "")
            message = "LLM call failed after retries"
            if error_detail:
                message = f"{message}: {error_detail}"
            return _fallback(perception, message)
        return _parse_action(raw, perception)

    def _system_message(
        self, perception: dict, available_tools: list[str]
    ) -> dict:
        """Build the shared system message.

        When the backend supports native tool calling and tool schemas are
        available, use the ``system_fc`` template.  That template omits the
        inline JSON-schema dump and the "reply with JSON" instruction, so the
        model uses the OpenAI ``tools`` parameter instead of emitting JSON text.
        """
        tool_descriptions = perception.get("tool_descriptions")
        supports_tools = getattr(self._model, "supports_tools", False)
        template = "system_fc" if (supports_tools and tool_descriptions) else "system"
        return {
            "role": "system",
            "content": render_prompt(
                template,
                tool_descriptions=tool_descriptions,
                available_tools=available_tools,
                behavior_guidance=perception.get("behavior_guidance"),
                project_instructions=perception.get("project_instructions"),
            ),
        }

    def _plan(
        self,
        perception: dict,
        available_tools: list[str],
        metadata: dict | None = None,
    ) -> dict:
        messages = [
            self._system_message(perception, available_tools),
            {
                "role": "user",
                "content": render_prompt(
                    "plan",
                    task=perception["task"],
                    max_steps=perception["max_steps"],
                    tool_descriptions=perception.get("tool_descriptions"),
                    available_tools=available_tools,
                ),
            },
        ]

        raw = self._model.generate(messages, metadata=metadata)
        if raw is None:
            return {"name": "plan", "args": {"steps": []}}
        return _parse_plan(raw)

    def _explore(
        self,
        perception: dict,
        available_tools: list[str],
        metadata: dict | None = None,
    ) -> dict:
        messages = [
            self._system_message(perception, available_tools),
            {
                "role": "user",
                "content": render_prompt(
                    "explore",
                    memory_context=perception.get("memory_context", ""),
                    tool_descriptions=perception.get("tool_descriptions"),
                    available_tools=available_tools,
                ),
            },
        ]

        raw = self._model.generate(messages, metadata=metadata)
        if raw is None:
            return {"name": "explore", "args": {"task": "explore", "hypothesis": ""}}
        return {"name": "explore", "args": parse_exploration_response(raw)}


def _merge_metadata(target: dict | None, source: dict) -> None:
    if target is not None:
        target.update(source)


def _parse_action(raw: str, perception: dict) -> dict[str, Any]:
    raw = strip_markdown_fences(raw)

    try:
        action = json.loads(raw)
    except json.JSONDecodeError:
        return _fallback(perception, raw)

    if not isinstance(action, dict) or "name" not in action:
        return _fallback(perception, raw)

    return action


def _parse_plan(raw: str) -> dict[str, Any]:
    raw = strip_markdown_fences(raw)

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        return {"name": "plan", "args": {"steps": []}}

    if isinstance(plan, list) and all(isinstance(s, str) for s in plan):
        return {"name": "plan", "args": {"steps": plan}}
    if isinstance(plan, dict):
        steps = plan.get("steps", plan.get("plan", []))
        if isinstance(steps, list) and all(isinstance(s, str) for s in steps):
            return {"name": "plan", "args": {"steps": steps}}

    return {"name": "plan", "args": {"steps": []}}


def _fallback(perception: dict, raw: str) -> dict[str, Any]:
    if perception["step"] + 1 >= perception["max_steps"]:
        return {"name": "done", "args": {}}
    return {"name": "echo", "args": {"message": raw[:200]}}
