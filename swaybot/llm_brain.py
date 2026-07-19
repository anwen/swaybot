import json
import os
import time
from typing import cast

from .models import FallbackModel, Model
from .prompts import render_prompt

try:
    from openai import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        OpenAI,
    )
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[misc, assignment]
    APIStatusError = None  # type: ignore[misc, assignment]
    APITimeoutError = None  # type: ignore[misc, assignment]
    APIConnectionError = None  # type: ignore[misc, assignment]


def _is_retryable(exc: Exception) -> bool:
    """Return True if the exception is worth retrying."""
    if APIStatusError is not None and isinstance(exc, APIStatusError):
        # Retry server errors and rate limits; not client errors.
        return exc.status_code >= 500 or exc.status_code == 429
    if APITimeoutError is not None and isinstance(exc, APITimeoutError):
        return True
    if APIConnectionError is not None and isinstance(exc, APIConnectionError):
        return True
    # Unknown/non-OpenAI exceptions (e.g. mock RuntimeError) are retried by default.
    return True


def _strip_markdown_fences(raw: str) -> str:
    """Remove leading/trailing ``` markdown fences if present."""
    raw = raw.strip()
    if not raw.startswith("```"):
        return raw
    lines = raw.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_exploration_response(raw: str) -> dict:
    """Parse a JSON response expected to contain task and hypothesis."""
    raw = _strip_markdown_fences(raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    if not isinstance(data, dict):
        return {}

    return {
        "task": data.get("task", data.get("explore", "")),
        "hypothesis": data.get("hypothesis", ""),
    }


class OpenAIModel:
    """OpenAI-compatible text generation backend."""

    supports_tools = True

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
    ):
        if OpenAI is None:
            raise ImportError(
                "OpenAI model requires the 'openai' package. "
                "Install it with: pip install 'swaybot[llm]'"
            )

        self.api_key = api_key or os.environ.get("SWAYBOT_API_KEY")
        self.base_url = base_url or os.environ.get("SWAYBOT_API_BASE")
        self.model = model or os.environ.get("SWAYBOT_MODEL")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.backoff = backoff
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "OpenAI model requires an API key. "
                "Set SWAYBOT_API_KEY or pass api_key."
            )
        if not self.base_url:
            raise ValueError(
                "OpenAI model requires a base URL. "
                "Set SWAYBOT_API_BASE or pass base_url."
            )
        if not self.model:
            raise ValueError(
                "OpenAI model requires a model name. "
                "Set SWAYBOT_MODEL or pass model."
            )

        self._client = OpenAI(
            api_key=cast(str, self.api_key),
            base_url=cast(str, self.base_url),
            timeout=self.timeout,
        )

    def generate(
        self,
        messages: list[dict],
        metadata: dict | None = None,
        tools: list[dict] | None = None,
        stream_callback=None,
        reasoning_callback=None,
    ) -> str | None:
        last_error = ""
        openai_tools = self._build_openai_tools(tools)
        for attempt in range(self.max_retries + 1):
            start = time.perf_counter()
            try:
                kwargs = self._prepare_kwargs(messages, openai_tools, stream_callback)
                if stream_callback:
                    return self._stream_generate(
                        messages,
                        kwargs,
                        metadata,
                        start,
                        stream_callback,
                        reasoning_callback,
                    )
                response = self._client.chat.completions.create(**kwargs)
                return self._handle_completion_response(
                    messages, response, metadata, start, reasoning_callback
                )
            except Exception as exc:
                last_error = str(exc)
                if metadata is not None:
                    metadata["error"] = last_error
                # Some providers accept chat completions but reject native tool-calling.
                # Drop the tools parameter and retry immediately; the system prompt
                # still describes available tools as JSON schema.
                if (
                    openai_tools
                    and "tool" in last_error.lower()
                    and "not supported" in last_error.lower()
                ):
                    openai_tools = []
                    continue
                if not _is_retryable(exc):
                    break
                if attempt < self.max_retries:
                    time.sleep(self.backoff * (2 ** attempt))
        return None

    def _prepare_kwargs(
        self,
        messages: list[dict],
        openai_tools: list[dict],
        stream_callback,
    ) -> dict:
        kwargs = {
            "model": cast(str, self.model),
            "messages": messages,  # type: ignore
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if openai_tools:
            kwargs["tools"] = openai_tools
            kwargs["tool_choice"] = "auto"
        if stream_callback:
            kwargs["stream"] = True
        return kwargs

    def _handle_completion_response(
        self,
        messages: list[dict],
        response,
        metadata: dict | None,
        start: float,
        reasoning_callback,
    ) -> str:
        message = response.choices[0].message
        content = message.content or ""
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning and reasoning_callback:
            reasoning_callback(reasoning)
        tool_calls = getattr(message, "tool_calls", None)
        if isinstance(tool_calls, list) and tool_calls:
            action = self._tool_call_to_action(tool_calls[0])
            if action is not None:
                content = json.dumps(action)
        self._record_completion_metadata(metadata, messages, content, start, response)
        return content

    def _record_completion_metadata(
        self,
        metadata: dict | None,
        messages: list[dict],
        content: str,
        start: float,
        response,
    ) -> None:
        self._record_base_metadata(metadata, messages, content, start)
        if metadata is None:
            return
        usage = response.usage
        if usage:
            metadata["token_usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }

    def _record_stream_metadata(
        self,
        metadata: dict | None,
        messages: list[dict],
        content: str,
        start: float,
    ) -> None:
        self._record_base_metadata(metadata, messages, content, start)

    def _record_base_metadata(
        self,
        metadata: dict | None,
        messages: list[dict],
        content: str,
        start: float,
    ) -> None:
        if metadata is None:
            return
        metadata["model_input_messages"] = list(messages)
        metadata["raw_output"] = content
        metadata["duration_ms"] = (time.perf_counter() - start) * 1000

    @staticmethod
    def _build_openai_tools(tools: list[dict] | None) -> list[dict]:
        if not tools:
            return []
        openai_tools = []
        for tool in tools:
            parameters = tool.get("parameters", tool.get("inputs", {}))
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": parameters,
                    },
                }
            )
        return openai_tools

    @staticmethod
    def _tool_call_to_action(tool_call) -> dict | None:
        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            arguments = {}
        return {
            "name": tool_call.function.name,
            "args": arguments,
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
        stream_callback=None,
        reasoning_callback=None,
    ) -> dict:
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

        messages = [
            {
                "role": "system",
                "content": render_prompt(
                    "system",
                    tool_descriptions=perception.get("tool_descriptions"),
                    available_tools=available_tools,
                    behavior_guidance=perception.get("behavior_guidance"),
                    project_instructions=perception.get("project_instructions"),
                ),
            }
        ]
        if perception.get("messages"):
            messages.extend(perception["messages"])
        else:
            messages.append(
                {"role": "user", "content": render_prompt("user", **perception)}
            )

        raw = self._model.generate(
            messages,
            metadata=call_info,
            tools=perception.get("tool_descriptions")
            if getattr(self._model, "supports_tools", False)
            else None,
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

    def _plan(
        self,
        perception: dict,
        available_tools: list[str],
        metadata: dict | None = None,
    ) -> dict:
        messages = [
            {
                "role": "system",
                "content": render_prompt(
                    "system",
                    tool_descriptions=perception.get("tool_descriptions"),
                    available_tools=available_tools,
                    behavior_guidance=perception.get("behavior_guidance"),
                    project_instructions=perception.get("project_instructions"),
                ),
            },
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
            {
                "role": "system",
                "content": render_prompt(
                    "system",
                    tool_descriptions=perception.get("tool_descriptions"),
                    available_tools=available_tools,
                    behavior_guidance=perception.get("behavior_guidance"),
                    project_instructions=perception.get("project_instructions"),
                ),
            },
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
        return {"name": "explore", "args": _parse_exploration_response(raw)}


def _merge_metadata(target: dict | None, source: dict) -> None:
    if target is not None:
        target.update(source)


def _parse_action(raw: str, perception: dict) -> dict:
    raw = _strip_markdown_fences(raw)

    try:
        action = json.loads(raw)
    except json.JSONDecodeError:
        return _fallback(perception, raw)

    if not isinstance(action, dict) or "name" not in action:
        return _fallback(perception, raw)

    return action


def _parse_plan(raw: str) -> dict:
    raw = _strip_markdown_fences(raw)

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


def _fallback(perception: dict, raw: str) -> dict:
    if perception["step"] + 1 >= perception["max_steps"]:
        return {"name": "done", "args": {}}
    return {"name": "echo", "args": {"message": raw[:200]}}
