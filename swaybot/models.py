"""Model abstraction for text generation backends."""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, cast, runtime_checkable


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


StreamCallback = Callable[[str], None]
ReasoningCallback = Callable[[str], None]


@runtime_checkable
class Model(Protocol):
    """A text generation backend that turns messages into text or tool calls."""

    def generate(
        self,
        messages: list[dict],
        metadata: dict | None = None,
        tools: list[dict] | None = None,
        stream_callback: StreamCallback | None = None,
        reasoning_callback: ReasoningCallback | None = None,
    ) -> str | None:
        """Return the model's text output, or None on failure.

        Implementations that support native tool calls may return a JSON
        string encoding ``{"name": ..., "args": ...}`` when a tool is
        chosen, or expose ``supports_tools = True`` and return a richer
        representation.
        """
        ...


class FallbackModel:
    """Try a chain of models until one succeeds."""

    def __init__(self, models: list[Model]) -> None:
        self.models = models

    def generate(
        self,
        messages: list[dict],
        metadata: dict | None = None,
        tools: list[dict] | None = None,
        stream_callback: StreamCallback | None = None,
        reasoning_callback: ReasoningCallback | None = None,
    ) -> str | None:
        last_error = ""
        for model in self.models:
            try:
                result = model.generate(
                    messages,
                    metadata=metadata,
                    tools=tools,
                    stream_callback=stream_callback,
                    reasoning_callback=reasoning_callback,
                )
                if result is not None:
                    return result
            except Exception as exc:  # pragma: no cover - defensive
                last_error = str(exc)
        if metadata is not None and last_error:
            metadata["error"] = f"all fallback models failed: {last_error}"
        return None


class ScriptedModelError(Exception):
    """Scripted model exhausted or a scripted step is malformed."""


class ScriptedModel:
    """Deterministic replay model for tests and offline demos.

    Each ``generate()`` call consumes the next scripted step:

    * ``{"text": "hello"}`` — plain prose output
    * ``{"action": {"name": "echo", "args": {...}}}`` — a native tool call,
      serialized to the ``{"name": ..., "args": ...}`` JSON the brain expects
    * ``{"raise": ValueError("boom")}`` — raise, to drive fallback/retry paths

    Text steps are delivered through ``stream_callback`` in small chunks so
    streaming consumers are exercised. Exhausting the script raises instead of
    returning None: a loop that asks for more steps than were scripted is a
    test bug, and a silent None would turn it into a confusing assertion
    against the fallback path.
    """

    supports_tools = True

    _CHUNK_CHARS = 8

    def __init__(self, steps: list[dict]):
        for step in steps:
            if sum(key in step for key in ("text", "action", "raise")) != 1:
                raise ScriptedModelError(
                    f"each scripted step needs exactly one of text/action/raise: {step!r}"
                )
        self.steps = [dict(step) for step in steps]
        self._cursor = 0
        self.calls: list[list[dict]] = []

    @property
    def remaining(self) -> int:
        return len(self.steps) - self._cursor

    def generate(
        self,
        messages: list[dict],
        metadata: dict | None = None,
        tools: list[dict] | None = None,
        stream_callback: StreamCallback | None = None,
        reasoning_callback: ReasoningCallback | None = None,
    ) -> str | None:
        if self._cursor >= len(self.steps):
            raise ScriptedModelError(
                f"script exhausted after {len(self.steps)} steps; "
                "the loop ran further than the test scripted"
            )
        step = self.steps[self._cursor]
        self._cursor += 1
        self.calls.append(list(messages))
        if "raise" in step:
            raise step["raise"]
        start = time.perf_counter()
        if "action" in step:
            action = step["action"]
            content = json.dumps({"name": action["name"], "args": action.get("args", {})})
        else:
            content = step.get("text", "")
        if metadata is not None:
            metadata["raw_output"] = content
            metadata["duration_ms"] = (time.perf_counter() - start) * 1000
            metadata["model_input_messages"] = list(messages)
        if stream_callback:
            for start_idx in range(0, len(content), self._CHUNK_CHARS):
                stream_callback(content[start_idx : start_idx + self._CHUNK_CHARS])
        return content


class RecordingModel:
    """Wrap any Model, appending one JSONL line per generate() call.

    Full-fidelity model I/O for debugging and replay: the exact messages
    and tool schemas sent, the raw output, the metadata dict (finish
    reason, token usage, duration), and errors. Enable by setting
    SWAYBOT_ROLLOUT_DIR (LLMBrain wraps its outermost backend) or by
    wrapping explicitly. Loop and provider code are untouched.

    No credentials are recorded: request bodies contain only messages and
    tool schemas, and base_url is reduced to scheme://host.
    """

    def __init__(self, inner: Model, directory: Path | str) -> None:
        self._inner = inner
        self._dir = Path(directory).expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self._path = self._dir / f"model-io-{stamp}-{id(self):x}.jsonl"

    @property
    def supports_tools(self) -> bool:  # passthrough for brain capability checks
        return bool(getattr(self._inner, "supports_tools", False))

    def generate(
        self,
        messages: list[dict],
        metadata: dict | None = None,
        tools: list[dict] | None = None,
        stream_callback: StreamCallback | None = None,
        reasoning_callback: ReasoningCallback | None = None,
    ) -> str | None:
        call_metadata: dict = {}
        start = time.perf_counter()
        error = ""
        result: str | None = None
        try:
            result = self._inner.generate(
                messages,
                metadata=call_metadata,
                tools=tools,
                stream_callback=stream_callback,
                reasoning_callback=reasoning_callback,
            )
            if result is None:
                error = str(call_metadata.get("error", "model returned None"))
            return result
        except Exception as exc:  # noqa: BLE001 —— 记录后原样上抛，不吞错
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._record(messages, tools, result, call_metadata, error, start)

    def _record(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        output: str | None,
        call_metadata: dict,
        error: str,
        start: float,
    ) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "model": getattr(self._inner, "model", None),
            "base_url_host": _host_of(getattr(self._inner, "base_url", None)),
            "duration_ms": (time.perf_counter() - start) * 1000,
            "tool_names": [t.get("name") for t in (tools or []) if isinstance(t, dict)],
            "message_count": len(messages),
            "messages": messages,
            "raw_output": output,
            "metadata": call_metadata,
            "error": error or None,
        }
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _host_of(base_url: str | None) -> str | None:
    """Reduce a base URL to scheme://host — query strings can carry keys."""
    if not base_url:
        return None
    parts = base_url.split("://", 1)
    if len(parts) != 2:
        return None
    host = parts[1].split("/", 1)[0]
    return f"{parts[0]}://{host}"


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

        # Some OpenAI-compatible deployments do not support native function
        # calling.  Let users (or benchmark configs) force text-mode generation.
        if os.environ.get("SWAYBOT_FORCE_TEXT_MODE", "").lower() in (
            "1",
            "true",
            "yes",
        ):
            self.supports_tools = False

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
            http_client=self._build_http_client(),
        )

    def _build_http_client(self):
        """Build an httpx.Client tuned for OpenAI-compatible endpoints.

        Two adjustments matter for self-hosted OpenAI-compatible gateways:

        * ``trust_env=False`` — httpx otherwise honors ``HTTP_PROXY`` /
          ``HTTPS_PROXY`` / ``ALL_PROXY`` / ``NO_PROXY`` and any system proxy
          configuration.  Even when those vars are unset in the shell, macOS
          system proxy / PAC settings can leak in and cause intermittent
          ``SSL: UNEXPECTED_EOF_WHILE_READING`` errors against the gateway.
        * ``http2=False`` — some gateways negotiate HTTP/2 badly under the
          default h2 transport; pinning HTTP/1.1 avoids that failure mode.

        Retries are handled at the ``generate`` level, not the transport.
        """
        import httpx

        transport = httpx.HTTPTransport(retries=0, http2=False)
        return httpx.Client(http2=False, transport=transport, trust_env=False)

    def generate(
        self,
        messages: list[dict],
        metadata: dict | None = None,
        tools: list[dict] | None = None,
        stream_callback: StreamCallback | None = None,
        reasoning_callback: ReasoningCallback | None = None,
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
                lowered = last_error.lower()
                if openai_tools and (
                    ("tool" in lowered and "not supported" in lowered)
                    or "tool choice requires" in lowered
                    or "enable-auto-tool-choice" in lowered
                    or "tool_call_parser" in lowered
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
        choice = response.choices[0]
        message = choice.message
        content = message.content or ""
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning and reasoning_callback:
            reasoning_callback(reasoning)
        finish_reason = getattr(choice, "finish_reason", None)
        tool_calls = getattr(message, "tool_calls", None)
        # A length truncation may leave tool_calls with an arguments string that
        # was cut mid-JSON; executing it would run a half-formed call. When the
        # stop reason is 'length', prefer the (complete) visible content and
        # drop tool_calls so think()'s prose->final_answer path handles it.
        if (
            finish_reason == "length"
            and isinstance(tool_calls, list)
            and tool_calls
            and content.strip()
        ):
            tool_calls = None
        if isinstance(tool_calls, list) and tool_calls:
            action = self._tool_call_to_action(tool_calls[0])
            if action is not None:
                content = json.dumps(action)
        self._record_completion_metadata(metadata, messages, content, start, response)
        if metadata is not None:
            metadata["finish_reason"] = finish_reason
        return content

    def _stream_generate(
        self,
        messages: list[dict],
        kwargs: dict,
        metadata: dict | None,
        start: float,
        stream_callback: StreamCallback | None,
        reasoning_callback: ReasoningCallback | None,
    ) -> str:
        """Stream a completion, invoking callbacks and recording metadata."""
        content_parts: list[str] = []
        response = self._client.chat.completions.create(**kwargs)
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning and reasoning_callback:
                reasoning_callback(reasoning)
            token = getattr(delta, "content", None) or ""
            if token:
                content_parts.append(token)
                if stream_callback:
                    stream_callback(token)
        content = "".join(content_parts)
        self._record_base_metadata(metadata, messages, content, start)
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
            parameters = tool.get("parameters", {})
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
