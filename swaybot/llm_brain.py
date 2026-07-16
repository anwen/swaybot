import json
import os
import time
from typing import cast

from .prompts import render_prompt

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[misc, assignment]


def _parse_exploration_response(raw: str) -> dict:
    """Parse a JSON response expected to contain task and hypothesis."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

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


class LLMBrain:
    """Brain backed by an OpenAI-compatible chat completion API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        max_retries: int = 3,
        backoff: float = 1.0,
    ):
        if OpenAI is None:
            raise ImportError(
                "LLMBrain requires the 'openai' package. "
                "Install it with: pip install 'swaybot[llm]'"
            )

        self.api_key = api_key or os.environ.get("SWAYBOT_API_KEY")
        self.base_url = base_url or os.environ.get("SWAYBOT_API_BASE")
        self.model = model or os.environ.get("SWAYBOT_MODEL")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.backoff = backoff

        if not self.api_key:
            raise ValueError(
                "LLMBrain requires an API key. "
                "Set SWAYBOT_API_KEY or pass api_key."
            )
        if not self.base_url:
            raise ValueError(
                "LLMBrain requires a base URL. "
                "Set SWAYBOT_API_BASE or pass base_url."
            )
        if not self.model:
            raise ValueError(
                "LLMBrain requires a model name. "
                "Set SWAYBOT_MODEL or pass model."
            )

        assert self.api_key is not None
        assert self.base_url is not None
        assert self.model is not None

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def think(
        self,
        perception: dict,
        available_tools: list[str],
        metadata: dict | None = None,
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
                ),
            }
        ]
        if perception.get("messages"):
            messages.extend(perception["messages"])
        else:
            messages.append(
                {"role": "user", "content": render_prompt("user", **perception)}
            )

        raw = self._chat(messages, metadata=call_info)
        _merge_metadata(metadata, call_info)
        if raw is None:
            return _fallback(perception, "LLM call failed after retries")
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

        raw = self._chat(messages, metadata=metadata)
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

        raw = self._chat(messages, metadata=metadata)
        if raw is None:
            return {"name": "explore", "args": {"task": "explore", "hypothesis": ""}}
        return {"name": "explore", "args": _parse_exploration_response(raw)}

    def _chat(
        self,
        messages: list[dict],
        metadata: dict | None = None,
    ) -> str | None:
        last_error = ""
        for attempt in range(self.max_retries + 1):
            start = time.perf_counter()
            try:
                response = self._client.chat.completions.create(
                    model=cast(str, self.model),
                    messages=messages,  # type: ignore
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content or ""
                if metadata is not None:
                    metadata["model_input_messages"] = list(messages)
                    metadata["raw_output"] = content
                    metadata["duration_ms"] = (time.perf_counter() - start) * 1000
                    usage = response.usage
                    if usage:
                        metadata["token_usage"] = {
                            "prompt_tokens": getattr(usage, "prompt_tokens", None),
                            "completion_tokens": getattr(
                                usage, "completion_tokens", None
                            ),
                            "total_tokens": getattr(usage, "total_tokens", None),
                        }
                return content
            except Exception as exc:
                last_error = str(exc)
                if metadata is not None:
                    metadata["error"] = last_error
                if attempt < self.max_retries:
                    time.sleep(self.backoff * (2 ** attempt))
        return None


def _merge_metadata(target: dict | None, source: dict) -> None:
    if target is not None:
        target.update(source)


def _parse_action(raw: str, perception: dict) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    try:
        action = json.loads(raw)
    except json.JSONDecodeError:
        return _fallback(perception, raw)

    if not isinstance(action, dict) or "name" not in action:
        return _fallback(perception, raw)

    return action


def _parse_plan(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

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
