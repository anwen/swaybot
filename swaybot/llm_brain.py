import json
import os
from typing import cast

from .prompts import render_prompt

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[misc, assignment]


class LLMBrain:
    """Brain backed by an OpenAI-compatible chat completion API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
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

    def think(self, perception: dict, available_tools: list[str]) -> dict:
        messages = [
            {
                "role": "system",
                "content": render_prompt(
                    "system",
                    tool_descriptions=perception.get("tool_descriptions"),
                    available_tools=available_tools,
                ),
            }
        ]
        if perception.get("messages"):
            messages.extend(perception["messages"])
        else:
            messages.append(
                {"role": "user", "content": render_prompt("user", **perception)}
            )

        try:
            response = self._client.chat.completions.create(
                model=cast(str, self.model),
                messages=messages,  # type: ignore
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            raw = response.choices[0].message.content or ""
        except Exception as exc:
            return _fallback(perception, f"LLM call failed: {exc}")

        return _parse_action(raw, perception)


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


def _fallback(perception: dict, raw: str) -> dict:
    if perception["step"] + 1 >= perception["max_steps"]:
        return {"name": "done", "args": {}}
    return {"name": "echo", "args": {"message": raw[:200]}}
