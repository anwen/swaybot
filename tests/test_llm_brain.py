from unittest.mock import MagicMock, patch

import pytest

from swaybot.llm_brain import LLMBrain


def _make_brain():
    return LLMBrain(api_key="test-key", base_url="http://localhost/v1", model="test-model")


def _mock_response(content: str):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


@patch("swaybot.llm_brain.OpenAI")
def test_llm_brain_parses_json_action(mock_openai):
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(
        '{"name": "add", "args": {"a": 1, "b": 2}}'
    )
    mock_openai.return_value = client

    brain = _make_brain()
    action = brain.think(
        {"task": "compute", "step": 0, "max_steps": 3, "history": []},
        ["echo", "add", "done"],
    )
    assert action == {"name": "add", "args": {"a": 1, "b": 2}}


@patch("swaybot.llm_brain.OpenAI")
def test_llm_brain_strips_markdown_fences(mock_openai):
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(
        '```json\n{"name": "done", "args": {}}\n```'
    )
    mock_openai.return_value = client

    brain = _make_brain()
    action = brain.think(
        {"task": "finish", "step": 0, "max_steps": 3, "history": []},
        ["echo", "done"],
    )
    assert action == {"name": "done", "args": {}}


@patch("swaybot.llm_brain.OpenAI")
def test_llm_brain_fallback_on_invalid_json(mock_openai):
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response("not json")
    mock_openai.return_value = client

    brain = _make_brain()
    action = brain.think(
        {"task": "fail", "step": 0, "max_steps": 3, "history": []},
        ["echo", "done"],
    )
    assert action["name"] == "echo"
    assert "not json" in action["args"]["message"]


@patch("swaybot.llm_brain.OpenAI")
def test_llm_brain_fallback_on_api_error(mock_openai):
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("connection failed")
    mock_openai.return_value = client

    brain = _make_brain()
    action = brain.think(
        {"task": "fail", "step": 0, "max_steps": 3, "history": []},
        ["echo", "done"],
    )
    assert action["name"] == "echo"
    assert "LLM call failed" in action["args"]["message"]


def test_llm_brain_requires_api_key():
    with pytest.raises(ValueError):
        LLMBrain(api_key=None, base_url="http://localhost/v1")


@patch("swaybot.llm_brain.OpenAI", None)
def test_llm_brain_requires_openai_package():
    with pytest.raises(ImportError):
        LLMBrain(api_key="test-key", base_url="http://localhost/v1")


@patch("swaybot.llm_brain.OpenAI")
def test_llm_brain_uses_env_variables(mock_openai, monkeypatch):
    monkeypatch.setenv("SWAYBOT_API_KEY", "env-key")
    monkeypatch.setenv("SWAYBOT_API_BASE", "http://env/v1")
    monkeypatch.setenv("SWAYBOT_MODEL", "env-model")
    mock_openai.return_value = MagicMock()

    brain = LLMBrain()
    assert brain.api_key == "env-key"
    assert brain.base_url == "http://env/v1"
    assert brain.model == "env-model"
