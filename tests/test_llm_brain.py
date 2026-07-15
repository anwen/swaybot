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
def test_llm_brain_system_prompt_includes_tool_schemas(mock_openai):
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(
        '{"name": "done", "args": {}}'
    )
    mock_openai.return_value = client

    schemas = [
        {
            "name": "add",
            "description": "Add two numbers.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
        {
            "name": "done",
            "description": "Finish the task.",
            "parameters": {"type": "object", "properties": {}},
        },
    ]

    brain = _make_brain()
    brain.think(
        {
            "task": "test",
            "step": 0,
            "max_steps": 3,
            "history": [],
            "tool_descriptions": schemas,
        },
        ["add", "done"],
    )
    call_kwargs = client.chat.completions.create.call_args.kwargs
    messages = call_kwargs["messages"]
    system_content = messages[0]["content"]
    assert "Add two numbers" in system_content
    assert '"type": "number"' in system_content
    assert "done" in system_content


@patch("swaybot.llm_brain.OpenAI")
def test_llm_brain_system_prompt_falls_back_to_tool_names(mock_openai):
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(
        '{"name": "done", "args": {}}'
    )
    mock_openai.return_value = client

    brain = _make_brain()
    brain.think(
        {"task": "test", "step": 0, "max_steps": 3, "history": []},
        ["echo", "done"],
    )
    call_kwargs = client.chat.completions.create.call_args.kwargs
    messages = call_kwargs["messages"]
    system_content = messages[0]["content"]
    assert "echo" in system_content
    assert "done" in system_content


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


def test_llm_brain_requires_api_key(monkeypatch):
    monkeypatch.delenv("SWAYBOT_API_KEY", raising=False)
    monkeypatch.delenv("SWAYBOT_API_BASE", raising=False)
    monkeypatch.delenv("SWAYBOT_MODEL", raising=False)
    with pytest.raises(ValueError):
        LLMBrain(api_key=None, base_url="http://localhost/v1")


@patch("swaybot.llm_brain.OpenAI", None)
def test_llm_brain_requires_openai_package():
    with pytest.raises(ImportError):
        LLMBrain(api_key="test-key", base_url="http://localhost/v1")


@patch("swaybot.llm_brain.OpenAI")
def test_llm_brain_uses_messages_when_present(mock_openai):
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(
        '{"name": "done", "args": {}}'
    )
    mock_openai.return_value = client

    brain = _make_brain()
    brain.think(
        {
            "task": "test",
            "step": 1,
            "max_steps": 3,
            "history": [],
            "messages": [
                {"role": "user", "content": "Task: test"},
                {"role": "assistant", "content": '{"name":"echo"}'},
            ],
        },
        ["echo", "done"],
    )
    call_kwargs = client.chat.completions.create.call_args.kwargs
    messages = call_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["content"] == "Task: test"
    assert messages[2]["role"] == "assistant"


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
