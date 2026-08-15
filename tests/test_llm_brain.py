import json
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


def test_llm_brain_fc_system_prompt_omits_inline_schemas():
    """Native FC mode should use system_fc and not embed JSON schemas."""
    backend = MagicMock()
    backend.supports_tools = True
    backend.generate.return_value = '{"name": "done", "args": {}}'

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

    brain = LLMBrain(backend=backend)
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
    messages = backend.generate.call_args.args[0]
    system_content = messages[0]["content"]
    assert "Reply with a single JSON object" not in system_content
    assert "Available tools (JSON schema)" not in system_content
    assert "Use the provided tools" in system_content
    assert "call the 'final_answer' tool" in system_content
    assert "call the 'done' tool" in system_content


def test_llm_brain_text_system_prompt_includes_inline_schemas():
    """Text-based backends still get the legacy inline-schema prompt."""
    backend = MagicMock()
    backend.supports_tools = False
    backend.generate.return_value = '{"name": "done", "args": {}}'

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
    ]

    brain = LLMBrain(backend=backend)
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
    messages = backend.generate.call_args.args[0]
    system_content = messages[0]["content"]
    assert "Reply with a single JSON object" in system_content
    assert "Available tools (JSON schema)" in system_content
    assert '"type": "number"' in system_content


def test_llm_brain_fc_prompt_requires_retrieval_before_answer():
    """The FC prompt must tell the model to search before final_answer."""
    backend = MagicMock()
    backend.supports_tools = True
    backend.generate.return_value = '{"name": "done", "args": {}}'
    brain = LLMBrain(backend=backend)
    brain.think(
        {"task": "test", "step": 0, "max_steps": 3, "history": [],
         "tool_descriptions": [{"name": "done", "parameters": {"type": "object", "properties": {}}}]},
        ["done"],
    )
    system_content = backend.generate.call_args.args[0][0]["content"]
    assert "before calling final_answer" in system_content.lower()
    assert "search/retrieval tool" in system_content.lower()


@patch("swaybot.models.OpenAI")
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


@patch("swaybot.models.OpenAI")
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


@patch("swaybot.models.OpenAI")
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


@patch("swaybot.models.OpenAI")
def test_llm_brain_fallback_on_invalid_json(mock_openai):
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response("not json")
    mock_openai.return_value = client

    brain = _make_brain()
    action = brain.think(
        {"task": "fail", "step": 0, "max_steps": 3, "history": []},
        ["echo", "done"],
    )
    # Prose that can't be parsed as a tool call is treated as the final
    # answer (full text), not an echo-truncated stub, so the turn completes
    # instead of looping on a "you got cut off" complaint.
    assert action["name"] == "final_answer"
    assert "not json" in action["args"]["answer"]


@patch("swaybot.models.OpenAI")
def test_llm_brain_prose_becomes_full_final_answer(mock_openai):
    """Prose output should become final_answer with the FULL text, not 200 chars."""
    client = MagicMock()
    long_prose = "Here is my answer. " * 200  # ~3200 chars, well over 200
    client.chat.completions.create.return_value = _mock_response(long_prose)
    mock_openai.return_value = client

    brain = _make_brain()
    action = brain.think(
        {"task": "answer me", "step": 0, "max_steps": 10, "history": []},
        ["echo", "done"],
    )
    assert action["name"] == "final_answer"
    # The full prose is preserved, not truncated to 200 chars.
    assert len(action["args"]["answer"]) > 200
    assert long_prose.strip()[:500] in action["args"]["answer"]


def test_llm_brain_fallback_preserves_full_prose():
    """_fallback should not truncate the echo message to 200 chars."""
    from swaybot.llm_brain import _fallback

    long_prose = "x" * 1000
    action = _fallback(
        {"step": 0, "max_steps": 10}, long_prose
    )
    assert action["name"] == "echo"
    assert len(action["args"]["message"]) == 1000


def test_llm_brain_fallback_caps_pathological_output():
    """_fallback should cap only truly enormous output (safety valve)."""
    from swaybot.llm_brain import _fallback

    huge = "y" * 10000
    action = _fallback({"step": 0, "max_steps": 10}, huge)
    assert action["name"] == "echo"
    assert len(action["args"]["message"]) == 4000


def test_llm_brain_fallback_done_at_max_steps():
    """At the final step, _fallback should signal done, not echo."""
    from swaybot.llm_brain import _fallback

    action = _fallback({"step": 9, "max_steps": 10}, "some prose")
    assert action["name"] == "done"


@patch("swaybot.models.OpenAI")
def test_llm_brain_records_finish_reason(mock_openai):
    """OpenAIModel should record finish_reason into metadata."""
    client = MagicMock()
    response = _mock_response('{"name": "done", "args": {}}')
    response.choices[0].finish_reason = "stop"
    client.chat.completions.create.return_value = response
    mock_openai.return_value = client

    brain = _make_brain()
    metadata: dict = {}
    brain.think(
        {"task": "test", "step": 0, "max_steps": 3, "history": []},
        ["done"],
        metadata=metadata,
    )
    assert metadata["finish_reason"] == "stop"


@patch("swaybot.models.OpenAI")
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


@patch("swaybot.models.OpenAI", None)
def test_llm_brain_requires_openai_package():
    with pytest.raises(ImportError):
        LLMBrain(api_key="test-key", base_url="http://localhost/v1")


@patch("swaybot.models.OpenAI")
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


@patch("swaybot.models.OpenAI")
def test_llm_brain_parses_planning_response_as_list(mock_openai):
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(
        '["search", "summarize", "finish"]'
    )
    mock_openai.return_value = client

    brain = _make_brain()
    action = brain.think(
        {
            "task": "demo",
            "max_steps": 3,
            "step": 0,
            "history": [],
            "planning": True,
        },
        ["echo", "done"],
    )
    assert action == {"name": "plan", "args": {"steps": ["search", "summarize", "finish"]}}


@patch("swaybot.models.OpenAI")
def test_llm_brain_parses_planning_response_as_object(mock_openai):
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(
        '{"steps": ["a", "b"]}'
    )
    mock_openai.return_value = client

    brain = _make_brain()
    action = brain.think(
        {"task": "demo", "max_steps": 3, "step": 0, "history": [], "planning": True},
        ["echo"],
    )
    assert action == {"name": "plan", "args": {"steps": ["a", "b"]}}


@patch("swaybot.models.OpenAI")
def test_llm_brain_planning_fallback_on_invalid_json(mock_openai):
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response("not a plan")
    mock_openai.return_value = client

    brain = _make_brain()
    action = brain.think(
        {"task": "demo", "max_steps": 3, "step": 0, "history": [], "planning": True},
        ["echo"],
    )
    assert action == {"name": "plan", "args": {"steps": []}}


@patch("swaybot.models.OpenAI")
def test_llm_brain_system_prompt_includes_behavior_guidance(mock_openai):
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(
        '{"name": "done", "args": {}}'
    )
    mock_openai.return_value = client

    brain = _make_brain()
    brain.think(
        {
            "task": "test",
            "step": 0,
            "max_steps": 3,
            "history": [],
            "behavior_guidance": "- Always verify the result.",
        },
        ["done"],
    )
    call_kwargs = client.chat.completions.create.call_args.kwargs
    system_content = call_kwargs["messages"][0]["content"]
    assert "Lessons learned" in system_content
    assert "Always verify" in system_content


@patch("swaybot.models.OpenAI")
def test_llm_brain_retries_then_succeeds(mock_openai):
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        RuntimeError("timeout"),
        RuntimeError("timeout"),
        _mock_response('{"name": "done", "args": {}}'),
    ]
    mock_openai.return_value = client

    brain = LLMBrain(
        api_key="test-key",
        base_url="http://localhost/v1",
        model="test-model",
        backoff=0.0,
    )
    action = brain.think(
        {"task": "test", "step": 0, "max_steps": 3, "history": []},
        ["done"],
    )
    assert action == {"name": "done", "args": {}}
    assert client.chat.completions.create.call_count == 3


@patch("swaybot.models.OpenAI")
def test_llm_brain_fallback_after_retries_exhausted(mock_openai):
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("timeout")
    mock_openai.return_value = client

    brain = LLMBrain(
        api_key="test-key",
        base_url="http://localhost/v1",
        model="test-model",
        max_retries=2,
        backoff=0.0,
    )
    action = brain.think(
        {"task": "test", "step": 0, "max_steps": 3, "history": []},
        ["echo", "done"],
    )
    assert action["name"] == "echo"
    assert "after retries" in action["args"]["message"]
    assert client.chat.completions.create.call_count == 3


@patch("swaybot.models.OpenAI")
def test_llm_brain_populates_metadata(mock_openai):
    client = MagicMock()
    response = _mock_response('{"name": "done", "args": {}}')
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 5
    response.usage.total_tokens = 15
    client.chat.completions.create.return_value = response
    mock_openai.return_value = client

    brain = _make_brain()
    metadata: dict = {}
    brain.think(
        {"task": "test", "step": 0, "max_steps": 3, "history": []},
        ["done"],
        metadata=metadata,
    )
    assert metadata["raw_output"] == '{"name": "done", "args": {}}'
    assert metadata["token_usage"]["prompt_tokens"] == 10
    assert metadata["token_usage"]["completion_tokens"] == 5
    assert metadata["token_usage"]["total_tokens"] == 15
    assert metadata["duration_ms"] >= 0


def _mock_tool_call_response(name: str, arguments: dict):
    response = MagicMock()
    response.choices = [MagicMock()]
    message = response.choices[0].message
    message.content = ""
    tool_call = MagicMock()
    tool_call.id = "call_1"
    tool_call.function.name = name
    tool_call.function.arguments = json.dumps(arguments)
    message.tool_calls = [tool_call]
    return response


@patch("swaybot.models.OpenAI")
def test_llm_brain_passes_native_tools_when_supported(mock_openai):
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(
        '{"name": "done", "args": {}}'
    )
    mock_openai.return_value = client

    brain = _make_brain()
    schemas = [
        {
            "name": "add",
            "description": "Add two numbers.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        }
    ]
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
    assert "tools" in call_kwargs
    assert call_kwargs["tools"][0]["type"] == "function"
    assert call_kwargs["tools"][0]["function"]["name"] == "add"


def test_llm_brain_injects_control_tools_for_native_fc():
    backend = MagicMock()
    backend.supports_tools = True
    backend.generate.return_value = '{"name": "done", "args": {}}'

    schemas = [
        {
            "name": "add",
            "description": "Add two numbers.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        }
    ]

    brain = LLMBrain(backend=backend)
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
    tools = backend.generate.call_args.kwargs["tools"]
    names = [t["name"] for t in tools]
    assert names == ["add", "final_answer", "done"]


def test_llm_brain_does_not_duplicate_control_tools():
    backend = MagicMock()
    backend.supports_tools = True
    backend.generate.return_value = '{"name": "done", "args": {}}'

    schemas = [
        {
            "name": "done",
            "description": "Finish.",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "final_answer",
            "description": "Answer.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
        {
            "name": "add",
            "description": "Add two numbers.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
    ]

    brain = LLMBrain(backend=backend)
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
    tools = backend.generate.call_args.kwargs["tools"]
    names = [t["name"] for t in tools]
    assert names == ["done", "final_answer", "add"]


def test_llm_brain_does_not_inject_control_tools_without_native_fc():
    backend = MagicMock()
    backend.supports_tools = False
    backend.generate.return_value = '{"name": "done", "args": {}}'

    schemas = [
        {
            "name": "add",
            "description": "Add two numbers.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        }
    ]

    brain = LLMBrain(backend=backend)
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
    assert backend.generate.call_args.kwargs["tools"] is None


@patch("swaybot.models.OpenAI")
def test_llm_brain_parses_native_tool_call(mock_openai):
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_tool_call_response(
        "add", {"a": 2, "b": 3}
    )
    mock_openai.return_value = client

    brain = _make_brain()
    action = brain.think(
        {
            "task": "compute",
            "step": 0,
            "max_steps": 3,
            "history": [],
            "tool_descriptions": [
                {
                    "name": "add",
                    "description": "Add two numbers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["a", "b"],
                    },
                }
            ],
        },
        ["add", "done"],
    )
    assert action == {"name": "add", "args": {"a": 2, "b": 3}}


def _mock_truncated_tool_call_response(prose: str, broken_args: str):
    """A response truncated mid-tool-call: finish_reason=length, prose present,
    and a tool_call whose arguments string was cut (invalid JSON)."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].finish_reason = "length"
    message = response.choices[0].message
    message.content = prose
    tool_call = MagicMock()
    tool_call.id = "call_1"
    tool_call.function.name = "final_answer"
    tool_call.function.arguments = broken_args  # e.g. '{"answer": "howev'
    message.tool_calls = [tool_call]
    return response


@patch("swaybot.models.OpenAI")
def test_length_truncation_drops_broken_tool_call(mock_openai):
    """finish_reason=length with prose + a half-formed tool_call must prefer the
    prose and NOT execute the truncated tool call."""
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_truncated_tool_call_response(
        "I need to clarify our compensation policy for flights.",
        '{"answer": "howev',  # truncated mid-string -> invalid JSON
    )
    mock_openai.return_value = client

    brain = _make_brain()
    action = brain.think(
        {
            "task": "refund policy",
            "step": 0,
            "max_steps": 3,
            "history": [],
            "tool_descriptions": [
                {
                    "name": "final_answer",
                    "description": "Answer.",
                    "parameters": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                }
            ],
        },
        ["final_answer", "done"],
    )
    # The broken tool_call was dropped; the prose became a final_answer via
    # think()'s prose->final_answer path (not the truncated tool_call).
    assert action["name"] == "final_answer"
    assert "compensation policy" in action["args"]["answer"]


def test_extract_tool_call_json_inside_tag():
    """_extract_tool_call should parse JSON inside <tool_call> tags."""
    from swaybot.llm_brain import _extract_tool_call

    raw = '<tool_call> {"name": "get_user", "args": {"id": "123"}} </tool_call>'
    action = _extract_tool_call(raw)
    assert action == {"name": "get_user", "args": {"id": "123"}}


def test_extract_tool_call_xml_parameters():
    """_extract_tool_call should parse Qwen XML parameter blocks."""
    from swaybot.llm_brain import _extract_tool_call

    raw = """<tool_call>
<function=get_user_details>
<parameter=user_id>
u_42
</parameter>
</function>
</tool_call>"""
    action = _extract_tool_call(raw)
    assert action == {"name": "get_user_details", "args": {"user_id": "u_42"}}


def test_extract_tool_call_returns_none_for_plain_json():
    """_extract_tool_call should not interfere with plain JSON actions."""
    from swaybot.llm_brain import _extract_tool_call

    assert _extract_tool_call('{"name": "done", "args": {}}') is None


def test_extract_tool_call_numeric_parameter():
    """_extract_tool_call should preserve plain numeric values as strings."""
    from swaybot.llm_brain import _extract_tool_call

    raw = "<tool_call><function=add><parameter=a>3</parameter><parameter=b>4</parameter></function></tool_call>"
    action = _extract_tool_call(raw)
    assert action == {"name": "add", "args": {"a": "3", "b": "4"}}


def test_llm_brain_repeat_guard_name_only_notices_rotating_calls():
    """The brain-level guard must fire a notice when the SAME tool name is
    called repeatedly with DIFFERENT arguments (the airline task-7 pattern),
    and fold that notice into the next turn's behavior_guidance."""
    backend = MagicMock()
    backend.supports_tools = True
    raws = [
        '{"name": "get_reservation_details", "args": {"id": "AAA"}}',
        '{"name": "get_reservation_details", "args": {"id": "BBB"}}',
        '{"name": "get_reservation_details", "args": {"id": "CCC"}}',
        '{"name": "get_reservation_details", "args": {"id": "DDD"}}',
        '{"name": "get_reservation_details", "args": {"id": "EEE"}}',
        '{"name": "final_answer", "args": {"answer": "done"}}',
    ]
    backend.generate.side_effect = list(raws)
    brain = LLMBrain(backend=backend)
    # Lower the name-only threshold so the test is fast.
    brain.repeat_guard = __import__(
        "swaybot.loop_guards", fromlist=["RepeatCallGuard"]
    ).RepeatCallGuard(thresholds=[99], name_only_thresholds=[5])

    schemas = [
        {"name": "get_reservation_details", "parameters": {"type": "object"}},
        {"name": "final_answer", "parameters": {"type": "object"}},
    ]
    perception = {
        "task": "cancel a reservation",
        "step": 0,
        "max_steps": 50,
        "history": [],
        "tool_descriptions": schemas,
    }
    notices = []
    for _ in raws:
        brain.think(dict(perception), ["get_reservation_details", "final_answer"])
        sys_content = backend.generate.call_args.args[0][0]["content"]
        notices.append(
            "repeating" in sys_content.lower()
            or "repeatedly" in sys_content.lower()
            or "repeated tool call" in sys_content.lower()
        )
    # Calls 0-3: below threshold -> no notice. Call 4: 5th same-name call ->
    # notice produced, folded into call 5's system prompt.
    assert not any(notices[:4])
    assert notices[4] or notices[5]


def test_llm_brain_repeat_guard_disabled_when_none():
    """Setting repeat_guard=None must not raise and must not inject notices."""
    backend = MagicMock()
    backend.supports_tools = True
    backend.generate.return_value = '{"name": "done", "args": {}}'
    brain = LLMBrain(backend=backend)
    brain.repeat_guard = None
    perception = {
        "task": "t", "step": 0, "max_steps": 3, "history": [],
        "tool_descriptions": [{"name": "done", "parameters": {"type": "object"}}],
    }
    action = brain.think(dict(perception), ["done"])
    assert action["name"] == "done"


@patch("swaybot.models.OpenAI")
def test_llm_brain_uses_env_variables(mock_openai, monkeypatch):
    monkeypatch.setenv("SWAYBOT_API_KEY", "env-key")
    monkeypatch.setenv("SWAYBOT_API_BASE", "http://env/v1")
    monkeypatch.setenv("SWAYBOT_MODEL", "env-model")
    mock_openai.return_value = MagicMock()

    brain = LLMBrain()
    assert brain.api_key == "env-key"
    assert brain.base_url == "http://env/v1"
    assert brain.model == "env-model"
