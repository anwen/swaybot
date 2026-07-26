"""Tests for benchmark adapters.

These tests verify the adapter logic without requiring the actual
benchmark packages (tau2-bench, agentdojo, bfcl-eval) to be installed.
They test the standalone helper functions from ``_helpers`` modules
which have no third-party dependencies.
"""


# ---------------------------------------------------------------------------
# tau3-bench helper tests
# ---------------------------------------------------------------------------


class TestTau3BenchHelpers:
    """Test tau3-bench standalone helpers."""

    def test_placeholder_tool_fn(self):
        """Placeholder functions should be callable and return a marker string."""
        from benchmarks.tau3bench._helpers import make_placeholder_fn

        fn = make_placeholder_fn("send_money")
        assert fn.__name__ == "send_money"
        result = fn(recipient="foo", amount=100)
        assert "send_money" in result
        assert "tau2 environment" in result

    def test_extract_user_content_assistant(self):
        """extract_user_content should handle AssistantMessage-like objects."""
        from benchmarks.tau3bench._helpers import extract_user_content

        class FakeMsg:
            content = "hello world"

        assert extract_user_content(FakeMsg()) == "hello world"

    def test_extract_user_content_empty(self):
        """extract_user_content should return empty string for objects with no content."""
        from benchmarks.tau3bench._helpers import extract_user_content

        class FakeMsg:
            pass

        assert extract_user_content(FakeMsg()) == ""

    def test_extract_user_content_tool_calls(self):
        """extract_user_content should handle MultiToolMessage-like objects."""
        from benchmarks.tau3bench._helpers import extract_user_content

        class FakeFn:
            arguments = '{"key": "val"}'

        class FakeTC:
            function = FakeFn()

        class FakeMsg:
            tool_calls = [FakeTC()]
            content = ""

        result = extract_user_content(FakeMsg())
        assert "key" in result


# ---------------------------------------------------------------------------
# AgentDojo helper tests
# ---------------------------------------------------------------------------


class TestAgentDojoHelpers:
    """Test AgentDojo standalone helpers."""

    def test_build_registry_from_functions(self):
        """build_registry_from_functions should create valid tool schemas."""
        from benchmarks.agentdojo._helpers import build_registry_from_functions

        class FakeFunc:
            name = "get_balance"
            description = "Get the account balance"
            parameters = {
                "type": "object",
                "properties": {"account_id": {"type": "string"}},
                "required": ["account_id"],
            }

        registry = build_registry_from_functions({"get_balance": FakeFunc()})
        assert "get_balance" in registry.names()
        schemas = registry.schemas()
        assert any(s["name"] == "get_balance" for s in schemas)

    def test_build_registry_from_functions_no_params(self):
        """build_registry_from_functions should handle objects without parameters."""
        from benchmarks.agentdojo._helpers import build_registry_from_functions

        class FakeFunc:
            name = "simple_action"
            description = "Do something"

        registry = build_registry_from_functions({"simple_action": FakeFunc()})
        assert "simple_action" in registry.names()

    def test_build_registry_from_functions_pydantic_params(self):
        """build_registry_from_functions should extract JSON schema from Pydantic parameter models."""
        from benchmarks.agentdojo._helpers import build_registry_from_functions

        class FakeParams:
            @classmethod
            def model_json_schema(cls):
                return {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                }

        class FakeFunc:
            name = "read_file"
            description = "Reads a file"
            parameters = FakeParams

        registry = build_registry_from_functions({"read_file": FakeFunc()})
        schemas = registry.schemas()
        schema = next(s for s in schemas if s["name"] == "read_file")
        assert schema["parameters"]["properties"]["file_path"]["type"] == "string"
        assert "file_path" in schema["parameters"]["required"]

    def test_build_registry_from_functions_pydantic_v1_params(self):
        """build_registry_from_functions should fall back to Pydantic v1 schema()."""
        from benchmarks.agentdojo._helpers import build_registry_from_functions

        class FakeParams:
            @classmethod
            def schema(cls):
                return {
                    "type": "object",
                    "properties": {"amount": {"type": "number"}},
                    "required": ["amount"],
                }

        class FakeFunc:
            name = "send_money"
            description = "Send money"
            parameters = FakeParams

        registry = build_registry_from_functions({"send_money": FakeFunc()})
        schemas = registry.schemas()
        schema = next(s for s in schemas if s["name"] == "send_money")
        assert schema["parameters"]["properties"]["amount"]["type"] == "number"

    def test_messages_to_history(self):
        """messages_to_history should convert chat messages to history dicts."""
        from benchmarks.agentdojo._helpers import messages_to_history

        messages = [
            {"role": "user", "content": "Send $100 to Alice"},
            {"role": "assistant", "content": "I'll send that for you."},
        ]
        history = messages_to_history(messages)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert "Send $100" in history[0]["content"]

    def test_messages_to_history_with_content_blocks(self):
        """messages_to_history should handle list content (content blocks)."""
        from benchmarks.agentdojo._helpers import messages_to_history

        class FakeBlock:
            text = "Hello from block"

        messages = [
            {"role": "assistant", "content": [FakeBlock()]},
        ]
        history = messages_to_history(messages)
        assert "Hello from block" in history[0]["content"]

    def test_messages_to_run_history_pairs_tool_call_and_result(self):
        """messages_to_run_history should pair assistant tool calls with tool results."""
        from benchmarks.agentdojo._helpers import messages_to_run_history

        class FakeToolCall:
            function = "read_file"
            args = {"file_path": "bill.txt"}

        messages = [
            {"role": "assistant", "tool_calls": [FakeToolCall()]},
            {"role": "tool", "content": "Bill amount: 100"},
        ]
        history = messages_to_run_history(messages)
        assert len(history) == 1
        assert history[0]["step"] == 1
        assert history[0]["action"] == {"name": "read_file", "args": {"file_path": "bill.txt"}}
        assert "100" in history[0]["result"]

    def test_messages_to_run_history_handles_final_answer(self):
        """messages_to_run_history should record a text-only assistant message as final_answer."""
        from benchmarks.agentdojo._helpers import messages_to_run_history

        messages = [
            {"role": "assistant", "content": "Payment completed."},
        ]
        history = messages_to_run_history(messages)
        assert len(history) == 1
        assert history[0]["action"]["name"] == "final_answer"
        assert "Payment completed" in history[0]["action"]["args"]["answer"]

    def test_placeholder_fn(self):
        """make_placeholder_fn should create a no-op function."""
        from benchmarks.agentdojo._helpers import make_placeholder_fn

        fn = make_placeholder_fn("send_email")
        assert fn.__name__ == "send_email"
        result = fn(to="alice@example.com")
        assert "send_email" in result
        assert "AgentDojo runtime" in result


# ---------------------------------------------------------------------------
# BFCL helper tests
# ---------------------------------------------------------------------------


class TestBFCLHelpers:
    """Test BFCL standalone helpers."""

    def test_build_registry(self):
        """build_registry should map OpenAI tool defs to SwayBot tools."""
        from benchmarks.bfcl._helpers import build_registry

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ]
        registry = build_registry(tools)
        assert "get_weather" in registry.names()

    def test_extract_task(self):
        """extract_task should return the last user message."""
        from benchmarks.bfcl._helpers import extract_task

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "Let me check."},
            {"role": "user", "content": "In Tokyo please"},
        ]
        assert extract_task(messages) == "In Tokyo please"

    def test_extract_task_no_user(self):
        """extract_task should return empty string if no user messages."""
        from benchmarks.bfcl._helpers import extract_task

        messages = [
            {"role": "system", "content": "You are helpful."},
        ]
        assert extract_task(messages) == ""

    def test_extract_task_multipart(self):
        """extract_task should handle list content (multi-part)."""
        from benchmarks.bfcl._helpers import extract_task

        messages = [
            {"role": "user", "content": [{"text": "Hello"}, {"text": "world"}]},
        ]
        result = extract_task(messages)
        assert "Hello" in result
        assert "world" in result

    def test_messages_to_history(self):
        """messages_to_history should convert OpenAI messages."""
        from benchmarks.bfcl._helpers import messages_to_history

        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Tokyo"}',
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": "Sunny, 22°C",
            },
        ]
        history = messages_to_history(messages)
        assert len(history) == 3
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        assert history[1]["action"]["name"] == "get_weather"
        assert history[2]["role"] == "tool"
        assert "Sunny" in history[2]["content"]

    def test_noop_fn(self):
        """make_noop_fn should return empty string."""
        from benchmarks.bfcl._helpers import make_noop_fn

        fn = make_noop_fn("search")
        assert fn.__name__ == "search"
        assert fn() == ""


# ---------------------------------------------------------------------------
# BFCL server integration (uses _helpers internally)
# ---------------------------------------------------------------------------


class TestBFCLServer:
    """Test BFCL server's internal wiring delegates to _helpers."""

    def test_server_uses_helpers(self):
        """Server's create_app should import and use _helpers functions."""
        from benchmarks.bfcl import server
        from benchmarks.bfcl._helpers import build_registry, extract_task, messages_to_history

        # Server module should have imported these from _helpers
        assert server.build_registry is build_registry
        assert server.extract_task is extract_task
        assert server.messages_to_history is messages_to_history
