"""Tests for benchmark adapters.

These tests verify the adapter logic without requiring the actual
benchmark packages (tau2-bench, agentdojo, bfcl-eval) to be installed.
They test the standalone helper functions from ``_helpers`` modules
which have no third-party dependencies.
"""

import json

import pytest

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

    def test_extract_message_content_assistant(self):
        """extract_message_content should handle content-based messages."""
        from benchmarks.tau3bench._helpers import extract_message_content

        class FakeMsg:
            content = "hello world"

        assert extract_message_content(FakeMsg()) == "hello world"

    def test_extract_message_content_empty(self):
        """extract_message_content should return empty string for objects with no content."""
        from benchmarks.tau3bench._helpers import extract_message_content

        class FakeMsg:
            pass

        assert extract_message_content(FakeMsg()) == ""

    def test_extract_message_content_multi_tool(self):
        """extract_message_content should aggregate MultiToolMessage tool_messages."""
        from benchmarks.tau3bench._helpers import extract_message_content

        class FakeToolMsg:
            content = "result a"

        class FakeMultiToolMsg:
            tool_messages = [FakeToolMsg(), FakeToolMsg()]
            content = ""

        result = extract_message_content(FakeMultiToolMsg())
        assert result == "result a\nresult a"

    def test_normalize_tool_args_zip_alias(self):
        """normalize_tool_args should map zip_code/zipcode to zip."""
        from benchmarks.tau3bench._helpers import normalize_tool_args

        result = normalize_tool_args(
            "find_user",
            {"zip_code": "12345", "name": "Alice"},
            ["name", "zip"],
        )
        assert result == {"name": "Alice", "zip": "12345"}

    def test_normalize_tool_args_string_encoded_array(self):
        """normalize_tool_args should decode JSON-string arrays."""
        from benchmarks.tau3bench._helpers import normalize_tool_args

        result = normalize_tool_args(
            "exchange_delivered_order_items",
            {
                "exchange_items": '["id1", "id2"]',
                "exchange_new_items": '["id3"]',
                "exchange_payment_method_id": "pm_1",
            },
            ["item_ids", "new_item_ids", "payment_method_id"],
        )
        assert result == {
            "item_ids": ["id1", "id2"],
            "new_item_ids": ["id3"],
            "payment_method_id": "pm_1",
        }

    def test_normalize_tool_args_unknown_key_passthrough(self):
        """normalize_tool_args should preserve unknown keys unchanged."""
        from benchmarks.tau3bench._helpers import normalize_tool_args

        result = normalize_tool_args(
            "foo",
            {"known": "x", "unknown_key": "y"},
            ["known"],
        )
        assert result["known"] == "x"
        assert result["unknown_key"] == "y"

    def test_normalize_tool_args_empty_params(self):
        """normalize_tool_args should return args unchanged when param list is empty."""
        from benchmarks.tau3bench._helpers import normalize_tool_args

        result = normalize_tool_args("foo", {"a": 1}, [])
        assert result == {"a": 1}

    def test_normalize_tool_args_non_dict(self):
        """normalize_tool_args should return empty dict for non-dict args."""
        from benchmarks.tau3bench._helpers import normalize_tool_args

        assert normalize_tool_args("foo", "not a dict", ["a"]) == {}

    def test_normalize_tool_args_customer_name_alias(self):
        """normalize_tool_args should map name/user_name to customer_name."""
        from benchmarks.tau3bench._helpers import normalize_tool_args

        result = normalize_tool_args(
            "get_user_information_by_name",
            {"name": "Alice Smith"},
            ["customer_name"],
        )
        assert result == {"customer_name": "Alice Smith"}

    def test_normalize_tool_args_schema_preserves_string_ids(self):
        """String-typed fields should stay strings even when numeric."""
        from benchmarks.tau3bench._helpers import normalize_tool_args

        schema = {
            "type": "object",
            "properties": {
                "product_id": {"type": "string"},
                "zip": {"type": "string"},
            },
        }
        result = normalize_tool_args(
            "get_product",
            {"product_id": "12345", "zip": "02142"},
            ["product_id", "zip"],
            schema,
        )
        assert result == {"product_id": "12345", "zip": "02142"}
        assert isinstance(result["product_id"], str)
        assert isinstance(result["zip"], str)

    def test_normalize_tool_args_schema_coerces_numeric_fields(self):
        """Integer/number fields should be coerced from strings."""
        from benchmarks.tau3bench._helpers import normalize_tool_args

        schema = {
            "type": "object",
            "properties": {
                "quantity": {"type": "integer"},
                "price": {"type": "number"},
            },
        }
        result = normalize_tool_args(
            "add_to_cart",
            {"quantity": "3", "price": "19.99"},
            ["quantity", "price"],
            schema,
        )
        assert result == {"quantity": 3, "price": 19.99}

    def test_summarize_tool_content_short_text_unchanged(self):
        """Short tool results should pass through unchanged."""
        from benchmarks.tau3bench._helpers import summarize_tool_content

        text = '{"status": "ok"}'
        assert summarize_tool_content("foo", text) == text

    def test_summarize_tool_content_generic_truncation(self):
        """Long non-JSON text should be truncated with a marker."""
        from benchmarks.tau3bench._helpers import summarize_tool_content

        long_text = "x " * 2000
        result = summarize_tool_content("foo", long_text)
        assert len(result) < len(long_text)
        assert "truncated" in result

    def test_summarize_product_variants_keeps_available_only(self):
        """Product details should keep only available variants."""
        from benchmarks.tau3bench._helpers import summarize_tool_content

        data = {
            "name": "Widget",
            "variants": {
                "a": {"available": True, "price": 10, "description": "x" * 1000},
                "b": {"available": False, "price": 20, "description": "y" * 1000},
                "c": {"available": True, "price": 30, "description": "z" * 1000},
            },
        }
        result = summarize_tool_content("get_product_details", data)
        parsed = json.loads(result)
        assert set(parsed["variants"].keys()) == {"a", "c"}

    def test_summarize_product_variants_caps_count(self):
        """Product details should cap available variants at five."""
        from benchmarks.tau3bench._helpers import summarize_tool_content

        variants = {
            f"v{i}": {"available": True, "price": i, "description": "x" * 200}
            for i in range(20)
        }
        data = {"name": "Widget", "variants": variants}
        result = summarize_tool_content("get_product_details", data)
        parsed = json.loads(result)
        assert len(parsed["variants"]) == 5
        assert "_variants_note" in parsed

    def test_summarize_user_details_truncates_orders(self):
        """User details should truncate very long order/payment lists."""
        from benchmarks.tau3bench._helpers import summarize_tool_content

        data = {
            "name": "Alice",
            "orders": {f"o{i}": {"total": i, "items": "x" * 200} for i in range(50)},
        }
        result = summarize_tool_content("get_user_details", data)
        parsed = json.loads(result)
        assert len(parsed["orders"]) == 3
        assert "_orders_note" in parsed

    def test_summarize_order_details_caps_items(self):
        """Order details should cap items at ten."""
        from benchmarks.tau3bench._helpers import summarize_tool_content

        data = {
            "order_id": "o1",
            "items": [{"id": f"item{i}", "desc": "x" * 200} for i in range(15)],
        }
        result = summarize_tool_content("get_order_details", data)
        parsed = json.loads(result)
        assert len(parsed["items"]) == 10
        assert "_items_note" in parsed

    def test_augment_tool_error_missing_args_appends_directive(self):
        """Missing-arg errors should get a directive naming the missing args."""
        from benchmarks.tau3bench._helpers import augment_tool_error

        content = (
            "Error: KnowledgeTools.log_verification() missing 7 required "
            "positional arguments: 'name', 'user_id', 'address', 'email', "
            "'phone_number', 'date_of_birth', and 'time_verified'"
        )
        params = ["name", "user_id", "address", "email", "phone_number",
                  "date_of_birth", "time_verified"]
        result = augment_tool_error("log_verification", content, params)
        assert result.startswith(content)
        assert "You must call 'log_verification' again." in result
        assert "Required arguments are:" in result
        assert "name, user_id, address" in result
        assert "'name', 'user_id', 'address'" in result

    def test_augment_tool_error_unexpected_kwarg_appends_directive(self):
        """Unexpected-kwarg errors should name the bad arg and valid args."""
        from benchmarks.tau3bench._helpers import augment_tool_error

        content = (
            "Error: update_reservation_flights() got an unexpected keyword "
            "argument 'payment_method_id'"
        )
        params = ["trip_id", "payment_id", "payment1", "payment2", "payment3"]
        result = augment_tool_error(
            "update_reservation_flights", content, params
        )
        assert result.startswith(content)
        assert "'payment_method_id' is not a valid argument" in result
        assert "Valid arguments are:" in result
        assert "payment_id" in result

    def test_augment_tool_error_passthrough_non_error(self):
        """Non-error (success) content should pass through unchanged."""
        from benchmarks.tau3bench._helpers import augment_tool_error

        content = '{"status": "ok", "order_id": "12345"}'
        result = augment_tool_error("get_order_details", content, ["order_id"])
        assert result == content

    def test_augment_tool_error_generic_fallback(self):
        """Error text without a recognizable pattern gets the generic hint."""
        from benchmarks.tau3bench._helpers import augment_tool_error

        content = "Error: something unusual happened"
        params = ["a", "b"]
        result = augment_tool_error("some_tool", content, params)
        assert result.startswith(content)
        assert "previous call to 'some_tool' failed" in result
        assert "a, b" in result

    def test_augment_tool_error_idempotent(self):
        """Augmenting an already-augmented error should not duplicate."""
        from benchmarks.tau3bench._helpers import augment_tool_error

        content = "Error: foo() missing 1 required positional argument: 'x'"
        params = ["x", "y"]
        once = augment_tool_error("foo", content, params)
        twice = augment_tool_error("foo", once, params)
        assert once == twice

    def test_is_kb_search_matches_variant_names(self):
        """The banking_knowledge 'alltools' variant exposes KB_search_bm25 /
        KB_search_dense, not bare 'KB_search'. The family check must catch all
        of them so the force-final / exhaustion logic actually fires."""
        from benchmarks.tau3bench.adapter import _is_kb_search

        assert _is_kb_search("KB_search") is True
        assert _is_kb_search("KB_search_bm25") is True
        assert _is_kb_search("KB_search_dense") is True
        # Non-KB tools must NOT match (the original bug: shell browsing ran
        # unchecked because the name check was an exact == "KB_search").
        assert _is_kb_search("shell") is False
        assert _is_kb_search("get_reservation_details") is False
        assert _is_kb_search("") is False

    def test_normalize_tool_args_reverse_alias_payment_method_id(self):
        """Airline wants payment_id; model emits payment_method_id → remapped."""
        from benchmarks.tau3bench._helpers import normalize_tool_args

        result = normalize_tool_args(
            "update_reservation_flights",
            {"trip_id": "t1", "payment_method_id": "pm_123"},
            ["trip_id", "payment_id", "payment1", "payment2", "payment3"],
        )
        assert "payment_id" in result
        assert result["payment_id"] == "pm_123"
        assert "payment_method_id" not in result

    def test_normalize_tool_args_reverse_alias_gb_amount(self):
        """Telecom refuel_data wants gb_amount; model emits amount_gb → remapped."""
        from benchmarks.tau3bench._helpers import normalize_tool_args

        result = normalize_tool_args(
            "refuel_data",
            {"customer_id": "c1", "amount_gb": 5},
            ["customer_id", "gb_amount"],
        )
        assert "gb_amount" in result
        assert result["gb_amount"] == 5
        assert "amount_gb" not in result

    def test_normalize_tool_args_no_reverse_when_variant_is_valid(self):
        """Retail payment_method_id IS valid — reverse alias must not fire."""
        from benchmarks.tau3bench._helpers import normalize_tool_args

        result = normalize_tool_args(
            "exchange_delivered_order_items",
            {"order_id": "o1", "payment_method_id": "pm_1", "item_ids": ["i1"]},
            ["order_id", "payment_method_id", "item_ids", "new_item_ids"],
        )
        assert "payment_method_id" in result
        assert result["payment_method_id"] == "pm_1"

    def test_tool_schema_from_tau2_unwraps_openai_function_wrapper(self):
        """tau2 openai_schema is {type, function:{parameters}} — must extract the inner parameters."""
        from benchmarks.tau3bench._helpers import tool_schema_from_tau2

        class FakeParams:
            @staticmethod
            def model_json_schema():
                return {"properties": {"fallback": {"type": "string"}}}

        class FakeTool:
            openai_schema = {
                "type": "function",
                "function": {
                    "name": "refuel_data",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "gb_amount": {"type": "number"},
                            "customer_id": {"type": "string"},
                        },
                        "required": ["gb_amount", "customer_id"],
                    },
                },
            }
            params = FakeParams

        schema = tool_schema_from_tau2(FakeTool())
        # Must return the inner parameters dict, not {} nor the whole wrapper.
        assert "properties" in schema
        assert "gb_amount" in schema["properties"]
        assert schema["properties"]["gb_amount"]["type"] == "number"

    def test_normalize_tool_args_coerces_number_from_openai_schema(self):
        """End-to-end: a number-typed param under the openai wrapper coerces str→float."""
        from benchmarks.tau3bench._helpers import tool_schema_from_tau2, normalize_tool_args

        class FakeTool:
            openai_schema = {
                "type": "function",
                "function": {
                    "name": "refuel_data",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "gb_amount": {"type": "number"},
                            "customer_id": {"type": "string"},
                            "line_id": {"type": "string"},
                        },
                        "required": ["customer_id", "line_id", "gb_amount"],
                    },
                },
            }
            params = None

        schema = tool_schema_from_tau2(FakeTool())
        result = normalize_tool_args(
            "refuel_data",
            {"customer_id": "C1001", "line_id": "L1002", "gb_amount": "2"},
            ["customer_id", "line_id", "gb_amount"],
            schema,
        )
        assert result["gb_amount"] == 2.0
        assert isinstance(result["gb_amount"], float)
        assert result["customer_id"] == "C1001"

    def test_tool_fallback_generate_retries_without_tools(self):
        """tau2 generate wrapper should fall back to text mode on tool rejection."""
        from benchmarks.tau3bench import _tool_fallback_generate

        calls = []

        def fake_generate(model, messages, tools=None, tool_choice=None, **kwargs):
            calls.append((tools, tool_choice))
            if tools is not None:
                raise RuntimeError(
                    '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set'
                )
            return "ok"

        result = _tool_fallback_generate(
            fake_generate, "m", [], tools=["dummy"], tool_choice="auto"
        )
        assert result == "ok"
        assert len(calls) == 2
        assert calls[0] == (["dummy"], "auto")
        assert calls[1] == (None, None)

    def test_tool_fallback_generate_reraises_non_tool_errors(self):
        """Non-tool-related errors should not trigger a retry."""
        from benchmarks.tau3bench import _tool_fallback_generate

        def fake_generate(model, messages, tools=None, tool_choice=None, **kwargs):
            raise RuntimeError("connection refused")

        with pytest.raises(RuntimeError, match="connection refused"):
            _tool_fallback_generate(fake_generate, "m", [], tools=["dummy"])


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

    def test_resolve_tool_name_maps_underscores_to_dots(self):
        """resolve_tool_name should map math_factorial back to math.factorial."""
        from benchmarks.bfcl._helpers import build_registry, resolve_tool_name

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "math.factorial",
                    "description": "Factorial",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        registry = build_registry(tools)
        assert resolve_tool_name(registry, "math_factorial") == "math.factorial"

    def test_resolve_tool_name_exact_match(self):
        """resolve_tool_name should return exact matches unchanged."""
        from benchmarks.bfcl._helpers import build_registry, resolve_tool_name

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        registry = build_registry(tools)
        assert resolve_tool_name(registry, "get_weather") == "get_weather"

    def test_normalize_tool_name(self):
        """normalize_tool_name should replace dots and dashes with underscores."""
        from benchmarks.bfcl._helpers import normalize_tool_name

        assert normalize_tool_name("math.factorial") == "math_factorial"
        assert normalize_tool_name("my-tool") == "my_tool"


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


def test_summarize_kb_search_keeps_all_entries():
    """_summarize_kb_search must preserve ALL returned documents (not just the
    top 3) — for multi-doc compare tasks the relevant card is often ranked 4th+,
    and dropping it leaves the agent unable to find the card to recommend."""
    from benchmarks.tau3bench.adapter import SwayBotHalfDuplexAgent as A

    content = """1. Gold Card
   ID: doc_gold
   Score: 10.0
   Content: - Cash back: 2.5%

2. Silver Card
   ID: doc_silver
   Score: 9.0
   Content: - Cash back: 1.0%

3. Bronze Card
   ID: doc_bronze
   Score: 8.0
   Content: - Cash back: 0.5%

4. Platinum Card
   ID: doc_plat
   Score: 7.0
   Content: - Cash back: 0.1%"""
    out = A._summarize_kb_search(content)
    assert "Platinum" in out  # was dropped when only top 3 kept
    assert "Gold" in out and "Silver" in out and "Bronze" in out
    assert "4 KB results" in out


def test_tool_residue_re_catches_bracketless_tool_prose():
    """Under force_final the model emits the blocked tool name as bracket-less
    prose (e.g. "KB_search_bm25 k 20 query ...") — _PROSE_TOOL_TAG_RE needs
    angle brackets and misses it, so _TOOL_RESIDUE_RE must catch a
    final_answer whose text *starts with* a retrieval tool name."""
    from benchmarks.tau3bench.adapter import _TOOL_RESIDUE_RE

    residue_lines = [
        "KB_search_bm25 k 20 query Bronze Rewards Card Diamond Elite Card personal",
        "KB_search_bm25 query personal credit card annual fee cash back rate k 20",
        "  KB_search_dense query Gold Rewards Card",  # leading whitespace
        "shell ls /documents/credit_cards",  # shell residue
    ]
    for line in residue_lines:
        assert _TOOL_RESIDUE_RE.search(line), f"missed residue: {line!r}"

    real_answers = [
        "Based on the Gold Rewards Card: 2.5% cash back, $0 annual fee. I recommend it.",
        "I recommend the Gold Rewards Card because it offers the best cash back rate.",
        "The KB_search_bm25 tool returns ranked documents.",  # tool name not at start
    ]
    for line in real_answers:
        assert not _TOOL_RESIDUE_RE.search(line), f"false positive: {line!r}"


def test_collect_kb_content_gathers_all_kb_results_not_just_last():
    """_collect_kb_content must walk the whole conversation and gather every
    KB_search tool result, not just the most recent one — for multi-card
    compare tasks the best-fit card is often in an earlier search result."""
    from benchmarks.tau3bench.adapter import SwayBotHalfDuplexAgent as A

    # A minimal agent instance — _collect_kb_content only uses
    # _tool_name_for_call_id (which reads self._tool_call_names; set it empty
    # so the method falls back to scanning the conversation messages).
    agent = A.__new__(A)
    agent._tool_call_names = {}

    messages = [
        {"role": "user", "content": "which credit card?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "KB_search_bm25", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "Gold Rewards Card: 2.5% cash back"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "KB_search_bm25", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_2", "content": "Silver Rewards Card: 1.0% cash back"},
        # A non-KB tool result must be excluded.
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_3",
                    "type": "function",
                    "function": {"name": "get_user_details", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_3", "content": "user data"},
        # A duplicate of call_1 must be de-duplicated.
        {"role": "tool", "tool_call_id": "call_1", "content": "Gold Rewards Card: 2.5% cash back"},
    ]
    out = agent._collect_kb_content(messages)
    assert "Gold Rewards Card" in out
    assert "Silver Rewards Card" in out
    assert "user data" not in out  # non-KB tool excluded
    # De-duplication: Gold content appears only once.
    assert out.count("Gold Rewards Card: 2.5% cash back") == 1



def test_summarize_kb_search_preserves_key_terms():
    """_summarize_kb_search must keep comparison-critical lines (cash back,
    annual fee, etc.) even when they fall beyond the first 3 content lines —
    a naive first-N truncation dropped Gold's "$0 annual fee / 2.5% cash back"
    and caused the agent to recommend the wrong card."""
    from benchmarks.tau3bench.adapter import SwayBotHalfDuplexAgent as A

    content = """1. Gold Rewards Card: Apply for a Premium Credit Card
   ID: doc_gold
   Score: 13.3
   Content: ## Eligibility and Key Terms
- Minimum credit score required: $720
- Rho-Bank+ premium subscription required: Yes
- Annual fee: $0.00
- Cash back on all purchases: 2.5%
- APR on carried balances: 17.99%
- Foreign transaction fee: 0%
- Late payment fee: $32.50"""
    out = A._summarize_kb_search(content)
    assert "Annual fee: $0.00" in out
    assert "Cash back on all purchases: 2.5%" in out
    assert "Foreign transaction fee: 0%" in out
    assert "Rho-Bank+" in out or "Rho‑Bank+" in out
