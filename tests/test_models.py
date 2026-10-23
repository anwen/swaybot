from swaybot.models import (
    FallbackModel,
    Model,
    RecordingModel,
    ScriptedModelError,
    ScriptedModel,
)


class FakeModel:
    def __init__(self, output: str | None = None, fail: bool = False):
        self.output = output
        self.fail = fail
        self.calls = 0

    def generate(
        self,
        _messages: list[dict],
        metadata: dict | None = None,
        tools: list[dict] | None = None,
        stream_callback=None,
        reasoning_callback=None,
    ) -> str | None:
        self.calls += 1
        if self.fail:
            raise RuntimeError("failed")
        return self.output


def test_fake_model_satisfies_protocol():
    assert isinstance(FakeModel(), Model)


def test_fallback_model_tries_primary_first():
    primary = FakeModel(output="primary")
    fallback = FakeModel(output="fallback")
    model = FallbackModel([primary, fallback])
    assert model.generate([{"role": "user", "content": "hi"}]) == "primary"
    assert primary.calls == 1
    assert fallback.calls == 0


def test_fallback_model_uses_backup_when_primary_fails():
    primary = FakeModel(fail=True)
    fallback = FakeModel(output="fallback")
    model = FallbackModel([primary, fallback])
    assert model.generate([{"role": "user", "content": "hi"}]) == "fallback"
    assert primary.calls == 1
    assert fallback.calls == 1


def test_fallback_model_returns_none_when_all_fail():
    model = FallbackModel([FakeModel(fail=True), FakeModel(fail=True)])
    assert model.generate([{"role": "user", "content": "hi"}]) is None


def test_scripted_model_satisfies_protocol():
    assert isinstance(ScriptedModel([{"text": "hi"}]), Model)


def test_scripted_model_replays_text_and_action_steps():
    model = ScriptedModel(
        [
            {"text": "thinking..."},
            {"action": {"name": "echo", "args": {"message": "hi"}}},
        ]
    )
    text_step = model.generate([{"role": "user", "content": "q"}])
    assert text_step.text == "thinking..."
    assert text_step.tool_call is None
    action_step = model.generate([{"role": "user", "content": "q"}])
    assert action_step.tool_call is not None
    assert action_step.tool_call.name == "echo"
    assert action_step.tool_call.args == {"message": "hi"}
    assert model.remaining == 0
    assert len(model.calls) == 2


def test_scripted_model_exhaustion_raises_instead_of_none():
    model = ScriptedModel([{"text": "only one"}])
    model.generate([{"role": "user", "content": "q"}])
    try:
        model.generate([{"role": "user", "content": "q"}])
    except ScriptedModelError as exc:
        assert "exhausted" in str(exc)
    else:
        raise AssertionError("expected ScriptedModelError on exhaustion")


def test_scripted_model_raise_step_drives_fallback():
    model = ScriptedModel([{"raise": RuntimeError("boom")}])
    chain = FallbackModel([model, FakeModel(output="recovered")])
    # The scripted model raises; the chain falls through to the next model.
    assert chain.generate([{"role": "user", "content": "q"}]) == "recovered"
    assert model.remaining == 0


def test_scripted_model_streams_text_in_chunks():
    chunks: list[str] = []
    model = ScriptedModel([{"text": "0123456789abcdef"}])
    out = model.generate(
        [{"role": "user", "content": "q"}], stream_callback=chunks.append
    )
    assert out.text == "0123456789abcdef"
    assert "".join(chunks) == out.text
    assert len(chunks) > 1  # chunked, exercising streaming consumers


def test_scripted_model_records_metadata():
    metadata: dict = {}
    ScriptedModel([{"text": "hi"}]).generate(
        [{"role": "user", "content": "q"}], metadata=metadata
    )
    assert metadata["raw_output"] == "hi"
    assert "duration_ms" in metadata
    assert metadata["model_input_messages"][0]["content"] == "q"


def test_scripted_model_rejects_malformed_step():
    try:
        ScriptedModel([{"text": "a", "action": {"name": "x"}}])
    except ScriptedModelError:
        pass
    else:
        raise AssertionError("expected ScriptedModelError for ambiguous step")


def test_recording_model_writes_one_line_per_call(tmp_path):
    import json

    model = RecordingModel(
        ScriptedModel(
            [
                {"text": "hello"},
                {"action": {"name": "done", "args": {}}},
            ]
        ),
        tmp_path,
    )
    assert model.generate([{"role": "user", "content": "q"}]).text == "hello"
    model.generate([{"role": "user", "content": "q"}], tools=[{"name": "done"}])
    files = list(tmp_path.glob("model-io-*.jsonl"))
    assert len(files) == 1
    lines = [json.loads(l) for l in files[0].read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["text"] == "hello"
    assert lines[0]["tool_call"] is None
    assert lines[0]["message_count"] == 1
    assert lines[0]["tools"] == []
    assert lines[1]["tools"][0]["name"] == "done"
    assert lines[1]["tool_call"]["name"] == "done"
    assert lines[1]["error"] is None


def test_recording_model_records_failure_without_swallowing(tmp_path):
    import json
    import pytest as _pytest

    model = RecordingModel(ScriptedModel([{"raise": RuntimeError("boom")}]), tmp_path)
    with _pytest.raises(RuntimeError):
        model.generate([{"role": "user", "content": "q"}])
    lines = [json.loads(l) for l in tmp_path.glob("model-io-*.jsonl").__next__().read_text().splitlines()]
    assert "RuntimeError: boom" in lines[0]["error"]


def test_recording_model_redacts_base_url_to_host(tmp_path):
    import json

    class _UrlModel(ScriptedModel):
        base_url = "https://gw.example.com/v1?apikey=secret"

    model = RecordingModel(_UrlModel([{"text": "hi"}]), tmp_path)
    model.generate([{"role": "user", "content": "q"}])
    line = json.loads(tmp_path.glob("model-io-*.jsonl").__next__().read_text().splitlines()[0])
    assert line["base_url_host"] == "https://gw.example.com"
    assert "secret" not in json.dumps(line)


def test_llm_brain_auto_wraps_with_rollout_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SWAYBOT_ROLLOUT_DIR", str(tmp_path))
    from swaybot.llm_brain import LLMBrain

    brain = LLMBrain(backend=ScriptedModel([{"text": "hi"}]))
    assert isinstance(brain._model, RecordingModel)
    brain.think(
        {"task": "t", "step": 0, "max_steps": 3, "history": [], "tool_descriptions": []},
        ["final_answer"],
    )
    assert list(tmp_path.glob("model-io-*.jsonl"))


def test_text_mode_fallback_logs_event(tmp_path):
    """自动降级必须显形（PRINCIPLES 移除标准依赖这个事件可观察）。"""
    import io
    import json
    import logging
    from unittest.mock import MagicMock, patch

    import swaybot.models as models_mod
    from swaybot.logging import JSONFormatter

    logger = logging.getLogger("swaybot.events")
    original_handlers, original_propagate = list(logger.handlers), logger.propagate
    logger.handlers, logger.propagate = [], False
    logger.setLevel(logging.INFO)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    try:
        with patch.object(models_mod, "OpenAI") as mock_openai:
            model = models_mod.OpenAIModel(
                api_key="k", base_url="http://x/v1", model="m", max_retries=1
            )
            ok = MagicMock()
            ok.choices = [MagicMock()]
            ok.choices[0].message.content = "done"
            ok.choices[0].finish_reason = "stop"
            ok.usage = None
            model._client.chat.completions.create = MagicMock(
                side_effect=[
                    RuntimeError("tool choice requires enable-auto-tool-choice"),
                    ok,
                ]
            )
            response = model.generate(
                [{"role": "user", "content": "q"}], tools=[{"name": "done"}]
            )
            assert response is not None and response.text == "done"
        lines = [l for l in stream.getvalue().splitlines() if l.strip()]
        assert lines, "fallback 事件未落日志"
        payload = json.loads(lines[-1])
        assert payload["event"]["type"] == "model.text_mode_fallback"
        assert "enable-auto-tool-choice" in payload["event"]["error"]
    finally:
        logger.handlers, logger.propagate = original_handlers, original_propagate


def test_recording_model_ids_and_usage(tmp_path):
    import json

    model = RecordingModel(
        ScriptedModel(
            [
                {"text": "one"},
                {"action": {"name": "done", "args": {}}},
            ]
        ),
        tmp_path,
    )
    session = model._session_id
    model.generate([{"role": "user", "content": "q"}])
    model.generate([{"role": "user", "content": "q"}])
    lines = [json.loads(l) for l in tmp_path.glob("model-io-*.jsonl").__next__().read_text().splitlines()]
    assert lines[0]["session_id"] == lines[1]["session_id"] == session
    assert [l["call_index"] for l in lines] == [1, 2]
    assert all("started_at" in l and "ts" in l for l in lines)
    # usage 来自 metadata.token_usage（OpenAIModel 约定），结构化提升到顶层
    assert "metadata" not in lines[0]
    assert "tool_names" not in lines[0]


def test_recording_model_captures_reasoning(tmp_path):
    import json

    class ReasoningModel(ScriptedModel):
        def generate(self, messages, metadata=None, tools=None, stream_callback=None, reasoning_callback=None):
            if reasoning_callback:
                reasoning_callback("I should compute 1+1 carefully.")
            return super().generate(messages, metadata=metadata, tools=tools)

    model = RecordingModel(ReasoningModel([{"text": "2"}]), tmp_path)
    assert model.generate([{"role": "user", "content": "q"}]).text == "2"
    line = json.loads(tmp_path.glob("model-io-*.jsonl").__next__().read_text().splitlines()[0])
    assert line["reasoning"] == "I should compute 1+1 carefully."
