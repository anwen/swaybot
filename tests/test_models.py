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
    assert lines[1]["tool_names"] == ["done"]
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
