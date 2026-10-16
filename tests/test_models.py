from swaybot.models import FallbackModel, Model, ScriptedModel, ScriptedModelError


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
    import json

    model = ScriptedModel(
        [
            {"text": "thinking..."},
            {"action": {"name": "echo", "args": {"message": "hi"}}},
        ]
    )
    assert model.generate([{"role": "user", "content": "q"}]) == "thinking..."
    action = json.loads(model.generate([{"role": "user", "content": "q"}]))
    assert action == {"name": "echo", "args": {"message": "hi"}}
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
    assert out == "0123456789abcdef"
    assert "".join(chunks) == out
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
