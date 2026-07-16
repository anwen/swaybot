from swaybot.models import FallbackModel, Model


class FakeModel:
    def __init__(self, output: str | None = None, fail: bool = False):
        self.output = output
        self.fail = fail
        self.calls = 0

    def generate(
        self,
        _messages: list[dict],
        metadata: dict | None = None,
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
