from swaybot.loop_guards import RepeatCallGuard, _canonicalize


def _call(name="get_reservation_details", **args):
    return {"name": name, "args": args}


def test_canonicalize_sorts_keys():
    assert _canonicalize({"b": 1, "a": 2}) == _canonicalize({"a": 2, "b": 1})


def test_canonicalize_nested():
    assert _canonicalize({"x": {"b": 1, "a": 2}}) == _canonicalize({"x": {"a": 2, "b": 1}})


def test_different_args_do_not_trigger():
    guard = RepeatCallGuard(thresholds=[3])
    assert guard.observe(_call(id=1)) is None
    assert guard.observe(_call(id=2)) is None  # different args -> reset
    assert guard.observe(_call(id=2)) is None
    # now two in a row for id=2; third triggers
    assert guard.observe(_call(id=2)) is not None


def test_key_order_independent():
    guard = RepeatCallGuard(thresholds=[3])
    a = {"name": "t", "args": {"b": 1, "a": 2}}
    b = {"name": "t", "args": {"a": 2, "b": 1}}
    assert guard.observe(a) is None
    assert guard.observe(b) is None
    assert guard.observe(b) is not None  # 3rd identical (order-independent)


def test_gentle_then_detailed_escalation():
    guard = RepeatCallGuard(thresholds=[3, 5])
    c = _call(id=7)
    for _ in range(2):
        assert guard.observe(c) is None
    gentle = guard.observe(c)
    assert gentle is not None
    assert "different approach" in gentle  # gentle reminder at thresholds[0]
    assert "tool:" not in gentle  # not the detailed form
    assert guard.observe(c) is None  # 4, no threshold
    detailed = guard.observe(c)
    assert detailed is not None
    assert "tool: get_reservation_details" in detailed
    assert "consecutive_calls: 5" in detailed


def test_reset_clears_chain():
    guard = RepeatCallGuard(thresholds=[3])
    c = _call(id=1)
    guard.observe(c)
    guard.observe(c)
    guard.reset()
    assert guard.observe(c) is None  # count restarted
    assert guard.observe(c) is None


def test_action_name_change_resets():
    guard = RepeatCallGuard(thresholds=[3])
    assert guard.observe(_call(id=1)) is None
    assert guard.observe(_call(id=1)) is None
    assert guard.observe({"name": "other", "args": {"id": 1}}) is None  # name changed


def test_invalid_thresholds_rejected():
    import pytest

    with pytest.raises(ValueError):
        RepeatCallGuard(thresholds=[])
    with pytest.raises(ValueError):
        RepeatCallGuard(thresholds=[1])
    with pytest.raises(ValueError):
        RepeatCallGuard(thresholds=[3, 3])


def test_preview_truncation():
    long_args = {"q": "x" * 600}
    guard = RepeatCallGuard(thresholds=[2], arguments_preview_chars=50)
    c = {"name": "search", "args": long_args}
    guard.observe(c)
    msg = guard.observe(c)
    assert msg is not None
    assert "+5" not in msg or "more chars" in msg
    assert len(msg) < 700  # bounded
