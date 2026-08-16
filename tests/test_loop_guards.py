from swaybot.loop_guards import NonConvergenceGuard, RepeatCallGuard, _canonicalize


def _call(name="get_reservation_details", **args):
    return {"name": name, "args": args}


def test_canonicalize_sorts_keys():
    assert _canonicalize({"b": 1, "a": 2}) == _canonicalize({"a": 2, "b": 1})


def test_canonicalize_nested():
    assert _canonicalize({"x": {"b": 1, "a": 2}}) == _canonicalize({"x": {"a": 2, "b": 1}})


def test_different_args_do_not_trigger_exact():
    """Different args must not trigger the EXACT-repeat channel (default thresholds)."""
    guard = RepeatCallGuard(thresholds=[3])
    assert guard.observe(_call(id=1)) is None
    assert guard.observe(_call(id=2)) is None  # different args -> exact chain resets
    assert guard.observe(_call(id=2)) is None
    # now two in a row for id=2; third triggers exact
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
    # name_only_thresholds validated the same way
    with pytest.raises(ValueError):
        RepeatCallGuard(name_only_thresholds=[])
    with pytest.raises(ValueError):
        RepeatCallGuard(name_only_thresholds=[1, 1])


def test_preview_truncation():
    long_args = {"q": "x" * 600}
    guard = RepeatCallGuard(thresholds=[2], arguments_preview_chars=50)
    c = {"name": "search", "args": long_args}
    guard.observe(c)
    msg = guard.observe(c)
    assert msg is not None
    assert "+5" not in msg or "more chars" in msg
    assert len(msg) < 700  # bounded


# --- name-only channel (catches argument rotation) -------------------------


def test_name_only_fires_on_rotating_args():
    """The airline-task-7 pattern: same tool name, different args every call.
    Exact-repeat must NOT fire; name-only must fire at its threshold."""
    guard = RepeatCallGuard(thresholds=[10], name_only_thresholds=[5])
    ids = ["XEHM4B", "59XX6W", "7WPL39", "3EMQJ6", "ZZZ999"]
    notices = []
    for i in ids[:4]:
        notices.append(guard.observe(_call(id=i)))
    assert not any(notices)  # first 4: below name-only threshold of 5
    fifth = guard.observe(_call(id="LAST1"))
    assert fifth is not None
    assert "repeatedly" in fifth or "repeating" in fifth.lower()


def test_name_only_gentle_then_detailed():
    guard = RepeatCallGuard(thresholds=[99], name_only_thresholds=[5, 7])
    for i in range(4):
        assert guard.observe(_call(id=i)) is None
    gentle = guard.observe(_call(id=4))
    assert gentle is not None
    assert "different arguments" in gentle
    assert guard.observe(_call(id=5)) is None  # 6, no threshold
    detailed = guard.observe(_call(id=6))
    assert detailed is not None
    assert "consecutive_calls: 7" in detailed


def test_exact_takes_precedence_over_name_only():
    """When both channels fire on the same call, exact-repeat wins (more specific)."""
    guard = RepeatCallGuard(thresholds=[3], name_only_thresholds=[3])
    c = _call(id=1)
    guard.observe(c)
    guard.observe(c)
    msg = guard.observe(c)  # both exact (3) and name-only (3) hit
    assert "exact same tool call" in msg  # exact gentle reminder, not name-only


def test_name_only_resets_on_name_change():
    guard = RepeatCallGuard(thresholds=[99], name_only_thresholds=[3])
    guard.observe(_call(id=1))
    guard.observe({"name": "other_tool", "args": {"id": 1}})  # name changes
    # name-only counter restarted for "other_tool"
    assert guard.observe({"name": "other_tool", "args": {"id": 2}}) is None
    assert guard.observe({"name": "other_tool", "args": {"id": 3}}) is not None


def test_reset_clears_name_only_chain():
    guard = RepeatCallGuard(thresholds=[99], name_only_thresholds=[3])
    guard.observe(_call(id=1))
    guard.observe(_call(id=2))
    guard.reset()
    assert guard.observe(_call(id=3)) is None
    assert guard.observe(_call(id=4)) is None


# --- NonConvergenceGuard ----------------------------------------------------


def test_nonconvergence_fires_after_threshold_tool_calls():
    """A run of N tool calls with no final_answer must nudge at the threshold."""
    guard = NonConvergenceGuard(thresholds=[5])
    for _ in range(4):
        assert guard.observe(_call(id=1)) is None
    msg = guard.observe(_call(id=1))  # 5th non-terminal call
    assert msg is not None
    assert "final_answer" in msg


def test_nonconvergence_rotating_tools_still_fires():
    """The airline-task-7 pattern: DIFFERENT tools, no final_answer. This is the
    case RepeatCallGuard misses (no single tool repeats) but NonConvergenceGuard
    must catch."""
    guard = NonConvergenceGuard(thresholds=[4])
    seq = [
        {"name": "get_reservation_details", "args": {"id": "A"}},
        {"name": "get_flight_status", "args": {}},
        {"name": "get_reservation_details", "args": {"id": "B"}},
        {"name": "get_flight_status", "args": {}},
    ]
    notices = [guard.observe(a) for a in seq]
    assert not any(notices[:3])
    assert notices[3] is not None  # 4th non-terminal -> nudge
    assert "converge" in notices[3].lower() or "final_answer" in notices[3]


def test_nonconvergence_resets_on_final_answer():
    """A final_answer resets the counter; subsequent tool calls start from 0."""
    guard = NonConvergenceGuard(thresholds=[5])
    for _ in range(4):
        guard.observe(_call(id=1))
    assert guard.observe({"name": "final_answer", "args": {"answer": "x"}}) is None
    # Counter reset; 4 more tool calls should NOT fire (threshold is 5).
    for _ in range(4):
        assert guard.observe(_call(id=1)) is None
    assert guard.observe(_call(id=1)) is not None  # 5th after reset


def test_nonconvergence_resets_on_done():
    """done is also terminal."""
    guard = NonConvergenceGuard(thresholds=[3])
    guard.observe(_call(id=1))
    guard.observe(_call(id=2))
    assert guard.observe({"name": "done", "args": {}}) is None  # reset
    assert guard.observe(_call(id=1)) is None
    assert guard.observe(_call(id=1)) is None  # only 2 since reset


def test_nonconvergence_gentle_then_detailed():
    guard = NonConvergenceGuard(thresholds=[5, 8])
    for _ in range(4):
        assert guard.observe(_call(id=1)) is None
    gentle = guard.observe(_call(id=1))
    assert gentle is not None
    assert "several tools" in gentle  # gentle form
    guard.observe(_call(id=1))  # 6
    guard.observe(_call(id=1))  # 7
    detailed = guard.observe(_call(id=1))  # 8
    assert detailed is not None
    assert "8 consecutive" in detailed


def test_nonconvergence_reset_clears_counter():
    guard = NonConvergenceGuard(thresholds=[3])
    guard.observe(_call(id=1))
    guard.observe(_call(id=2))
    guard.reset()
    assert guard.observe(_call(id=1)) is None
    assert guard.observe(_call(id=1)) is None  # only 2 since reset


def test_nonconvergence_invalid_thresholds():
    import pytest

    with pytest.raises(ValueError):
        NonConvergenceGuard(thresholds=[])
    with pytest.raises(ValueError):
        NonConvergenceGuard(thresholds=[1])
    with pytest.raises(ValueError):
        NonConvergenceGuard(thresholds=[3, 3])
