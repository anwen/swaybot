from pathlib import Path

from swaybot.run_log import (
    append_run,
    build_run_record,
    format_run,
    load_runs,
    run_log_path_for_memory,
    select_runs,
)


def test_run_log_round_trip(tmp_path: Path):
    path = tmp_path / "runs.jsonl"
    steps = [
        {
            "step": 1,
            "action": {"name": "echo", "args": {"message": "hi"}},
            "result": "hi",
        }
    ]
    record = build_run_record(
        "demo",
        3,
        steps,
        hypothesis="echo preserves input",
        reflections=["echo returned the same message"],
    )
    append_run(path, record)
    append_run(path, {"task": "other", "steps": []})

    runs = load_runs(path)
    assert len(runs) == 2
    selected = select_runs(runs, task="demo", limit=1)
    assert selected[0]["hypothesis"] == "echo preserves input"
    assert selected[0]["reflections"] == ["echo returned the same message"]
    assert run_log_path_for_memory(tmp_path / "memory.json") == tmp_path / "runs.jsonl"


def test_format_run_includes_monitoring_details():
    record = build_run_record(
        "compute",
        3,
        [
            {
                "step": 1,
                "action": {"name": "add", "args": {"a": 2, "b": 2}},
                "result": "4",
                "raw_output": '{"name": "add", "args": {"a": 2, "b": 2}}',
                "token_usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 5,
                    "total_tokens": 8,
                },
                "duration_ms": 12.5,
            },
            {
                "step": 2,
                "action": {"name": "final_answer", "args": {"answer": "4"}},
                "result": "4",
                "error": "temporary failure",
            },
        ],
    )

    output = format_run(record)
    assert "Task: compute" in output
    assert "Final answer: 4" in output
    assert "raw output:" in output
    assert "tokens: prompt=3 completion=5 total=8" in output
    assert "duration: 12.5 ms" in output
    assert "Totals: prompt=3 completion=5 total=8 duration=12.5 ms" in output
    assert "error: temporary failure" in output
