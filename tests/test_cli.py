from pathlib import Path

from swaybot.cli import _default_data_dir, main
from swaybot.memory import MemoryStore


def test_default_data_dir_is_dot_swaybot():
    assert _default_data_dir() == Path.home() / ".swaybot"


def test_cli_default_memory_path(tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "swaybot"
    monkeypatch.setenv("SWAYBOT_DATA_DIR", str(data_dir))
    main(["hello", "--max-steps", "2", "--data-dir", str(data_dir)])
    captured = capsys.readouterr()
    assert "Step 1:" in captured.out
    assert "Step 2:" in captured.out
    memory_path = data_dir / "memory.json"
    assert memory_path.exists()
    store = MemoryStore(path=memory_path)
    assert len(store.memories) == 5  # task + 2 actions + 2 observations
    assert all(m.scope == "short_term" for m in store.memories)
    assert any(m.source == "user" for m in store.memories)


def test_cli_no_memory(tmp_path, capsys):
    main(["hello", "--max-steps", "2", "--data-dir", str(tmp_path / "empty"), "--no-memory"])
    captured = capsys.readouterr()
    assert "Step 1:" in captured.out
    memory_path = tmp_path / "empty" / "memory.json"
    assert not memory_path.exists()


def test_cli_plan_flag_creates_planning_step(tmp_path, capsys):
    from swaybot.memory import MemoryStore, PlanningStep

    data_dir = tmp_path / "plan"
    main(["hello", "--max-steps", "2", "--data-dir", str(data_dir), "--plan"])
    captured = capsys.readouterr()
    assert "Step 1:" in captured.out
    memory_path = data_dir / "memory.json"
    store = MemoryStore(path=memory_path)
    assert any(isinstance(m, PlanningStep) for m in store.memories)


def test_cli_explore_flag_generates_and_runs_task(tmp_path, capsys):
    from swaybot.memory import MemoryStore

    data_dir = tmp_path / "explore"
    main(["--explore", "--max-steps", "2", "--data-dir", str(data_dir)])
    captured = capsys.readouterr()
    assert "Exploration:" in captured.out
    assert "Step 1:" in captured.out
    memory_path = data_dir / "memory.json"
    store = MemoryStore(path=memory_path)
    assert any(m.scope == "long_term" for m in store.memories)


def test_cli_verbose_output(tmp_path, capsys):
    main(["hello", "--max-steps", "1", "--data-dir", str(tmp_path / "verb"), "--verbose"])
    captured = capsys.readouterr()
    assert "Step 1:" in captured.out
    assert "'name':" in captured.out or '"name":' in captured.out


def test_cli_writes_run_log_and_inspect_last(tmp_path, capsys):
    data_dir = tmp_path / "inspect"
    main(["hello", "--max-steps", "1", "--data-dir", str(data_dir)])
    capsys.readouterr()

    run_log = data_dir / "runs.jsonl"
    assert run_log.exists()

    exit_code = main(["inspect", "--last", "--data-dir", str(data_dir)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Task: hello" in captured.out
    assert "Steps: 1" in captured.out
    assert "Totals:" in captured.out
    assert "Step 1:" in captured.out


def test_cli_inspect_survives_reflection_pruning(tmp_path, capsys):
    data_dir = tmp_path / "reflect-inspect"
    main(["hello", "--max-steps", "1", "--data-dir", str(data_dir), "--reflect"])
    capsys.readouterr()

    store = MemoryStore(path=data_dir / "memory.json")
    assert store.query(scope="short_term", tag="hello") == []

    exit_code = main(["inspect", "--task", "hell", "--data-dir", str(data_dir)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Task: hello" in captured.out
    assert "Reflections:" in captured.out
