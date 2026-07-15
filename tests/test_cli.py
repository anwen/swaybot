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
    assert len(store.memories) == 2
    assert all(m.scope == "short_term" for m in store.memories)


def test_cli_no_memory(tmp_path, capsys):
    main(["hello", "--max-steps", "2", "--data-dir", str(tmp_path / "empty"), "--no-memory"])
    captured = capsys.readouterr()
    assert "Step 1:" in captured.out
    memory_path = tmp_path / "empty" / "memory.json"
    assert not memory_path.exists()


def test_cli_verbose_output(tmp_path, capsys):
    main(["hello", "--max-steps", "1", "--data-dir", str(tmp_path / "verb"), "--verbose"])
    captured = capsys.readouterr()
    assert "Step 1:" in captured.out
    assert "'name':" in captured.out or '"name":' in captured.out
