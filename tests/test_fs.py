import os

import pytest

from swaybot.tools.fs import FileSystem, WorkspaceError, list_directory, read_file, search_files, write_file


@pytest.fixture
def fs(tmp_path, monkeypatch):
    monkeypatch.setenv("SWAYBOT_WORKSPACE", str(tmp_path))
    # Rebuild the module-level filesystem so it picks up the env var.
    import swaybot.tools.fs as fs_module

    fs_module._fs = FileSystem(str(tmp_path))
    return fs_module._fs


def test_read_file(fs):
    fs.root.joinpath("hello.txt").write_text("world", encoding="utf-8")
    assert read_file(path="hello.txt") == "world"


def test_write_file(fs):
    result = write_file(path="dir/sub.txt", content="data")
    assert "Wrote 4 characters" in result
    assert fs.root.joinpath("dir/sub.txt").read_text(encoding="utf-8") == "data"


def test_list_directory(fs):
    fs.root.joinpath("a.txt").touch()
    fs.root.joinpath("b").mkdir()
    items = list_directory(path=".")
    assert "a.txt" in items
    assert "b/" in items


def test_search_files(fs):
    fs.root.joinpath("foo.py").touch()
    fs.root.joinpath("bar.txt").touch()
    assert search_files(query="*.py", path=".") == ["foo.py"]


def test_path_traversal_blocked(fs):
    with pytest.raises(WorkspaceError):
        read_file(path="../outside.txt")


def test_read_missing_file(fs):
    assert "is not a file" in read_file(path="missing.txt")
