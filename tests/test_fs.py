import os

import pytest

from swaybot.tools.fs import (
    FileSystem,
    WorkspaceError,
    edit_file,
    grep,
    list_directory,
    read_file,
    search_files,
    write_file,
)


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


def test_edit_file_replaces_single_occurrence(fs):
    fs.root.joinpath("code.py").write_text("hello world", encoding="utf-8")
    result = edit_file(path="code.py", old_string="hello", new_string="hi")
    assert "replaced 1 occurrence" in result
    assert fs.root.joinpath("code.py").read_text(encoding="utf-8") == "hi world"


def test_edit_file_rejects_ambiguous_replacement(fs):
    fs.root.joinpath("code.py").write_text("foo foo foo", encoding="utf-8")
    result = edit_file(path="code.py", old_string="foo", new_string="bar")
    assert "appears 3 times" in result
    assert fs.root.joinpath("code.py").read_text(encoding="utf-8") == "foo foo foo"


def test_edit_file_replace_all(fs):
    fs.root.joinpath("code.py").write_text("foo foo foo", encoding="utf-8")
    result = edit_file(
        path="code.py", old_string="foo", new_string="bar", replace_all=True
    )
    assert "replaced 3 occurrence" in result
    assert fs.root.joinpath("code.py").read_text(encoding="utf-8") == "bar bar bar"


def test_edit_file_stale_content(fs):
    fs.root.joinpath("code.py").write_text("current", encoding="utf-8")
    result = edit_file(path="code.py", old_string="old", new_string="new")
    assert "StaleContentError" in result
    assert fs.root.joinpath("code.py").read_text(encoding="utf-8") == "current"


def test_grep_finds_content(fs):
    fs.root.joinpath("a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    fs.root.joinpath("b.py").write_text("def bar():\n    pass\n", encoding="utf-8")
    results = grep(pattern="def foo", path=".")
    assert any("a.py:1:" in r for r in results)
    assert not any("b.py" in r for r in results)


def test_grep_respects_max_results(fs):
    for i in range(5):
        fs.root.joinpath(f"f{i}.py").write_text("target\n", encoding="utf-8")
    results = grep(pattern="target", path=".", max_results=3)
    assert len(results) == 3


def test_grep_fallback_when_rg_missing(fs, monkeypatch):
    monkeypatch.setattr("swaybot.tools.fs.shutil.which", lambda _name: None)
    fs.root.joinpath("c.py").write_text("# magic value\n", encoding="utf-8")
    results = grep(pattern="magic", path=".")
    assert any("c.py:1:" in r for r in results)
