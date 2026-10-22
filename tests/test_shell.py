"""Tests for the tiered sandboxed shell tool."""

import pytest

from swaybot.tools.shell import Shell, ShellError, create_shell_tool


@pytest.fixture
def tool_fn(tmp_path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    return create_shell_tool(str(tmp_path))


@pytest.fixture
def ws_tool(tmp_path):
    """workspace-write 档：允许解释器、禁元字符。"""
    tool = create_shell_tool(str(tmp_path))
    tool._shell.set_guard_mode("workspace-write")
    return tool


@pytest.fixture
def danger_tool(tmp_path):
    tool = create_shell_tool(str(tmp_path))
    tool._shell.set_guard_mode("danger")
    return tool


def test_read_only_allows_inspection(tool_fn):
    assert tool_fn(command="cat a.txt") == "hello"


def test_read_only_denies_destruction(tool_fn):
    with pytest.raises(ShellError, match="not allowed in read-only mode"):
        tool_fn(command="rm a.txt")


def test_read_only_denies_metacharacters(tool_fn):
    with pytest.raises(ShellError, match="metacharacters"):
        tool_fn(command="echo hi; rm -rf /")


def test_denials_are_final_and_actionable(tool_fn):
    with pytest.raises(ShellError, match="denial is final"):
        tool_fn(command="python3 x.py")


def test_workspace_escape_denied(tool_fn):
    with pytest.raises(ShellError):
        tool_fn(command="pwd", cwd="../outside")


def test_timeout_returns_error(ws_tool):
    (ws_tool._shell.root / "s.py").write_text(
        "import time; time.sleep(5)", encoding="utf-8"
    )
    out = ws_tool(command="python3 s.py", timeout=1)
    assert out.startswith("Error: command timed out")


def test_absolute_path_argument_denied(tool_fn):
    with pytest.raises(ShellError, match="absolute paths"):
        tool_fn(command="cat /etc/passwd")


def test_parent_path_argument_denied(tool_fn):
    with pytest.raises(ShellError, match="path traversal"):
        tool_fn(command="cat ../a.txt")


def test_find_exec_denied(tool_fn):
    with pytest.raises(ShellError, match="find action"):
        tool_fn(command="find . -exec sh -c 'echo pwned' +")


def test_workspace_write_runs_python(ws_tool):
    (ws_tool._shell.root / "s.py").write_text("print(6 * 7)", encoding="utf-8")
    assert "42" in ws_tool(command="python3 s.py")


def test_workspace_write_still_denies_pipes(ws_tool):
    with pytest.raises(ShellError, match="metacharacters"):
        ws_tool(command="python3 s.py | tail -1")


def test_danger_runs_full_shell(danger_tool):
    root = danger_tool._shell.root
    (root / "s.py").write_text("print(6 * 7)", encoding="utf-8")
    out = danger_tool(command="python3 s.py | tail -1 && echo done")
    assert "42" in out and "done" in out


def test_danger_reports_nonzero_exit(danger_tool):
    out = danger_tool(command="ls /definitely/not/here")
    assert "No such file" in out or out.startswith("Error:")


def test_shell_standalone_default_is_read_only(tmp_path):
    assert Shell(str(tmp_path)).command_guard.mode == "read-only"
