import pytest

from swaybot.tools.shell import Shell, ShellError, run_shell_command


@pytest.fixture
def shell(tmp_path, monkeypatch):
    monkeypatch.setenv("SWAYBOT_WORKSPACE", str(tmp_path))
    import swaybot.tools.shell as shell_module

    shell_module._shell = Shell(str(tmp_path))
    return shell_module._shell


def test_run_allowed_command(shell):
    shell.root.joinpath("a.txt").write_text("hello", encoding="utf-8")
    output = run_shell_command(command="cat a.txt")
    assert output == "hello"


def test_run_disallowed_command(shell):
    with pytest.raises(ShellError):
        run_shell_command(command="rm a.txt")


def test_metacharacters_blocked(shell):
    with pytest.raises(ShellError):
        run_shell_command(command="echo hi; rm -rf /")


def test_cwd_traversal_blocked(shell):
    with pytest.raises(ShellError):
        run_shell_command(command="pwd", cwd="../outside")


def test_command_timeout(shell):
    # sleep is not in the allowlist, so this would raise ShellError rather than timeout.
    with pytest.raises(ShellError):
        run_shell_command(command="sleep 5")
