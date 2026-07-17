import subprocess

import pytest

from swaybot.instructions import load_project_instructions


@pytest.fixture
def project_tree(tmp_path, monkeypatch):
    """Create a git repo with instruction files and cd into a subdir."""
    subprocess.run(
        ["git", "init", "-q"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )
    (tmp_path / "AGENTS.md").write_text("# Root rules\nAlways use type hints.", encoding="utf-8")
    sub = tmp_path / "src"
    sub.mkdir()
    swaybot_dir = sub / ".swaybot"
    swaybot_dir.mkdir()
    (swaybot_dir / "rules.md").write_text("# Module rules\nPrefer pathlib.", encoding="utf-8")
    monkeypatch.chdir(str(sub))
    return tmp_path, sub


def test_loads_root_agents_md(project_tree, monkeypatch):
    tmp_path, _sub = project_tree
    monkeypatch.chdir(str(tmp_path))
    text = load_project_instructions()
    assert "Always use type hints" in text
    assert "Prefer pathlib" not in text


def test_loads_nested_rules(project_tree):
    _, _sub = project_tree
    text = load_project_instructions()
    # Root instructions come first, then subdir rules.
    root_pos = text.find("Always use type hints")
    nested_pos = text.find("Prefer pathlib")
    assert root_pos != -1
    assert nested_pos != -1
    assert root_pos < nested_pos


def test_returns_empty_when_no_instructions(tmp_path, monkeypatch):
    monkeypatch.chdir(str(tmp_path))
    assert load_project_instructions() == ""


def test_ignores_nonexistent_start_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(str(tmp_path))
    nonexistent = tmp_path / "missing"
    assert load_project_instructions(str(nonexistent)) == ""
