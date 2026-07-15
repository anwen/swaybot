from pathlib import Path

import pytest

from swaybot.prompts import load_prompt, render_prompt


def test_load_prompt_reads_template(tmp_path: Path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "test.j2").write_text("Hello {{ name }}", encoding="utf-8")
    assert load_prompt("test", prompts_dir=prompts_dir) == "Hello {{ name }}"


def test_render_prompt_renders_variables(tmp_path: Path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "test.j2").write_text("Hello {{ subject }}", encoding="utf-8")
    assert render_prompt("test", prompts_dir=prompts_dir, subject="Sway") == "Hello Sway"


def test_render_prompt_conditional_and_loops(tmp_path: Path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "test.j2").write_text(
        "{% if items %}Items:{% for item in items %} {{ item }}{% endfor %}{% else %}None{% endif %}",
        encoding="utf-8",
    )
    assert (
        render_prompt("test", prompts_dir=prompts_dir, items=["a", "b"])
        == "Items: a b"
    )
    assert render_prompt("test", prompts_dir=prompts_dir, items=[]) == "None"


def test_load_default_system_prompt_exists():
    text = load_prompt("system")
    assert "SwayBot" in text
    assert "JSON" in text


def test_load_default_user_prompt_exists():
    text = load_prompt("user")
    assert "Task:" in text
    assert "Step:" in text


def test_render_missing_prompt_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_prompt("missing", prompts_dir=tmp_path)
