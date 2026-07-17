from pathlib import Path

import pytest

from swaybot.code_quality import run_quality_hooks


def test_quality_hooks_run_configured_commands(tmp_path):
    target = tmp_path / "demo.txt"
    target.write_text("hello", encoding="utf-8")
    hooks = [
        ("hello", ["python", "-c", "print('hi from hook')"]),
    ]
    report = run_quality_hooks(target, hooks=hooks)
    assert "hi from hook" in report


def test_quality_hooks_skip_missing_commands(tmp_path, monkeypatch):
    target = tmp_path / "demo.py"
    target.write_text("x = 1", encoding="utf-8")
    monkeypatch.setattr("swaybot.code_quality.shutil.which", lambda _name: None)
    report = run_quality_hooks(target)
    assert report == ""


def test_quality_hooks_non_python_file_skipped_by_default(tmp_path):
    target = tmp_path / "data.json"
    target.write_text("{}", encoding="utf-8")
    report = run_quality_hooks(target)
    assert report == ""


def test_quality_hooks_reports_failure(tmp_path):
    target = tmp_path / "demo.py"
    target.write_text("x = 1", encoding="utf-8")
    hooks = [
        ("fails", ["python", "-c", "import sys; sys.exit(1)"]),
    ]
    report = run_quality_hooks(target, hooks=hooks)
    assert "failed" in report
