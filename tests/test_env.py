"""Tests for dotenv loading."""

import os
from pathlib import Path

from swaybot.env import load_dotenv


def test_load_dotenv_reads_key_value_pairs(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "SWAYBOT_TEST_KEY=secret\nSWAYBOT_TEST_QUOTED=\"quoted value\"\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("SWAYBOT_TEST_KEY", raising=False)
    monkeypatch.delenv("SWAYBOT_TEST_QUOTED", raising=False)

    load_dotenv(env_file)

    assert os.environ["SWAYBOT_TEST_KEY"] == "secret"
    assert os.environ["SWAYBOT_TEST_QUOTED"] == "quoted value"


def test_load_dotenv_does_not_override_existing_vars(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("SWAYBOT_TEST_PRESERVE=from_file\n", encoding="utf-8")
    monkeypatch.setenv("SWAYBOT_TEST_PRESERVE", "from_env")

    load_dotenv(env_file)

    assert os.environ["SWAYBOT_TEST_PRESERVE"] == "from_env"


def test_load_dotenv_ignores_comments_and_blank_lines(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# this is a comment\n\nSWAYBOT_TEST_REAL=value\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("SWAYBOT_TEST_REAL", raising=False)

    load_dotenv(env_file)

    assert os.environ["SWAYBOT_TEST_REAL"] == "value"


def test_load_dotenv_missing_file_is_silent(tmp_path):
    missing = tmp_path / "nonexistent.env"
    load_dotenv(missing)  # should not raise
    assert not missing.exists()
