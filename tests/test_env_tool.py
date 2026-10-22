"""Tests for the on-demand env_info tool."""

import json

from swaybot.tools import build_default_registry
from swaybot.tools.env import env_info


def test_env_info_reports_core_fields():
    payload = json.loads(env_info())
    for key in ("cwd", "platform", "os_version", "date_utc"):
        assert key in payload
    assert "is_git_repository" in payload  # True 或 False，但字段必须在


def test_env_info_in_default_registry_read_only():
    registry = build_default_registry()
    tool_obj = registry.get("env_info")
    assert tool_obj is not None
    assert tool_obj.read_only is True
    assert tool_obj.required_permission == "read"


def test_env_info_git_snapshot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    payload = json.loads(env_info())
    # 临时目录不在 git 仓库内（CI 环境下 tmp 通常独立）——两种结果都合法，
    # 但快照字段必须与 is_git_repository 一致。
    if payload["is_git_repository"]:
        assert payload["git_branch"]
        assert payload["git_status"] in ("clean", "dirty")
    else:
        assert "git_branch" not in payload
