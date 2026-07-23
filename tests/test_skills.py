"""Tests for the Markdown skill library."""

from swaybot.skills import SkillLibrary


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_load_nanobot_style_skill(tmp_path):
    _write(
        tmp_path / "github" / "SKILL.md",
        """---
name: github
description: Interact with GitHub using the gh CLI.
tags: git, github
---

# GitHub

Use `gh` to interact with GitHub.
""",
    )

    lib = SkillLibrary(tmp_path).load()
    assert lib.list_skills() == ["github"]
    skill = lib.get("github")
    assert skill is not None
    assert skill.description.startswith("Interact with GitHub")
    assert "gh" in skill.content
    assert skill.tags == ["git", "github"]


def test_load_flat_style_skill(tmp_path):
    _write(
        tmp_path / "weather.md",
        """---
name: weather
description: Get weather info.
---

# Weather
Use wttr.in.
""",
    )

    lib = SkillLibrary(tmp_path).load()
    assert "weather" in lib.list_skills()


def test_skill_falls_back_to_directory_name(tmp_path):
    _write(
        tmp_path / "cron" / "SKILL.md",
        """---
description: Schedule reminders.
---

# Cron
""",
    )
    lib = SkillLibrary(tmp_path).load()
    skill = lib.get("cron")
    assert skill is not None
    assert skill.name == "cron"
    assert skill.tags == ["cron"]


def test_query_matches_by_name_description_or_tags(tmp_path):
    _write(
        tmp_path / "github" / "SKILL.md",
        """---
name: github
description: GitHub CLI
tags: git
---
body
""",
    )
    _write(
        tmp_path / "weather" / "SKILL.md",
        """---
name: weather
description: Get weather info
tags: forecast
---
body
""",
    )

    lib = SkillLibrary(tmp_path).load()
    assert {s.name for s in lib.query("git")} == {"github"}
    assert {s.name for s in lib.query("weather")} == {"weather"}
    assert {s.name for s in lib.query("forecast")} == {"weather"}


def test_render_context_returns_empty_when_no_match(tmp_path):
    _write(
        tmp_path / "github" / "SKILL.md",
        """---
name: github
description: GitHub
---
body
""",
    )
    lib = SkillLibrary(tmp_path).load()
    assert lib.render_context("nonexistent") == ""
    context = lib.render_context("github")
    assert "### Skill: github" in context
    assert "body" in context


def test_load_missing_directory_is_empty(tmp_path):
    lib = SkillLibrary(tmp_path / "does-not-exist").load()
    assert lib.list_skills() == []


def test_json_metadata_field_parsed(tmp_path):
    _write(
        tmp_path / "clawhub" / "SKILL.md",
        """---
name: clawhub
description: Search skills.
metadata: {"nanobot": {"emoji": "🦞"}}
---
body
""",
    )
    lib = SkillLibrary(tmp_path).load()
    skill = lib.get("clawhub")
    assert skill is not None
    assert skill.metadata["metadata"] == {"nanobot": {"emoji": "🦞"}}
