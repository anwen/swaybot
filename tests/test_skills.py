from swaybot.skills import SkillManager, load_skills


def test_skill_manager_loads_markdown(tmp_path):
    skill = tmp_path / "coding.md"
    skill.write_text(
        "---\nname: coding\ntags: code, python\n---\n\nWrite clean Python code.",
        encoding="utf-8",
    )
    manager = SkillManager(tmp_path)
    loaded = manager.load()
    assert len(loaded) == 1
    assert loaded[0].name == "coding"
    assert "clean Python" in loaded[0].content
    assert loaded[0].tags == ["code", "python"]


def test_skill_manager_query_by_tag(tmp_path):
    (tmp_path / "a.md").write_text("---\ntags: code\n---\nA", encoding="utf-8")
    (tmp_path / "b.md").write_text("---\ntags: docs\n---\nB", encoding="utf-8")
    manager = load_skills(tmp_path)
    assert [s.name for s in manager.query("code")] == ["a"]


def test_skill_manager_render_context(tmp_path):
    (tmp_path / "x.md").write_text("---\nname: x\ntags: test\n---\nDo X.", encoding="utf-8")
    manager = load_skills(tmp_path)
    ctx = manager.render_context("test")
    assert "Skill: x" in ctx
    assert "Do X." in ctx


def test_skill_manager_empty_directory(tmp_path):
    manager = load_skills(tmp_path)
    assert manager.list_skills() == []
    assert manager.render_context("anything") == ""


def test_skill_manager_missing_directory():
    manager = load_skills("/nonexistent/path/for/skills")
    assert manager.list_skills() == []
