"""Markdown skill system: load and retrieve skill instructions."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Skill:
    """A skill defined by a Markdown file."""

    name: str
    content: str
    tags: list[str]
    metadata: dict[str, Any]

    def matches(self, query: str) -> bool:
        """Return True if ``query`` matches name or tags."""
        low = query.lower()
        if low in self.name.lower():
            return True
        return any(low in tag.lower() for tag in self.tags)


class SkillManager:
    """Load and query Markdown skill files."""

    def __init__(self, directory: Path | str | None = None) -> None:
        if directory is None:
            directory = Path.cwd() / "skills"
        self.directory = Path(directory)
        self._skills: dict[str, Skill] = {}

    def load(self) -> list[Skill]:
        """Load all ``*.md`` files from ``directory``."""
        self._skills.clear()
        if not self.directory.exists():
            return []
        for path in sorted(self.directory.glob("*.md")):
            skill = self._parse(path)
            self._skills[skill.name] = skill
        return list(self._skills.values())

    def _parse(self, path: Path) -> Skill:
        content = path.read_text(encoding="utf-8")
        metadata: dict[str, Any] = {}
        tags: list[str] = []
        body = content

        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                header = parts[1].strip()
                body = parts[2].strip()
                for line in header.splitlines():
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip().lower()
                        value = value.strip()
                        if key == "tags":
                            tags = [t.strip() for t in value.split(",") if t.strip()]
                        else:
                            metadata[key] = value

        name = metadata.get("name", path.stem)
        tags = tags or [path.stem]
        return Skill(name=name, content=body, tags=tags, metadata=metadata)

    def list_skills(self) -> list[str]:
        return sorted(self._skills.keys())

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def query(self, query: str) -> list[Skill]:
        """Return skills whose name or tags match ``query``."""
        return [s for s in self._skills.values() if s.matches(query)]

    def render_context(self, query: str) -> str:
        """Render matching skills as a single context string."""
        skills = self.query(query)
        if not skills:
            return ""
        parts = []
        for skill in skills:
            parts.append(f"### Skill: {skill.name}\n{skill.content}")
        return "\n\n".join(parts)


def load_skills(directory: Path | str | None = None) -> SkillManager:
    manager = SkillManager(directory)
    manager.load()
    return manager
