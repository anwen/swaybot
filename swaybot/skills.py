"""Load and query Markdown skill files."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class Skill:
    """A skill defined by a Markdown file."""

    name: str
    description: str
    content: str
    tags: list[str]
    metadata: dict[str, Any]

    def matches(self, query: str) -> bool:
        """Return True if ``query`` matches name, description or tags."""
        low = query.lower()
        if low in self.name.lower():
            return True
        if low in self.description.lower():
            return True
        return any(low in tag.lower() for tag in self.tags)


class SkillLibrary:
    """Load and query Markdown skill files from a directory."""

    def __init__(self, directory: Path | str | None = None) -> None:
        if directory is None:
            directory = Path.cwd() / "skills"
        self.directory = Path(directory)
        self._skills: dict[str, Skill] = {}

    def load(self) -> "SkillLibrary":
        """Load all ``SKILL.md`` and top-level ``*.md`` files."""
        self._skills.clear()
        if not self.directory.exists():
            return self

        # nanobot-style: skills/<skill-name>/SKILL.md
        for path in sorted(self.directory.rglob("SKILL.md")):
            skill = self._parse(path)
            self._skills[skill.name] = skill

        # Flat style: skills/*.md
        for path in sorted(self.directory.glob("*.md")):
            if path.name == "README.md":
                continue
            skill = self._parse(path)
            self._skills[skill.name] = skill

        return self

    def _parse(self, path: Path) -> Skill:
        content = path.read_text(encoding="utf-8")
        metadata: dict[str, Any] = {}
        tags: list[str] = []
        body = content

        match = FRONTMATTER_RE.match(content)
        if match:
            body = content[match.end() :].strip()
            metadata, tags = self._parse_frontmatter(match.group(1))

        name = metadata.get("name", path.parent.name if path.name == "SKILL.md" else path.stem)
        description = metadata.get("description", "")
        tags = tags or [name]
        return Skill(
            name=name,
            description=description,
            content=body,
            tags=tags,
            metadata=metadata,
        )

    @staticmethod
    def _parse_frontmatter(text: str) -> tuple[dict[str, Any], list[str]]:
        """Parse a simple key-value frontmatter, with JSON metadata support."""
        metadata: dict[str, Any] = {}
        tags: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()

            # Unwrap surrounding quotes.
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]

            if key == "tags":
                tags = [t.strip() for t in value.split(",") if t.strip()]
            elif key == "metadata":
                try:
                    metadata[key] = json.loads(value)
                except json.JSONDecodeError:
                    metadata[key] = value
            else:
                metadata[key] = value
        return metadata, tags

    def list_skills(self) -> list[str]:
        return sorted(self._skills.keys())

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def query(self, query: str) -> list[Skill]:
        """Return skills whose name, description or tags match ``query``."""
        return [s for s in self._skills.values() if s.matches(query)]

    def render_context(self, query: str) -> str:
        """Render matching skills as a single context string."""
        skills = self.query(query)
        if not skills:
            return ""
        parts = []
        for skill in skills:
            parts.append(f"### Skill: {skill.name}\n{skill.description}\n\n{skill.content}")
        return "\n\n".join(parts)
