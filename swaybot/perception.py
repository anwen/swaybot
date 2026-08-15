"""Build the perception context fed to the brain each turn."""

from .instructions import load_project_instructions
from .memory import MemoryStore, format_memory_bullets
from .skills import SkillLibrary
from .tools import ToolRegistry


class ContextBuilder:
    """Assemble memory, guidance, history and tool context for the brain."""

    def __init__(
        self,
        memory: MemoryStore | None,
        tools: ToolRegistry,
        skills: SkillLibrary | None = None,
    ) -> None:
        self.memory = memory
        self.tools = tools
        self.skills = skills

    def build(
        self,
        task: str,
        step: int,
        max_steps: int,
        history: list[dict],
        notice: str | None = None,
    ) -> dict:
        """Return a perception dict ready for ``Brain.think``."""
        perception: dict = {
            "task": task,
            "step": step,
            "max_steps": max_steps,
            "history": history,
        }
        if self.memory is not None:
            relevant = self.memory.query_relevant(task, scope="long_term", limit=5)
            memory_context = self._memory_context(task, relevant)
            perception["memory_context"] = memory_context
            guidance = self._behavior_guidance(task, relevant)
            if notice:
                guidance = f"{guidance}\n{notice}".strip() if guidance else notice
            perception["behavior_guidance"] = guidance
            perception["messages"] = self._build_messages(
                task, relevant, memory_context
            )
        elif notice:
            perception["behavior_guidance"] = notice
        perception["tool_descriptions"] = self.tools.schemas()
        perception["available_tools"] = self.tools.names()
        perception["project_instructions"] = load_project_instructions()
        if self.skills is not None:
            perception["skill_context"] = self.skills.render_context(task)
        return perception

    def _memory_context(self, task: str, relevant: list) -> str:
        if not relevant:
            return ""
        return format_memory_bullets(relevant)

    def _behavior_guidance(self, task: str, relevant: list) -> str:
        lines: list[str] = []
        for theory in relevant:
            if getattr(theory, "credibility", 0.0) < 0.5:
                continue
            content = getattr(theory, "content", str(theory))
            if content:
                lines.append(f"- {content}")
        return "\n".join(lines)

    def _build_messages(self, task: str, relevant: list, memory_context: str = "") -> list[dict]:
        if self.memory is None:
            return []
        messages: list[dict] = []
        if memory_context:
            messages.append(
                {"role": "user", "content": f"Relevant memories:\n{memory_context}"}
            )
        for step in self.memory.iter_short_term(task):
            messages.extend(step.to_messages())
        return messages
