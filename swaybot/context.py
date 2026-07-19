"""Build the perception context fed to the brain each turn."""

from .instructions import load_project_instructions
from .memory import MemoryStore
from .tools import ToolRegistry


class ContextBuilder:
    """Assemble memory, guidance, history and tool context for the brain."""

    def __init__(
        self,
        memory: MemoryStore | None,
        tools: ToolRegistry,
    ) -> None:
        self.memory = memory
        self.tools = tools

    def build(
        self,
        task: str,
        step: int,
        max_steps: int,
        history: list[dict],
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
            perception["memory_context"] = self._memory_context(task, relevant)
            perception["behavior_guidance"] = self._behavior_guidance(task, relevant)
            perception["messages"] = self._build_messages(task, relevant)
        perception["tool_descriptions"] = self.tools.schemas()
        perception["available_tools"] = self.tools.names()
        perception["project_instructions"] = load_project_instructions()
        return perception

    def _memory_context(self, task: str, relevant: list) -> str:
        if not relevant:
            return ""
        return "\n".join(
            f"- {getattr(m, 'content', str(m))}" for m in relevant
        )

    def _behavior_guidance(self, task: str, relevant: list) -> str:
        lines: list[str] = []
        for theory in relevant:
            if getattr(theory, "credibility", 0.0) < 0.5:
                continue
            content = getattr(theory, "content", str(theory))
            if content:
                lines.append(f"- {content}")
        return "\n".join(lines)

    def _build_messages(self, task: str, relevant: list) -> list[dict]:
        if self.memory is None:
            return []
        messages: list[dict] = []
        context = self._memory_context(task, relevant)
        if context:
            messages.append(
                {"role": "user", "content": f"Relevant memories:\n{context}"}
            )
        for step in self.memory.iter_short_term(task):
            messages.extend(step.to_messages())
        return messages
