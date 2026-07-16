"""Build the perception context fed to the brain each turn."""

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
            perception["memory_context"] = self._memory_context(task)
            perception["behavior_guidance"] = self._behavior_guidance(task)
            perception["messages"] = self._build_messages(task)
        perception["tool_descriptions"] = self.tools.schemas()
        perception["available_tools"] = self.tools.names()
        return perception

    def _memory_context(self, task: str) -> str:
        if self.memory is None:
            return ""
        relevant = self.memory.query_relevant(task, scope="long_term", limit=5)
        if not relevant:
            return ""
        return "\n".join(
            f"- {getattr(m, 'content', str(m))}" for m in relevant
        )

    def _behavior_guidance(self, task: str) -> str:
        if self.memory is None:
            return ""
        theories = self.memory.query_relevant(task, scope="long_term", limit=5)
        if not theories:
            return ""
        lines: list[str] = []
        for theory in theories:
            if getattr(theory, "credibility", 0.0) < 0.5:
                continue
            content = getattr(theory, "content", str(theory))
            if content:
                lines.append(f"- {content}")
        return "\n".join(lines)

    def _build_messages(self, task: str) -> list[dict]:
        if self.memory is None:
            return []
        messages: list[dict] = []
        context = self._memory_context(task)
        if context:
            messages.append(
                {"role": "user", "content": f"Relevant memories:\n{context}"}
            )
        for step in self.memory.memories:
            if task not in step.tags or step.scope != "short_term":
                continue
            messages.extend(step.to_messages())
        return messages
