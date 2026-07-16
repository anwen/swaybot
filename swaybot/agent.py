from .brain import Brain, EchoBrain
from .context import ContextBuilder
from .environment import Environment
from .hook import AgentHook, CompositeHook
from .memory import (
    ActionStep,
    AutoCompact,
    MemoryStore,
    ObservationStep,
    PlanningStep,
    TaskStep,
)
from .reflection import Reflector, reflection_to_memory
from .run_log import append_run, build_run_record, run_log_path_for_memory
from .tools import ToolRegistry, build_default_registry


_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


class Agent:
    """Minimal agent: perceive, think, act, observe, loop, reflect."""

    def __init__(
        self,
        brain: Brain | None = None,
        tools: ToolRegistry | None = None,
        memory: MemoryStore | None = None,
        reflector: Reflector | None = None,
        hooks: list[AgentHook] | None = None,
        permission_level: str = "medium",
        auto_compact: bool = False,
        compactor: AutoCompact | None = None,
    ):
        self.brain = brain or EchoBrain()
        self.tools = tools or build_default_registry()
        self.memory = memory
        self.reflector = reflector
        self.hooks = CompositeHook(hooks)
        self.permission_level = permission_level
        self.auto_compact = auto_compact
        self.compactor = compactor
        self.context_builder = ContextBuilder(self.memory, self.tools)

    def _allowed(self, action_name: str) -> tuple[bool, str]:
        """Check whether the current permission level allows ``action_name``."""
        tool = self.tools.get(action_name)
        if tool is None:
            return True, ""
        if _RISK_ORDER[tool.risk_level] > _RISK_ORDER[self.permission_level]:
            return False, (
                f"Permission denied: '{action_name}' requires "
                f"{tool.risk_level} permission (current: {self.permission_level})"
            )
        return True, ""

    def run(
        self,
        task: str,
        max_steps: int = 10,
        reflect: bool = True,
        plan: bool = False,
        hypothesis: str | None = None,
    ) -> Environment:
        env = Environment(task=task, max_steps=max_steps)
        run_steps: list[dict] = []
        if self.memory is not None:
            self.memory.add(
                TaskStep(task=task, max_steps=max_steps, tags=[task])
            )
        if plan and self.memory is not None:
            self._create_plan(task, max_steps)
        while not env.done:
            perception = self.context_builder.build(
                task, env.step, max_steps, env.history
            )
            self.hooks.before_iteration(task, env.step, perception)
            call_info: dict = {}
            try:
                action = self.brain.think(
                    perception, self.tools.names(), metadata=call_info
                )
            except TypeError:
                action = self.brain.think(perception, self.tools.names())
                call_info = {}
            error = None
            allowed, permission_msg = self._allowed(action.get("name", ""))
            if allowed:
                try:
                    result = self.tools.execute(action)
                except Exception as exc:
                    error = str(exc)
                    result = f"Error: {exc}"
            else:
                error = permission_msg
                result = f"Error: {permission_msg}"
            if error is None and call_info.get("error"):
                error = str(call_info["error"])
            env.observe(action, result)
            self.hooks.after_iteration(
                task, env.step, action, result, metadata=call_info
            )
            step_record = {
                "step": env.step,
                "action": action,
                "result": str(result),
            }
            if error:
                step_record["error"] = error
            for key in (
                "model_input_messages",
                "raw_output",
                "token_usage",
                "duration_ms",
            ):
                if call_info.get(key) is not None:
                    step_record[key] = call_info[key]
            run_steps.append(step_record)
            if self.memory is not None:
                self.memory.add(
                    ActionStep(
                        step=env.step,
                        action=action,
                        tags=[task],
                        model_input_messages=call_info.get("model_input_messages"),
                        raw_output=call_info.get("raw_output"),
                        token_usage=call_info.get("token_usage"),
                        duration_ms=call_info.get("duration_ms"),
                    )
                )
                self.memory.add(
                    ObservationStep(
                        step=env.step,
                        observation=str(result),
                        tags=[task],
                    )
                )
                if self.auto_compact and self.compactor is not None:
                    self.compactor.compact(self.memory, tag=task)

        reflections: list[str] = []
        if reflect and self.memory is not None and self.reflector is not None:
            for reflection in self.reflector.reflect_on_run(
                task, env.history, hypothesis=hypothesis
            ):
                memory_step = reflection_to_memory(reflection)
                self.memory.add(memory_step)
                content = getattr(memory_step, "content", "")
                if content:
                    reflections.append(content)
            self.memory.prune(scope="short_term", tag=task)

        if self.memory is not None and self.memory.path is not None:
            run_log_path = run_log_path_for_memory(self.memory.path)
            if run_log_path is not None:
                append_run(
                    run_log_path,
                    build_run_record(
                        task,
                        max_steps,
                        run_steps,
                        hypothesis=hypothesis,
                        reflections=reflections,
                    ),
                )

        self.hooks.after_run(task, env, reflections)
        return env

    def _create_plan(self, task: str, max_steps: int) -> None:
        """Ask the brain for a plan and store it as a PlanningStep."""
        assert self.memory is not None
        perception = {
            "task": task,
            "max_steps": max_steps,
            "step": 0,
            "planning": True,
            "tool_descriptions": self.tools.schemas(),
        }
        action = self.brain.think(perception, self.tools.names())
        if action.get("name") == "plan":
            steps = action.get("args", {}).get("steps", [])
            if steps:
                self.memory.add(PlanningStep(plan=steps, tags=[task]))
