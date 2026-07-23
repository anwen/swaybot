from typing import Any

from .brain import Brain, EchoBrain
from .context import ContextBuilder
from .task_state import Environment
from .hook import AgentHook, CompositeHook
from .request_context import RequestContext, current_context, set_context
from .security import PermissionPolicy, permission_level_to_mode
from .memory import (
    ActionStep,
    AutoCompact,
    MemoryStore,
    ObservationStep,
    PlanningStep,
    TaskStep,
)
from .metrics import LoggingMetricSink, MetricsHook
from .reflection import Reflector, reflection_to_memory
from .run_log import append_run, build_run_record, run_log_path_for_memory
from .tools import ToolRegistry, build_default_registry


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
        permission_policy: PermissionPolicy | None = None,
        auto_compact: bool = False,
        compactor: AutoCompact | None = None,
    ):
        self.brain = brain or EchoBrain()
        self.tools = tools or build_default_registry()
        self.memory = memory
        self.reflector = reflector
        if hooks is None:
            hooks = [MetricsHook(LoggingMetricSink())]
        self.hooks = CompositeHook(hooks)
        if permission_policy is not None:
            self.permission_policy = permission_policy
        else:
            self.permission_policy = PermissionPolicy(
                mode=permission_level_to_mode(permission_level)
            )
        self.auto_compact = auto_compact
        self.compactor = compactor
        self.context_builder = ContextBuilder(self.memory, self.tools)

    def _allowed(self, action_name: str) -> tuple[bool, str]:
        """Check whether the active permission policy allows ``action_name``."""
        tool = self.tools.get(action_name)
        if tool is None:
            return True, ""
        allowed, reason = self.permission_policy.allows(
            tool.name, tool.required_permission
        )
        if not allowed:
            return False, f"Permission denied: {reason}"
        return True, ""

    def run(
        self,
        task: str,
        max_steps: int = 10,
        reflect: bool = True,
        plan: bool = False,
        hypothesis: str | None = None,
        request_context: RequestContext | None = None,
        stream: bool = False,
    ) -> Environment:
        ctx = request_context or current_context()
        with set_context(ctx):
            return self._run(
                task, max_steps, reflect, plan, hypothesis, stream, ctx
            )

    def _run(
        self,
        task: str,
        max_steps: int = 10,
        reflect: bool = True,
        plan: bool = False,
        hypothesis: str | None = None,
        stream: bool = False,
        request_context: RequestContext | None = None,
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
            action, call_info = self._think(perception, stream)
            result, error = self._execute_action(action, request_context)
            if error is None and call_info.get("error"):
                error = str(call_info["error"])
            env.observe(action, result)
            self.hooks.after_iteration(
                task, env.step, action, result, metadata=call_info
            )
            run_steps.append(
                self._build_step_record(env.step, action, result, error, call_info)
            )
            if self.memory is not None:
                self._persist_step(task, env.step, action, result, call_info)

        reflections = self._reflect(task, env.history, hypothesis, reflect)
        self._append_run_log(task, max_steps, run_steps, hypothesis, reflections)

        self.hooks.after_run(task, env, reflections)
        return env

    def _think(self, perception: dict, stream: bool) -> tuple[dict, dict]:
        """Ask the brain for the next action, capturing call metadata."""
        call_info: dict = {}
        if stream:
            action = self.brain.think(
                perception,
                self.tools.names(),
                metadata=call_info,
                stream_callback=lambda token: self.hooks.on_token(token),
                reasoning_callback=lambda reason: self.hooks.on_reasoning(reason),
            )
        else:
            action = self.brain.think(
                perception,
                self.tools.names(),
                metadata=call_info,
            )
        return action, call_info

    def _execute_action(
        self, action: dict, request_context: RequestContext | None = None
    ) -> tuple[Any, str | None]:
        """Run the chosen tool if permitted, returning (result, error)."""
        allowed, permission_msg = self._allowed(action.get("name", ""))
        if not allowed:
            return f"Error: {permission_msg}", permission_msg
        try:
            return self.tools.execute(action, request_context), None
        except Exception as exc:
            return f"Error: {exc}", str(exc)

    def _build_step_record(
        self,
        step: int,
        action: dict,
        result: Any,
        error: str | None,
        call_info: dict,
    ) -> dict:
        """Create a serializable record of one step."""
        record: dict = {
            "step": step,
            "action": action,
            "result": str(result),
        }
        if error:
            record["error"] = error
        for key in (
            "model_input_messages",
            "raw_output",
            "token_usage",
            "duration_ms",
        ):
            if call_info.get(key) is not None:
                record[key] = call_info[key]
        return record

    def _persist_step(
        self,
        task: str,
        step: int,
        action: dict,
        result: Any,
        call_info: dict,
    ) -> None:
        """Store an action/observation pair in memory and optionally compact."""
        assert self.memory is not None
        steps = [
            ActionStep(
                step=step,
                action=action,
                tags=[task],
                model_input_messages=call_info.get("model_input_messages"),
                raw_output=call_info.get("raw_output"),
                token_usage=call_info.get("token_usage"),
                duration_ms=call_info.get("duration_ms"),
            ),
            ObservationStep(
                step=step,
                observation=str(result),
                tags=[task],
            ),
        ]
        self.memory.add_steps(steps)
        if self.auto_compact and self.compactor is not None:
            self.compactor.compact(self.memory, tag=task)

    def _reflect(
        self,
        task: str,
        history: list[dict],
        hypothesis: str | None,
        reflect: bool,
    ) -> list[str]:
        """Generate reflection memories and prune short-term steps."""
        reflections: list[str] = []
        if not reflect or self.memory is None or self.reflector is None:
            return reflections
        for reflection in self.reflector.reflect_on_run(
            task, history, hypothesis=hypothesis
        ):
            memory_step = reflection_to_memory(reflection)
            self.memory.add(memory_step)
            content = getattr(memory_step, "content", "")
            if content:
                reflections.append(content)
        self.memory.prune(scope="short_term", tag=task)
        return reflections

    def _append_run_log(
        self,
        task: str,
        max_steps: int,
        run_steps: list[dict],
        hypothesis: str | None,
        reflections: list[str],
    ) -> None:
        """Persist the run record if memory is backed by a file."""
        if self.memory is None or self.memory.path is None:
            return
        run_log_path = run_log_path_for_memory(self.memory.path)
        if run_log_path is None:
            return
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

    def _create_plan(self, task: str, max_steps: int) -> None:
        """Ask the brain for a plan and store it as a PlanningStep."""
        assert self.memory is not None
        perception = self.context_builder.build(task, 0, max_steps, [])
        perception["planning"] = True
        action = self.brain.think(perception, self.tools.names())
        if action.get("name") == "plan":
            steps = action.get("args", {}).get("steps", [])
            if steps:
                self.memory.add(PlanningStep(plan=steps, tags=[task]))
