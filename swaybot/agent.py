from contextlib import nullcontext
from typing import Any, Callable

from .agent_events import (
    Event,
    ReasoningDelta,
    RunBegin,
    RunEnd,
    StepBegin,
    StepEnd,
    TokenDelta,
)
from .brain import Brain, EchoBrain
from .instructions import load_project_instructions
from .perception import ContextBuilder
from .task_state import Environment
from .request_context import RequestContext, current_context, set_context
from .permission import PermissionPolicy, permission_level_to_mode
from .memory import (
    ActionStep,
    AutoCompact,
    MemoryStore,
    ObservationStep,
    PlanningStep,
    TaskStep,
)
from .reflection import Reflector, reflection_to_memory
from .run_log import RunLogRecorder
from .skills import SkillLibrary
from .tools import ToolRegistry, build_default_registry


class Agent:
    """Minimal agent: perceive, think, act, observe, loop, reflect."""

    def __init__(
        self,
        brain: Brain | None = None,
        tools: ToolRegistry | None = None,
        memory: MemoryStore | None = None,
        reflector: Reflector | None = None,
        permission_level: str = "medium",
        permission_policy: PermissionPolicy | None = None,
        auto_compact: bool = True,
        compactor: AutoCompact | None = None,
        skills: SkillLibrary | None = None,
        on_event: Callable[[Event], None] | None = None,
    ):
        self.brain = brain or EchoBrain()
        self.tools = tools or build_default_registry()
        self.memory = memory
        self.reflector = reflector
        self.on_event = on_event
        if permission_policy is not None:
            self.permission_policy = permission_policy
        else:
            self.permission_policy = PermissionPolicy(
                mode=permission_level_to_mode(permission_level)
            )
        self.auto_compact = auto_compact
        # Default-on since 0.3.x: experience accumulation is the point of
        # the memory system, and long runs need bounded context. The
        # summarizer asks the brain via final_answer; brains that cannot
        # summarize (EchoBrain) yield an empty answer and the compactor's
        # deterministic digest takes over — safe by construction.
        if compactor is None and auto_compact:
            compactor = AutoCompact(brain=self._summarize_steps)
        self.compactor = compactor
        # The repeat-call guard now lives on the LLMBrain (so it also fires on
        # the tau3bench path, which calls brain.think() directly and bypasses
        # this loop). Agent._run no longer feeds the guard itself; it only
        # resets the brain's guard chain when a new task starts, mirroring the
        # old behavior where a fresh user turn cleared the repeat counter.
        self._last_task: str | None = None
        self._sinks: list[Callable[[Event], None]] = []
        if skills is not None and not skills._skills:
            skills.load()
        self.context_builder = ContextBuilder(self.memory, self.tools, skills=skills)
        # The shell tool's CommandGuard enforces the same tiered policy the
        # PermissionPolicy applies at the tool level: read-only -> inspection
        # commands, workspace-write -> interpreters too, danger -> full shell.
        shell_tool = self.tools.get("run_shell_command")
        shell_impl = getattr(shell_tool, "_shell", None)
        if shell_impl is not None:
            shell_impl.set_guard_mode(self.permission_policy.mode.value)

    def _summarize_steps(self, step_messages: list[dict]) -> str:
        """Brain-backed summarizer for AutoCompact (empty string = fallback)."""
        perception = {
            "task": "Summarize the previous agent steps for future reference.",
            "step": 0,
            # 2 而非 1：见 LifeRunner._ask_brain 的注释——1 会把散文兜底变 done
            "max_steps": 2,
            "messages": step_messages,
            "tool_descriptions": [],
        }
        action = self.brain.think(perception, ["final_answer"])
        if action.get("name") == "final_answer":
            return str(action.get("args", {}).get("answer", "")).strip()
        return ""

    def _emit(self, event: Event) -> None:
        """Deliver an event to every active sink (run-log recorder, callers)."""
        for sink in self._sinks:
            sink(event)

    def _on_token(self, token: str) -> None:
        self._emit(TokenDelta(text=token))

    def _on_reasoning(self, text: str) -> None:
        self._emit(ReasoningDelta(text=text))

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
        # Instruction files (AGENTS.md etc.) are read once per run, not once
        # per step — collecting them walks directories and spawns git.
        instructions = load_project_instructions()
        # The run-log recorder and the caller's sink consume the same event
        # stream; recording is disabled when memory has no backing file.
        self._sinks = []
        if self.memory is not None:
            self._sinks.append(RunLogRecorder.for_memory(self.memory.path))
        if self.on_event is not None:
            self._sinks.append(self.on_event)
        self._emit(RunBegin(task=task, max_steps=max_steps, hypothesis=hypothesis))
        # Reset the brain's guard chains when the task changes: a repeat or
        # non-convergence streak across a fresh user/task turn is not a loop.
        if task != self._last_task:
            for attr in ("repeat_guard", "nonconvergence_guard"):
                guard = getattr(self.brain, attr, None)
                if guard is not None:
                    guard.reset()
            self._last_task = task
        batch = self.memory.batch() if self.memory is not None else nullcontext()
        with batch:
            self._run_body(
                env, task, max_steps, reflect, plan, hypothesis, stream,
                request_context, instructions,
            )
        return env

    def _run_body(
        self,
        env: Environment,
        task: str,
        max_steps: int,
        reflect: bool,
        plan: bool,
        hypothesis: str | None,
        stream: bool,
        request_context: RequestContext | None,
        instructions: str,
    ) -> None:
        if self.memory is not None:
            self.memory.add(
                TaskStep(task=task, max_steps=max_steps, tags=[task])
            )
        if plan and self.memory is not None:
            self._create_plan(task, max_steps, instructions)
        while not env.done:
            perception = self.context_builder.build(
                task, env.step, max_steps, env.history, instructions=instructions
            )
            action, call_info = self._think(perception, stream)
            self._emit(StepBegin(step=env.step + 1, action=dict(action)))
            result, error = self._execute_action(action, request_context)
            if error is None and call_info.get("error"):
                error = str(call_info["error"])
            env.observe(action, result)
            self._emit(
                StepEnd(
                    step=env.step,
                    action=action,
                    result=result,
                    error=error,
                    metadata=call_info,
                )
            )
            if self.memory is not None:
                self._persist_step(task, env.step, action, result, call_info)

        reflections = self._reflect(task, env.history, hypothesis, reflect)
        self._emit(
            RunEnd(
                steps=env.step,
                done=env.done,
                status="finished" if env.stop_reason == "tool" else "steps_exhausted",
                reflections=reflections,
            )
        )
        # Detach run-scoped sinks so late emissions (e.g. a model stream
        # callback outliving the run) no-op instead of hitting a stale
        # recorder. NOTE: one Agent runs one task at a time — concurrent
        # run() calls on the same instance would still race; use an agent
        # per session (as AsyncAgent does).
        self._sinks = []

    def _think(self, perception: dict, stream: bool) -> tuple[dict, dict]:
        """Ask the brain for the next action, capturing call metadata."""
        call_info: dict = {}
        if stream:
            action = self.brain.think(
                perception,
                self.tools.names(),
                metadata=call_info,
                stream_callback=self._on_token,
                reasoning_callback=self._on_reasoning,
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

    def _create_plan(self, task: str, max_steps: int, instructions: str = "") -> None:
        """Ask the brain for a plan and store it as a PlanningStep."""
        assert self.memory is not None
        perception = self.context_builder.build(
            task, 0, max_steps, [], instructions=instructions
        )
        perception["planning"] = True
        action = self.brain.think(perception, self.tools.names())
        if action.get("name") == "plan":
            steps = action.get("args", {}).get("steps", [])
            if steps:
                self.memory.add(PlanningStep(plan=steps, tags=[task]))
