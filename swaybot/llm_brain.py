import os
from typing import Any

from .brain import ReasoningCallback, StreamCallback
from .loop_guards import NonConvergenceGuard, RepeatCallGuard
from .models import (
    FallbackModel,
    Model,
    ModelResponse,
    OpenAIModel,
    RecordingModel,
    normalize_response,
)
from .parsing import (
    PROSE_CAP_CHARS,
    PROSE_TOOL_TAG_RE,
    is_explicit_tool_call,
    fallback_action,
    parse_action,
    parse_exploration_response,
    parse_plan,
    strip_prose_tool_tags,
)
from .prompts import render_prompt


# Schemas for the control tools used to finish a task.  These are injected into
# native function-calling requests when the backend supports ``tools`` so that
# the model can explicitly signal completion instead of emitting JSON text.
FINAL_ANSWER_SCHEMA: dict[str, Any] = {
    "name": "final_answer",
    "description": "Provide the final answer to the task.",
    "parameters": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The final answer text to return to the user.",
            },
        },
        "required": ["answer"],
    },
}

DONE_SCHEMA: dict[str, Any] = {
    "name": "done",
    "description": "Signal that the task is complete or cannot proceed.",
    "parameters": {
        "type": "object",
        "properties": {},
    },
}


class LLMBrain:
    """Brain backed by a Model text-generation backend."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        max_retries: int = 3,
        backoff: float = 1.0,
        timeout: float = 60.0,
        backend: Model | None = None,
        fallback_models: list[Model] | None = None,
    ):
        if backend is None:
            backend = OpenAIModel(
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
                backoff=backoff,
                timeout=timeout,
            )
        if fallback_models:
            backend = FallbackModel([backend] + fallback_models)
        # Opt-in full-fidelity model I/O recording (one JSONL line per call):
        # set SWAYBOT_ROLLOUT_DIR and every generate() lands with the exact
        # messages/tools/output/metadata. Zero loop changes, no credentials.
        rollout_dir = os.environ.get("SWAYBOT_ROLLOUT_DIR", "").strip()
        if rollout_dir and not isinstance(backend, RecordingModel):
            backend = RecordingModel(backend, rollout_dir)
        self._model = backend
        self.api_key = getattr(backend, "api_key", None)
        self.base_url = getattr(backend, "base_url", None)
        self.model = getattr(backend, "model", None)
        # Advisory guards wired into the brain itself (not the Agent._run
        # loop) so they fire on every think() path, including the tau3bench
        # adapter which calls brain.think() directly and bypasses Agent.
        # observe() runs after the action is chosen; the resulting notice is
        # folded into the NEXT perception's behavior_guidance. Disabled by
        # passing the guard as None.
        self.repeat_guard: RepeatCallGuard | None = RepeatCallGuard()
        # Non-convergence guard catches runs that rotate across DIFFERENT tools
        # (so no single tool repeats) yet never emit final_answer — the airline
        # task-7 failure mode. Independent of repeat_guard's identity keying.
        self.nonconvergence_guard: NonConvergenceGuard | None = NonConvergenceGuard()
        self._repeat_notice: str | None = None

    def think(
        self,
        perception: dict,
        available_tools: list[str],
        metadata: dict | None = None,
        stream_callback: StreamCallback | None = None,
        reasoning_callback: ReasoningCallback | None = None,
    ) -> dict[str, Any]:
        call_info: dict = {}
        if perception.get("planning"):
            plan = self._plan(perception, available_tools, metadata=call_info)
            _merge_metadata(metadata, call_info)
            return plan
        if perception.get("exploring"):
            exploration = self._explore(
                perception, available_tools, metadata=call_info
            )
            _merge_metadata(metadata, call_info)
            return exploration

        # Fold any pending repeat-call notice into this turn's behavior guidance
        # so the model sees the nudge before choosing its next action.
        perception = self._with_repeat_notice(perception)

        # Feed the current step to the non-convergence guard so it counts on the
        # REAL task progress (perception["step"]) rather than on how many times
        # think() happened to be called. In the tau3bench half-duplex adapter,
        # think() is called once per inbound message but some messages take a
        # force_final / echo shortcut and never reach _observe_action — so a
        # call-count anchor undercounts. The step anchor reflects true progress.
        step = perception.get("step", 0)
        if self.nonconvergence_guard is not None:
            self.nonconvergence_guard.set_step(step)

        messages = [self._system_message(perception, available_tools)]
        if perception.get("messages"):
            messages.extend(perception["messages"])
        else:
            messages.append(
                {"role": "user", "content": render_prompt("user", **perception)}
            )

        # When using a backend that supports native tool calling, augment the
        # provided tool schemas with the control tools so the model can finish.
        supports_tools = getattr(self._model, "supports_tools", False)
        tool_descriptions = perception.get("tool_descriptions")
        tools_for_api = tool_descriptions
        if supports_tools and tool_descriptions:
            tools_for_api = list(tool_descriptions)
            existing = {t.get("name") for t in tools_for_api if isinstance(t, dict)}
            if "final_answer" not in existing:
                tools_for_api.append(FINAL_ANSWER_SCHEMA)
            if "done" not in existing:
                tools_for_api.append(DONE_SCHEMA)

        result = self._model.generate(
            messages,
            metadata=call_info,
            tools=tools_for_api if supports_tools else None,
            stream_callback=stream_callback,
            reasoning_callback=reasoning_callback,
        )
        _merge_metadata(metadata, call_info)
        response = normalize_response(result)
        if response is None:
            error_detail = call_info.get("error", "")
            message = "LLM call failed after retries"
            if error_detail:
                message = f"{message}: {error_detail}"
            action = fallback_action(perception, message)
        elif response.tool_call is not None:
            # Structural path: the model's action arrives as a ToolCall, never
            # as text to be re-parsed — no recovery logic can misfire here.
            action = {"name": response.tool_call.name, "args": response.tool_call.args}
        else:
            action = self._text_action(response.text, perception, available_tools)
        # If the model returned prose instead of a tool call, treat the prose as
        # the final answer rather than echo-truncating it.  This commonly happens
        # with reasoning models: reasoning tokens exhaust the output budget so
        # the final_answer tool call is never emitted (finish_reason=length), and
        # the model's visible content is a prose answer that parse_action can't
        # parse as JSON/XML.  Wrapping it as final_answer lets the turn complete
        # with the full answer instead of looping on a 200-char echo.
        # NOTE the is_explicit_tool_call guard: a genuine JSON/XML echo call
        # must execute as a tool call; only fallback-wrapped PROSE becomes the
        # final answer.  (Text-mode branch only; structural tool calls above
        # can never be hijacked.)
        if (
            action.get("name") == "echo"
            and response is not None
            and response.text
            and response.text.strip()
            and not is_explicit_tool_call(response.text, available_tools)
        ):
            prose = response.text.strip()
            if len(prose) > PROSE_CAP_CHARS:
                prose = prose[:PROSE_CAP_CHARS]
            # If the prose contains tool-call tags the model meant to invoke
            # (e.g. "<KB_search_bm25 query=...>") but couldn't because the tool
            # wasn't available this turn (force_final / text mode), those tags
            # are not part of the answer to the user — strip them so the user
            # never sees raw tool syntax. The surrounding prose is kept.
            if PROSE_TOOL_TAG_RE.search(prose):
                stripped = strip_prose_tool_tags(prose)
                if stripped.strip():
                    prose = stripped.strip()
            action = {"name": "final_answer", "args": {"answer": prose}}
        # Observe the FINAL action (after any rewrite) so the repeat guard sees
        # what the agent actually commits to, and can produce a notice for the
        # next think() call. Done last so fallbacks/rewrites are counted too.
        self._observe_action(action)
        return action

    def _with_repeat_notice(self, perception: dict) -> dict:
        """Fold the pending repeat-call notice into behavior_guidance.

        Returns a shallow copy so the caller's perception dict is not mutated.
        The notice is consumed (cleared) once folded.
        """
        notice = self._repeat_notice
        self._repeat_notice = None
        if not notice:
            return perception
        guidance = perception.get("behavior_guidance")
        new_guidance = f"{guidance}\n{notice}" if guidance else notice
        updated = dict(perception)
        updated["behavior_guidance"] = new_guidance
        return updated

    def _observe_action(self, action: dict) -> None:
        """Feed the chosen action to the repeat + non-convergence guards; stash
        any notice for the next think() call."""
        # Prefer the repeat guard's notice (more specific: it identifies the
        # offending tool/args); fall back to the non-convergence nudge.
        notice: str | None = None
        if self.repeat_guard is not None:
            notice = self.repeat_guard.observe(action)
        if notice is None and self.nonconvergence_guard is not None:
            notice = self.nonconvergence_guard.observe(action)
        if notice:
            self._repeat_notice = notice

    def _system_message(
        self, perception: dict, available_tools: list[str]
    ) -> dict:
        """Build the shared system message.

        When the backend supports native tool calling and tool schemas are
        available, use the ``system_fc`` template.  That template omits the
        inline JSON-schema dump and the "reply with JSON" instruction, so the
        model uses the OpenAI ``tools`` parameter instead of emitting JSON text.
        """
        tool_descriptions = perception.get("tool_descriptions")
        supports_tools = getattr(self._model, "supports_tools", False)
        template = "system_fc" if (supports_tools and tool_descriptions) else "system"
        return {
            "role": "system",
            "content": render_prompt(
                template,
                tool_descriptions=tool_descriptions,
                available_tools=available_tools,
                behavior_guidance=perception.get("behavior_guidance"),
                project_instructions=perception.get("project_instructions"),
            ),
        }

    @staticmethod
    def _text_action(
        raw: str, perception: dict, available_tools: list[str]
    ) -> dict[str, Any]:
        """Text-mode fallback: parse an action out of plain model output.

        Only reached when the backend produced no structural tool call —
        text-only backends or forced text mode. XML-tag recovery and the
        prose fallback live in parsing.py, scoped exactly to this path.
        """
        return parse_action(raw, perception, available_tools)

    def _plan(
        self,
        perception: dict,
        available_tools: list[str],
        metadata: dict | None = None,
    ) -> dict:
        messages = [
            self._system_message(perception, available_tools),
            {
                "role": "user",
                "content": render_prompt(
                    "plan",
                    task=perception["task"],
                    max_steps=perception["max_steps"],
                    tool_descriptions=perception.get("tool_descriptions"),
                    available_tools=available_tools,
                ),
            },
        ]

        response = normalize_response(self._model.generate(messages, metadata=metadata))
        if response is None:
            return {"name": "plan", "args": {"steps": []}}
        return parse_plan(response.text)

    def _explore(
        self,
        perception: dict,
        available_tools: list[str],
        metadata: dict | None = None,
    ) -> dict:
        messages = [
            self._system_message(perception, available_tools),
            {
                "role": "user",
                "content": render_prompt(
                    "explore",
                    memory_context=perception.get("memory_context", ""),
                    tool_descriptions=perception.get("tool_descriptions"),
                    available_tools=available_tools,
                ),
            },
        ]

        response = normalize_response(self._model.generate(messages, metadata=metadata))
        if response is None:
            return {"name": "explore", "args": {"task": "explore", "hypothesis": ""}}
        return {"name": "explore", "args": parse_exploration_response(response.text)}


def _merge_metadata(target: dict | None, source: dict) -> None:
    if target is not None:
        target.update(source)
