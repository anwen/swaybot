import json
from typing import Any

from .brain import ReasoningCallback, StreamCallback
from .loop_guards import NonConvergenceGuard, RepeatCallGuard
from .models import FallbackModel, Model, OpenAIModel
from .parsing import parse_exploration_response, strip_markdown_fences
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

        raw = self._model.generate(
            messages,
            metadata=call_info,
            tools=tools_for_api if supports_tools else None,
            stream_callback=stream_callback,
            reasoning_callback=reasoning_callback,
        )
        _merge_metadata(metadata, call_info)
        if raw is None:
            error_detail = call_info.get("error", "")
            message = "LLM call failed after retries"
            if error_detail:
                message = f"{message}: {error_detail}"
            action = _fallback(perception, message)
        else:
            action = _parse_action(raw, perception, available_tools)
        # If the model returned prose instead of a tool call, treat the prose as
        # the final answer rather than echo-truncating it.  This commonly happens
        # with reasoning models: reasoning tokens exhaust the output budget so
        # the final_answer tool call is never emitted (finish_reason=length), and
        # the model's visible content is a prose answer that _parse_action can't
        # parse as JSON/XML.  Wrapping it as final_answer lets the turn complete
        # with the full answer instead of looping on a 200-char echo.
        if action.get("name") == "echo" and raw and raw.strip():
            prose = raw.strip()
            if len(prose) > 4000:
                prose = prose[:4000]
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

        raw = self._model.generate(messages, metadata=metadata)
        if raw is None:
            return {"name": "plan", "args": {"steps": []}}
        return _parse_plan(raw)

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

        raw = self._model.generate(messages, metadata=metadata)
        if raw is None:
            return {"name": "explore", "args": {"task": "explore", "hypothesis": ""}}
        return {"name": "explore", "args": parse_exploration_response(raw)}


def _merge_metadata(target: dict | None, source: dict) -> None:
    if target is not None:
        target.update(source)


def _extract_tool_call(
    raw: str, available_tools: list[str] | None = None
) -> dict[str, Any] | None:
    """Normalize Qwen/DeepSeek-style XML tool calls into JSON actions.

    Models fine-tuned for ``<tool_call>`` formatting sometimes emit either
    ``<tool_call>{"name": ..., "args": ...}</tool_call>`` or a nested
    ``<function=name><parameter=key>value</parameter>...</function>`` block.
    We convert both into SwayBot's standard ``{"name": ..., "args": ...}``.

    A third variant is ``<ToolName key="value">`` prose tags: some models
    (notably glm-5.1-fp8 on retrieval tools) emit a known tool name directly
    as an XML-like tag instead of a native tool_call. When ``available_tools``
    is provided we recognize such tags ONLY for names that are actually
    registered tools, so ordinary HTML/markdown prose is not misparsed.
    """
    text = raw.strip()
    if not text.startswith("<"):
        return None

    import re

    # Variant 1: <tool_call> { "name": "...", "args": {...} } </tool_call>
    if "tool_call" in text:
        stripped = text
        for tag in ("<tool_call>", "</tool_call>", "<tool_call>", "</tool_call>"):
            stripped = stripped.replace(tag, "")
        stripped = stripped.strip()
        try:
            data = json.loads(stripped)
            if isinstance(data, dict) and "name" in data:
                return {"name": data["name"], "args": data.get("args", {})}
        except json.JSONDecodeError:
            pass

    # Variant 2: <function=name><parameter=key>value</parameter>...</function>
    func_match = re.search(r"<function=([^>\s]+)>", text)
    if func_match:
        return _parse_function_tag(text, func_match.group(1))

    # Variant 3: <ToolName key="value">  (prose-embedded tool tag). Only
    # recognized when the tag name is a registered tool, so arbitrary HTML is
    # left untouched. The model often emits several such tags per turn; we
    # take the first (the brain processes one action per turn).
    if available_tools:
        tool_set = set(available_tools)
        for m in re.finditer(r"<([A-Za-z_][\w-]*)((\s+[^<>]*?)?)\s*/?>", text):
            name = m.group(1)
            if name in tool_set:
                return {"name": name, "args": _parse_tag_attrs(m.group(2))}
    return None


def _parse_function_tag(text: str, name: str) -> dict[str, Any]:
    import re

    args: dict[str, Any] = {}
    for param_match in re.finditer(
        r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", text, re.DOTALL
    ):
        key = param_match.group(1)
        value_text = param_match.group(2).strip()
        # XML parameter values are plain text. Only interpret values that are
        # clearly JSON literals (objects, arrays, booleans, null). Plain
        # numbers are kept as strings so ID fields such as product_id/zip
        # are not accidentally coerced to integers.
        lowered = value_text.lower()
        if value_text.startswith(("{", "[")) or lowered in (
            "true",
            "false",
            "null",
        ):
            try:
                value = json.loads(value_text)
            except json.JSONDecodeError:
                value = value_text
        else:
            value = value_text
        args[key] = value
    return {"name": name, "args": args}


def _parse_tag_attrs(attr_text: str) -> dict[str, Any]:
    """Parse ``key="value"`` / ``key='value'`` / ``key=value`` attributes from
    a prose tool tag like ``<KB_search query="gold card">``. Bare (quoteless)
    values run to the next whitespace; quoted values may contain spaces."""
    import re

    args: dict[str, Any] = {}
    pattern = r'''(\w[\w-]*)\s*=\s*("([^"]*)"|'([^']*)'|(\S+))'''
    for m in re.finditer(pattern, attr_text):
        value = m.group(3) if m.group(3) is not None else (
            m.group(4) if m.group(4) is not None else m.group(5)
        )
        args[m.group(1)] = value if value is not None else ""
    return args


def _parse_action(
    raw: str, perception: dict, available_tools: list[str] | None = None
) -> dict[str, Any]:
    raw = strip_markdown_fences(raw)

    # Some Qwen/DeepSeek-style models emit tool calls in XML tags even in
    # text mode. Try to normalize those into the standard JSON action shape.
    tool_action = _extract_tool_call(raw, available_tools)
    if tool_action is not None:
        return tool_action

    try:
        action = json.loads(raw)
    except json.JSONDecodeError:
        return _fallback(perception, raw)

    if not isinstance(action, dict) or "name" not in action:
        return _fallback(perception, raw)

    return action


def _parse_plan(raw: str) -> dict[str, Any]:
    raw = strip_markdown_fences(raw)

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        return {"name": "plan", "args": {"steps": []}}

    if isinstance(plan, list) and all(isinstance(s, str) for s in plan):
        return {"name": "plan", "args": {"steps": plan}}
    if isinstance(plan, dict):
        steps = plan.get("steps", plan.get("plan", []))
        if isinstance(steps, list) and all(isinstance(s, str) for s in steps):
            return {"name": "plan", "args": {"steps": steps}}

    return {"name": "plan", "args": {"steps": []}}


def _fallback(perception: dict, raw: str) -> dict[str, Any]:
    if perception["step"] + 1 >= perception["max_steps"]:
        return {"name": "done", "args": {}}
    # The model emitted prose instead of a parseable tool call. This is common
    # for reasoning models whose visible-content budget is eaten by reasoning
    # tokens, so the final_answer tool call is never emitted. Rather than
    # truncate to 200 chars (which causes a "you got cut off" -> re-emit loop),
    # pass the full prose through as the user-facing message so the turn
    # completes. A generous cap only guards against pathological output.
    message = (raw or "").strip()
    if len(message) > 4000:
        message = message[:4000]
    return {"name": "echo", "args": {"message": message}}
