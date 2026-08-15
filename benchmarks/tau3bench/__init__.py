"""tau3-bench integration for SwayBot.

Wraps SwayBot's Agent/Brain/ToolRegistry to implement tau2's
``HalfDuplexAgent`` interface so that SwayBot can be evaluated on
multi-turn domain tasks (airline, retail, telecom, banking_knowledge, mock).

Usage::

    python -m benchmarks.tau3bench --domain mock --llm-agent gpt-4o
"""

from __future__ import annotations

import os
import time
from typing import Any

try:
    import tau2.utils.llm_utils as _tau2_llm_utils
except ImportError:  # pragma: no cover
    _tau2_llm_utils = None  # type: ignore[assignment]


def _tool_fallback_generate(
    original_generate: Any,
    model: str,
    messages: Any,
    tools: Any = None,
    tool_choice: str | None = None,
    call_name: str | None = None,
    **kwargs: Any,
) -> Any:
    """Wrap tau2's ``generate`` for two gateway-compatibility concerns.

    1. **Model remapping.**  tau2 hard-codes OpenAI/Anthropic model names for
       internal LLM calls that are *not* the agent or user simulator — e.g.
       ``DEFAULT_LLM_NL_ASSERTIONS = "gpt-4.1-2025-04-14"`` (used by the
       NL-assertion reward evaluator on retail/airline tasks) and
       ``DEFAULT_LLM_ENV_INTERFACE``.  Our remote gateway only serves the
       single model in ``SWAYBOT_MODEL``, so any other name returns
       ``No available channel for model <name>`` and crashes the run.
       We rewrite every model that is not our target to the target, keeping
       the litellm ``openai/`` provider prefix that the user simulator uses
       successfully.

    2. **Tool-calling fallback.**  Some OpenAI-compatible endpoints (e.g.
       vLLM without an explicit tool-call parser) reject requests that set
       ``tools``/``tool_choice``.  When we see a tool-related rejection,
       retry the same call without tools instead of crashing the
       evaluation.  If ``SWAYBOT_FORCE_TEXT_MODE`` is set, skip the doomed
       tool attempt entirely.
    """
    model = _remap_model(model)

    force_text = os.environ.get("SWAYBOT_FORCE_TEXT_MODE", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if force_text and tools is not None:
        return original_generate(
            model, messages, tools=None, tool_choice=None, **kwargs
        )

    # The remote gateway is slow (60-300s per request is common) and
    # occasionally drops connections mid-stream ("Connection error", "upstream
    # error", SSL EOF).  litellm's defaults — 3 retries, and crucially a long
    # request timeout that lets a hung socket block for minutes — combine to
    # stall whole runs.  We pin a hard per-request timeout and add an outer
    # retry loop for transient errors so a single flaky/hung request does not
    # abort 20 tasks' worth of work.
    request_timeout = float(
        os.environ.get("SWAYBOT_LITELLM_TIMEOUT", "180")
    )
    max_outer_retries = int(os.environ.get("SWAYBOT_LITELLM_OUTER_RETRIES", "5"))
    # litellm reads ``timeout`` from kwargs; tau2 forwards unknown kwargs.
    kwargs.setdefault("timeout", request_timeout)
    kwargs.setdefault("num_retries", 0)  # we handle retries ourselves

    message = None
    for attempt in range(max_outer_retries + 1):
        try:
            message = original_generate(
                model,
                messages,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs,
            )
            break
        except Exception as exc:
            error_text = str(exc)
            if _is_tool_call_rejection(error_text):
                message = original_generate(
                    model,
                    messages,
                    tools=None,
                    tool_choice=None,
                    **kwargs,
                )
                break
            if _is_rate_limited(error_text) and attempt < max_outer_retries:
                # The gateway caps at 100 req/min. When two domains run in
                # parallel (agent + user_simulator + NL-judge) we burst past it
                # and get 429s on individual tasks. Wait out the window so the
                # request succeeds on retry instead of being recorded as a 0.
                _sleep_for_rate_limit()
                continue
            if _is_transient(error_text) and attempt < max_outer_retries:
                time.sleep(3.0 * (attempt + 1))
                continue
            # Non-transient error, or retries exhausted: if this looks like an
            # LLM-judge call, fall back to a sentinel so the *evaluator*
            # degrades to reward=0 instead of crashing the domain.  Otherwise
            # re-raise.
            if call_name and "nl_assertion" in str(call_name):
                message = _empty_judge_message()
                break
            raise

    # tau2's NL-assertion evaluator (and similar internal callers) does
    # ``json.loads(assistant_message.content)`` on whatever the model returns.
    # Our reasoning-capable gateway model emits reasoning tokens and can return
    # ``content=None`` (finish_reason=length) or prose instead of strict JSON,
    # which raises ``JSONDecodeError`` and crashes the whole domain.  When this
    # looks like an LLM-judge call, coerce the content into valid JSON so the
    # evaluator degrades gracefully (treats the assertion as not-met) instead of
    # crashing the run.
    if call_name and "nl_assertion" in str(call_name):
        message = _coerce_judge_json(message)
    return message


def _empty_judge_message() -> Any:
    """Return a minimal assistant-message-like object with empty JSON content.

    Used when an NL-assertion judge call fails after all retries: instead of
    letting ``json.loads(None)`` crash the domain, we return an object whose
    ``content`` is valid ``{"results": []}`` so the evaluator scores the
    assertion as not-met (reward contribution 0).
    """
    import json as _json

    class _Stub:
        role = "assistant"
        content = _json.dumps({"results": []})
        tool_calls = None
        cost = 0.0
        usage = None
        raw_data = None
        generation_time_seconds = 0.0

    return _Stub()


def _is_transient(error_text: str) -> bool:
    """Return True if the error looks like a flaky connection/timeout/5xx."""
    lowered = error_text.lower()
    markers = (
        "connection error",
        "connection reset",
        "connection aborted",
        "remote disconnected",
        "ssl: unexpected_eof",
        "unexpected_eof_while_reading",
        "upstream error",
        "do request failed",
        "503",
        "502",
        "504",
        "timeout",
        "timed out",
        "read timeout",
        "api connection error",
        "serviceunavailableerror",
        "internalservererror",
    )
    return any(m in lowered for m in markers)


def _is_rate_limited(error_text: str) -> bool:
    """Return True if the error is a 429 / request-rate-cap hit."""
    lowered = error_text.lower()
    return (
        "rate limit" in lowered
        or "ratelimit" in lowered
        or "429" in lowered
        or "请求数限制" in error_text
        or "rate_limit" in lowered
    )


def _sleep_for_rate_limit() -> None:
    """Sleep long enough for the gateway's per-minute request cap to reset.

    The gateway caps at 100 req/min. When we hit it mid-run the safest action
    is to wait out the rest of the current minute window before retrying.
    """
    import datetime as _dt

    wait = 62.0 - (_dt.datetime.now().second % 60)
    wait = max(wait, 5.0)
    time.sleep(wait)


def _coerce_judge_json(message: Any) -> Any:
    """Ensure an LLM-judge response has parseable JSON in ``content``.

    tau2's NL-assertion evaluator calls ``json.loads(message.content)`` and
    expects ``{"results": [...]}``.  Our gateway model may return ``None``
    (truncated by reasoning tokens) or prose.  Return a sentinel "no
    expectations met" payload so the reward is 0 rather than a crash.
    """
    import json as _json

    sentinel = _json.dumps({"results": []})
    content = getattr(message, "content", None)
    if not content:
        try:
            message.content = sentinel
        except Exception:
            pass
        return message
    if isinstance(content, str):
        stripped = _strip_code_fence(content)
        try:
            _json.loads(stripped)
        except _json.JSONDecodeError:
            # Try to salvage a JSON object buried in prose.
            first = stripped.find("{")
            last = stripped.rfind("}")
            if first != -1 and last > first:
                candidate = stripped[first : last + 1]
                try:
                    _json.loads(candidate)
                    message.content = candidate
                    return message
                except _json.JSONDecodeError:
                    pass
            try:
                message.content = sentinel
            except Exception:
                pass
    return message


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def _remap_model(model: str) -> str:
    """Rewrite any model name to the single gateway-served model.

    tau2's internal evaluators (NL assertions, env interface) and reviewers
    hard-code ``gpt-4.1-*`` / ``claude-*`` defaults.  Our gateway only serves
    ``SWAYBOT_MODEL``, so unrelated names 503.  We keep the litellm
    ``openai/`` prefix (the form the user simulator uses successfully).
    """
    target = os.environ.get("SWAYBOT_MODEL") or "glm-5.1-fp8"
    litellm_target = target if "/" in target else f"openai/{target}"
    # If the caller already asked for our model (bare or prefixed), keep it.
    if model in (target, litellm_target, target.split("/")[-1]):
        return litellm_target
    return litellm_target


def _is_tool_call_rejection(error_text: str) -> bool:
    lowered = error_text.lower()
    markers = (
        "tool choice requires",
        "enable-auto-tool-choice",
        "tool_call_parser",
        "does not support tool",
        "tool choice",
        "tools are not supported",
    )
    return any(marker in lowered for marker in markers)


if _tau2_llm_utils is not None:
    _original_generate = _tau2_llm_utils.generate

    def _wrapped_generate(*args: Any, **kwargs: Any) -> Any:
        return _tool_fallback_generate(_original_generate, *args, **kwargs)

    _tau2_llm_utils.generate = _wrapped_generate

    # tau2 modules that import ``generate`` directly into their own namespace
    # captured the original function object.  Patch any already-loaded tau2
    # modules so the fallback applies everywhere (user simulator, agents, ...).
    import sys

    for _mod_name, _mod in list(sys.modules.items()):
        if _mod_name.startswith("tau2.") and getattr(_mod, "generate", None) is _original_generate:
            setattr(_mod, "generate", _wrapped_generate)

    # Patch the NL-assertion evaluator so a JSON parse failure on the judge's
    # response degrades to reward=0 instead of crashing the whole domain.
    # Our gateway model sometimes returns ``content=None`` (reasoning tokens
    # exhaust the output budget) or non-JSON prose.  ``_tool_fallback_generate``
    # already coerces content for judge calls, but as a belt-and-suspenders
    # guard we also wrap the evaluator itself.
    try:
        from tau2.evaluator import evaluator_nl_assertions as _nl_mod

        _orig_eval = _nl_mod.NLAssertionsEvaluator.evaluate_nl_assertions

        # ``RewardInfo`` (a pydantic model) validates the ``nl_assertions``
        # field against the real ``NLAssertionCheck`` class, so our fallback
        # must return genuine pydantic instances — a plain ``type(...)()``
        # object is rejected with a ValidationError that crashes the task.
        from tau2.data_model.simulation import NLAssertionCheck as _NLAC

        @classmethod  # type: ignore[misc]
        def _safe_eval_nl(cls, trajectory, nl_assertions):  # type: ignore[no-untyped-def]
            try:
                return _orig_eval(trajectory, nl_assertions)
            except Exception:
                # Treat all assertions as not-met; reward becomes 0.
                return [
                    _NLAC(
                        nl_assertion=a,
                        met=False,
                        justification="judge parse error",
                    )
                    for a in nl_assertions
                ]

        _nl_mod.NLAssertionsEvaluator.evaluate_nl_assertions = _safe_eval_nl  # type: ignore[attr-defined]
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Local sentence-transformer embedding fallback for banking_knowledge.
#
# tau2's banking_knowledge domain defaults to an OpenAI text-embedding model
# for dense retrieval.  That fails when no OpenAI key is available.  We swap
# the OpenAI embedder for a locally cached sentence-transformer model so the
# evaluation can run offline against the remote chat endpoint.
# ---------------------------------------------------------------------------

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from tau2.knowledge import embedders as _tau2_embedders
    from tau2.knowledge.embedders import openai_embedder as _tau2_openai_embedder
except Exception:  # pragma: no cover
    _tau2_embedders = None  # type: ignore[assignment]
    _tau2_openai_embedder = None  # type: ignore[assignment]


class _LocalSentenceTransformerEmbedder:
    """Drop-in replacement for tau2's OpenAIEmbedder using a local model."""

    _model: "SentenceTransformer | None" = None

    def __init__(self, model: str | None = None, **_kwargs: Any) -> None:
        # tau2 passes OpenAI model names such as ``text-embedding-3-large``.
        # If the supplied name looks like an OpenAI model, fall back to a local
        # sentence-transformer model selected by the environment variable.
        default_name = os.environ.get(
            "SENTENCE_TRANSFORMERS_MODEL",
            "paraphrase-multilingual-mpnet-base-v2",
        )
        if model and "text-embedding" not in model.lower():
            self._model_name = model
        else:
            self._model_name = default_name

    def _load(self) -> "SentenceTransformer":
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: list[str]) -> "np.ndarray":
        if not texts:
            raise ValueError("No text to embed.")
        model = self._load()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings

    def get_name(self) -> str:
        return f"local_sentence_transformer_{self._model_name}"


if _tau2_openai_embedder is not None:
    _tau2_openai_embedder.OpenAIEmbedder = _LocalSentenceTransformerEmbedder
if _tau2_embedders is not None:
    _tau2_embedders.OpenAIEmbedder = _LocalSentenceTransformerEmbedder
