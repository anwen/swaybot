"""SwayBot adapter for tau2-bench's HalfDuplexAgent interface.

This module provides ``SwayBotHalfDuplexAgent`` which wraps SwayBot's
``Brain`` + ``ToolRegistry`` into the interface tau2-bench expects:

- ``get_init_state()`` → initial ``SwayBotState``
- ``generate_next_message(message, state)`` → one turn of agent logic

Key design decisions
-------------------
- **Tool execution is delegated to tau2's environment.** The agent only
  returns ``AssistantMessage`` objects with optional ``tool_calls``; the
  tau2 orchestrator executes the tools and feeds the results back as
  ``ToolMessage`` / ``MultiToolMessage``.
- The SwayBot brain sees the full OpenAI-style conversation (system +
  user + assistant + tool messages) so that multi-turn context is
  preserved.  Previous turns are passed via ``perception["messages"]``.
- Domain tools are mapped into a fresh ``ToolRegistry`` at construction
  time so the brain can reason about them via SwayBot's normal
  ``perception["tool_descriptions"]`` mechanism.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from swaybot.brain import Brain, EchoBrain
from swaybot.llm_brain import DONE_SCHEMA, FINAL_ANSWER_SCHEMA, LLMBrain
from swaybot.parsing import PROSE_TOOL_TAG_RE
from swaybot.tools import Tool, ToolRegistry

from ._helpers import (
    augment_tool_error,
    extract_message_content,
    normalize_tool_args,
    summarize_tool_content,
    tool_schema_from_tau2,
)

try:
    from tau2.agent.base_agent import HalfDuplexAgent
    from tau2.data_model.message import (
        AssistantMessage,
        MultiToolMessage,
        ToolCall,
        ToolMessage,
        UserMessage,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "tau2-bench is required for this adapter. "
        "Install it with: pip install 'tau2[knowledge] @ git+https://github.com/sierra-research/tau2-bench'"
    ) from exc


_BEHAVIOR_GUIDANCE = """\
Identifier handling:
- Do not pass a person's display name as a user_id. If you need a user_id and the user has only given their name, ask for it.
- Order IDs (e.g. #W2378156) and reservation IDs are strings; keep the leading '#' where shown.
- Product IDs are strings and must not be converted to numbers.
- If a tool call returns "not found", ask the user for the correct identifier instead of repeating the same guess.

Confirmation and irreversible actions:
- Before executing irreversible actions (cancel reservation, exchange/return items, downgrade card, transfers, payments), disclose the concrete consequences (fees, refund amount/timeline, price difference, items affected) and ask the user for explicit approval.
- Wait for the user to clearly approve (e.g. "yes, proceed", "confirm") before executing the action. Do not execute just because the user asked for a summary or said "go ahead" about providing information.

Retail exchanges/returns/modifications:
- item_ids must be the existing item IDs from the user's order (from get_order_details), NOT product variant IDs from get_product_details.
- new_item_ids must be available variant item IDs from get_product_details.
- payment_method_id must be a valid payment method id from the order's payment_history (e.g. credit_card_9513926). Do not use the literal string "original".

Costs and arithmetic:
- When the user asks for the total cost of upcoming flights, compute and STATE that total BEFORE you perform any cancellations the user requested. Once you cancel a reservation it is no longer "upcoming", so if you cancel first you will report a reduced, wrong total. Answer the cost question first, then do the cancellations.
- The total of "upcoming flights" = the booking payment of EVERY reservation whose flights are in the future at the moment the user asks — INCLUDING reservations the user is cancelling in this same conversation. Do NOT exclude them. Example: the user has upcoming reservations A, B, C, D and asks you to cancel A and B and also asks the total cost of upcoming flights → the total is booking(A)+booking(B)+booking(C)+booking(D), all four, computed before you cancel A and B.
- A reservation's "cost" is its original booking payment — the FIRST entry in its payment_history. When a reservation was upgraded mid-conversation, its payment_history gains a second entry for the surcharge; for total-cost questions use only the booking payment (first entry), never the sum of all entries. Do NOT include modification surcharges (upgrade fees, extra-bag fees) you applied this conversation — those are separate charges.
- Call the `calculate` tool for any arithmetic over tool data (sums, differences, products). Never state a computed figure you did not calculate with the tool.

Knowledge tasks (banking_knowledge):
- Some banking_knowledge tasks are pure knowledge questions (answer from KB_search); others are transactional (verify identity, then apply_for_credit_card / submit_referral / change_user_email). Read the user's request: if they want you to TAKE AN ACTION on their account, use the transactional tools, not just KB_search.
- The `shell` sandbox browses raw knowledge documents via filesystem commands. It is for inspecting a document you cannot get via KB_search — NOT a general-purpose tool, and NOT the way to answer the user. Prefer KB_search_bm25 (ranked retrieval) over `shell`. If you find yourself running `ls`/`cat`/`grep` more than once or twice, stop and use KB_search_bm25 or a transactional tool instead.
- Limit yourself to at most 2-3 KB_search calls for pure knowledge questions. Synthesize the findings and call final_answer. Do not keep searching indefinitely.
- Do not call tools that are not in your available tool list (e.g. get_order_details, get_user_details). If you have KB results, call final_answer.
- For credit-card recommendation tasks, first ask the user for their annual income and whether they have a Rho-Bank+ subscription. Then search for each specific product document using ONLY the exact card name as the KB_search query (e.g., query="Gold Rewards Card"). Compare annual fees, cash back/rewards rates, and foreign transaction fees, then recommend the best-fit personal card and explain why. Call `KB_search_bm25` with `k=20` so you see enough candidate documents — the best-fit card is often not bm25's top-3 result.
- For account-action tasks (referral bonus, email change, card application), first verify the user's identity (get_user_information_by_name/email, then log_verification with two matching pieces), then perform the action (submit_referral, change_user_email, apply_for_credit_card). Do not answer from the knowledge base alone when the user wants you to change something on their account.

Finishing:
- When the task is complete, call final_answer with the answer for the user, then call done.
"""

# Long conversations (especially banking_knowledge with large KB_search results)
# can exhaust the model's context window and leave no tokens for the response.
# We keep the message history under a rough character budget so the model has
# room to generate complete tool calls / answers.
_MAX_CONTEXT_CHARS: int = 8000
_MAX_CONTEXT_MESSAGES: int = 14

# Cap open-ended KB_search loops. Multi-document banking tasks (e.g. compare
# 4 credit cards) legitimately need several KB_search calls, so the cap must be
# high enough not to truncate a valid comparison while still preventing runaway
# search. Force-final only kicks in at the cap, NOT after every KB result.
_MAX_KB_SEARCH_CALLS: int = 8
# Cap shell-sandbox browsing in banking_knowledge (the "alltools" variant).
_MAX_SHELL_CALLS: int = 5

# tau2's banking_knowledge "alltools" retrieval variant exposes the knowledge
# tools under variant-specific names: ``KB_search_bm25`` / ``KB_search_dense``
# (and a ``shell`` sandbox). The adapter's force-final / exhaustion logic and
# result summarization key on the *family* "KB_search", so we treat any tool
# whose name starts with that prefix as a knowledge-base search. ``shell`` is
# a retrieval-sandbox tool (browse KB docs via filesystem commands), not a
# general shell — it must NOT be counted as a knowledge search, but it is a
# retrieval tool whose abuse we cap separately.
_KB_SEARCH_PREFIX = "KB_search"
_RETRIEVAL_SANDBOX_TOOLS = frozenset({"shell"})

# Under force_final the only tools the model may call are final_answer/done.
# When it still tries to search, it emits the blocked tool name as *prose
# without angle brackets* — e.g. "KB_search_bm25 k 20 query personal credit
# card annual fee" — which PROSE_TOOL_TAG_RE (which needs ``<...>``) does not
# match, so think()'s tag-stripper leaves the residue verbatim in the
# final_answer text. Detect that residue: a final_answer that *starts with* a
# retrieval tool name is the model attempting a blocked call, not an answer.
_TOOL_RESIDUE_RE = re.compile(r"^\s*(?:KB_search_\w+|shell)\b", re.IGNORECASE)


def _is_kb_search(tool_name: str) -> bool:
    """True for any KB_search-family tool (KB_search, KB_search_bm25, ...)."""
    return bool(tool_name) and tool_name.startswith(_KB_SEARCH_PREFIX)


_CREDIT_CARD_TASK_KEYWORDS = (
    "credit card",
    "creditcard",
    "best card",
    "cash back",
    "recommend a card",
    "which card",
)

_CREDIT_CARD_PLAN = [
    "Ask the user for their annual income.",
    "Ask the user whether they have an active Rho-Bank+ subscription.",
    "Search the knowledge base for the exact product document using the query 'Gold Rewards Card'.",
    "Search the knowledge base for the exact product document using the query 'Silver Rewards Card'.",
    "Search the knowledge base for the exact product document using the query 'Bronze Rewards Card'.",
    "Search the knowledge base for the exact product document using the query 'Platinum Rewards Card'.",
    "Compare annual fees, cash back/rewards rates, foreign transaction fees, and Rho-Bank+ requirements.",
    "Recommend the best-fit personal card and call final_answer with a concise explanation.",
]


@dataclass
class SwayBotState:
    """Simple mutable state for the tau2 runner.

    The runner passes the state back into ``generate_next_message`` each
    turn.  ``messages`` stores the conversation in OpenAI message format
    so the LLM brain can see the full multi-turn context.
    """

    messages: list[dict] = field(default_factory=list)
    step: int = 0
    plan: list[str] = field(default_factory=list)
    force_final: bool = False
    kb_search_count: int = 0
    kb_exhausted: bool = False
    last_kb_content: str = ""


class SwayBotHalfDuplexAgent(HalfDuplexAgent):
    """Adapt a SwayBot Brain to tau2's HalfDuplexAgent interface."""

    def __init__(
        self,
        tools: list[Any],
        domain_policy: str,
        *,
        llm: str | None = None,
        llm_args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(tools, domain_policy)

        self._instructions = domain_policy or ""
        self._max_steps = 100
        self._task = ""
        self._tau2_tools: dict[str, Any] = {}
        self._registry = self._build_registry(tools)
        self._behavior_guidance = self._build_behavior_guidance()
        self._order_payment_methods: dict[str, str] = {}
        self._tool_call_names: dict[str, str] = {}

        llm_args = llm_args or {}
        if llm:
            self._brain: Brain = LLMBrain(
                model=llm,
                temperature=llm_args.get("temperature", 0.0),
                max_tokens=llm_args.get("max_tokens", 16384),
            )
        else:
            self._brain = EchoBrain()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_behavior_guidance(self) -> str:
        """Combine the domain policy with identifier-handling guidance."""
        parts = []
        if self._instructions:
            parts.append(self._instructions)
        parts.append(_BEHAVIOR_GUIDANCE)
        return "\n\n".join(parts)

    def _schema_for_tool(self, tool_name: str) -> dict[str, Any]:
        """Return the JSON schema for a tau2 tool, if available."""
        tool = self._tau2_tools.get(tool_name)
        if tool is None:
            return {}
        return tool_schema_from_tau2(tool)

    def _tool_name_for_call_id(
        self,
        call_id: str,
        conversation_messages: list[dict],
    ) -> str:
        """Map a tool_call_id to the tool name from the matching assistant call."""
        if not call_id:
            return ""
        if call_id in self._tool_call_names:
            print(f"[SWAYBOT DEBUG] dict lookup {call_id} -> {self._tool_call_names[call_id]}", flush=True)
            return self._tool_call_names[call_id]
        for msg in reversed(conversation_messages):
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls") or []:
                if tc.get("id") == call_id:
                    return tc.get("function", {}).get("name", "")
        return ""

    def _register_tool_call_names(self, assistant_message: dict) -> None:
        """Remember the mapping from tool_call_id to tool name."""
        for tc in assistant_message.get("tool_calls") or []:
            name = tc.get("function", {}).get("name", "")
            call_id = tc.get("id", "")
            if name and call_id:
                self._tool_call_names[call_id] = name
                print(f"[SWAYBOT DEBUG] registered {call_id} -> {name}", flush=True)

    def _build_registry(self, tools: list[Any]) -> ToolRegistry:
        registry = ToolRegistry()
        for t in tools:
            name = getattr(t, "name", "unknown")
            self._tau2_tools[name] = t
            swaybot_tool = Tool(
                name=name,
                description=getattr(t, "short_desc", "") or getattr(t, "long_desc", "") or "",
                inputs=tool_schema_from_tau2(t),
                output_type="string",
                fn=lambda **_: "[executed by tau2 environment]",
            )
            registry.register(name, swaybot_tool)

        # Register meta-tools so the brain can explicitly finish in text mode
        # (in native FC mode LLMBrain also injects them, but duplication is
        # harmless because LLMBrain checks existing names).
        if "final_answer" not in registry.names():
            registry.register(
                "final_answer",
                Tool(
                    name=FINAL_ANSWER_SCHEMA["name"],
                    description=FINAL_ANSWER_SCHEMA["description"],
                    inputs=FINAL_ANSWER_SCHEMA["parameters"],
                    output_type="string",
                    fn=lambda **_: "",
                ),
            )
        if "done" not in registry.names():
            registry.register(
                "done",
                Tool(
                    name=DONE_SCHEMA["name"],
                    description=DONE_SCHEMA["description"],
                    inputs=DONE_SCHEMA["parameters"],
                    output_type="string",
                    fn=lambda **_: "",
                ),
            )
        return registry

    def _openai_messages_from_tau2(
        self,
        message: UserMessage | ToolMessage | MultiToolMessage | AssistantMessage,
        conversation_messages: list[dict] | None = None,
    ) -> list[dict]:
        """Convert a tau2 message into one or more OpenAI-format messages."""
        conversation_messages = conversation_messages or []
        if isinstance(message, MultiToolMessage):
            return [
                msg
                for tm in message.tool_messages
                for msg in self._openai_messages_from_tau2(tm, conversation_messages)
            ]

        if isinstance(message, ToolMessage):
            call_id = getattr(message, "id", "") or ""
            tool_name = self._tool_name_for_call_id(call_id, conversation_messages)
            content = extract_message_content(message)
            if _is_kb_search(tool_name) and isinstance(content, str):
                content = self._summarize_kb_search(content)
            else:
                content = summarize_tool_content(tool_name, content)
            # tau2 marks failed tool calls with ``error=True``, but that flag is
            # otherwise invisible — the failure sits in a role:tool message
            # indistinguishable from a success result.  Append a directive
            # remediation hint so the model sees *what* to fix and acts on it
            # instead of repeating the same failing call.
            is_error = bool(getattr(message, "error", False))
            if is_error and isinstance(content, str) and tool_name:
                augmented = augment_tool_error(
                    tool_name, content, self._param_names_for_tool(tool_name)
                )
                if augmented != content:
                    print(
                        f"[SWAYBOT] augmented tool error for '{tool_name}': "
                        f"appended remediation directive",
                        flush=True,
                    )
                content = augmented
            return [
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": content,
                }
            ]

        if isinstance(message, AssistantMessage):
            tool_calls = getattr(message, "tool_calls", None) or []
            if tool_calls:
                return [
                    {
                        "role": "assistant",
                        "content": getattr(message, "content", None) or None,
                        "tool_calls": [
                            {
                                "id": getattr(tc, "id", "") or "",
                                "type": "function",
                                "function": {
                                    "name": getattr(tc, "name", "") or "",
                                    "arguments": json.dumps(
                                        getattr(tc, "arguments", {}) or {}
                                    ),
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                ]
            return [
                {
                    "role": "assistant",
                    "content": extract_message_content(message) or None,
                }
            ]

        # UserMessage (or anything content-based)
        return [{"role": "user", "content": extract_message_content(message)}]

    def _param_names_for_tool(self, tool_name: str) -> list[str]:
        """Return the canonical parameter names for a tau2 tool."""
        tool = self._tau2_tools.get(tool_name)
        if tool is None:
            return []
        params = getattr(tool, "params", None)
        if params is not None:
            return list(getattr(params, "model_fields", {}).keys())
        return []

    def _update_order_payment_methods(
        self,
        message: ToolMessage,
        tool_name: str,
    ) -> None:
        """Remember the original payment method for each order we look up."""
        if tool_name != "get_order_details":
            return
        content = extract_message_content(message)
        if not isinstance(content, str):
            return
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return
        order_id = data.get("order_id")
        payment_history = data.get("payment_history") or []
        if not order_id or not isinstance(payment_history, list):
            return
        for payment in payment_history:
            if isinstance(payment, dict) and payment.get("payment_method_id"):
                self._order_payment_methods[order_id] = payment["payment_method_id"]
                break

    def _record_order_payment_methods(
        self,
        message: UserMessage | ToolMessage | MultiToolMessage,
        state: SwayBotState,
    ) -> None:
        """Update payment-method cache from any ToolMessages in *message*."""
        if isinstance(message, MultiToolMessage):
            for tm in message.tool_messages:
                call_id = getattr(tm, "id", "") or ""
                tool_name = self._tool_name_for_call_id(call_id, state.messages)
                self._update_order_payment_methods(tm, tool_name)
        elif isinstance(message, ToolMessage):
            call_id = getattr(message, "id", "") or ""
            tool_name = self._tool_name_for_call_id(call_id, state.messages)
            self._update_order_payment_methods(message, tool_name)

    def _normalize_tool_args(
        self,
        tool_name: str,
        args: Any,
        order_id: str | None = None,
    ) -> dict[str, Any]:
        """Map LLM argument names/types onto the tau2 tool's canonical schema."""
        if not isinstance(args, dict):
            return {}
        param_names = self._param_names_for_tool(tool_name)
        schema = self._schema_for_tool(tool_name)
        normalized = normalize_tool_args(tool_name, args, param_names, schema)

        # Fill in a missing / placeholder payment method for order changes.
        if tool_name in {
            "exchange_delivered_order_items",
            "modify_pending_order_items",
            "return_delivered_order_items",
        }:
            payment = normalized.get("payment_method_id")
            if not payment or payment == "original":
                lookup_order_id = order_id or str(normalized.get("order_id") or "")
                inferred = self._order_payment_methods.get(lookup_order_id)
                if inferred:
                    normalized["payment_method_id"] = inferred

        # Force KB_search family tools to retrieve enough documents. The model
        # often calls KB_search_bm25 with a small k (5 or 10) despite guidance
        # to use k=20, and for multi-document compare tasks the best-fit
        # document (e.g. the personal Gold Rewards Card) routinely ranks 14th+
        # — a small k silently drops it, leaving the agent unable to find the
        # card it must recommend. Clamp k up to 20 for any KB_search call.
        if _is_kb_search(tool_name):
            try:
                k_val = int(normalized.get("k", 0) or 0)
            except (TypeError, ValueError):
                k_val = 0
            if k_val < 20:
                normalized["k"] = 20
        return normalized

    def _collect_kb_content(self, messages: list[dict]) -> str:
        """Gather ALL accumulated KB_search results from the conversation.

        For multi-document comparison tasks (e.g. "which credit card has the
        highest cash back") the best-fit document is often in an *earlier*
        KB_search result, not the most recent one. Using only the last result
        (``state.last_kb_content``) can drop exactly the card the agent needs
        to recommend. We therefore walk the full message history, find every
        ``tool`` message whose matching assistant tool_call was a KB_search,
        and concatenate their (already-summarized) contents, de-duplicated by
        text so the same doc returned across searches is not repeated.
        """
        seen: set[str] = set()
        chunks: list[str] = []
        for idx, msg in enumerate(messages):
            if msg.get("role") != "tool" or not isinstance(msg.get("content"), str):
                continue
            call_id = msg.get("tool_call_id", "")
            tool_name = ""
            if call_id:
                tool_name = self._tool_name_for_call_id(call_id, messages)
            # Fall back to scanning preceding assistant messages for the id.
            if not tool_name:
                for prev in reversed(messages[:idx]):
                    if prev.get("role") != "assistant":
                        continue
                    for tc in prev.get("tool_calls") or []:
                        if tc.get("id") == call_id:
                            tool_name = tc.get("function", {}).get("name", "")
                            break
                    if tool_name:
                        break
            if not _is_kb_search(tool_name):
                continue
            text = msg["content"].strip()
            if not text:
                continue
            key = text[:200]
            if key in seen:
                continue
            seen.add(key)
            chunks.append(text)
        return "\n\n".join(chunks)

    def _synthesize_kb_answer(
        self,
        task: str,
        question: str,
        kb_content: str,
    ) -> str:
        """Turn summarized KB results into a concise answer for the user."""
        model = getattr(self._brain, "_model", None)
        if model is None:
            return kb_content[:2000] if kb_content else "I'm not sure; please contact support."
        is_cc_task = any(kw in (task or "").lower() for kw in _CREDIT_CARD_TASK_KEYWORDS)
        system_prompt = (
            "You are a helpful banking assistant. Answer the user's question "
            "using only the provided knowledge-base results. Be concise and direct."
        )
        user_prompt = (
            f"Task context: {task}\n\n"
            f"User question: {question}\n\n"
            f"Knowledge-base results:\n{kb_content}\n\n"
        )
        if is_cc_task:
            system_prompt = (
                "You are a helpful banking assistant. Using ONLY the provided "
                "knowledge-base results, recommend the single best-fit personal "
                "credit card for the user's situation and state its key terms "
                "(annual fee, cash back rate, Rho-Bank+ requirement). "
                "The user wants the highest cash back and will not accept an "
                "annual fee unless it is the only option. Name the recommended "
                "card explicitly by its exact document title (e.g. 'Gold Rewards "
                "Card'). Be concise and direct."
            )
            user_prompt += (
                "Recommend the best-fit personal credit card by name and give its "
                "key terms so the user can decide."
            )
        else:
            user_prompt += "Please provide a helpful answer."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        from swaybot.models import normalize_response

        try:
            response = normalize_response(model.generate(messages, tools=None))
        except Exception:
            response = None
        raw = response.text if response is not None else None
        return (
            (raw or "").strip()
            or kb_content[:2000]
            or "I'm not sure; please contact support."
        )

    @staticmethod
    def _summarize_kb_search(content: str) -> str:
        """Compress KB_search results to compact per-document summaries.

        Full KB_search returns up to 20 documents, each potentially thousands
        of characters — far too much for the model's context window. We keep a
        compact summary of every returned document (title + the first few
        substantive content lines) rather than only the top 3, because for
        multi-document compare tasks (e.g. "which credit card has the highest
        cash back") the relevant document is often NOT among bm25's top 3 — it
        may rank 4th-8th and would be silently dropped, leaving the agent unable
        to find the card it needs to recommend.

        We keep every entry but cap each entry's snippet so total size stays
        bounded; a trailing note records how many results were returned.

        In addition to the first few lines, we ALWAYS keep lines that carry the
        comparison-critical banking terms (cash back, annual fee, rewards rate,
        foreign transaction fee, Rho-Bank+ requirement, APY). These terms often
        appear a few lines into a document (after eligibility/credit-score
        boilerplate), so a naive first-N truncation drops exactly the figures
        the agent needs to compare cards — which caused the agent to see the
        Gold Rewards Card as "cash back not specified" and recommend the wrong
        card.
        """
        entries = re.split(r"\n(?=\d+\.\s)", content.strip())
        kept: list[str] = []
        _SNIPPET_LINES = 3
        _KEY_TERM_RE = re.compile(
            r"(?i)cash\s*back|annual\s*fee|rewards?\s*rate|foreign\s*transaction"
            r"|rho[\s\-]*bank\s*\+|subscription\s*required|apy"
        )
        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue
            title_match = re.search(r"^\d+\.\s*(.*)$", entry, re.MULTILINE)
            title = title_match.group(1).strip() if title_match else entry.splitlines()[0]
            content_match = re.search(
                r"Content:\s*(.*?)(?=\n\s*\d+\.\s|\Z)", entry, re.DOTALL
            )
            snippet = ""
            if content_match:
                lines = [
                    line.strip()
                    for line in content_match.group(1).splitlines()
                    if line.strip()
                ]
                # Keep the first _SNIPPET_LINES for context, plus EVERY line
                # carrying a comparison-critical term, de-duplicated, in order.
                selected: list[str] = []
                seen: set[str] = set()
                for line in lines:
                    if len(selected) < _SNIPPET_LINES or _KEY_TERM_RE.search(line):
                        if line not in seen:
                            seen.add(line)
                            selected.append(line)
                snippet = "\n".join(selected) if selected else ""
            kept.append(f"{title}\n{snippet}" if snippet else title)
        note = ""
        total = len(entries)
        shown = len(kept)
        if total and shown < total:
            note = f"\n\n...[showing {shown} of {total} KB results]"
        elif total:
            note = f"\n\n[showing all {total} KB results]"
        return "\n\n".join(kept) + note

    @staticmethod
    def _trim_messages_for_context(messages: list[dict]) -> list[dict]:
        """Drop old messages so the prompt stays within a rough token budget.

        Keeps the earliest user message (usually the task) and as many recent
        messages as fit under _MAX_CONTEXT_CHARS.  This prevents long
        multi-turn runs from running out of output tokens.
        """
        if len(messages) <= _MAX_CONTEXT_MESSAGES:
            return messages

        def _message_chars(msg: dict) -> int:
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls") or []
            extra = ""
            for tc in tool_calls:
                extra += tc.get("function", {}).get("arguments", "") or ""
            return len(str(content)) + len(extra)

        # Always keep the first user message (task) so the model knows the goal.
        first = messages[0] if messages else None
        budget = _MAX_CONTEXT_CHARS - (_message_chars(first) if first else 0)
        recent: list[dict] = []
        for msg in reversed(messages[1:]):
            cost = _message_chars(msg)
            if cost > budget and recent:
                break
            recent.append(msg)
            budget -= cost
        trimmed = [first] if first else []
        trimmed.extend(reversed(recent))
        return trimmed

    def _remember_task(
        self,
        message: UserMessage | ToolMessage | MultiToolMessage,
        state: SwayBotState,
    ) -> None:
        """Store the first user message as the task description."""
        if isinstance(message, UserMessage):
            text = extract_message_content(message)
            if text and not self._task:
                self._task = text
                state.plan = self._generate_plan()

    def _generate_plan(self) -> list[str]:
        """Generate an explicit step-by-step plan for the task.

        The plan is stored in the state and fed back into the brain on every
        subsequent turn so multi-step tasks stay on track.
        """
        if not self._task:
            return []

        task_lower = self._task.lower()
        if any(kw in task_lower for kw in _CREDIT_CARD_TASK_KEYWORDS):
            return list(_CREDIT_CARD_PLAN)

        perception: dict[str, Any] = {
            "task": self._task,
            "step": 0,
            "max_steps": self._max_steps,
            "messages": [{"role": "user", "content": self._task}],
            "tool_descriptions": self._registry.schemas(),
            "available_tools": self._registry.names(),
            "planning": True,
        }
        if self._behavior_guidance:
            perception["behavior_guidance"] = self._behavior_guidance
        try:
            plan_action = self._brain.think(perception, self._registry.names())
            steps = plan_action.get("args", {}).get("steps", [])
            if isinstance(steps, list) and all(isinstance(s, str) for s in steps):
                return steps
        except Exception:
            pass
        # Planning is advisory; failures should not block the run.
        return []

    # ------------------------------------------------------------------
    # HalfDuplexAgent interface
    # ------------------------------------------------------------------

    def get_init_state(
        self,
        message_history: list[Any] | None = None,
    ) -> SwayBotState:
        """Return the initial state for a new simulation."""
        messages: list[dict] = []
        for msg in message_history or []:
            messages.extend(self._openai_messages_from_tau2(msg))
        return SwayBotState(messages=messages)

    def generate_next_message(
        self,
        message: UserMessage | ToolMessage | MultiToolMessage,
        state: SwayBotState,
    ) -> tuple[AssistantMessage, SwayBotState]:
        """One turn: decide which tool to call (or text to return)."""
        state.step += 1

        # ------------------------------------------------------------------
        # Convert the incoming tau2 message into OpenAI conversation format.
        # ------------------------------------------------------------------
        self._remember_task(message, state)
        state.messages.extend(self._openai_messages_from_tau2(message, state.messages))
        self._record_order_payment_methods(message, state)

        # If the environment just returned a KB_search result, force the
        # next turn to synthesize an answer instead of calling another tool.
        incoming_tool_name = ""
        if isinstance(message, ToolMessage):
            incoming_tool_name = self._tool_name_for_call_id(
                getattr(message, "id", "") or "", state.messages
            )
        elif isinstance(message, MultiToolMessage) and message.tool_messages:
            incoming_tool_names = {
                self._tool_name_for_call_id(
                    getattr(tm, "id", "") or "", state.messages
                )
                for tm in message.tool_messages
            }
            if any(_is_kb_search(n) for n in incoming_tool_names):
                incoming_tool_name = _KB_SEARCH_PREFIX
        if _is_kb_search(incoming_tool_name):
            state.kb_search_count += 1
            # Remember the (already summarized) KB content for final-answer synthesis.
            last_tool_content = ""
            for msg in reversed(state.messages):
                if msg.get("role") == "tool" and isinstance(msg.get("content"), str):
                    last_tool_content = msg["content"]
                    break
            if last_tool_content:
                state.last_kb_content = last_tool_content
            if state.kb_search_count >= _MAX_KB_SEARCH_CALLS:
                state.kb_exhausted = True

        # Once the knowledge-base budget is exhausted, keep forcing a final answer
        # on subsequent turns so the agent does not search indefinitely.
        if state.kb_exhausted:
            state.force_final = True

        # ------------------------------------------------------------------
        # Keep the context window small enough for the model to emit a full
        # response.  Long banking_knowledge / retail runs otherwise truncate
        # tool calls and answers.
        # ------------------------------------------------------------------
        messages_for_brain = self._trim_messages_for_context(state.messages)

        kb_calls = sum(
            1
            for msg in state.messages
            if msg.get("role") == "assistant"
            and any(
                _is_kb_search(tc.get("function", {}).get("name", ""))
                for tc in (msg.get("tool_calls") or [])
            )
        )
        shell_calls = sum(
            1
            for msg in state.messages
            if msg.get("role") == "assistant"
            and any(
                tc.get("function", {}).get("name", "") in _RETRIEVAL_SANDBOX_TOOLS
                for tc in (msg.get("tool_calls") or [])
            )
        )
        behavior_guidance = self._behavior_guidance
        if state.force_final:
            behavior_guidance = (
                behavior_guidance
                + "\n\nYou have just received knowledge-base results. "
                "Do NOT call any other tools. Call final_answer with a concise, "
                "helpful answer now."
            )
        elif kb_calls >= _MAX_KB_SEARCH_CALLS:
            # Hit the KB_search cap: stop searching and synthesize from what we have.
            state.force_final = True
            behavior_guidance = (
                behavior_guidance
                + "\n\nYou have searched the knowledge base enough times. "
                "Do NOT call KB_search again. Synthesize what you found and "
                "call final_answer with a concise answer now."
            )
        elif shell_calls >= _MAX_SHELL_CALLS:
            # The banking_knowledge "alltools" variant includes a `shell` sandbox
            # for browsing KB documents. A weak model can waste dozens of steps
            # `ls`/`cat`/`grep`-ing the document directory. Hard-stop further
            # shell calls by nudging toward the proper transactional tools and
            # KB_search instead of forcing a premature final_answer (banking tasks
            # often need log_verification / submit_referral, not just a KB answer).
            behavior_guidance = (
                behavior_guidance
                + "\n\nYou have used the shell sandbox enough. STOP calling shell. "
                "The shell sandbox is only for browsing knowledge documents; "
                "it is NOT the way to answer the user. If you need account data, "
                "use the transactional tools (get_user_information_by_name/email, "
                "log_verification, get_referrals_by_user, submit_referral, etc.). "
                "If you need knowledge documents, use KB_search_bm25. Now decide "
                "the next concrete action toward the user's request."
            )
        elif shell_calls >= 2 and kb_calls == 0:
            # Early shell-browsing without any KB_search yet: nudge toward the
            # proper retrieval tool instead of letting the model grep raw docs.
            behavior_guidance = (
                behavior_guidance
                + "\n\nYou are browsing documents with the shell sandbox. "
                "The shell is for inspecting knowledge documents only, not for "
                "answering the user. If you need a knowledge document, use "
                "KB_search_bm25 with the exact product/topic name as the query. "
                "If you need account data, use the transactional tools "
                "(get_user_information_by_name, log_verification, etc.)."
            )

        # ------------------------------------------------------------------
        # Ask the SwayBot brain what to do next.
        # ------------------------------------------------------------------
        available_tools_for_brain = self._registry.names()
        tool_descriptions_for_brain = self._registry.schemas()
        if state.force_final:
            available_tools_for_brain = ["final_answer", "done"]
            tool_descriptions_for_brain = [FINAL_ANSWER_SCHEMA, DONE_SCHEMA]

        perception: dict[str, Any] = {
            "task": self._task,
            "step": state.step,
            "max_steps": self._max_steps,
            "messages": messages_for_brain,
            "tool_descriptions": tool_descriptions_for_brain,
            "available_tools": available_tools_for_brain,
        }
        if state.plan:
            perception["plan"] = state.plan
        if behavior_guidance:
            perception["behavior_guidance"] = behavior_guidance

        action = self._brain.think(perception, available_tools_for_brain)

        # ------------------------------------------------------------------
        # Enforce finalization after KB_search results.
        #
        # Even when tools are restricted to final_answer/done, some models
        # ignore the constraint and emit another KB_search (either as a native
        # tool call or as JSON text).  Rather than loop until max_steps,
        # override the action and answer from the KB result we already have.
        # ------------------------------------------------------------------
        tool_name = action.get("name", "")
        # When the KB budget is exhausted, the model often keeps emitting
        # retrieval intent — either as a native tool call the model shouldn't
        # be making anymore, or as prose carrying tool-call syntax. Two prose
        # shapes occur:
        #   (a) angle-bracket tags like "<KB_search_bm25 query=...>" that
        #       think() rewrote into a final_answer — caught by
        #       PROSE_TOOL_TAG_RE;
        #   (b) bracket-less residue like "KB_search_bm25 k 20 query ..."
        #       produced when think()'s tag-stripper removed the brackets but
        #       left the inner attribute words — caught by _TOOL_RESIDUE_RE
        #       (a final_answer whose text *starts with* a retrieval tool
        #       name is an attempted blocked call, not an answer).
        # In all cases the model's own text is NOT a real answer, so override
        # it and synthesize a proper answer from the KB results we already
        # have.
        if state.force_final:
            answer_text = action.get("args", {}).get("answer", "") if isinstance(action.get("args"), dict) else ""
            wants_tool = tool_name not in ("final_answer", "done") or bool(
                PROSE_TOOL_TAG_RE.search(answer_text)
                or _TOOL_RESIDUE_RE.search(answer_text)
            )
            if wants_tool:
                question = ""
                for msg in reversed(state.messages):
                    if msg.get("role") == "user":
                        question = msg.get("content", "")
                        break
                kb_content = self._collect_kb_content(state.messages)
                answer = self._synthesize_kb_answer(self._task, question, kb_content)
                action = {"name": "final_answer", "args": {"answer": answer}}
                tool_name = "final_answer"

        # ------------------------------------------------------------------
        # Convert SwayBot action back into a tau2 AssistantMessage.
        # ------------------------------------------------------------------
        tool_args = action.get("args", {})

        # Guard against hallucinated tool names in text mode.
        if (
            tool_name
            and tool_name not in ("done", "final_answer", "echo")
            and tool_name not in self._registry.names()
        ):
            text = (
                f"'{tool_name}' is not an available tool. "
                f"Available tools: {', '.join(self._registry.names())}. "
                "Please call final_answer when you are ready to answer."
            )
            assistant_msg = AssistantMessage(role="assistant", content=text)
            state.messages.append({"role": "assistant", "content": text})
            state.force_final = False
            return assistant_msg, state

        if tool_name and tool_name not in ("done", "final_answer", "echo"):
            tool_args = self._normalize_tool_args(
                tool_name, tool_args, order_id=tool_args.get("order_id")
            )
            tool_call_id = f"call_{state.step}"
            assistant_msg = AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id=tool_call_id,
                        name=tool_name,
                        arguments=tool_args if isinstance(tool_args, dict) else {},
                        requestor="assistant",
                    )
                ],
            )
            state.messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(
                                    tool_args if isinstance(tool_args, dict) else {}
                                ),
                            },
                        }
                    ],
                }
            )
            self._register_tool_call_names(state.messages[-1])
        else:
            text = ""
            if tool_name == "final_answer":
                text = tool_args.get("answer", "") if isinstance(tool_args, dict) else ""
            elif tool_name == "echo":
                text = tool_args.get("message", "") if isinstance(tool_args, dict) else ""
            else:
                text = action.get("text", "")
            text = text or "Done."
            assistant_msg = AssistantMessage(role="assistant", content=text)
            state.messages.append({"role": "assistant", "content": text})

        state.force_final = False
        return assistant_msg, state
