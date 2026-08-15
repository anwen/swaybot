"""Standalone helpers for tau3-bench adapter.

These functions have no dependency on tau2-bench and can be tested
without installing it.
"""

from __future__ import annotations

from typing import Any, cast
import json
import re


def make_placeholder_fn(tool_name: str):
    """Create a no-op placeholder function for a tau2 tool name.

    The actual execution goes through the tau2 environment, so
    the SwayBot registry only needs the function for schema metadata.
    """
    def _placeholder(**_):
        return f"[{tool_name} executed via tau2 environment]"
    _placeholder.__name__ = tool_name
    _placeholder.__doc__ = f"Placeholder for tau2 tool: {tool_name}"
    return _placeholder


def extract_message_content(message: object) -> str:
    """Extract text content from a tau2 message object.

    Handles duck-typed objects that look like tau2 messages
    (AssistantMessage, ToolMessage, MultiToolMessage, UserMessage) without
    requiring the actual tau2 imports.
    """
    # MultiToolMessage-like: has tool_messages list
    tool_messages = getattr(message, "tool_messages", None)
    if tool_messages and isinstance(tool_messages, list):
        parts = [extract_message_content(tm) for tm in tool_messages]
        return "\n".join(part for part in parts if part)

    # Simple content-based message
    content = getattr(message, "content", "")
    if content:
        return str(content)

    return ""


def _canonicalize_key(key: str) -> str:
    """Lowercase + strip underscores/dashes for fuzzy key comparison."""
    return key.lower().replace("_", "").replace("-", "")


def _flatten_structured_tool_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Convert array-of-dicts argument styles into the canonical array-of-ids form.

    Some models emit ``items_to_exchange: [{"item_id": ..., "new_item_id": ...}]``
    instead of the expected ``item_ids``/``new_item_ids`` arrays.  We rewrite
    those before alias mapping so the downstream tool receives the right shape.
    """
    if tool_name not in {
        "exchange_delivered_order_items",
        "modify_pending_order_items",
        "return_delivered_order_items",
    }:
        return args

    result = dict(args)
    for key, value in list(result.items()):
        if not isinstance(value, list) or not value:
            continue
        if not all(isinstance(entry, dict) for entry in value):
            continue
        sample = value[0]
        if "item_id" in sample and "new_item_id" in sample:
            result.setdefault("item_ids", [entry["item_id"] for entry in value])
            result.setdefault("new_item_ids", [entry["new_item_id"] for entry in value])
            del result[key]
        elif "item_id" in sample:
            result.setdefault("item_ids", [entry["item_id"] for entry in value])
            del result[key]
    return result


def normalize_tool_args(tool_name: str, args: Any, param_names: list[str], schema: dict[str, Any] | None = None) -> dict[str, Any]:
    """Map LLM argument names/types onto a tau2 tool's canonical schema.

    Some models invent parameter names or encode arrays as strings.  We try
    exact matches first, then fuzzy aliases, then domain-specific corrections,
    so minor schema-adherence errors don't become hard failures.

    If *schema* is provided, values are also coerced to the declared JSON
    Schema types (e.g. plain-text numbers are kept as strings for ``string``
    fields, but converted to ``int``/``float`` for numeric fields).
    """
    if not isinstance(args, dict) or not param_names:
        return args if isinstance(args, dict) else {}

    args = _flatten_structured_tool_args(tool_name, args)

    exact = {name: name for name in param_names}
    canonical_to_name = {
        _canonicalize_key(name): name for name in param_names
    }

    alias_map: dict[str, str] = {
        "zipcode": "zip",
        "zip_code": "zip",
        "postalcode": "zip",
        "postal_code": "zip",
        "firstname": "first_name",
        "first_name": "first_name",
        "lastname": "last_name",
        "last_name": "last_name",
        "phonenumber": "phone_number",
        "phone_number": "phone_number",
        "emailaddress": "email_address",
        "email_address": "email_address",
        "orderid": "order_id",
        "order_id": "order_id",
        "itemid": "item_id",
        "item_id": "item_id",
        "paymentid": "payment_method_id",
        "payment_id": "payment_method_id",
        "paymentmethodid": "payment_method_id",
        "payment_method_id": "payment_method_id",
        "name": "customer_name",
        "user_name": "customer_name",
        "full_name": "customer_name",
        "username": "customer_name",
        "userid": "user_id",
        "user_id": "user_id",
        "newemail": "new_email",
        "new_email": "new_email",
    }
    tool_specific: dict[str, dict[str, str]] = {
        "exchange_delivered_order_items": {
            "items_to_exchange": "item_ids",
            "items_to_modify": "item_ids",
            "exchange_items": "item_ids",
            "exchange_new_items": "new_item_ids",
            "exchange_payment_method_id": "payment_method_id",
        },
        "modify_pending_order_items": {
            "items_to_modify": "item_ids",
            "items_to_exchange": "item_ids",
            "exchange_items": "item_ids",
            "exchange_new_items": "new_item_ids",
        },
        "return_delivered_order_items": {
            "items_to_return": "item_ids",
            "items_to_exchange": "item_ids",
            "return_items": "item_ids",
        },
    }
    if tool_name in tool_specific:
        alias_map.update(tool_specific[tool_name])

    normalized: dict[str, Any] = {}
    for key, value in args.items():
        if key in exact:
            target = key
        elif key.lower() in alias_map:
            target = alias_map[key.lower()]
        elif _canonicalize_key(key) in canonical_to_name:
            target = canonical_to_name[_canonicalize_key(key)]
        else:
            target = key

        # Reverse-direction fallback: the model emitted a known alias variant
        # (e.g. airline 'payment_method_id') but this tool wants a different
        # form ('payment_id'), so exact-match + alias_map both missed.  Map to
        # whichever alias the target tool actually accepts.  Only fires when
        # the resolved target is not a real param name, so tools where the
        # variant IS valid (retail 'payment_method_id') are unaffected.
        if target not in param_names:
            reverse_aliases = {
                "payment_method_id": "payment_id",
                "payment_method": "payment_id",
                "payment_id": "payment_method_id",
                "amount_gb": "gb_amount",
                "gb": "gb_amount",
                "data_amount": "gb_amount",
            }
            candidate = reverse_aliases.get(key.lower())
            if candidate and candidate in param_names:
                target = candidate

        if isinstance(value, str):
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass

        normalized[target] = value

    if schema:
        normalized = _coerce_args_to_schema(normalized, schema)

    return normalized


def _coerce_value_to_schema(value: Any, prop: dict[str, Any]) -> Any:
    """Coerce a single argument value to the JSON Schema type declared in *prop*."""
    prop_type = (prop.get("type") or "string").lower()

    if prop_type == "string":
        return str(value)

    if prop_type == "integer":
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                try:
                    f = float(value)
                    if f.is_integer():
                        return int(f)
                    return f
                except ValueError:
                    return value
        return value

    if prop_type == "number":
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return value
        return value

    if prop_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.lower()
            if lowered in ("true", "1", "yes", "on"):
                return True
            if lowered in ("false", "0", "no", "off"):
                return False
        return bool(value)

    if prop_type == "array":
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return [value]
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        return [value]

    if prop_type == "object":
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return value
            if isinstance(parsed, dict):
                return parsed
            return value
        return value

    return value


def _coerce_args_to_schema(
    args: dict[str, Any],
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Coerce all argument values to the types declared in the tool schema."""
    # ``schema`` may be the raw parameters dict (type+properties) or a wrapper
    # such as {"parameters": {...}} / {"function": {"parameters": {...}}}.
    if isinstance(schema, dict) and "properties" in schema:
        parameters = schema
    else:
        parameters = (
            schema.get("parameters")
            or schema.get("function", {}).get("parameters", {})
            or {}
        )
    properties = parameters.get("properties", {})
    if not properties:
        return args

    coerced: dict[str, Any] = {}
    for key, value in args.items():
        prop = properties.get(key, {})
        coerced[key] = _coerce_value_to_schema(value, prop)
    return coerced


def tool_schema_from_tau2(tool: Any) -> dict[str, Any]:
    """Extract a JSON schema dict from a tau2 Tool object.

    tau2 Tools store parameters as a Pydantic model class (``params``).
    We prefer the tool's own ``openai_schema`` representation and fall
    back to ``params.model_json_schema()`` if needed.
    """
    # If the tool already exposes an OpenAI-compatible schema, use it.
    # tau2's ``openai_schema`` is the full ``{"type":"function","function":{...}}``
    # wrapper, so the parameters live under ``function.parameters`` rather than
    # the top level.  If we only look at the top-level ``parameters`` key we
    # silently get ``{}`` and downstream type-coercion (e.g. telecom
    # ``refuel_data`` ``gb_amount`` string→number) never runs, which surfaces as
    # ``TypeError: '<=' not supported between instances of 'str' and 'int'``.
    openai_schema = getattr(tool, "openai_schema", None)
    if isinstance(openai_schema, dict):
        parameters = openai_schema.get("parameters")
        if not isinstance(parameters, dict) or not parameters:
            parameters = openai_schema.get("function", {}).get("parameters", {})
        if isinstance(parameters, dict) and parameters:
            return parameters

    # Fall back to the Pydantic parameter model.
    params = getattr(tool, "params", None)
    if params is not None:
        to_json_schema = getattr(params, "model_json_schema", None)
        if callable(to_json_schema):
            return cast(dict[str, Any], to_json_schema())
        to_schema = getattr(params, "schema", None)
        if callable(to_schema):
            return cast(dict[str, Any], to_schema())

    return {}


# ---------------------------------------------------------------------------
# Tool-result summarization
# ---------------------------------------------------------------------------


MAX_TOOL_CHARS: int = 2500


def summarize_tool_content(tool_name: str, content: Any) -> str:
    """Compress long tool results so the agent is not overwhelmed.

    Some tau2 tools (e.g. ``get_product_details``) return huge JSON blobs
    with dozens of variants.  We keep the most relevant parts and truncate
    the rest, preserving enough information for the agent to act.
    """
    text = str(content) if content is not None else ""
    if isinstance(content, (dict, list)):
        text = json.dumps(content, ensure_ascii=False)
    if len(text) <= MAX_TOOL_CHARS:
        return text

    # Try JSON-aware summarization first.
    parsed: Any = None
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            pass
    else:
        parsed = content

    if isinstance(parsed, dict):
        summarized = _summarize_parsed_tool(tool_name, parsed)
        if summarized:
            return summarized

    # Generic truncation with a clear marker.
    keep = text[:MAX_TOOL_CHARS]
    # Try to end on a sensible boundary.
    for boundary in ("\n}\n", "\n]\n", ",\n", " "):
        idx = keep.rfind(boundary)
        if idx > MAX_TOOL_CHARS * 0.8:
            keep = keep[: idx + len(boundary)]
            break
    return keep + "\n...[content truncated]"


def _summarize_parsed_tool(tool_name: str, data: dict[str, Any]) -> str | None:
    """Domain-aware summarization for known heavy tools."""
    lowered = tool_name.lower()

    # Product variants: keep only available variants, cap at 5.
    if "product" in lowered and "variants" in data and isinstance(data["variants"], dict):
        selected_variants = {
            k: v
            for k, v in data["variants"].items()
            if isinstance(v, dict) and v.get("available") is True
        }
        capped_variants = dict(list(selected_variants.items())[:5])
        summary = {**data, "variants": capped_variants}
        if len(selected_variants) > 5:
            summary["_variants_note"] = f"{len(selected_variants)} available variants total; showing first 5"
        return json.dumps(summary, ensure_ascii=False, default=str)

    # User details: keep full, but truncate very long order/payment lists.
    if "user" in lowered and ("orders" in data or "payment_methods" in data):
        summary = dict(data)
        for key in ("orders", "payment_methods"):
            if isinstance(summary.get(key), dict) and len(str(summary[key])) > 800:
                items = list(summary[key].items())
                summary[key] = dict(items[:3])
                summary[f"_{key}_note"] = f"{len(items)} total; showing first 3"
        return json.dumps(summary, ensure_ascii=False, default=str)

    # Order details: keep all items but clamp if extremely long.
    if "order" in lowered and "items" in data and isinstance(data["items"], list):
        if len(data["items"]) > 10:
            summary = dict(data)
            summary["items"] = data["items"][:10]
            summary["_items_note"] = f"{len(data['items'])} items total; showing first 10"
            return json.dumps(summary, ensure_ascii=False, default=str)

    return None


# ---------------------------------------------------------------------------
# Tool-error feedback augmentation
# ---------------------------------------------------------------------------


_MISSING_ARGS_RE = re.compile(
    r"missing\s+(?:\d+\s+)?required\s+positional\s+argument[s]?:\s*(.*)",
    re.IGNORECASE,
)
_UNEXPECTED_KWARG_RE = re.compile(
    r"got\s+an\s+unexpected\s+keyword\s+argument\s+'([^']+)'",
    re.IGNORECASE,
)


def augment_tool_error(tool_name: str, content: str, param_names: list[str]) -> str:
    """Append a directive remediation hint to a failed tool call's content.

    tau2 stringifies the Python ``TypeError`` verbatim into the tool result
    content as ``"Error: <exception>"`` (including missing/extra arg names),
    but that signal is buried in a ``role: "tool"`` message indistinguishable
    from a success result.  The model often ignores it and repeats the same
    failing call.  We append an explicit, directive remediation hint naming
    the exact required args / correct param names so the model corrects on the
    next turn instead of looping.

    Returns *content* unchanged when it does not look like an error.
    """
    if not isinstance(content, str) or not content:
        return content

    stripped = content.lstrip()
    if not stripped.lower().startswith("error:"):
        return content

    params_str = ", ".join(param_names) if param_names else "(unknown)"
    directive: str

    missing_match = _MISSING_ARGS_RE.search(stripped)
    kwarg_match = _UNEXPECTED_KWARG_RE.search(stripped)

    if missing_match:
        missing = missing_match.group(1).strip().rstrip(".")
        directive = (
            f"You must call '{tool_name}' again. Required arguments are: "
            f"{params_str}. You are missing: {missing}. Gather the missing "
            f"values from earlier tool results or the user, then call "
            f"'{tool_name}' with ALL required arguments."
        )
    elif kwarg_match:
        bad = kwarg_match.group(1)
        directive = (
            f"'{bad}' is not a valid argument for '{tool_name}'. Valid "
            f"arguments are: {params_str}. Call '{tool_name}' again using "
            f"only valid argument names."
        )
    else:
        directive = (
            f"The previous call to '{tool_name}' failed. Check the error "
            f"above, then call '{tool_name}' again with the correct "
            f"arguments: {params_str}."
        )

    if directive in content:
        return content
    return f"{content}\n\n{directive}"
