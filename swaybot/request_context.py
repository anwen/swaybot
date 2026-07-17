"""Request-scoped context propagation via a context variable."""

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator


@dataclass
class RequestContext:
    """Request-scoped identity and trace metadata.

    Carried from the API layer through Agent -> Tools -> Models for audit,
    multi-user safety, and structured logging.
    """

    session_id: str = ""
    request_id: str = ""
    principal: str = "anonymous"
    permission_level: str = "medium"
    metadata: dict[str, Any] = field(default_factory=dict)


_current_context: contextvars.ContextVar[RequestContext | None] = contextvars.ContextVar(
    "swaybot_request_context", default=None
)


def current_context() -> RequestContext:
    """Return the active request context, or a default context if none is set."""
    ctx = _current_context.get()
    if ctx is None:
        return RequestContext()
    return ctx


@contextmanager
def set_context(ctx: RequestContext) -> Generator[RequestContext, None, None]:
    """Set ``ctx`` for the duration of the with-block."""
    token = _current_context.set(ctx)
    try:
        yield ctx
    finally:
        _current_context.reset(token)
