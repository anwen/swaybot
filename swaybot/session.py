"""Persistent per-session JSONL history manager."""

import asyncio
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .bus import InboundMessage, OutboundMessage
from .storage import StorageBackend, default_session_backend


class SessionError(ValueError):
    """Raised for invalid session identifiers or operations."""


class SessionManager:
    """Store chat history as one JSONL stream per session.

    The backend is pluggable; by default a JSON-lines file is used under
    ``~/.swaybot/sessions/{safe_session_id}.jsonl``.
    """

    def __init__(
        self,
        base_dir: Path | str | None = None,
        backend: StorageBackend | None = None,
    ) -> None:
        if backend is not None:
            self.backend = backend
        else:
            self.backend = default_session_backend(base_dir)
        self.base_dir = getattr(self.backend, "base_dir", None)

    @staticmethod
    def _safe_id(session_id: str) -> str:
        """Sanitize a session id into a safe key component."""
        if not session_id:
            raise SessionError("session_id must not be empty")
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)
        if safe != session_id:
            raise SessionError(
                f"invalid session_id: {session_id!r}"
            )
        return safe

    def _key(self, session_id: str) -> str:
        return self._safe_id(session_id)

    def create(self, session_id: str) -> Path | None:
        """Create an empty session stream if it does not exist."""
        key = self._key(session_id)
        self.backend.create(key)
        return getattr(self.backend, "_path", lambda _k: None)(key)

    def exists(self, session_id: str) -> bool:
        return self.backend.exists(self._key(session_id))

    def delete(self, session_id: str) -> None:
        self.backend.delete(self._key(session_id))

    def list_sessions(self) -> list[str]:
        """Return all session ids stored in this backend."""
        return self.backend.list_keys(prefix="")

    def _normalize(self, message: Any) -> dict[str, Any]:
        if isinstance(message, (InboundMessage, OutboundMessage)):
            data = asdict(message)
        elif isinstance(message, dict):
            data = dict(message)
        else:
            raise SessionError(
                f"unsupported message type: {type(message).__name__}"
            )
        data.setdefault(
            "timestamp", datetime.now(timezone.utc).isoformat()
        )
        return data

    def append(self, session_id: str, message: Any) -> None:
        """Append a message to the session's stream."""
        self.backend.append(self._key(session_id), self._normalize(message))

    def load(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        """Load messages for ``session_id`` in chronological order."""
        return self.backend.load_stream(self._key(session_id), limit=limit)

    def clear(self, session_id: str) -> None:
        """Truncate the session history without deleting the stream."""
        self.backend.save(self._key(session_id), [])

    async def acreate(self, session_id: str) -> Path | None:
        return await asyncio.to_thread(self.create, session_id)

    async def aexists(self, session_id: str) -> bool:
        return await asyncio.to_thread(self.exists, session_id)

    async def adelete(self, session_id: str) -> None:
        return await asyncio.to_thread(self.delete, session_id)

    async def alist_sessions(self) -> list[str]:
        return await asyncio.to_thread(self.list_sessions)

    async def aappend(self, session_id: str, message: Any) -> None:
        return await asyncio.to_thread(self.append, session_id, message)

    async def aload(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.load, session_id, limit=limit)

    async def aclear(self, session_id: str) -> None:
        return await asyncio.to_thread(self.clear, session_id)
