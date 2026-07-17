"""Persistent per-session JSONL history manager."""

import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .bus import InboundMessage, OutboundMessage


class SessionError(ValueError):
    """Raised for invalid session identifiers or operations."""


class SessionManager:
    """Store chat history as one JSONL file per session.

    Each line is a JSON object with at least ``role``, ``content`` and
    ``timestamp``. The directory layout is ``{base_dir}/{safe_session_id}.jsonl``.
    """

    def __init__(self, base_dir: Path | str | None = None) -> None:
        if base_dir is None:
            base_dir = Path.home() / ".swaybot" / "sessions"
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_id(session_id: str) -> str:
        """Sanitize a session id into a safe filename component."""
        if not session_id:
            raise SessionError("session_id must not be empty")
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)
        if safe != session_id:
            raise SessionError(
                f"invalid session_id: {session_id!r}"
            )
        return safe

    def _path(self, session_id: str) -> Path:
        safe = self._safe_id(session_id)
        path = (self.base_dir / f"{safe}.jsonl").resolve()
        # Ensure the resolved path is still inside base_dir.
        if not str(path).startswith(str(self.base_dir.resolve())):
            raise SessionError(f"session path escapes base_dir: {session_id}")
        return path

    def create(self, session_id: str) -> Path:
        """Create an empty session file if it does not exist."""
        path = self._path(session_id)
        path.touch(exist_ok=True)
        return path

    def exists(self, session_id: str) -> bool:
        return self._path(session_id).exists()

    def delete(self, session_id: str) -> None:
        self._path(session_id).unlink(missing_ok=True)

    def list_sessions(self) -> list[str]:
        """Return all session ids stored under ``base_dir``."""
        sessions: list[str] = []
        for path in sorted(self.base_dir.glob("*.jsonl")):
            sessions.append(path.stem)
        return sessions

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
        """Append a message to the session's JSONL file."""
        path = self._path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self._normalize(message)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(data, ensure_ascii=False) + "\n")

    def load(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        """Load messages for ``session_id`` in chronological order."""
        path = self._path(session_id)
        if not path.exists():
            return []
        messages: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                messages.append(json.loads(line))
        if limit is not None:
            messages = messages[-limit:]
        return messages

    def clear(self, session_id: str) -> None:
        """Truncate the session history without deleting the file."""
        path = self._path(session_id)
        if path.exists():
            path.write_text("", encoding="utf-8")
