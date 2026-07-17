"""Minimal Python SDK for the SwayBot HTTP API."""

import json
import urllib.parse
import urllib.request
from typing import Any


class SwayBotClient:
    """Sync client for a SwayBot API server."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        self.base_url = base_url.rstrip("/")

    def _request(
        self,
        method: str,
        path: str,
        data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        headers = {"Accept": "application/json"}
        body = None
        if data is not None:
            body = json.dumps(data).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(
            url, data=body, headers=headers, method=method
        )
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def chat_completions(
        self, messages: list[dict[str, str]], model: str = "swaybot"
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/chat/completions",
            data={"model": model, "messages": messages},
        )

    def post_message(self, session_id: str, content: str, role: str = "user") -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v1/sessions/{urllib.parse.quote(session_id, safe='')}/messages",
            data={"role": role, "content": content},
        )

    def get_history(
        self, session_id: str, limit: int | None = None
    ) -> dict[str, Any]:
        params = {}
        if limit is not None:
            params["limit"] = limit
        return self._request(
            "GET",
            f"/v1/sessions/{urllib.parse.quote(session_id, safe='')}/history",
            params=params or None,
        )
