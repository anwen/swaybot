"""Minimal stdio MCP client for dynamic tool loading."""

import json
import queue
import subprocess
import threading
from pathlib import Path
from typing import Any


class McpClient:
    """A tiny stdio MCP client that lists and calls tools."""

    def __init__(self, command: list[str], cwd: str | Path | None = None) -> None:
        self.command = command
        self.cwd = Path(cwd) if cwd else None
        self._process: subprocess.Popen | None = None
        self._lines: queue.Queue[str] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._request_id = 0

    def start(self) -> "McpClient":
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=self.cwd,
        )
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        self._call(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "swaybot", "version": "0.1.0"},
            },
        )
        self._notify("notifications/initialized")
        return self

    def close(self) -> None:
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:  # pragma: no cover
                self._process.kill()
        self._process = None

    def list_tools(self) -> list[dict]:
        response = self._call("tools/list")
        result = response.get("result", {})
        return result.get("tools", [])

    def call_tool(self, name: str, arguments: dict) -> str:
        response = self._call("tools/call", {"name": name, "arguments": arguments})
        result = response.get("result", {})
        parts = []
        for item in result.get("content", []):
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts) if parts else ""

    def _read_stdout(self) -> None:
        if self._process is None or self._process.stdout is None:
            return
        try:
            for line in self._process.stdout:
                self._lines.put(line.strip())
        except Exception:  # pragma: no cover
            pass

    def _send(self, message: dict) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("MCP client is not started")
        self._process.stdin.write(json.dumps(message) + "\n")
        self._process.stdin.flush()

    def _notify(self, method: str, params: dict | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def _call(self, method: str, params: dict | None = None) -> dict:
        self._request_id += 1
        request_id = self._request_id
        self._send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params or {},
            }
        )
        while True:
            try:
                line = self._lines.get(timeout=30)
            except queue.Empty as exc:
                raise RuntimeError(f"MCP request {method} timed out") from exc
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if message.get("id") == request_id:
                if "error" in message:
                    raise RuntimeError(message["error"])
                return message


def register_mcp_tools(registry: Any, command: list[str]) -> McpClient:
    """Start an MCP server and register its tools into a ToolRegistry."""
    from .tools import Tool

    client = McpClient(command).start()
    for tool in client.list_tools():
        schema = tool.get("inputSchema", {"type": "object"})
        name = tool["name"]
        description = tool.get("description", "")

        def _make_fn(tool_name: str, mcp_client: McpClient):
            def fn(**kwargs: Any) -> str:
                return mcp_client.call_tool(tool_name, kwargs)

            return fn

        registry.register(
            name,
            Tool(
                name=name,
                description=description,
                inputs=schema,
                output_type="string",
                fn=_make_fn(name, client),
            ),
        )
    return client
