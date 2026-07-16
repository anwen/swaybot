#!/usr/bin/env python3
"""A tiny fake MCP server for tests."""

import json
import sys


def send(message: dict) -> None:
    print(json.dumps(message), flush=True)


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = request.get("method")
        req_id = request.get("id")
        if method == "initialize":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "serverInfo": {"name": "fake-mcp", "version": "0.1"},
                        "capabilities": {},
                    },
                }
            )
        elif method == "notifications/initialized":
            pass
        elif method == "tools/list":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "tools": [
                            {
                                "name": "greet",
                                "description": "Greet someone by name.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"}
                                    },
                                    "required": ["name"],
                                },
                            }
                        ]
                    },
                }
            )
        elif method == "tools/call":
            params = request.get("params", {})
            name = params.get("name")
            arguments = params.get("arguments", {})
            if name == "greet":
                text = f"Hello, {arguments.get('name', 'stranger')}!"
            else:
                text = f"Unknown tool: {name}"
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": text}]},
                }
            )


if __name__ == "__main__":
    main()
