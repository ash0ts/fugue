"""Tiny dependency-free, read-only MCP server for the standalone template."""

from __future__ import annotations

import json
import sys
from typing import Any

CATALOG = {
    "legacy": {
        "refund_window_days": {"value": 14, "unit": "days"},
        "cache_ttl_seconds": {"value": 120, "unit": "seconds"},
    },
    "current": {
        "refund_window_days": {"value": 30, "unit": "days"},
        "cache_ttl_seconds": {"value": 45, "unit": "seconds"},
    },
}


def _revision() -> str:
    try:
        index = sys.argv.index("--revision")
        revision = sys.argv[index + 1]
    except (ValueError, IndexError):
        revision = "legacy"
    return revision if revision in CATALOG else "legacy"


def _result(request: dict[str, Any]) -> dict[str, Any] | None:
    request_id = request.get("id")
    method = request.get("method")
    if request_id is None:
        return None
    if method == "initialize":
        result: dict[str, Any] = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "offline-catalog", "version": "1"},
        }
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {
            "tools": [
                {
                    "name": "lookup_setting",
                    "description": "Read one named setting from the locked catalogue.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"setting": {"type": "string"}},
                        "required": ["setting"],
                        "additionalProperties": False,
                    },
                }
            ]
        }
    elif method == "tools/call":
        params = request.get("params") or {}
        setting = str((params.get("arguments") or {}).get("setting") or "")
        revision = _revision()
        item = CATALOG[revision].get(setting)
        if item is None:
            payload = {
                "setting": setting,
                "available": False,
                "revision": revision,
                "source": "offline-catalog",
            }
        else:
            payload = {
                "setting": setting,
                **item,
                "revision": revision,
                "source": "offline-catalog",
            }
        result = {
            "content": [
                {"type": "text", "text": json.dumps(payload, sort_keys=True)}
            ],
            "structuredContent": payload,
            "isError": item is None,
        }
    else:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": "method not found"},
        }
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def main() -> None:
    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = _result(request)
        except (TypeError, ValueError) as exc:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": str(exc)},
            }
        if response is not None:
            sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
