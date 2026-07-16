from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

_SERVER_NAME = re.compile(r"[A-Za-z0-9_-]+")


def render_codex_mcp_toml(servers: Iterable[Any]) -> str:
    sections: list[str] = []
    names: set[str] = set()
    for server in servers:
        name = str(getattr(server, "name", "") or "")
        if not _SERVER_NAME.fullmatch(name):
            raise ValueError(f"invalid Codex MCP server name: {name!r}")
        if name in names:
            raise ValueError(f"duplicate Codex MCP server name: {name}")
        names.add(name)
        transport = str(getattr(server, "transport", "") or "")
        lines = [f"[mcp_servers.{name}]"]
        if transport == "stdio":
            command = str(getattr(server, "command", "") or "")
            if not command:
                raise ValueError(f"Codex MCP server {name} has no command")
            args = [str(value) for value in (getattr(server, "args", None) or [])]
            lines.append(f"command = {json.dumps(command)}")
            lines.append("args = " + json.dumps(args, separators=(",", ":")))
        elif transport in {"sse", "streamable-http"}:
            url = str(getattr(server, "url", "") or "")
            if not url:
                raise ValueError(f"Codex MCP server {name} has no URL")
            lines.append(f"url = {json.dumps(url)}")
        else:
            raise ValueError(
                f"Codex MCP server {name} has unsupported transport {transport!r}"
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections) + ("\n" if sections else "")
