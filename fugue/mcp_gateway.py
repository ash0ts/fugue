from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Expose one pinned stdio MCP server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("upstream", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    upstream = list(args.upstream)
    if upstream[:1] == ["--"]:
        upstream = upstream[1:]
    if not upstream:
        parser.error("an upstream MCP command is required after --")

    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.server.fastmcp import FastMCP
    from mcp.server.lowlevel.helper_types import ReadResourceContents

    @asynccontextmanager
    async def lifespan(_: Any) -> AsyncIterator[ClientSession]:
        parameters = StdioServerParameters(
            command=upstream[0],
            args=upstream[1:],
            env=dict(os.environ),
        )
        async with stdio_client(parameters) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                _event("gateway_ready", upstream=upstream[0])
                yield session

    gateway = FastMCP(
        "fugue-mcp-gateway",
        host=args.host,
        port=args.port,
        streamable_http_path="/mcp",
        stateless_http=False,
        json_response=True,
        lifespan=lifespan,
    )
    # FastMCP 1.28.1 has no public forwarding hook. Keep this private access
    # pinned with the SDK version instead of pretending proxy tools are static.
    server = gateway._mcp_server

    @server.list_tools()
    async def list_tools():
        return (await _upstream(server).list_tools()).tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]):
        call_id = uuid.uuid4().hex
        started = time.monotonic()
        _event("tool_start", tool=name, gateway_call_id=call_id)
        try:
            result = await _upstream(server).call_tool(name, arguments)
        except asyncio.CancelledError:
            _event("tool_cancelled", tool=name, gateway_call_id=call_id)
            raise
        except Exception as exc:
            _event(
                "tool_failed",
                tool=name,
                gateway_call_id=call_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        _event(
            "tool_end",
            tool=name,
            gateway_call_id=call_id,
            duration_ms=round((time.monotonic() - started) * 1_000, 3),
            is_error=bool(result.isError),
        )
        return result.model_copy(
            update={
                "meta": {
                    **(result.meta or {}),
                    **_correlation(),
                    "fugue_gateway_call_id": call_id,
                }
            }
        )

    @server.list_resources()
    async def list_resources():
        return (await _upstream(server).list_resources()).resources

    @server.read_resource()
    async def read_resource(uri: Any):
        result = await _upstream(server).read_resource(uri)
        contents: list[ReadResourceContents] = []
        for item in result.contents:
            if hasattr(item, "text"):
                content: str | bytes = item.text
            else:
                content = base64.b64decode(item.blob)
            contents.append(
                ReadResourceContents(
                    content=content,
                    mime_type=item.mimeType,
                    meta=item.meta,
                )
            )
        return contents

    @server.list_resource_templates()
    async def list_resource_templates():
        return (await _upstream(server).list_resource_templates()).resourceTemplates

    @server.list_prompts()
    async def list_prompts():
        return (await _upstream(server).list_prompts()).prompts

    @server.get_prompt()
    async def get_prompt(name: str, arguments: dict[str, str] | None):
        return await _upstream(server).get_prompt(name, arguments)

    gateway.run("streamable-http")
    return 0


def _event(event: str, **values: Any) -> None:
    correlation = _correlation()
    print(
        json.dumps(
            {
                "event": event,
                **correlation,
                **values,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _correlation() -> dict[str, str]:
    values = {
        "fugue_run_id": os.environ.get("FUGUE_RUN_ID"),
        "fugue_candidate_id": os.environ.get("FUGUE_CANDIDATE_ID"),
        "fugue_comparison_example_id": os.environ.get("FUGUE_COMPARISON_EXAMPLE_ID"),
        "fugue_trial_index": os.environ.get("FUGUE_TRIAL_INDEX"),
        "fugue_execution_fingerprint": os.environ.get("FUGUE_EXECUTION_FINGERPRINT"),
        "fugue_context_system_id": os.environ.get("FUGUE_CONTEXT_SYSTEM_ID"),
    }
    return {key: value for key, value in values.items() if value}


def _upstream(server: Any) -> Any:
    return server.request_context.lifespan_context


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
