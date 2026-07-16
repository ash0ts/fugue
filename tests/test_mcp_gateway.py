from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from fugue.mcp_gateway import _UpstreamStderr


def test_gateway_collects_vector_telemetry_across_stderr_chunks(
    capsys,
) -> None:
    stderr = _UpstreamStderr()
    try:
        stderr.write("GitNexus starting\nFUGUE_GITNEXUS_VEC")
        stderr.write(
            'TOR {"vector_search_attempted":true,"semantic_result_count":3}\n'
        )
        stderr.write(
            'FUGUE_GITNEXUS_VECTOR {"vector_search_succeeded":true,'
            '"bm25_result_count":2,"model_digest":"sha256:model"}\n'
        )

        assert stderr.vector() == {
            "vector_search_attempted": True,
            "vector_search_succeeded": True,
            "semantic_result_count": 3,
            "bm25_result_count": 2,
            "model_digest": "sha256:model",
        }
        assert "GitNexus starting" in capsys.readouterr().err

        stderr.reset_vector()
        assert stderr.vector() == {}
    finally:
        stderr.close()


def test_gateway_forwards_a_pinned_stdio_tool(tmp_path: Path) -> None:
    upstream = tmp_path / "echo_server.py"
    upstream.write_text(
        """from mcp.server.fastmcp import FastMCP

server = FastMCP("fixture")

@server.tool()
def echo(value: str) -> str:
    return value

server.run("stdio")
"""
    )
    port = _unused_port()
    event_log = tmp_path / "gateway-events.jsonl"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "fugue.mcp_gateway",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--",
            sys.executable,
            upstream.as_posix(),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env={
            **os.environ,
            "FUGUE_RUN_ID": "run-a",
            "FUGUE_CANDIDATE_ID": "candidate-a",
            "FUGUE_GATEWAY_EVENT_LOG": event_log.as_posix(),
        },
    )
    try:
        _wait_for_port(port, process)
        tools, result, meta = asyncio.run(
            asyncio.wait_for(_call_echo(port), timeout=10)
        )
        assert tools == ["echo"]
        assert result == "hello"
        assert meta["fugue_run_id"] == "run-a"
        assert meta["fugue_candidate_id"] == "candidate-a"
        assert meta["fugue_gateway_call_id"]
    finally:
        subprocess.run(
            ["pkill", "-TERM", "-P", str(process.pid)],
            check=False,
            capture_output=True,
        )
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate(timeout=10)
    assert process.returncode in {0, -signal.SIGTERM}
    assert '"event": "gateway_ready"' in stdout
    assert '"event": "tool_end"' in stdout
    assert '"fugue_run_id": "run-a"' in stdout
    assert "ERROR" not in stderr
    persisted = [json.loads(line) for line in event_log.read_text().splitlines()]
    assert [event["event"] for event in persisted] == [
        "gateway_ready",
        "tool_start",
        "tool_end",
    ]
    assert persisted[-1]["fugue_candidate_id"] == "candidate-a"


async def _call_echo(port: int) -> tuple[list[str], str, dict[str, str]]:
    async with streamable_http_client(f"http://127.0.0.1:{port}/mcp") as streams:
        async with ClientSession(*streams[:2]) as session:
            await session.initialize()
            tools = await session.list_tools()
            result = await session.call_tool("echo", {"value": "hello"})
    return [tool.name for tool in tools.tools], result.content[0].text, result.meta or {}


def _unused_port() -> int:
    with socket.socket() as value:
        value.bind(("127.0.0.1", 0))
        return int(value.getsockname()[1])


def _wait_for_port(port: int, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(f"gateway exited early\n{stdout}\n{stderr}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise AssertionError("gateway did not listen within 15 seconds")
