from __future__ import annotations

import json
import os
import subprocess
import sys
from io import BytesIO

from fugue.mcp_proxy import _Recorder, _relay_requests, _sanitize


def test_mcp_telemetry_redacts_and_correlates_requests(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    recorder = _Recorder(name="search", path=path)
    recorder.request(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "search",
                "arguments": {"query": "auth flow", "api_key": "secret-value"},
            },
        },
        120,
    )
    recorder.response(
        {"jsonrpc": "2.0", "id": 7, "result": {"content": "large local value"}},
        240,
    )

    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [item["event"] for item in events] == [
        "mcp_tool_request",
        "mcp_tool_response",
    ]
    assert events[0]["arguments"] == {
        "api_key": "[redacted]",
        "query": "auth flow",
    }
    assert events[1]["response_bytes"] == 240
    assert events[1]["latency_ms"] >= 0
    assert [item["layer"] for item in events] == ["proxy", "upstream"]
    assert events[0]["elapsed_ms"] >= 0
    assert "large local value" not in path.read_text()


def test_mcp_payload_sanitizer_caps_text_and_nested_lists() -> None:
    value = _sanitize(
        {"authorization": "bearer", "max_tokens": 2_000, "text": "x" * 2_000}
    )
    assert value["authorization"] == "[redacted]"
    assert value["max_tokens"] == 2_000
    assert len(value["text"]) == 1_000


def test_mcp_proxy_denies_tools_outside_the_allowlist(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    recorder = _Recorder(name="search", path=path, allowed_tools={"search"})
    request = BytesIO(
        b'{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"delete"}}\n'
    )
    upstream = BytesIO()
    client = BytesIO()

    _relay_requests(request, upstream, client, recorder)

    assert upstream.getvalue() == b""
    assert b"Tool denied by Fugue policy" in client.getvalue()
    [event] = [json.loads(line) for line in path.read_text().splitlines()]
    assert event["event"] == "mcp_tool_denied"
    assert event["tool"] == "delete"


def test_mcp_proxy_process_blocks_denied_call_and_relays_allowed_call(
    tmp_path,
) -> None:
    server = tmp_path / "server.py"
    server.write_text(
        """
import json
import sys

request = json.loads(sys.stdin.readline())
print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": {"ok": True}}), flush=True)
"""
    )
    events = tmp_path / "events.jsonl"
    payload = "\n".join(
        [
            '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"delete","arguments":{}}}',
            '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"search","arguments":{"query":"safe"}}}',
            "",
        ]
    )
    env = dict(os.environ)
    env["FUGUE_CONTEXT_EVENTS_PATH"] = events.as_posix()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fugue.mcp_proxy",
            "--name",
            "fake",
            "--allow-tool",
            "search",
            "--",
            sys.executable,
            server.as_posix(),
        ],
        input=payload,
        capture_output=True,
        text=True,
        env=env,
        timeout=5,
    )

    assert result.returncode == 0
    responses = [json.loads(line) for line in result.stdout.splitlines()]
    assert {item["id"] for item in responses} == {1, 2}
    denied = next(item for item in responses if item["id"] == 1)
    allowed = next(item for item in responses if item["id"] == 2)
    assert denied["error"]["code"] == -32601
    assert allowed["result"] == {"ok": True}
    recorded = [json.loads(line) for line in events.read_text().splitlines()]
    assert [item["event"] for item in recorded] == [
        "mcp_tool_denied",
        "mcp_tool_request",
        "mcp_tool_response",
    ]
