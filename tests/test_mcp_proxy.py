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
    assert events[1]["terminal_status"] == "succeeded"
    assert events[1]["successful"] is True
    assert events[1]["error"] is False
    assert [item["layer"] for item in events] == ["proxy", "upstream"]
    assert events[0]["elapsed_ms"] >= 0
    assert "large local value" not in path.read_text()


def test_mcp_response_telemetry_exports_only_safe_terminal_metadata(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    recorder = _Recorder(name="wandb", path=path)
    recorder.request(
        {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {"name": "query_wandb_tool", "arguments": {}},
        },
        100,
    )
    recorder.response(
        {
            "jsonrpc": "2.0",
            "id": 9,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "result": {
                                    "items": [
                                        {"id": "private-run-a"},
                                        {"id": "private-run-b"},
                                    ],
                                    "returned_count": 2,
                                    "total_count": 2,
                                    "has_more": False,
                                    "project_exhaustive": True,
                                    "truncated": False,
                                    "truncation": {"applied": False},
                                }
                            }
                        ),
                    }
                ]
            },
        },
        500,
    )
    recorder.request(
        {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {"name": "get_run_history_tool", "arguments": {}},
        },
        100,
    )
    recorder.response(
        {
            "jsonrpc": "2.0",
            "id": 10,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "result": json.dumps(
                                    {
                                        "error": (
                                            "Failed to fetch private-run-a with "
                                            "credential abc123"
                                        )
                                    }
                                )
                            }
                        ),
                    }
                ]
            },
        },
        500,
    )

    responses = [
        item
        for item in (
            json.loads(line) for line in path.read_text().splitlines()
        )
        if item["event"] == "mcp_tool_response"
    ]
    assert responses[0] == {
        **{
            key: responses[0][key]
            for key in (
                "elapsed_ms",
                "latency_ms",
                "request_id",
                "response_bytes",
            )
        },
        "coverage_status": "project-exhaustive",
        "error": False,
        "event": "mcp_tool_response",
        "has_more": False,
        "layer": "upstream",
        "project_exhaustive": True,
        "returned_count": 2,
        "server": "wandb",
        "successful": True,
        "terminal_status": "succeeded",
        "tool": "query_wandb_tool",
        "total_count": 2,
        "truncation_applied": False,
    }
    assert responses[1]["terminal_status"] == "structured_error"
    assert responses[1]["successful"] is False
    assert responses[1]["structured_error_code"] == "tool_error"
    serialized = path.read_text()
    assert "private-run-a" not in serialized
    assert "abc123" not in serialized


def test_mcp_response_telemetry_sums_safe_evaluation_prediction_counts(
    tmp_path,
) -> None:
    path = tmp_path / "events.jsonl"
    recorder = _Recorder(name="wandb", path=path)
    recorder.request(
        {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "summarize_evaluation_tool",
                "arguments": {
                    "entity_name": "wandb",
                    "project_name": "source",
                    "max_evals": 25,
                },
            },
        },
        100,
    )
    recorder.response(
        {
            "jsonrpc": "2.0",
            "id": 11,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "evaluations": [
                                    {
                                        "eval_id": "private-root-a",
                                        "total_predictions": 8,
                                    },
                                    {
                                        "eval_id": "private-root-b",
                                        "total_predictions": 8,
                                    },
                                ]
                            }
                        ),
                    }
                ]
            },
        },
        500,
    )

    response = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if json.loads(line)["event"] == "mcp_tool_response"
    ][0]
    assert response["prediction_count"] == 16
    assert "private-root-a" not in path.read_text()
    assert "private-root-b" not in path.read_text()


def test_mcp_response_telemetry_normalizes_nested_graphql_and_trace_counts(
    tmp_path,
) -> None:
    path = tmp_path / "events.jsonl"
    recorder = _Recorder(name="wandb", path=path)
    for request_id, tool in (
        (11, "query_wandb_tool"),
        (12, "query_weave_traces_tool"),
    ):
        recorder.request(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {"name": tool, "arguments": {}},
            },
            100,
        )
    recorder.response(
        {
            "jsonrpc": "2.0",
            "id": 11,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "data": {
                                    "project": {
                                        "runCount": 6,
                                        "runs": {
                                            "totalCount": 6,
                                            "edges": [
                                                {"node": {"id": f"private-{index}"}}
                                                for index in range(6)
                                            ],
                                            "pageInfo": {"hasNextPage": False},
                                        },
                                    }
                                }
                            }
                        ),
                    }
                ]
            },
        },
        500,
    )
    recorder.response(
        {
            "jsonrpc": "2.0",
            "id": 12,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "traces": [
                                    {"id": f"private-call-{index}"}
                                    for index in range(9)
                                ],
                                "metadata": {
                                    "total_matching_count": 9,
                                    "truncation_applied": False,
                                },
                            }
                        ),
                    }
                ]
            },
        },
        500,
    )

    responses = [
        item
        for item in (
            json.loads(line) for line in path.read_text().splitlines()
        )
        if item["event"] == "mcp_tool_response"
    ]
    assert responses[0]["returned_count"] == 6
    assert responses[0]["total_count"] == 6
    assert responses[0]["has_more"] is False
    assert responses[0]["project_exhaustive"] is True
    assert responses[0]["coverage_status"] == "project-exhaustive"
    assert responses[1]["returned_count"] == 9
    assert responses[1]["total_count"] == 9
    assert responses[1]["truncation_applied"] is False
    serialized = path.read_text()
    assert "private-0" not in serialized
    assert "private-call-0" not in serialized


def test_mcp_response_telemetry_uses_closed_error_taxonomy_and_envelopes(
    tmp_path,
) -> None:
    path = tmp_path / "events.jsonl"
    recorder = _Recorder(name="wandb", path=path)
    responses = (
        {
            "jsonrpc": "2.0",
            "id": 20,
            "error": {"code": "sk-ant-api03-PRIVATE-TOKEN"},
        },
        {
            "jsonrpc": "2.0",
            "id": 21,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {"error": {"code": "customer-private-label"}}
                        ),
                    }
                ]
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 22,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "traces": [
                                    {
                                        "id": "call-a",
                                        "error": {"code": "model_failed"},
                                    }
                                ],
                                "total_matching_count": 1,
                            }
                        ),
                    }
                ]
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 23,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {"error": {"code": "server_busy"}}
                        ),
                    }
                ]
            },
        },
    )
    for response in responses:
        recorder.request(
            {
                "jsonrpc": "2.0",
                "id": response["id"],
                "method": "tools/call",
                "params": {
                    "name": "query_weave_traces_tool",
                    "arguments": {},
                },
            },
            100,
        )
        recorder.response(response, 500)

    recorded = [
        item
        for item in (
            json.loads(line) for line in path.read_text().splitlines()
        )
        if item["event"] == "mcp_tool_response"
    ]
    assert [item["terminal_status"] for item in recorded] == [
        "protocol_error",
        "structured_error",
        "succeeded",
        "structured_error",
    ]
    assert [item.get("structured_error_code") for item in recorded] == [
        "jsonrpc_error",
        "tool_error",
        None,
        "server_busy",
    ]
    assert recorded[2]["returned_count"] == 1
    serialized = path.read_text()
    assert "PRIVATE-TOKEN" not in serialized
    assert "customer-private-label" not in serialized
    assert "model_failed" not in serialized
    assert "call-a" not in serialized


def test_mcp_payload_sanitizer_caps_text_and_nested_lists() -> None:
    value = _sanitize(
        {"authorization": "bearer", "max_tokens": 2_000, "text": "x" * 2_000}
    )
    assert value["authorization"] == "[redacted]"
    assert value["max_tokens"] == 2_000
    assert len(value["text"]) == 1_000


def test_mcp_graphql_request_records_shape_without_query_or_variables(
    tmp_path,
) -> None:
    path = tmp_path / "events.jsonl"
    recorder = _Recorder(name="wandb", path=path)
    recorder.request(
        {
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {
                "name": "query_wandb_tool",
                "arguments": {
                    "query": """
query PrivateInventory($entity: String!, $project: String!) {
  project(name: $project, entityName: $entity) {
    runCount
    runs(first: 50) {
      edges { node { privateAlias: name config summaryMetrics } }
    }
  }
}
""",
                    "variables": {
                        "entity": "wandb",
                        "project": "release-project",
                        "private_filter": "customer-secret",
                    },
                    "max_items": 50,
                },
            },
        },
        500,
    )

    [event] = [json.loads(line) for line in path.read_text().splitlines()]
    arguments = event["arguments"]
    assert "query" not in arguments
    assert "variables" not in arguments
    assert arguments["entity_name"] == "wandb"
    assert arguments["project_name"] == "release-project"
    assert arguments["max_items"] == 50
    assert arguments["raw_graphql_shape"] == {
        "broad_projection": True,
        "graphql_operation_type": "query",
        "graphql_limit_resolved": True,
        "graphql_projection_resolved": True,
        "graphql_requested_limit": 50,
        "graphql_scope_resolved": True,
        "projected_fields": ["config.*", "id", "summary.*"],
        "raw_graphql": True,
        "resource": "runs",
        "response_modes": ["count", "items"],
    }
    serialized = path.read_text()
    assert "PrivateInventory" not in serialized
    assert "privateAlias" not in serialized
    assert "customer-secret" not in serialized


def test_mcp_graphql_request_resolves_custom_scope_and_limit_variables(
    tmp_path,
) -> None:
    path = tmp_path / "events.jsonl"
    recorder = _Recorder(name="wandb", path=path)
    recorder.request(
        {
            "jsonrpc": "2.0",
            "id": 24,
            "method": "tools/call",
            "params": {
                "name": "query_wandb_tool",
                "arguments": {
                    "query": """
query Inventory($account: String!, $workspace: String!, $pageSize: Int!) {
  project(entityName: $account, name: $workspace) {
    runs(first: $pageSize) {
      edges { node { id displayName state } }
    }
  }
}
""",
                    "variables": {
                        "account": "wandb",
                        "workspace": "release-project",
                        "pageSize": 6,
                    },
                },
            },
        },
        500,
    )

    [event] = [json.loads(line) for line in path.read_text().splitlines()]
    arguments = event["arguments"]
    assert arguments["entity_name"] == "wandb"
    assert arguments["project_name"] == "release-project"
    assert "project_ref" not in arguments
    assert arguments["raw_graphql_shape"] == {
        "broad_projection": False,
        "graphql_limit_resolved": True,
        "graphql_operation_type": "query",
        "graphql_projection_resolved": True,
        "graphql_requested_limit": 6,
        "graphql_scope_resolved": True,
        "projected_fields": ["display_name", "id", "state"],
        "raw_graphql": True,
        "resource": "runs",
        "response_modes": ["items"],
    }


def test_mcp_graphql_request_fails_closed_for_fragments_and_unknown_scope(
    tmp_path,
) -> None:
    path = tmp_path / "events.jsonl"
    recorder = _Recorder(name="wandb", path=path)
    recorder.request(
        {
            "jsonrpc": "2.0",
            "id": 25,
            "method": "tools/call",
            "params": {
                "name": "query_wandb_tool",
                "arguments": {
                    "query": """
query {
  project(entityName: "other-team", name: "private-project") {
    runs(first: 6) {
      edges { node { id ...BroadRun } }
    }
  }
}
fragment BroadRun on Run { config summaryMetrics }
""",
                    "variables": {
                        "entity": "wandb",
                        "project": "locked-project-decoy",
                        "private_filter": "customer-secret",
                    },
                },
            },
        },
        500,
    )

    [event] = [json.loads(line) for line in path.read_text().splitlines()]
    arguments = event["arguments"]
    assert arguments["project_ref"] == "*/*"
    assert arguments["raw_graphql_shape"]["graphql_scope_resolved"] is False
    assert arguments["raw_graphql_shape"]["graphql_projection_resolved"] is False
    assert arguments["raw_graphql_shape"]["broad_projection"] is True
    assert arguments["raw_graphql_shape"]["graphql_limit_resolved"] is True
    serialized = path.read_text()
    assert "other-team" not in serialized
    assert "private-project" not in serialized
    assert "locked-project-decoy" not in serialized
    assert "BroadRun" not in serialized
    assert "customer-secret" not in serialized


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
