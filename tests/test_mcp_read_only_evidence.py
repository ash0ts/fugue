from __future__ import annotations

import io
import json
from pathlib import Path

from fugue.bench import export
from fugue.mcp_proxy import _Recorder, _relay_requests

MCP_CONFIG = Path(
    "examples/comparisons/wandb-mcp-maintenance/mcp.json"
)
CURRENT_CANDIDATE = Path(
    "examples/comparisons/wandb-mcp-maintenance/current-candidate.json"
)
WRITE_TOOLS = {
    "create_wandb_report_tool",
    "log_analysis_to_wandb",
}
EXACT_V10_SERVERS = {
    "wandb-main": "53b199a5f4af29aa82077e2c7f1e2c5e5e0c2ca0",
    "wandb-0-4-current": "5c6cc1c9a1079296daf6613ea6d12daebdd8bcba",
}


def test_exact_mcp_release_servers_are_locked_read_only() -> None:
    servers = json.loads(MCP_CONFIG.read_text())["mcpServers"]
    selection = json.loads(CURRENT_CANDIDATE.read_text())
    selected_servers = {
        selection[role]["server"]: selection[role]["commit"]
        for role in ("baseline", "candidate")
    }

    assert selected_servers == EXACT_V10_SERVERS
    assert set(selected_servers).issubset(servers)
    for server in servers.values():
        assert server["env"]["WANDB_MCP_READ_ONLY"] == "true"
        assert server["allowed_tools"]
        assert WRITE_TOOLS.isdisjoint(server["allowed_tools"])
    for name, revision in selected_servers.items():
        server = servers[name]
        assert any(
            f"wandb-mcp-server@{revision}" in argument
            for argument in server["args"]
        )
        assert server["version"] == f"git:{revision}"


def test_proxy_persists_only_sanitized_weave_operation_counts(
    tmp_path: Path,
) -> None:
    events_path = tmp_path / "events.jsonl"
    recorder = _Recorder(name="wandb", path=events_path)
    recorder.request(
        {
            "jsonrpc": "2.0",
            "id": 41,
            "method": "tools/call",
            "params": {
                "name": "query_weave_traces_tool",
                "arguments": {},
            },
        },
        100,
    )
    recorder.response(
        {
            "jsonrpc": "2.0",
            "id": 41,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "returned_count": 3,
                                "calls": [
                                    {
                                        "id": "private-prediction-call",
                                        "parent_id": "private-evaluation-root",
                                        "op_name": (
                                            "weave:///wandb/source/op/"
                                            "Evaluation.predict_and_score:*"
                                        ),
                                        "output": "private prediction content",
                                    },
                                    {
                                        "id": "private-summary-call",
                                        "parent_id": "private-evaluation-root",
                                        "op_name": "Evaluation.summarize",
                                        "input": "private summary content",
                                    },
                                    {
                                        "id": "private-nested-call",
                                        "parent_id": "private-prediction-call",
                                        "op_name": "maintainer.private_helper",
                                        "output": "customer-private-value",
                                    },
                                ],
                            }
                        ),
                    }
                ]
            },
        },
        500,
    )

    [event] = [
        json.loads(line)
        for line in events_path.read_text().splitlines()
        if json.loads(line)["event"] == "mcp_tool_response"
    ]
    assert event["operation_counts"] == {
        "Evaluation.predict_and_score": 1,
        "Evaluation.summarize": 1,
        "other": 1,
    }
    assert event["returned_count"] == 3

    serialized = events_path.read_text()
    for private_value in (
        "private-prediction-call",
        "private-summary-call",
        "private-nested-call",
        "private-evaluation-root",
        "private prediction content",
        "private summary content",
        "customer-private-value",
        "maintainer.private_helper",
    ):
        assert private_value not in serialized
    assert '"call_id"' not in serialized
    assert '"parent_id"' not in serialized
    assert '"op_name"' not in serialized
    assert '"content"' not in serialized


def _record_parent_filter_response(
    tmp_path: Path,
    *,
    returned_parent_ids: tuple[str, ...],
) -> dict[str, object]:
    events_path = tmp_path / "parent-filter-events.jsonl"
    recorder = _Recorder(name="wandb", path=events_path)
    recorder.request(
        {
            "jsonrpc": "2.0",
            "id": 42,
            "method": "tools/call",
            "params": {
                "name": "query_weave_traces_tool",
                "arguments": {
                    "filters": {
                        "parent_ids": [
                            "private-evaluation-root-a",
                            "private-evaluation-root-b",
                        ]
                    }
                },
            },
        },
        100,
    )
    recorder.response(
        {
            "jsonrpc": "2.0",
            "id": 42,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "returned_count": len(returned_parent_ids),
                                "calls": [
                                    {
                                        "id": f"private-call-{index}",
                                        "parent_id": parent_id,
                                        "op_name": (
                                            "Evaluation.predict_and_score"
                                        ),
                                        "output": (
                                            f"private response content {index}"
                                        ),
                                    }
                                    for index, parent_id in enumerate(
                                        returned_parent_ids
                                    )
                                ],
                            }
                        ),
                    }
                ]
            },
        },
        500,
    )

    [response] = [
        json.loads(line)
        for line in events_path.read_text().splitlines()
        if json.loads(line)["event"] == "mcp_tool_response"
    ]
    return response


def test_proxy_confirms_exact_requested_evaluation_parents(
    tmp_path: Path,
) -> None:
    response = _record_parent_filter_response(
        tmp_path,
        returned_parent_ids=(
            "private-evaluation-root-a",
            "private-evaluation-root-b",
        ),
    )

    assert response["returned_parent_filter_match"] is True
    assert response["operation_counts"] == {
        "Evaluation.predict_and_score": 2
    }
    serialized = json.dumps(response, sort_keys=True)
    assert "private-evaluation-root" not in serialized
    assert "private-call" not in serialized
    assert "private response content" not in serialized


def test_proxy_rejects_response_from_unrequested_evaluation_parent(
    tmp_path: Path,
) -> None:
    response = _record_parent_filter_response(
        tmp_path,
        returned_parent_ids=(
            "private-evaluation-root-a",
            "unrequested-evaluation-root",
        ),
    )

    assert response["returned_parent_filter_match"] is False
    serialized = json.dumps(response, sort_keys=True)
    assert "private-evaluation-root" not in serialized
    assert "unrequested-evaluation-root" not in serialized
    assert "private-call" not in serialized
    assert "private response content" not in serialized


def _tool_request(
    *,
    request_id: int,
    tool: str,
    arguments: dict[str, object],
) -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    }


def test_proxy_enforces_exact_source_scope_before_relay(
    tmp_path: Path,
) -> None:
    events_path = tmp_path / "source-policy-events.jsonl"
    source_project = "wandb/fugue-mcp-release-source-v2"
    recorder = _Recorder(
        name="wandb",
        path=events_path,
        source_project=source_project,
    )

    assert recorder.request(
        _tool_request(
            request_id=1,
            tool="get_run_history_tool",
            arguments={
                "entity_name": "wandb",
                "project_name": "fugue-mcp-release-source-v2",
                "run_name": "maint-r18-06",
            },
        ),
        100,
    )
    assert not recorder.request(
        _tool_request(
            request_id=2,
            tool="get_run_history_tool",
            arguments={
                "entity_name": "wandb",
                "project_name": "news-research-agent",
                "run_name": "unrelated",
            },
        ),
        100,
    )
    assert not recorder.request(
        _tool_request(
            request_id=3,
            tool="get_run_history_tool",
            arguments={"run_name": "unscoped"},
        ),
        100,
    )

    events = [
        json.loads(line) for line in events_path.read_text().splitlines()
    ]
    assert [event["event"] for event in events] == [
        "mcp_tool_request",
        "mcp_tool_denied",
        "mcp_tool_denied",
    ]
    assert all(
        event.get("reason") == "source_scope_policy"
        for event in events[1:]
    )
    assert "news-research-agent" not in events_path.read_text()


def test_proxy_rejects_graphql_mutation_and_accepts_exact_scoped_query(
    tmp_path: Path,
) -> None:
    events_path = tmp_path / "graphql-policy-events.jsonl"
    recorder = _Recorder(
        name="wandb",
        path=events_path,
        source_project="wandb/fugue-mcp-release-source-v2",
    )
    variables = {
        "entity": "wandb",
        "project": "fugue-mcp-release-source-v2",
    }
    mutation = """
        mutation Rename($entity: String!, $project: String!) {
          project(entityName: $entity, name: $project) { id }
        }
    """
    query = """
        query Runs($entity: String!, $project: String!) {
          project(entityName: $entity, name: $project) { runCount }
        }
    """

    assert not recorder.request(
        _tool_request(
            request_id=10,
            tool="query_wandb_tool",
            arguments={"query": mutation, "variables": variables},
        ),
        100,
    )
    assert recorder.request(
        _tool_request(
            request_id=11,
            tool="query_wandb_tool",
            arguments={"query": query, "variables": variables},
        ),
        100,
    )

    events = [
        json.loads(line) for line in events_path.read_text().splitlines()
    ]
    assert events[0]["event"] == "mcp_tool_denied"
    assert events[0]["reason"] == "source_scope_policy"
    assert events[1]["event"] == "mcp_tool_request"
    assert events[1]["arguments"]["raw_graphql_shape"][
        "graphql_operation_type"
    ] == "query"
    serialized = events_path.read_text()
    assert "mutation Rename" not in serialized
    assert "$entity" not in serialized


def test_proxy_hashes_all_parent_filter_aliases_without_persisting_ids(
    tmp_path: Path,
) -> None:
    events_path = tmp_path / "parent-alias-events.jsonl"
    recorder = _Recorder(name="wandb", path=events_path)
    private_ids = (
        "private-evaluation-root-a",
        "private-evaluation-root-b",
    )

    assert recorder.request(
        _tool_request(
            request_id=20,
            tool="query_weave_traces_tool",
            arguments={
                "filters": {
                    "parent_call_ids": list(private_ids),
                }
            },
        ),
        100,
    )

    [event] = [
        json.loads(line) for line in events_path.read_text().splitlines()
    ]
    arguments = event["arguments"]
    assert arguments["filters"] == {}
    assert arguments["parent_filter_count"] == 2
    assert len(arguments["parent_filter_digest"]) == 64
    serialized = events_path.read_text()
    assert "parent_call_ids" not in serialized
    assert all(private_id not in serialized for private_id in private_ids)


def test_denied_source_requests_never_reach_the_upstream_transport(
    tmp_path: Path,
) -> None:
    requests = (
        _tool_request(
            request_id=30,
            tool="get_run_history_tool",
            arguments={
                "entity_name": "wandb",
                "project_name": "news-research-agent",
            },
        ),
        _tool_request(
            request_id=31,
            tool="get_run_history_tool",
            arguments={"run_name": "unscoped"},
        ),
        _tool_request(
            request_id=32,
            tool="query_wandb_tool",
            arguments={
                "query": (
                    "mutation M($entity: String!, $project: String!) "
                    "{ project(entityName: $entity, name: $project) { id } }"
                ),
                "variables": {
                    "entity": "wandb",
                    "project": "fugue-mcp-release-source-v2",
                },
            },
        ),
    )
    source = io.BytesIO(
        b"".join(
            json.dumps(request, separators=(",", ":")).encode() + b"\n"
            for request in requests
        )
    )
    upstream = io.BytesIO()
    client = io.BytesIO()
    recorder = _Recorder(
        name="wandb",
        path=tmp_path / "relay-events.jsonl",
        source_project="wandb/fugue-mcp-release-source-v2",
    )

    _relay_requests(source, upstream, client, recorder)

    assert upstream.getvalue() == b""
    responses = [
        json.loads(line)
        for line in client.getvalue().decode().splitlines()
    ]
    assert [response["id"] for response in responses] == [30, 31, 32]
    assert all(
        response["error"]["message"] == "Tool denied by Fugue policy"
        for response in responses
    )


def test_export_retains_only_valid_sanitized_operation_counts() -> None:
    response = {
        "tool": "query_weave_traces_tool",
        "terminal_status": "succeeded",
        "successful": True,
        "returned_count": 10,
        "operation_counts": {
            "Evaluation.predict_and_score": 8,
            "Evaluation.summarize": 1,
            "other": 1,
        },
        "call_ids": ["private-call"],
        "content": "private call content",
    }

    normalized = export._normalized_mcp_response(
        [response],
        tool="query_weave_traces_tool",
    )

    assert normalized["operation_counts"] == {
        "Evaluation.predict_and_score": 8,
        "Evaluation.summarize": 1,
        "other": 1,
    }
    assert normalized["returned_count"] == 10
    assert "call_ids" not in normalized
    assert "content" not in normalized

    mismatched = export._normalized_mcp_response(
        [
            {
                **response,
                "operation_counts": {
                    "Evaluation.predict_and_score": 8,
                    "Evaluation.summarize": 1,
                },
            }
        ],
        tool="query_weave_traces_tool",
    )
    assert "operation_counts" not in mismatched
