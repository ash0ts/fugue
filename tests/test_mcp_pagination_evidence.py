from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fugue.bench import export
from fugue.mcp_proxy import _Recorder, _safe_response_evidence

PROJECT = "wandb/fugue-mcp-release-source-v2"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _request(recorder: _Recorder, request_id: int, cursor: object) -> None:
    recorder.request(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {
                "name": "query_wandb_tool",
                "arguments": {
                    "entity_name": "wandb",
                    "project_name": "fugue-mcp-release-source-v2",
                    "resource": "runs",
                    "cursor": cursor,
                    "limit": 3,
                    "filters": {
                        "attempt_label": "private-failure-label",
                    },
                    "expected_value": "private-answer",
                },
            },
        },
        100,
    )


def _response(
    recorder: _Recorder,
    request_id: int,
    *,
    ids: tuple[str, ...],
    next_cursor: str | None,
    has_more: bool,
) -> None:
    recorder.response(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "result": {
                                    "items": [
                                        {
                                            "id": value,
                                            "label": "private-result-label",
                                        }
                                        for value in ids
                                    ],
                                    "returned_count": len(ids),
                                    "total_count": 6,
                                    "has_more": has_more,
                                    "next_cursor": next_cursor,
                                }
                            }
                        ),
                    }
                ]
            },
        },
        500,
    )


def test_proxy_and_export_verify_private_cursor_chain_and_disjoint_pages(
    tmp_path: Path,
) -> None:
    event_log = (
        tmp_path / "trial" / "artifacts" / "fugue-context-events.jsonl"
    )
    recorder = _Recorder(name="wandb-mcp-0-4", path=event_log)
    _request(recorder, 1, None)
    _response(
        recorder,
        1,
        ids=("private-run-a", "private-run-b", "private-run-c"),
        next_cursor="opaque-private-page-2",
        has_more=True,
    )
    _request(recorder, 2, "opaque-private-page-2")
    _response(
        recorder,
        2,
        ids=("private-run-d", "private-run-e", "private-run-f"),
        next_cursor=None,
        has_more=False,
    )

    serialized = event_log.read_text()
    for private_value in (
        "opaque-private-page-2",
        "private-run-a",
        "private-run-f",
        "private-failure-label",
        "private-result-label",
        "private-answer",
    ):
        assert private_value not in serialized

    events = [json.loads(line) for line in serialized.splitlines()]
    requests = [
        event for event in events if event["event"] == "mcp_tool_request"
    ]
    responses = [
        event for event in events if event["event"] == "mcp_tool_response"
    ]
    assert requests[0]["cursor_present"] is False
    assert requests[0]["cursor_metadata_verified"] is True
    assert "cursor_digest" not in requests[0]
    assert requests[1]["cursor_present"] is True
    assert requests[1]["cursor_metadata_verified"] is True
    assert requests[1]["cursor_digest"] == _digest("opaque-private-page-2")
    assert requests[1]["arguments"]["cursor"] == "[opaque-cursor]"
    assert responses[0]["next_cursor_present"] is True
    assert responses[0]["next_cursor_metadata_verified"] is True
    assert responses[0]["next_cursor_digest"] == requests[1]["cursor_digest"]
    assert responses[0]["pagination_metadata_verified"] is True
    assert responses[1]["next_cursor_present"] is False
    assert responses[1]["next_cursor_metadata_verified"] is True
    assert responses[1]["pagination_metadata_verified"] is True
    assert "next_cursor_digest" not in responses[1]

    summary = export._context_event_summary(tmp_path / "trial")
    first, second = summary["mcp_tool_calls"]
    assert first["next_cursor_digest"] == second["cursor_digest"]
    assert first["cursor_present"] is False
    assert second["cursor_present"] is True
    assert first["returned_object_id_count"] == 3
    assert second["returned_object_id_count"] == 3
    assert first["returned_object_ids_unique"] is True
    assert second["returned_object_ids_unique"] is True
    assert set(first["returned_object_id_hashes"]).isdisjoint(
        second["returned_object_id_hashes"]
    )
    assert first["pagination_metadata_verified"] is True
    assert second["pagination_metadata_verified"] is True
    assert second["next_cursor_present"] is False


def test_duplicate_ids_are_detected_and_hash_list_is_bounded() -> None:
    duplicate = _safe_response_evidence(
        {
            "result": {
                "items": [{"id": "same-private-id"}, {"id": "same-private-id"}],
                "returned_count": 2,
            }
        }
    )

    assert duplicate["returned_object_id_metadata_verified"] is True
    assert duplicate["returned_object_id_count"] == 2
    assert duplicate["returned_object_id_hashes"] == [
        _digest("same-private-id"),
        _digest("same-private-id"),
    ]
    assert duplicate["returned_object_ids_unique"] is False
    assert "same-private-id" not in json.dumps(duplicate, sort_keys=True)

    oversized = _safe_response_evidence(
        {
            "result": {
                "items": [{"id": f"private-{index}"} for index in range(51)],
                "returned_count": 51,
            }
        }
    )
    assert oversized["returned_object_id_metadata_verified"] is False
    assert "returned_object_id_hashes" not in oversized


def test_malformed_cursor_and_object_metadata_fail_closed(
    tmp_path: Path,
) -> None:
    event_log = (
        tmp_path / "trial" / "artifacts" / "fugue-context-events.jsonl"
    )
    recorder = _Recorder(name="wandb-mcp-0-4", path=event_log)
    _request(recorder, 3, 7)
    recorder.response(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "result": {
                                    "items": [
                                        {
                                            "id": "private-id-a",
                                            "run_id": "private-id-b",
                                        }
                                    ],
                                    "returned_count": 1,
                                    "has_more": True,
                                    "next_cursor": "private-next-a",
                                    "pagination": {
                                        "endCursor": "private-next-b",
                                    },
                                }
                            }
                        ),
                    }
                ]
            },
        },
        500,
    )

    serialized = event_log.read_text()
    for private_value in (
        "private-id-a",
        "private-id-b",
        "private-next-a",
        "private-next-b",
    ):
        assert private_value not in serialized
    request, response = [
        json.loads(line) for line in serialized.splitlines()
    ]
    assert request["cursor_present"] is True
    assert request["cursor_metadata_verified"] is False
    assert "cursor_digest" not in request
    assert response["next_cursor_present"] is True
    assert response["next_cursor_metadata_verified"] is False
    assert response["pagination_metadata_verified"] is False
    assert "next_cursor_digest" not in response
    assert response["returned_object_id_metadata_verified"] is False

    [call] = export._context_event_summary(tmp_path / "trial")[
        "mcp_tool_calls"
    ]
    assert call["cursor_metadata_verified"] is False
    assert call["next_cursor_metadata_verified"] is False
    assert call["pagination_metadata_verified"] is False
    assert call["returned_object_id_metadata_verified"] is False


def test_ambiguous_responses_do_not_export_pagination_claims(
    tmp_path: Path,
) -> None:
    trial = tmp_path / "trial"
    event_log = trial / "artifacts" / "fugue-context-events.jsonl"
    event_log.parent.mkdir(parents=True)
    request = {
        "event": "mcp_tool_request",
        "server": "wandb",
        "tool": "query_wandb_tool",
        "request_id": "same",
        "arguments": {
            "entity_name": "wandb",
            "project_name": "fugue-mcp-release-source-v2",
        },
        "cursor_present": False,
        "cursor_metadata_verified": True,
    }
    response = {
        "event": "mcp_tool_response",
        "server": "wandb",
        "tool": "query_wandb_tool",
        "request_id": "same",
        "terminal_status": "succeeded",
        "successful": True,
        "next_cursor_present": True,
        "next_cursor_digest": "a" * 64,
        "next_cursor_metadata_verified": True,
        "returned_object_id_count": 1,
        "returned_object_id_hashes": ["b" * 64],
        "returned_object_id_metadata_verified": True,
        "returned_object_ids_unique": True,
    }
    event_log.write_text(
        "\n".join(json.dumps(value) for value in (request, response, response))
        + "\n"
    )

    [call] = export._context_event_summary(trial)["mcp_tool_calls"]
    assert call["terminal_status"] == "ambiguous"
    assert call["successful"] is False
    assert call["response_metadata_verified"] is False
    assert "next_cursor_digest" not in call
    assert "returned_object_id_hashes" not in call


def test_trace_specific_evidence_remains_separate_from_object_hashes() -> None:
    evidence = _safe_response_evidence(
        {
            "result": {
                "traces": [
                    {"trace_id": "private-trace-a"},
                    {"trace_id": "private-trace-b"},
                ],
                "returned_count": 2,
            }
        }
    )

    assert evidence["returned_trace_ids_count"] == 2
    assert len(evidence["returned_trace_ids_digest"]) == 64
    assert evidence["returned_object_id_count"] == 2
    assert evidence["returned_object_id_hashes"] == [
        _digest("private-trace-a"),
        _digest("private-trace-b"),
    ]
    serialized = json.dumps(evidence, sort_keys=True)
    assert "private-trace-a" not in serialized
    assert "private-trace-b" not in serialized
