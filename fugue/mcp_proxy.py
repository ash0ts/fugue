from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO

from fugue.mcp_evidence import (
    safe_graphql_event_arguments,
    safe_structured_error_code,
    validated_graphql_query_shape,
)
from fugue.redaction import redact_text, secrets_from_env, sensitive_key

_MAX_TEXT = 1_000
_MAX_EVENT_BYTES = 16_384
_MAX_RESPONSE_INSPECTION_BYTES = 1_000_000
_ENV_SECRETS = secrets_from_env(os.environ)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fugue.mcp_proxy")
    parser.add_argument("--name", required=True)
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--allow-tool", action="append", default=[])
    parser.add_argument(
        "--source-project",
        default=os.environ.get("FUGUE_SOURCE_EVIDENCE_PROJECT", ""),
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("an upstream command is required after --")

    process = subprocess.Popen(
        command,
        cwd=args.cwd,
        env=dict(os.environ),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
    )
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("failed to open MCP proxy pipes")

    recorder = _Recorder(
        name=args.name,
        allowed_tools=set(args.allow_tool) or None,
        source_project=str(args.source_project or "").strip() or None,
        path=Path(
            os.environ.get(
                "FUGUE_CONTEXT_EVENTS_PATH",
                "/logs/artifacts/fugue-context-events.jsonl",
            )
        ),
    )
    request_thread = threading.Thread(
        target=_relay_requests,
        args=(sys.stdin.buffer, process.stdin, sys.stdout.buffer, recorder),
        daemon=True,
    )
    response_thread = threading.Thread(
        target=_relay_responses,
        args=(process.stdout, sys.stdout.buffer, recorder),
        daemon=True,
    )
    request_thread.start()
    response_thread.start()
    return_code = process.wait()
    request_thread.join(timeout=1)
    response_thread.join(timeout=1)
    return return_code


class _Recorder:
    def __init__(
        self,
        *,
        name: str,
        path: Path,
        allowed_tools: set[str] | None = None,
        source_project: str | None = None,
    ) -> None:
        self.name = name
        self.path = path
        self.allowed_tools = allowed_tools
        self.source_project = source_project
        self.started_at = time.perf_counter()
        self.pending: dict[
            str,
            tuple[float, str | None, tuple[str, ...]],
        ] = {}
        self.lock = threading.Lock()

    def request(self, payload: dict[str, Any], size: int) -> bool:
        request_id = _request_id(payload)
        method = str(payload.get("method") or "")
        tool = None
        if method == "tools/call":
            params = payload.get("params") or {}
            tool = str(params.get("name") or "") if isinstance(params, dict) else None
            if self.allowed_tools is not None and tool not in self.allowed_tools:
                self.write(
                    {
                        "event": "mcp_tool_denied",
                        "layer": "proxy",
                        "server": self.name,
                        "tool": tool,
                        "request_id": request_id,
                        "request_bytes": size,
                    }
                )
                return False
        raw_arguments = (
            (payload.get("params") or {}).get("arguments")
            if method == "tools/call"
            else None
        )
        if (
            method == "tools/call"
            and self.source_project is not None
            and not _request_is_source_scoped(
                tool=tool,
                arguments=raw_arguments,
                source_project=self.source_project,
            )
        ):
            self.write(
                {
                    "event": "mcp_tool_denied",
                    "layer": "proxy",
                    "server": self.name,
                    "tool": tool,
                    "request_id": request_id,
                    "request_bytes": size,
                    "reason": "source_scope_policy",
                }
            )
            return False
        requested_parent_ids = _requested_parent_ids(raw_arguments)
        if request_id:
            with self.lock:
                self.pending[request_id] = (
                    time.perf_counter(),
                    tool,
                    requested_parent_ids,
                )
        if method == "tools/call":
            shaped_arguments = (
                safe_graphql_event_arguments(raw_arguments)
                if tool == "query_wandb_tool"
                and isinstance(raw_arguments, Mapping)
                else raw_arguments
            )
            recorded_arguments = _safe_request_arguments(
                shaped_arguments,
                parent_ids=requested_parent_ids,
            )
            self.write(
                {
                    "event": "mcp_tool_request",
                    "layer": "proxy",
                    "server": self.name,
                    "tool": tool,
                    "request_id": request_id,
                    "request_bytes": size,
                    "arguments": _sanitize(recorded_arguments),
                }
            )
        return True

    def response(self, payload: dict[str, Any], size: int) -> None:
        request_id = _request_id(payload)
        with self.lock:
            pending = self.pending.pop(request_id, None) if request_id else None
        if pending is None:
            return
        started, tool, expected_parent_ids = pending
        if not tool:
            return
        response_evidence = _safe_response_evidence(
            payload,
            expected_parent_ids=expected_parent_ids,
        )
        self.write(
            {
                "event": "mcp_tool_response",
                "layer": "upstream",
                "server": self.name,
                "tool": tool,
                "request_id": request_id,
                "response_bytes": size,
                "latency_ms": (time.perf_counter() - started) * 1_000,
                "error": response_evidence["terminal_status"] != "succeeded",
                **response_evidence,
            }
        )

    def write(self, event: dict[str, Any]) -> None:
        event.setdefault("elapsed_ms", (time.perf_counter() - self.started_at) * 1_000)
        line = json.dumps(event, sort_keys=True, default=str)
        if len(line.encode()) > _MAX_EVENT_BYTES:
            event = {
                key: value
                for key, value in event.items()
                if key not in {"arguments", "error"}
            }
            event["payload_truncated"] = True
            line = json.dumps(event, sort_keys=True, default=str)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as handle:
                handle.write(line + "\n")
        except OSError:
            pass


def _relay_requests(
    source: BinaryIO,
    target: BinaryIO,
    client: BinaryIO,
    recorder: _Recorder,
) -> None:
    while line := source.readline():
        try:
            payload = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            target.write(line)
            target.flush()
            continue
        if not isinstance(payload, dict) or recorder.request(payload, len(line)):
            target.write(line)
            target.flush()
            continue
        response = {
            "jsonrpc": "2.0",
            "id": payload.get("id"),
            "error": {"code": -32601, "message": "Tool denied by Fugue policy"},
        }
        client.write((json.dumps(response, separators=(",", ":")) + "\n").encode())
        client.flush()


def _relay_responses(source: BinaryIO, target: BinaryIO, recorder: _Recorder) -> None:
    _relay(source, target, recorder.response)


def _relay(
    source: BinaryIO,
    target: BinaryIO,
    observe: Any,
) -> None:
    while line := source.readline():
        target.write(line)
        target.flush()
        try:
            payload = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(payload, dict):
            observe(payload, len(line))


def _request_id(payload: dict[str, Any]) -> str | None:
    value = payload.get("id")
    return str(value) if value is not None else None


def _safe_response_evidence(
    payload: Mapping[str, Any],
    *,
    expected_parent_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Summarize a tool response without persisting result content."""

    protocol_error = payload.get("error")
    if protocol_error not in (None, False, ""):
        return {
            "terminal_status": "protocol_error",
            "successful": False,
            "structured_error_code": safe_structured_error_code(
                protocol_error,
                "jsonrpc_error",
            ),
        }

    result = payload.get("result")
    objects = list(_response_objects(result))
    structured_error = _mcp_structured_error_code(result)
    if structured_error is not None:
        return {
            "terminal_status": "structured_error",
            "successful": False,
            "structured_error_code": structured_error,
        }

    evidence: dict[str, Any] = {
        "terminal_status": "succeeded",
        "successful": True,
    }
    for key, aliases in (
        ("returned_count", ("returned_count", "returnedCount", "resolved")),
        (
            "total_count",
            (
                "total_count",
                "totalCount",
                "run_count",
                "runCount",
                "total_matching_count",
                "totalMatchingCount",
                "total_requested",
                "totalRequested",
            ),
        ),
        ("rows_scanned", ("rows_scanned", "rowsScanned")),
        ("total_steps", ("total_steps", "totalSteps")),
    ):
        value = _unique_non_negative_int(objects, aliases)
        if value is not None:
            evidence[key] = value
    prediction_count = _unique_list_integer_sum(
        objects,
        collection_keys=("evaluations",),
        item_keys=(
            "total_predictions",
            "totalPredictions",
            "prediction_count",
            "predictionCount",
        ),
    )
    if prediction_count is not None:
        evidence["prediction_count"] = prediction_count
    if "returned_count" not in evidence:
        derived = _unique_list_count(
            objects,
            (
                "items",
                "rows",
                "runs",
                "edges",
                "traces",
                "calls",
                "evaluations",
                "versions",
                "artifacts",
            ),
        )
        if derived is not None:
            evidence["returned_count"] = derived
    operation_counts = _safe_weave_operation_counts(objects)
    if operation_counts is not None and (
        evidence.get("returned_count") is None
        or sum(operation_counts.values()) == evidence["returned_count"]
    ):
        evidence["operation_counts"] = operation_counts
    parent_filter_match = _safe_weave_parent_filter_match(
        objects,
        expected_parent_ids=expected_parent_ids,
    )
    if parent_filter_match is not None:
        evidence["returned_parent_filter_match"] = parent_filter_match
    for key, aliases in (
        ("has_more", ("has_more", "hasMore", "hasNextPage")),
        ("project_exhaustive", ("project_exhaustive", "projectExhaustive")),
        ("digest_match", ("digest_match", "digestMatch")),
    ):
        value = _unique_bool(objects, aliases)
        if value is not None:
            evidence[key] = value
    artifact_digest = _unique_string(
        objects,
        ("artifact_digest", "artifactDigest", "digest"),
    )
    if artifact_digest is not None:
        evidence["artifact_digest"] = artifact_digest
    returned_trace_ids = _unique_strings(objects, ("trace_id", "traceId"))
    if returned_trace_ids:
        evidence["returned_trace_ids_digest"] = _stable_digest(list(returned_trace_ids))
        evidence["returned_trace_ids_count"] = len(returned_trace_ids)
    if (
        evidence.get("project_exhaustive") is None
        and evidence.get("has_more") is False
        and evidence.get("returned_count") is not None
        and evidence.get("returned_count") == evidence.get("total_count")
    ):
        evidence["project_exhaustive"] = True
    truncation = _response_truncation(objects)
    if truncation is not None:
        evidence["truncation_applied"] = truncation
    coverage_status = _derived_coverage_status(evidence, objects)
    if coverage_status is not None:
        evidence["coverage_status"] = coverage_status
    return evidence


def _requested_parent_ids(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, Mapping):
        return ()
    values: set[str] = set()
    for key, value in raw.items():
        if key in _PARENT_ID_KEYS:
            candidates = value if isinstance(value, list) else [value]
            values.update(
                str(item).strip()
                for item in candidates
                if isinstance(item, str) and item.strip()
            )
        elif isinstance(value, Mapping):
            values.update(_requested_parent_ids(value))
    return tuple(sorted(values))


_ENTITY_KEYS = ("entity", "entity_name", "entityName", "wandb_entity")
_PROJECT_KEYS = ("project", "project_name", "projectName", "wandb_project")
_PROJECT_REF_KEYS = (
    "project_path",
    "project_ref",
    "project_slug",
    "entity_project",
)
_ARTIFACT_REF_KEYS = (
    "artifact_name",
    "artifact_name_a",
    "artifact_name_b",
)
_PARENT_ID_KEYS = frozenset(
    {
        "parent_id",
        "parent_ids",
        "parentId",
        "parentIds",
        "parent_call_id",
        "parent_call_ids",
        "parentCallId",
        "parentCallIds",
    }
)


def _request_is_source_scoped(
    *,
    tool: str | None,
    arguments: Any,
    source_project: str,
) -> bool:
    if not tool or not isinstance(arguments, Mapping):
        return False
    shaped = (
        safe_graphql_event_arguments(arguments)
        if tool == "query_wandb_tool"
        else dict(arguments)
    )
    raw_shape = validated_graphql_query_shape(shaped.get("raw_graphql_shape"))
    if raw_shape and (
        raw_shape.get("graphql_operation_type") != "query"
        or raw_shape.get("graphql_scope_resolved") is not True
    ):
        return False
    return _requested_projects(shaped) == {source_project}


def _requested_projects(raw: Mapping[str, Any]) -> set[str]:
    projects: set[str] = set()
    for value in _nested_mappings(raw):
        entity = _first_text(value, _ENTITY_KEYS)
        project = _first_text(value, _PROJECT_KEYS)
        if project:
            projects.add(
                project
                if "/" in project
                else f"{entity}/{project}"
                if entity
                else f"*/{project}"
            )
        for key in _PROJECT_REF_KEYS:
            reference = value.get(key)
            if isinstance(reference, str) and reference.strip():
                projects.add(reference.strip())
        for key in _ARTIFACT_REF_KEYS:
            reference = value.get(key)
            project = _qualified_artifact_project(reference)
            if project:
                projects.add(project)
    return projects


def _qualified_artifact_project(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    parts = raw.strip().split("/")
    if (
        len(parts) != 3
        or any(not part or part in {".", ".."} for part in parts)
        or ":" not in parts[2]
    ):
        return None
    return f"{parts[0]}/{parts[1]}"


def _nested_mappings(raw: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result = [raw]
    for value in raw.values():
        if isinstance(value, Mapping):
            result.extend(_nested_mappings(value))
    return result


def _first_text(raw: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _safe_request_arguments(
    raw: Any,
    *,
    parent_ids: tuple[str, ...],
) -> Any:
    if not isinstance(raw, Mapping):
        return raw
    result = {
        str(key): _safe_request_arguments(value, parent_ids=())
        for key, value in raw.items()
        if key not in _PARENT_ID_KEYS
    }
    if parent_ids:
        result["parent_filter_digest"] = _stable_digest(list(parent_ids))
        result["parent_filter_count"] = len(parent_ids)
    return result


def _stable_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _safe_weave_parent_filter_match(
    objects: list[Mapping[str, Any]],
    *,
    expected_parent_ids: tuple[str, ...],
) -> bool | None:
    expected = set(expected_parent_ids)
    if not expected:
        return None
    observed: list[str] = []
    for value in objects:
        if not (
            value.get("id")
            or value.get("call_id")
            or value.get("callId")
        ):
            continue
        operation = value.get("op_name") or value.get("opName")
        if not isinstance(operation, str) or not operation.strip():
            continue
        parent = value.get("parent_id") or value.get("parentId")
        if not isinstance(parent, str) or not parent.strip():
            return False
        observed.append(parent.strip())
    return bool(observed) and all(item in expected for item in observed)


def _safe_weave_operation_counts(
    objects: list[Mapping[str, Any]],
) -> dict[str, int] | None:
    """Count only release-relevant operation classes without persisting rows."""
    allowed = {
        "Evaluation.predict_and_score",
        "Evaluation.summarize",
    }
    by_id: dict[str, str] = {}
    for value in objects:
        call_id = value.get("id") or value.get("call_id") or value.get(
            "callId"
        )
        parent_id = value.get("parent_id") or value.get(
            "parentId"
        )
        operation = value.get("op_name") or value.get("opName")
        if not all(
            isinstance(item, str) and item.strip()
            for item in (call_id, parent_id, operation)
        ):
            continue
        normalized = str(operation).strip()
        if "/op/" in normalized:
            normalized = normalized.rsplit("/op/", 1)[-1]
        normalized = normalized.split(":", 1)[0]
        label = normalized if normalized in allowed else "other"
        prior = by_id.setdefault(str(call_id), label)
        if prior != label:
            return None
    if not by_id:
        return None
    counts: dict[str, int] = {}
    for label in by_id.values():
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _response_objects(value: Any, *, depth: int = 0) -> list[Mapping[str, Any]]:
    if depth > 8:
        return []
    if isinstance(value, str):
        try:
            encoded_size = len(value.encode())
        except UnicodeEncodeError:
            return []
        if encoded_size > _MAX_RESPONSE_INSPECTION_BYTES:
            return []
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return []
        return _response_objects(decoded, depth=depth + 1)
    if isinstance(value, list):
        result: list[Mapping[str, Any]] = []
        for item in value[:50]:
            result.extend(_response_objects(item, depth=depth + 1))
        return result
    if not isinstance(value, Mapping):
        return []

    result = [value]
    for key in (
        "result",
        "structuredContent",
        "data",
        "project",
        "runs",
        "pageInfo",
        "traces",
        "metadata",
        "calls",
        "evaluations",
        "artifact",
    ):
        if key in value:
            result.extend(_response_objects(value[key], depth=depth + 1))
    content = value.get("content")
    if isinstance(content, str | list | Mapping):
        result.extend(_response_objects(content, depth=depth + 1))
    if value.get("type") == "text" and isinstance(value.get("text"), str):
        result.extend(_response_objects(value["text"], depth=depth + 1))
    return result


def _mcp_structured_error_code(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    envelopes = [value, *_mcp_content_envelopes(value)]
    if value.get("isError") is True:
        for envelope in envelopes:
            code = _error_code_from_envelope(envelope, fallback="mcp_is_error")
            if code != "mcp_is_error":
                return code
        return "mcp_is_error"
    for envelope in envelopes:
        code = _error_code_from_envelope(envelope, fallback="tool_error")
        if code is not None:
            return code
    return None


def _mcp_content_envelopes(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    envelopes: list[Mapping[str, Any]] = []
    structured = value.get("structuredContent")
    if isinstance(structured, Mapping):
        envelopes.extend(_result_envelope_chain(structured))
    content = value.get("content")
    items = content if isinstance(content, list) else [content]
    for item in items[:50]:
        if isinstance(item, str):
            decoded = _decode_json_mapping(item)
        elif isinstance(item, Mapping) and item.get("type") == "text":
            decoded = _decode_json_mapping(item.get("text"))
        else:
            decoded = None
        if decoded is not None:
            envelopes.extend(_result_envelope_chain(decoded))
    return envelopes


def _result_envelope_chain(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result = [value]
    current: Mapping[str, Any] = value
    for _ in range(3):
        nested: Any = current.get("result")
        if isinstance(nested, str):
            nested = _decode_json_mapping(nested)
        if not isinstance(nested, Mapping):
            break
        result.append(nested)
        current = nested
    return result


def _decode_json_mapping(value: Any) -> Mapping[str, Any] | None:
    if not isinstance(value, str):
        return None
    try:
        if len(value.encode()) > _MAX_RESPONSE_INSPECTION_BYTES:
            return None
    except UnicodeEncodeError:
        return None
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, Mapping) else None


def _error_code_from_envelope(
    value: Mapping[str, Any],
    *,
    fallback: str,
) -> str | None:
    error = value.get("error")
    if error not in (None, False, ""):
        return safe_structured_error_code(error, fallback)
    failed = (
        value.get("isError") is True
        or value.get("success") is False
        or value.get("successful") is False
        or value.get("ok") is False
        or str(value.get("status") or "").strip().lower()
        in {"error", "failed", "failure"}
    )
    if not failed:
        return None
    for key in ("error_code", "errorCode", "code"):
        if key in value:
            return safe_structured_error_code(value[key], fallback)
    return fallback


def _unique_non_negative_int(
    values: list[Mapping[str, Any]], keys: tuple[str, ...]
) -> int | None:
    observed = {
        value[key]
        for value in values
        for key in keys
        if type(value.get(key)) is int and value[key] >= 0
    }
    return next(iter(observed)) if len(observed) == 1 else None


def _unique_bool(
    values: list[Mapping[str, Any]], keys: tuple[str, ...]
) -> bool | None:
    observed = {
        value[key]
        for value in values
        for key in keys
        if type(value.get(key)) is bool
    }
    return next(iter(observed)) if len(observed) == 1 else None


def _unique_string(
    values: list[Mapping[str, Any]], keys: tuple[str, ...]
) -> str | None:
    observed = {
        value[key].strip()
        for value in values
        for key in keys
        if isinstance(value.get(key), str) and value[key].strip()
    }
    return next(iter(observed)) if len(observed) == 1 else None


def _unique_strings(
    values: list[Mapping[str, Any]], keys: tuple[str, ...]
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                value[key].strip()
                for value in values
                for key in keys
                if isinstance(value.get(key), str) and value[key].strip()
            }
        )
    )


def _unique_list_count(
    values: list[Mapping[str, Any]], keys: tuple[str, ...]
) -> int | None:
    observed = {
        len(value[key])
        for value in values
        for key in keys
        if isinstance(value.get(key), list)
    }
    return next(iter(observed)) if len(observed) == 1 else None


def _unique_list_integer_sum(
    values: list[Mapping[str, Any]],
    *,
    collection_keys: tuple[str, ...],
    item_keys: tuple[str, ...],
) -> int | None:
    observed: set[int] = set()
    for value in values:
        for collection_key in collection_keys:
            items = value.get(collection_key)
            if not isinstance(items, list):
                continue
            counts: list[int] = []
            valid = True
            for item in items:
                if not isinstance(item, Mapping):
                    valid = False
                    break
                item_values = {
                    item[key]
                    for key in item_keys
                    if type(item.get(key)) is int and item[key] >= 0
                }
                if len(item_values) != 1:
                    valid = False
                    break
                counts.append(next(iter(item_values)))
            if valid and counts:
                observed.add(sum(counts))
    return next(iter(observed)) if len(observed) == 1 else None


def _response_truncation(values: list[Mapping[str, Any]]) -> bool | None:
    observed = {
        value[key]
        for value in values
        for key in ("truncated", "truncation_applied", "truncationApplied")
        if type(value.get(key)) is bool
    }
    observed.update(
        truncation["applied"]
        for value in values
        if isinstance((truncation := value.get("truncation")), Mapping)
        and type(truncation.get("applied")) is bool
    )
    return next(iter(observed)) if len(observed) == 1 else None


def _derived_coverage_status(
    evidence: Mapping[str, Any], values: list[Mapping[str, Any]]
) -> str | None:
    if (
        evidence.get("project_exhaustive") is True
        and evidence.get("has_more") is False
    ):
        return "project-exhaustive"
    if evidence.get("has_more") is True:
        return "bounded-page"
    exact = _unique_bool(values, ("exact",))
    if exact is True:
        return "exact-target"
    rows_scanned = evidence.get("rows_scanned")
    total_steps = evidence.get("total_steps")
    if (
        type(rows_scanned) is int
        and type(total_steps) is int
        and total_steps > 0
        and rows_scanned == total_steps
    ):
        return "bounded-full-history"
    return None


def _sanitize(value: Any, *, key: str = "") -> Any:
    if sensitive_key(key):
        return "[redacted]"
    if isinstance(value, dict):
        return {
            str(name): _sanitize(item, key=str(name)) for name, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize(item, key=key) for item in value[:50]]
    if isinstance(value, str):
        return redact_text(value[:_MAX_TEXT], _ENV_SECRETS)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:_MAX_TEXT]


if __name__ == "__main__":
    raise SystemExit(main())
