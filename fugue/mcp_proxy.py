from __future__ import annotations

import argparse
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
        self, *, name: str, path: Path, allowed_tools: set[str] | None = None
    ) -> None:
        self.name = name
        self.path = path
        self.allowed_tools = allowed_tools
        self.started_at = time.perf_counter()
        self.pending: dict[str, tuple[float, str | None]] = {}
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
        if request_id:
            with self.lock:
                self.pending[request_id] = (time.perf_counter(), tool)
        if method == "tools/call":
            raw_arguments = (payload.get("params") or {}).get("arguments")
            recorded_arguments = (
                safe_graphql_event_arguments(raw_arguments)
                if tool == "query_wandb_tool"
                and isinstance(raw_arguments, Mapping)
                else raw_arguments
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
        started, tool = pending
        if not tool:
            return
        response_evidence = _safe_response_evidence(payload)
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


def _safe_response_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
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
        ("returned_count", ("returned_count", "returnedCount")),
        (
            "total_count",
            (
                "total_count",
                "totalCount",
                "runCount",
                "total_matching_count",
                "totalMatchingCount",
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
            ("items", "rows", "runs", "edges", "traces", "calls", "evaluations"),
        )
        if derived is not None:
            evidence["returned_count"] = derived
    for key, aliases in (
        ("has_more", ("has_more", "hasMore", "hasNextPage")),
        ("project_exhaustive", ("project_exhaustive", "projectExhaustive")),
    ):
        value = _unique_bool(objects, aliases)
        if value is not None:
            evidence[key] = value
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
