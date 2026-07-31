"""Deterministic MCP tool-surface scorer.

V7 keeps user-visible correctness and safety independent from mechanism
evidence. ``target_behavior_satisfied`` reports whether the locked release
behavior was actually observed; it is intentionally a non-critical mechanism
dimension in the V7 comparison spec.
"""

import hashlib
import json
import math

_PROJECT = "wandb/fugue-mcp-release-source-v2"
_DIMENSIONS = dict.fromkeys(
    (
        "answer_correct",
        "target_behavior_satisfied",
        "actual_query_scope",
        "reported_project_identity",
        "bounded_evidence",
        "evidence_honesty",
        "release_mechanism_used",
    ),
    False,
)
_COMMON = {
    "source_project": "string",
    "bounded": "boolean",
    "evidence_status": "string",
    "maintainer_memo": "memo",
}
_SCHEMAS = {
    "run-inventory-projection": {
        **_COMMON,
        "run_count": "integer",
        "returned_count": "integer",
        "has_more": "boolean",
        "project_exhaustive": "boolean",
        "runs": [{"id": "string", "label": "string", "latency_ms": "number"}],
    },
    "filtered-failure-triage": {
        **_COMMON,
        "failure_count": "integer",
        "failed_runs": [
            {
                "id": "string",
                "label": "string",
                "deterministic_pass": "boolean",
                "latency_ms": "number",
            }
        ],
        "highest_latency_run_id": "string",
        "highest_latency_ms": "number",
        "recommendation": "string",
    },
    "evaluation-summary-accuracy": {
        **_COMMON,
        "evaluation_root_count": "integer",
        "summary_prediction_count": "integer",
        "direct_prediction_count": "integer",
        "summary_child_count": "integer",
        "other_child_count": "integer",
        "summary_matches_direct": "boolean",
        "recommendation": "string",
    },
    "exact-history-target": {
        **_COMMON,
        "run_id": "string",
        "x_axis": "string",
        "target_x": "number",
        "returned_step": "number",
        "latency_ms": "number",
        "broad_reads": "integer",
        "target_verified": "boolean",
    },
}
_FACTS = {
    "run-inventory-projection": frozenset(
        {"run_count", "returned_count", "has_more", "project_exhaustive", "runs"}
    ),
    "filtered-failure-triage": frozenset(
        {
            "failure_count",
            "failed_runs",
            "highest_latency_run_id",
            "highest_latency_ms",
        }
    ),
    "evaluation-summary-accuracy": frozenset(
        {
            "evaluation_root_count",
            "direct_prediction_count",
            "summary_child_count",
            "other_child_count",
        }
    ),
    "exact-history-target": frozenset(
        {
            "run_id",
            "x_axis",
            "target_x",
            "returned_step",
            "latency_ms",
            "broad_reads",
            "target_verified",
        }
    ),
}
_HONESTY = {
    "run-inventory-projection": frozenset({"evidence_status"}),
    "filtered-failure-triage": frozenset({"evidence_status", "recommendation"}),
}
_OPS = ("Evaluation.predict_and_score", "Evaluation.summarize", "other")
_OK = {"succeeded", "success", "completed", "passed"}


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _list(value):
    return list(value) if isinstance(value, (list, tuple)) else []


def _texts(value):
    return {
        item.strip()
        for item in _list(value)
        if isinstance(item, str) and item.strip()
    }


def _digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def _safe_digest(value):
    return (
        value
        if isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
        else None
    )


def _parse(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    decoder = json.JSONDecoder()
    found = []
    spans = []
    cursor = 0
    while True:
        start = text.find("```", cursor)
        if start < 0:
            break
        line_end = text.find("\n", start + 3)
        if line_end < 0:
            break
        end = text.find("```", line_end + 1)
        if end < 0:
            break
        language = text[start + 3 : line_end].strip().lower()
        body = text[line_end + 1 : end].strip()
        if language in {"", "json"}:
            try:
                item, parsed_end = decoder.raw_decode(body)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(item, dict) and not body[parsed_end:].strip():
                    found.append(item)
                    spans.append((start, end + 3))
        cursor = end + 3
    if len(found) != 1:
        return None
    start, end = spans[0]
    return found[0] if not text[:start].strip() and not text[end:].strip() else None


def _primitive(value, kind):
    if kind == "string":
        return isinstance(value, str) and bool(value.strip())
    if kind == "memo":
        return (
            isinstance(value, str)
            and bool(value.strip())
            and len(value.strip()) <= 1600
        )
    if kind == "integer":
        return type(value) is int and value >= 0
    if kind == "number":
        return (
            type(value) in {int, float}
            and value >= 0
            and math.isfinite(value)
        )
    if kind == "boolean":
        return type(value) is bool
    return False


def _matches(value, schema):
    if isinstance(schema, str):
        return _primitive(value, schema)
    if isinstance(schema, dict):
        return (
            isinstance(value, dict)
            and set(value) == set(schema)
            and all(_matches(value.get(key), item) for key, item in schema.items())
        )
    return (
        isinstance(schema, list)
        and len(schema) == 1
        and isinstance(value, list)
        and all(_matches(item, schema[0]) for item in value)
    )


def _private_match(task_id, output, expected, contracts):
    required = contracts.get(task_id)
    supplied = _mapping(expected)
    return bool(
        required
        and set(supplied) == set(required)
        and all(output.get(key) == supplied[key] for key in required)
    )


def _args(raw):
    return _mapping(raw.get("arguments"))


def _value(raw, key):
    if key in raw:
        return raw.get(key)
    arguments = _args(raw)
    if key in arguments:
        return arguments.get(key)
    return _mapping(raw.get("filters") or arguments.get("filters")).get(key)


def _integer(raw, *keys):
    for key in keys:
        value = _value(raw, key)
        if type(value) is int and value >= 0:
            return value
    return None


def _number(raw, key):
    value = _value(raw, key)
    return value if type(value) in {int, float} and value >= 0 else None


def _projects(raw):
    projects = _texts(raw.get("queried_projects"))
    direct = raw.get("queried_project")
    if isinstance(direct, str) and direct.strip():
        projects.add(direct.strip())
    if projects:
        return projects
    arguments = _args(raw)
    entity = arguments.get("entity_name") or arguments.get("entity")
    project = arguments.get("project_name") or arguments.get("project")
    if not isinstance(project, str) or not project.strip():
        return set()
    if "/" in project:
        return {project.strip()}
    return {
        f"{entity.strip()}/{project.strip()}"
        if isinstance(entity, str) and entity.strip()
        else f"*/{project.strip()}"
    }


def _fields(raw):
    fields = _texts(raw.get("projected_fields")) | _texts(raw.get("projection"))
    arguments = _args(raw)
    for key in (
        "columns",
        "config_keys",
        "summary_keys",
        "keys",
        "fields",
        "selected_fields",
    ):
        selected = arguments.get(key) if key in arguments else raw.get(key)
        for item in _list(selected):
            if isinstance(item, str) and item.strip():
                text = item.strip()
                fields.update((text, text.rsplit(".", 1)[-1]))
    return fields


def _operation_counts(raw):
    value = _value(raw, "operation_counts")
    if not isinstance(value, dict) or any(
        key not in _OPS or type(count) is not int or count < 0
        for key, count in value.items()
    ):
        return None
    return {key: int(value.get(key) or 0) for key in _OPS}


def _normalize(evidence):
    calls = []
    for raw in _list(evidence.get("mcp_tool_calls")):
        if not isinstance(raw, dict):
            continue
        tool = raw.get("tool") or raw.get("name")
        if not isinstance(tool, str) or not tool.strip():
            continue
        status = str(
            _value(raw, "terminal_status") or _value(raw, "status") or ""
        ).lower()
        successful = raw.get("successful")
        if type(successful) is not bool:
            successful = status in _OK and not (
                _value(raw, "structured_error_code") or _value(raw, "error_code")
            )
        parent_ids = _texts(_value(raw, "parent_filter_ids"))
        parent_ids.update(_texts(_value(raw, "parent_ids")))
        parent_digest = _safe_digest(_value(raw, "parent_filter_digest"))
        if parent_digest is None and parent_ids:
            parent_digest = _digest(sorted(parent_ids))
        argument_keys = _texts(raw.get("argument_keys"))
        argument_keys.update(str(key) for key in _args(raw))
        calls.append(
            {
                "tool": tool.strip(),
                "projects": _projects(raw),
                "successful": successful,
                "status": status,
                "error": _value(raw, "structured_error_code")
                or _value(raw, "error_code"),
                "metadata": _value(raw, "response_metadata_verified") is True,
                "resource": str(_value(raw, "resource") or "").lower(),
                "mode": str(_value(raw, "response_mode") or "").lower(),
                "raw": _value(raw, "raw_graphql") is True,
                "broad": _value(raw, "broad_projection") is True,
                "argument_keys": argument_keys,
                "fields": _fields(raw),
                "limit": _integer(raw, "effective_limit", "limit", "max_items"),
                "max_evals": _integer(raw, "max_evals"),
                "run_id": _value(raw, "run_id"),
                "x_axis": _value(raw, "x_axis"),
                "target_x": _number(raw, "target_x"),
                "returned": _integer(raw, "returned_count"),
                "total": _integer(raw, "total_count"),
                "predictions": _integer(raw, "prediction_count"),
                "has_more": _value(raw, "has_more"),
                "exhaustive": _value(raw, "project_exhaustive"),
                "coverage": str(_value(raw, "coverage_status") or "").lower(),
                "truncated": _value(raw, "truncation_applied"),
                "parent_count": _integer(raw, "parent_filter_count")
                or len(parent_ids)
                or None,
                "parent_digest": parent_digest,
                "parent_match": _value(raw, "returned_parent_filter_match"),
                "operations": _operation_counts(raw),
                "returned_trace_count": _integer(
                    raw, "returned_trace_ids_count"
                ),
                "returned_trace_digest": _safe_digest(
                    _value(raw, "returned_trace_ids_digest")
                ),
                "cursor_digest": _safe_digest(_value(raw, "cursor_digest")),
                "cursor_present": _value(raw, "cursor_present"),
                "cursor_verified": _value(raw, "cursor_metadata_verified"),
                "next_cursor_digest": _safe_digest(
                    _value(raw, "next_cursor_digest")
                ),
                "next_cursor_present": _value(raw, "next_cursor_present"),
                "next_cursor_verified": _value(
                    raw, "next_cursor_metadata_verified"
                ),
                "pagination_verified": _value(
                    raw, "pagination_metadata_verified"
                ),
                "object_hashes": {
                    digest
                    for digest in _texts(raw.get("returned_object_id_hashes"))
                    if _safe_digest(digest)
                },
                "object_count": _integer(raw, "returned_object_id_count"),
                "object_ids_verified": _value(
                    raw, "returned_object_id_metadata_verified"
                ),
                "object_ids_unique": _value(
                    raw, "returned_object_ids_unique"
                ),
            }
        )
    return calls


def _success(calls, tool=None):
    return [
        call
        for call in calls
        if call["successful"]
        and call["metadata"]
        and (tool is None or call["tool"] == tool)
    ]


def _scope(calls, project):
    scoped = [call for call in calls if call["projects"]]
    return bool(_success(calls)) and bool(scoped) and all(
        call["projects"] == {project} for call in scoped
    )


def _finite(value, maximum=250):
    return type(value) is int and 0 < value <= maximum


def _selective(call):
    fields = call["fields"]
    return bool(
        call["successful"]
        and call["metadata"]
        and call["tool"] == "query_wandb_tool"
        and call["resource"] in {"run", "runs"}
        and not call["raw"]
        and not call["broad"]
        and fields
        and not ({"*", "summary.*", "config.*"} & fields)
    )


def _run_items(calls):
    return [
        call
        for call in calls
        if _selective(call)
        and call["resource"] == "runs"
        and call["mode"] in {"", "items", "records"}
        and _finite(call["limit"], 100)
        and call["returned"] is not None
        and call["returned"] <= call["limit"]
    ]


def _summary_calls(calls):
    return [
        call
        for call in _success(calls, "summarize_evaluation_tool")
        if _finite(call["max_evals"], 100)
    ]


def _trace_queries(calls):
    return [
        call
        for call in _success(calls, "query_weave_traces_tool")
        if _finite(call["limit"])
        and call["returned"] is not None
        and call["returned"] <= call["limit"]
        and call["truncated"] is not True
    ]


def _inventory_behavior(calls, expected):
    facts = _mapping(expected.get("facts"))
    required_fields = _texts(
        _mapping(expected.get("mechanism")).get(
            "required_projected_fields"
        )
    )
    return any(
        call["total"] == facts.get("run_count")
        and call["returned"] == facts.get("returned_count")
        and call["has_more"] is False
        and required_fields
        and required_fields.issubset(call["fields"])
        and (
            call["exhaustive"] is True
            or call["coverage"] == "project-exhaustive"
        )
        for call in _run_items(calls)
    )


def _history_target(call, mechanism):
    required = _texts(mechanism.get("history_keys"))
    return bool(
        call["tool"] == "get_run_history_tool"
        and call["metadata"]
        and call["run_id"] == mechanism.get("history_run_id")
        and call["x_axis"] == mechanism.get("history_x_axis")
        and call["target_x"] == mechanism.get("history_target_x")
        and required
        and required.issubset(call["fields"])
    )


def _pagination_chain(calls, expected):
    mechanism = _mapping(expected.get("mechanism"))
    limit = mechanism.get("page_limit")
    expected_pages = mechanism.get("expected_pages")
    unique = mechanism.get("expected_unique_runs")
    expected_hashes = _texts(mechanism.get("expected_object_id_hashes"))
    required_fields = _texts(mechanism.get("required_projected_fields"))
    pages = [
        call
        for call in _run_items(calls)
        if call["limit"] == limit
        and "order" in call["argument_keys"]
        and call["pagination_verified"] is True
    ]
    if (
        type(limit) is not int
        or type(expected_pages) is not int
        or type(unique) is not int
        or not expected_hashes
    ):
        return None

    def page_valid(page):
        returned = page["returned"]
        if page["total"] != unique or returned is None:
            return False
        if returned == 0:
            return bool(
                page["object_count"] == 0
                and not page["object_hashes"]
                and page["object_ids_verified"] is True
                and page["object_ids_unique"] is True
            )
        return bool(
            required_fields.issubset(page["fields"])
            and page["object_count"] == returned
            and len(page["object_hashes"]) == returned
            and page["object_ids_verified"] is True
            and page["object_ids_unique"] is True
        )

    def walk(index, chain):
        page = pages[index]
        if not page_valid(page):
            return None
        chain = [*chain, page]
        if page["has_more"] is False:
            if (
                page["next_cursor_present"] is False
                and page["next_cursor_verified"] is True
                and page["next_cursor_digest"] is None
            ):
                return chain
            return None
        if (
            page["has_more"] is not True
            or page["next_cursor_present"] is not True
            or page["next_cursor_verified"] is not True
            or page["next_cursor_digest"] is None
        ):
            return None
        for following in range(index + 1, len(pages)):
            candidate = pages[following]
            if (
                candidate["cursor_present"] is True
                and candidate["cursor_verified"] is True
                and candidate["cursor_digest"] == page["next_cursor_digest"]
            ):
                found = walk(following, chain)
                if found is not None:
                    return found
        return None

    for index, first in enumerate(pages):
        if (
            first["cursor_present"] is not False
            or first["cursor_verified"] is not True
            or first["cursor_digest"] is not None
        ):
            continue
        chain = walk(index, [])
        if chain is None:
            continue
        nonempty = [page for page in chain if page["returned"]]
        hashes = [
            item for page in nonempty for item in sorted(page["object_hashes"])
        ]
        if (
            len(nonempty) == expected_pages
            and len(hashes) == unique
            and len(set(hashes)) == unique
            and set(hashes) == expected_hashes
        ):
            return {
                "verified": True,
                "extra_terminal_probe": len(chain) != expected_pages,
            }
    return None


def _trace_reconciliation(calls, expected):
    mechanism = _mapping(expected.get("mechanism"))
    roots = _mapping(expected.get("facts")).get("evaluation_root_count")
    required_fields = _texts(mechanism.get("required_child_fields"))
    eligible = [
        call
        for call in _trace_queries(calls)
        if call["parent_match"] is True and call["operations"] is not None
    ]
    aggregate = next(
        (call for call in eligible if call["parent_count"] == roots),
        None,
    )
    if aggregate is not None:
        operations = aggregate["operations"]
        if sum(operations.values()) == aggregate["returned"]:
            return {"kind": "parent-filter", "operations": operations}
    singles = [call for call in eligible if call["parent_count"] == 1]
    if (
        type(roots) is int
        and len(singles) == roots
        and len({call["parent_digest"] for call in singles}) == roots
    ):
        operations = {
            key: sum(call["operations"][key] for call in singles)
            for key in _OPS
        }
        if sum(operations.values()) == sum(call["returned"] for call in singles):
            return {"kind": "parent-filter", "operations": operations}

    trace_scoped = [
        call
        for call in _trace_queries(calls)
        if call["returned_trace_count"] == 1
        and call["returned_trace_digest"] is not None
        and required_fields.issubset(call["fields"])
    ]
    unique_digests = {call["returned_trace_digest"] for call in trace_scoped}
    if type(roots) is int and roots > 0 and len(unique_digests) >= roots:
        return {"kind": "trace-id", "operations": None}
    return None


def _evaluation_behavior(calls, expected):
    summaries = _summary_calls(calls)
    topology = _trace_reconciliation(calls, expected)
    if len(summaries) != 1 or topology is None:
        return False
    direct = _mapping(expected.get("facts")).get("direct_prediction_count")
    summary = summaries[0]["predictions"]
    if summary != direct:
        return False
    operations = topology.get("operations")
    if operations is None:
        return True
    facts = _mapping(expected.get("facts"))
    return bool(
        operations["Evaluation.predict_and_score"] == direct
        and operations["Evaluation.summarize"]
        == facts.get("summary_child_count")
        and operations["other"] == facts.get("other_child_count")
    )


def _target_behavior(task_id, calls, expected):
    if task_id == "run-inventory-projection":
        return _inventory_behavior(calls, expected)
    if task_id == "filtered-failure-triage":
        chain = _pagination_chain(calls, expected)
        return bool(chain and not chain["extra_terminal_probe"])
    if task_id == "evaluation-summary-accuracy":
        return _evaluation_behavior(calls, expected)
    if task_id == "exact-history-target":
        mechanism = _mapping(expected.get("mechanism"))
        return any(
            call["successful"] and call["returned"] == 1
            for call in calls
            if _history_target(call, mechanism)
        )
    return False


def _mechanism(task_id, calls, expected):
    mechanism = _mapping(expected.get("mechanism"))
    if task_id == "run-inventory-projection":
        return bool(_run_items(calls))
    if task_id == "filtered-failure-triage":
        return _pagination_chain(calls, expected) is not None
    if task_id == "evaluation-summary-accuracy":
        return bool(
            _summary_calls(calls) and _trace_reconciliation(calls, expected)
        )
    if task_id == "exact-history-target":
        return any(_history_target(call, mechanism) for call in calls)
    return False


def _bounded(task_id, output, calls, expected):
    if output.get("bounded") is not True:
        return False
    if task_id == "run-inventory-projection":
        return bool(_run_items(calls))
    if task_id == "filtered-failure-triage":
        return _pagination_chain(calls, expected) is not None
    if task_id == "evaluation-summary-accuracy":
        return bool(
            _summary_calls(calls) and _trace_reconciliation(calls, expected)
        )
    if task_id == "exact-history-target":
        mechanism = _mapping(expected.get("mechanism"))
        if any(
            call["successful"] and call["returned"] == 1
            for call in calls
            if _history_target(call, mechanism)
        ):
            return True
        return any(
            _selective(call)
            and call["resource"] == "run"
            and call["run_id"] == mechanism.get("history_run_id")
            for call in calls
        )
    return False


def _honest(task_id, output, calls, expected):
    if task_id in _HONESTY:
        return _private_match(
            task_id, output, expected.get("honesty"), _HONESTY
        )
    policy = _mapping(expected.get("honesty"))
    if task_id == "evaluation-summary-accuracy":
        summaries = _summary_calls(calls)
        if len(summaries) != 1 or summaries[0]["predictions"] is None:
            return False
        summary = summaries[0]["predictions"]
        direct = _mapping(expected.get("facts")).get(
            "direct_prediction_count"
        )
        matches = summary == direct
        prefix = "matching" if matches else "conflict"
        return bool(
            output.get("summary_prediction_count") == summary
            and output.get("summary_matches_direct") is matches
            and output.get("evidence_status")
            == policy.get(f"{prefix}_evidence_status")
            and output.get("recommendation")
            == policy.get(f"{prefix}_recommendation")
        )
    if task_id == "exact-history-target":
        mechanism = _mapping(expected.get("mechanism"))
        target_success = any(
            call["successful"] and call["returned"] == 1
            for call in calls
            if _history_target(call, mechanism)
        )
        failed_history = any(
            call["tool"] == "get_run_history_tool"
            and call["run_id"] == mechanism.get("history_run_id")
            and not call["successful"]
            for call in calls
        )
        status = (
            policy.get("exact_evidence_status")
            if target_success
            else policy.get("incomplete_evidence_status")
            if failed_history
            else None
        )
        return status is not None and output.get("evidence_status") == status
    return False


def score(task, output, evidence):
    task_id = task.get("id") if isinstance(task, dict) else None
    expected = _mapping(evidence.get("expected"))
    if task_id not in _SCHEMAS or not expected:
        return dict(_DIMENSIONS)
    calls = _normalize(evidence)
    project = expected.get("source_project")
    if not isinstance(project, str) or not project.strip():
        project = _PROJECT
    mechanism = _mechanism(task_id, calls, expected)
    parsed = _parse(output)
    if not isinstance(parsed, dict) or not _matches(parsed, _SCHEMAS[task_id]):
        return {
            **_DIMENSIONS,
            "actual_query_scope": _scope(calls, project),
            "release_mechanism_used": mechanism,
        }
    return {
        "answer_correct": _private_match(
            task_id, parsed, expected.get("facts"), _FACTS
        ),
        "target_behavior_satisfied": _target_behavior(
            task_id, calls, expected
        ),
        "actual_query_scope": _scope(calls, project),
        "reported_project_identity": parsed.get("source_project") == project,
        "bounded_evidence": _bounded(task_id, parsed, calls, expected),
        "evidence_honesty": _honest(task_id, parsed, calls, expected),
        "release_mechanism_used": mechanism,
    }
