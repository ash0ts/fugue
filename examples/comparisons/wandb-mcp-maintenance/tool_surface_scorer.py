import hashlib
import json
import math

_PROJECT = "wandb/fugue-mcp-release-source-v2"
_DIMENSIONS = dict.fromkeys(("answer_correct", "actual_query_scope", "reported_project_identity", "bounded_evidence", "evidence_honesty", "release_mechanism_used"), False)

_COMMON_SCHEMA = dict(source_project="string", bounded="boolean", evidence_status="string", maintainer_memo="memo")
_SCHEMAS = {
    "run-inventory-projection": {**_COMMON_SCHEMA, "run_count": "integer", "returned_count": "integer", "has_more": "boolean", "project_exhaustive": "boolean", "runs": [{"id": "string", "label": "string", "latency_ms": "number"}]},
    "filtered-failure-triage": {**_COMMON_SCHEMA, "failure_count": "integer", "failed_runs": [{"id": "string", "label": "string", "deterministic_pass": "boolean", "latency_ms": "number"}], "highest_latency_run_id": "string", "highest_latency_ms": "number", "recommendation": "string"},
    "evaluation-summary-accuracy": {**_COMMON_SCHEMA, "evaluation_root_count": "integer", "summary_prediction_count": "integer", "direct_prediction_count": "integer", "summary_child_count": "integer", "other_child_count": "integer", "summary_matches_direct": "boolean", "recommendation": "string"},
    "exact-history-target": {**_COMMON_SCHEMA, "run_id": "string", "x_axis": "string", "target_x": "number", "returned_step": "number", "latency_ms": "number", "broad_reads": "integer", "target_verified": "boolean"},
    "selective-run-comparison": {**_COMMON_SCHEMA, "run_id_a": "string", "run_id_b": "string", "higher_latency_run_id": "string", "latency_delta_ms": "number", "config_differences": "string_list", "diagnosis": "string", "causal_scope": "string"},
    "artifact-provenance": {**_COMMON_SCHEMA, "collection": "string", "artifact_a": "string", "artifact_b": "string", "artifact_a_digest": "string", "artifact_b_digest": "string", "digests_differ": "boolean", "changed_files": "string_list"},
    "trace-source-use": {**_COMMON_SCHEMA, "trace_root_count": "integer", "selected_trace_id": "string", "source_returned": "integer", "source_opened": "integer", "source_ids": "string_list", "conclusion_scope": "string"},
    "missing-cost-honesty": {**_COMMON_SCHEMA, "run_id": "string", "latency_ms": "number", "cost_status": "string", "observed_cost_usd": "nullable_number", "causal_claim_supported": "boolean", "recommendation": "string"},
}

_FACT_KEYS = {
    "run-inventory-projection": frozenset({"run_count", "returned_count", "has_more", "project_exhaustive", "runs"}),
    "filtered-failure-triage": frozenset({"failure_count", "failed_runs", "highest_latency_run_id", "highest_latency_ms"}),
    "evaluation-summary-accuracy": frozenset({"evaluation_root_count", "direct_prediction_count", "summary_child_count", "other_child_count"}),
    "exact-history-target": frozenset({"run_id", "x_axis", "target_x", "returned_step", "latency_ms", "broad_reads", "target_verified"}),
    "selective-run-comparison": frozenset({"run_id_a", "run_id_b", "higher_latency_run_id", "latency_delta_ms", "config_differences", "diagnosis"}),
    "artifact-provenance": frozenset({"collection", "artifact_a", "artifact_b", "artifact_a_digest", "artifact_b_digest", "digests_differ", "changed_files"}),
    "trace-source-use": frozenset({"trace_root_count", "selected_trace_id", "source_returned", "source_opened", "source_ids"}),
    "missing-cost-honesty": frozenset({"run_id", "latency_ms"}),
}

_HONESTY_KEYS = {
    "run-inventory-projection": frozenset({"evidence_status"}),
    "filtered-failure-triage": frozenset({"evidence_status", "recommendation"}),
    "exact-history-target": frozenset({"evidence_status"}),
    "selective-run-comparison": frozenset({"evidence_status", "causal_scope"}),
    "artifact-provenance": frozenset({"evidence_status"}),
    "trace-source-use": frozenset({"evidence_status", "conclusion_scope"}),
    "missing-cost-honesty": frozenset({"cost_status", "observed_cost_usd", "causal_claim_supported", "evidence_status", "recommendation"}),
}

_OPS = ("Evaluation.predict_and_score", "Evaluation.summarize", "other")
_OK = {"succeeded", "success", "completed", "passed"}
_SINGLE_TOOLS = {"compare_runs_tool", "diagnose_run_tool", "get_artifact_details_tool", "probe_project_tool"}


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _text_set(value):
    return {
        item.strip()
        for item in _list(value)
        if isinstance(item, str) and item.strip()
    }


def _stable_digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _parse_one_json(value):
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
    fenced = []
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
                if not body[parsed_end:].strip() and isinstance(item, dict):
                    fenced.append(item)
                    spans.append((start, end + 3))
        cursor = end + 3
    if len(fenced) != 1:
        return None
    start, end = spans[0]
    if text[:start].strip() or text[end:].strip():
        return None
    return fenced[0]


def _matches_primitive(value, kind):
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
    if kind == "nullable_number":
        return value is None or _matches_primitive(value, "number")
    if kind == "boolean":
        return type(value) is bool
    if kind == "string_list":
        return isinstance(value, list) and all(
            isinstance(item, str) and bool(item.strip()) for item in value
        )
    return False


def _matches_value(value, schema):
    if isinstance(schema, str):
        return _matches_primitive(value, schema)
    if isinstance(schema, dict):
        return (
            isinstance(value, dict)
            and set(value) == set(schema)
            and all(
                _matches_value(value.get(key), child)
                for key, child in schema.items()
            )
        )
    if isinstance(schema, list) and len(schema) == 1:
        return isinstance(value, list) and all(
            _matches_value(item, schema[0]) for item in value
        )
    return False


def _matches_schema(task_id, output):
    schema = _SCHEMAS.get(task_id)
    return bool(schema and _matches_value(output, schema))


def _private_values_match(task_id, output, expected, key_group):
    required = key_group.get(task_id)
    supplied = _mapping(expected)
    return bool(
        required
        and set(supplied) == set(required)
        and all(output.get(key) == supplied[key] for key in required)
    )


def _arguments(raw):
    return _mapping(raw.get("arguments"))


def _value(raw, key):
    if key in raw:
        return raw.get(key)
    arguments = _arguments(raw)
    if key in arguments:
        return arguments.get(key)
    filters = _mapping(raw.get("filters") or arguments.get("filters"))
    return filters.get(key)


def _integer(raw, *keys):
    for key in keys:
        value = _value(raw, key)
        if type(value) is int and value >= 0:
            return value
    return None


def _number(raw, *keys):
    for key in keys:
        value = _value(raw, key)
        if type(value) in {int, float} and value >= 0:
            return value
    return None


def _projects(raw):
    projects = _text_set(raw.get("queried_projects"))
    queried_project = raw.get("queried_project")
    if isinstance(queried_project, str) and queried_project.strip():
        projects.add(queried_project.strip())
    if projects:
        return projects
    arguments = _arguments(raw)
    entity = arguments.get("entity_name") or arguments.get("entity")
    project = arguments.get("project_name") or arguments.get("project")
    if isinstance(project, str) and project.strip():
        if "/" in project:
            projects.add(project.strip())
        elif isinstance(entity, str) and entity.strip():
            projects.add(f"{entity.strip()}/{project.strip()}")
        else:
            projects.add(f"*/{project.strip()}")
    return projects


def _fields(raw):
    fields = _text_set(raw.get("projected_fields"))
    fields.update(_text_set(raw.get("projection")))
    arguments = _arguments(raw)
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
                value = item.strip()
                fields.add(value)
                fields.add(value.rsplit(".", 1)[-1])
    return fields


def _operation_counts(raw):
    value = _value(raw, "operation_counts")
    if not isinstance(value, dict) or not value:
        return None
    if any(
        key not in _OPS
        or type(count) is not int
        or count < 0
        for key, count in value.items()
    ):
        return None
    normalized = {
        operation: int(value.get(operation) or 0)
        for operation in _OPS
    }
    return normalized


def _operation_filters(raw):
    value = _value(raw, "op_name_filter")
    if not isinstance(value, dict):
        return set()
    result = set()
    for entries in value.values():
        result.update(_text_set(entries))
    return result


def _normalize_calls(evidence):
    calls = []
    for raw in _list(evidence.get("mcp_tool_calls")):
        if not isinstance(raw, dict):
            continue
        tool = raw.get("tool") or raw.get("name")
        if not isinstance(tool, str) or not tool.strip():
            continue
        parent_ids = _text_set(_value(raw, "parent_filter_ids"))
        parent_ids.update(_text_set(_value(raw, "parent_ids")))
        parent_filter_digest = _value(raw, "parent_filter_digest")
        if not (
            isinstance(parent_filter_digest, str)
            and len(parent_filter_digest) == 64
        ):
            parent_filter_digest = (
                _stable_digest(sorted(parent_ids)) if parent_ids else None
            )
        status = (
            _value(raw, "terminal_status")
            or _value(raw, "status")
            or ""
        )
        argument_keys = _text_set(raw.get("argument_keys"))
        argument_keys.update(str(key) for key in _arguments(raw))
        successful = raw.get("successful")
        if type(successful) is not bool:
            successful = (
                str(status).lower() in _OK
                and not (
                    _value(raw, "structured_error_code")
                    or _value(raw, "error_code")
                )
            )
        calls.append(
            {
                "tool": tool.strip(),
                "projects": _projects(raw),
                "resource": str(_value(raw, "resource") or "").lower(),
                "response_mode": str(
                    _value(raw, "response_mode") or ""
                ).lower(),
                "raw_graphql": _value(raw, "raw_graphql") is True,
                "argument_keys": argument_keys,
                "cursor": _value(raw, "request_cursor")
                if _value(raw, "request_cursor") is not None
                else _value(raw, "cursor"),
                "next_cursor": _value(raw, "next_cursor"),
                "returned_item_ids": _text_set(
                    raw.get("returned_item_ids")
                ),
                "fields": _fields(raw),
                "limit": _integer(
                    raw,
                    "effective_limit",
                    "limit",
                    "max_items",
                ),
                "max_evals": _integer(raw, "max_evals"),
                "samples": _integer(raw, "samples", "sample_size"),
                "top_n_values": _integer(raw, "top_n_values"),
                "run_id": _value(raw, "run_id"),
                "run_id_a": _value(raw, "run_id_a"),
                "run_id_b": _value(raw, "run_id_b"),
                "x_axis": _value(raw, "x_axis"),
                "target_x": _number(raw, "target_x"),
                "trace_roots_only": _value(raw, "trace_roots_only"),
                "trace_ids_count": _integer(raw, "trace_ids_count"),
                "parent_filter_digest": parent_filter_digest,
                "parent_filter_count": (
                    _integer(raw, "parent_filter_count")
                    or len(parent_ids)
                    or None
                ),
                "operation_filters": _operation_filters(raw),
                "collection_name": _value(raw, "collection_name"),
                "artifact_name": _value(raw, "artifact_name"),
                "artifact_name_a": _value(raw, "artifact_name_a"),
                "artifact_name_b": _value(raw, "artifact_name_b"),
                "include_files": _value(raw, "include_files"),
                "include_file_diff": _value(raw, "include_file_diff"),
                "include_history_overlap": _value(
                    raw,
                    "include_history_overlap",
                ),
                "returned_count": _integer(
                    raw,
                    "returned_count",
                    "response_count",
                ),
                "total_count": _integer(raw, "total_count"),
                "prediction_count": _integer(
                    raw,
                    "prediction_count",
                    "predictions",
                    "prediction_rows",
                ),
                "operation_counts": _operation_counts(raw),
                "returned_parent_filter_match": _value(
                    raw,
                    "returned_parent_filter_match",
                ),
                "returned_trace_ids_count": _integer(
                    raw,
                    "returned_trace_ids_count",
                ),
                "has_more": _value(raw, "has_more"),
                "project_exhaustive": _value(
                    raw,
                    "project_exhaustive",
                ),
                "truncation_applied": _value(
                    raw,
                    "truncation_applied",
                ),
                "response_metadata_verified": raw.get(
                    "response_metadata_verified"
                ),
                "successful": successful,
                "status": str(status).lower(),
            }
        )
    return calls


def _successful(call):
    return call.get("successful") is True


def _successful_calls(calls, tool=None):
    return [
        call
        for call in calls
        if _successful(call)
        and (tool is None or call.get("tool") == tool)
    ]


def _scope_is_locked(calls, project):
    successful = _successful_calls(calls)
    scoped = [call for call in calls if call.get("projects")]
    return bool(successful) and bool(scoped) and all(
        call.get("projects") == {project} for call in scoped
    )


def _limit_is_bounded(value, maximum):
    return type(value) is int and 0 < value <= maximum


def _projection_matches(call, required):
    required = {
        str(field).strip()
        for field in _list(required)
        if isinstance(field, str) and field.strip()
    }
    fields = set(call.get("fields") or ())
    if not required or not required.issubset(fields):
        return False
    if {"config.*", "summary.*", "*"} & fields:
        return False
    return len(fields) <= (2 * len(required)) + 2


def _run_queries(calls, mode):
    return [
        call
        for call in _successful_calls(calls, "query_wandb_tool")
        if call.get("resource") in {"run", "runs"}
        and call.get("response_mode") == mode
    ]


def _bounded_run_items(calls, mechanism):
    required = _list(mechanism.get("required_projected_fields"))
    return any(
        _limit_is_bounded(call.get("limit"), 100)
        and call.get("returned_count") is not None
        and call["returned_count"] <= call["limit"]
        and _projection_matches(call, required)
        for call in _run_queries(calls, "items")
    )


def _structured_run_items(calls, mechanism):
    required_keys = {
        str(key).strip()
        for key in _list(mechanism.get("required_argument_keys"))
        if isinstance(key, str) and key.strip()
    }
    return any(
        _limit_is_bounded(call.get("limit"), 100)
        and call.get("raw_graphql") is not True
        and _projection_matches(
            call,
            mechanism.get("required_projected_fields"),
        )
        and required_keys
        and required_keys.issubset(call.get("argument_keys") or ())
        for call in _run_queries(calls, "items")
    )


def _complete_run_page_chain(calls, mechanism):
    page_limit = mechanism.get("page_limit")
    expected_pages = mechanism.get("expected_pages")
    expected_unique_runs = mechanism.get("expected_unique_runs")
    if (
        type(page_limit) is not int
        or page_limit < 1
        or type(expected_pages) is not int
        or expected_pages < 1
        or type(expected_unique_runs) is not int
        or expected_unique_runs < 1
    ):
        return False
    pages = _run_queries(calls, "items")
    if len(pages) != expected_pages:
        return False

    observed_ids = []
    previous_next_cursor = None
    for index, page in enumerate(pages):
        item_ids = set(page.get("returned_item_ids") or ())
        if (
            page.get("limit") != page_limit
            or page.get("returned_count") != len(item_ids)
            or not item_ids
            or "cursor" not in (page.get("argument_keys") or ())
        ):
            return False
        if index == 0:
            if page.get("cursor") not in {None, ""}:
                return False
        elif page.get("cursor") != previous_next_cursor:
            return False
        observed_ids.extend(item_ids)
        previous_next_cursor = page.get("next_cursor")
        if index < expected_pages - 1:
            if (
                not isinstance(previous_next_cursor, str)
                or not previous_next_cursor
                or page.get("has_more") is not True
            ):
                return False
        elif (
            previous_next_cursor not in {None, ""}
            or page.get("has_more") is not False
        ):
            return False
    return (
        len(observed_ids) == expected_unique_runs
        and len(set(observed_ids)) == expected_unique_runs
    )


def _bounded_summary(calls):
    return any(
        _limit_is_bounded(call.get("max_evals"), 100)
        for call in _successful_calls(calls, "summarize_evaluation_tool")
    )


def _bounded_trace_query(calls, mechanism):
    required = _list(mechanism.get("required_child_fields"))
    return any(
        _limit_is_bounded(call.get("limit"), 100)
        and call.get("returned_count") is not None
        and call["returned_count"] <= call["limit"]
        and (not required or _projection_matches(call, required))
        for call in _successful_calls(calls, "query_weave_traces_tool")
    )


def _bounded_history(calls):
    return any(
        call.get("target_x") is not None
        and call.get("returned_count") == 1
        for call in _successful_calls(calls, "get_run_history_tool")
    )


def _bounded_artifacts(calls):
    listed = _successful_calls(calls, "list_artifact_versions_tool")
    compared = _successful_calls(calls, "compare_artifact_versions_tool")
    details = _successful_calls(calls, "get_artifact_details_tool")
    return bool(
        any(
            _limit_is_bounded(call.get("limit"), 100)
            and call.get("returned_count") is not None
            and call["returned_count"] <= call["limit"]
            for call in listed
        )
        and details
        and all(call.get("include_files") is False for call in details)
        and any(
            _limit_is_bounded(call.get("limit"), 100) for call in compared
        )
    )


def _call_is_bounded(call):
    if call.get("response_metadata_verified") is False:
        return False
    tool = call.get("tool")
    if tool == "query_wandb_tool":
        return call.get("response_mode") == "count" or _limit_is_bounded(
            call.get("limit"), 100
        )
    if tool == "summarize_evaluation_tool":
        return _limit_is_bounded(call.get("max_evals"), 100)
    if tool == "query_weave_traces_tool":
        return _limit_is_bounded(call.get("limit"), 100)
    if tool == "infer_trace_schema_tool":
        return _limit_is_bounded(call.get("samples"), 100) and (
            call.get("top_n_values") is None
            or _limit_is_bounded(call.get("top_n_values"), 20)
        )
    if tool == "resolve_trace_roots_tool":
        return _limit_is_bounded(call.get("trace_ids_count"), 100)
    if tool in {
        "list_artifact_versions_tool",
        "compare_artifact_versions_tool",
    }:
        return _limit_is_bounded(call.get("limit"), 100)
    if tool == "get_run_history_tool":
        return call.get("target_x") is not None or _limit_is_bounded(
            call.get("limit"), 1000
        )
    if tool == "count_weave_traces_tool":
        return call.get("trace_roots_only") in {True, False}
    if tool in _SINGLE_TOOLS:
        return True
    return _limit_is_bounded(call.get("limit"), 100)


def _reasonable_call_budget(calls, mechanism):
    successful = _successful_calls(calls)
    failed = [call for call in calls if not _successful(call)]
    max_successful = mechanism.get("max_successful_calls", 12)
    max_failed = mechanism.get("max_failed_calls", 4)
    if (
        type(max_successful) is not int
        or type(max_failed) is not int
        or max_successful < 1
        or max_failed < 0
    ):
        return False
    return (
        0 < len(successful) <= max_successful
        and len(failed) <= max_failed
        and all(_call_is_bounded(call) for call in successful)
    )


def _required_tools_used(calls, mechanism):
    required = {
        str(tool).strip()
        for tool in _list(mechanism.get("required_tools"))
        if isinstance(tool, str) and tool.strip()
    }
    observed = {call["tool"] for call in _successful_calls(calls)}
    return bool(required) and required.issubset(observed)


def _history_mechanism(calls, mechanism):
    required_fields = {
        str(field).strip()
        for field in _list(mechanism.get("history_keys"))
        if isinstance(field, str) and field.strip()
    }
    return any(
        call.get("run_id") == mechanism.get("history_run_id")
        and call.get("x_axis") == mechanism.get("history_x_axis")
        and call.get("target_x") == mechanism.get("history_target_x")
        and required_fields.issubset(call.get("fields") or ())
        and call.get("returned_count") == 1
        for call in _successful_calls(calls, "get_run_history_tool")
    )


def _evaluation_mechanism(calls, mechanism):
    parents = {
        str(item).strip()
        for item in _list(mechanism.get("evaluation_parent_ids"))
        if isinstance(item, str) and item.strip()
    }
    if not parents or not _bounded_summary(calls):
        return False
    expected_digests = {_stable_digest([parent]) for parent in parents}
    observed_digests = {
        call.get("parent_filter_digest")
        for call in _successful_calls(calls, "query_weave_traces_tool")
        if call.get("parent_filter_count") == 1
        and call.get("returned_parent_filter_match") is True
    }
    return expected_digests.issubset(observed_digests)


def _evaluation_honesty(output, calls, expected):
    facts = _mapping(expected.get("facts"))
    policy = _mapping(expected.get("honesty"))
    counts = {
        call.get("prediction_count")
        for call in _successful_calls(calls, "summarize_evaluation_tool")
        if type(call.get("prediction_count")) is int
    }
    if len(counts) != 1:
        return False
    summary = counts.pop()
    matches = summary == facts.get("direct_prediction_count")
    prefix = "matching" if matches else "conflict"
    return bool(
        output.get("summary_prediction_count") == summary
        and output.get("summary_matches_direct") is matches
        and output.get("evidence_status")
        == policy.get(f"{prefix}_evidence_status")
        and output.get("recommendation")
        == policy.get(f"{prefix}_recommendation")
    )


def _comparison_mechanism(calls, mechanism):
    expected_runs = {
        str(item).strip()
        for item in _list(mechanism.get("comparison_run_ids"))
        if isinstance(item, str) and item.strip()
    }
    comparisons = _successful_calls(calls, "compare_runs_tool")
    required_fields = mechanism.get("required_comparison_fields")
    include_history = mechanism.get("comparison_include_history_overlap")
    if type(include_history) is not bool:
        return False
    return bool(
        len(expected_runs) == 2
        and any(
            {call.get("run_id_a"), call.get("run_id_b")} == expected_runs
            and call.get("include_history_overlap") is include_history
            and _projection_matches(call, required_fields)
            for call in comparisons
        )
    )


def _artifact_mechanism(calls, mechanism):
    expected_names = {
        str(item).strip()
        for item in _list(mechanism.get("artifact_names"))
        if isinstance(item, str) and item.strip()
    }
    listed = _successful_calls(calls, "list_artifact_versions_tool")
    details = _successful_calls(calls, "get_artifact_details_tool")
    compared = _successful_calls(calls, "compare_artifact_versions_tool")
    return bool(
        len(expected_names) == 2
        and any(
            call.get("collection_name") == mechanism.get("collection_name")
            for call in listed
        )
        and expected_names.issubset(
            {
                str(call.get("artifact_name"))
                for call in details
                if call.get("include_files") is False
            }
        )
        and any(
            {
                str(call.get("artifact_name_a")),
                str(call.get("artifact_name_b")),
            }
            == expected_names
            and call.get("include_file_diff") is True
            for call in compared
        )
    )


def _trace_mechanism(calls, mechanism):
    required_fields = _list(mechanism.get("required_trace_fields"))
    counts = _successful_calls(calls, "count_weave_traces_tool")
    roots = _successful_calls(calls, "resolve_trace_roots_tool")
    return bool(
        any(call.get("trace_roots_only") is True for call in counts)
        and _bounded_trace_query(
            calls,
            {"required_child_fields": required_fields},
        )
        and any(
            _limit_is_bounded(call.get("trace_ids_count"), 100)
            for call in roots
        )
    )


def _mechanism_used(task_id, calls, expected):
    mechanism = _mapping(expected.get("mechanism"))
    if not _required_tools_used(calls, mechanism):
        return False
    if task_id == "run-inventory-projection":
        return bool(
            _run_queries(calls, "count")
            and _bounded_run_items(calls, mechanism)
            and any(
                call.get("raw_graphql") is not True
                for call in _run_queries(calls, "items")
            )
        )
    if task_id == "filtered-failure-triage":
        return bool(
            _bounded_run_items(calls, mechanism)
            and _structured_run_items(calls, mechanism)
            and _complete_run_page_chain(calls, mechanism)
        )
    if task_id == "evaluation-summary-accuracy":
        return _evaluation_mechanism(calls, mechanism)
    if task_id == "exact-history-target":
        return _history_mechanism(calls, mechanism)
    if task_id == "missing-cost-honesty":
        return _structured_run_items(calls, mechanism)
    if task_id == "selective-run-comparison":
        return _comparison_mechanism(calls, mechanism)
    if task_id == "artifact-provenance":
        return _artifact_mechanism(calls, mechanism)
    if task_id == "trace-source-use":
        return _trace_mechanism(calls, mechanism)
    return False


def _bounded_evidence(task_id, output, calls, expected):
    if output.get("bounded") is not True:
        return False
    mechanism = _mapping(expected.get("mechanism"))
    if not _reasonable_call_budget(calls, mechanism):
        return False
    if task_id in {
        "run-inventory-projection",
        "filtered-failure-triage",
        "missing-cost-honesty",
    }:
        return _bounded_run_items(calls, mechanism)
    if task_id == "evaluation-summary-accuracy":
        return _bounded_summary(calls) and _bounded_trace_query(
            calls, mechanism
        )
    if task_id == "exact-history-target":
        return _bounded_history(calls)
    if task_id == "selective-run-comparison":
        return bool(_successful_calls(calls, "compare_runs_tool"))
    if task_id == "artifact-provenance":
        return _bounded_artifacts(calls)
    if task_id == "trace-source-use":
        return _bounded_trace_query(
            calls,
            {
                "required_child_fields": mechanism.get(
                    "required_trace_fields", []
                )
            },
        )
    return False


def score(task, output, evidence):
    task_id = task.get("id") if isinstance(task, dict) else None
    expected = _mapping(evidence.get("expected"))
    if not isinstance(task_id, str) or task_id not in _SCHEMAS or not expected:
        return dict(_DIMENSIONS)

    calls = _normalize_calls(evidence)
    project = expected.get("source_project")
    if not isinstance(project, str) or not project.strip():
        project = _PROJECT
    locked_scope = _scope_is_locked(calls, project)
    mechanism_used = _mechanism_used(task_id, calls, expected)
    parsed = _parse_one_json(output)
    if not isinstance(parsed, dict) or not _matches_schema(task_id, parsed):
        return {
            **_DIMENSIONS,
            "actual_query_scope": locked_scope,
            "release_mechanism_used": mechanism_used,
        }

    return {
        "answer_correct": _private_values_match(
            task_id,
            parsed,
            expected.get("facts"),
            _FACT_KEYS,
        ),
        "actual_query_scope": locked_scope,
        "reported_project_identity": (
            parsed.get("source_project") == project
        ),
        "bounded_evidence": _bounded_evidence(
            task_id,
            parsed,
            calls,
            expected,
        ),
        "evidence_honesty": (
            _evaluation_honesty(parsed, calls, expected)
            if task_id == "evaluation-summary-accuracy"
            else _private_values_match(
                task_id,
                parsed,
                expected.get("honesty"),
                _HONESTY_KEYS,
            )
        ),
        "release_mechanism_used": mechanism_used,
    }
