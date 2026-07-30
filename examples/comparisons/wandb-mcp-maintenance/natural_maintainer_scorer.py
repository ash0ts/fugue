import hashlib
import json

_SOURCE_PROJECT = "wandb/fugue-mcp-release-source-v2"
_DIMENSIONS = {
    "answer_correct": False,
    "actual_query_scope": False,
    "reported_project_identity": False,
    "bounded_evidence": False,
    "evidence_honesty": False,
    "release_mechanism_used": False,
}
_WEAVE_OPERATION_CLASSES = ("Evaluation.predict_and_score", "Evaluation.summarize", "other")
_RECONCILIATION_FACTS = frozenset(
    "evaluation_root_count direct_child_count summary_reported_predictions "
    "direct_prediction_rows summary_children summary_matches_direct".split()
)
_HISTORY_FACTS = frozenset({"run_id", "step", "latency_ms", "broad_reads"})
_FACT_KEYS = {
    "maintainer-evaluation-reconciliation": _RECONCILIATION_FACTS,
    "maintainer-evaluation-trace-topology": _RECONCILIATION_FACTS,
    "maintainer-project-health": frozenset(
        "run_count evaluation_root_count largest_latency_run largest_latency_ms "
        "missing_cost_run".split()
    ),
    "maintainer-source-inventory": frozenset(
        "run_count evaluation_root_count summary_reported_predictions".split()
    ),
    "maintainer-history-hotspot": _HISTORY_FACTS,
    "maintainer-run-triage": _HISTORY_FACTS,
    "maintainer-artifact-provenance": frozenset(
        "collection listed_version artifact_a artifact_b artifact_a_digest "
        "artifact_b_digest digests_differ".split()
    ),
}

_COMMON_SCHEMA = {
    "source_project": "string",
    "bounded": "boolean",
    "evidence_status": "string",
    "maintainer_memo": "memo",
}
_RECONCILIATION_SCHEMA = {
    **_COMMON_SCHEMA,
    "evaluation_root_count": "integer",
    "direct_child_count": "integer",
    "summary_reported_predictions": "integer",
    "direct_prediction_rows": "integer",
    "summary_children": "integer",
    "summary_matches_direct": "boolean",
    "unrelated_children_excluded": "boolean",
    "recommendation": "string",
}
_HISTORY_SCHEMA = {
    **_COMMON_SCHEMA,
    "run_id": "string",
    "step": "integer",
    "latency_ms": "integer",
    "broad_reads": "integer",
    "observed_cost_status": "string",
    "causal_scope": "string",
}
_SCHEMAS = {
    "maintainer-evaluation-reconciliation": _RECONCILIATION_SCHEMA,
    "maintainer-evaluation-trace-topology": _RECONCILIATION_SCHEMA,
    "maintainer-project-health": {
        **_COMMON_SCHEMA,
        "run_count": "integer",
        "evaluation_root_count": "integer",
        "coverage": "string",
        "largest_latency_run": "string",
        "largest_latency_ms": "integer",
        "missing_cost_run": "string",
        "recommendation": "string",
        "causal_scope": "string",
    },
    "maintainer-source-inventory": {
        **_COMMON_SCHEMA,
        "run_count": "integer",
        "evaluation_root_count": "integer",
        "summary_reported_predictions": "integer",
    },
    "maintainer-history-hotspot": _HISTORY_SCHEMA,
    "maintainer-run-triage": _HISTORY_SCHEMA,
    "maintainer-artifact-provenance": {
        **_COMMON_SCHEMA,
        "collection": "string",
        "listed_version": "string",
        "artifact_a": "string",
        "artifact_b": "string",
        "artifact_a_digest": "string",
        "artifact_b_digest": "string",
        "digests_differ": "boolean",
    },
}


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
        str(item).strip()
        for item in _list(value)
        if isinstance(item, str) and item.strip()
    }


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


def _matches_type(value, kind):
    if kind == "string":
        return isinstance(value, str) and bool(value.strip())
    if kind == "memo":
        return (
            isinstance(value, str)
            and bool(value.strip())
            and len(value.strip()) <= 1200
        )
    if kind == "integer":
        return type(value) is int and value >= 0
    if kind == "boolean":
        return type(value) is bool
    return False


def _matches_schema(task_id, output):
    schema = _SCHEMAS.get(task_id)
    return bool(
        schema
        and set(output) == set(schema)
        and all(_matches_type(output.get(key), kind) for key, kind in schema.items())
    )


def _contains(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                any(_contains(candidate, item) for candidate in actual)
                for item in expected
            )
        )
    return actual == expected


def _task_facts(task_id, expected):
    allowed = _FACT_KEYS.get(task_id, frozenset())
    return {
        key: value
        for key, value in _mapping(expected.get("facts")).items()
        if key in allowed
    }


def _stable_digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


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


def _projects(raw):
    projects = _text_set(raw.get("queried_projects"))
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
    raw_fields = _text_set(raw.get("projected_fields"))
    raw_fields.update(_text_set(raw.get("projection")))
    arguments = _arguments(raw)
    for key in ("columns", "config_keys", "summary_keys", "keys"):
        for item in _list(arguments.get(key) if key in arguments else raw.get(key)):
            if isinstance(item, str) and item.strip():
                raw_fields.add(item.strip())
    return {field.rsplit(".", 1)[-1] for field in raw_fields}


def _operation_counts(raw):
    value = _value(raw, "operation_counts")
    if not isinstance(value, dict) or not value:
        return None
    if any(
        key not in _WEAVE_OPERATION_CLASSES or type(count) is not int or count < 0
        for key, count in value.items()
    ):
        return None
    normalized = {
        operation: int(value.get(operation) or 0)
        for operation in _WEAVE_OPERATION_CLASSES
    }
    return normalized if sum(normalized.values()) > 0 else None


def _normalize_calls(evidence):
    calls = []
    for raw in _list(evidence.get("mcp_tool_calls")):
        if not isinstance(raw, dict):
            continue
        tool = raw.get("tool") or raw.get("name")
        if not isinstance(tool, str) or not tool.strip():
            continue
        parent_ids = _text_set(_value(raw, "parent_filter_ids")) or _text_set(
            _value(raw, "parent_ids")
        )
        parent_filter_digest = _value(raw, "parent_filter_digest")
        if not (
            isinstance(parent_filter_digest, str) and len(parent_filter_digest) == 64
        ):
            parent_filter_digest = (
                _stable_digest(sorted(parent_ids)) if parent_ids else None
            )
        calls.append(
            {
                "tool": tool.strip(),
                "projects": _projects(raw),
                "resource": _value(raw, "resource"),
                "response_mode": _value(raw, "response_mode"),
                "fields": _fields(raw),
                "limit": _integer(raw, "limit", "max_items"),
                "max_evals": _integer(raw, "max_evals"),
                "samples": _integer(raw, "samples", "sample_size"),
                "sample_runs": _integer(raw, "sample_runs"),
                "top_n_values": _integer(raw, "top_n_values"),
                "run_id": _value(raw, "run_id"),
                "run_id_a": _value(raw, "run_id_a"),
                "run_id_b": _value(raw, "run_id_b"),
                "x_axis": _value(raw, "x_axis"),
                "target_x": _value(raw, "target_x"),
                "trace_roots_only": _value(raw, "trace_roots_only"),
                "trace_ids_digest": _value(raw, "trace_ids_digest"),
                "trace_ids_count": _integer(raw, "trace_ids_count"),
                "returned_trace_ids_digest": _value(
                    raw,
                    "returned_trace_ids_digest",
                ),
                "returned_trace_ids_count": _integer(
                    raw,
                    "returned_trace_ids_count",
                ),
                "artifact_name": _value(raw, "artifact_name"),
                "artifact_name_a": _value(raw, "artifact_name_a"),
                "artifact_name_b": _value(raw, "artifact_name_b"),
                "collection_name": _value(raw, "collection_name"),
                "include_file_diff": _value(raw, "include_file_diff"),
                "include_files": _value(raw, "include_files"),
                "include_history_overlap": _value(
                    raw,
                    "include_history_overlap",
                ),
                "parent_filter_digest": parent_filter_digest,
                "parent_filter_count": (
                    _integer(raw, "parent_filter_count") or len(parent_ids) or None
                ),
                "returned_count": _integer(raw, "returned_count", "response_count"),
                "total_count": _integer(raw, "total_count"),
                "prediction_count": _integer(
                    raw,
                    "prediction_count",
                    "predictions",
                    "prediction_rows",
                ),
                "operation_counts": _operation_counts(raw),
                "artifact_digest": _value(raw, "artifact_digest"),
                "digest_match": _value(raw, "digest_match"),
                "returned_parent_filter_match": _value(
                    raw,
                    "returned_parent_filter_match",
                ),
                "has_more": _value(raw, "has_more"),
                "project_exhaustive": _value(raw, "project_exhaustive"),
                "truncation_applied": _value(raw, "truncation_applied"),
                "structured_error": (
                    _value(raw, "structured_error_code") or _value(raw, "error_code")
                ),
                "status": (_value(raw, "terminal_status") or _value(raw, "status")),
            }
        )
    return calls


def _successful(call):
    return not call.get("structured_error") and call.get("status") in {
        "succeeded",
        "success",
        "completed",
        "passed",
    }


def _scope_is_locked(evidence, calls, project):
    declared_projects = _text_set(evidence.get("mcp_queried_projects"))
    declared_projects.update(_text_set(evidence.get("queried_projects")))
    successful_calls = [call for call in calls if _successful(call)]
    return (
        bool(successful_calls)
        and (not declared_projects or declared_projects == {project})
        and all(call["projects"] == {project} for call in successful_calls)
    )


def _calls(calls, tool):
    return [call for call in calls if _successful(call) and call["tool"] == tool]


def _run_calls(calls, response_mode):
    return [
        call
        for call in _calls(calls, "query_wandb_tool")
        if str(call.get("resource") or "").lower() in {"run", "runs"}
        and str(call.get("response_mode") or "").lower() == response_mode
    ]


def _bounded_run_items(calls, required_fields=()):
    item_calls = _run_calls(calls, "items")
    return len(item_calls) == 1 and (
        item_calls[0]["limit"] is not None
        and 0 < item_calls[0]["limit"] <= 50
        and item_calls[0]["returned_count"] is not None
        and item_calls[0]["returned_count"] <= item_calls[0]["limit"]
        and item_calls[0]["has_more"] is False
        and item_calls[0]["project_exhaustive"] is True
        and item_calls[0]["truncation_applied"] is False
        and item_calls[0]["fields"] == set(required_fields)
    )


def _run_count(calls, expected):
    count_calls = _run_calls(calls, "count")
    return len(count_calls) == 1 and count_calls[0]["total_count"] == expected


def _has_one_run_count(calls):
    return len(_run_calls(calls, "count")) == 1


def _evaluation_summary(calls):
    summaries = _calls(calls, "summarize_evaluation_tool")
    if (
        len(summaries) != 1
        or summaries[0]["max_evals"] is None
        or not 0 < summaries[0]["max_evals"] <= 25
    ):
        return []
    return summaries


def _evaluation_child_queries(calls, expected):
    parents = set(expected.get("evaluation_parent_ids") or ())
    required_fields = set(
        _mapping(expected.get("mechanism")).get("required_child_fields") or ()
    )
    if len(parents) != 2:
        return None
    if not required_fields:
        return None
    child_calls = _calls(calls, "query_weave_traces_tool")
    if len(child_calls) != len(parents):
        return None
    digest_to_parent = {_stable_digest([parent]): parent for parent in sorted(parents)}
    result = {}
    for call in child_calls:
        if call["parent_filter_count"] != 1:
            return None
        parent = digest_to_parent.get(call["parent_filter_digest"])
        if parent not in parents or parent in result:
            return None
        if (
            call["limit"] is None
            or not 0 < call["limit"] <= 20
            or call["returned_count"] is None
            or call["returned_count"] > call["limit"]
            or call["truncation_applied"] is not False
            or call["fields"] != required_fields
            or call["returned_parent_filter_match"] is not True
        ):
            return None
        result[parent] = call
    return result if set(result) == parents else None


def _evaluation_children(calls, expected):
    parents = set(expected.get("evaluation_parent_ids") or ())
    raw_expected_counts = _mapping(expected.get("mechanism")).get(
        "evaluation_parent_operation_counts"
    )
    if len(parents) != 2 or not isinstance(raw_expected_counts, dict):
        return None
    expected_counts = {}
    for parent, value in raw_expected_counts.items():
        normalized = _operation_counts({"operation_counts": value})
        if not isinstance(parent, str) or not parent.strip() or normalized is None:
            return None
        expected_counts[parent.strip()] = normalized
    if set(expected_counts) != parents:
        return None

    child_queries = _evaluation_child_queries(calls, expected)
    if child_queries is None:
        return None
    result = {}
    for parent, call in child_queries.items():
        operation_counts = call["operation_counts"]
        if (
            operation_counts is None
            or sum(operation_counts.values()) != call["returned_count"]
            or operation_counts != expected_counts[parent]
        ):
            return None
        result[parent] = operation_counts
    return result if set(result) == parents else None


def _history_call(calls, expected):
    mechanism = _mapping(expected.get("mechanism"))
    required = set(mechanism.get("history_keys") or ())
    history_calls = _calls(calls, "get_run_history_tool")
    if len(history_calls) != 1:
        return None
    call = history_calls[0]
    return (
        call
        if call["run_id"] == mechanism.get("history_run_id")
        and call["x_axis"] == mechanism.get("history_x_axis")
        and call["target_x"] == mechanism.get("history_target_x")
        and call["fields"] == required
        and call["returned_count"] == 1
        and call["truncation_applied"] is False
        else None
    )


def _probe_project_call(calls, expected):
    mechanism = _mapping(expected.get("mechanism"))
    facts = _mapping(expected.get("facts"))
    probes = _calls(calls, "probe_project_tool")
    if len(probes) != 1:
        return None
    probe = probes[0]
    return (
        probe
        if probe["sample_runs"] == mechanism.get("probe_sample_runs")
        and probe["total_count"] == facts.get("run_count")
        else None
    )


def _trace_topology_calls(calls, expected):
    mechanism = _mapping(expected.get("mechanism"))
    counts = _calls(calls, "count_weave_traces_tool")
    schemas = _calls(calls, "infer_trace_schema_tool")
    roots = _calls(calls, "resolve_trace_roots_tool")
    child_queries = _evaluation_child_queries(calls, expected)
    if (
        len(counts) != 1
        or len(schemas) != 1
        or len(roots) != 1
        or child_queries is None
    ):
        return None
    count = counts[0]
    schema = schemas[0]
    root = roots[0]
    returned_trace_digests = {
        call["returned_trace_ids_digest"]
        for call in child_queries.values()
        if call["returned_trace_ids_count"] == 1
        and isinstance(call["returned_trace_ids_digest"], str)
    }
    if len(returned_trace_digests) != len(child_queries):
        return None
    if (
        count["trace_roots_only"] is not True
        or count["total_count"] != mechanism.get("root_trace_count")
        or schema["samples"] != mechanism.get("schema_sample_size")
        or schema["top_n_values"] != mechanism.get("schema_top_n_values")
        or root["trace_ids_count"] != 1
        or root["trace_ids_digest"] not in returned_trace_digests
        or root["returned_count"] != 1
        or root["total_count"] != 1
    ):
        return None
    return {
        "count": count,
        "schema": schema,
        "root": root,
    }


def _run_triage_calls(calls, expected):
    mechanism = _mapping(expected.get("mechanism"))
    facts = _mapping(expected.get("facts"))
    comparisons = _calls(calls, "compare_runs_tool")
    diagnoses = _calls(calls, "diagnose_run_tool")
    if len(comparisons) != 1 or len(diagnoses) != 1:
        return None
    comparison = comparisons[0]
    diagnosis = diagnoses[0]
    expected_run = facts.get("run_id")
    expected_peer = mechanism.get("comparison_peer_run_id")
    if (
        {comparison["run_id_a"], comparison["run_id_b"]}
        != {expected_run, expected_peer}
        or comparison["include_history_overlap"] is not False
        or diagnosis["run_id"] != expected_run
    ):
        return None
    return {
        "comparison": comparison,
        "diagnosis": diagnosis,
    }


def _artifact_calls(calls, expected):
    facts = _mapping(expected.get("facts"))
    versions = _calls(calls, "list_artifact_versions_tool")
    details = _calls(calls, "get_artifact_details_tool")
    comparisons = _calls(calls, "compare_artifact_versions_tool")
    if len(versions) != 1 or len(details) != 2 or len(comparisons) != 1:
        return None
    version = versions[0]
    details_by_artifact = {
        detail["artifact_name"]: detail
        for detail in details
        if isinstance(detail["artifact_name"], str)
    }
    if set(details_by_artifact) != {
        facts.get("artifact_a"),
        facts.get("artifact_b"),
    }:
        return None
    detail_a = details_by_artifact[facts.get("artifact_a")]
    detail_b = details_by_artifact[facts.get("artifact_b")]
    comparison = comparisons[0]
    if (
        version["collection_name"] != facts.get("collection")
        or version["limit"] != 10
        or version["returned_count"] is None
        or not 1 <= version["returned_count"] <= 10
        or detail_a["include_files"] is not False
        or detail_a["artifact_digest"] != facts.get("artifact_a_digest")
        or detail_b["include_files"] is not False
        or detail_b["artifact_digest"] != facts.get("artifact_b_digest")
        or comparison["artifact_name_a"] != facts.get("artifact_a")
        or comparison["artifact_name_b"] != facts.get("artifact_b")
        or comparison["include_file_diff"] is not True
        or comparison["limit"] != 10
        or comparison["digest_match"] is not False
    ):
        return None
    return {
        "versions": version,
        "details": details_by_artifact,
        "comparison": comparison,
    }


def _reconciliation(task_output, calls, expected, *, task_id):
    summaries = _evaluation_summary(calls)
    child_queries = _evaluation_child_queries(calls, expected)
    locked_root_count = len(set(expected.get("evaluation_parent_ids") or ()))
    summary_root_count = summaries[0]["total_count"] if len(summaries) == 1 else None
    bounded = (
        task_output.get("bounded") is True
        and len(summaries) == 1
        and locked_root_count == 2
        and summary_root_count == locked_root_count
        and child_queries is not None
        and (
            task_id != "maintainer-evaluation-trace-topology"
            or _trace_topology_calls(calls, expected) is not None
        )
    )
    children = _evaluation_children(calls, expected)
    if len(summaries) != 1 or children is None:
        return False, bounded, False
    summary = summaries[0]
    observed_summary = summary["prediction_count"]
    if observed_summary is None:
        observed_summary = summary["total_count"]
    direct_predictions = sum(
        counts["Evaluation.predict_and_score"] for counts in children.values()
    )
    summary_children = sum(
        counts["Evaluation.summarize"] for counts in children.values()
    )
    unrelated_children = sum(counts["other"] for counts in children.values())
    direct_children = sum(sum(counts.values()) for counts in children.values())
    summary_matches = observed_summary == direct_predictions
    locked_roots_match = summary_root_count == locked_root_count == len(children)
    reconciled = summary_matches and locked_roots_match
    facts = _task_facts(task_id, expected)
    factual = (
        _contains(task_output, facts)
        and task_output.get("evaluation_root_count") == len(children)
        and locked_roots_match
        and task_output.get("direct_child_count") == direct_children
        and task_output.get("summary_reported_predictions") == observed_summary
        and task_output.get("direct_prediction_rows") == direct_predictions
        and task_output.get("summary_children") == summary_children
        and task_output.get("summary_matches_direct") is summary_matches
    )
    recommendation = (
        "advance-to-qualification" if reconciled else "block-and-investigate"
    )
    honest = (
        task_output.get("recommendation") == recommendation
        and task_output.get("unrelated_children_excluded") is (unrelated_children == 0)
        and task_output.get("evidence_status")
        == ("reconciled" if reconciled else "conflicted")
    )
    return factual, bounded, honest


def _source_inventory(task_output, calls, expected):
    facts = _task_facts("maintainer-source-inventory", expected)
    summaries = _evaluation_summary(calls)
    observed_summary = summaries[0]["prediction_count"] if len(summaries) == 1 else None
    factual = (
        _contains(task_output, facts)
        and _run_count(calls, int(facts.get("run_count") or 0))
        and len(summaries) == 1
        and summaries[0]["total_count"] == facts.get("evaluation_root_count")
        and observed_summary == facts.get("summary_reported_predictions")
        and task_output.get("summary_reported_predictions") == observed_summary
        and _probe_project_call(calls, expected) is not None
    )
    bounded = (
        task_output.get("bounded") is True
        and _bounded_run_items(calls, ("id", "attempt_label", "latency_ms"))
        and _has_one_run_count(calls)
        and len(summaries) == 1
        and _probe_project_call(calls, expected) is not None
    )
    honest = task_output.get("evidence_status") == "summary-only"
    return factual, bounded, honest


def _history(task_output, calls, expected, *, task_id):
    facts = _task_facts(task_id, expected)
    history = _history_call(calls, expected)
    factual = _contains(task_output, facts)
    if task_id == "maintainer-run-triage":
        factual = factual and _run_triage_calls(calls, expected) is not None
    bounded = (
        task_output.get("bounded") is True
        and _bounded_run_items(calls, ("id", "attempt_label", "latency_ms"))
        and history is not None
    )
    honest = (
        task_output.get("observed_cost_status") == "unavailable"
        and task_output.get("causal_scope") == "observed-mechanism-only"
        and task_output.get("evidence_status") == "reconciled"
    )
    return factual, bounded, honest


def _artifact_provenance(task_output, calls, expected):
    facts = _task_facts("maintainer-artifact-provenance", expected)
    artifacts = _artifact_calls(calls, expected)
    factual = _contains(task_output, facts) and artifacts is not None
    bounded = task_output.get("bounded") is True and artifacts is not None
    honest = (
        task_output.get("evidence_status") == "reconciled"
        and task_output.get("digests_differ") is True
    )
    return factual, bounded, honest


def _project_health(task_output, calls, expected):
    facts = _task_facts("maintainer-project-health", expected)
    summaries = _evaluation_summary(calls)
    history = _history_call(calls, expected)
    factual = (
        _contains(task_output, facts)
        and _run_count(calls, int(facts.get("run_count") or 0))
        and len(summaries) == 1
        and summaries[0]["total_count"] == facts.get("evaluation_root_count")
    )
    bounded = (
        task_output.get("bounded") is True
        and _has_one_run_count(calls)
        and _bounded_run_items(calls, ("id", "attempt_label", "latency_ms"))
        and len(summaries) == 1
        and history is not None
    )
    honest = (
        task_output.get("coverage") == "locked-cohort-exhaustive"
        and task_output.get("recommendation") == "investigate"
        and task_output.get("causal_scope") == "observed-mechanism-only"
        and task_output.get("evidence_status") == "reconciled"
    )
    return factual, bounded, honest


def _evaluate(task_id, output, calls, expected):
    if task_id == "maintainer-evaluation-reconciliation":
        return _reconciliation(output, calls, expected, task_id=task_id)
    if task_id == "maintainer-evaluation-trace-topology":
        return _reconciliation(output, calls, expected, task_id=task_id)
    if task_id == "maintainer-source-inventory":
        return _source_inventory(output, calls, expected)
    if task_id == "maintainer-history-hotspot":
        return _history(output, calls, expected, task_id=task_id)
    if task_id == "maintainer-run-triage":
        return _history(output, calls, expected, task_id=task_id)
    if task_id == "maintainer-artifact-provenance":
        return _artifact_provenance(output, calls, expected)
    if task_id == "maintainer-project-health":
        return _project_health(output, calls, expected)
    return False, False, False


def _required_call_counts(calls, expected, include_failed):
    required = _mapping(expected.get("mechanism")).get("required_tool_counts")
    if not isinstance(required, dict) or not required:
        return None
    observed = calls if include_failed else [call for call in calls if _successful(call)]
    if any(
        not isinstance(tool, str)
        or type(count) is not int
        or count < 1
        or len([call for call in observed if call["tool"] == tool]) != count
        for tool, count in required.items()
    ):
        return None
    if len(observed) != sum(required.values()) or {
        call["tool"] for call in observed
    } != set(required):
        return None
    return True


def _mechanism_used(task_id, calls, expected):
    if _required_call_counts(calls, expected, True) is None:
        return False
    if task_id == "maintainer-evaluation-reconciliation":
        return (
            len(_evaluation_summary(calls)) == 1
            and _evaluation_child_queries(calls, expected) is not None
        )
    if task_id == "maintainer-evaluation-trace-topology":
        return (
            len(_evaluation_summary(calls)) == 1
            and _evaluation_child_queries(calls, expected) is not None
            and _trace_topology_calls(calls, expected) is not None
        )
    if task_id == "maintainer-source-inventory":
        return (
            _has_one_run_count(calls)
            and _bounded_run_items(calls, ("id", "attempt_label", "latency_ms"))
            and len(_evaluation_summary(calls)) == 1
            and _probe_project_call(calls, expected) is not None
        )
    if task_id == "maintainer-history-hotspot":
        return (
            _bounded_run_items(calls, ("id", "attempt_label", "latency_ms"))
            and _history_call(calls, expected) is not None
        )
    if task_id == "maintainer-run-triage":
        return (
            _bounded_run_items(calls, ("id", "attempt_label", "latency_ms"))
            and _history_call(calls, expected) is not None
            and _run_triage_calls(calls, expected) is not None
        )
    if task_id == "maintainer-artifact-provenance":
        return _artifact_calls(calls, expected) is not None
    if task_id == "maintainer-project-health":
        return (
            _has_one_run_count(calls)
            and _bounded_run_items(calls, ("id", "attempt_label", "latency_ms"))
            and len(_evaluation_summary(calls)) == 1
            and _history_call(calls, expected) is not None
        )
    return False


def score(task, output, evidence):
    task_id = task.get("id") if isinstance(task, dict) else None
    expected = _mapping(evidence.get("expected"))
    parsed = _parse_one_json(output)
    if not isinstance(task_id, str) or not expected:
        return dict(_DIMENSIONS)
    calls = _normalize_calls(evidence)
    project = str(expected.get("source_project") or _SOURCE_PROJECT)
    locked_scope = _scope_is_locked(evidence, calls, project)
    mechanism = _mechanism_used(task_id, calls, expected)
    if not isinstance(parsed, dict) or not _matches_schema(task_id, parsed):
        return {
            **_DIMENSIONS,
            "actual_query_scope": locked_scope,
            "release_mechanism_used": mechanism,
        }
    factual, bounded, honest = _evaluate(task_id, parsed, calls, expected)
    return {
        "answer_correct": factual,
        "actual_query_scope": locked_scope,
        "reported_project_identity": (parsed.get("source_project") == project),
        "bounded_evidence": (
            bounded and _required_call_counts(calls, expected, False) is not None
        ),
        "evidence_honesty": honest,
        "release_mechanism_used": mechanism,
    }
