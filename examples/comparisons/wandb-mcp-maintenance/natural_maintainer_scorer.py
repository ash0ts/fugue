from __future__ import annotations

import json

_SOURCE_PROJECT = "wandb/fugue-mcp-release-source-v1"
_DIMENSIONS = {
    "answer_correct": False,
    "locked_source_scope": False,
    "bounded_evidence": False,
    "evidence_honesty": False,
    "release_mechanism_used": False,
}

_SCHEMAS = {
    "maintainer-evaluation-reconciliation": {
        "source_project": "string",
        "evaluation_root_count": "integer",
        "direct_child_count": "integer",
        "summary_reported_predictions": "integer",
        "direct_prediction_rows": "integer",
        "summary_children": "integer",
        "summary_matches_direct": "boolean",
        "unrelated_children_excluded": "boolean",
        "recommendation": "string",
        "bounded": "boolean",
        "evidence_status": "string",
        "maintainer_memo": "memo",
    },
    "maintainer-project-health": {
        "source_project": "string",
        "run_count": "integer",
        "evaluation_root_count": "integer",
        "coverage": "string",
        "largest_latency_run": "string",
        "largest_latency_ms": "integer",
        "missing_cost_run": "string",
        "recommendation": "string",
        "causal_scope": "string",
        "bounded": "boolean",
        "evidence_status": "string",
        "maintainer_memo": "memo",
    },
    "maintainer-source-inventory": {
        "source_project": "string",
        "run_count": "integer",
        "evaluation_root_count": "integer",
        "direct_prediction_rows": "integer",
        "bounded": "boolean",
        "evidence_status": "string",
        "maintainer_memo": "memo",
    },
    "maintainer-history-hotspot": {
        "source_project": "string",
        "run_id": "string",
        "step": "integer",
        "latency_ms": "integer",
        "broad_reads": "integer",
        "observed_cost_status": "string",
        "causal_scope": "string",
        "bounded": "boolean",
        "evidence_status": "string",
        "maintainer_memo": "memo",
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
    fields = _text_set(raw.get("projected_fields"))
    fields.update(_text_set(raw.get("projection")))
    arguments = _arguments(raw)
    for key in ("columns", "config_keys", "summary_keys", "keys"):
        for item in _list(
            arguments.get(key) if key in arguments else raw.get(key)
        ):
            if isinstance(item, str) and item.strip():
                field = item.strip()
                fields.add(field)
                fields.add(field.rsplit(".", 1)[-1])
    return fields


def _normalize_calls(evidence):
    calls = []
    for raw in _list(evidence.get("mcp_tool_calls")):
        if not isinstance(raw, dict):
            continue
        tool = raw.get("tool") or raw.get("name")
        if not isinstance(tool, str) or not tool.strip():
            continue
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
                "run_id": _value(raw, "run_id"),
                "x_axis": _value(raw, "x_axis"),
                "target_x": _value(raw, "target_x"),
                "parent_ids": (
                    _text_set(_value(raw, "parent_filter_ids"))
                    or _text_set(_value(raw, "parent_ids"))
                ),
                "returned_count": _integer(
                    raw, "returned_count", "response_count"
                ),
                "total_count": _integer(raw, "total_count"),
                "prediction_count": _integer(
                    raw,
                    "prediction_count",
                    "predictions",
                    "prediction_rows",
                ),
                "has_more": _value(raw, "has_more"),
                "project_exhaustive": _value(raw, "project_exhaustive"),
                "truncation_applied": _value(raw, "truncation_applied"),
                "structured_error": (
                    _value(raw, "structured_error_code")
                    or _value(raw, "error_code")
                ),
                "status": (
                    _value(raw, "terminal_status")
                    or _value(raw, "status")
                ),
            }
        )
    return calls


def _successful(call):
    return (
        not call.get("structured_error")
        and call.get("status")
        in {"succeeded", "success", "completed", "passed"}
    )


def _scope_is_locked(evidence, calls, project):
    projects = _text_set(evidence.get("mcp_queried_projects"))
    projects.update(_text_set(evidence.get("queried_projects")))
    for call in calls:
        projects.update(call["projects"])
    return bool(projects) and projects == {project}


def _calls(calls, tool):
    return [
        call for call in calls if _successful(call) and call["tool"] == tool
    ]


def _run_calls(calls, response_mode):
    return [
        call
        for call in _calls(calls, "query_wandb_tool")
        if str(call.get("resource") or "").lower() in {"run", "runs"}
        and str(call.get("response_mode") or "").lower() == response_mode
    ]


def _bounded_run_items(calls, required_fields=()):
    return any(
        call["limit"] is not None
        and 0 < call["limit"] <= 50
        and call["returned_count"] is not None
        and call["returned_count"] <= call["limit"]
        and call["has_more"] is False
        and call["project_exhaustive"] is True
        and call["truncation_applied"] is False
        and set(required_fields) <= call["fields"]
        and "*" not in call["fields"]
        for call in _run_calls(calls, "items")
    )


def _run_count(calls, expected):
    return any(
        call["total_count"] == expected
        for call in _run_calls(calls, "count")
    )


def _evaluation_summary(calls):
    return [
        call
        for call in _calls(calls, "summarize_evaluation_tool")
        if call["max_evals"] is not None and 0 < call["max_evals"] <= 25
    ]


def _evaluation_children(calls, expected):
    parents = set(expected.get("evaluation_parent_ids") or ())
    if parents and len(parents) != 2:
        return None
    result = {}
    for call in _calls(calls, "query_weave_traces_tool"):
        if len(call["parent_ids"]) != 1:
            continue
        parent = next(iter(call["parent_ids"]))
        if (parents and parent not in parents) or parent in result:
            continue
        if (
            call["limit"] is None
            or not 0 < call["limit"] <= 20
            or call["returned_count"] is None
            or call["returned_count"] > call["limit"]
            or call["truncation_applied"] is not False
            or not {"id", "op_name", "parent_id"} <= call["fields"]
        ):
            return None
        result[parent] = call["returned_count"]
    if parents:
        return result if set(result) == parents else None
    return result if len(result) == 2 else None


def _history_call(calls, expected):
    mechanism = _mapping(expected.get("mechanism"))
    required = set(mechanism.get("history_keys") or ())
    matches = [
        call
        for call in _calls(calls, "get_run_history_tool")
        if call["run_id"] == mechanism.get("history_run_id")
        and call["x_axis"] == mechanism.get("history_x_axis")
        and call["target_x"] == mechanism.get("history_target_x")
        and required <= call["fields"]
        and call["returned_count"] == 1
        and call["truncation_applied"] is False
    ]
    return matches[0] if len(matches) == 1 else None


def _reconciliation(task_output, calls, expected):
    summaries = _evaluation_summary(calls)
    children = _evaluation_children(calls, expected)
    if len(summaries) != 1 or children is None:
        return False, False, False
    summary = summaries[0]
    observed_summary = summary["prediction_count"]
    if observed_summary is None:
        observed_summary = summary["total_count"]
    direct_predictions = int(
        _mapping(expected.get("facts")).get("direct_prediction_rows") or 0
    )
    direct_children = sum(children.values())
    summary_matches = observed_summary == direct_predictions
    factual = (
        _contains(task_output, expected.get("facts") or {})
        and task_output.get("summary_reported_predictions") == observed_summary
        and task_output.get("summary_matches_direct") is summary_matches
    )
    recommendation = (
        "advance-to-qualification"
        if summary_matches
        else "block-and-investigate"
    )
    honest = (
        task_output.get("recommendation") == recommendation
        and task_output.get("direct_child_count") == direct_children
        and task_output.get("unrelated_children_excluded") is True
        and task_output.get("evidence_status") == "reconciled"
    )
    bounded = (
        task_output.get("bounded") is True
        and direct_children == 18
        and all(count == 9 for count in children.values())
    )
    return factual, bounded, honest


def _source_inventory(task_output, calls, expected):
    facts = _mapping(expected.get("facts"))
    summaries = _evaluation_summary(calls)
    factual = (
        _contains(task_output, facts)
        and _run_count(calls, int(facts.get("run_count") or 0))
        and len(summaries) == 1
        and summaries[0]["total_count"] == facts.get("evaluation_root_count")
        and summaries[0]["prediction_count"]
        == facts.get("direct_prediction_rows")
    )
    bounded = (
        task_output.get("bounded") is True
        and _bounded_run_items(calls, ("id", "attempt_label", "latency_ms"))
    )
    honest = (
        factual
        and task_output.get("evidence_status") == "reconciled"
    )
    return factual, bounded, honest


def _history(task_output, calls, expected):
    facts = _mapping(expected.get("facts"))
    history = _history_call(calls, expected)
    factual = _contains(task_output, facts)
    bounded = (
        task_output.get("bounded") is True
        and _bounded_run_items(
            calls, ("id", "attempt_label", "latency_ms")
        )
        and history is not None
    )
    honest = (
        factual
        and task_output.get("observed_cost_status") == "unavailable"
        and task_output.get("causal_scope") == "observed-mechanism-only"
        and task_output.get("evidence_status") == "reconciled"
    )
    return factual, bounded, honest


def _project_health(task_output, calls, expected):
    facts = _mapping(expected.get("facts"))
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
        and _bounded_run_items(
            calls, ("id", "attempt_label", "latency_ms")
        )
        and history is not None
    )
    honest = (
        factual
        and task_output.get("recommendation") == "investigate"
        and task_output.get("causal_scope") == "observed-mechanism-only"
        and task_output.get("evidence_status") == "reconciled"
    )
    return factual, bounded, honest


def _evaluate(task_id, output, calls, expected):
    if task_id == "maintainer-evaluation-reconciliation":
        return _reconciliation(output, calls, expected)
    if task_id == "maintainer-source-inventory":
        return _source_inventory(output, calls, expected)
    if task_id == "maintainer-history-hotspot":
        return _history(output, calls, expected)
    if task_id == "maintainer-project-health":
        return _project_health(output, calls, expected)
    return False, False, False


def _mechanism_used(task_id, calls, expected):
    if task_id == "maintainer-evaluation-reconciliation":
        return (
            len(_evaluation_summary(calls)) == 1
            and _evaluation_children(calls, expected) is not None
        )
    if task_id == "maintainer-source-inventory":
        facts = _mapping(expected.get("facts"))
        return (
            _run_count(calls, int(facts.get("run_count") or 0))
            and _bounded_run_items(
                calls, ("id", "attempt_label", "latency_ms")
            )
            and len(_evaluation_summary(calls)) == 1
        )
    if task_id == "maintainer-history-hotspot":
        return (
            _bounded_run_items(
                calls, ("id", "attempt_label", "latency_ms")
            )
            and _history_call(calls, expected) is not None
        )
    if task_id == "maintainer-project-health":
        facts = _mapping(expected.get("facts"))
        return (
            _run_count(calls, int(facts.get("run_count") or 0))
            and _bounded_run_items(
                calls, ("id", "attempt_label", "latency_ms")
            )
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
            "locked_source_scope": locked_scope,
            "release_mechanism_used": mechanism,
        }
    factual, bounded, honest = _evaluate(task_id, parsed, calls, expected)
    return {
        "answer_correct": factual,
        "locked_source_scope": (
            parsed.get("source_project") == project and locked_scope
        ),
        "bounded_evidence": bounded,
        "evidence_honesty": honest,
        "release_mechanism_used": mechanism,
    }
