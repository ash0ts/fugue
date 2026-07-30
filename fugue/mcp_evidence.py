from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_GRAPHQL_SAFE_PROJECTION_FIELDS = {
    "id": "id",
    "name": "id",
    "displayName": "display_name",
    "state": "state",
    "config": "config.*",
    "summaryMetrics": "summary.*",
    "createdAt": "created_at",
    "heartbeatAt": "heartbeat_at",
    "duration": "duration",
    "group": "group",
    "jobType": "job_type",
}
_MAX_GRAPHQL_LIMIT = 1_000_000
_SAFE_STRUCTURED_ERROR_CODES = frozenset(
    {
        "cancelled",
        "capability_unavailable",
        "deadline_exceeded",
        "internal_error",
        "invalid_argument",
        "invalid_params",
        "invalid_request",
        "jsonrpc_error",
        "method_not_found",
        "mcp_is_error",
        "not_found",
        "parse_error",
        "permission_denied",
        "query_timeout",
        "rate_limited",
        "reference_not_locked",
        "response_too_large",
        "server_busy",
        "storage_error",
        "storage_read_error",
        "timeout",
        "tool_error",
        "unauthorized",
        "unavailable",
        "validation_error",
    }
)
_JSONRPC_ERROR_CODES = {
    -32700: "parse_error",
    -32600: "invalid_request",
    -32601: "method_not_found",
    -32602: "invalid_params",
    -32603: "internal_error",
}


def safe_structured_error_code(value: Any, fallback: str) -> str:
    """Map untrusted error metadata into Fugue's closed public taxonomy."""

    if fallback not in _SAFE_STRUCTURED_ERROR_CODES:
        fallback = "tool_error"
    direct = _allowlisted_error_code(value)
    if direct is not None:
        return direct
    if isinstance(value, Mapping):
        for key in ("code", "type", "kind", "status"):
            candidate = _allowlisted_error_code(value.get(key))
            if candidate is not None:
                return candidate
    return fallback


def _allowlisted_error_code(value: Any) -> str | None:
    if type(value) is int:
        return _JSONRPC_ERROR_CODES.get(value)
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("-", "_")
    return normalized if normalized in _SAFE_STRUCTURED_ERROR_CODES else None


def safe_graphql_query_shape(
    query: str,
    *,
    variables: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an allowlisted GraphQL shape with no literals or aliases."""

    scrubbed = _strip_graphql_literals(query)
    operation_match = re.search(r"\b(query|mutation|subscription)\b", scrubbed)
    operation_type = operation_match.group(1) if operation_match else "query"
    response_modes: list[str] = []
    if re.search(r"\brunCount\b", scrubbed):
        response_modes.append("count")
    if re.search(r"\bruns\s*(?:\(|\{)", scrubbed):
        response_modes.append("items")
    if re.search(r"\brun\s*(?:\(|\{)", scrubbed) and "items" not in response_modes:
        response_modes.append("item")
    resource = "runs" if response_modes else None
    node_selections = _graphql_node_selections(scrubbed)
    node_selection = "\n".join(node_selections)
    node_fields = {
        normalized
        for raw, normalized in _GRAPHQL_SAFE_PROJECTION_FIELDS.items()
        if re.search(rf"\b{re.escape(raw)}\b", node_selection)
    }
    query_limit = _graphql_numeric_argument(scrubbed, "first")
    if query_limit is None:
        variable_name = _graphql_variable_argument(scrubbed, "first")
        variable_value = variables.get(variable_name) if variables and variable_name else None
        if (
            type(variable_value) is int
            and 0 <= variable_value <= _MAX_GRAPHQL_LIMIT
        ):
            query_limit = variable_value
    projection_required = bool({"items", "item"} & set(response_modes))
    projection_resolved = (
        not projection_required
        or (
            len(node_selections) == 1
            and "..." not in scrubbed
            and not re.search(r"\bfragment\b", scrubbed)
            and _graphql_projection_is_allowlisted(node_selections[0])
        )
    )
    limit_resolved = "items" not in response_modes or query_limit is not None
    scope_variables = _graphql_project_scope_variables(query)
    scope_values = {
        target: variables.get(source)
        for target, source in scope_variables.items()
    } if variables else {}
    scope_resolved = all(
        isinstance(scope_values.get(key), str)
        and bool(str(scope_values[key]).strip())
        for key in ("entity_name", "project_name")
    )
    return {
        "raw_graphql": True,
        "graphql_operation_type": operation_type,
        "resource": resource,
        "response_modes": response_modes,
        "projected_fields": sorted(node_fields),
        "broad_projection": (
            not projection_resolved
            or bool({"config.*", "summary.*"} & node_fields)
        ),
        "graphql_requested_limit": query_limit,
        "graphql_limit_resolved": limit_resolved,
        "graphql_projection_resolved": projection_resolved,
        "graphql_scope_resolved": scope_resolved,
    }


def safe_graphql_event_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Replace raw GraphQL and variables with their safe execution shape."""

    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        return dict(arguments)
    result = {
        str(key): value
        for key, value in arguments.items()
        if key not in {"query", "variables"}
    }
    variables = arguments.get("variables")
    variable_values = variables if isinstance(variables, Mapping) else None
    result["raw_graphql_shape"] = safe_graphql_query_shape(
        query,
        variables=variable_values,
    )
    if isinstance(variables, Mapping):
        scope_variables = _graphql_project_scope_variables(query)
        for target, source in scope_variables.items():
            value = variables.get(source)
            if isinstance(value, str) and value.strip() and target not in result:
                result[target] = value.strip()
    entity_name = _first_nonempty_text(
        result,
        ("entity", "entity_name", "entityName"),
    )
    project_name = _first_nonempty_text(
        result,
        ("project", "project_name", "projectName"),
    )
    scope_resolved = entity_name is not None and project_name is not None
    result["raw_graphql_shape"]["graphql_scope_resolved"] = scope_resolved
    if not scope_resolved:
        # A raw query whose exact destination cannot be proven must fail
        # cross-project qualification instead of appearing unscoped.
        result.setdefault("project_ref", "*/*")
    return result


def validated_graphql_query_shape(value: Any) -> dict[str, Any]:
    """Validate a proxy-produced shape before it enters scorer evidence."""

    if not isinstance(value, Mapping) or value.get("raw_graphql") is not True:
        return {}
    operation = str(value.get("graphql_operation_type") or "")
    if operation not in {"query", "mutation", "subscription"}:
        return {}
    resource = value.get("resource")
    if resource not in {None, "runs"}:
        return {}
    modes = value.get("response_modes")
    if not isinstance(modes, list) or any(
        mode not in {"count", "items", "item"} for mode in modes
    ):
        return {}
    projected = value.get("projected_fields")
    allowed_fields = set(_GRAPHQL_SAFE_PROJECTION_FIELDS.values())
    if not isinstance(projected, list) or any(
        field not in allowed_fields for field in projected
    ):
        return {}
    requested_limit = value.get("graphql_requested_limit")
    if requested_limit is not None and (
        type(requested_limit) is not int
        or requested_limit < 0
        or requested_limit > _MAX_GRAPHQL_LIMIT
    ):
        return {}
    projection_resolved = value.get("graphql_projection_resolved", False)
    limit_resolved = value.get(
        "graphql_limit_resolved",
        "items" not in modes or requested_limit is not None,
    )
    scope_resolved = value.get("graphql_scope_resolved", False)
    if any(
        type(item) is not bool
        for item in (projection_resolved, limit_resolved, scope_resolved)
    ):
        return {}
    return {
        "raw_graphql": True,
        "graphql_operation_type": operation,
        "resource": resource,
        "response_modes": list(dict.fromkeys(modes)),
        "projected_fields": sorted(set(projected)),
        "broad_projection": bool(
            {"config.*", "summary.*"} & set(projected)
        )
        or not projection_resolved,
        "graphql_requested_limit": requested_limit,
        "graphql_limit_resolved": limit_resolved,
        "graphql_projection_resolved": projection_resolved,
        "graphql_scope_resolved": scope_resolved,
    }


def _strip_graphql_literals(query: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(query):
        if query.startswith('"""', index):
            end = query.find('"""', index + 3)
            index = len(query) if end < 0 else end + 3
            result.append(" ")
            continue
        char = query[index]
        if char == '"':
            index += 1
            while index < len(query):
                if query[index] == "\\":
                    index += 2
                    continue
                if query[index] == '"':
                    index += 1
                    break
                index += 1
            result.append(" ")
            continue
        if char == "#":
            end = query.find("\n", index)
            index = len(query) if end < 0 else end
            result.append("\n")
            continue
        result.append(char)
        index += 1
    return "".join(result)


def _graphql_node_selections(query: str) -> list[str]:
    selections: list[str] = []
    for match in re.finditer(r"\bnode\s*\{", query):
        start = query.find("{", match.start())
        depth = 0
        for index in range(start, len(query)):
            if query[index] == "{":
                depth += 1
            elif query[index] == "}":
                depth -= 1
                if depth == 0:
                    selections.append(query[start + 1 : index])
                    break
    return selections


def _graphql_projection_is_allowlisted(selection: str) -> bool:
    if not selection.strip() or "..." in selection:
        return False
    without_aliases = re.sub(
        r"\b[_A-Za-z][_0-9A-Za-z]*\s*:\s*",
        "",
        selection,
    )
    identifiers = set(re.findall(r"\b[_A-Za-z][_0-9A-Za-z]*\b", without_aliases))
    return bool(identifiers) and identifiers <= set(_GRAPHQL_SAFE_PROJECTION_FIELDS)


def _graphql_numeric_argument(query: str, name: str) -> int | None:
    matches = list(
        re.finditer(rf"\b{re.escape(name)}\s*:\s*(\d+)\b", query)
    )
    if len(matches) != 1:
        return None
    value = int(matches[0].group(1))
    return value if value <= _MAX_GRAPHQL_LIMIT else None


def _graphql_variable_argument(query: str, name: str) -> str | None:
    matches = list(
        re.finditer(
            rf"\b{re.escape(name)}\s*:\s*\$([_A-Za-z][_0-9A-Za-z]*)\b",
            query,
        )
    )
    return matches[0].group(1) if len(matches) == 1 else None


def _graphql_project_scope_variables(query: str) -> dict[str, str]:
    scrubbed = _strip_graphql_literals(query)
    projects = list(
        re.finditer(r"\bproject\s*\(([^)]*)\)", scrubbed, re.DOTALL)
    )
    if len(projects) != 1:
        return {}
    arguments = projects[0].group(1)
    result: dict[str, str] = {}
    for target, names in (
        ("entity_name", ("entityName", "entity", "entity_name")),
        ("project_name", ("name", "projectName", "project", "project_name")),
    ):
        for name in names:
            match = re.search(
                rf"\b{re.escape(name)}\s*:\s*\$([_A-Za-z][_0-9A-Za-z]*)\b",
                arguments,
            )
            if match is not None:
                result[target] = match.group(1)
                break
    return result


def _first_nonempty_text(
    values: Mapping[str, Any],
    keys: tuple[str, ...],
) -> str | None:
    for key in keys:
        value = values.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
