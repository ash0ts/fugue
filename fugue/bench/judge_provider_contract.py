from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fugue.bench.candidates import stable_digest


class ProviderOutputValidationError(ValueError):
    """A typed, content-free provider-output contract failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def provider_response_schema(
    rubric: Mapping[str, Any], modality: str
) -> dict[str, Any]:
    dimensions = _dimensions(rubric, modality)
    labels = list(_labels(rubric))
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["label", "dimension_labels", "reason", "missing_evidence"],
        "properties": {
            "label": {"type": "string", "enum": labels},
            "dimension_labels": {
                "type": "object",
                "additionalProperties": False,
                "required": list(dimensions),
                "properties": {
                    dimension: {"type": "string", "enum": labels}
                    for dimension in dimensions
                },
            },
            "reason": {
                "type": "string",
                "description": "Non-empty evidence reason; at most 2000 characters.",
            },
            "missing_evidence": {"type": "boolean"},
        },
    }


def response_schema_digest(rubric: Mapping[str, Any]) -> str:
    modalities = _mapping(rubric.get("modalities"), "rubric modalities")
    return stable_digest({
        "schema_version": 2,
        "provider_schemas": {
            name: provider_response_schema(rubric, name) for name in sorted(modalities)
        },
        "host_bound_fields": ["case_ref", "modality"],
        "normalization": "strict-no-coercion",
        "local_constraints": {"reason_nonempty": True, "reason_max_characters": 2000},
    })


def validate_provider_output(
    value: Mapping[str, Any], rubric: Mapping[str, Any], modality: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("json_not_object", "provider model output must be an object")
    expected = {"label", "dimension_labels", "reason", "missing_evidence"}
    if set(value) != expected:
        _fail("field_set_mismatch", "provider model output fields do not match")
    selected = value.get("dimension_labels")
    if not isinstance(selected, Mapping):
        _fail("dimension_labels_not_object", "dimension labels must be an object")
    dimensions, labels = _dimensions(rubric, modality), _labels(rubric)
    if set(selected) != set(dimensions):
        _fail("dimension_set_mismatch", "dimension label fields do not match")
    if value.get("label") not in labels or any(row not in labels for row in selected.values()):
        _fail("label_value_invalid", "provider label is not a qualitative anchor")
    if not isinstance(value.get("missing_evidence"), bool):
        _fail("missing_evidence_not_boolean", "missing evidence must be a boolean")
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 2000:
        _fail("reason_invalid", "provider reason is invalid")
    return dict(value)


def _dimensions(rubric: Mapping[str, Any], modality: str) -> tuple[str, ...]:
    selected = _mapping(_mapping(rubric.get("modalities"), "modalities").get(modality), "modality")
    dimensions = selected.get("dimensions")
    if not isinstance(dimensions, list) or not all(isinstance(row, str) for row in dimensions):
        raise ValueError("rubric modality dimensions are invalid")
    return tuple(dimensions)


def _labels(rubric: Mapping[str, Any]) -> tuple[str, ...]:
    labels = _mapping(rubric.get("labels"), "rubric labels")
    return tuple(str(row) for row in labels)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _fail(code: str, message: str) -> None:
    raise ProviderOutputValidationError(code, message)
