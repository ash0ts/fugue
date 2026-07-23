from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Any, Literal

from fugue.bench.candidates import stable_digest
from fugue.bench.library import validate_id
from fugue.research.contracts import RESEARCH_SCHEMA_VERSION, ResearchError

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_FILTER_FIELDS = frozenset(
    {
        "conversation_id",
        "harness",
        "model",
        "operation",
        "run_id",
        "status",
        "tag",
        "trace_id",
    }
)
_TRACE_FIELDS = frozenset(
    {
        "artifacts",
        "conversation",
        "cost",
        "errors",
        "final_output",
        "latency",
        "operation",
        "status",
        "tokens",
        "tools",
    }
)

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class TraceSourceRefV1:
    schema_version: int
    source_id: str
    adapter: Literal["jsonl", "weave"]
    source_digest: str

    def to_dict(self) -> dict[str, Any]:
        return _json(asdict(self))


@dataclass(frozen=True)
class TraceAuditDraftV1:
    schema_version: int
    study_id: str
    source_id: str
    objective: str
    fields: tuple[str, ...]
    filters: dict[str, JsonValue]
    max_traces: int
    started_after: str | None = None
    started_before: str | None = None
    draft_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(_json(asdict(self)))


@dataclass(frozen=True)
class TraceAuditPreviewV1:
    schema_version: int
    study_id: str
    audit_id: str
    source: TraceSourceRefV1
    draft: dict[str, Any]
    maximum_traces: int
    available_fields: tuple[str, ...]
    estimated_calls: dict[str, int]
    estimated_cost_usd: float
    approval_required: bool
    redactions: tuple[str, ...]
    eligible: bool
    blockers: tuple[str, ...]
    preview_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json(asdict(self))


@dataclass(frozen=True)
class TraceAuditV1:
    schema_version: int
    id: str
    study_id: str
    preview_digest: str
    source: TraceSourceRefV1
    source_snapshot_digest: str
    cohort_count: int
    coverage: dict[str, JsonValue]
    clusters: tuple[dict[str, JsonValue], ...]
    trace_refs: tuple[dict[str, JsonValue], ...]
    evidence_samples: tuple[dict[str, JsonValue], ...]
    suggested_tasks: tuple[dict[str, JsonValue], ...]
    redactions: tuple[str, ...]
    warnings: tuple[str, ...]
    created_at: str
    audit_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json(asdict(self))


@dataclass(frozen=True)
class CandidateRefV1:
    schema_version: int
    repository_id: str
    source_kind: Literal["git_commit", "artifact"]
    source_digest: str
    revision: str
    content_digest: str
    registered_experiment_id: str
    registered_variant_id: str

    def to_dict(self) -> dict[str, Any]:
        return _json(asdict(self))


@dataclass(frozen=True)
class ExecutionApprovalV1:
    schema_version: int
    approval_id: str
    subject_kind: Literal["experiment", "trace_audit"]
    preview_digest: str
    maximum_cost_usd: float
    maximum_cells: int | None
    approved_by: str
    operation_id: str
    created_at: str
    expires_at: str
    approval_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(_json(asdict(self)), preserve_zero=True)


def trace_source_ref_from_dict(raw: Mapping[str, Any]) -> TraceSourceRefV1:
    _reject_unknown(raw, TraceSourceRefV1, "trace source")
    adapter = str(raw.get("adapter") or "")
    if adapter not in {"jsonl", "weave"}:
        raise ValueError("trace source adapter must be jsonl or weave")
    return TraceSourceRefV1(
        schema_version=_schema(raw, "trace source"),
        source_id=validate_id(str(raw.get("source_id") or ""), kind="trace source id"),
        adapter=adapter,  # type: ignore[arg-type]
        source_digest=_digest_value(raw.get("source_digest"), "trace source digest"),
    )


def trace_audit_draft_from_dict(
    raw: Mapping[str, Any], *, require_digest: bool = True
) -> TraceAuditDraftV1:
    _reject_unknown(raw, TraceAuditDraftV1, "trace audit draft")
    fields = _strings(raw.get("fields"), "trace field")
    unknown_fields = sorted(set(fields) - _TRACE_FIELDS)
    if unknown_fields:
        raise ValueError("unknown trace fields: " + ", ".join(unknown_fields))
    filters = _mapping(raw.get("filters") or {}, "trace filters")
    unknown_filters = sorted(set(filters) - _FILTER_FIELDS)
    if unknown_filters:
        raise ValueError("unknown trace filters: " + ", ".join(unknown_filters))
    for key, value in filters.items():
        _filter_value(value, key)
    draft = TraceAuditDraftV1(
        schema_version=_schema(raw, "trace audit draft"),
        study_id=validate_id(str(raw.get("study_id") or ""), kind="study id"),
        source_id=validate_id(str(raw.get("source_id") or ""), kind="trace source id"),
        objective=_text(raw.get("objective"), "trace audit objective", 4000),
        fields=fields,
        filters=_json(filters),
        max_traces=_bounded_int(raw.get("max_traces"), "maximum traces", 1, 1000),
        started_after=_optional_timestamp(raw.get("started_after"), "started after"),
        started_before=_optional_timestamp(raw.get("started_before"), "started before"),
        draft_digest=str(raw.get("draft_digest") or ""),
    )
    if draft.started_after and draft.started_before:
        if draft.started_after >= draft.started_before:
            raise ValueError("started_after must precede started_before")
    digest = _artifact_digest(draft.to_dict(), "draft_digest")
    if require_digest and draft.draft_digest != digest:
        raise ValueError("draft_digest does not match the trace audit draft")
    if draft.draft_digest and draft.draft_digest != digest:
        raise ValueError("draft_digest does not match the trace audit draft")
    return replace(draft, draft_digest=digest)


def build_trace_audit_draft(**values: Any) -> TraceAuditDraftV1:
    return trace_audit_draft_from_dict(
        {"schema_version": RESEARCH_SCHEMA_VERSION, **values, "draft_digest": ""},
        require_digest=False,
    )


def trace_audit_preview_from_dict(raw: Mapping[str, Any]) -> TraceAuditPreviewV1:
    _reject_unknown(raw, TraceAuditPreviewV1, "trace audit preview")
    preview = TraceAuditPreviewV1(
        schema_version=_schema(raw, "trace audit preview"),
        study_id=validate_id(str(raw.get("study_id") or ""), kind="study id"),
        audit_id=validate_id(str(raw.get("audit_id") or ""), kind="trace audit id"),
        source=trace_source_ref_from_dict(_mapping(raw.get("source"), "trace source")),
        draft=_mapping(raw.get("draft"), "trace audit draft"),
        maximum_traces=_bounded_int(
            raw.get("maximum_traces"), "maximum traces", 1, 1000
        ),
        available_fields=_strings(raw.get("available_fields"), "available field"),
        estimated_calls={
            validate_id(str(key), kind="call kind"): _bounded_int(
                value, f"{key} calls", 0, 1_000_000
            )
            for key, value in _mapping(
                raw.get("estimated_calls") or {}, "estimated calls"
            ).items()
        },
        estimated_cost_usd=_non_negative_number(
            raw.get("estimated_cost_usd"), "estimated cost"
        ),
        approval_required=_boolean(raw.get("approval_required"), "approval required"),
        redactions=_strings(raw.get("redactions"), "redaction", allow_empty=True),
        eligible=_boolean(raw.get("eligible"), "preview eligible"),
        blockers=_strings(raw.get("blockers"), "blocker", allow_empty=True),
        preview_digest=_digest_value(raw.get("preview_digest"), "preview digest"),
    )
    _verify_artifact(preview.to_dict(), "preview_digest", "trace audit preview")
    trace_audit_draft_from_dict(preview.draft)
    return preview


def sign_trace_audit_preview(preview: TraceAuditPreviewV1) -> TraceAuditPreviewV1:
    return replace(
        preview,
        preview_digest=_artifact_digest(preview.to_dict(), "preview_digest"),
    )


def trace_audit_from_dict(raw: Mapping[str, Any]) -> TraceAuditV1:
    _reject_unknown(raw, TraceAuditV1, "trace audit")
    audit = TraceAuditV1(
        schema_version=_schema(raw, "trace audit"),
        id=validate_id(str(raw.get("id") or ""), kind="trace audit id"),
        study_id=validate_id(str(raw.get("study_id") or ""), kind="study id"),
        preview_digest=_digest_value(raw.get("preview_digest"), "preview digest"),
        source=trace_source_ref_from_dict(_mapping(raw.get("source"), "trace source")),
        source_snapshot_digest=_digest_value(
            raw.get("source_snapshot_digest"), "source snapshot digest"
        ),
        cohort_count=_bounded_int(raw.get("cohort_count"), "cohort count", 0, 1000),
        coverage=_json(_mapping(raw.get("coverage"), "trace coverage")),
        clusters=tuple(
            _json(_mapping(value, "trace cluster"))
            for value in _sequence(raw.get("clusters"), "trace clusters")
        ),
        trace_refs=tuple(
            _json(_mapping(value, "trace reference"))
            for value in _sequence(raw.get("trace_refs"), "trace references")
        ),
        evidence_samples=tuple(
            _json(_mapping(value, "trace evidence sample"))
            for value in _sequence(
                raw.get("evidence_samples"), "trace evidence samples"
            )
        ),
        suggested_tasks=tuple(
            _json(_mapping(value, "suggested task"))
            for value in _sequence(raw.get("suggested_tasks"), "suggested tasks")
        ),
        redactions=_strings(raw.get("redactions"), "redaction", allow_empty=True),
        warnings=_strings(raw.get("warnings"), "warning", allow_empty=True),
        created_at=_timestamp(raw.get("created_at"), "audit creation time"),
        audit_digest=_digest_value(raw.get("audit_digest"), "audit digest"),
    )
    _verify_artifact(audit.to_dict(), "audit_digest", "trace audit")
    return audit


def sign_trace_audit(audit: TraceAuditV1) -> TraceAuditV1:
    return replace(
        audit, audit_digest=_artifact_digest(audit.to_dict(), "audit_digest")
    )


def candidate_ref_from_dict(raw: Mapping[str, Any]) -> CandidateRefV1:
    _reject_unknown(raw, CandidateRefV1, "candidate reference")
    kind = str(raw.get("source_kind") or "")
    if kind not in {"git_commit", "artifact"}:
        raise ValueError("candidate source_kind must be git_commit or artifact")
    revision = str(raw.get("revision") or "")
    if not _GIT_REVISION.fullmatch(revision):
        raise ValueError("candidate revision must be an immutable hexadecimal digest")
    return CandidateRefV1(
        schema_version=_schema(raw, "candidate reference"),
        repository_id=validate_id(
            str(raw.get("repository_id") or ""), kind="repository id"
        ),
        source_kind=kind,  # type: ignore[arg-type]
        source_digest=_digest_value(raw.get("source_digest"), "source digest"),
        revision=revision,
        content_digest=_digest_value(raw.get("content_digest"), "content digest"),
        registered_experiment_id=validate_id(
            str(raw.get("registered_experiment_id") or ""), kind="experiment id"
        ),
        registered_variant_id=validate_id(
            str(raw.get("registered_variant_id") or ""), kind="variant id"
        ),
    )


def execution_approval_from_dict(raw: Mapping[str, Any]) -> ExecutionApprovalV1:
    _reject_unknown(raw, ExecutionApprovalV1, "execution approval")
    subject_kind = str(raw.get("subject_kind") or "")
    if subject_kind not in {"experiment", "trace_audit"}:
        raise ValueError("approval subject_kind must be experiment or trace_audit")
    approval = ExecutionApprovalV1(
        schema_version=_schema(raw, "execution approval"),
        approval_id=validate_id(str(raw.get("approval_id") or ""), kind="approval id"),
        subject_kind=subject_kind,  # type: ignore[arg-type]
        preview_digest=_digest_value(raw.get("preview_digest"), "preview digest"),
        maximum_cost_usd=_non_negative_number(
            raw.get("maximum_cost_usd"), "maximum cost"
        ),
        maximum_cells=(
            _bounded_int(raw.get("maximum_cells"), "maximum cells", 1, 1_000_000)
            if raw.get("maximum_cells") is not None
            else None
        ),
        approved_by=_text(raw.get("approved_by"), "approver", 300),
        operation_id=validate_id(
            str(raw.get("operation_id") or ""), kind="operation id"
        ),
        created_at=_timestamp(raw.get("created_at"), "approval creation time"),
        expires_at=_timestamp(raw.get("expires_at"), "approval expiry"),
        approval_digest=_digest_value(raw.get("approval_digest"), "approval digest"),
    )
    if approval.expires_at <= approval.created_at:
        raise ValueError("approval expiry must follow its creation time")
    _verify_artifact(approval.to_dict(), "approval_digest", "execution approval")
    return approval


def sign_execution_approval(approval: ExecutionApprovalV1) -> ExecutionApprovalV1:
    return replace(
        approval,
        approval_digest=_artifact_digest(approval.to_dict(), "approval_digest"),
    )


def ensure_registered_candidate(
    candidate: CandidateRefV1, *, experiment_id: str, variants: Sequence[str]
) -> None:
    if candidate.registered_experiment_id != experiment_id:
        raise ResearchError(
            "candidate_experiment_mismatch",
            "candidate reference belongs to another registered experiment",
            category="policy",
        )
    if candidate.registered_variant_id not in variants:
        raise ResearchError(
            "candidate_variant_mismatch",
            "candidate reference is not selected by the experiment draft",
            category="policy",
        )


def _schema(raw: Mapping[str, Any], label: str) -> int:
    value = raw.get("schema_version")
    if type(value) is not int or value != RESEARCH_SCHEMA_VERSION:
        raise ValueError(f"{label} schema_version must be {RESEARCH_SCHEMA_VERSION}")
    return value


def _artifact_digest(value: Mapping[str, Any], field: str) -> str:
    return stable_digest({key: item for key, item in value.items() if key != field})


def _verify_artifact(value: Mapping[str, Any], field: str, label: str) -> None:
    if value.get(field) != _artifact_digest(value, field):
        raise ValueError(f"{field} does not match the {label}")


def _digest_value(value: Any, label: str) -> str:
    text = str(value or "")
    if not _DIGEST.fullmatch(text):
        raise ValueError(f"{label} must be a sha256 digest")
    return text


def _text(value: Any, label: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    text = value.strip()
    if len(text) > limit:
        raise ValueError(f"{label} exceeds {limit} characters")
    return text


def _timestamp(value: Any, label: str) -> str:
    text = _text(value, label, 64)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _optional_timestamp(value: Any, label: str) -> str | None:
    return _timestamp(value, label) if value is not None else None


def _bounded_int(value: Any, label: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}")
    return value


def _non_negative_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a non-negative finite number") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be a non-negative finite number")
    return number


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean")
    return value


def _strings(value: Any, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    values = _sequence(value if value is not None else [], label)
    result = tuple(_text(item, label, 300) for item in values)
    if not allow_empty and not result:
        raise ValueError(f"{label} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicates")
    return result


def _sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a list")
    return list(value)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return {str(key): item for key, item in value.items()}


def _filter_value(value: Any, label: str) -> None:
    values = value if isinstance(value, list) else [value]
    if not values or len(values) > 100:
        raise ValueError(f"trace filter {label} must contain 1 to 100 values")
    for item in values:
        if not isinstance(item, (str, int, bool)) or isinstance(item, float):
            raise ValueError(f"trace filter {label} contains an unsupported value")
        if isinstance(item, str) and (not item.strip() or len(item) > 500):
            raise ValueError(f"trace filter {label} contains invalid text")


def _json(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("value must contain only finite JSON data") from exc


def _drop_empty(
    value: dict[str, Any], *, preserve_zero: bool = False
) -> dict[str, Any]:
    empty = (None, "", [], {}, ()) if preserve_zero else (None, "", [], {}, (), 0)
    return {key: item for key, item in value.items() if item not in empty}


def _reject_unknown(raw: Mapping[str, Any], cls: type[Any], label: str) -> None:
    allowed = set(cls.__dataclass_fields__)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown {label} fields: " + ", ".join(unknown))
