from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Literal

from fugue.bench.candidates import stable_digest
from fugue.bench.library import validate_id

RESEARCH_SCHEMA_VERSION = 1
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_STUDY_FIELDS = {
    "question",
    "background",
    "plan",
    "findings",
    "conclusion",
    "next_questions",
}
TERMINAL_EXPERIMENT_STATES = frozenset(
    {"completed", "blocked", "cancelled", "interrupted"}
)

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


class ResearchError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        category: str = "validation",
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = validate_id(code, kind="research error code")
        self.category = validate_id(category, kind="research error category")
        self.retryable = bool(retryable)
        self.details = _json_mapping(details or {}, "research error details")

    def to_dict(self) -> dict[str, Any]:
        unsigned = {
            "schema_version": RESEARCH_SCHEMA_VERSION,
            "code": self.code,
            "category": self.category,
            "retryable": self.retryable,
            "message": str(self),
            "details": self.details,
            "error_digest": "",
        }
        return {**unsigned, "error_digest": _digest(unsigned, "error_digest")}


@dataclass(frozen=True)
class AttributionV1:
    actor_type: Literal["human", "agent", "service"] = "human"
    name: str = "unknown"
    identity: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(_json_value(asdict(self)))


@dataclass(frozen=True)
class EvidenceRefV1:
    kind: Literal[
        "outcome",
        "analysis",
        "evaluation",
        "run",
        "conversation",
        "trace_audit",
        "artifact",
        "resource",
        "note",
        "result",
    ]
    ref: str
    digest: str | None = None
    version: str | None = None
    uri: str | None = None
    selector: dict[str, JsonValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(_json_value(asdict(self)))


@dataclass(frozen=True)
class StudyBriefV1:
    question: str
    background: str = ""
    plan: str = ""
    findings: str = ""
    conclusion: str = ""
    next_questions: tuple[str, ...] = ()
    provenance: dict[str, tuple[EvidenceRefV1, ...]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class StudyNoteV1:
    id: str
    revision: int
    text: str
    kind: str
    sources: tuple[EvidenceRefV1, ...]
    created_at: str
    attribution: AttributionV1
    supersedes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(_json_value(asdict(self)))


@dataclass(frozen=True)
class StudyResourceV1:
    id: str
    uri: str
    kind: str
    digest: str | None
    version: str | None
    title: str
    summary: str
    added_revision: int
    added_at: str
    attribution: AttributionV1

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(_json_value(asdict(self)))


@dataclass(frozen=True)
class ResultUncertaintyV1:
    kind: str
    value: float | None = None
    lower: float | None = None
    upper: float | None = None
    level: float | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(_json_value(asdict(self)), preserve_false=True)


@dataclass(frozen=True)
class ResultEstimateV1:
    value: float | int | str
    kind: str = ""
    unit: str = ""
    uncertainty: ResultUncertaintyV1 | None = None

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(_json_value(asdict(self)), preserve_false=True)


@dataclass(frozen=True)
class ResultComparisonV1:
    condition: str
    comparator: str
    condition_sources: tuple[EvidenceRefV1, ...]
    comparator_sources: tuple[EvidenceRefV1, ...]

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class StudyResultV1:
    id: str
    revision: int
    statement: str
    kind: str
    outcome: str
    estimate: ResultEstimateV1 | None
    comparison: ResultComparisonV1 | None
    population: str
    conditions: dict[str, JsonValue]
    sample_size: int | None
    aggregation: str
    exclusions: tuple[str, ...]
    sources: tuple[EvidenceRefV1, ...]
    analysis_source: EvidenceRefV1 | None
    created_at: str
    attribution: AttributionV1
    supersedes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(_json_value(asdict(self)), preserve_false=True)


@dataclass(frozen=True)
class StudyExperimentRefV1:
    experiment_id: str
    state: str
    proposal_digest: str | None
    preview_digest: str
    task_suite_digest: str | None
    plan_digest: str | None
    parent_experiment_ids: tuple[str, ...]
    run_id: str | None
    outcome_digest: str | None
    evaluation_digest: str | None
    analysis_digest: str | None
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(_json_value(asdict(self)))


@dataclass(frozen=True)
class StudyV1:
    schema_version: int
    id: str
    title: str
    campaign_id: str
    brief: StudyBriefV1
    revision: int
    notes: tuple[StudyNoteV1, ...]
    resources: tuple[StudyResourceV1, ...]
    results: tuple[StudyResultV1, ...]
    experiments: tuple[StudyExperimentRefV1, ...]
    run_refs: tuple[EvidenceRefV1, ...]
    baseline_refs: tuple[EvidenceRefV1, ...]
    primary_baseline_ref: EvidenceRefV1 | None
    parent_study_ids: tuple[str, ...]
    created_at: str
    updated_at: str
    created_by: AttributionV1
    updated_by: AttributionV1
    study_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class StudyContextV1:
    schema_version: int
    study_id: str
    revision: int
    title: str
    campaign_id: str
    brief: dict[str, Any]
    baseline: dict[str, Any]
    experiments: tuple[dict[str, Any], ...]
    results: tuple[dict[str, Any], ...]
    notes: tuple[dict[str, Any], ...]
    resources: tuple[dict[str, Any], ...]
    omissions: dict[str, int]
    context_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class StudyUpdateV1:
    message: str | None = None
    note_kind: str = "observation"
    note_sources: tuple[EvidenceRefV1, ...] = ()
    note_supersedes: str | None = None
    brief_patch: dict[str, JsonValue] = field(default_factory=dict)
    brief_sources: dict[str, tuple[EvidenceRefV1, ...]] = field(default_factory=dict)
    resources: tuple[dict[str, Any], ...] = ()
    results: tuple[dict[str, Any], ...] = ()
    run_refs: tuple[EvidenceRefV1, ...] = ()
    baseline_refs: tuple[EvidenceRefV1, ...] | None = None
    primary_baseline_ref: EvidenceRefV1 | None = None
    attribution: AttributionV1 = field(default_factory=AttributionV1)

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(_json_value(asdict(self)), preserve_false=True)


@dataclass(frozen=True)
class ExperimentDraftV1:
    schema_version: int
    study_id: str
    campaign_id: str
    proposal_id: str
    stage_id: str
    question: str
    hypothesis: str
    fixed_dimensions: tuple[str, ...]
    varied_dimensions: tuple[str, ...]
    measured_dimensions: tuple[str, ...]
    experiment_id: str
    model: str
    n_attempts: int
    n_concurrent: int
    preset_id: str | None = None
    workloads: tuple[str, ...] = ()
    harnesses: tuple[str, ...] = ()
    context_systems: tuple[str, ...] = ()
    variants: tuple[str, ...] = ()
    n_tasks: int | None = None
    trace_content: str = "full"
    analysis_ids: tuple[str, ...] = ()
    parent_experiment_ids: tuple[str, ...] = ()
    parent_outcome_id: str | None = None
    decision_rationale: str = ""
    task_suite_digest: str | None = None
    task_suite_draft: dict[str, Any] | None = None
    task_recipe_preview: dict[str, Any] | None = None
    candidate_refs: tuple[dict[str, Any], ...] = ()
    scoring_revision: dict[str, Any] | None = None
    task_analysis_id: str | None = None
    draft_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(_json_value(asdict(self)), preserve_false=True)


@dataclass(frozen=True)
class ExperimentPreviewV1:
    schema_version: int
    study_id: str
    experiment_id: str
    campaign_id: str
    catalog_digest: str
    policy_digest: str
    draft: dict[str, Any]
    task_suite_preview: dict[str, Any] | None
    plan_receipt: dict[str, Any] | None
    estimated_cells: int
    estimated_calls: dict[str, int]
    estimated_cost_usd: float
    eligible: bool
    blockers: tuple[str, ...]
    preview_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(_json_value(asdict(self)), preserve_false=True)


@dataclass(frozen=True)
class ExperimentRecordV1:
    schema_version: int
    id: str
    study_id: str
    campaign_id: str
    state: str
    draft: dict[str, Any]
    preview: dict[str, Any]
    approval: dict[str, Any] | None
    parent_experiment_ids: tuple[str, ...]
    proposal: dict[str, Any] | None
    plan: dict[str, Any] | None
    task_suite_lock: dict[str, Any] | None
    prepared_plan: dict[str, Any] | None
    admission: dict[str, Any] | None
    run_id: str | None
    outcome: dict[str, Any] | None
    evaluation: dict[str, Any] | None
    analysis: dict[str, Any] | None
    error: dict[str, Any] | None
    created_at: str
    updated_at: str
    record_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(_json_value(asdict(self)), preserve_false=True)


@dataclass(frozen=True)
class ExperimentEventV1:
    schema_version: int
    event_id: str
    sequence: int
    study_id: str
    experiment_id: str
    state: str
    event_type: str
    message: str
    artifact_type: str | None
    artifact_digest: str | None
    created_at: str
    event_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(_json_value(asdict(self)))


# Public research vocabulary.  These aliases deliberately preserve the persisted
# V1 field names and digests: the former programme-level Study is presented as a
# Research record, while one controlled Experiment is presented as a Study.
ResearchV1 = StudyV1
ResearchContextV1 = StudyContextV1
ControlledStudyDraftV1 = ExperimentDraftV1
ControlledStudyPreviewV1 = ExperimentPreviewV1
ControlledStudyV1 = ExperimentRecordV1
ControlledStudyEventV1 = ExperimentEventV1


def study_from_dict(raw: Mapping[str, Any]) -> StudyV1:
    fields = {item.name for item in StudyV1.__dataclass_fields__.values()}
    _reject_unknown(raw, fields, "study")
    study = StudyV1(
        schema_version=_schema(raw, "study"),
        id=validate_id(raw.get("id") or "", kind="study id"),
        title=_text(raw.get("title"), "study title", 300),
        campaign_id=validate_id(raw.get("campaign_id") or "", kind="campaign id"),
        brief=brief_from_dict(_mapping(raw.get("brief"), "study brief")),
        revision=_positive_int(raw.get("revision"), "study revision"),
        notes=tuple(
            note_from_dict(item) for item in _mappings(raw.get("notes"), "notes")
        ),
        resources=tuple(
            resource_from_dict(item)
            for item in _mappings(raw.get("resources"), "resources")
        ),
        results=tuple(
            result_from_dict(item) for item in _mappings(raw.get("results"), "results")
        ),
        experiments=tuple(
            study_experiment_ref_from_dict(item)
            for item in _mappings(raw.get("experiments"), "experiments")
        ),
        run_refs=_evidence_refs(raw.get("run_refs"), "run refs"),
        baseline_refs=_evidence_refs(raw.get("baseline_refs"), "baseline refs"),
        primary_baseline_ref=(
            evidence_ref_from_dict(
                _mapping(raw["primary_baseline_ref"], "primary baseline")
            )
            if raw.get("primary_baseline_ref")
            else None
        ),
        parent_study_ids=_ids(
            raw.get("parent_study_ids"), "parent study", allow_empty=True
        ),
        created_at=_timestamp(raw.get("created_at"), "created at"),
        updated_at=_timestamp(raw.get("updated_at"), "updated at"),
        created_by=attribution_from_dict(_mapping(raw.get("created_by"), "created by")),
        updated_by=attribution_from_dict(_mapping(raw.get("updated_by"), "updated by")),
        study_digest=str(raw.get("study_digest") or ""),
    )
    if study.primary_baseline_ref and _ref_key(study.primary_baseline_ref) not in {
        _ref_key(item) for item in study.baseline_refs
    }:
        raise ValueError("primary baseline must be one of baseline_refs")
    if study.id in study.parent_study_ids:
        raise ValueError("a Study cannot parent itself")
    _unique([item.id for item in study.notes], "study note")
    _unique([item.id for item in study.resources], "study resource")
    _unique([item.id for item in study.results], "study result")
    _unique([item.experiment_id for item in study.experiments], "study experiment")
    digest = _digest(study.to_dict(), "study_digest")
    if study.study_digest and study.study_digest != digest:
        raise ValueError("study_digest does not match the study")
    return replace(study, study_digest=digest)


def brief_from_dict(raw: Mapping[str, Any]) -> StudyBriefV1:
    fields = {*_STUDY_FIELDS, "provenance"}
    _reject_unknown(raw, fields, "study brief")
    provenance_raw = _mapping(raw.get("provenance") or {}, "brief provenance")
    if set(provenance_raw) - _STUDY_FIELDS:
        raise ValueError("brief provenance names an unknown field")
    return StudyBriefV1(
        question=_text(raw.get("question"), "research question", 8000),
        background=_optional_text(raw.get("background"), "background", 16000),
        plan=_optional_text(raw.get("plan"), "plan", 16000),
        findings=_optional_text(raw.get("findings"), "findings", 16000),
        conclusion=_optional_text(raw.get("conclusion"), "conclusion", 16000),
        next_questions=_texts(
            raw.get("next_questions"), "next question", allow_empty=True
        ),
        provenance={
            key: _evidence_refs(value, f"{key} provenance")
            for key, value in provenance_raw.items()
        },
    )


def attribution_from_dict(raw: Mapping[str, Any]) -> AttributionV1:
    _reject_unknown(raw, {"actor_type", "name", "identity"}, "attribution")
    actor_type = str(raw.get("actor_type") or "human")
    if actor_type not in {"human", "agent", "service"}:
        raise ValueError("attribution actor_type must be human, agent, or service")
    return AttributionV1(
        actor_type=actor_type,  # type: ignore[arg-type]
        name=_text(raw.get("name") or "unknown", "attribution name", 300),
        identity=_optional(raw.get("identity"), "attribution identity", 1000),
    )


def evidence_ref_from_dict(raw: Mapping[str, Any]) -> EvidenceRefV1:
    _reject_unknown(
        raw,
        {"kind", "ref", "digest", "version", "uri", "selector"},
        "evidence ref",
    )
    kind = str(raw.get("kind") or "")
    allowed = {
        "outcome",
        "analysis",
        "evaluation",
        "run",
        "conversation",
        "trace_audit",
        "artifact",
        "resource",
        "note",
        "result",
    }
    if kind not in allowed:
        raise ValueError(f"unknown evidence kind: {kind}")
    digest = _optional(raw.get("digest"), "evidence digest", 64)
    if digest and not _DIGEST.fullmatch(digest):
        raise ValueError("evidence digest must be sha256")
    return EvidenceRefV1(
        kind=kind,  # type: ignore[arg-type]
        ref=_text(raw.get("ref"), "evidence ref", 2000),
        digest=digest,
        version=_optional(raw.get("version"), "evidence version", 300),
        uri=_optional(raw.get("uri"), "evidence uri", 4000),
        selector=_json_mapping(raw.get("selector") or {}, "evidence selector"),
    )


def note_from_dict(raw: Mapping[str, Any]) -> StudyNoteV1:
    fields = {item.name for item in StudyNoteV1.__dataclass_fields__.values()}
    _reject_unknown(raw, fields, "study note")
    return StudyNoteV1(
        id=validate_id(raw.get("id") or "", kind="note id"),
        revision=_positive_int(raw.get("revision"), "note revision"),
        text=_text(raw.get("text"), "note text", 64000),
        kind=_text(raw.get("kind") or "observation", "note kind", 100),
        sources=_evidence_refs(raw.get("sources"), "note sources"),
        created_at=_timestamp(raw.get("created_at"), "note created at"),
        attribution=attribution_from_dict(
            _mapping(raw.get("attribution"), "note attribution")
        ),
        supersedes=_optional_id(raw.get("supersedes"), "superseded note"),
    )


def resource_from_dict(raw: Mapping[str, Any]) -> StudyResourceV1:
    fields = {item.name for item in StudyResourceV1.__dataclass_fields__.values()}
    _reject_unknown(raw, fields, "study resource")
    digest = _optional(raw.get("digest"), "resource digest", 64)
    if digest and not _DIGEST.fullmatch(digest):
        raise ValueError("resource digest must be sha256")
    return StudyResourceV1(
        id=validate_id(raw.get("id") or "", kind="resource id"),
        uri=_text(raw.get("uri"), "resource uri", 4000),
        kind=_text(raw.get("kind") or "other", "resource kind", 100),
        digest=digest,
        version=_optional(raw.get("version"), "resource version", 300),
        title=_optional_text(raw.get("title"), "resource title", 300),
        summary=_optional_text(raw.get("summary"), "resource summary", 4000),
        added_revision=_positive_int(raw.get("added_revision"), "resource revision"),
        added_at=_timestamp(raw.get("added_at"), "resource added at"),
        attribution=attribution_from_dict(
            _mapping(raw.get("attribution"), "resource attribution")
        ),
    )


def result_from_dict(raw: Mapping[str, Any]) -> StudyResultV1:
    fields = {item.name for item in StudyResultV1.__dataclass_fields__.values()}
    _reject_unknown(raw, fields, "study result")
    estimate = (
        estimate_from_dict(_mapping(raw["estimate"], "result estimate"))
        if raw.get("estimate") is not None
        else None
    )
    comparison = (
        comparison_from_dict(_mapping(raw["comparison"], "result comparison"))
        if raw.get("comparison") is not None
        else None
    )
    sources = _evidence_refs(raw.get("sources"), "result sources")
    if not sources:
        raise ValueError("a StudyResult requires at least one evidence source")
    analysis_source = (
        evidence_ref_from_dict(_mapping(raw["analysis_source"], "analysis source"))
        if raw.get("analysis_source")
        else None
    )
    source_keys = {_ref_key(item) for item in sources}
    special = [analysis_source]
    if comparison:
        special.extend(comparison.condition_sources)
        special.extend(comparison.comparator_sources)
    if any(item and _ref_key(item) not in source_keys for item in special):
        raise ValueError("comparison and analysis sources must also appear in sources")
    sample_size = raw.get("sample_size")
    if sample_size is not None and (
        isinstance(sample_size, bool) or int(sample_size) < 0
    ):
        raise ValueError("sample_size must be a non-negative integer")
    return StudyResultV1(
        id=validate_id(raw.get("id") or "", kind="result id"),
        revision=_positive_int(raw.get("revision"), "result revision"),
        statement=_text(raw.get("statement"), "result statement", 16000),
        kind=_text(raw.get("kind") or "result", "result kind", 100),
        outcome=_optional_text(raw.get("outcome"), "result outcome", 1000),
        estimate=estimate,
        comparison=comparison,
        population=_optional_text(raw.get("population"), "result population", 2000),
        conditions=_json_mapping(raw.get("conditions") or {}, "result conditions"),
        sample_size=int(sample_size) if sample_size is not None else None,
        aggregation=_optional_text(raw.get("aggregation"), "result aggregation", 1000),
        exclusions=_texts(raw.get("exclusions"), "result exclusion", allow_empty=True),
        sources=sources,
        analysis_source=analysis_source,
        created_at=_timestamp(raw.get("created_at"), "result created at"),
        attribution=attribution_from_dict(
            _mapping(raw.get("attribution"), "result attribution")
        ),
        supersedes=_optional_id(raw.get("supersedes"), "superseded result"),
    )


def estimate_from_dict(raw: Mapping[str, Any]) -> ResultEstimateV1:
    _reject_unknown(raw, {"value", "kind", "unit", "uncertainty"}, "result estimate")
    value = raw.get("value")
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("result estimate value must be a string or finite number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("result estimate value must be finite")
    return ResultEstimateV1(
        value=value,
        kind=_optional_text(raw.get("kind"), "estimate kind", 100),
        unit=_optional_text(raw.get("unit"), "estimate unit", 100),
        uncertainty=(
            uncertainty_from_dict(_mapping(raw["uncertainty"], "uncertainty"))
            if raw.get("uncertainty")
            else None
        ),
    )


def uncertainty_from_dict(raw: Mapping[str, Any]) -> ResultUncertaintyV1:
    _reject_unknown(
        raw,
        {"kind", "value", "lower", "upper", "level", "description"},
        "result uncertainty",
    )
    values = {
        key: _finite_optional(raw.get(key), key)
        for key in ("value", "lower", "upper", "level")
    }
    if (
        values["lower"] is not None
        and values["upper"] is not None
        and values["lower"] > values["upper"]
    ):
        raise ValueError("uncertainty lower cannot exceed upper")
    if values["level"] is not None and not 0 < values["level"] <= 1:
        raise ValueError("uncertainty level must be in (0, 1]")
    return ResultUncertaintyV1(
        kind=_text(raw.get("kind"), "uncertainty kind", 100),
        value=values["value"],
        lower=values["lower"],
        upper=values["upper"],
        level=values["level"],
        description=_optional_text(
            raw.get("description"), "uncertainty description", 1000
        ),
    )


def comparison_from_dict(raw: Mapping[str, Any]) -> ResultComparisonV1:
    _reject_unknown(
        raw,
        {"condition", "comparator", "condition_sources", "comparator_sources"},
        "result comparison",
    )
    condition_sources = _evidence_refs(
        raw.get("condition_sources"), "condition sources"
    )
    comparator_sources = _evidence_refs(
        raw.get("comparator_sources"), "comparator sources"
    )
    if not condition_sources or not comparator_sources:
        raise ValueError("result comparison requires sources for both arms")
    return ResultComparisonV1(
        condition=_text(raw.get("condition"), "comparison condition", 1000),
        comparator=_text(raw.get("comparator"), "comparison comparator", 1000),
        condition_sources=condition_sources,
        comparator_sources=comparator_sources,
    )


def study_experiment_ref_from_dict(raw: Mapping[str, Any]) -> StudyExperimentRefV1:
    fields = {item.name for item in StudyExperimentRefV1.__dataclass_fields__.values()}
    _reject_unknown(raw, fields, "study experiment ref")
    return StudyExperimentRefV1(
        experiment_id=validate_id(
            raw.get("experiment_id") or "", kind="experiment record id"
        ),
        state=_text(raw.get("state"), "experiment state", 100),
        proposal_digest=_optional_digest(raw.get("proposal_digest"), "proposal digest"),
        preview_digest=_required_digest(raw.get("preview_digest"), "preview digest"),
        task_suite_digest=_optional_digest(
            raw.get("task_suite_digest"), "task suite digest"
        ),
        plan_digest=_optional_digest(raw.get("plan_digest"), "plan digest"),
        parent_experiment_ids=_ids(
            raw.get("parent_experiment_ids"), "parent experiment", allow_empty=True
        ),
        run_id=_optional_id(raw.get("run_id"), "run id"),
        outcome_digest=_optional_digest(raw.get("outcome_digest"), "outcome digest"),
        evaluation_digest=_optional_digest(
            raw.get("evaluation_digest"), "evaluation digest"
        ),
        analysis_digest=_optional_digest(raw.get("analysis_digest"), "analysis digest"),
        updated_at=_timestamp(raw.get("updated_at"), "experiment updated at"),
    )


def study_update_from_dict(raw: Mapping[str, Any]) -> StudyUpdateV1:
    fields = {item.name for item in StudyUpdateV1.__dataclass_fields__.values()}
    _reject_unknown(raw, fields, "study update")
    brief_patch = _json_mapping(raw.get("brief_patch") or {}, "brief patch")
    if set(brief_patch) - _STUDY_FIELDS:
        raise ValueError("brief patch names an unknown field")
    brief_sources_raw = _mapping(raw.get("brief_sources") or {}, "brief sources")
    if set(brief_sources_raw) - _STUDY_FIELDS:
        raise ValueError("brief sources name an unknown field")
    baseline = raw.get("baseline_refs")
    return StudyUpdateV1(
        message=_optional(raw.get("message"), "study note", 64000),
        note_kind=_text(raw.get("note_kind") or "observation", "note kind", 100),
        note_sources=_evidence_refs(raw.get("note_sources"), "note sources"),
        note_supersedes=_optional_id(raw.get("note_supersedes"), "superseded note"),
        brief_patch=brief_patch,
        brief_sources={
            key: _evidence_refs(value, f"{key} sources")
            for key, value in brief_sources_raw.items()
        },
        resources=tuple(_mappings(raw.get("resources"), "resources")),
        results=tuple(_mappings(raw.get("results"), "results")),
        run_refs=_evidence_refs(raw.get("run_refs"), "run refs"),
        baseline_refs=(
            _evidence_refs(baseline, "baseline refs") if baseline is not None else None
        ),
        primary_baseline_ref=(
            evidence_ref_from_dict(
                _mapping(raw["primary_baseline_ref"], "primary baseline")
            )
            if raw.get("primary_baseline_ref")
            else None
        ),
        attribution=attribution_from_dict(
            _mapping(raw.get("attribution") or {}, "attribution")
        ),
    )


def experiment_draft_from_dict(
    raw: Mapping[str, Any], *, require_digest: bool = True
) -> ExperimentDraftV1:
    fields = {item.name for item in ExperimentDraftV1.__dataclass_fields__.values()}
    _reject_unknown(raw, fields, "experiment draft")
    if raw.get("task_suite_digest") and raw.get("task_suite_draft"):
        raise ValueError(
            "experiment draft accepts a task suite digest or draft, not both"
        )
    task_digest = _optional_digest(raw.get("task_suite_digest"), "task suite digest")
    draft = ExperimentDraftV1(
        schema_version=_schema(raw, "experiment draft"),
        study_id=validate_id(raw.get("study_id") or "", kind="study id"),
        campaign_id=validate_id(raw.get("campaign_id") or "", kind="campaign id"),
        proposal_id=validate_id(raw.get("proposal_id") or "", kind="proposal id"),
        stage_id=validate_id(raw.get("stage_id") or "", kind="stage id"),
        question=_text(raw.get("question"), "experiment question", 8000),
        hypothesis=_text(raw.get("hypothesis"), "experiment hypothesis", 8000),
        fixed_dimensions=_texts(raw.get("fixed_dimensions"), "fixed dimension"),
        varied_dimensions=_texts(raw.get("varied_dimensions"), "varied dimension"),
        measured_dimensions=_texts(
            raw.get("measured_dimensions"), "measured dimension"
        ),
        experiment_id=validate_id(
            raw.get("experiment_id") or "", kind="registered experiment id"
        ),
        model=_text(raw.get("model"), "model", 300),
        n_attempts=_positive_int(raw.get("n_attempts"), "attempts"),
        n_concurrent=_positive_int(raw.get("n_concurrent"), "concurrency"),
        preset_id=_optional_id(raw.get("preset_id"), "preset id"),
        workloads=_ids(raw.get("workloads"), "workload", allow_empty=True),
        harnesses=_ids(raw.get("harnesses"), "harness", allow_empty=True),
        context_systems=_ids(
            raw.get("context_systems"), "context system", allow_empty=True
        ),
        variants=_ids(raw.get("variants"), "variant", allow_empty=True),
        n_tasks=(
            _positive_int(raw.get("n_tasks"), "tasks")
            if raw.get("n_tasks") is not None
            else None
        ),
        trace_content=_text(raw.get("trace_content") or "full", "trace content", 32),
        analysis_ids=_ids(raw.get("analysis_ids"), "analysis", allow_empty=True),
        parent_experiment_ids=_ids(
            raw.get("parent_experiment_ids"), "parent experiment", allow_empty=True
        ),
        parent_outcome_id=_optional_id(raw.get("parent_outcome_id"), "parent outcome"),
        decision_rationale=_optional_text(
            raw.get("decision_rationale"), "decision rationale", 8000
        ),
        task_suite_digest=task_digest,
        task_suite_draft=(
            _mapping(raw["task_suite_draft"], "task suite draft")
            if raw.get("task_suite_draft") is not None
            else None
        ),
        task_recipe_preview=(
            _mapping(raw["task_recipe_preview"], "task recipe preview")
            if raw.get("task_recipe_preview") is not None
            else None
        ),
        candidate_refs=tuple(
            _mapping(item, "candidate reference")
            for item in _mappings(raw.get("candidate_refs"), "candidate references")
        ),
        scoring_revision=(
            _mapping(raw["scoring_revision"], "scoring revision")
            if raw.get("scoring_revision") is not None
            else None
        ),
        task_analysis_id=_optional_id(raw.get("task_analysis_id"), "task analysis id"),
        draft_digest=str(raw.get("draft_digest") or ""),
    )
    digest = _digest(draft.to_dict(), "draft_digest")
    if require_digest and draft.draft_digest != digest:
        raise ValueError("draft_digest does not match the experiment draft")
    if draft.draft_digest and draft.draft_digest != digest:
        raise ValueError("draft_digest does not match the experiment draft")
    return replace(draft, draft_digest=digest)


def build_experiment_draft(**values: Any) -> ExperimentDraftV1:
    raw = {"schema_version": RESEARCH_SCHEMA_VERSION, **values, "draft_digest": ""}
    return experiment_draft_from_dict(raw, require_digest=False)


def experiment_preview_from_dict(raw: Mapping[str, Any]) -> ExperimentPreviewV1:
    fields = {item.name for item in ExperimentPreviewV1.__dataclass_fields__.values()}
    _reject_unknown(raw, fields, "experiment preview")
    preview = ExperimentPreviewV1(
        schema_version=_schema(raw, "experiment preview"),
        study_id=validate_id(raw.get("study_id") or "", kind="study id"),
        experiment_id=validate_id(
            raw.get("experiment_id") or "", kind="experiment record id"
        ),
        campaign_id=validate_id(raw.get("campaign_id") or "", kind="campaign id"),
        catalog_digest=_required_digest(raw.get("catalog_digest"), "catalog digest"),
        policy_digest=_required_digest(raw.get("policy_digest"), "policy digest"),
        draft=_mapping(raw.get("draft"), "experiment draft"),
        task_suite_preview=(
            _mapping(raw["task_suite_preview"], "task suite preview")
            if raw.get("task_suite_preview")
            else None
        ),
        plan_receipt=(
            _mapping(raw["plan_receipt"], "plan receipt")
            if raw.get("plan_receipt")
            else None
        ),
        estimated_cells=_non_negative_int(
            raw.get("estimated_cells"), "estimated cells"
        ),
        estimated_calls={
            key: _non_negative_int(value, f"{key} calls")
            for key, value in _mapping(
                raw.get("estimated_calls") or {}, "estimated calls"
            ).items()
        },
        estimated_cost_usd=_non_negative_number(
            raw.get("estimated_cost_usd"), "estimated cost"
        ),
        eligible=_strict_bool(raw.get("eligible"), "preview eligible"),
        blockers=_texts(raw.get("blockers"), "preview blocker", allow_empty=True),
        preview_digest=_required_digest(raw.get("preview_digest"), "preview digest"),
    )
    _verify_digest(preview.to_dict(), "preview_digest", "experiment preview")
    experiment_draft_from_dict(preview.draft)
    return preview


def experiment_record_from_dict(raw: Mapping[str, Any]) -> ExperimentRecordV1:
    fields = {item.name for item in ExperimentRecordV1.__dataclass_fields__.values()}
    _reject_unknown(raw, fields, "experiment record")
    record = ExperimentRecordV1(
        schema_version=_schema(raw, "experiment record"),
        id=validate_id(raw.get("id") or "", kind="experiment record id"),
        study_id=validate_id(raw.get("study_id") or "", kind="study id"),
        campaign_id=validate_id(raw.get("campaign_id") or "", kind="campaign id"),
        state=_text(raw.get("state"), "experiment state", 100),
        draft=_mapping(raw.get("draft"), "experiment draft"),
        preview=_mapping(raw.get("preview"), "experiment preview"),
        approval=_optional_mapping(raw.get("approval"), "execution approval"),
        parent_experiment_ids=_ids(
            raw.get("parent_experiment_ids"), "parent experiment", allow_empty=True
        ),
        proposal=_optional_mapping(raw.get("proposal"), "proposal"),
        plan=_optional_mapping(raw.get("plan"), "plan"),
        task_suite_lock=_optional_mapping(
            raw.get("task_suite_lock"), "task suite lock"
        ),
        prepared_plan=_optional_mapping(raw.get("prepared_plan"), "prepared plan"),
        admission=_optional_mapping(raw.get("admission"), "admission"),
        run_id=_optional_id(raw.get("run_id"), "run id"),
        outcome=_optional_mapping(raw.get("outcome"), "outcome"),
        evaluation=_optional_mapping(raw.get("evaluation"), "evaluation"),
        analysis=_optional_mapping(raw.get("analysis"), "analysis"),
        error=_optional_mapping(raw.get("error"), "error"),
        created_at=_timestamp(raw.get("created_at"), "created at"),
        updated_at=_timestamp(raw.get("updated_at"), "updated at"),
        record_digest=_required_digest(raw.get("record_digest"), "record digest"),
    )
    _verify_digest(record.to_dict(), "record_digest", "experiment record")
    experiment_draft_from_dict(record.draft)
    experiment_preview_from_dict(record.preview)
    return record


def experiment_event_from_dict(raw: Mapping[str, Any]) -> ExperimentEventV1:
    fields = {item.name for item in ExperimentEventV1.__dataclass_fields__.values()}
    _reject_unknown(raw, fields, "experiment event")
    event = ExperimentEventV1(
        schema_version=_schema(raw, "experiment event"),
        event_id=validate_id(raw.get("event_id") or "", kind="event id"),
        sequence=_positive_int(raw.get("sequence"), "event sequence"),
        study_id=validate_id(raw.get("study_id") or "", kind="study id"),
        experiment_id=validate_id(raw.get("experiment_id") or "", kind="experiment id"),
        state=_text(raw.get("state"), "event state", 100),
        event_type=_text(raw.get("event_type"), "event type", 100),
        message=_optional_text(raw.get("message"), "event message", 4000),
        artifact_type=_optional(raw.get("artifact_type"), "artifact type", 200),
        artifact_digest=_optional_digest(raw.get("artifact_digest"), "artifact digest"),
        created_at=_timestamp(raw.get("created_at"), "event created at"),
        event_digest=_required_digest(raw.get("event_digest"), "event digest"),
    )
    _verify_digest(event.to_dict(), "event_digest", "experiment event")
    return event


def sign_study(study: StudyV1) -> StudyV1:
    return replace(study, study_digest=_digest(study.to_dict(), "study_digest"))


def sign_preview(preview: ExperimentPreviewV1) -> ExperimentPreviewV1:
    return replace(preview, preview_digest=_digest(preview.to_dict(), "preview_digest"))


def sign_record(record: ExperimentRecordV1) -> ExperimentRecordV1:
    return replace(record, record_digest=_digest(record.to_dict(), "record_digest"))


def sign_event(event: ExperimentEventV1) -> ExperimentEventV1:
    return replace(event, event_digest=_digest(event.to_dict(), "event_digest"))


def sign_context(context: StudyContextV1) -> StudyContextV1:
    return replace(context, context_digest=_digest(context.to_dict(), "context_digest"))


def now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _schema(raw: Mapping[str, Any], label: str) -> int:
    value = raw.get("schema_version")
    if isinstance(value, bool) or int(value or 0) != RESEARCH_SCHEMA_VERSION:
        raise ValueError(f"{label} must use schema_version 1")
    return RESEARCH_SCHEMA_VERSION


def _digest(value: Mapping[str, Any], field: str) -> str:
    unsigned = {**value, field: ""}
    return stable_digest(unsigned)


def _verify_digest(value: Mapping[str, Any], field: str, label: str) -> None:
    if value.get(field) != _digest(value, field):
        raise ValueError(f"{field} does not match the {label}")


def _required_digest(value: Any, label: str) -> str:
    result = str(value or "")
    if not _DIGEST.fullmatch(result):
        raise ValueError(f"{label} must be sha256")
    return result


def _optional_digest(value: Any, label: str) -> str | None:
    if value in (None, ""):
        return None
    return _required_digest(value, label)


def _text(value: Any, label: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if len(value) > limit:
        raise ValueError(f"{label} exceeds {limit} characters")
    return value.strip()


def _optional_text(value: Any, label: str, limit: int) -> str:
    if value in (None, ""):
        return ""
    return _text(value, label, limit)


def _optional(value: Any, label: str, limit: int) -> str | None:
    if value in (None, ""):
        return None
    return _text(value, label, limit)


def _optional_id(value: Any, label: str) -> str | None:
    if value in (None, ""):
        return None
    return validate_id(str(value), kind=label)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    result = int(value or 0)
    if result < 1:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    result = int(value or 0)
    if result < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return result


def _strict_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _finite_optional(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _non_negative_number(value: Any, label: str) -> float:
    result = _finite_optional(value, label)
    if result is None or result < 0:
        raise ValueError(f"{label} must be a non-negative finite number")
    return result


def _timestamp(value: Any, label: str) -> str:
    text = _text(value, label, 100)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be ISO 8601") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return text


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _optional_mapping(value: Any, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    return _mapping(value, label)


def _mappings(value: Any, label: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    return [_mapping(item, label) for item in value]


def _texts(value: Any, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if value is None:
        value = []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    result = tuple(_text(item, label, 4000) for item in value)
    if not result and not allow_empty:
        raise ValueError(f"{label} must not be empty")
    _unique(result, label)
    return result


def _ids(value: Any, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    values = _texts(value, label, allow_empty=allow_empty)
    return tuple(validate_id(item, kind=label) for item in values)


def _evidence_refs(value: Any, label: str) -> tuple[EvidenceRefV1, ...]:
    refs = tuple(evidence_ref_from_dict(item) for item in _mappings(value, label))
    _unique([_ref_key(item) for item in refs], label)
    return refs


def _ref_key(value: EvidenceRefV1) -> str:
    return json.dumps(value.to_dict(), sort_keys=True, separators=(",", ":"))


def _unique(values: Sequence[Any], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} values must be unique")


def _json_mapping(value: Any, label: str) -> dict[str, JsonValue]:
    result = _mapping(value, label)
    return {str(key): _json_value(item) for key, item in result.items()}


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise ValueError(f"unsupported JSON value: {type(value).__name__}")


def _drop_empty(
    value: dict[str, Any], *, preserve_false: bool = False
) -> dict[str, Any]:
    empty = (None, "", [], {}, ()) if preserve_false else (None, "", [], {}, (), False)
    return {key: item for key, item in value.items() if item not in empty}


def _reject_unknown(raw: Mapping[str, Any], fields: set[str], label: str) -> None:
    unknown = sorted(set(raw) - fields)
    if unknown:
        raise ValueError(f"{label} contains unknown field(s): {', '.join(unknown)}")
