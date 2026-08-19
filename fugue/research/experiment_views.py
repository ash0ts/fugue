from __future__ import annotations

import math
import urllib.parse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from fugue.bench.analysis_contracts import (
    aligned_analysis_from_dict,
    evidence_topology_from_dict,
    lock_descriptor_from_dict,
    superseded_result_from_dict,
    task_validity_from_dict,
)
from fugue.bench.candidates import attempt_id as canonical_attempt_id
from fugue.bench.candidates import stable_digest
from fugue.bench.local_evidence import LocalEvidenceDestinationV1
from fugue.redaction import redact_text
from fugue.research.display_labels import humanize_display_id

# Existing design/progress builders and V2 result projections keep writing the
# last canonical shape they implement.  V3 evaluation projections opt in with
# an explicit schema version only after supplying every V3-required contract.
EXPERIMENT_VIEW_SCHEMA_VERSION = 2
EXPERIMENT_VIEW_V3_SCHEMA_VERSION = 3
EXPERIMENT_VIEW_READABLE_SCHEMA_VERSIONS = frozenset({1, 2, 3})
EXPERIMENT_VIEW_CELL_LIMIT = 256

ExperimentViewKind = Literal["design", "progress", "evaluation"]
ExecutionStatus = Literal[
    "queued",
    "preparing",
    "running",
    "completed",
    "failed",
    "cancelled",
    "interrupted",
    "not_applicable",
]
OutcomeStatus = Literal["pending", "passed", "failed", "unavailable", "not_applicable"]
EvidenceStatus = Literal["pending", "reconciled", "missing", "not_applicable"]
SummaryStatus = Literal["passed", "failed", "unavailable", "not_applicable"]
ScoreStatus = Literal["passed", "failed", "observed", "unavailable", "not_applicable"]

_EXECUTION_STATES = {
    "queued",
    "preparing",
    "running",
    "completed",
    "failed",
    "cancelled",
    "interrupted",
    "not_applicable",
}
_OUTCOME_STATES = {"pending", "passed", "failed", "unavailable", "not_applicable"}
_EVIDENCE_STATES = {"pending", "reconciled", "missing", "not_applicable"}
_VIEW_KINDS = {"design", "progress", "evaluation"}
_RECONCILIATION_STATUSES = {"resolved", "unresolved", "unavailable"}
_PROVISIONAL_V2_EVALUATION_FIELDS = frozenset(
    {
        "aligned_comparisons",
        "arm_totals",
        "cell_limit",
        "cells",
        "mechanism_funnel",
        "omitted_cells",
        "outcome_summaries",
        "score_summaries",
    }
)
_SAFE_BEHAVIORAL_MEASURES = (
    "agent_runtime_completed",
    "context_invoked",
    "localization_recall_at_5",
    "localization_recall_at_10",
    "localization_mrr",
    "relevant_retrieval_returned",
    "relevant_retrieval_opened",
    "relevant_retrieval_open_rate",
    "relevant_retrieval_used",
    "relevant_retrieval_change_rate",
    "off_target_change_only",
    "premature_completion",
    "document_search_available",
    "document_search_invoked",
    "relevant_document_returned",
    "relevant_document_opened",
    "current_document_cited",
    "current_document_used",
    "artifact_schema_valid",
    "answer_facts_correct",
    "unsupported_claims_absent",
    "skill_ids_invoked",
    "integration_ids_invoked",
    "mcp_tool_names",
    "mcp_queried_projects",
    "prompt_injection_action_gate_allowed",
    "prompt_injection_action_gate_blocked",
    "prompt_injection_attack_encountered",
    "prompt_injection_compromised",
    "prompt_injection_evidence_preserved",
    "prompt_injection_false_positive_refusal",
    "prompt_injection_incorrect",
    "prompt_injection_safe_and_useful",
    "prompt_injection_safe_but_failed_or_refused",
    "prompt_injection_sensitive_action_attempted",
    "prompt_injection_task_complete",
    "tool_calls",
    "turns",
    "input_tokens",
    "output_tokens",
    "wall_time_sec",
)


@dataclass(frozen=True)
class ExperimentDescriptorV1:
    id: str
    label: str
    digest: str | None = None
    details: dict[str, str | int | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self))


@dataclass(frozen=True)
class ExperimentEvidenceScopeV1:
    entity: str
    project: str
    evidence_types: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self))


@dataclass(frozen=True)
class ExperimentFactorV1:
    name: str
    levels: tuple[str, ...]
    label: str | None = None
    level_labels: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self))


@dataclass(frozen=True)
class ExperimentTreatmentArmV1:
    id: str
    label: str
    factor_levels: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self))


@dataclass(frozen=True)
class ExperimentOutcomeSummaryV1:
    id: str
    label: str
    status: SummaryStatus
    passed: int | None = None
    total: int | None = None
    unavailable: int = 0

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass(frozen=True)
class ExperimentTaskDesignV1:
    title: str
    summary: str
    interaction_mode: str | None = None
    tools: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()
    evidence_links: tuple[dict[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass(frozen=True)
class ExperimentPromptDesignV1:
    base_instruction_summary: str
    treatment_summaries: dict[str, str] = field(default_factory=dict)
    evidence_links: tuple[dict[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass(frozen=True)
class ExperimentScoreDefinitionV1:
    id: str
    label: str
    description: str | None = None
    source_key: str | None = None
    target: str | float | int | bool | None = None
    primary: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass(frozen=True)
class ExperimentScorerDesignV1:
    id: str
    label: str
    kind: Literal["benchmark", "deterministic", "criteria", "llm_judge"]
    description: str
    required: bool
    threshold: float | None = None
    aggregation: str | None = None
    evidence_inputs: tuple[str, ...] = ()
    revision: str | None = None
    model: str | None = None
    rubric_summary: str | None = None
    blind_fields: tuple[str, ...] = ()
    dimensions: tuple[ExperimentScoreDefinitionV1, ...] = ()
    evidence_links: tuple[dict[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass(frozen=True)
class ExperimentEvaluationDesignV1:
    pass_rule: str
    scorers: tuple[ExperimentScorerDesignV1, ...]
    llm_judge_used: bool

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass(frozen=True)
class ExperimentScoreResultV1:
    id: str
    label: str
    status: ScoreStatus
    value: str | float | int | bool | None = None
    scorer_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass(frozen=True)
class ExperimentScoreSummaryV1:
    id: str
    label: str
    observed: int
    passed: int | None = None
    failed: int | None = None
    unavailable: int = 0
    mean: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass(frozen=True)
class ExperimentMechanismArmV1:
    arm: str
    harness: str
    eligible: int
    reached: int

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass(frozen=True)
class ExperimentMechanismStageV1:
    id: str
    label: str
    eligible: int
    reached: int
    by_arm: tuple[ExperimentMechanismArmV1, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass(frozen=True)
class ExperimentCellViewV1:
    cell_id: str
    task_label: str
    factor_levels: dict[str, str]
    attempt: int
    execution_status: ExecutionStatus
    task_outcome: OutcomeStatus
    evaluation_status: OutcomeStatus
    evidence_status: EvidenceStatus
    reason_code: str | None = None
    cost_usd: float | None = None
    latency_sec: float | None = None
    evidence_links: tuple[dict[str, str], ...] = ()
    measures: dict[str, str | int | float | bool | None] = field(default_factory=dict)
    scores: tuple[ExperimentScoreResultV1, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass(frozen=True)
class ExperimentViewV1:
    schema_version: int
    kind: ExperimentViewKind
    research_label: str | None = None
    study_label: str | None = None
    question: str | None = None
    hypothesis: str | None = None
    context: str | None = None
    observation: str | None = None
    rationale: str | None = None
    alternative_explanations: tuple[str, ...] = ()
    success_definition: str | None = None
    task_design: ExperimentTaskDesignV1 | None = None
    prompt_design: ExperimentPromptDesignV1 | None = None
    evaluation_design: ExperimentEvaluationDesignV1 | None = None
    source_cohort: ExperimentDescriptorV1 | None = None
    fixed_conditions: tuple[ExperimentFactorV1, ...] = ()
    varied_factors: tuple[ExperimentFactorV1, ...] = ()
    treatment_arms: tuple[ExperimentTreatmentArmV1, ...] = ()
    measured_outcomes: tuple[str, ...] = ()
    taskset: ExperimentDescriptorV1 | None = None
    harnesses: tuple[ExperimentDescriptorV1, ...] = ()
    runtime: ExperimentDescriptorV1 | None = None
    matrix_size: int = 0
    preview_digest: str | None = None
    approval_state: str | None = None
    cell_limit: int | None = None
    reserved_cost_usd: float | None = None
    phase: str | None = None
    completed_cells: int | None = None
    observed_cost_usd: float | None = None
    state_counts: dict[str, int] = field(default_factory=dict)
    cells: tuple[ExperimentCellViewV1, ...] = ()
    omitted_cells: int = 0
    infrastructure_health: str | None = None
    arm_totals: tuple[dict[str, Any], ...] = ()
    aligned_comparisons: tuple[dict[str, Any], ...] = ()
    behavioral_measures: dict[str, Any] = field(default_factory=dict)
    mechanism_funnel: tuple[ExperimentMechanismStageV1, ...] = ()
    outcome_summaries: tuple[ExperimentOutcomeSummaryV1, ...] = ()
    score_summaries: tuple[ExperimentScoreSummaryV1, ...] = ()
    evidence_eligible: bool | None = None
    evidence_scope: ExperimentEvidenceScopeV1 | None = None
    limitations: tuple[str, ...] = ()
    evidence_links: tuple[dict[str, str], ...] = ()
    decision: dict[str, Any] | None = None
    integrity_status: str | None = None
    evidence_grade: str | None = None
    release_target: str | None = None
    candidate_sha: str | None = None
    release_note_coverage: tuple[dict[str, Any], ...] = ()
    infrastructure_gates: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass(frozen=True)
class ExperimentViewV2:
    """Strict canonical V2 evaluation view.

    V2 evaluation evidence is represented exactly once through ``paired_cases``.
    The legacy V1 cell and aggregate projections intentionally do not exist on
    this type.
    """

    schema_version: int
    kind: Literal["evaluation"]
    matrix_size: int
    infrastructure_health: str
    evidence_eligible: bool
    integrity_status: str
    evidence_grade: str
    behavioral_summary: dict[str, Any] | None
    paired_cases: tuple[dict[str, Any], ...]
    preview_digest: str | None = None
    approval_state: str | None = None
    phase: str | None = None
    completed_cells: int | None = None
    observed_cost_usd: float | None = None
    state_counts: dict[str, int] = field(default_factory=dict)
    evidence_scope: ExperimentEvidenceScopeV1 | None = None
    limitations: tuple[str, ...] = ()
    evidence_links: tuple[dict[str, str], ...] = ()
    decision: dict[str, Any] | None = None
    release_target: str | None = None
    candidate_sha: str | None = None
    release_note_coverage: tuple[dict[str, Any], ...] = ()
    infrastructure_gates: tuple[dict[str, Any], ...] = ()
    backend: str | None = None
    candidate_source_revisions: tuple[dict[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        serialized = _drop_empty(asdict(self), preserve_false=True)
        # ``paired_cases`` is the sole canonical V2 attempt surface. Preserve
        # the empty array explicitly so invalid evidence cannot be confused
        # with a producer that omitted the contract.
        serialized["paired_cases"] = [dict(item) for item in self.paired_cases]
        return serialized


@dataclass(frozen=True)
class ExperimentViewV3:
    """Strict, source-isolated evaluation view for decision-ready Studies."""

    schema_version: Literal[3]
    kind: Literal["evaluation"]
    matrix_size: int
    infrastructure_health: str
    evidence_eligible: bool
    integrity_status: str
    evidence_grade: str
    behavioral_summary: dict[str, Any] | None
    paired_cases: tuple[dict[str, Any], ...]
    evidence_topology: dict[str, Any]
    aligned_analysis: dict[str, Any]
    task_validity: tuple[dict[str, Any], ...]
    scorer_revisions: tuple[dict[str, Any], ...]
    runtime_locks: tuple[dict[str, Any], ...]
    result_digest: str | None = None
    qualification_digest: str | None = None
    runtime_lock_digest: str | None = None
    preview_digest: str | None = None
    approval_state: str | None = None
    phase: str | None = None
    completed_cells: int | None = None
    observed_cost_usd: float | None = None
    state_counts: dict[str, int] = field(default_factory=dict)
    evidence_scope: ExperimentEvidenceScopeV1 | None = None
    limitations: tuple[str, ...] = ()
    evidence_links: tuple[dict[str, str], ...] = ()
    decision: dict[str, Any] | None = None
    release_target: str | None = None
    candidate_sha: str | None = None
    release_note_coverage: tuple[dict[str, Any], ...] = ()
    infrastructure_gates: tuple[dict[str, Any], ...] = ()
    backend: str | None = None
    candidate_source_revisions: tuple[dict[str, str], ...] = ()
    supersedes: tuple[dict[str, str], ...] = ()
    judge_summary: dict[str, Any] = field(
        default_factory=lambda: {"status": "not_used"}
    )

    def to_dict(self) -> dict[str, Any]:
        serialized = _drop_empty(asdict(self), preserve_false=True)
        for name in (
            "paired_cases",
            "task_validity",
            "scorer_revisions",
            "runtime_locks",
            "supersedes",
        ):
            serialized[name] = [dict(item) for item in getattr(self, name)]
        return serialized


def _experiment_view_v3_from_dict(
    raw: Mapping[str, Any],
) -> ExperimentViewV3:
    _reject_unknown(
        raw,
        {item.name for item in ExperimentViewV3.__dataclass_fields__.values()},
        "V3 experiment view",
    )
    required = {
        "evidence_topology",
        "aligned_analysis",
        "task_validity",
        "scorer_revisions",
        "runtime_locks",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(
            "V3 evaluation view is missing required field(s): " + ", ".join(missing)
        )
    evidence_eligible = _optional_bool(
        raw.get("evidence_eligible"), "evidence_eligible"
    )
    if evidence_eligible is None:
        raise ValueError("V3 evaluation view requires evidence eligibility")
    topology = evidence_topology_from_dict(
        _mapping(raw.get("evidence_topology"), "evidence_topology")
    )
    aligned = aligned_analysis_from_dict(
        _mapping(raw.get("aligned_analysis"), "aligned_analysis")
    )
    task_validity = tuple(
        task_validity_from_dict(_mapping(item, "task_validity")).to_dict()
        for item in _sequence(raw.get("task_validity"), "task_validity")
    )
    scorer_revisions = tuple(
        lock_descriptor_from_dict(_mapping(item, "scorer_revision")).to_dict()
        for item in _sequence(raw.get("scorer_revisions"), "scorer_revisions")
    )
    runtime_locks = tuple(
        lock_descriptor_from_dict(_mapping(item, "runtime_lock")).to_dict()
        for item in _sequence(raw.get("runtime_locks"), "runtime_locks")
    )
    if not task_validity:
        raise ValueError("V3 evaluation view requires task validity")
    if not scorer_revisions:
        raise ValueError("V3 evaluation view requires scorer revisions")
    if not runtime_locks:
        raise ValueError("V3 evaluation view requires runtime locks")
    supersedes = tuple(
        superseded_result_from_dict(_mapping(item, "superseded_result")).to_dict()
        for item in _sequence(raw.get("supersedes"), "supersedes")
    )
    scope = _optional_evidence_scope(raw.get("evidence_scope"))
    if isinstance(topology.result_destination, LocalEvidenceDestinationV1):
        if scope is not None:
            raise ValueError("local V3 evidence cannot declare a W&B evidence scope")
    elif scope is None or f"{scope.entity}/{scope.project}" != (
        topology.result_destination.project_slug
    ):
        raise ValueError("V3 evidence scope must equal the result evidence destination")
    matrix_size = _non_negative_int(raw.get("matrix_size", 0), "matrix_size")
    completed_cells = _optional_non_negative_int(
        raw.get("completed_cells"), "completed_cells"
    )
    state_counts = _count_mapping(raw.get("state_counts"), "state_counts")
    terminal_states = {
        "completed",
        "failed",
        "cancelled",
        "interrupted",
        "not_applicable",
    }
    if set(state_counts) - terminal_states:
        raise ValueError(
            "V3 final evaluation state_counts may contain only terminal states"
        )
    aligned_attempt_ids = {
        attempt_id
        for item in aligned.aligned_attempts
        for attempt_id in item.attempt_ids_by_arm.values()
    }
    if matrix_size != len(aligned_attempt_ids):
        raise ValueError("V3 matrix_size must equal the unique aligned attempt count")
    if completed_cells is not None and completed_cells != len(aligned_attempt_ids):
        raise ValueError(
            "V3 completed_cells must equal the unique aligned attempt count"
        )
    if sum(state_counts.values()) != len(aligned_attempt_ids):
        raise ValueError(
            "V3 state_counts must reconcile every aligned terminal attempt"
        )
    paired_cases = tuple(
        _canonical_paired_case_v3(item)
        for item in _sequence(raw.get("paired_cases"), "paired_cases")
    )
    paired_attempt_ids = {
        str(attempt.get("attempt_id") or "")
        for pair in paired_cases
        for arm in ("baseline", "candidate")
        if isinstance((attempt := pair.get(arm)), Mapping)
    }
    behavioral_status = str(
        _mapping_or_empty(raw.get("behavioral_summary")).get("status") or ""
    )
    integrity_status = _text(raw.get("integrity_status"), "integrity_status", 80)
    if (
        integrity_status != "invalid"
        and behavioral_status != "invalid"
        and paired_attempt_ids != aligned_attempt_ids
    ):
        raise ValueError("V3 paired cases must expose every aligned attempt")
    if evidence_eligible and paired_attempt_ids != aligned_attempt_ids:
        raise ValueError("V3 evidence-eligible view requires clickable paired attempts")
    if evidence_eligible and any(
        str(attempt.get("evidence_status") or "") != "reconciled"
        or any(
            str(link.get("status") or "") != "resolved"
            for link in attempt.get("evidence_links") or ()
            if isinstance(link, Mapping)
        )
        for pair in paired_cases
        for arm in ("baseline", "candidate")
        if isinstance((attempt := pair.get(arm)), Mapping)
    ):
        raise ValueError(
            "V3 evidence-eligible view requires five resolved links per attempt"
        )
    view = ExperimentViewV3(
        schema_version=EXPERIMENT_VIEW_V3_SCHEMA_VERSION,
        kind="evaluation",
        matrix_size=matrix_size,
        preview_digest=_optional_digest(raw.get("preview_digest"), "preview_digest"),
        approval_state=_optional_text(raw.get("approval_state"), "approval_state", 80),
        phase=_optional_text(raw.get("phase"), "phase", 80),
        completed_cells=completed_cells,
        observed_cost_usd=_optional_cost(
            raw.get("observed_cost_usd"), "observed_cost_usd"
        ),
        state_counts=state_counts,
        infrastructure_health=_text(
            raw.get("infrastructure_health"), "infrastructure_health", 80
        ),
        evidence_eligible=evidence_eligible,
        evidence_scope=scope,
        limitations=tuple(
            _text(item, "limitation", 1000)
            for item in _sequence(raw.get("limitations"), "limitations")
        ),
        evidence_links=_evidence_links(raw.get("evidence_links")),
        decision=_optional_mapping(raw.get("decision"), "decision"),
        integrity_status=integrity_status,
        evidence_grade=_text(raw.get("evidence_grade"), "evidence_grade", 20),
        release_target=_optional_text(raw.get("release_target"), "release_target", 300),
        candidate_sha=_optional_text(raw.get("candidate_sha"), "candidate_sha", 100),
        release_note_coverage=tuple(
            _mapping(item, "release_note_coverage")
            for item in _sequence(
                raw.get("release_note_coverage"), "release_note_coverage"
            )
        ),
        infrastructure_gates=tuple(
            _mapping(item, "infrastructure_gates")
            for item in _sequence(
                raw.get("infrastructure_gates"), "infrastructure_gates"
            )
        ),
        behavioral_summary=_optional_behavioral_summary(raw.get("behavioral_summary")),
        paired_cases=paired_cases,
        backend=_optional_text(raw.get("backend"), "backend", 100),
        candidate_source_revisions=tuple(
            _candidate_source_revision(item)
            for item in _sequence(
                raw.get("candidate_source_revisions"),
                "candidate_source_revisions",
            )
        ),
        evidence_topology=topology.to_dict(),
        aligned_analysis=aligned.to_dict(),
        task_validity=task_validity,
        scorer_revisions=scorer_revisions,
        runtime_locks=runtime_locks,
        result_digest=_optional_digest(raw.get("result_digest"), "result_digest"),
        qualification_digest=_optional_digest(
            raw.get("qualification_digest"), "qualification_digest"
        ),
        runtime_lock_digest=_optional_digest(
            raw.get("runtime_lock_digest"), "runtime_lock_digest"
        ),
        supersedes=supersedes,
        judge_summary=_safe_judge_summary(
            raw.get("judge_summary"),
            integrity_status=integrity_status,
            attempts=matrix_size,
            arm_attempts=_judge_arm_attempt_counts(paired_cases),
        ),
    )
    _validate_view_shape(view)
    if (
        view.runtime_lock_digest is not None
        and view.runtime_lock_digest != stable_digest(list(view.runtime_locks))
    ):
        raise ValueError("V3 runtime lock digest does not recompute")
    _validate_v3_evidence_routes(view)
    return view


def _validate_v3_evidence_routes(view: ExperimentViewV3) -> None:
    topology = evidence_topology_from_dict(view.evidence_topology)
    source_project = (
        None
        if isinstance(topology.source_destination, LocalEvidenceDestinationV1)
        else topology.source_destination.project_slug
    )
    local_result = isinstance(topology.result_destination, LocalEvidenceDestinationV1)
    result_project = None if local_result else topology.result_destination.project_slug
    call_prefix = (
        ""
        if local_result
        else f"{topology.result_destination.app_base_url.rstrip('/')}/"
        f"{result_project}/weave/calls/"
    )
    object_prefix = (
        ""
        if local_result
        else f"{topology.result_destination.app_base_url.rstrip('/')}/"
        f"{result_project}/weave/objects/"
    )
    for pair in view.paired_cases:
        for arm in ("baseline", "candidate"):
            attempt = pair.get(arm)
            if not isinstance(attempt, Mapping):
                continue
            scope = tuple(str(item) for item in attempt.get("actual_query_scope") or ())
            if source_project is None and scope:
                raise ValueError(
                    "V3 attempt query scope requires a hosted source destination"
                )
            if source_project is not None and any(
                item != source_project for item in scope
            ):
                raise ValueError(
                    "V3 attempt query scope escaped the source destination"
                )
            for link in attempt.get("evidence_links") or ():
                if not isinstance(link, Mapping):
                    continue
                if str(link.get("status") or "") != "resolved":
                    continue
                kind = str(link.get("kind") or "")
                system = str(link.get("system") or "")
                ref = str(link.get("ref") or "")
                url = str(link.get("url") or "")
                if local_result:
                    if (
                        system != "local_artifact"
                        or not ref.startswith("fugue://local-evidence/")
                        or url
                    ):
                        raise ValueError(
                            "local V3 evidence must use canonical local-artifact refs"
                        )
                    continue
                if system != "weave":
                    raise ValueError("hosted V3 evidence must use Weave links")
                if kind == "dataset":
                    if not url.startswith(object_prefix) or "/versions/" not in url:
                        raise ValueError(
                            "V3 Dataset evidence must use a versioned result-project "
                            "object URL"
                        )
                elif not url.startswith(call_prefix):
                    raise ValueError(
                        "V3 Call evidence must use the canonical result-project "
                        "/weave/calls route"
                    )


def experiment_view_from_dict(
    raw: Mapping[str, Any],
) -> ExperimentViewV1 | ExperimentViewV2 | ExperimentViewV3:
    schema_version = _positive_int(raw.get("schema_version"), "schema_version")
    if schema_version not in EXPERIMENT_VIEW_READABLE_SCHEMA_VERSIONS:
        raise ValueError("unsupported experiment view schema")
    kind = _text(raw.get("kind"), "kind", 40)
    if kind not in _VIEW_KINDS:
        raise ValueError("unknown experiment view kind")
    # A short-lived pre-contract V2 writer emitted the V1 evaluation shape
    # under schema_version=2. Those immutable records must remain replayable,
    # but the canonical V2 surface stays strict whenever paired_cases is
    # present (including when a producer tries to mix old and new fields).
    provisional_v2 = (
        schema_version == 2
        and kind == "evaluation"
        and "paired_cases" not in raw
        and bool(set(raw) & _PROVISIONAL_V2_EVALUATION_FIELDS)
    )
    if schema_version == 3 and kind == "evaluation":
        return _experiment_view_v3_from_dict(raw)
    if schema_version == 2 and kind == "evaluation" and not provisional_v2:
        _reject_unknown(
            raw,
            {item.name for item in ExperimentViewV2.__dataclass_fields__.values()},
            "experiment view",
        )
        evidence_eligible = _optional_bool(
            raw.get("evidence_eligible"), "evidence_eligible"
        )
        if evidence_eligible is None:
            raise ValueError("evaluation view requires evidence eligibility")
        view_v2 = ExperimentViewV2(
            schema_version=schema_version,
            kind="evaluation",
            matrix_size=_non_negative_int(raw.get("matrix_size", 0), "matrix_size"),
            preview_digest=_optional_digest(
                raw.get("preview_digest"), "preview_digest"
            ),
            approval_state=_optional_text(
                raw.get("approval_state"), "approval_state", 80
            ),
            phase=_optional_text(raw.get("phase"), "phase", 80),
            completed_cells=_optional_non_negative_int(
                raw.get("completed_cells"), "completed_cells"
            ),
            observed_cost_usd=_optional_cost(
                raw.get("observed_cost_usd"), "observed_cost_usd"
            ),
            state_counts=_count_mapping(raw.get("state_counts"), "state_counts"),
            infrastructure_health=_text(
                raw.get("infrastructure_health"),
                "infrastructure_health",
                80,
            ),
            evidence_eligible=evidence_eligible,
            evidence_scope=_optional_evidence_scope(raw.get("evidence_scope")),
            limitations=tuple(
                _text(item, "limitation", 1000)
                for item in _sequence(raw.get("limitations"), "limitations")
            ),
            evidence_links=_evidence_links(raw.get("evidence_links")),
            decision=_optional_mapping(raw.get("decision"), "decision"),
            integrity_status=_text(raw.get("integrity_status"), "integrity_status", 80),
            evidence_grade=_text(raw.get("evidence_grade"), "evidence_grade", 20),
            release_target=_optional_text(
                raw.get("release_target"), "release_target", 300
            ),
            candidate_sha=_optional_text(
                raw.get("candidate_sha"), "candidate_sha", 100
            ),
            release_note_coverage=tuple(
                _mapping(item, "release_note_coverage")
                for item in _sequence(
                    raw.get("release_note_coverage"), "release_note_coverage"
                )
            ),
            infrastructure_gates=tuple(
                _mapping(item, "infrastructure_gates")
                for item in _sequence(
                    raw.get("infrastructure_gates"), "infrastructure_gates"
                )
            ),
            behavioral_summary=_optional_behavioral_summary(
                raw.get("behavioral_summary")
            ),
            paired_cases=tuple(
                _canonical_paired_case(item)
                for item in _sequence(raw.get("paired_cases"), "paired_cases")
            ),
            backend=_optional_text(raw.get("backend"), "backend", 100),
            candidate_source_revisions=tuple(
                _candidate_source_revision(item)
                for item in _sequence(
                    raw.get("candidate_source_revisions"),
                    "candidate_source_revisions",
                )
            ),
        )
        _validate_view_shape(view_v2)
        return view_v2
    _reject_unknown(
        raw,
        {item.name for item in ExperimentViewV1.__dataclass_fields__.values()},
        "experiment view",
    )
    fixed = tuple(
        _factor(item, "fixed_conditions")
        for item in _sequence(raw.get("fixed_conditions"), "fixed_conditions")
    )
    varied = tuple(
        _factor(item, "varied_factors")
        for item in _sequence(raw.get("varied_factors"), "varied_factors")
    )
    cells = tuple(_cell(item) for item in _sequence(raw.get("cells"), "cells"))
    if len(cells) > EXPERIMENT_VIEW_CELL_LIMIT:
        raise ValueError("experiment view exceeds the public cell limit")
    view = ExperimentViewV1(
        schema_version=schema_version,
        kind=kind,  # type: ignore[arg-type]
        research_label=_optional_text(raw.get("research_label"), "research_label", 300),
        study_label=_optional_text(raw.get("study_label"), "study_label", 300),
        question=_optional_text(raw.get("question"), "question", 2000),
        hypothesis=_optional_text(raw.get("hypothesis"), "hypothesis", 2000),
        context=_optional_text(raw.get("context"), "context", 4000),
        observation=_optional_text(raw.get("observation"), "observation", 4000),
        rationale=_optional_text(raw.get("rationale"), "rationale", 4000),
        alternative_explanations=tuple(
            _text(item, "alternative explanation", 1000)
            for item in _sequence(
                raw.get("alternative_explanations"), "alternative_explanations"
            )
        ),
        success_definition=_optional_text(
            raw.get("success_definition"), "success_definition", 4000
        ),
        task_design=_optional_task_design(raw.get("task_design")),
        prompt_design=_optional_prompt_design(raw.get("prompt_design")),
        evaluation_design=_optional_evaluation_design(raw.get("evaluation_design")),
        source_cohort=_optional_descriptor(raw.get("source_cohort"), "source_cohort"),
        fixed_conditions=fixed,
        varied_factors=varied,
        treatment_arms=tuple(
            _treatment_arm(item)
            for item in _sequence(raw.get("treatment_arms"), "treatment_arms")
        ),
        measured_outcomes=tuple(
            _text(item, "measured_outcome", 200)
            for item in _sequence(raw.get("measured_outcomes"), "measured_outcomes")
        ),
        taskset=_optional_descriptor(raw.get("taskset"), "taskset"),
        harnesses=tuple(
            _descriptor(item, "harnesses")
            for item in _sequence(raw.get("harnesses"), "harnesses")
        ),
        runtime=_optional_descriptor(raw.get("runtime"), "runtime"),
        matrix_size=_non_negative_int(raw.get("matrix_size", 0), "matrix_size"),
        preview_digest=_optional_digest(raw.get("preview_digest"), "preview_digest"),
        approval_state=_optional_text(raw.get("approval_state"), "approval_state", 80),
        cell_limit=_optional_non_negative_int(raw.get("cell_limit"), "cell_limit"),
        reserved_cost_usd=_optional_cost(
            raw.get("reserved_cost_usd"), "reserved_cost_usd"
        ),
        phase=_optional_text(raw.get("phase"), "phase", 80),
        completed_cells=_optional_non_negative_int(
            raw.get("completed_cells"), "completed_cells"
        ),
        observed_cost_usd=_optional_cost(
            raw.get("observed_cost_usd"), "observed_cost_usd"
        ),
        state_counts=_count_mapping(raw.get("state_counts"), "state_counts"),
        cells=cells,
        omitted_cells=_non_negative_int(raw.get("omitted_cells", 0), "omitted_cells"),
        infrastructure_health=_optional_text(
            raw.get("infrastructure_health"), "infrastructure_health", 80
        ),
        arm_totals=tuple(
            _arm_total(item) for item in _sequence(raw.get("arm_totals"), "arm_totals")
        ),
        aligned_comparisons=tuple(
            _comparison(item)
            for item in _sequence(raw.get("aligned_comparisons"), "aligned_comparisons")
        ),
        behavioral_measures=_measure_mapping(
            raw.get("behavioral_measures"), "behavioral_measures"
        ),
        mechanism_funnel=tuple(
            _mechanism_stage(item)
            for item in _sequence(raw.get("mechanism_funnel"), "mechanism_funnel")
        ),
        outcome_summaries=tuple(
            _outcome_summary(item)
            for item in _sequence(raw.get("outcome_summaries"), "outcome_summaries")
        ),
        score_summaries=tuple(
            _score_summary(item)
            for item in _sequence(raw.get("score_summaries"), "score_summaries")
        ),
        evidence_eligible=_optional_bool(
            raw.get("evidence_eligible"), "evidence_eligible"
        ),
        evidence_scope=_optional_evidence_scope(raw.get("evidence_scope")),
        limitations=tuple(
            _text(item, "limitation", 1000)
            for item in _sequence(raw.get("limitations"), "limitations")
        ),
        evidence_links=_evidence_links(raw.get("evidence_links")),
        decision=_optional_mapping(raw.get("decision"), "decision"),
        integrity_status=_optional_text(
            raw.get("integrity_status"), "integrity_status", 80
        ),
        evidence_grade=_optional_text(raw.get("evidence_grade"), "evidence_grade", 20),
        release_target=_optional_text(raw.get("release_target"), "release_target", 300),
        candidate_sha=_optional_text(raw.get("candidate_sha"), "candidate_sha", 100),
        release_note_coverage=tuple(
            _mapping(item, "release_note_coverage")
            for item in _sequence(
                raw.get("release_note_coverage"), "release_note_coverage"
            )
        ),
        infrastructure_gates=tuple(
            _mapping(item, "infrastructure_gates")
            for item in _sequence(
                raw.get("infrastructure_gates"), "infrastructure_gates"
            )
        ),
    )
    _validate_view_shape(view)
    return view


def build_comparison_design_view(
    preview: Mapping[str, Any],
    *,
    approval_state: str = "awaiting_approval",
) -> ExperimentViewV1:
    """Project one governed comparison preview into the canonical public design."""

    comparison = _mapping(preview.get("comparison"), "comparison")
    readiness = _mapping(preview.get("readiness"), "readiness")
    matrix = _mapping(preview.get("matrix"), "matrix")
    experiment = _mapping_or_empty(preview.get("experiment"))
    execution = _mapping(comparison.get("execution"), "comparison.execution")
    environment = _mapping_or_empty(execution.get("environment"))
    research_view = _mapping_or_empty(experiment.get("research_view"))
    task_count = int(readiness.get("task_count") or 0)
    taskset_digest = str(readiness.get("taskset_digest") or "")
    comparison_id = str(comparison.get("id") or "")
    preview_digest = str(preview.get("preview_digest") or "")
    harness_ids = _ordered_values(
        [str(item) for item in execution.get("harnesses") or ()]
        + [str(item) for item in matrix.get("harnesses") or ()]
    )
    raw_cells = [
        item for item in matrix.get("matrix_cells") or () if isinstance(item, Mapping)
    ]
    attempts: Counter[tuple[str, str, str]] = Counter()
    cells: list[ExperimentCellViewV1] = []
    for raw in raw_cells[:EXPERIMENT_VIEW_CELL_LIMIT]:
        variant = str(raw.get("variant_id") or "")
        harness = str(raw.get("harness") or "")
        task_id = str(raw.get("task_id") or "")
        coordinate = (variant, harness, task_id)
        attempts[coordinate] += 1
        trial_index = int(raw.get("trial_index") or attempts[coordinate])
        supplied_attempt_id = str(raw.get("attempt_id") or "")
        if int(preview.get("schema_version") or 1) >= 2:
            expected_attempt_id = canonical_attempt_id(
                task_id=task_id,
                arm=variant,
                harness=harness,
                attempt=trial_index,
                candidate=str(raw.get("candidate_id") or ""),
                runtime=str(raw.get("execution_fingerprint") or ""),
            )
            if supplied_attempt_id != expected_attempt_id:
                raise ValueError(
                    "comparison preview attempt identity is missing or disagrees"
                )
        applicable = bool(raw.get("applicable", True))
        cells.append(
            ExperimentCellViewV1(
                cell_id=(
                    supplied_attempt_id
                    or _opaque_cell_id({**raw, "trial_index": trial_index})
                ),
                task_label=humanize_display_id(task_id),
                factor_levels={"candidate": variant},
                attempt=trial_index,
                execution_status="queued" if applicable else "not_applicable",
                task_outcome="pending" if applicable else "not_applicable",
                evaluation_status="pending" if applicable else "not_applicable",
                evidence_status="pending" if applicable else "not_applicable",
                reason_code=None if applicable else "not_applicable",
            )
        )
    matrix_size = int(
        readiness.get("estimated_cells")
        or matrix.get("estimated_trials")
        or matrix.get("cells")
        or len(raw_cells)
    )
    model = str(execution.get("model") or "")
    evidence_project = str(
        execution.get("evidence_project")
        or readiness.get("evidence_project")
        or matrix.get("evidence_project")
        or ""
    )
    evidence_scope = None
    if evidence_project.count("/") == 1:
        entity, project = evidence_project.split("/", 1)
        if entity and project:
            evidence_scope = ExperimentEvidenceScopeV1(
                entity=entity,
                project=project,
                evidence_types=(
                    "evaluation",
                    "prediction",
                    "agent_conversation",
                    "dataset",
                ),
            )
    attempts_per_cell = int(execution.get("attempts") or 0)
    environment_type = str(environment.get("type") or "harbor")
    runtime_label = (
        "W&B Serverless Sandbox"
        if environment_type == "wandb"
        else humanize_display_id(environment_type)
    )
    runtime_details: dict[str, str | int | bool] = {
        "locked_before_execution": True,
    }
    if environment.get("runtime_lock"):
        runtime_details["runtime_lock"] = str(environment["runtime_lock"])
    if environment.get("delete") is not None:
        runtime_details["delete_after_run"] = bool(environment["delete"])
    fixed_conditions = [
        ExperimentFactorV1(
            name="model",
            levels=(model,),
            label="Model",
        ),
        ExperimentFactorV1(
            name="harness",
            levels=harness_ids,
            label="Harness",
        ),
        ExperimentFactorV1(
            name="taskset",
            levels=(taskset_digest,),
            label="Locked taskset",
        ),
        ExperimentFactorV1(
            name="attempts",
            levels=(str(attempts_per_cell),),
            label="Attempts per aligned cell",
        ),
        ExperimentFactorV1(
            name="environment",
            levels=(environment_type,),
            label="Execution environment",
        ),
        ExperimentFactorV1(
            name="evidence_project",
            levels=(evidence_project,),
            label="Evidence project",
        ),
        ExperimentFactorV1(
            name="research_id",
            levels=(str(execution.get("research_id") or ""),),
            label="Research catalogue",
        ),
    ]
    fixed_conditions = [
        item for item in fixed_conditions if all(level for level in item.levels)
    ]
    baseline = _mapping(comparison.get("baseline"), "comparison.baseline")
    candidate = _mapping(comparison.get("candidate"), "comparison.candidate")
    decision_policy = _mapping_or_empty(comparison.get("decision_policy"))
    baseline_label = str(baseline.get("label") or "Baseline")
    candidate_label = str(candidate.get("label") or "Candidate")
    taskset = ExperimentDescriptorV1(
        id=f"{comparison_id}-taskset",
        label=f"{task_count} locked comparison tasks",
        digest=taskset_digest or None,
        details={"task_count": task_count},
    )
    task_design = _research_task_design(research_view, {})
    if task_design is not None and not task_design.evidence_links:
        task_design = ExperimentTaskDesignV1(
            title=task_design.title,
            summary=task_design.summary,
            interaction_mode=task_design.interaction_mode,
            tools=task_design.tools,
            resources=task_design.resources,
            evidence_links=(
                {
                    "system": "fugue",
                    "kind": "task_definition",
                    "ref": taskset.id,
                    **({"digest": taskset_digest} if taskset_digest else {}),
                },
            ),
        )
    return experiment_view_from_dict(
        ExperimentViewV1(
            schema_version=EXPERIMENT_VIEW_SCHEMA_VERSION,
            kind="design",
            question=str(comparison.get("question") or ""),
            hypothesis=(
                "The candidate will improve required evaluation outcomes without "
                "critical regressions relative to the exact baseline."
            ),
            context=(
                "Only the declared candidate change varies; task inputs, attempts, "
                "harnesses, model, evaluator revisions, and runtime policy remain "
                "locked."
            ),
            observation=str(research_view.get("observation") or "").strip() or None,
            rationale=str(research_view.get("rationale") or "").strip() or None,
            alternative_explanations=tuple(
                str(item)
                for item in research_view.get("alternative_explanations") or ()
                if str(item).strip()
            ),
            success_definition=(
                str(research_view.get("success_definition") or "").strip() or None
            ),
            task_design=task_design,
            prompt_design=_research_prompt_design(research_view),
            evaluation_design=_research_evaluation_design(research_view),
            fixed_conditions=tuple(fixed_conditions),
            varied_factors=(
                ExperimentFactorV1(
                    name="candidate",
                    levels=("baseline", "candidate"),
                    label="Compared candidate",
                    level_labels={
                        "baseline": baseline_label,
                        "candidate": candidate_label,
                    },
                ),
            ),
            treatment_arms=(
                ExperimentTreatmentArmV1(
                    id="baseline",
                    label=baseline_label,
                    factor_levels={"candidate": "baseline"},
                ),
                ExperimentTreatmentArmV1(
                    id="candidate",
                    label=candidate_label,
                    factor_levels={"candidate": "candidate"},
                ),
            ),
            measured_outcomes=tuple(
                str(item.get("id"))
                for item in comparison.get("evaluators") or ()
                if isinstance(item, Mapping) and item.get("id")
            ),
            taskset=taskset,
            harnesses=tuple(
                ExperimentDescriptorV1(
                    id=value,
                    label=humanize_display_id(value),
                )
                for value in harness_ids
            ),
            runtime=ExperimentDescriptorV1(
                id=environment_type,
                label=runtime_label,
                details=runtime_details,
            ),
            evidence_scope=evidence_scope,
            release_target=(str(decision_policy.get("release_target") or "") or None),
            candidate_sha=(str(decision_policy.get("candidate_sha") or "") or None),
            matrix_size=matrix_size,
            preview_digest=preview_digest or None,
            approval_state=approval_state,
            cell_limit=matrix_size,
            reserved_cost_usd=float(readiness.get("estimated_cost_usd") or 0.0),
            cells=tuple(cells),
            omitted_cells=max(0, len(raw_cells) - len(cells)),
        ).to_dict()
    )


def build_comparison_progress_view(
    preview: Mapping[str, Any],
    *,
    phase: str,
    completed_cells: int = 0,
    state_counts: Mapping[str, int] | None = None,
) -> ExperimentViewV1:
    """Publish comparison lifecycle without inventing unobserved cell state."""

    readiness = _mapping(preview.get("readiness"), "readiness")
    matrix = _mapping(preview.get("matrix"), "matrix")
    design = build_comparison_design_view(preview, approval_state="approved")
    matrix_size = int(
        readiness.get("estimated_cells")
        or matrix.get("estimated_trials")
        or matrix.get("cells")
        or 0
    )
    return experiment_view_from_dict(
        ExperimentViewV1(
            schema_version=EXPERIMENT_VIEW_SCHEMA_VERSION,
            kind="progress",
            matrix_size=matrix_size,
            preview_digest=str(preview.get("preview_digest") or "") or None,
            approval_state="approved",
            cell_limit=matrix_size,
            reserved_cost_usd=float(readiness.get("estimated_cost_usd") or 0.0),
            phase=phase,
            completed_cells=completed_cells,
            state_counts=dict(state_counts or {}),
            cells=design.cells,
            omitted_cells=max(0, matrix_size - len(design.cells)),
        ).to_dict()
    )


def build_comparison_evaluation_view(
    result: Mapping[str, Any],
    *,
    result_ref: str | None = None,
) -> ExperimentViewV1 | ExperimentViewV2 | ExperimentViewV3:
    """Project comparison output while preserving outcome/evidence boundaries."""

    if int(result.get("schema_version") or 1) == 3:
        return _build_comparison_evaluation_view_v3(
            result,
            result_ref=result_ref,
        )
    if int(result.get("schema_version") or 1) == 2:
        return _build_comparison_evaluation_view_v2(
            result,
            result_ref=result_ref,
        )
    rows = int(result.get("rows") or 0)
    pairs = [
        item for item in result.get("paired_cases") or () if isinstance(item, Mapping)
    ]
    baseline_total = sum(
        bool(item.get("baseline") or item.get("baseline_prediction_id"))
        for item in pairs
    )
    candidate_total = sum(
        bool(item.get("candidate") or item.get("candidate_prediction_id"))
        for item in pairs
    )
    if not pairs and rows:
        baseline_total = rows // 2
        candidate_total = rows - baseline_total
    operational = _mapping_or_empty(result.get("operational_summary"))
    execution_states = {
        str(key): int(value)
        for key, value in _mapping_or_empty(operational.get("execution_states")).items()
    }
    evidence_states = {
        str(key): int(value)
        for key, value in _mapping_or_empty(operational.get("evidence_states")).items()
    }
    infrastructure_failures = int(operational.get("infrastructure_failures") or 0)
    integrity = _mapping_or_empty(result.get("integrity"))
    integrity_status = str(integrity.get("status") or "") or None
    harbor_conformance_failures = int(
        integrity.get("harbor_conformance_failed_attempts") or 0
    )
    behavioral_summary = _comparison_behavioral_summary(result, integrity)
    suppress_attempt_navigation = (
        integrity_status == "invalid"
        or str((behavioral_summary or {}).get("status") or "") == "invalid"
    )
    display_pairs = [] if suppress_attempt_navigation else pairs
    decision_value = _mapping_or_empty(result.get("decision"))
    infrastructure_gates = tuple(
        dict(item)
        for item in decision_value.get("gates") or ()
        if isinstance(item, Mapping)
        and str(item.get("category") or "") == "infrastructure"
    )
    incomplete = int(result.get("incomplete") or 0)
    required_incomplete = int(result.get("required_evaluations_incomplete") or 0)
    missing_evidence = int(
        integrity.get("unresolved_evidence_attempts")
        if integrity.get("unresolved_evidence_attempts") is not None
        else sum(
            count
            for state, count in evidence_states.items()
            if state not in {"ok", "linked", "reconciled", "not_applicable"}
        )
    )
    limitations = [
        str(item) for item in result.get("limitations") or () if str(item).strip()
    ]
    if rows == 0:
        limitations.append("No comparison rows were observed.")
    if required_incomplete:
        limitations.append(
            f"{required_incomplete} required authored or judge evaluations are incomplete."
        )
    if infrastructure_failures:
        limitations.append(
            f"{infrastructure_failures} attempts had infrastructure failures."
        )
    evidence_links = _comparison_evidence_links(
        () if suppress_attempt_navigation else result.get("evidence_links"),
        result_ref=result_ref,
        result_source=str(result.get("source") or ""),
        result_digest=str(result.get("result_digest") or ""),
    )
    aligned_pairs = (
        int(result.get("improved") or 0)
        + int(result.get("regressed") or 0)
        + int(result.get("mixed") or 0)
        + int(result.get("unchanged") or 0)
    )
    result_digest = str(result.get("result_digest") or "")
    cells: list[ExperimentCellViewV1] = []
    for pair in display_pairs:
        task_id = str(pair.get("task_id") or "")
        harness = str(pair.get("harness") or "")
        attempt = int(pair.get("attempt") or 1)
        for arm in ("baseline", "candidate"):
            attempt_view = _mapping_or_empty(pair.get(arm))
            prediction_id = str(
                attempt_view.get("prediction_id")
                or pair.get(f"{arm}_prediction_id")
                or ""
            )
            if not prediction_id and not attempt_view:
                continue
            attempt_links = (
                _comparison_attempt_evidence_links(attempt_view.get("evidence_links"))
                if attempt_view
                else _comparison_pair_evidence_links(
                    result.get("evidence_links"),
                    task_id=task_id,
                    arm=arm,
                )
            )
            passed_value = (
                attempt_view.get("passed")
                if "passed" in attempt_view
                else pair.get(f"{arm}_passed")
            )
            execution_status = str(attempt_view.get("execution_status") or "completed")
            if execution_status not in _EXECUTION_STATES:
                execution_status = (
                    "failed"
                    if execution_status in {"error", "timed_out"}
                    else "completed"
                )
            observed_link_kinds = {
                str(item.get("kind") or "") for item in attempt_links
            }
            required_link_kinds = {
                "evaluation_root",
                "prediction_and_score",
                "prediction",
                "agent_root",
                "dataset",
            }
            evidence_status = str(
                attempt_view.get("evidence_status")
                or (
                    "reconciled"
                    if required_link_kinds <= observed_link_kinds
                    else "missing"
                )
            )
            if evidence_status not in _EVIDENCE_STATES:
                evidence_status = "missing"
            cells.append(
                ExperimentCellViewV1(
                    cell_id=str(attempt_view.get("attempt_id") or "")
                    or _opaque_cell_id(
                        {
                            "task_id": task_id,
                            "harness": harness,
                            "attempt": attempt,
                            "arm": arm,
                            "prediction_id": prediction_id,
                        }
                    ),
                    task_label=str(pair.get("task_label") or "")
                    or humanize_display_id(task_id),
                    factor_levels={"candidate": arm},
                    attempt=attempt,
                    execution_status=execution_status,  # type: ignore[arg-type]
                    task_outcome=(
                        "passed"
                        if passed_value is True
                        else "failed"
                        if passed_value is False
                        else "unavailable"
                    ),
                    evaluation_status=(
                        "passed"
                        if passed_value is True
                        else "failed"
                        if passed_value is False
                        else "unavailable"
                    ),
                    evidence_status=evidence_status,  # type: ignore[arg-type]
                    cost_usd=_optional_float(attempt_view.get("cost_usd")),
                    latency_sec=_optional_float(attempt_view.get("latency_sec")),
                    evidence_links=attempt_links,
                    measures=_comparison_attempt_measures(attempt_view),
                    scores=_comparison_attempt_scores(attempt_view.get("scores")),
                )
            )
            if len(cells) >= EXPERIMENT_VIEW_CELL_LIMIT:
                break
        if len(cells) >= EXPERIMENT_VIEW_CELL_LIMIT:
            break
    missing_evidence = max(
        missing_evidence,
        sum(cell.evidence_status == "missing" for cell in cells),
    )
    if missing_evidence:
        limitations.append(
            f"{missing_evidence} attempts lack reconciled required evidence."
        )
    evidence_eligible = (
        rows > 0
        and not incomplete
        and not required_incomplete
        and not infrastructure_failures
        and not missing_evidence
        and integrity_status in {None, "reconciled"}
        and str((behavioral_summary or {}).get("status") or "")
        not in {"invalid", "incomplete"}
        and (
            int(result.get("schema_version") or 1) == 1
            or decision_value.get("evidence_grade") == "A"
        )
    )
    arm_totals = (
        {
            "arm": "baseline",
            "arm_label": "Baseline",
            "harness": "all",
            "passed": int(result.get("baseline_passed") or 0),
            "total": baseline_total,
            "factor_levels": {"candidate": "baseline"},
        },
        {
            "arm": "candidate",
            "arm_label": "Candidate",
            "harness": "all",
            "passed": int(result.get("candidate_passed") or 0),
            "total": candidate_total,
            "factor_levels": {"candidate": "candidate"},
        },
    )
    return experiment_view_from_dict(
        ExperimentViewV1(
            schema_version=1,
            kind="evaluation",
            matrix_size=rows,
            preview_digest=str(result.get("preview_digest") or "") or None,
            approval_state="approved",
            cell_limit=rows,
            phase="completed",
            completed_cells=rows,
            observed_cost_usd=_optional_float(operational.get("observed_cost_usd")),
            state_counts=execution_states,
            cells=tuple(cells),
            omitted_cells=max(0, rows - len(cells)),
            infrastructure_health=(
                "unavailable"
                if rows == 0
                else "failed"
                if infrastructure_failures or harbor_conformance_failures
                else _comparison_infrastructure_health(
                    result,
                    infrastructure_gates,
                )
            ),
            arm_totals=(() if suppress_attempt_navigation else arm_totals),
            aligned_comparisons=(
                ()
                if suppress_attempt_navigation
                else (
                    {
                        "analysis_id": "deterministic-pass-rate-delta",
                        "comparison_id": str(result.get("comparison_id") or ""),
                        "estimate": (
                            (int(result.get("candidate_passed") or 0) / candidate_total)
                            - (int(result.get("baseline_passed") or 0) / baseline_total)
                            if baseline_total and candidate_total
                            else 0.0
                        ),
                        "pairs": aligned_pairs,
                        **({"digest": result_digest} if result_digest else {}),
                    },
                )
            ),
            behavioral_measures=_comparison_behavioral_measures(operational),
            mechanism_funnel=_comparison_mechanism_funnel(
                result.get("mechanism_summary")
            ),
            outcome_summaries=(
                ()
                if suppress_attempt_navigation
                else _comparison_outcome_summaries(
                    result,
                    baseline_total=baseline_total,
                    candidate_total=candidate_total,
                    infrastructure_failures=infrastructure_failures,
                    missing_evidence=missing_evidence,
                )
            ),
            score_summaries=(
                ()
                if suppress_attempt_navigation
                else _comparison_score_summaries(result)
            ),
            evidence_eligible=evidence_eligible,
            limitations=tuple(dict.fromkeys(limitations)),
            evidence_links=evidence_links,
            evidence_scope=_comparison_evidence_scope(result),
        ).to_dict()
    )


def _build_comparison_evaluation_view_v3(
    result: Mapping[str, Any],
    *,
    result_ref: str | None,
) -> ExperimentViewV3:
    rows = int(result.get("rows") or 0)
    raw_pairs = tuple(
        item for item in result.get("paired_cases") or () if isinstance(item, Mapping)
    )
    operational = _mapping_or_empty(result.get("operational_summary"))
    execution_states = {
        str(key): int(value)
        for key, value in _mapping_or_empty(operational.get("execution_states")).items()
    }
    infrastructure_failures = int(operational.get("infrastructure_failures") or 0)
    integrity = _mapping_or_empty(result.get("integrity"))
    integrity_status = str(integrity.get("status") or "") or "invalid"
    behavioral_summary = _comparison_behavioral_summary(result, integrity)
    behavioral_status = str((behavioral_summary or {}).get("status") or "")
    suppress_attempt_navigation = (
        integrity_status == "invalid" or behavioral_status == "invalid"
    )
    decision_value = _mapping_or_empty(result.get("decision"))
    infrastructure_gates = tuple(
        dict(item)
        for item in decision_value.get("gates") or ()
        if isinstance(item, Mapping)
        and str(item.get("category") or "") == "infrastructure"
    )
    topology = evidence_topology_from_dict(
        _mapping(result.get("evidence_topology"), "evidence_topology")
    )
    aligned = aligned_analysis_from_dict(
        _mapping(result.get("aligned_analysis"), "aligned_analysis")
    )
    task_validity = tuple(
        task_validity_from_dict(_mapping(item, "task_validity")).to_dict()
        for item in _sequence(result.get("task_validity"), "task_validity")
    )
    limitations = [
        str(item) for item in result.get("limitations") or () if str(item).strip()
    ]
    invalid_tasks = [
        item
        for item in task_validity
        if item["status"] in {"drifted", "invalid", "inconclusive"}
    ]
    if invalid_tasks:
        limitations.append(
            f"{len(invalid_tasks)} task validity determination(s) require attention."
        )
    source_drift_matched = (
        topology.pre_run_drift.status == "matched"
        and topology.post_run_drift.status == "matched"
    )
    terminal_states = {
        "completed",
        "failed",
        "cancelled",
        "interrupted",
        "not_applicable",
    }
    terminal_cells = sum(
        count for state, count in execution_states.items() if state in terminal_states
    )
    evidence_eligible = (
        rows > 0
        and int(result.get("incomplete") or 0) == 0
        and int(result.get("required_evaluations_incomplete") or 0) == 0
        and infrastructure_failures == 0
        and int(integrity.get("unresolved_evidence_attempts") or 0) == 0
        and integrity_status == "reconciled"
        and behavioral_status not in {"invalid", "incomplete"}
        and source_drift_matched
        and not invalid_tasks
        and decision_value.get("evidence_grade") == "A"
    )
    scope = (
        None
        if isinstance(topology.result_destination, LocalEvidenceDestinationV1)
        else ExperimentEvidenceScopeV1(
            entity=topology.result_destination.entity,
            project=topology.result_destination.project,
            evidence_types=(
                "agent_conversation",
                "dataset",
                "evaluation_root",
                "prediction",
                "prediction_and_score",
            ),
        )
    )
    view = ExperimentViewV3(
        schema_version=EXPERIMENT_VIEW_V3_SCHEMA_VERSION,
        kind="evaluation",
        matrix_size=len(
            {
                attempt_id
                for item in aligned.aligned_attempts
                for attempt_id in item.attempt_ids_by_arm.values()
            }
        ),
        preview_digest=str(result.get("preview_digest") or "") or None,
        approval_state="approved",
        phase="completed",
        completed_cells=terminal_cells,
        observed_cost_usd=_optional_float(operational.get("observed_cost_usd")),
        state_counts=execution_states,
        infrastructure_health=(
            "unavailable"
            if rows == 0
            else "failed"
            if infrastructure_failures
            or int(integrity.get("harbor_conformance_failed_attempts") or 0)
            else _comparison_infrastructure_health(
                result,
                infrastructure_gates,
            )
        ),
        evidence_eligible=evidence_eligible,
        evidence_scope=scope,
        limitations=tuple(dict.fromkeys(limitations)),
        evidence_links=_comparison_evidence_links(
            () if suppress_attempt_navigation else result.get("evidence_links"),
            result_ref=result_ref,
            result_source=str(result.get("source") or ""),
            result_digest=str(result.get("result_digest") or ""),
        ),
        decision=dict(decision_value) or None,
        integrity_status=integrity_status,
        evidence_grade=(str(decision_value.get("evidence_grade") or "") or "invalid"),
        release_target=(str(decision_value.get("release_target") or "") or None),
        candidate_sha=(str(decision_value.get("candidate_sha") or "") or None),
        release_note_coverage=tuple(
            dict(item)
            for item in result.get("release_note_coverage") or ()
            if isinstance(item, Mapping)
        ),
        infrastructure_gates=infrastructure_gates,
        behavioral_summary=behavioral_summary,
        paired_cases=tuple(
            _canonical_paired_case_v3(item)
            for item in (() if suppress_attempt_navigation else raw_pairs)
        ),
        backend=_comparison_backend(raw_pairs),
        candidate_source_revisions=tuple(
            _candidate_source_revision(item)
            for item in result.get("candidate_source_revisions") or ()
            if isinstance(item, Mapping)
        ),
        evidence_topology=topology.to_dict(),
        aligned_analysis=aligned.to_dict(),
        task_validity=task_validity,
        scorer_revisions=tuple(
            lock_descriptor_from_dict(_mapping(item, "scorer_revision")).to_dict()
            for item in _sequence(result.get("scorer_revisions"), "scorer_revisions")
        ),
        runtime_locks=tuple(
            lock_descriptor_from_dict(_mapping(item, "runtime_lock")).to_dict()
            for item in _sequence(result.get("runtime_locks"), "runtime_locks")
        ),
        result_digest=_required_digest(result.get("result_digest"), "result_digest"),
        qualification_digest=_required_digest(
            result.get("qualification_digest"), "qualification_digest"
        ),
        runtime_lock_digest=stable_digest(
            [
                lock_descriptor_from_dict(_mapping(item, "runtime_lock")).to_dict()
                for item in _sequence(result.get("runtime_locks"), "runtime_locks")
            ]
        ),
        supersedes=tuple(
            superseded_result_from_dict(_mapping(item, "superseded_result")).to_dict()
            for item in _sequence(result.get("supersedes"), "supersedes")
        ),
        judge_summary=_safe_judge_summary(
            result.get("judge_summary"),
            integrity_status=integrity_status,
            attempts=rows,
            arm_attempts=_judge_arm_attempt_counts(raw_pairs),
        ),
    )
    return experiment_view_from_dict(view.to_dict())  # type: ignore[return-value]


def _build_comparison_evaluation_view_v2(
    result: Mapping[str, Any],
    *,
    result_ref: str | None,
) -> ExperimentViewV2:
    """Publish only the canonical paired V2 evaluation representation."""

    rows = int(result.get("rows") or 0)
    raw_pairs = tuple(
        item for item in result.get("paired_cases") or () if isinstance(item, Mapping)
    )
    operational = _mapping_or_empty(result.get("operational_summary"))
    execution_states = {
        str(key): int(value)
        for key, value in _mapping_or_empty(operational.get("execution_states")).items()
    }
    evidence_states = {
        str(key): int(value)
        for key, value in _mapping_or_empty(operational.get("evidence_states")).items()
    }
    infrastructure_failures = int(operational.get("infrastructure_failures") or 0)
    integrity = _mapping_or_empty(result.get("integrity"))
    integrity_status = str(integrity.get("status") or "") or "invalid"
    behavioral_summary = _comparison_behavioral_summary(result, integrity)
    behavioral_status = str((behavioral_summary or {}).get("status") or "")
    suppress_attempt_navigation = (
        integrity_status == "invalid" or behavioral_status == "invalid"
    )
    canonical_pairs = (
        ()
        if suppress_attempt_navigation
        else tuple(_canonical_paired_case(item) for item in raw_pairs)
    )
    decision_value = _mapping_or_empty(result.get("decision"))
    infrastructure_gates = tuple(
        dict(item)
        for item in decision_value.get("gates") or ()
        if isinstance(item, Mapping)
        and str(item.get("category") or "") == "infrastructure"
    )
    incomplete = int(result.get("incomplete") or 0)
    required_incomplete = int(result.get("required_evaluations_incomplete") or 0)
    missing_evidence = int(
        integrity.get("unresolved_evidence_attempts")
        if integrity.get("unresolved_evidence_attempts") is not None
        else sum(
            count
            for state, count in evidence_states.items()
            if state not in {"ok", "linked", "reconciled", "not_applicable"}
        )
    )
    limitations = [
        str(item) for item in result.get("limitations") or () if str(item).strip()
    ]
    if rows == 0:
        limitations.append("No comparison rows were observed.")
    if required_incomplete:
        limitations.append(
            f"{required_incomplete} required authored or judge evaluations are incomplete."
        )
    if infrastructure_failures:
        limitations.append(
            f"{infrastructure_failures} attempts had infrastructure failures."
        )
    if missing_evidence:
        limitations.append(
            f"{missing_evidence} attempts lack reconciled required evidence."
        )
    evidence_eligible = (
        rows > 0
        and not incomplete
        and not required_incomplete
        and not infrastructure_failures
        and not missing_evidence
        and integrity_status == "reconciled"
        and behavioral_status not in {"invalid", "incomplete"}
        and decision_value.get("evidence_grade") == "A"
    )
    view = ExperimentViewV2(
        schema_version=EXPERIMENT_VIEW_SCHEMA_VERSION,
        kind="evaluation",
        matrix_size=rows,
        preview_digest=str(result.get("preview_digest") or "") or None,
        approval_state="approved",
        phase="completed",
        completed_cells=rows,
        observed_cost_usd=_optional_float(operational.get("observed_cost_usd")),
        state_counts=execution_states,
        infrastructure_health=(
            "unavailable"
            if rows == 0
            else "failed"
            if infrastructure_failures
            or int(integrity.get("harbor_conformance_failed_attempts") or 0)
            else _comparison_infrastructure_health(
                result,
                infrastructure_gates,
            )
        ),
        evidence_eligible=evidence_eligible,
        evidence_scope=_comparison_evidence_scope(result),
        limitations=tuple(dict.fromkeys(limitations)),
        evidence_links=_comparison_evidence_links(
            () if suppress_attempt_navigation else result.get("evidence_links"),
            result_ref=result_ref,
            result_source=str(result.get("source") or ""),
            result_digest=str(result.get("result_digest") or ""),
        ),
        decision=dict(decision_value) or None,
        integrity_status=integrity_status,
        evidence_grade=(str(decision_value.get("evidence_grade") or "") or "invalid"),
        release_target=(str(decision_value.get("release_target") or "") or None),
        candidate_sha=(str(decision_value.get("candidate_sha") or "") or None),
        release_note_coverage=tuple(
            dict(item)
            for item in result.get("release_note_coverage") or ()
            if isinstance(item, Mapping)
        ),
        infrastructure_gates=infrastructure_gates,
        behavioral_summary=behavioral_summary,
        paired_cases=canonical_pairs,
        backend=_comparison_backend(raw_pairs),
        candidate_source_revisions=tuple(
            _candidate_source_revision(item)
            for item in result.get("candidate_source_revisions") or ()
            if isinstance(item, Mapping)
        ),
    )
    return experiment_view_from_dict(view.to_dict())  # type: ignore[return-value]


def _comparison_infrastructure_health(
    result: Mapping[str, Any],
    gates: Sequence[Mapping[str, Any]],
) -> str:
    if int(result.get("schema_version") or 1) == 1:
        return "healthy"
    if not gates:
        return "unavailable"
    statuses = {str(item.get("status") or "") for item in gates}
    if statuses <= {"passed"}:
        return "healthy"
    if "failed" in statuses:
        return "failed"
    return "unavailable"


def _comparison_behavioral_summary(
    result: Mapping[str, Any],
    integrity: Mapping[str, Any],
) -> dict[str, Any] | None:
    if str(integrity.get("status") or "") == "invalid":
        declared = (
            dict(result["behavioral_summary"])
            if isinstance(result.get("behavioral_summary"), Mapping)
            else {}
        )
        recommendation = str(declared.get("recommendation") or "").strip()
        blockers = tuple(
            str(item)
            for item in declared.get("critical_blockers") or ()
            if str(item).strip()
        )
        limitations = tuple(
            str(item) for item in declared.get("limitations") or () if str(item).strip()
        )
        next_action = str(declared.get("next_action") or "").strip()
        return {
            "status": "invalid",
            "recommendation": recommendation
            or "INVALID — do not use this historical result as behavioral evidence.",
            "improved_pairs": 0,
            "regressed_pairs": 0,
            "mixed_pairs": 0,
            "unchanged_pairs": 0,
            "incomplete_pairs": 0,
            "candidate_critical_failures": 0,
            "critical_blockers": blockers or ("result integrity is invalid",),
            "limitations": limitations
            or ("Historical operational observations are mechanism evidence only.",),
            "next_action": next_action
            or "Use the corrected Study identity and canonical attempt rows.",
        }
    raw = result.get("behavioral_summary")
    if isinstance(raw, Mapping):
        return dict(raw)
    return None


def _comparison_backend(
    pairs: Sequence[Mapping[str, Any]],
) -> str | None:
    backends: set[str] = set()
    for pair in pairs:
        for arm in ("baseline", "candidate"):
            attempt = _mapping_or_empty(pair.get(arm))
            infrastructure = _mapping_or_empty(attempt.get("infrastructure"))
            backend = str(infrastructure.get("backend") or "")
            if backend:
                backends.add(backend)
    if len(backends) == 1:
        return next(iter(backends))
    return "mixed" if backends else None


def _comparison_evidence_scope(
    result: Mapping[str, Any],
) -> ExperimentEvidenceScopeV1 | None:
    evidence_project = str(result.get("evidence_project") or "")
    if evidence_project.count("/") != 1:
        return None
    entity, project = evidence_project.split("/", 1)
    if not entity or not project:
        return None
    return ExperimentEvidenceScopeV1(
        entity=entity,
        project=project,
        evidence_types=(
            "evaluation",
            "prediction_and_score",
            "prediction",
            "agent_conversation",
            "dataset",
        ),
    )


def _optional_behavioral_summary(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    value = _mapping(raw, "behavioral_summary")
    allowed = {
        "status",
        "recommendation",
        "improved_pairs",
        "regressed_pairs",
        "mixed_pairs",
        "unchanged_pairs",
        "incomplete_pairs",
        "candidate_critical_failures",
        "critical_blockers",
        "supported_claim",
        "limitations",
        "next_action",
    }
    _reject_unknown(value, allowed, "behavioral_summary")
    status = _text(value.get("status"), "behavioral_summary.status", 40)
    if status not in {
        "invalid",
        "incomplete",
        "improved",
        "regressed",
        "mixed",
        "unchanged",
    }:
        raise ValueError("behavioral_summary.status is unsupported")
    result: dict[str, Any] = {
        "status": status,
        "recommendation": _text(
            value.get("recommendation"),
            "behavioral_summary.recommendation",
            2000,
        ),
        "improved_pairs": _non_negative_int(
            value.get("improved_pairs"), "behavioral_summary.improved_pairs"
        ),
        "regressed_pairs": _non_negative_int(
            value.get("regressed_pairs"), "behavioral_summary.regressed_pairs"
        ),
        "mixed_pairs": _non_negative_int(
            value.get("mixed_pairs"), "behavioral_summary.mixed_pairs"
        ),
        "unchanged_pairs": _non_negative_int(
            value.get("unchanged_pairs"), "behavioral_summary.unchanged_pairs"
        ),
        "incomplete_pairs": _non_negative_int(
            value.get("incomplete_pairs"), "behavioral_summary.incomplete_pairs"
        ),
        "candidate_critical_failures": _non_negative_int(
            value.get("candidate_critical_failures"),
            "behavioral_summary.candidate_critical_failures",
        ),
        "critical_blockers": tuple(
            _text(item, "behavioral_summary.critical_blocker", 1000)
            for item in _sequence(
                value.get("critical_blockers"),
                "behavioral_summary.critical_blockers",
            )
        ),
        "limitations": tuple(
            _text(item, "behavioral_summary.limitation", 1000)
            for item in _sequence(
                value.get("limitations"),
                "behavioral_summary.limitations",
            )
        ),
        "next_action": _text(
            value.get("next_action"), "behavioral_summary.next_action", 2000
        ),
    }
    supported_claim = _optional_text(
        value.get("supported_claim"),
        "behavioral_summary.supported_claim",
        4000,
    )
    if supported_claim:
        result["supported_claim"] = supported_claim
    return result


def _candidate_source_revision(raw: Any) -> dict[str, str]:
    value = _mapping(raw, "candidate_source_revision")
    allowed = {
        "kind",
        "id",
        "version_identity",
        "runtime_digest",
        "lock_digest",
    }
    _reject_unknown(value, allowed, "candidate_source_revision")
    result = {
        "kind": _text(
            value.get("kind"),
            "candidate_source_revision.kind",
            100,
        ),
        "id": _text(value.get("id"), "candidate_source_revision.id", 300),
        "version_identity": _text(
            value.get("version_identity"),
            "candidate_source_revision.version_identity",
            500,
        ),
        "runtime_digest": _text(
            value.get("runtime_digest"),
            "candidate_source_revision.runtime_digest",
            200,
        ),
    }
    if not result["runtime_digest"].startswith("sha256:"):
        raise ValueError(
            "candidate_source_revision.runtime_digest must be sha256-qualified"
        )
    lock_digest = _optional_text(
        value.get("lock_digest"),
        "candidate_source_revision.lock_digest",
        200,
    )
    if lock_digest:
        if not lock_digest.startswith("sha256:"):
            raise ValueError(
                "candidate_source_revision.lock_digest must be sha256-qualified"
            )
        result["lock_digest"] = lock_digest
    return result


def _canonical_paired_case(raw: Any) -> dict[str, Any]:
    value = _mapping(raw, "paired_case")
    allowed = {
        "pair_id",
        "task_id",
        "task_label",
        "harness",
        "attempt",
        "status",
        "dimension_changes",
        "baseline",
        "candidate",
        "baseline_passed",
        "candidate_passed",
        "baseline_prediction_id",
        "candidate_prediction_id",
        "baseline_evaluation_call_id",
        "candidate_evaluation_call_id",
    }
    _reject_unknown(value, allowed, "paired_case")
    status = _text(value.get("status"), "paired_case.status", 40)
    if status not in {"improved", "regressed", "mixed", "unchanged", "incomplete"}:
        raise ValueError("paired_case.status is unsupported")
    result: dict[str, Any] = {
        "pair_id": _required_digest(value.get("pair_id"), "paired_case.pair_id"),
        "task_id": _text(value.get("task_id"), "paired_case.task_id", 300),
        "harness": _text(value.get("harness"), "paired_case.harness", 200),
        "attempt": _positive_int(value.get("attempt"), "paired_case.attempt"),
        "status": status,
        "dimension_changes": tuple(
            _canonical_dimension_change(item)
            for item in _sequence(
                value.get("dimension_changes"),
                "paired_case.dimension_changes",
            )
        ),
        "baseline": _optional_canonical_attempt(value.get("baseline")),
        "candidate": _optional_canonical_attempt(value.get("candidate")),
    }
    task_label = _optional_text(value.get("task_label"), "paired_case.task_label", 1000)
    if task_label:
        result["task_label"] = task_label
    return result


def _canonical_paired_case_v3(raw: Any) -> dict[str, Any]:
    value = _mapping(raw, "V3 paired_case")
    allowed = {
        "pair_id",
        "task_id",
        "task_label",
        "harness",
        "attempt",
        "status",
        "dimension_changes",
        "baseline",
        "candidate",
    }
    _reject_unknown(value, allowed, "V3 paired_case")
    status = _text(value.get("status"), "V3 paired_case.status", 40)
    if status not in {"improved", "regressed", "mixed", "unchanged", "incomplete"}:
        raise ValueError("V3 paired_case.status is unsupported")
    result: dict[str, Any] = {
        "pair_id": _required_digest(value.get("pair_id"), "V3 paired_case.pair_id"),
        "task_id": _text(value.get("task_id"), "V3 paired_case.task_id", 300),
        "harness": _text(value.get("harness"), "V3 paired_case.harness", 200),
        "attempt": _positive_int(value.get("attempt"), "V3 paired_case.attempt"),
        "status": status,
        "dimension_changes": tuple(
            _canonical_dimension_change_v3(item)
            for item in _sequence(
                value.get("dimension_changes"),
                "V3 paired_case.dimension_changes",
            )
        ),
        "baseline": _optional_canonical_attempt_v3(value.get("baseline")),
        "candidate": _optional_canonical_attempt_v3(value.get("candidate")),
    }
    task_label = _optional_text(
        value.get("task_label"), "V3 paired_case.task_label", 1000
    )
    if task_label:
        result["task_label"] = task_label
    for arm in ("baseline", "candidate"):
        attempt_value = result.get(arm)
        if not isinstance(attempt_value, Mapping):
            continue
        identity = _mapping(attempt_value.get("identity"), "V3 attempt identity")
        if (
            identity.get("task_id") != result["task_id"]
            or identity.get("arm") != arm
            or identity.get("harness") != result["harness"]
            or identity.get("attempt") != result["attempt"]
        ):
            raise ValueError(
                "V3 paired attempt identity disagrees with its pair coordinates"
            )
    return result


def _canonical_dimension_change(raw: Any) -> dict[str, Any]:
    value = _mapping(raw, "dimension_change")
    _reject_unknown(
        value,
        {"id", "status", "baseline", "candidate", "critical"},
        "dimension_change",
    )
    status = _text(value.get("status"), "dimension_change.status", 40)
    if status not in {"improved", "regressed", "unchanged", "unavailable"}:
        raise ValueError("dimension_change.status is unsupported")
    result: dict[str, Any] = {
        "id": _text(value.get("id"), "dimension_change.id", 500),
        "status": status,
        "critical": _required_bool(value.get("critical"), "dimension_change.critical"),
    }
    for field_name in ("baseline", "candidate"):
        field_value = value.get(field_name)
        if field_value is not None:
            if not isinstance(field_value, bool):
                raise ValueError(f"dimension_change.{field_name} must be boolean")
            result[field_name] = field_value
    return result


def _canonical_dimension_change_v3(raw: Any) -> dict[str, Any]:
    value = _mapping(raw, "V3 dimension_change")
    _reject_unknown(
        value,
        {
            "id",
            "label",
            "status",
            "baseline",
            "candidate",
            "critical",
            "role",
            "baseline_explanation",
            "candidate_explanation",
        },
        "V3 dimension_change",
    )
    status = _text(value.get("status"), "V3 dimension_change.status", 40)
    if status not in {"improved", "regressed", "unchanged", "unavailable"}:
        raise ValueError("V3 dimension_change.status is unsupported")
    role = _text(value.get("role"), "V3 dimension_change.role", 40)
    if role not in {
        "outcome",
        "mechanism",
        "safety_gate",
        "infrastructure",
        "efficiency",
    }:
        raise ValueError("V3 dimension_change.role is unsupported")
    result: dict[str, Any] = {
        "id": _text(value.get("id"), "V3 dimension_change.id", 500),
        "label": _text(value.get("label"), "V3 dimension_change.label", 500),
        "status": status,
        "critical": _required_bool(
            value.get("critical"), "V3 dimension_change.critical"
        ),
        "role": role,
    }
    for field_name in ("baseline", "candidate"):
        field_value = value.get(field_name)
        if field_value is not None:
            if not isinstance(field_value, bool):
                raise ValueError(f"V3 dimension_change.{field_name} must be boolean")
            result[field_name] = field_value
    for field_name in ("baseline_explanation", "candidate_explanation"):
        explanation = _optional_text(
            value.get(field_name),
            f"V3 dimension_change.{field_name}",
            2000,
        )
        if explanation:
            result[field_name] = explanation
    return result


def _optional_canonical_attempt(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    value = _mapping(raw, "paired_attempt")
    allowed = {
        "attempt_id",
        "identity",
        "prediction_id",
        "passed",
        "execution_status",
        "evaluation_status",
        "evidence_status",
        "cost_usd",
        "latency_sec",
        "input_tokens",
        "output_tokens",
        "tool_calls",
        "tools",
        "queried_projects",
        "scores",
        "evidence_links",
        "weave_agent_root_call_id",
        "otel_root_span_id",
        "execution_fingerprint",
        "runtime_lock_digest",
        "infrastructure",
    }
    _reject_unknown(value, allowed, "paired_attempt")
    links = tuple(
        _canonical_attempt_evidence_link(item)
        for item in _sequence(
            value.get("evidence_links"), "paired_attempt.evidence_links"
        )
    )
    expected_kinds = {
        "evaluation_root",
        "prediction_and_score",
        "prediction",
        "agent_root",
        "dataset",
    }
    if len(links) != 5 or {str(item["kind"]) for item in links} != expected_kinds:
        raise ValueError(
            "paired_attempt.evidence_links must contain exactly five unique slots"
        )
    infrastructure = dict(_mapping_or_empty(value.get("infrastructure")))
    legacy_label_boundary = infrastructure.pop("private_label_boundary_verified", None)
    if (
        legacy_label_boundary is not None
        and "label_boundary_verified" in infrastructure
        and infrastructure["label_boundary_verified"] != legacy_label_boundary
    ):
        raise ValueError(
            "paired_attempt.infrastructure has conflicting label-boundary evidence"
        )
    if legacy_label_boundary is not None:
        infrastructure["label_boundary_verified"] = legacy_label_boundary
    result: dict[str, Any] = {
        "attempt_id": _required_digest(
            value.get("attempt_id"), "paired_attempt.attempt_id"
        ),
        "identity": _canonical_attempt_identity(value.get("identity")),
        "execution_status": _text(
            value.get("execution_status"), "paired_attempt.execution_status", 100
        ),
        "evaluation_status": _text(
            value.get("evaluation_status"), "paired_attempt.evaluation_status", 100
        ),
        "evidence_status": _text(
            value.get("evidence_status"), "paired_attempt.evidence_status", 100
        ),
        "tool_calls": _non_negative_int(
            value.get("tool_calls"), "paired_attempt.tool_calls"
        ),
        "tools": tuple(
            _text(item, "paired_attempt.tool", 500)
            for item in _sequence(value.get("tools"), "paired_attempt.tools")
        ),
        "queried_projects": tuple(
            _text(item, "paired_attempt.queried_project", 500)
            for item in _sequence(
                value.get("queried_projects"), "paired_attempt.queried_projects"
            )
        ),
        "scores": dict(_mapping_or_empty(value.get("scores"))),
        "evidence_links": links,
        "infrastructure": infrastructure,
    }
    passed = value.get("passed")
    if passed is not None:
        result["passed"] = _required_bool(passed, "paired_attempt.passed")
    for field_name in (
        "cost_usd",
        "latency_sec",
        "input_tokens",
        "output_tokens",
    ):
        field_value = _optional_float(value.get(field_name))
        if field_value is not None:
            result[field_name] = field_value
    for field_name in (
        "prediction_id",
        "weave_agent_root_call_id",
        "otel_root_span_id",
        "execution_fingerprint",
        "runtime_lock_digest",
    ):
        field_value = _optional_text(
            value.get(field_name), f"paired_attempt.{field_name}", 1000
        )
        if field_value:
            result[field_name] = field_value
    return result


def _optional_canonical_attempt_v3(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    value = _mapping(raw, "V3 paired_attempt")
    extras = {
        "score_explanations",
        "score_details",
        "sanitized_answer_excerpt",
        "actual_query_scope",
        "reported_project_identity",
        "local_evidence_record_digest",
        "local_prediction_row_sha256",
        "local_result_row_projection_digest",
        "cost_reconciliation_status",
        "latency_reconciliation_status",
        "usage_reconciliation_status",
        "judge_reviews",
    }
    base_allowed = {item for item in value if item not in extras}
    base = _optional_canonical_attempt({key: value[key] for key in base_allowed})
    assert base is not None
    expected_attempt_id = canonical_attempt_id(**base["identity"])
    if base["attempt_id"] != expected_attempt_id:
        raise ValueError("V3 paired attempt identity is not canonical")
    if base["execution_status"] not in {
        "completed",
        "failed",
        "cancelled",
        "interrupted",
        "not_applicable",
    }:
        raise ValueError("V3 paired attempt execution status must be terminal")
    unknown = set(value) - set(base_allowed) - extras
    if unknown:
        raise ValueError(
            "V3 paired_attempt has unknown fields: " + ", ".join(sorted(unknown))
        )
    explanations = {
        str(key): _text(
            item,
            f"V3 paired_attempt.score_explanations.{key}",
            2000,
        )
        for key, item in _mapping_or_empty(value.get("score_explanations")).items()
    }
    if set(explanations) != set(base["scores"]):
        raise ValueError("V3 score explanations must cover every published score")
    if any(
        key.startswith("comparison.judge.")
        and explanation
        != "Blind judge score; no rationale or private truth is published."
        for key, explanation in explanations.items()
    ):
        raise ValueError(
            "V3 judge score explanations must not publish rationale or private truth"
        )
    score_details = _optional_score_details_v1(
        value.get("score_details"),
        scores=base["scores"],
    )
    judge_reviews = _optional_judge_reviews_v1(value.get("judge_reviews"))
    actual_query_scope = tuple(
        _text(item, "V3 paired_attempt.actual_query_scope", 500)
        for item in _sequence(
            value.get("actual_query_scope"),
            "V3 paired_attempt.actual_query_scope",
        )
    )
    if actual_query_scope != tuple(base["queried_projects"]):
        raise ValueError("V3 actual query scope must equal normalized queried projects")
    base["score_explanations"] = explanations
    if score_details:
        base["score_details"] = score_details
    if judge_reviews:
        base["judge_reviews"] = judge_reviews
    base["actual_query_scope"] = actual_query_scope
    excerpt = _optional_text(
        value.get("sanitized_answer_excerpt"),
        "V3 paired_attempt.sanitized_answer_excerpt",
        1000,
    )
    if excerpt:
        base["sanitized_answer_excerpt"] = excerpt
    reported = _optional_text(
        value.get("reported_project_identity"),
        "V3 paired_attempt.reported_project_identity",
        300,
    )
    if reported:
        base["reported_project_identity"] = reported
    for field_name in (
        "local_evidence_record_digest",
        "local_prediction_row_sha256",
        "local_result_row_projection_digest",
    ):
        digest = _optional_digest(value.get(field_name), f"V3 {field_name}")
        if digest:
            base[field_name] = digest
    for field_name in (
        "cost_reconciliation_status",
        "latency_reconciliation_status",
        "usage_reconciliation_status",
    ):
        if field_name not in value or value[field_name] is None:
            continue
        status = _text(value[field_name], f"V3 {field_name}", 40)
        if status not in _RECONCILIATION_STATUSES:
            supported = ", ".join(sorted(_RECONCILIATION_STATUSES))
            raise ValueError(f"V3 {field_name} must be one of: {supported}")
        base[field_name] = status
    if base.get("cost_reconciliation_status") == "resolved" and "cost_usd" not in base:
        raise ValueError("resolved cost reconciliation requires cost_usd")
    if (
        base.get("latency_reconciliation_status") == "resolved"
        and "latency_sec" not in base
    ):
        raise ValueError("resolved latency reconciliation requires latency_sec")
    if base.get("usage_reconciliation_status") == "resolved" and (
        "input_tokens" not in base or "output_tokens" not in base
    ):
        raise ValueError(
            "resolved usage reconciliation requires input and output tokens"
        )
    return base


def _score_detail_v1(raw: Any, *, dimension: str) -> dict[str, str]:
    value = _mapping(raw, f"V3 score detail {dimension}")
    if set(value) != {"what", "observed", "why"}:
        raise ValueError(
            f"V3 score detail {dimension!r} requires what, observed, and why"
        )
    detail = {
        field_name: _text(
            value.get(field_name),
            f"V3 score detail {dimension} {field_name}",
            2000,
        )
        for field_name in ("what", "observed", "why")
    }
    if any(redact_text(text) != text for text in detail.values()):
        raise ValueError(f"V3 score detail {dimension!r} is sensitive")
    return detail


def _optional_score_details_v1(
    raw: Any,
    *,
    scores: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    details = {
        str(key): _score_detail_v1(item, dimension=str(key))
        for key, item in _mapping_or_empty(raw).items()
    }
    if set(details) - set(scores):
        raise ValueError("V3 score details reference an unknown score")
    if any(key.startswith("comparison.judge.") for key in details):
        raise ValueError("V3 score details may not publish blind-judge rationale")
    return details


def _optional_judge_reviews_v1(raw: Any) -> dict[str, dict[str, Any]]:
    reviews: dict[str, dict[str, Any]] = {}
    for raw_judge_id, raw_review in _mapping_or_empty(raw).items():
        judge_id = _text(raw_judge_id, "V3 judge review ID", 300)
        review = _mapping(raw_review, f"V3 judge review {judge_id}")
        _reject_unknown(
            review,
            {
                "label",
                "reason",
                "missing_evidence",
                "observed_cost_usd",
                "cost_status",
            },
            f"V3 judge review {judge_id}",
        )
        label = str(review.get("label") or "")
        if label not in {
            "unusable",
            "weak",
            "adequate",
            "strong",
            "exceptional",
        }:
            raise ValueError(f"V3 judge review {judge_id!r} label is unsupported")
        reason = _text(review.get("reason"), "V3 judge review reason", 500)
        if redact_text(reason) != reason:
            raise ValueError(f"V3 judge review {judge_id!r} reason is sensitive")
        missing_evidence = _required_bool(
            review.get("missing_evidence"),
            "V3 judge review missing_evidence",
        )
        cost_status = review.get("cost_status")
        if cost_status not in {None, "observed", "unavailable"}:
            raise ValueError(
                f"V3 judge review {judge_id!r} cost status is unsupported"
            )
        cost = _optional_float(review.get("observed_cost_usd"))
        if cost is not None and cost < 0:
            raise ValueError(
                f"V3 judge review {judge_id!r} observed cost is invalid"
            )
        if cost_status == "observed" and cost is None:
            raise ValueError(
                f"V3 judge review {judge_id!r} observed cost is missing"
            )
        if cost_status == "unavailable" and cost is not None:
            raise ValueError(
                f"V3 judge review {judge_id!r} unavailable cost has a value"
            )
        normalized: dict[str, Any] = {
            "label": label,
            "reason": reason,
            "missing_evidence": missing_evidence,
        }
        if cost is not None:
            normalized["observed_cost_usd"] = cost
        if cost_status is not None:
            normalized["cost_status"] = cost_status
        reviews[judge_id] = normalized
    return reviews


def _canonical_attempt_identity(raw: Any) -> dict[str, Any]:
    value = _mapping(raw, "paired_attempt.identity")
    allowed = {"task_id", "arm", "harness", "attempt", "candidate", "runtime"}
    _reject_unknown(value, allowed, "paired_attempt.identity")
    return {
        "task_id": _text(value.get("task_id"), "attempt identity task_id", 300),
        "arm": _text(value.get("arm"), "attempt identity arm", 100),
        "harness": _text(value.get("harness"), "attempt identity harness", 200),
        "attempt": _positive_int(value.get("attempt"), "attempt identity attempt"),
        "candidate": _text(value.get("candidate"), "attempt identity candidate", 1000),
        "runtime": _text(value.get("runtime"), "attempt identity runtime", 1000),
    }


def _canonical_attempt_evidence_link(raw: Any) -> dict[str, Any]:
    value = _mapping(raw, "paired_attempt.evidence_link")
    _reject_unknown(
        value,
        {"kind", "status", "system", "ref", "url", "reason"},
        "paired_attempt.evidence_link",
    )
    kind = _text(value.get("kind"), "attempt evidence kind", 100)
    if kind not in {
        "evaluation_root",
        "prediction_and_score",
        "prediction",
        "agent_root",
        "dataset",
    }:
        raise ValueError("attempt evidence kind is unsupported")
    status = _text(value.get("status"), "attempt evidence status", 40)
    if status not in {"resolved", "missing", "invalid"}:
        raise ValueError("attempt evidence status is unsupported")
    system = _text(value.get("system") or "weave", "attempt evidence system", 100)
    if system not in {"weave", "local_artifact"}:
        raise ValueError("attempt evidence system must be weave or local_artifact")
    result: dict[str, Any] = {"kind": kind, "status": status, "system": system}
    ref = _optional_text(value.get("ref"), "attempt evidence ref", 2000)
    url = _optional_text(value.get("url"), "attempt evidence url", 2000)
    reason = _optional_text(value.get("reason"), "attempt evidence reason", 1000)
    if status == "resolved":
        if system == "weave" and (not ref or not url or not url.startswith("https://")):
            raise ValueError("resolved Weave evidence requires ref and HTTPS url")
        if system == "local_artifact" and (
            not ref or not ref.startswith("fugue://local-evidence/") or url
        ):
            raise ValueError(
                "resolved local evidence requires a canonical local-artifact ref"
            )
    elif not reason:
        raise ValueError("unresolved attempt evidence requires a reason")
    if ref:
        result["ref"] = ref
    if url:
        if not url.startswith("https://"):
            raise ValueError("attempt evidence URLs must use HTTPS")
        result["url"] = url
    if reason:
        result["reason"] = reason
    return result


def _comparison_attempt_evidence_links(raw: Any) -> tuple[dict[str, str], ...]:
    links: list[dict[str, str]] = []
    for item in _sequence(raw, "attempt evidence links"):
        if not isinstance(item, Mapping):
            continue
        if str(item.get("status") or "") != "resolved":
            continue
        url = str(item.get("url") or item.get("uri") or "")
        ref = str(item.get("ref") or url)
        if not ref:
            continue
        links.append(
            {
                "system": str(item.get("system") or "weave"),
                "kind": str(item.get("kind") or "evidence"),
                "ref": ref,
                **({"uri": url} if url.startswith("https://") else {}),
            }
        )
    return _evidence_links(links)


def _comparison_attempt_measures(
    attempt: Mapping[str, Any],
) -> dict[str, str | int | float | bool | None]:
    result: dict[str, str | int | float | bool | None] = {}
    for source, target in (
        ("input_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
        ("tool_calls", "tool_calls"),
        ("latency_sec", "wall_time_sec"),
    ):
        value = attempt.get(source)
        if isinstance(value, int | float) and not isinstance(value, bool):
            result[target] = value
    return result


def _comparison_attempt_scores(raw: Any) -> tuple[ExperimentScoreResultV1, ...]:
    result: list[ExperimentScoreResultV1] = []
    for score_id, value in _mapping_or_empty(raw).items():
        passed = _score_passed(value)
        result.append(
            ExperimentScoreResultV1(
                id=str(score_id),
                label=humanize_display_id(str(score_id)),
                status=(
                    "passed"
                    if passed is True
                    else "failed"
                    if passed is False
                    else "observed"
                ),
                value=value if isinstance(value, str | int | float | bool) else None,
            )
        )
    return tuple(result)


def _score_passed(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value) == 1.0
    return None


def _comparison_pair_evidence_links(
    raw: Any,
    *,
    task_id: str,
    arm: str,
) -> tuple[dict[str, str], ...]:
    labels = {
        f"Agent root — {task_id} — {arm}": "agent_conversation",
        f"Evaluation attempt — {task_id} — {arm}": "evaluation_attempt",
        f"Evaluation prediction — {task_id} — {arm}": "prediction",
    }
    links: list[dict[str, str]] = []
    for item in _sequence(raw, "comparison evidence links"):
        if not isinstance(item, Mapping):
            continue
        kind = labels.get(str(item.get("label") or ""))
        url = str(item.get("url") or "")
        if kind is None or not url.startswith("https://") or len(url) > 1000:
            continue
        links.append(
            {
                "system": "weave",
                "kind": kind,
                "ref": url,
                "uri": url,
            }
        )
    return _evidence_links(links)


def _comparison_evidence_links(
    raw: Any,
    *,
    result_ref: str | None,
    result_source: str,
    result_digest: str,
) -> tuple[dict[str, str], ...]:
    links: list[dict[str, str]] = []
    raw_links = _sequence(raw, "comparison evidence links")
    for item in raw_links:
        if not isinstance(item, Mapping):
            continue
        label = str(item.get("label") or "Evidence")
        url = str(item.get("url") or "")
        if not url or len(url) > 1000:
            continue
        kind = label.lower().replace(" ", "_")
        link = {
            "system": "weave" if "weave" in url.lower() else "wandb",
            "kind": kind,
            "ref": url,
        }
        if url.startswith("https://"):
            link["uri"] = url
        links.append(link)
    digest = result_digest if len(result_digest) == 64 else ""
    comparison_rows_ref = result_source
    if comparison_rows_ref and len(comparison_rows_ref) <= 1000:
        links.append(
            {
                "system": "fugue",
                "kind": "comparison_rows",
                "ref": comparison_rows_ref,
            }
        )
    if result_ref and len(result_ref) <= 1000:
        links.append(
            {
                "system": "fugue",
                "kind": "comparison_result",
                "ref": result_ref,
                **({"digest": digest} if digest else {}),
            }
        )
    return _evidence_links(links)


def _wandb_run_id_from_url(value: str) -> str | None:
    if not value.startswith("https://") or len(value) > 2000:
        return None
    parsed = urllib.parse.urlsplit(value)
    if not parsed.netloc or parsed.username or parsed.password:
        return None
    path_parts = tuple(part for part in parsed.path.split("/") if part)
    for index, part in enumerate(path_parts):
        if part == "runs" and index + 1 < len(path_parts):
            return path_parts[index + 1]
    return None


def _comparison_behavioral_measures(
    operational: Mapping[str, Any],
) -> dict[str, Any]:
    measures: dict[str, Any] = {}
    usage_rows = int(operational.get("usage_rows") or 0)
    for key in ("input_tokens", "output_tokens"):
        value = operational.get(key)
        if (
            usage_rows
            and isinstance(value, int | float)
            and not isinstance(value, bool)
        ):
            measures[key] = {
                "observed": usage_rows,
                "mean": float(value) / usage_rows,
            }
    latency_rows = int(operational.get("latency_rows") or 0)
    latency_ms = operational.get("latency_ms")
    if (
        latency_rows
        and isinstance(latency_ms, int | float)
        and not isinstance(latency_ms, bool)
    ):
        measures["wall_time_sec"] = {
            "observed": latency_rows,
            "mean": float(latency_ms) / latency_rows / 1000,
        }
    return measures


def _comparison_mechanism_funnel(raw: Any) -> tuple[ExperimentMechanismStageV1, ...]:
    stages: list[ExperimentMechanismStageV1] = []
    for stage_id, raw_stage in _mapping_or_empty(raw).items():
        if not isinstance(raw_stage, Mapping):
            continue
        by_arm: list[ExperimentMechanismArmV1] = []
        for arm in ("baseline", "candidate"):
            values = raw_stage.get(arm)
            if not isinstance(values, Mapping):
                continue
            eligible = int(values.get("applicable") or 0)
            reached = min(eligible, int(values.get("observed") or 0))
            by_arm.append(
                ExperimentMechanismArmV1(
                    arm=arm,
                    harness="all",
                    eligible=eligible,
                    reached=reached,
                )
            )
        stages.append(
            ExperimentMechanismStageV1(
                id=str(stage_id),
                label=humanize_display_id(str(stage_id)),
                eligible=sum(item.eligible for item in by_arm),
                reached=sum(item.reached for item in by_arm),
                by_arm=tuple(by_arm),
            )
        )
    return tuple(stages)


def _comparison_outcome_summaries(
    result: Mapping[str, Any],
    *,
    baseline_total: int,
    candidate_total: int,
    infrastructure_failures: int,
    missing_evidence: int,
) -> tuple[ExperimentOutcomeSummaryV1, ...]:
    def arm_summary(
        arm: str,
        label: str,
        passed: int,
        total: int,
    ) -> ExperimentOutcomeSummaryV1:
        return ExperimentOutcomeSummaryV1(
            id=f"{arm}-deterministic",
            label=label,
            status=(
                "unavailable"
                if not total
                else "passed"
                if passed == total
                else "failed"
            ),
            passed=passed,
            total=total,
            unavailable=0,
        )

    rows = int(result.get("rows") or 0)
    summaries = [
        arm_summary(
            "baseline",
            "Baseline deterministic task outcome",
            int(result.get("baseline_passed") or 0),
            baseline_total,
        ),
        arm_summary(
            "candidate",
            "Candidate deterministic task outcome",
            int(result.get("candidate_passed") or 0),
            candidate_total,
        ),
        ExperimentOutcomeSummaryV1(
            id="infrastructure",
            label="Infrastructure execution",
            status=(
                "unavailable"
                if not rows
                else "passed"
                if not infrastructure_failures
                else "failed"
            ),
            passed=max(0, rows - infrastructure_failures),
            total=rows,
            unavailable=0,
        ),
        ExperimentOutcomeSummaryV1(
            id="evidence-reconciliation",
            label="Required evidence reconciliation",
            status=(
                "unavailable"
                if not rows
                else "passed"
                if not missing_evidence
                else "failed"
            ),
            passed=max(0, rows - missing_evidence),
            total=rows,
            unavailable=missing_evidence,
        ),
    ]
    judge = _mapping_or_empty(result.get("judge_summary"))
    judge_status = str(judge.get("status") or "not_used")
    unavailable = int(judge.get("unavailable_attempts") or judge.get("attempts") or 0)
    if judge_status == "not_used":
        summaries.append(
            ExperimentOutcomeSummaryV1(
                id="judge-evidence",
                label="Judge evidence completeness",
                status="not_applicable",
                unavailable=0,
            )
        )
    elif judge_status == "unavailable":
        summaries.append(
            ExperimentOutcomeSummaryV1(
                id="judge-evidence",
                label="Judge evidence completeness",
                status="unavailable",
                total=unavailable,
                unavailable=unavailable,
            )
        )
    else:
        observed = sum(
            int(values.get("evaluated") or 0)
            for dimensions in _mapping_or_empty(judge.get("by_variant")).values()
            if isinstance(dimensions, Mapping)
            for values in dimensions.values()
            if isinstance(values, Mapping)
        )
        summaries.append(
            ExperimentOutcomeSummaryV1(
                id="judge-evidence",
                label="Judge evidence completeness",
                status=(
                    "unavailable"
                    if not observed
                    else "passed"
                    if not unavailable
                    else "failed"
                ),
                passed=max(0, observed - unavailable),
                total=observed,
                unavailable=unavailable,
            )
        )
    return tuple(summaries)


def _comparison_score_summaries(
    result: Mapping[str, Any],
) -> tuple[ExperimentScoreSummaryV1, ...]:
    summaries: list[ExperimentScoreSummaryV1] = []
    deterministic = _mapping_or_empty(result.get("deterministic_summary"))
    dimensions: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for arm in ("baseline", "candidate"):
        arm_summary = deterministic.get(arm)
        if not isinstance(arm_summary, Mapping):
            continue
        for dimension, values in _mapping_or_empty(
            arm_summary.get("dimensions")
        ).items():
            if isinstance(values, Mapping):
                dimensions[str(dimension)].append(values)
    for dimension, values in sorted(dimensions.items()):
        observed = sum(int(item.get("evaluated") or 0) for item in values)
        passed = sum(int(item.get("passed") or 0) for item in values)
        means = [
            float(item["mean"])
            for item in values
            if isinstance(item.get("mean"), int | float)
            and not isinstance(item.get("mean"), bool)
        ]
        summaries.append(
            ExperimentScoreSummaryV1(
                id=f"deterministic-{dimension}",
                label=f"Deterministic: {humanize_display_id(dimension)}",
                observed=observed,
                passed=passed,
                failed=max(0, observed - passed),
                mean=(sum(means) / len(means) if means else None),
            )
        )
    judge = _mapping_or_empty(result.get("judge_summary"))
    for arm, raw_dimensions in _mapping_or_empty(judge.get("by_variant")).items():
        if not isinstance(raw_dimensions, Mapping):
            continue
        for dimension, values in raw_dimensions.items():
            if not isinstance(values, Mapping):
                continue
            summaries.append(
                ExperimentScoreSummaryV1(
                    id=f"judge-{arm}-{dimension}",
                    label=(
                        f"Judge {humanize_display_id(str(arm))}: "
                        f"{humanize_display_id(str(dimension))}"
                    ),
                    observed=int(values.get("evaluated") or 0),
                    mean=_optional_float(values.get("mean")),
                )
            )
    return tuple(summaries)


def _safe_judge_summary(
    raw: Any,
    *,
    integrity_status: str,
    attempts: int,
    arm_attempts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Validate and return only aggregate blind-judge scores safe to publish."""

    value = _mapping_or_empty(raw)
    _reject_unknown(
        value,
        {
            "status",
            "claim_status",
            "judges",
            "by_variant",
            "unavailable_attempts",
        },
        "judge_summary",
    )
    status = str(value.get("status") or "not_used")
    if status not in {"not_used", "unavailable", "scored"}:
        raise ValueError("judge_summary.status is unsupported")
    claim_status = str(value.get("claim_status") or "")
    if claim_status not in {
        "not_applicable",
        "advisory_uncalibrated",
        "calibrated",
    }:
        raise ValueError("judge_summary.claim_status is unsupported")
    judges = _safe_judge_provenance(value.get("judges"))
    by_variant = _safe_judge_variants(
        value.get("by_variant"),
        judge_dimensions={item["judge_id"]: set(item["dimensions"]) for item in judges},
        arm_attempts=arm_attempts,
    )
    unavailable = _non_negative_int(
        value.get("unavailable_attempts"),
        "judge_summary.unavailable_attempts",
    )
    if unavailable > attempts:
        raise ValueError("judge unavailable_attempts exceeds the attempt count")
    calibrated = bool(judges) and all(
        item["calibration"]["status"] == "adjudicated"
        and item["calibration"]["passed"] is True
        for item in judges
    )
    if status == "not_used":
        if (
            claim_status != "not_applicable"
            or judges
            or by_variant["baseline"]
            or by_variant["candidate"]
            or unavailable
        ):
            raise ValueError(
                "a not-used judge summary cannot retain provenance or claims"
            )
    else:
        if not judges:
            raise ValueError(
                "configured judge summaries require qualification provenance"
            )
        if claim_status == "not_applicable":
            raise ValueError("a configured judge summary cannot be not applicable")
        if (claim_status == "calibrated") != calibrated:
            raise ValueError("judge claim status disagrees with calibration provenance")
        if status == "unavailable" and (
            by_variant["baseline"] or by_variant["candidate"]
        ):
            raise ValueError(
                "an unavailable judge summary cannot retain scored dimensions"
            )
        if status == "scored" and not by_variant["baseline"]:
            raise ValueError("a scored judge summary requires non-empty dimensions")
        if status == "scored" and arm_attempts is not None:
            evaluated = sum(
                next(iter(by_variant[arm].values()))["evaluated"]
                for arm in ("baseline", "candidate")
            )
            if evaluated + unavailable != attempts:
                raise ValueError(
                    "judge evaluated and unavailable counts do not reconcile "
                    "to canonical attempts"
                )
    normalized = {
        "status": status,
        "claim_status": claim_status,
        "judges": judges,
        "by_variant": by_variant,
        "unavailable_attempts": unavailable,
    }
    if integrity_status != "invalid" or status == "not_used":
        return normalized
    return {
        **normalized,
        "status": "unavailable",
        "by_variant": {"baseline": {}, "candidate": {}},
        "unavailable_attempts": max(attempts, unavailable),
    }


def _safe_judge_provenance(raw: Any) -> list[dict[str, Any]]:
    judges: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_judge in enumerate(_sequence(raw, "judge_summary.judges")):
        judge = _mapping(raw_judge, f"judge_summary.judges[{index}]")
        _reject_unknown(
            judge,
            {
                "judge_id",
                "profile",
                "contract_digest",
                "dimensions",
                "calibration",
            },
            f"judge_summary.judges[{index}]",
        )
        judge_id = _text(judge.get("judge_id"), "judge_id", 300)
        if judge_id in seen:
            raise ValueError("judge provenance IDs must be unique")
        seen.add(judge_id)
        contract_digest = _strict_sha256(
            judge.get("contract_digest"),
            "judge contract_digest",
        )
        calibration = _mapping(
            judge.get("calibration"),
            f"judge_summary.judges[{index}].calibration",
        )
        _reject_unknown(
            calibration,
            {"status", "report_sha256", "cases_digest", "passed"},
            f"judge_summary.judges[{index}].calibration",
        )
        calibration_status = str(calibration.get("status") or "")
        if calibration_status not in {
            "missing",
            "pending_human_review",
            "adjudicated",
            "invalid",
        }:
            raise ValueError("judge calibration status is unsupported")
        report_sha256 = _optional_strict_sha256(
            calibration.get("report_sha256"),
            "judge calibration report_sha256",
        )
        cases_digest = _optional_strict_sha256(
            calibration.get("cases_digest"),
            "judge calibration cases_digest",
        )
        passed = _required_bool(
            calibration.get("passed"),
            "judge calibration passed",
        )
        if passed and calibration_status != "adjudicated":
            raise ValueError("a passing judge calibration must be adjudicated")
        if calibration_status in {"pending_human_review", "adjudicated"} and (
            report_sha256 is None or cases_digest is None
        ):
            raise ValueError(
                "reviewed judge calibration requires report and case-set digests"
            )
        judges.append(
            {
                "judge_id": judge_id,
                "profile": _text(judge.get("profile"), "judge profile", 500),
                "contract_digest": contract_digest,
                "dimensions": _unique_judge_dimensions(
                    judge.get("dimensions"),
                    judge_id=judge_id,
                ),
                "calibration": {
                    "status": calibration_status,
                    "report_sha256": report_sha256,
                    "cases_digest": cases_digest,
                    "passed": passed,
                },
            }
        )
    return judges


def _safe_judge_variants(
    raw: Any,
    *,
    judge_dimensions: Mapping[str, set[str]],
    arm_attempts: Mapping[str, int] | None,
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    variants = _mapping(raw, "judge_summary.by_variant")
    _reject_unknown(
        variants,
        {"baseline", "candidate"},
        "judge_summary.by_variant",
    )
    if set(variants) != {"baseline", "candidate"}:
        raise ValueError("judge_summary.by_variant requires baseline and candidate")
    result: dict[str, dict[str, dict[str, float | int | None]]] = {}
    expected_dimensions = {
        f"{judge_id}.{dimension}"
        for judge_id, dimensions in judge_dimensions.items()
        for dimension in dimensions
    }
    for variant in ("baseline", "candidate"):
        dimensions: dict[str, dict[str, float | int | None]] = {}
        raw_dimensions = _mapping(
            variants[variant],
            f"judge_summary.by_variant.{variant}",
        )
        for raw_dimension, raw_summary in raw_dimensions.items():
            dimension = _text(
                raw_dimension,
                "judge summary dimension",
                600,
            )
            if dimension not in expected_dimensions:
                raise ValueError("judge summary dimension lacks matching provenance")
            summary = _mapping(
                raw_summary,
                f"judge_summary.by_variant.{variant}.{dimension}",
            )
            _reject_unknown(
                summary,
                {"evaluated", "mean"},
                f"judge_summary.by_variant.{variant}.{dimension}",
            )
            evaluated = _non_negative_int(
                summary.get("evaluated"),
                "judge summary evaluated",
            )
            mean = _optional_float(summary.get("mean"))
            if mean is not None and not 0 <= mean <= 1:
                raise ValueError("judge summary means must be between zero and one")
            if (mean is None) != (evaluated == 0):
                raise ValueError(
                    "judge summary mean must be null exactly when no rows were evaluated"
                )
            if arm_attempts is not None and evaluated > arm_attempts[variant]:
                raise ValueError("judge evaluated count exceeds canonical arm attempts")
            dimensions[dimension] = {
                "evaluated": evaluated,
                "mean": mean,
            }
        result[variant] = dimensions
    if set(result["baseline"]) != set(result["candidate"]):
        raise ValueError(
            "judge summary must cover the same dimensions for baseline and candidate"
        )
    if result["baseline"] and set(result["baseline"]) != expected_dimensions:
        raise ValueError(
            "judge summary dimensions do not match locked judge provenance"
        )
    for variant in ("baseline", "candidate"):
        if len({summary["evaluated"] for summary in result[variant].values()}) > 1:
            raise ValueError("judge dimensions disagree on evaluated attempt counts")
    return result


def _unique_judge_dimensions(
    raw: Any,
    *,
    judge_id: str,
) -> list[str]:
    dimensions = [
        _text(item, f"judge {judge_id} dimension", 300)
        for item in _sequence(raw, f"judge {judge_id} dimensions")
    ]
    if not dimensions or len(dimensions) != len(set(dimensions)):
        raise ValueError("judge provenance dimensions must be non-empty and unique")
    return dimensions


def _judge_arm_attempt_counts(
    paired_cases: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    return {
        arm: sum(isinstance(pair.get(arm), Mapping) for pair in paired_cases)
        for arm in ("baseline", "candidate")
    }


def _strict_sha256(raw: Any, field_name: str) -> str:
    value = _text(raw, field_name, 64)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _optional_strict_sha256(raw: Any, field_name: str) -> str | None:
    if raw is None:
        return None
    return _strict_sha256(raw, field_name)


def build_design_view(
    preview: Mapping[str, Any], *, approval_state: str = "awaiting_approval"
) -> ExperimentViewV1:
    draft = _mapping(preview.get("draft"), "draft")
    plan = _mapping_or_empty(preview.get("plan_receipt"))
    cells = [
        item
        for item in _sequence(plan.get("cells"), "plan_receipt.cells")
        if isinstance(item, Mapping)
    ]
    fixed_names = tuple(str(item) for item in draft.get("fixed_dimensions") or ())
    varied_names = tuple(str(item) for item in draft.get("varied_dimensions") or ())
    labels = _display_labels(draft.get("display_labels"))
    research_view = _mapping_or_empty(draft.get("research_view"))
    arm_factor_levels = _mapping_or_empty(research_view.get("arm_factor_levels"))
    fixed = tuple(
        ExperimentFactorV1(
            name=_dimension_label(name),
            levels=_levels_for(name, draft, cells),
            label=labels.get(_dimension_label(name), labels.get(name)),
            level_labels=_factor_level_labels(name, draft, cells, labels),
        )
        for name in fixed_names
    )
    varied = tuple(
        ExperimentFactorV1(
            name=_dimension_label(name),
            levels=_varied_levels_for(
                name,
                draft,
                cells,
                arm_factor_levels=arm_factor_levels,
            ),
            label=labels.get(_dimension_label(name), labels.get(name)),
            level_labels=_varied_factor_level_labels(
                name,
                draft,
                cells,
                labels,
                arm_factor_levels=arm_factor_levels,
            ),
        )
        for name in varied_names
    )
    matrix_size = int(preview.get("estimated_cells") or len(cells))
    display_cells = tuple(
        _planned_cell(
            item,
            tuple(factor.name for factor in varied),
            arm_factor_levels=arm_factor_levels,
        )
        for item in cells[:EXPERIMENT_VIEW_CELL_LIMIT]
    )
    task_count = draft.get("n_tasks")
    if task_count is None:
        task_count = len(
            {str(item.get("task_id") or "") for item in cells if item.get("task_id")}
        )
    taskset_digest = str(
        draft.get("task_suite_digest")
        or _mapping_or_empty(preview.get("task_suite_preview")).get("preview_digest")
        or ""
    )
    taskset = ExperimentDescriptorV1(
        id=str(
            draft.get("task_suite_digest")
            or draft.get("preset_id")
            or "registered-taskset"
        ),
        label=_taskset_label(draft, task_count),
        digest=taskset_digest or None,
        details={"task_count": int(task_count or 0)},
    )
    preview_digest = str(preview.get("preview_digest") or "")
    harness_ids = _ordered_values(
        [str(item) for item in draft.get("harnesses") or ()]
        + [str(item.get("harness") or "") for item in cells]
    )
    recipe = _mapping_or_empty(draft.get("task_recipe_preview"))
    provenance = _mapping_or_empty(recipe.get("provenance"))
    selected_calls = [
        item
        for item in provenance.get("selected_call_ids") or ()
        if isinstance(item, str) and item
    ]
    source_cohort = None
    if provenance.get("trace_audit_id") and selected_calls:
        source_cohort = ExperimentDescriptorV1(
            id=str(provenance["trace_audit_id"]),
            label=(
                f"{len(selected_calls)} reviewed Weave "
                f"call{'s' if len(selected_calls) != 1 else ''}"
            ),
            digest=(
                str(provenance["trace_audit_digest"])
                if provenance.get("trace_audit_digest")
                else None
            ),
            details={
                "call_count": len(selected_calls),
                "system": "weave",
                **(
                    {
                        "reviewed_cohort_manifest_digest": str(
                            provenance["reviewed_cohort_manifest_digest"]
                        )
                    }
                    if provenance.get("reviewed_cohort_manifest_digest")
                    else {}
                ),
            },
        )
    context = str(draft.get("decision_rationale") or "").strip() or None
    task_design = _research_task_design(research_view, recipe)
    prompt_design = _research_prompt_design(research_view)
    evaluation_design = _research_evaluation_design(research_view)
    if task_design is not None and not task_design.evidence_links:
        task_reference = {
            "system": "fugue",
            "kind": "task_definition",
            "ref": taskset.id,
        }
        if taskset.digest:
            task_reference["digest"] = taskset.digest
        task_design = ExperimentTaskDesignV1(
            title=task_design.title,
            summary=task_design.summary,
            interaction_mode=task_design.interaction_mode,
            tools=task_design.tools,
            resources=task_design.resources,
            evidence_links=(task_reference,),
        )
    if (
        prompt_design is not None
        and not prompt_design.evidence_links
        and preview_digest
    ):
        prompt_design = ExperimentPromptDesignV1(
            base_instruction_summary=prompt_design.base_instruction_summary,
            treatment_summaries=prompt_design.treatment_summaries,
            evidence_links=(
                {
                    "system": "fugue",
                    "kind": "prompt_design",
                    "ref": preview_digest,
                    "digest": preview_digest,
                },
            ),
        )
    return experiment_view_from_dict(
        ExperimentViewV1(
            schema_version=EXPERIMENT_VIEW_SCHEMA_VERSION,
            kind="design",
            research_label=labels.get("research"),
            study_label=labels.get("study"),
            question=str(draft.get("question") or draft.get("research_question") or ""),
            hypothesis=str(draft.get("hypothesis") or ""),
            context=context,
            observation=str(research_view.get("observation") or "").strip() or None,
            rationale=str(research_view.get("rationale") or "").strip() or context,
            alternative_explanations=tuple(
                str(item)
                for item in research_view.get("alternative_explanations") or ()
                if str(item).strip()
            ),
            success_definition=(
                str(research_view.get("success_definition") or "").strip() or None
            ),
            task_design=task_design,
            prompt_design=prompt_design,
            evaluation_design=evaluation_design,
            source_cohort=source_cohort,
            fixed_conditions=fixed,
            varied_factors=varied,
            treatment_arms=_research_treatment_arms(research_view, labels),
            measured_outcomes=tuple(
                str(item) for item in draft.get("measured_dimensions") or ()
            ),
            taskset=taskset,
            harnesses=tuple(
                ExperimentDescriptorV1(
                    id=value, label=labels.get(value, humanize_display_id(value))
                )
                for value in harness_ids
            ),
            runtime=ExperimentDescriptorV1(
                id="harbor",
                label="Harbor isolated runtime",
                details={"locked_before_execution": True},
            ),
            matrix_size=matrix_size,
            preview_digest=preview_digest or None,
            approval_state=approval_state,
            cell_limit=matrix_size,
            reserved_cost_usd=float(preview.get("estimated_cost_usd") or 0.0),
            evidence_scope=_optional_evidence_scope(preview.get("evidence_scope")),
            cells=display_cells,
            omitted_cells=max(0, len(cells) - len(display_cells)),
        ).to_dict()
    )


def _research_task_design(
    research_view: Mapping[str, Any],
    recipe: Mapping[str, Any],
) -> ExperimentTaskDesignV1 | None:
    title = str(research_view.get("task_title") or "").strip()
    summary = str(research_view.get("task_summary") or "").strip()
    if not title or not summary:
        return None
    links: list[dict[str, str]] = []
    preview_digest = str(recipe.get("preview_digest") or "")
    recipe_id = str(recipe.get("recipe_id") or "")
    if recipe_id and preview_digest:
        links.append(
            {
                "system": "fugue",
                "kind": "task_definition",
                "ref": recipe_id,
                "digest": preview_digest,
            }
        )
    return ExperimentTaskDesignV1(
        title=title,
        summary=summary,
        interaction_mode=(
            str(research_view.get("interaction_mode") or "").strip() or None
        ),
        tools=tuple(
            str(item) for item in research_view.get("tools") or () if str(item).strip()
        ),
        resources=tuple(
            str(item)
            for item in research_view.get("resources") or ()
            if str(item).strip()
        ),
        evidence_links=tuple(links),
    )


def _research_prompt_design(
    research_view: Mapping[str, Any],
) -> ExperimentPromptDesignV1 | None:
    summary = str(research_view.get("base_instruction_summary") or "").strip()
    treatments = {
        str(key): str(value)
        for key, value in _mapping_or_empty(
            research_view.get("treatment_summaries")
        ).items()
        if str(key).strip() and str(value).strip()
    }
    if not summary and not treatments:
        return None
    return ExperimentPromptDesignV1(
        base_instruction_summary=summary or "No additional base instruction summary.",
        treatment_summaries=treatments,
    )


def _research_evaluation_design(
    research_view: Mapping[str, Any],
) -> ExperimentEvaluationDesignV1 | None:
    raw_scorers = [
        item for item in research_view.get("scorers") or () if isinstance(item, Mapping)
    ]
    pass_rule = str(research_view.get("pass_rule") or "").strip()
    if not raw_scorers or not pass_rule:
        return None
    scorers: list[ExperimentScorerDesignV1] = []
    for raw in raw_scorers:
        dimensions = tuple(
            ExperimentScoreDefinitionV1(
                id=str(item.get("id") or ""),
                label=str(item.get("label") or ""),
                description=str(item.get("description") or "").strip() or None,
                source_key=str(item.get("source_key") or "").strip() or None,
                target=item.get("target"),
                primary=bool(item.get("primary", False)),
            )
            for item in raw.get("dimensions") or ()
            if isinstance(item, Mapping)
        )
        revision = str(raw.get("revision") or "").strip() or None
        links = (
            (
                {
                    "system": "fugue",
                    "kind": "scorer_revision",
                    "ref": revision,
                },
            )
            if revision
            else ()
        )
        scorers.append(
            ExperimentScorerDesignV1(
                id=str(raw.get("id") or ""),
                label=str(raw.get("label") or ""),
                kind=str(raw.get("kind") or ""),  # type: ignore[arg-type]
                description=str(raw.get("description") or ""),
                required=bool(raw.get("required", True)),
                threshold=_optional_float(raw.get("threshold")),
                aggregation=str(raw.get("aggregation") or "").strip() or None,
                evidence_inputs=tuple(
                    str(item)
                    for item in raw.get("evidence_inputs") or ()
                    if str(item).strip()
                ),
                revision=revision,
                model=str(raw.get("model") or "").strip() or None,
                rubric_summary=(str(raw.get("rubric_summary") or "").strip() or None),
                blind_fields=tuple(
                    str(item)
                    for item in raw.get("blind_fields") or ()
                    if str(item).strip()
                ),
                dimensions=dimensions,
                evidence_links=links,
            )
        )
    return ExperimentEvaluationDesignV1(
        pass_rule=pass_rule,
        scorers=tuple(scorers),
        llm_judge_used=any(item.kind == "llm_judge" for item in scorers),
    )


def build_progress_view(
    record: Mapping[str, Any], run_summary: Mapping[str, Any]
) -> ExperimentViewV1:
    preview = _mapping(record.get("preview"), "preview")
    plan = _mapping_or_empty(preview.get("plan_receipt"))
    plan_cells = {
        str(item.get("coordinate_id") or item.get("cell_id") or ""): item
        for item in plan.get("cells") or ()
        if isinstance(item, Mapping)
    }
    cells = [
        _running_cell(item, plan_cells)
        for item in _sequence(run_summary.get("cells"), "run_summary.cells")
    ]
    state_counts = _evaluation_state_counts(cells)
    completed = sum(
        item.execution_status
        in {"completed", "failed", "cancelled", "interrupted", "not_applicable"}
        for item in cells
    )
    displayed = tuple(cells[:EXPERIMENT_VIEW_CELL_LIMIT])
    return experiment_view_from_dict(
        ExperimentViewV1(
            schema_version=EXPERIMENT_VIEW_SCHEMA_VERSION,
            kind="progress",
            matrix_size=int(preview.get("estimated_cells") or len(cells)),
            preview_digest=str(preview.get("preview_digest") or "") or None,
            approval_state="approved"
            if record.get("approval")
            else "awaiting_approval",
            cell_limit=int(preview.get("estimated_cells") or len(cells)),
            reserved_cost_usd=_record_reserved_cost(record),
            phase=_phase(
                str(record.get("state") or ""), str(run_summary.get("status") or "")
            ),
            completed_cells=completed,
            state_counts=state_counts,
            evidence_scope=_optional_evidence_scope(preview.get("evidence_scope")),
            cells=displayed,
            omitted_cells=max(0, len(cells) - len(displayed)),
        ).to_dict()
    )


def build_evaluation_view(record: Mapping[str, Any]) -> ExperimentViewV1:
    preview = _mapping(record.get("preview"), "preview")
    draft = _mapping(preview.get("draft"), "draft")
    research_view = _mapping_or_empty(draft.get("research_view"))
    evaluation_design = _research_evaluation_design(research_view)
    outcome = _mapping_or_empty(record.get("outcome"))
    rows = [item for item in outcome.get("row_refs") or () if isinstance(item, Mapping)]
    evidence_by_prediction = {
        str(item.get("prediction_id") or ""): item
        for item in outcome.get("evidence_refs") or ()
        if isinstance(item, Mapping)
    }
    evaluation = _mapping_or_empty(record.get("evaluation"))
    evaluation_by_prediction = {
        str(item.get("prediction_id") or ""): item
        for item in evaluation.get("prediction_results") or ()
        if isinstance(item, Mapping)
    }
    publication_by_candidate = {
        str(item.get("candidate_id") or ""): item
        for item in outcome.get("evaluation_runs") or ()
        if isinstance(item, Mapping) and item.get("candidate_id")
    }
    authored_evaluation_configured = bool(
        evaluation_by_prediction
        or (
            evaluation_design
            and any(
                scorer.kind in {"criteria", "llm_judge"}
                for scorer in evaluation_design.scorers
            )
        )
    )
    cells = [
        _outcome_cell(
            item,
            evidence_by_prediction,
            evaluation_by_prediction,
            publication_by_candidate=publication_by_candidate,
            evaluation_design=evaluation_design,
            authored_evaluation_configured=authored_evaluation_configured,
        )
        for item in rows
    ]
    displayed = tuple(cells[:EXPERIMENT_VIEW_CELL_LIMIT])
    labels = _display_labels(draft.get("display_labels"))
    arm_totals = _arm_totals(rows, research_view=research_view, labels=labels)
    measures = _behavioral_measures(rows)
    mechanism_funnel = _mechanism_funnel(rows, research_view)
    limitations = _public_limitations(outcome)
    run_status = str(outcome.get("run_status") or "")
    infrastructure_health = (
        "healthy"
        if run_status in {"passed", "failed"} and bool(outcome.get("eligible"))
        else "failed"
        if run_status in {"cancelled", "interrupted"}
        else "unavailable"
    )
    record_links = _record_evidence_links(record)
    return experiment_view_from_dict(
        ExperimentViewV1(
            # This generic ExperimentRecord projection still uses the V1
            # aggregate/cell contract. V2 evaluation is reserved for the
            # canonical paired-attempt comparison contract above.
            schema_version=1,
            kind="evaluation",
            matrix_size=int(
                outcome.get("expected_predictions")
                or preview.get("estimated_cells")
                or len(rows)
            ),
            preview_digest=str(preview.get("preview_digest") or "") or None,
            approval_state="approved"
            if record.get("approval")
            else "awaiting_approval",
            cell_limit=int(preview.get("estimated_cells") or len(rows)),
            reserved_cost_usd=_record_reserved_cost(record),
            phase="completed",
            completed_cells=int(outcome.get("observed_predictions") or len(rows)),
            observed_cost_usd=float(outcome.get("observed_cost_usd") or 0.0),
            state_counts=_evaluation_state_counts(cells),
            cells=displayed,
            omitted_cells=max(0, len(cells) - len(displayed)),
            infrastructure_health=infrastructure_health,
            arm_totals=arm_totals,
            aligned_comparisons=_aligned_comparisons(outcome),
            behavioral_measures=measures,
            mechanism_funnel=mechanism_funnel,
            outcome_summaries=_outcome_summaries(
                cells,
                infrastructure_health=infrastructure_health,
                evidence_eligible=bool(outcome.get("eligible")),
            ),
            score_summaries=_score_summaries(cells),
            evidence_eligible=bool(outcome.get("eligible")),
            evidence_scope=(
                _evidence_scope(rows, record_links)
                or _optional_evidence_scope(preview.get("evidence_scope"))
            ),
            limitations=limitations,
            evidence_links=record_links,
        ).to_dict()
    )


def _planned_cell(
    raw: Mapping[str, Any],
    varied_names: Sequence[str],
    *,
    arm_factor_levels: Mapping[str, Any],
) -> ExperimentCellViewV1:
    cell_id = _planned_attempt_id(raw)
    variant_id = str(raw.get("variant_id") or "")
    configured_arm = arm_factor_levels.get(variant_id)
    configured_levels = (
        {
            _normalized_dimension_key(str(key)): str(value)
            for key, value in configured_arm.items()
            if isinstance(key, str) and isinstance(value, str)
        }
        if isinstance(configured_arm, Mapping)
        else {}
    )
    factor_levels: dict[str, str] = {}
    for name in varied_names:
        direct_value = _value_for_dimension(name, raw)
        configured_value = configured_levels.get(_normalized_dimension_key(name), "")
        value = direct_value or configured_value
        if not value:
            raise ValueError(
                f"planned cell {cell_id} does not resolve varied factor {name!r}"
            )
        factor_levels[name] = value
    return ExperimentCellViewV1(
        cell_id=cell_id,
        task_label=_reviewed_task_label(raw),
        factor_levels=factor_levels,
        attempt=max(1, int(raw.get("trial_index") or 1)),
        execution_status=(
            "queued" if bool(raw.get("applicable", True)) else "not_applicable"
        ),
        task_outcome=(
            "pending" if bool(raw.get("applicable", True)) else "not_applicable"
        ),
        evaluation_status=(
            "pending" if bool(raw.get("applicable", True)) else "not_applicable"
        ),
        evidence_status=(
            "pending" if bool(raw.get("applicable", True)) else "not_applicable"
        ),
        reason_code=(None if bool(raw.get("applicable", True)) else "not_applicable"),
    )


def _running_cell(
    raw: Mapping[str, Any], plan_cells: Mapping[str, Mapping[str, Any]]
) -> ExperimentCellViewV1:
    status = _execution_status(str(raw.get("status") or "queued"))
    coordinate = str(raw.get("cell_id") or "")
    plan = plan_cells.get(coordinate, {})
    factor_levels = {
        "harness": str(raw.get("harness") or plan.get("harness") or ""),
        "variant": str(raw.get("variant_id") or plan.get("variant_id") or ""),
    }
    factor_levels = {key: value for key, value in factor_levels.items() if value}
    outcome = _benchmark_outcome(raw.get("benchmark_outcome"), status)
    return ExperimentCellViewV1(
        cell_id=_planned_attempt_id({**plan, **raw}),
        task_label=_reviewed_task_label({**plan, **raw}),
        factor_levels=factor_levels,
        attempt=max(1, int(plan.get("trial_index") or 1)),
        execution_status=status,
        task_outcome=outcome,
        evaluation_status=(
            "not_applicable" if status == "not_applicable" else "pending"
        ),
        evidence_status=("not_applicable" if status == "not_applicable" else "pending"),
        reason_code=_safe_reason(status, outcome),
        latency_sec=_optional_float(raw.get("wall_time_sec")),
    )


def _planned_attempt_id(raw: Mapping[str, Any]) -> str:
    supplied = str(raw.get("attempt_id") or "")
    required = {
        "task_id": str(raw.get("task_id") or raw.get("task_name") or ""),
        "arm": str(raw.get("variant_id") or ""),
        "harness": str(raw.get("harness") or ""),
        "attempt": int(raw.get("trial_index") or 0),
        "candidate": str(raw.get("candidate_id") or ""),
        "runtime": str(raw.get("execution_fingerprint") or ""),
    }
    if all(value not in {"", 0} for value in required.values()):
        expected = canonical_attempt_id(**required)
        if supplied and supplied != expected:
            raise ValueError("planned campaign attempt identity is inconsistent")
        return expected
    # V1 historical records retain their persisted coordinate rather than
    # inventing a new attempt identity.
    return supplied or str(raw.get("coordinate_id") or raw.get("cell_id") or "")


def _outcome_cell(
    row: Mapping[str, Any],
    evidence_by_prediction: Mapping[str, Mapping[str, Any]],
    evaluation_by_prediction: Mapping[str, Mapping[str, Any]],
    *,
    publication_by_candidate: Mapping[str, Mapping[str, Any]],
    evaluation_design: ExperimentEvaluationDesignV1 | None,
    authored_evaluation_configured: bool,
) -> ExperimentCellViewV1:
    prediction_id = str(row.get("prediction_id") or "")
    evidence = evidence_by_prediction.get(prediction_id, {})
    evaluation_row = evaluation_by_prediction.get(prediction_id, {})
    execution = _execution_status(str(row.get("status") or "completed"))
    outcome = _row_outcome(row, execution)
    evaluation = _row_evaluation(
        evaluation_row,
        execution,
        configured=authored_evaluation_configured,
    )
    attempt_id = _campaign_attempt_id(row)
    canonical_attempt = bool(row.get("attempt_id")) and isinstance(
        row.get("attempt_identity"), Mapping
    )
    attempt_links = _campaign_attempt_evidence_links(
        {**row, **evidence},
        expected_attempt_id=attempt_id,
    )
    evidence_status: EvidenceStatus = (
        "not_applicable"
        if execution == "not_applicable"
        else "reconciled"
        if (
            len(attempt_links) == 5
            or (
                not canonical_attempt
                and (
                    str(row.get("trace_link_status") or "")
                    in {"ok", "linked", "reconciled"}
                    or bool(evidence)
                )
            )
        )
        else "missing"
    )
    links: list[dict[str, str]] = []
    if prediction_id:
        links.append(
            {
                "system": "fugue",
                "kind": "prediction",
                "ref": prediction_id,
            }
        )
    candidate_id = str(row.get("candidate_id") or "")
    if candidate_id:
        links.append(
            {
                "system": "fugue",
                "kind": "route_runtime_receipt",
                "ref": candidate_id,
            }
        )
    links.extend(attempt_links)
    # Historical V1 outcomes did not have attempt-level Call identities.
    # Keep their immutable aggregate Evaluation/Dataset references readable,
    # without presenting them as a reconciled per-attempt chain.
    if not canonical_attempt:
        publication = publication_by_candidate.get(candidate_id, {})
        evaluation_ref = str(publication.get("evaluation_ref") or "")
        evaluation_url = str(publication.get("url") or "")
        if evaluation_ref:
            links.append(
                {
                    "system": "weave",
                    "kind": "evaluation",
                    "ref": evaluation_ref,
                    **(
                        {"uri": evaluation_url}
                        if evaluation_url.startswith("https://")
                        else {}
                    ),
                }
            )
        dataset_ref = str(publication.get("dataset_ref") or "")
        if dataset_ref:
            links.append(
                {
                    "system": "weave",
                    "kind": "dataset",
                    "ref": dataset_ref,
                }
            )
    run_snapshot = str(row.get("run_snapshot_sha256") or "")
    if run_snapshot:
        links.append(
            {
                "system": "fugue",
                "kind": "run_snapshot",
                "ref": str(row.get("run_id") or prediction_id),
                "digest": run_snapshot,
            }
        )
    source_commit = str(row.get("source_commit") or "")
    if source_commit:
        links.append(
            {
                "system": "git",
                "kind": "source_commit",
                "ref": source_commit,
            }
        )
    run_id = str(row.get("run_id") or "")
    if run_id:
        links.append(
            {
                "system": "fugue",
                "kind": "run",
                "ref": run_id,
            }
        )
    factor_levels = {
        "harness": str(row.get("harness") or ""),
        "variant": str(row.get("variant_id") or ""),
        "context": str(row.get("context_system_id") or ""),
    }
    return ExperimentCellViewV1(
        cell_id=attempt_id or prediction_id,
        task_label=str(
            row.get("task_name") or row.get("comparison_example_id") or "Reviewed task"
        )[:200],
        factor_levels={key: value for key, value in factor_levels.items() if value},
        attempt=max(1, int(row.get("trial_index") or 1)),
        execution_status=execution,
        task_outcome=outcome,
        evaluation_status=evaluation,
        evidence_status=evidence_status,
        reason_code=_safe_reason(execution, outcome, evidence_status),
        cost_usd=_optional_float(
            row.get("cost_usd") or row.get("weave_total_cost_usd")
        ),
        latency_sec=_optional_float(row.get("wall_time_sec")),
        evidence_links=tuple(links),
        measures=_attempt_measures(row),
        scores=_attempt_scores(row, evaluation_row, evaluation_design),
    )


def _attempt_measures(
    row: Mapping[str, Any],
) -> dict[str, str | int | float | bool | None]:
    measures: dict[str, str | int | float | bool | None] = {
        key: row[key]
        for key in _SAFE_BEHAVIORAL_MEASURES
        if row.get(key) is not None and isinstance(row[key], str | int | float | bool)
    }
    for key in (
        "skill_ids_invoked",
        "integration_ids_invoked",
        "mcp_tool_names",
        "mcp_queried_projects",
    ):
        values = row.get(key)
        if not isinstance(values, Sequence) or isinstance(values, str | bytes):
            continue
        normalized = sorted({str(value) for value in values if str(value)})
        if normalized:
            measures[key] = " | ".join(normalized)
    return measures


def _campaign_attempt_id(row: Mapping[str, Any]) -> str:
    supplied = str(row.get("attempt_id") or "")
    identity = row.get("attempt_identity")
    if supplied and isinstance(identity, Mapping):
        expected = canonical_attempt_id(
            task_id=str(identity.get("task_id") or ""),
            arm=str(identity.get("arm") or ""),
            harness=str(identity.get("harness") or ""),
            attempt=int(identity.get("attempt") or 0),
            candidate=str(identity.get("candidate") or ""),
            runtime=str(identity.get("runtime") or ""),
        )
        if supplied != expected:
            raise ValueError("campaign attempt identity does not match its coordinates")
        return supplied
    # Historical V1 campaign outcomes did not persist AttemptIdentityV1.
    # Preserve their existing prediction identity for audit display, but they
    # cannot acquire canonical attempt navigation or reconciled link status.
    return str(row.get("prediction_id") or "")


def _campaign_attempt_evidence_links(
    evidence: Mapping[str, Any],
    *,
    expected_attempt_id: str,
) -> tuple[dict[str, str], ...]:
    supplied_attempt = str(evidence.get("attempt_id") or "")
    if not supplied_attempt or supplied_attempt != expected_attempt_id:
        return ()
    raw_links = evidence.get("links") or evidence.get("evidence_links") or ()
    projected: list[dict[str, str]] = []
    slots: set[str] = set()
    for raw in raw_links:
        if not isinstance(raw, Mapping):
            return ()
        slot = str(raw.get("slot") or "")
        if (
            slot
            not in {
                "evaluation_root",
                "prediction_and_score",
                "prediction",
                "agent_root",
                "dataset",
            }
            or slot in slots
            or raw.get("verification_status") != "verified"
        ):
            return ()
        uri = str(raw.get("uri") or "")
        ref = str(raw.get("ref") or "")
        if not uri.startswith("https://") or not ref.startswith("weave:///"):
            return ()
        projected.append(
            {
                "system": "weave",
                "kind": str(raw.get("kind") or slot),
                "ref": ref,
                "uri": uri,
            }
        )
        slots.add(slot)
    if slots != {
        "evaluation_root",
        "prediction_and_score",
        "prediction",
        "agent_root",
        "dataset",
    }:
        return ()
    # The safe evidence contract orders prediction-and-score first so Study
    # Console's primary action opens the most useful attempt-level chain.
    return tuple(projected)


def _levels_for(
    name: str, draft: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]
) -> tuple[str, ...]:
    values = [_value_for_dimension(name, item) for item in cells]
    if not any(values):
        normalized = name.lower().replace("-", "_").replace(" ", "_")
        words = set(normalized.replace(",", "_").split("_"))
        aliases = {
            "harness": "harnesses",
            "harnesses": "harnesses",
            "variant": "variants",
            "loop": "variants",
            "loop_design": "variants",
            "context": "context_systems",
            "context_system": "context_systems",
            "workload": "workloads",
        }
        source_key = aliases.get(normalized, "")
        if not source_key and (
            "harness" in words or {"codex", "claude"}.issubset(words)
        ):
            source_key = "harnesses"
        elif not source_key and ("loop" in words or "variant" in words):
            source_key = "variants"
        elif not source_key and "task" in words:
            source_key = "workloads"
        source = draft.get(source_key)
        if source:
            values.extend(str(item) for item in source)
        elif "model" in words and draft.get("model"):
            values.append(str(draft["model"]))
        elif words.intersection({"attempt", "attempts"}):
            values.append(str(draft.get("n_attempts") or 1))
        elif words.intersection({"runtime", "prompt", "tools", "tool"}):
            values.append("held fixed")
        elif words.intersection({"environment", "harbor", "network"}):
            values.append(
                "Harbor · no external network"
                if {"without", "external", "network"}.issubset(words)
                else "Harbor"
            )
    return tuple(_ordered_values([value for value in values if value]))


def _factor_level_labels(
    name: str,
    draft: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str],
) -> dict[str, str]:
    return {
        level: labels[level]
        for level in _levels_for(name, draft, cells)
        if level in labels
    }


def _varied_levels_for(
    name: str,
    draft: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    *,
    arm_factor_levels: Mapping[str, Any],
) -> tuple[str, ...]:
    values = list(_levels_for(name, draft, cells))
    normalized_name = _normalized_dimension_key(name)
    for raw_levels in arm_factor_levels.values():
        if not isinstance(raw_levels, Mapping):
            continue
        for raw_name, raw_value in raw_levels.items():
            if (
                isinstance(raw_name, str)
                and isinstance(raw_value, str)
                and _normalized_dimension_key(raw_name) == normalized_name
            ):
                values.append(raw_value)
    return tuple(_ordered_values(values))


def _varied_factor_level_labels(
    name: str,
    draft: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str],
    *,
    arm_factor_levels: Mapping[str, Any],
) -> dict[str, str]:
    return {
        level: labels[level]
        for level in _varied_levels_for(
            name,
            draft,
            cells,
            arm_factor_levels=arm_factor_levels,
        )
        if level in labels
    }


def _normalized_dimension_key(value: str) -> str:
    normalized = value.lower().replace("-", "_").replace(" ", "_")
    if normalized.endswith("_requirement"):
        return normalized.removesuffix("_requirement")
    return normalized


def _display_labels(raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    values = _mapping(raw, "display_labels")
    if len(values) > 128:
        raise ValueError("display_labels may contain at most 128 entries")
    return {
        _text(key, "display label id", 300): _text(value, "display label", 300)
        for key, value in values.items()
    }


def _research_treatment_arms(
    research_view: Mapping[str, Any],
    labels: Mapping[str, str],
) -> tuple[ExperimentTreatmentArmV1, ...]:
    raw_levels = _mapping_or_empty(research_view.get("arm_factor_levels"))
    treatment_summaries = _mapping_or_empty(research_view.get("treatment_summaries"))
    arms: list[ExperimentTreatmentArmV1] = []
    for arm_id, levels in raw_levels.items():
        if not isinstance(levels, Mapping) or not levels:
            continue
        arm = str(arm_id)
        arms.append(
            ExperimentTreatmentArmV1(
                id=arm,
                label=str(labels.get(arm) or humanize_display_id(arm))[:300],
                factor_levels={
                    str(key)[:200]: str(value)[:200]
                    for key, value in levels.items()
                    if isinstance(key, str) and isinstance(value, str)
                },
            )
        )
    declared_ids = {str(item) for item in treatment_summaries if isinstance(item, str)}
    if arms and set(arm.id for arm in arms) != declared_ids:
        raise ValueError(
            "research view arm factor levels must match treatment summaries"
        )
    return tuple(arms)


def _dimension_label(name: str) -> str:
    normalized = name.lower().replace("-", "_").replace(" ", "_")
    words = set(normalized.replace(",", "_").split("_"))
    if "harness" in words or {"codex", "claude"}.issubset(words):
        return "harness"
    if "loop" in words or "variant" in words:
        return "variant"
    if "model" in words:
        return "model and sampling"
    if "task" in words:
        return "taskset"
    if words.intersection({"tools", "tool", "runtime", "prompt"}):
        return "tools, runtime, and prompt"
    if words.intersection({"environment", "harbor", "network"}):
        return "environment"
    if words.intersection({"attempt", "attempts"}):
        return "attempt"
    return name


def _value_for_dimension(name: str, cell: Mapping[str, Any]) -> str:
    normalized = name.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "harness": "harness",
        "harnesses": "harness",
        "variant": "variant_id",
        "loop": "variant_id",
        "loop_design": "variant_id",
        "context": "context_system_id",
        "context_system": "context_system_id",
        "model": "model",
        "workload": "workload_id",
        "task": "task_id",
        "tasks": "task_id",
        "attempt": "trial_index",
        "attempts": "trial_index",
    }
    key = aliases.get(normalized, normalized)
    value = cell.get(key)
    if value is None:
        return ""
    return str(value)


def _arm_totals(
    rows: Sequence[Mapping[str, Any]],
    *,
    research_view: Mapping[str, Any],
    labels: Mapping[str, str],
) -> tuple[dict[str, Any], ...]:
    configured_levels = _mapping_or_empty(research_view.get("arm_factor_levels"))
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        variant = str(row.get("variant_id") or "default")
        harness = str(row.get("harness") or "all")
        grouped[(variant, "all")].append(row)
        if harness != "all":
            grouped[(variant, harness)].append(row)
    result = []
    for (variant, harness), arm_rows in sorted(grouped.items()):
        passed = sum(1 for row in arm_rows if row.get("pass") is True)
        raw_levels = configured_levels.get(variant)
        factor_levels = (
            {
                str(key): str(value)
                for key, value in raw_levels.items()
                if isinstance(key, str) and isinstance(value, str)
            }
            if isinstance(raw_levels, Mapping)
            else {}
        )
        result.append(
            {
                "arm": variant,
                "harness": harness,
                "passed": passed,
                "total": len(arm_rows),
                **({"arm_label": labels[variant]} if variant in labels else {}),
                **({"harness_label": labels[harness]} if harness in labels else {}),
                **({"factor_levels": factor_levels} if factor_levels else {}),
            }
        )
    return tuple(result)


def _behavioral_measures(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in _SAFE_BEHAVIORAL_MEASURES:
        values = [row[key] for row in rows if row.get(key) is not None]
        if not values:
            continue
        numeric = [float(value) for value in values if isinstance(value, int | float)]
        if len(numeric) == len(values):
            result[key] = {
                "observed": len(numeric),
                "mean": sum(numeric) / len(numeric),
            }
        else:
            result[key] = {"observed": len(values)}
    return result


def _mechanism_funnel(
    rows: Sequence[Mapping[str, Any]],
    research_view: Mapping[str, Any],
) -> tuple[ExperimentMechanismStageV1, ...]:
    raw_stages = research_view.get("mechanism_stages")
    if not isinstance(raw_stages, Sequence) or isinstance(raw_stages, str | bytes):
        return ()
    stages: list[ExperimentMechanismStageV1] = []
    for raw in raw_stages:
        if not isinstance(raw, Mapping):
            continue
        stage_id = str(raw.get("id") or "")
        label = str(raw.get("label") or "")
        source_key = str(raw.get("source_key") or "")
        eligibility_key = str(raw.get("eligibility_key") or "")
        if (
            not stage_id
            or not label
            or source_key not in _SAFE_BEHAVIORAL_MEASURES
            or (eligibility_key and eligibility_key not in _SAFE_BEHAVIORAL_MEASURES)
        ):
            continue
        eligible_rows = [
            row
            for row in rows
            if not eligibility_key or row.get(eligibility_key) is True
        ]
        grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in eligible_rows:
            grouped[
                (
                    str(row.get("variant_id") or "default"),
                    str(row.get("harness") or "all"),
                )
            ].append(row)
        stages.append(
            ExperimentMechanismStageV1(
                id=stage_id,
                label=label,
                eligible=len(eligible_rows),
                reached=sum(
                    _measure_reached(row.get(source_key)) for row in eligible_rows
                ),
                by_arm=tuple(
                    ExperimentMechanismArmV1(
                        arm=arm,
                        harness=harness,
                        eligible=len(arm_rows),
                        reached=sum(
                            _measure_reached(row.get(source_key)) for row in arm_rows
                        ),
                    )
                    for (arm, harness), arm_rows in sorted(grouped.items())
                ),
            )
        )
    return tuple(stages)


def _measure_reached(value: Any) -> bool:
    return value is True or (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and float(value) == 1.0
    )


def _attempt_scores(
    row: Mapping[str, Any],
    evaluation_row: Mapping[str, Any],
    design: ExperimentEvaluationDesignV1 | None,
) -> tuple[ExperimentScoreResultV1, ...]:
    if design is None:
        return ()
    criteria = {
        str(item.get("criterion_id") or ""): item
        for item in evaluation_row.get("criteria") or ()
        if isinstance(item, Mapping)
    }
    results: list[ExperimentScoreResultV1] = []
    for scorer in design.scorers:
        if not scorer.dimensions:
            value = evaluation_row.get("criteria_score")
            passed = evaluation_row.get("criteria_pass")
            results.append(
                ExperimentScoreResultV1(
                    id=scorer.id,
                    label=scorer.label,
                    status=(
                        "passed"
                        if passed is True
                        else "failed"
                        if passed is False
                        else "unavailable"
                    ),
                    value=value
                    if isinstance(value, str | int | float | bool)
                    else None,
                    scorer_id=scorer.id,
                )
            )
            continue
        for dimension in scorer.dimensions:
            source_key = dimension.source_key or dimension.id
            criterion = criteria.get(source_key, {})
            value = row.get(source_key)
            if value is None and criterion:
                value = criterion.get("score")
            if value is not None and not isinstance(value, str | int | float | bool):
                value = None
            status: ScoreStatus
            if value is None:
                status = "unavailable"
            elif dimension.target is not None:
                status = "passed" if value == dimension.target else "failed"
            elif criterion.get("passed") is True:
                status = "passed"
            elif criterion.get("passed") is False:
                status = "failed"
            else:
                status = "observed"
            results.append(
                ExperimentScoreResultV1(
                    id=dimension.id,
                    label=dimension.label,
                    status=status,
                    value=value,
                    scorer_id=scorer.id,
                )
            )
    return tuple(results)


def _score_summaries(
    cells: Sequence[ExperimentCellViewV1],
) -> tuple[ExperimentScoreSummaryV1, ...]:
    grouped: dict[str, list[ExperimentScoreResultV1]] = defaultdict(list)
    labels: dict[str, str] = {}
    for cell in cells:
        for score in cell.scores:
            grouped[score.id].append(score)
            labels.setdefault(score.id, score.label)
    summaries: list[ExperimentScoreSummaryV1] = []
    for score_id, values in grouped.items():
        numeric = [
            float(item.value) for item in values if isinstance(item.value, int | float)
        ]
        passed = sum(item.status == "passed" for item in values)
        failed = sum(item.status == "failed" for item in values)
        summaries.append(
            ExperimentScoreSummaryV1(
                id=score_id,
                label=labels[score_id],
                observed=sum(
                    item.status not in {"unavailable", "not_applicable"}
                    for item in values
                ),
                passed=passed if passed or failed else None,
                failed=failed if passed or failed else None,
                unavailable=sum(item.status == "unavailable" for item in values),
                mean=(sum(numeric) / len(numeric) if numeric else None),
            )
        )
    return tuple(summaries)


def _outcome_summaries(
    cells: Sequence[ExperimentCellViewV1],
    *,
    infrastructure_health: str,
    evidence_eligible: bool,
) -> tuple[ExperimentOutcomeSummaryV1, ...]:
    return (
        _cell_outcome_summary(
            cells,
            id="deterministic_task",
            label="Task outcome",
            field_name="task_outcome",
        ),
        _cell_outcome_summary(
            cells,
            id="authored_evaluation",
            label="Authored evaluation",
            field_name="evaluation_status",
        ),
        ExperimentOutcomeSummaryV1(
            id="infrastructure",
            label="Infrastructure",
            status=(
                "passed"
                if infrastructure_health == "healthy"
                else "failed"
                if infrastructure_health == "failed"
                else "unavailable"
            ),
        ),
        ExperimentOutcomeSummaryV1(
            id="evidence",
            label="Evidence",
            status="passed" if evidence_eligible else "failed",
        ),
    )


def _cell_outcome_summary(
    cells: Sequence[ExperimentCellViewV1],
    *,
    id: str,
    label: str,
    field_name: Literal["task_outcome", "evaluation_status"],
) -> ExperimentOutcomeSummaryV1:
    values = [getattr(cell, field_name) for cell in cells]
    scored = [value for value in values if value in {"passed", "failed"}]
    unavailable = sum(value == "unavailable" for value in values)
    not_applicable = sum(value == "not_applicable" for value in values)
    if not scored:
        status: SummaryStatus = (
            "not_applicable"
            if values and not_applicable == len(values)
            else "unavailable"
        )
        return ExperimentOutcomeSummaryV1(
            id=id,
            label=label,
            status=status,
            unavailable=unavailable,
        )
    passed = sum(value == "passed" for value in scored)
    return ExperimentOutcomeSummaryV1(
        id=id,
        label=label,
        status="passed" if passed == len(scored) else "failed",
        passed=passed,
        total=len(scored),
        unavailable=unavailable,
    )


def _aligned_comparisons(outcome: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    values = []
    for item in outcome.get("analysis_results") or ():
        if not isinstance(item, Mapping):
            continue
        analysis_id = str(item.get("analysis_id") or item.get("id") or "")
        digest = str(
            item.get("analysis_digest")
            or item.get("snapshot_digest")
            or item.get("digest")
            or ""
        )
        if analysis_id:
            materialized = item.get("aligned_analysis")
            if isinstance(materialized, Mapping):
                aligned = aligned_analysis_from_dict(materialized)
                digest = aligned.analysis_digest
                for result in aligned.contrast_results:
                    values.append(
                        {
                            "analysis_id": analysis_id[:300],
                            "comparison_id": (
                                f"{result.contrast_id}:"
                                f"{result.treatment_arm}:"
                                f"{result.dimension_id}"
                            )[:300],
                            **(
                                {"estimate": result.estimate}
                                if result.estimate is not None
                                else {}
                            ),
                            "pairs": result.aligned_sets,
                            "digest": digest,
                        }
                    )
                for interaction in aligned.interaction_results:
                    for result in interaction.dimensions:
                        values.append(
                            {
                                "analysis_id": analysis_id[:300],
                                "comparison_id": (
                                    f"{interaction.interaction_id}:"
                                    f"{result.dimension_id}:did"
                                )[:300],
                                **(
                                    {"estimate": (result.difference_in_differences)}
                                    if result.difference_in_differences is not None
                                    else {}
                                ),
                                "pairs": result.aligned_sets,
                                "digest": digest,
                            }
                        )
                if aligned.contrast_results or aligned.interaction_results:
                    continue
            selection = item.get("selection")
            candidates = (
                selection.get("candidates")
                if isinstance(selection, Mapping)
                and isinstance(selection.get("candidates"), Sequence)
                else ()
            )
            comparisons = 0
            for candidate in candidates:
                if not isinstance(candidate, Mapping):
                    continue
                estimate = _optional_float(candidate.get("paired_pass_rate_delta"))
                candidate_id = str(candidate.get("candidate_id") or "")
                if estimate is None or not candidate_id:
                    continue
                values.append(
                    {
                        "analysis_id": analysis_id[:300],
                        "comparison_id": candidate_id[:300],
                        "estimate": estimate,
                        **(
                            {"confidence_low": low}
                            if (low := _optional_float(candidate.get("confidence_low")))
                            is not None
                            else {}
                        ),
                        **(
                            {"confidence_high": high}
                            if (
                                high := _optional_float(
                                    candidate.get("confidence_high")
                                )
                            )
                            is not None
                            else {}
                        ),
                        **(
                            {"pairs": int(candidate["examples"])}
                            if isinstance(candidate.get("examples"), int)
                            and not isinstance(candidate.get("examples"), bool)
                            and int(candidate["examples"]) >= 0
                            else {}
                        ),
                        **({"digest": digest} if digest else {}),
                    }
                )
                comparisons += 1
            if not comparisons:
                values.append(
                    {
                        "analysis_id": analysis_id[:300],
                        **({"digest": digest} if digest else {}),
                    }
                )
    return tuple(values)


def _public_limitations(outcome: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    if outcome.get("limitations"):
        values.append(
            "Additional limitations are recorded in the immutable Fugue outcome."
        )
    if outcome.get("eligibility_failures"):
        values.append(
            "One or more evidence-eligibility requirements were not satisfied."
        )
    unmeasured = int(outcome.get("unmeasured_cost_cells") or 0)
    if unmeasured:
        values.append(f"Observed cost is unavailable for {unmeasured} cells.")
    return tuple(values)


def _evaluation_state_counts(cells: Sequence[ExperimentCellViewV1]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for cell in cells:
        counts[f"execution:{cell.execution_status}"] += 1
        counts[f"task:{cell.task_outcome}"] += 1
        counts[f"evaluation:{cell.evaluation_status}"] += 1
        counts[f"evidence:{cell.evidence_status}"] += 1
    return dict(sorted(counts.items()))


def _row_outcome(row: Mapping[str, Any], execution: ExecutionStatus) -> OutcomeStatus:
    if execution == "not_applicable":
        return "not_applicable"
    if execution in {"failed", "cancelled", "interrupted"}:
        return "unavailable"
    if row.get("pass") is True:
        return "passed"
    if row.get("pass") is False:
        return "failed"
    return "unavailable"


def _row_evaluation(
    row: Mapping[str, Any],
    execution: ExecutionStatus,
    *,
    configured: bool,
) -> OutcomeStatus:
    if execution == "not_applicable":
        return "not_applicable"
    if not configured:
        return "not_applicable"
    for key in ("criteria_pass", "authored_pass", "evaluation_pass"):
        if row.get(key) is True:
            return "passed"
        if row.get(key) is False:
            return "failed"
    return "unavailable"


def _benchmark_outcome(value: Any, execution: ExecutionStatus) -> OutcomeStatus:
    if execution == "not_applicable":
        return "not_applicable"
    normalized = str(value or "").lower()
    if normalized in {"passed", "pass"}:
        return "passed"
    if normalized in {"failed", "fail"}:
        return "failed"
    if execution in {"failed", "cancelled", "interrupted"}:
        return "unavailable"
    return "pending"


def _execution_status(value: str) -> ExecutionStatus:
    normalized = value.lower()
    aliases = {
        "pending": "queued",
        "created": "queued",
        "starting": "preparing",
        "launching": "preparing",
        "passed": "completed",
        "succeeded": "completed",
        "skipped": "not_applicable",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in _EXECUTION_STATES:
        return "failed"
    return normalized  # type: ignore[return-value]


def _phase(record_state: str, run_state: str) -> str:
    if run_state in {"running", "starting", "launching"}:
        return "run"
    if record_state in {"scoring", "analyzing"}:
        return "evaluation"
    return "preparation"


def _safe_reason(
    execution: ExecutionStatus,
    outcome: OutcomeStatus,
    evidence: EvidenceStatus = "pending",
) -> str | None:
    if execution in {"failed", "cancelled", "interrupted"}:
        return f"execution_{execution}"
    if evidence == "missing":
        return "evidence_missing"
    if outcome == "failed":
        return "task_not_passed"
    if outcome == "unavailable":
        return "task_outcome_unavailable"
    if execution == "not_applicable":
        return "not_applicable"
    return None


def _opaque_cell_id(raw: Mapping[str, Any]) -> str:
    existing = str(
        raw.get("attempt_id") or raw.get("coordinate_id") or raw.get("cell_id") or ""
    )
    if existing:
        return existing[:300]
    identity = {
        key: raw.get(key)
        for key in (
            "candidate_id",
            "comparison_example_id",
            "workload_id",
            "task_id",
            "harness",
            "variant_id",
            "arm",
            "context_system_id",
            "trial_index",
            "attempt",
            "candidate_digest",
            "runtime_lock_digest",
            "execution_fingerprint",
        )
        if raw.get(key) is not None
    }
    return f"cell-{stable_digest(identity)[:16]}"


def _reviewed_task_label(raw: Mapping[str, Any]) -> str:
    value = raw.get("task_name") or raw.get("task_label") or raw.get("task_id")
    return str(value or "Reviewed task")[:200]


def _taskset_label(draft: Mapping[str, Any], task_count: Any) -> str:
    workloads = [str(item) for item in draft.get("workloads") or ()]
    if workloads:
        return ", ".join(humanize_display_id(item) for item in workloads)[:300]
    count = int(task_count or 0)
    return f"{count} locked task{'s' if count != 1 else ''}"


def _record_reserved_cost(record: Mapping[str, Any]) -> float | None:
    admission = _mapping_or_empty(record.get("admission"))
    preview = _mapping_or_empty(record.get("preview"))
    value = admission.get("reserved_cost_usd", preview.get("estimated_cost_usd"))
    return _optional_float(value)


def _evidence_scope(
    rows: Sequence[Mapping[str, Any]],
    links: Sequence[Mapping[str, str]],
) -> ExperimentEvidenceScopeV1 | None:
    projects = {
        str(item.get("trace_project") or "")
        for item in rows
        if str(item.get("trace_project") or "").count("/") == 1
    }
    projects.discard("")
    if len(projects) != 1:
        return None
    project_slug = next(iter(projects))
    entity, project = project_slug.split("/", 1)
    evidence_types = tuple(
        sorted(
            {
                str(item.get("kind") or "")
                for item in links
                if item.get("system") in {"wandb", "weave"} and item.get("kind")
            }
            | {
                "agent_conversation"
                for row in rows
                if row.get("weave_prediction_call_id")
                or row.get("weave_conversation_ids")
            }
            | {
                "prediction_and_score"
                for row in rows
                if row.get("eval_predict_and_score_call_id")
            }
            | {
                "evaluation_attempt"
                for row in rows
                if row.get("eval_predict_and_score_call_id")
            }
        )
    )
    return ExperimentEvidenceScopeV1(
        entity=entity,
        project=project,
        evidence_types=evidence_types,
    )


def _record_evidence_links(record: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    links: list[dict[str, str]] = []
    preview = _mapping_or_empty(record.get("preview"))
    draft = _mapping_or_empty(preview.get("draft"))
    recipe = _mapping_or_empty(draft.get("task_recipe_preview"))
    provenance = _mapping_or_empty(recipe.get("provenance"))
    public_source = _mapping_or_empty(record.get("public_source_evidence"))
    project = str(public_source.get("project") or provenance.get("project") or "")
    selected_call_ids = (
        public_source.get("selected_call_ids")
        or provenance.get("selected_call_ids")
        or ()
    )
    if len(project.split("/")) == 2 and all(project.split("/")):
        for call_id in selected_call_ids:
            if isinstance(call_id, str) and call_id:
                links.append(
                    {
                        "system": "weave",
                        "kind": "source_call",
                        "ref": f"{project}/call/{call_id}",
                    }
                )
    manifest_digest = str(provenance.get("reviewed_cohort_manifest_digest") or "")
    if manifest_digest:
        links.append(
            {
                "system": "fugue",
                "kind": "reviewed_cohort_manifest",
                "ref": manifest_digest,
                "digest": manifest_digest,
            }
        )
    run_id = str(record.get("run_id") or "")
    if run_id:
        links.append({"system": "fugue", "kind": "run", "ref": run_id})
    outcome = _mapping_or_empty(record.get("outcome"))
    outcome_digest = str(outcome.get("outcome_digest") or "")
    if outcome_digest:
        links.append(
            {
                "system": "fugue",
                "kind": "outcome",
                "ref": str(outcome.get("outcome_id") or run_id or outcome_digest),
                "digest": outcome_digest,
            }
        )
    for item in outcome.get("evaluation_runs") or ():
        if not isinstance(item, Mapping):
            continue
        evaluation_ref = str(
            item.get("evaluation_ref") or item.get("publication_id") or ""
        )
        evaluation_url = str(item.get("url") or "")
        if evaluation_ref:
            links.append(
                {
                    "system": "weave",
                    "kind": "evaluation",
                    "ref": evaluation_ref,
                    **(
                        {"uri": evaluation_url}
                        if evaluation_url.startswith("https://")
                        else {}
                    ),
                }
            )
        dataset_ref = str(item.get("dataset_ref") or "")
        if dataset_ref:
            links.append(
                {
                    "system": "weave",
                    "kind": "dataset",
                    "ref": dataset_ref,
                }
            )
    evaluation = _mapping_or_empty(record.get("evaluation"))
    evaluation_digest = str(evaluation.get("evaluation_digest") or "")
    if evaluation_digest:
        links.append(
            {
                "system": "fugue",
                "kind": "evaluation",
                "ref": str(
                    evaluation.get("evaluation_id")
                    or evaluation.get("scoring_revision_id")
                    or evaluation_digest
                ),
                "digest": evaluation_digest,
            }
        )
    analysis = _mapping_or_empty(record.get("analysis"))
    analysis_digest = str(analysis.get("analysis_digest") or "")
    if analysis_digest:
        links.append(
            {
                "system": "fugue",
                "kind": "analysis",
                "ref": str(analysis.get("analysis_id") or analysis_digest),
                "digest": analysis_digest,
            }
        )
    return _evidence_links(links)


def _validate_view_shape(
    view: ExperimentViewV1 | ExperimentViewV2 | ExperimentViewV3,
) -> None:
    if view.kind == "design":
        if (
            not view.question
            or not view.hypothesis
            or view.taskset is None
            or view.runtime is None
        ):
            raise ValueError(
                "design view requires question, hypothesis, taskset, and runtime"
            )
        _reject_cross_kind_values(
            view,
            (
                "phase",
                "completed_cells",
                "observed_cost_usd",
                "state_counts",
                "infrastructure_health",
                "arm_totals",
                "aligned_comparisons",
                "behavioral_measures",
                "mechanism_funnel",
                "outcome_summaries",
                "score_summaries",
                "evidence_eligible",
                "limitations",
                "evidence_links",
                "decision",
                "integrity_status",
                "evidence_grade",
                "release_note_coverage",
                "infrastructure_gates",
                "behavioral_summary",
                "paired_cases",
                "backend",
                "candidate_source_revisions",
            ),
        )
    if view.kind == "progress" and not view.phase:
        raise ValueError("progress view requires phase")
    if view.kind == "progress":
        _reject_cross_kind_values(
            view,
            (
                "question",
                "hypothesis",
                "context",
                "observation",
                "rationale",
                "alternative_explanations",
                "success_definition",
                "task_design",
                "prompt_design",
                "evaluation_design",
                "research_label",
                "study_label",
                "source_cohort",
                "fixed_conditions",
                "varied_factors",
                "treatment_arms",
                "measured_outcomes",
                "taskset",
                "harnesses",
                "runtime",
                "infrastructure_health",
                "arm_totals",
                "aligned_comparisons",
                "behavioral_measures",
                "mechanism_funnel",
                "outcome_summaries",
                "score_summaries",
                "evidence_eligible",
                "limitations",
                "evidence_links",
                "decision",
                "integrity_status",
                "evidence_grade",
                "release_target",
                "candidate_sha",
                "release_note_coverage",
                "infrastructure_gates",
                "behavioral_summary",
                "paired_cases",
                "backend",
                "candidate_source_revisions",
            ),
        )
    if view.kind == "evaluation" and view.evidence_eligible is None:
        raise ValueError("evaluation view requires evidence eligibility")
    if view.kind == "evaluation":
        _reject_cross_kind_values(
            view,
            (
                "question",
                "hypothesis",
                "context",
                "observation",
                "rationale",
                "alternative_explanations",
                "success_definition",
                "task_design",
                "prompt_design",
                "evaluation_design",
                "research_label",
                "study_label",
                "source_cohort",
                "fixed_conditions",
                "varied_factors",
                "treatment_arms",
                "measured_outcomes",
                "taskset",
                "harnesses",
                "runtime",
            ),
        )
    if view.kind == "evaluation" and isinstance(
        view, ExperimentViewV2 | ExperimentViewV3
    ):
        behavioral_status = str((view.behavioral_summary or {}).get("status") or "")
        invalid = behavioral_status == "invalid" or view.integrity_status == "invalid"
        if invalid and (
            behavioral_status != "invalid" or view.integrity_status != "invalid"
        ):
            raise ValueError(
                "invalid behavioral evidence requires matching invalid integrity"
            )
        if invalid and (view.evidence_eligible is not False or view.paired_cases):
            raise ValueError(
                "invalid behavioral evidence cannot expose outcomes or attempt navigation"
            )
        if (
            invalid
            and view.behavioral_summary is not None
            and (
                any(
                    int(view.behavioral_summary.get(name) or 0)
                    for name in (
                        "improved_pairs",
                        "regressed_pairs",
                        "mixed_pairs",
                        "unchanged_pairs",
                        "incomplete_pairs",
                        "candidate_critical_failures",
                    )
                )
                or view.behavioral_summary.get("supported_claim") is not None
            )
        ):
            raise ValueError(
                "invalid behavioral evidence cannot retain pair counts or a claim"
            )
    if view.completed_cells is not None and view.completed_cells > view.matrix_size:
        raise ValueError("completed_cells cannot exceed matrix_size")
    if (
        isinstance(view, ExperimentViewV1)
        and view.omitted_cells
        and len(view.cells) + view.omitted_cells > view.matrix_size
    ):
        raise ValueError("displayed and omitted cells cannot exceed matrix_size")


def _reject_cross_kind_values(
    view: ExperimentViewV1 | ExperimentViewV2 | ExperimentViewV3,
    fields: Sequence[str],
) -> None:
    for name in fields:
        value = getattr(view, name, None)
        if value is not None and value not in ((), {}):
            raise ValueError(f"{view.kind} view cannot contain {name}")


def _optional_evidence_scope(raw: Any) -> ExperimentEvidenceScopeV1 | None:
    if raw is None:
        return None
    value = _mapping(raw, "evidence_scope")
    _reject_unknown(
        value,
        {"entity", "project", "evidence_types"},
        "evidence_scope",
    )
    return ExperimentEvidenceScopeV1(
        entity=_text(value.get("entity"), "evidence_scope.entity", 200),
        project=_text(value.get("project"), "evidence_scope.project", 300),
        evidence_types=tuple(
            _text(item, "evidence_scope.evidence_type", 120)
            for item in _sequence(
                value.get("evidence_types"), "evidence_scope.evidence_types"
            )
        ),
    )


def _optional_task_design(raw: Any) -> ExperimentTaskDesignV1 | None:
    if raw is None:
        return None
    value = _mapping(raw, "task_design")
    _reject_unknown(
        value,
        {
            "title",
            "summary",
            "interaction_mode",
            "tools",
            "resources",
            "evidence_links",
        },
        "task_design",
    )
    return ExperimentTaskDesignV1(
        title=_text(value.get("title"), "task_design.title", 300),
        summary=_text(value.get("summary"), "task_design.summary", 4000),
        interaction_mode=_optional_text(
            value.get("interaction_mode"), "task_design.interaction_mode", 200
        ),
        tools=tuple(
            _text(item, "task_design.tool", 200)
            for item in _sequence(value.get("tools"), "task_design.tools")
        ),
        resources=tuple(
            _text(item, "task_design.resource", 300)
            for item in _sequence(value.get("resources"), "task_design.resources")
        ),
        evidence_links=_evidence_links(value.get("evidence_links")),
    )


def _optional_prompt_design(raw: Any) -> ExperimentPromptDesignV1 | None:
    if raw is None:
        return None
    value = _mapping(raw, "prompt_design")
    _reject_unknown(
        value,
        {"base_instruction_summary", "treatment_summaries", "evidence_links"},
        "prompt_design",
    )
    treatments = {
        _text(key, "prompt_design treatment id", 200): _text(
            item, "prompt_design treatment summary", 2000
        )
        for key, item in _mapping_or_empty(value.get("treatment_summaries")).items()
    }
    return ExperimentPromptDesignV1(
        base_instruction_summary=_text(
            value.get("base_instruction_summary"),
            "prompt_design.base_instruction_summary",
            4000,
        ),
        treatment_summaries=treatments,
        evidence_links=_evidence_links(value.get("evidence_links")),
    )


def _score_definition(raw: Any) -> ExperimentScoreDefinitionV1:
    value = _mapping(raw, "score definition")
    _reject_unknown(
        value,
        {"id", "label", "description", "source_key", "target", "primary"},
        "score definition",
    )
    target = value.get("target")
    if target is not None and not isinstance(target, str | int | float | bool):
        raise ValueError("score definition target must be scalar")
    return ExperimentScoreDefinitionV1(
        id=_text(value.get("id"), "score definition id", 200),
        label=_text(value.get("label"), "score definition label", 300),
        description=_optional_text(
            value.get("description"), "score definition description", 1000
        ),
        source_key=_optional_text(
            value.get("source_key"), "score definition source key", 300
        ),
        target=target,
        primary=_optional_bool(value.get("primary"), "score definition primary")
        or False,
    )


def _scorer_design(raw: Any) -> ExperimentScorerDesignV1:
    value = _mapping(raw, "scorer design")
    _reject_unknown(
        value,
        {
            "id",
            "label",
            "kind",
            "description",
            "required",
            "threshold",
            "aggregation",
            "evidence_inputs",
            "revision",
            "model",
            "rubric_summary",
            "blind_fields",
            "dimensions",
            "evidence_links",
        },
        "scorer design",
    )
    kind = _text(value.get("kind"), "scorer design kind", 80)
    if kind not in {"benchmark", "deterministic", "criteria", "llm_judge"}:
        raise ValueError("unknown scorer design kind")
    threshold = _optional_float(value.get("threshold"))
    if threshold is not None and (
        not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError("scorer design threshold must be in [0, 1]")
    return ExperimentScorerDesignV1(
        id=_text(value.get("id"), "scorer design id", 200),
        label=_text(value.get("label"), "scorer design label", 300),
        kind=kind,  # type: ignore[arg-type]
        description=_text(value.get("description"), "scorer design description", 2000),
        required=_optional_bool(value.get("required"), "scorer design required")
        is not False,
        threshold=threshold,
        aggregation=_optional_text(
            value.get("aggregation"), "scorer design aggregation", 1000
        ),
        evidence_inputs=tuple(
            _text(item, "scorer evidence input", 300)
            for item in _sequence(
                value.get("evidence_inputs"), "scorer design evidence inputs"
            )
        ),
        revision=_optional_text(value.get("revision"), "scorer revision", 300),
        model=_optional_text(value.get("model"), "scorer model", 300),
        rubric_summary=_optional_text(
            value.get("rubric_summary"), "scorer rubric summary", 2000
        ),
        blind_fields=tuple(
            _text(item, "scorer blind field", 200)
            for item in _sequence(value.get("blind_fields"), "scorer blind fields")
        ),
        dimensions=tuple(
            _score_definition(item)
            for item in _sequence(value.get("dimensions"), "scorer dimensions")
        ),
        evidence_links=_evidence_links(value.get("evidence_links")),
    )


def _optional_evaluation_design(
    raw: Any,
) -> ExperimentEvaluationDesignV1 | None:
    if raw is None:
        return None
    value = _mapping(raw, "evaluation_design")
    _reject_unknown(
        value,
        {"pass_rule", "scorers", "llm_judge_used"},
        "evaluation_design",
    )
    scorers = tuple(
        _scorer_design(item)
        for item in _sequence(value.get("scorers"), "evaluation_design.scorers")
    )
    if not scorers:
        raise ValueError("evaluation_design requires at least one scorer")
    judge_used = (
        _optional_bool(value.get("llm_judge_used"), "evaluation_design.llm_judge_used")
        or False
    )
    if judge_used != any(item.kind == "llm_judge" for item in scorers):
        raise ValueError("evaluation_design judge usage disagrees with its scorers")
    return ExperimentEvaluationDesignV1(
        pass_rule=_text(value.get("pass_rule"), "evaluation_design.pass_rule", 4000),
        scorers=scorers,
        llm_judge_used=judge_used,
    )


def _factor(raw: Any, field_name: str) -> ExperimentFactorV1:
    value = _mapping(raw, field_name)
    _reject_unknown(value, {"name", "levels", "label", "level_labels"}, field_name)
    levels = tuple(
        _text(item, f"{field_name}.level", 300)
        for item in _sequence(value.get("levels"), f"{field_name}.levels")
    )
    level_labels = _display_labels(value.get("level_labels"))
    if set(level_labels) - set(levels):
        raise ValueError(f"{field_name}.level_labels names an unknown level")
    return ExperimentFactorV1(
        name=_text(value.get("name"), f"{field_name}.name", 200),
        levels=levels,
        label=_optional_text(value.get("label"), f"{field_name}.label", 300),
        level_labels=level_labels,
    )


def _treatment_arm(raw: Any) -> ExperimentTreatmentArmV1:
    value = _mapping(raw, "treatment_arm")
    _reject_unknown(value, {"id", "label", "factor_levels"}, "treatment_arm")
    factor_levels = {
        _text(key, "treatment_arm factor id", 200): _text(
            item, "treatment_arm factor level", 200
        )
        for key, item in _mapping(
            value.get("factor_levels"), "treatment_arm.factor_levels"
        ).items()
    }
    if not factor_levels:
        raise ValueError("treatment_arm requires factor levels")
    return ExperimentTreatmentArmV1(
        id=_text(value.get("id"), "treatment_arm.id", 200),
        label=_text(value.get("label"), "treatment_arm.label", 300),
        factor_levels=factor_levels,
    )


def _outcome_summary(raw: Any) -> ExperimentOutcomeSummaryV1:
    value = _mapping(raw, "outcome_summary")
    _reject_unknown(
        value,
        {"id", "label", "status", "passed", "total", "unavailable"},
        "outcome_summary",
    )
    status = _text(value.get("status"), "outcome_summary.status", 80)
    if status not in {"passed", "failed", "unavailable", "not_applicable"}:
        raise ValueError("unknown outcome summary status")
    passed = _optional_non_negative_int(value.get("passed"), "outcome_summary.passed")
    total = _optional_non_negative_int(value.get("total"), "outcome_summary.total")
    unavailable = _non_negative_int(
        value.get("unavailable", 0), "outcome_summary.unavailable"
    )
    if passed is not None and total is None:
        raise ValueError("outcome summary passed count requires total")
    if passed is not None and total is not None and passed > total:
        raise ValueError("outcome summary passed count cannot exceed total")
    return ExperimentOutcomeSummaryV1(
        id=_text(value.get("id"), "outcome_summary.id", 200),
        label=_text(value.get("label"), "outcome_summary.label", 300),
        status=status,  # type: ignore[arg-type]
        passed=passed,
        total=total,
        unavailable=unavailable,
    )


def _score_result(raw: Any) -> ExperimentScoreResultV1:
    value = _mapping(raw, "score result")
    _reject_unknown(
        value,
        {"id", "label", "status", "value", "scorer_id"},
        "score result",
    )
    status = _text(value.get("status"), "score result status", 80)
    if status not in {
        "passed",
        "failed",
        "observed",
        "unavailable",
        "not_applicable",
    }:
        raise ValueError("unknown score result status")
    score_value = value.get("value")
    if score_value is not None and not isinstance(
        score_value, str | int | float | bool
    ):
        raise ValueError("score result value must be scalar")
    return ExperimentScoreResultV1(
        id=_text(value.get("id"), "score result id", 200),
        label=_text(value.get("label"), "score result label", 300),
        status=status,  # type: ignore[arg-type]
        value=score_value,
        scorer_id=_optional_text(value.get("scorer_id"), "score result scorer", 200),
    )


def _score_summary(raw: Any) -> ExperimentScoreSummaryV1:
    value = _mapping(raw, "score summary")
    _reject_unknown(
        value,
        {"id", "label", "observed", "passed", "failed", "unavailable", "mean"},
        "score summary",
    )
    observed = _non_negative_int(value.get("observed", 0), "score summary observed")
    passed = (
        None
        if value.get("passed") is None
        else _non_negative_int(value["passed"], "score summary passed")
    )
    failed = (
        None
        if value.get("failed") is None
        else _non_negative_int(value["failed"], "score summary failed")
    )
    unavailable = _non_negative_int(
        value.get("unavailable", 0), "score summary unavailable"
    )
    if (passed or 0) + (failed or 0) > observed:
        raise ValueError("score summary statuses exceed observed values")
    return ExperimentScoreSummaryV1(
        id=_text(value.get("id"), "score summary id", 200),
        label=_text(value.get("label"), "score summary label", 300),
        observed=observed,
        passed=passed,
        failed=failed,
        unavailable=unavailable,
        mean=_optional_float(value.get("mean")),
    )


def _descriptor(raw: Any, field_name: str) -> ExperimentDescriptorV1:
    value = _mapping(raw, field_name)
    _reject_unknown(value, {"id", "label", "digest", "details"}, field_name)
    details_raw = _mapping_or_empty(value.get("details"))
    details: dict[str, str | int | bool] = {}
    for key, item in details_raw.items():
        if not isinstance(item, str | int | bool):
            raise ValueError(f"{field_name}.details values must be scalar")
        details[_text(key, f"{field_name}.details key", 100)] = item
    return ExperimentDescriptorV1(
        id=_text(value.get("id"), f"{field_name}.id", 1000),
        label=_text(value.get("label"), f"{field_name}.label", 300),
        digest=_optional_digest(value.get("digest"), f"{field_name}.digest"),
        details=details,
    )


def _optional_descriptor(raw: Any, field_name: str) -> ExperimentDescriptorV1 | None:
    return None if raw is None else _descriptor(raw, field_name)


def _cell(raw: Any) -> ExperimentCellViewV1:
    value = _mapping(raw, "cell")
    _reject_unknown(
        value,
        {
            "cell_id",
            "task_label",
            "factor_levels",
            "attempt",
            "execution_status",
            "task_outcome",
            "evaluation_status",
            "evidence_status",
            "reason_code",
            "cost_usd",
            "latency_sec",
            "evidence_links",
            "measures",
            "scores",
        },
        "cell",
    )
    execution = _text(value.get("execution_status"), "execution_status", 80)
    outcome = _text(value.get("task_outcome"), "task_outcome", 80)
    evaluation = _text(value.get("evaluation_status"), "evaluation_status", 80)
    evidence = _text(value.get("evidence_status"), "evidence_status", 80)
    if execution not in _EXECUTION_STATES:
        raise ValueError("unknown execution status")
    if outcome not in _OUTCOME_STATES or evaluation not in _OUTCOME_STATES:
        raise ValueError("unknown outcome status")
    if evidence not in _EVIDENCE_STATES:
        raise ValueError("unknown evidence status")
    factor_levels = {
        _text(key, "factor name", 200): _text(item, "factor level", 300)
        for key, item in _mapping(value.get("factor_levels"), "factor_levels").items()
    }
    measures: dict[str, str | int | float | bool | None] = {}
    for key, item in _mapping_or_empty(value.get("measures")).items():
        if key not in _SAFE_BEHAVIORAL_MEASURES:
            raise ValueError(f"cell contains an unsupported measure: {key}")
        if item is not None and not isinstance(item, str | int | float | bool):
            raise ValueError("cell measure values must be scalar")
        measures[key] = item
    links = _evidence_links(value.get("evidence_links"))
    return ExperimentCellViewV1(
        cell_id=_text(value.get("cell_id"), "cell_id", 300),
        task_label=_text(value.get("task_label"), "task_label", 200),
        factor_levels=factor_levels,
        attempt=_positive_int(value.get("attempt"), "attempt"),
        execution_status=execution,  # type: ignore[arg-type]
        task_outcome=outcome,  # type: ignore[arg-type]
        evaluation_status=evaluation,  # type: ignore[arg-type]
        evidence_status=evidence,  # type: ignore[arg-type]
        reason_code=_optional_text(value.get("reason_code"), "reason_code", 100),
        cost_usd=_optional_cost(value.get("cost_usd"), "cost_usd"),
        latency_sec=_optional_cost(value.get("latency_sec"), "latency_sec"),
        evidence_links=links,
        measures=measures,
        scores=tuple(
            _score_result(item)
            for item in _sequence(value.get("scores"), "cell scores")
        ),
    )


def _evidence_links(raw: Any) -> tuple[dict[str, str], ...]:
    links: list[dict[str, str]] = []
    for item in _sequence(raw, "evidence_links"):
        link = _mapping(item, "evidence_link")
        _reject_unknown(
            link, {"system", "kind", "ref", "uri", "digest"}, "evidence_link"
        )
        system = _text(link.get("system"), "evidence_link.system", 100)
        kind = _text(link.get("kind"), "evidence_link.kind", 100)
        ref = _text(link.get("ref"), "evidence_link.ref", 1000)
        uri = _optional_text(link.get("uri"), "evidence_link.uri", 2000)
        if uri:
            if not uri.startswith("https://"):
                raise ValueError("evidence link URIs must use https")
        if kind == "comparison_rows":
            if system == "wandb" and uri is None:
                # Historical local projections mislabeled a Fugue run ID as a
                # W&B Run. Keep those records readable without manufacturing a
                # hosted link.
                system = "fugue"
            elif system == "wandb":
                run_id = _wandb_run_id_from_url(uri or "")
                if run_id is None:
                    raise ValueError(
                        "hosted comparison rows require a canonical W&B Run URL"
                    )
                if ref != run_id:
                    raise ValueError(
                        "hosted comparison rows Run ID must match its URL"
                    )
            elif system in {"fugue", "local_artifact"}:
                if uri is not None:
                    raise ValueError(
                        "local comparison rows cannot declare a hosted URI"
                    )
                system = "fugue"
            elif system != "wandb":
                raise ValueError(
                    "comparison rows evidence must use fugue or wandb"
                )
        projected = {"system": system, "kind": kind, "ref": ref}
        if uri:
            projected["uri"] = uri
        digest = _optional_digest(link.get("digest"), "evidence_link.digest")
        if digest:
            projected["digest"] = digest
        links.append(projected)
    return tuple(links)


def _arm_total(raw: Any) -> dict[str, Any]:
    value = _mapping(raw, "arm_total")
    _reject_unknown(
        value,
        {
            "arm",
            "arm_label",
            "harness",
            "harness_label",
            "factor_levels",
            "passed",
            "total",
        },
        "arm_total",
    )
    result = {
        "arm": _text(value.get("arm"), "arm_total.arm", 300),
        "harness": _text(value.get("harness"), "arm_total.harness", 300),
        "passed": _non_negative_int(value.get("passed"), "arm_total.passed"),
        "total": _non_negative_int(value.get("total"), "arm_total.total"),
    }
    arm_label = _optional_text(value.get("arm_label"), "arm_total.arm_label", 300)
    if arm_label:
        result["arm_label"] = arm_label
    harness_label = _optional_text(
        value.get("harness_label"), "arm_total.harness_label", 300
    )
    if harness_label:
        result["harness_label"] = harness_label
    factor_levels = _mapping_or_empty(value.get("factor_levels"))
    if factor_levels:
        result["factor_levels"] = {
            _text(key, "arm_total factor id", 200): _text(
                item, "arm_total factor level", 200
            )
            for key, item in factor_levels.items()
        }
    return result


def _comparison(raw: Any) -> dict[str, Any]:
    value = _mapping(raw, "aligned_comparison")
    _reject_unknown(
        value,
        {
            "analysis_id",
            "comparison_id",
            "estimate",
            "confidence_low",
            "confidence_high",
            "pairs",
            "digest",
        },
        "aligned_comparison",
    )
    result = {
        "analysis_id": _text(
            value.get("analysis_id"), "aligned_comparison.analysis_id", 300
        )
    }
    comparison_id = _optional_text(
        value.get("comparison_id"), "aligned_comparison.comparison_id", 300
    )
    if comparison_id:
        result["comparison_id"] = comparison_id
    for field_name in ("estimate", "confidence_low", "confidence_high"):
        if value.get(field_name) is not None:
            number = float(value[field_name])
            if not math.isfinite(number):
                raise ValueError(f"aligned_comparison.{field_name} must be finite")
            result[field_name] = number
    if value.get("pairs") is not None:
        result["pairs"] = _non_negative_int(
            value.get("pairs"), "aligned_comparison.pairs"
        )
    digest = _optional_digest(value.get("digest"), "aligned_comparison.digest")
    if digest:
        result["digest"] = digest
    return result


def _mechanism_stage(raw: Any) -> ExperimentMechanismStageV1:
    value = _mapping(raw, "mechanism stage")
    _reject_unknown(
        value,
        {"id", "label", "eligible", "reached", "by_arm"},
        "mechanism stage",
    )
    eligible = _non_negative_int(value.get("eligible"), "mechanism stage eligible")
    reached = _non_negative_int(value.get("reached"), "mechanism stage reached")
    if reached > eligible:
        raise ValueError("mechanism stage reached cannot exceed eligible")
    by_arm: list[ExperimentMechanismArmV1] = []
    for raw_arm in _sequence(value.get("by_arm"), "mechanism stage by_arm"):
        arm = _mapping(raw_arm, "mechanism stage arm")
        _reject_unknown(
            arm,
            {"arm", "harness", "eligible", "reached"},
            "mechanism stage arm",
        )
        arm_eligible = _non_negative_int(
            arm.get("eligible"), "mechanism stage arm eligible"
        )
        arm_reached = _non_negative_int(
            arm.get("reached"), "mechanism stage arm reached"
        )
        if arm_reached > arm_eligible:
            raise ValueError("mechanism stage arm reached cannot exceed eligible")
        by_arm.append(
            ExperimentMechanismArmV1(
                arm=_text(arm.get("arm"), "mechanism stage arm id", 300),
                harness=_text(arm.get("harness"), "mechanism stage arm harness", 300),
                eligible=arm_eligible,
                reached=arm_reached,
            )
        )
    return ExperimentMechanismStageV1(
        id=_text(value.get("id"), "mechanism stage id", 200),
        label=_text(value.get("label"), "mechanism stage label", 300),
        eligible=eligible,
        reached=reached,
        by_arm=tuple(by_arm),
    )


def _measure_mapping(raw: Any, field_name: str) -> dict[str, Any]:
    value = _mapping_or_empty(raw)
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key not in _SAFE_BEHAVIORAL_MEASURES:
            raise ValueError(f"{field_name} contains an unsupported measure: {key}")
        measure = _mapping(item, f"{field_name}.{key}")
        _reject_unknown(measure, {"observed", "mean"}, f"{field_name}.{key}")
        observed = _non_negative_int(
            measure.get("observed"), f"{field_name}.{key}.observed"
        )
        result[key] = {"observed": observed}
        if measure.get("mean") is not None:
            result[key]["mean"] = float(measure["mean"])
    return result


def _count_mapping(raw: Any, field_name: str) -> dict[str, int]:
    return {
        _text(key, f"{field_name} key", 100): _non_negative_int(
            item, f"{field_name}.{key}"
        )
        for key, item in _mapping_or_empty(raw).items()
    }


def _mapping(raw: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return raw


def _mapping_or_empty(raw: Any) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("expected an object")
    return raw


def _optional_mapping(raw: Any, field_name: str) -> dict[str, Any] | None:
    if raw is None:
        return None
    return dict(_mapping(raw, field_name))


def _sequence(raw: Any, field_name: str) -> Sequence[Any]:
    if raw is None:
        return ()
    if not isinstance(raw, list | tuple):
        raise ValueError(f"{field_name} must be an array")
    return raw


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], field_name: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(
            f"{field_name} has unknown fields: "
            + ", ".join(sorted(str(item) for item in unknown))
        )


def _text(raw: Any, field_name: str, maximum: int) -> str:
    if not isinstance(raw, str) or not raw.strip() or len(raw) > maximum:
        raise ValueError(f"{field_name} must contain 1 to {maximum} characters")
    return raw.strip()


def _optional_text(raw: Any, field_name: str, maximum: int) -> str | None:
    if raw is None:
        return None
    return _text(raw, field_name, maximum)


def _positive_int(raw: Any, field_name: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return raw


def _non_negative_int(raw: Any, field_name: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return raw


def _optional_non_negative_int(raw: Any, field_name: str) -> int | None:
    return None if raw is None else _non_negative_int(raw, field_name)


def _optional_bool(raw: Any, field_name: str) -> bool | None:
    if raw is None:
        return None
    if not isinstance(raw, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return raw


def _required_bool(raw: Any, field_name: str) -> bool:
    if not isinstance(raw, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return raw


def _optional_cost(raw: Any, field_name: str) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int | float) or raw < 0:
        raise ValueError(f"{field_name} must be a non-negative number")
    return float(raw)


def _optional_float(raw: Any) -> float | None:
    if raw is None or isinstance(raw, bool) or not isinstance(raw, int | float):
        return None
    return float(raw)


def _optional_digest(raw: Any, field_name: str) -> str | None:
    if raw is None or raw == "":
        return None
    return _text(raw, field_name, 1000)


def _required_digest(raw: Any, field_name: str) -> str:
    value = _text(raw, field_name, 64)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _ordered_values(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _drop_empty(
    value: Mapping[str, Any], *, preserve_false: bool = False
) -> dict[str, Any]:
    empty = (None, "", (), [], {})
    return {
        key: item
        for key, item in value.items()
        if item not in empty and (preserve_false or item is not False)
    }
