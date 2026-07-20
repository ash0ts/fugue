from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from fugue.bench.candidates import stable_digest
from fugue.bench.task_authoring import TaskAuthoringPolicyV1

CAMPAIGN_SCHEMA_VERSION = 1
_EVIDENCE_REQUIREMENTS = {
    "terminal_rows",
    "agent_identity",
    "runtime_lock",
    "route_receipt",
    "cost_accounting",
}
_ERROR_CATEGORIES = {
    "validation",
    "policy",
    "preparation",
    "admission",
    "execution",
    "reconciliation",
    "evidence",
    "scope",
}

EvidenceScope = Literal["summary", "rows", "traces"]


class CampaignError(ValueError):
    """A stable machine-facing campaign failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        category: str,
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        if category not in _ERROR_CATEGORIES:
            raise ValueError(f"unknown campaign error category: {category}")
        self.code = code
        self.category = category
        self.retryable = retryable
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        unsigned = {
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "code": self.code,
            "category": self.category,
            "safe_to_repeat": self.retryable,
            "message": str(self),
            "details": _json_value(self.details),
            "error_digest": "",
        }
        return {
            **unsigned,
            "error_digest": _artifact_digest(unsigned, "error_digest"),
        }


@dataclass(frozen=True)
class CampaignLimitsV1:
    total_cost_usd: float
    initial_cell_reserve_usd: float
    safety_margin: float
    max_cells_per_proposal: int
    max_total_cells: int
    max_attempts_per_cell: int
    max_concurrent: int
    max_active_runs: int = 1


@dataclass(frozen=True)
class CampaignStagePolicyV1:
    id: str
    predecessors: tuple[str, ...]
    max_proposals: int
    max_cells: int
    require_eligible_parent: bool
    required_evidence: tuple[str, ...] = tuple(sorted(_EVIDENCE_REQUIREMENTS))


@dataclass(frozen=True)
class ResearchCampaignSpecV1:
    schema_version: int
    id: str
    revision: str
    title: str
    objective: str
    allowed_experiments: tuple[str, ...]
    allowed_models: tuple[str, ...]
    allowed_harnesses: tuple[str, ...]
    allowed_workloads: tuple[str, ...]
    allowed_variants: tuple[str, ...]
    allowed_context_systems: tuple[str, ...]
    allowed_analyses: tuple[str, ...]
    allowed_trace_content: tuple[str, ...]
    stages: tuple[CampaignStagePolicyV1, ...]
    limits: CampaignLimitsV1
    task_authoring: TaskAuthoringPolicyV1 | None = None
    evidence_scope: EvidenceScope = "rows"
    require_clean_source: bool = True
    campaign_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(
            {
                "schema_version": self.schema_version,
                "id": self.id,
                "revision": self.revision,
                "title": self.title,
                "objective": self.objective,
                "allowed": {
                    "experiments": self.allowed_experiments,
                    "models": self.allowed_models,
                    "harnesses": self.allowed_harnesses,
                    "workloads": self.allowed_workloads,
                    "variants": self.allowed_variants,
                    "context_systems": self.allowed_context_systems,
                    "analyses": self.allowed_analyses,
                    "trace_content": self.allowed_trace_content,
                },
                "stages": [asdict(stage) for stage in self.stages],
                "limits": asdict(self.limits),
                "task_authoring": (
                    self.task_authoring.to_dict() if self.task_authoring else None
                ),
                "evidence_scope": self.evidence_scope,
                "require_clean_source": self.require_clean_source,
                "campaign_digest": self.campaign_digest,
            }
        )


@dataclass(frozen=True)
class CampaignCatalogSnapshotV1:
    schema_version: int
    campaign_id: str
    policy_digest: str
    source_provenance: dict[str, Any]
    experiments: tuple[dict[str, Any], ...]
    models: tuple[dict[str, Any], ...]
    harnesses: tuple[str, ...]
    context_systems: tuple[dict[str, Any], ...]
    analyses: tuple[dict[str, Any], ...]
    task_authoring: dict[str, Any] | None
    component_digests: dict[str, str]
    catalog_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class ExperimentProposalV1:
    schema_version: int
    proposal_id: str
    campaign_id: str
    catalog_digest: str
    stage_id: str
    research_question: str
    hypothesis: str
    fixed_dimensions: tuple[str, ...]
    varied_dimensions: tuple[str, ...]
    measured_dimensions: tuple[str, ...]
    experiment_id: str
    preset_id: str | None
    workloads: tuple[str, ...]
    harnesses: tuple[str, ...]
    context_systems: tuple[str, ...]
    variants: tuple[str, ...]
    model: str
    n_attempts: int
    n_tasks: int | None
    n_concurrent: int
    trace_content: str
    task_suite_digest: str | None = None
    analysis_ids: tuple[str, ...] = ()
    parent_outcome_id: str | None = None
    decision_rationale: str = ""
    proposal_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class PlanReceiptV1:
    schema_version: int
    campaign_id: str
    proposal_id: str
    proposal_digest: str
    policy_digest: str
    catalog_digest: str
    source_provenance: dict[str, Any]
    proposal: dict[str, Any]
    request: dict[str, Any]
    cells: tuple[dict[str, Any], ...]
    cell_count: int
    applicable_cells: int
    expected_predictions: int
    max_concurrent: int
    component_digests: dict[str, str]
    qualification_requirements: tuple[str, ...]
    plan_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class PreparedPlanV1:
    schema_version: int
    campaign_id: str
    proposal_id: str
    plan_digest: str
    policy_digest: str
    source_provenance: dict[str, Any]
    plan: dict[str, Any]
    preparation: dict[str, Any]
    preflight: tuple[dict[str, Any], ...]
    component_digests: dict[str, str]
    route_locks: tuple[dict[str, Any], ...]
    integration_locks: dict[str, str]
    prepared_at: str
    prepared_plan_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class AdmissionReceiptV1:
    schema_version: int
    admission_id: str
    campaign_id: str
    proposal_id: str
    stage_id: str
    prepared_plan_digest: str
    policy_digest: str
    operation_id: str
    parent_outcome_id: str | None
    cell_count: int
    reserved_cell_cost_usd: float
    reserved_cost_usd: float
    prepared_plan: dict[str, Any]
    admitted_at: str
    admission_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class OutcomePacketV1:
    schema_version: int
    outcome_id: str
    campaign_id: str
    proposal_id: str
    stage_id: str
    admission_id: str
    run_id: str
    run_status: str
    expected_predictions: int
    observed_predictions: int
    passed: int
    failed: int
    not_applicable: int
    eligible: bool
    eligibility_failures: tuple[str, ...]
    limitations: tuple[str, ...]
    observed_cost_usd: float
    accounted_cost_usd: float
    measured_cost_cells: int
    unmeasured_cost_cells: int
    maximum_measured_cell_cost_usd: float | None
    input_lock_sha256: str | None
    run_snapshot_sha256: str | None
    export_sha256: str
    export_path: str
    row_refs: tuple[dict[str, Any], ...]
    evidence_refs: tuple[dict[str, Any], ...]
    analysis_results: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]
    finalized_at: str
    outcome_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class CampaignEventV1:
    schema_version: int
    sequence_number: int
    event_id: str
    campaign_id: str
    event: str
    recorded_at: str
    operation_id: str | None = None
    proposal_id: str | None = None
    admission_id: str | None = None
    run_id: str | None = None
    artifact_type: str | None = None
    artifact_digest: str | None = None
    error: dict[str, Any] | None = None
    previous_event_digest: str | None = None
    event_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class CampaignStatusV1:
    schema_version: int
    campaign_id: str
    subject_id: str
    state: str
    policy_digest: str
    active_runs: tuple[str, ...]
    runs: tuple[dict[str, Any], ...]
    admissions: int
    outcomes: int
    admitted_cells: int
    total_cost_usd: float
    accounted_cost_usd: float
    reserved_cost_usd: float
    remaining_cost_usd: float
    next_actions: tuple[str, ...]
    blockers: tuple[str, ...]
    status_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


def _artifact_digest(value: Mapping[str, Any], digest_field: str) -> str:
    unsigned = dict(_json_value(value))
    unsigned[digest_field] = ""
    return stable_digest(unsigned)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value
