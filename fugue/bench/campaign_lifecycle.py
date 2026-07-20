from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from filelock import FileLock

from fugue.bench.ai import get_analysis, list_analyses
from fugue.bench.campaign_accounting import (
    account_prediction_costs,
    reserve_campaign_cost,
)
from fugue.bench.campaign_contracts import (
    _EVIDENCE_REQUIREMENTS,
    CAMPAIGN_SCHEMA_VERSION,
    AdmissionReceiptV1,
    CampaignCatalogSnapshotV1,
    CampaignError,
    CampaignEventV1,
    CampaignLimitsV1,
    CampaignStagePolicyV1,
    CampaignStatusV1,
    ExperimentProposalV1,
    OutcomePacketV1,
    PlanReceiptV1,
    PreparedPlanV1,
    ResearchCampaignSpecV1,
)
from fugue.bench.campaign_evidence import (
    outcome_eligibility_failures as _outcome_eligibility_failures,
)
from fugue.bench.campaign_evidence import (
    outcome_metrics as _outcome_metrics,
)
from fugue.bench.campaign_evidence import (
    safe_agent_evidence as _safe_agent_evidence,
)
from fugue.bench.campaign_evidence import (
    safe_prediction_row as _safe_prediction_row,
)
from fugue.bench.campaign_store import (
    CampaignStore,
)
from fugue.bench.campaign_store import (
    event_id_in_log as _event_id_in_log,
)
from fugue.bench.campaign_store import (
    read_json_object as _read_json_object,
)
from fugue.bench.campaign_store import (
    read_jsonl as _read_jsonl,
)
from fugue.bench.campaign_store import (
    read_last_json_object as _read_last_json_object,
)
from fugue.bench.campaign_store import (
    sha256_path as _sha256_path,
)
from fugue.bench.candidates import stable_digest
from fugue.bench.context import list_context_systems
from fugue.bench.execution import new_run_id
from fugue.bench.library import (
    ExperimentSpec,
    IntegrationSelection,
    get_experiment,
    get_prompt,
    get_skill,
    list_experiments,
    validate_id,
)
from fugue.bench.operator import (
    ExperimentRequest,
    OperatorService,
    ResolvedRunPlan,
    SetupPreparation,
)
from fugue.bench.reproducibility import read_evaluation_asset_lock
from fugue.bench.runtime_provenance import resolve_fugue_source_provenance
from fugue.bench.task_authoring import (
    TaskAuthoringPolicyV1,
    TaskEvaluationV1,
    TaskScoringRevisionV1,
    TaskStudyAnalysisV1,
    TaskSuiteDraftV1,
    TaskSuiteLockV1,
    TaskSuitePreviewV1,
    analyze_task_evaluation,
    evaluate_task_rows,
    load_task_profiles,
    materialize_task_suite_lock,
    preview_task_suite,
    read_task_suite_lock,
    scoring_revision_from_dict,
    task_authoring_policy_from_dict,
    task_evaluation_call_estimate,
    task_evaluation_from_dict,
    task_study_analysis_from_dict,
    task_suite_draft_from_dict,
    task_suite_lock_dir,
    task_suite_lock_from_dict,
    task_suite_preview_from_dict,
)
from fugue.model_plane import (
    model_route_identity,
    resolve_harness_model_route,
    resolve_model_route,
)
from fugue.redaction import redact_value, secrets_from_env

CAMPAIGNS_DIR = Path("configs/fugue/campaigns")
CAMPAIGN_RUNTIME_DIR = Path(".fugue/runtime/campaigns")
_TERMINAL_RUN_STATES = {"passed", "failed", "cancelled", "interrupted"}
_SAFE_TRACE_CONTENT = {"full", "metadata"}
_SAFE_EVIDENCE_SCOPES = {"summary", "rows", "traces"}
_IDEMPOTENT_ACTIONS = {
    "prepare",
    "admit",
    "launch",
    "cancel",
    "finalize",
    "lock_task_suite",
    "score_task_suite",
    "analyze_task_study",
}
_RECEIPT_KINDS = {
    "plans",
    "prepared",
    "admissions",
    "outcomes",
    "task-suites",
    "task-evaluations",
    "task-analyses",
}
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")


def list_campaigns(repo_root: Path | None = None) -> tuple[ResearchCampaignSpecV1, ...]:
    root = (repo_root or Path.cwd()) / CAMPAIGNS_DIR
    if not root.is_dir():
        return ()
    return tuple(
        get_campaign(path.stem, repo_root) for path in sorted(root.glob("*.yaml"))
    )


def get_campaign(
    campaign_id: str, repo_root: Path | None = None
) -> ResearchCampaignSpecV1:
    campaign_id = validate_id(campaign_id, kind="campaign id")
    path = (repo_root or Path.cwd()) / CAMPAIGNS_DIR / f"{campaign_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"campaign not found: {campaign_id}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: campaign must be a mapping")
    return campaign_spec_from_dict(raw, item_id=campaign_id)


def campaign_spec_from_dict(
    raw: Mapping[str, Any], *, item_id: str | None = None
) -> ResearchCampaignSpecV1:
    _reject_unknown(
        raw,
        {
            "schema_version",
            "id",
            "revision",
            "title",
            "objective",
            "allowed",
            "stages",
            "limits",
            "task_authoring",
            "evidence_scope",
            "require_clean_source",
            "campaign_digest",
        },
        "campaign",
    )
    if int(raw.get("schema_version") or 0) != CAMPAIGN_SCHEMA_VERSION:
        raise ValueError("campaign must use schema_version 1")
    campaign_id = validate_id(raw.get("id") or item_id or "", kind="campaign id")
    if item_id and campaign_id != item_id:
        raise ValueError(
            f"campaign file {item_id!r} declares mismatched id {campaign_id!r}"
        )
    allowed = _mapping(raw.get("allowed"), "campaign allowed")
    _reject_unknown(
        allowed,
        {
            "experiments",
            "models",
            "harnesses",
            "workloads",
            "variants",
            "context_systems",
            "analyses",
            "trace_content",
        },
        "campaign allowed",
    )
    limits_raw = _mapping(raw.get("limits"), "campaign limits")
    _reject_unknown(
        limits_raw,
        {
            "total_cost_usd",
            "initial_cell_reserve_usd",
            "safety_margin",
            "max_cells_per_proposal",
            "max_total_cells",
            "max_attempts_per_cell",
            "max_concurrent",
            "max_active_runs",
        },
        "campaign limits",
    )
    limits = CampaignLimitsV1(
        total_cost_usd=_positive_number(
            limits_raw.get("total_cost_usd"), "campaign total_cost_usd"
        ),
        initial_cell_reserve_usd=_non_negative_number(
            limits_raw.get("initial_cell_reserve_usd"),
            "campaign initial_cell_reserve_usd",
        ),
        safety_margin=_at_least_one_number(
            limits_raw.get("safety_margin"), "campaign safety_margin"
        ),
        max_cells_per_proposal=_positive_int(
            limits_raw.get("max_cells_per_proposal"),
            "campaign max_cells_per_proposal",
        ),
        max_total_cells=_positive_int(
            limits_raw.get("max_total_cells"), "campaign max_total_cells"
        ),
        max_attempts_per_cell=_positive_int(
            limits_raw.get("max_attempts_per_cell"),
            "campaign max_attempts_per_cell",
        ),
        max_concurrent=_positive_int(
            limits_raw.get("max_concurrent"), "campaign max_concurrent"
        ),
        max_active_runs=_positive_int(
            limits_raw.get("max_active_runs", 1), "campaign max_active_runs"
        ),
    )
    stages = tuple(
        _stage_policy(item) for item in _sequence(raw.get("stages"), "stages")
    )
    _validate_stages(stages)
    experiments = _id_tuple(allowed.get("experiments"), "experiment")
    models = _text_tuple(allowed.get("models"), "model")
    harnesses = _id_tuple(allowed.get("harnesses"), "harness")
    workloads = _id_tuple(allowed.get("workloads"), "workload")
    variants = _id_tuple(allowed.get("variants"), "variant")
    context_systems = _id_tuple(allowed.get("context_systems"), "context system")
    analyses = _id_tuple(allowed.get("analyses"), "analysis", allow_empty=True)
    trace_content = _text_tuple(allowed.get("trace_content"), "trace content")
    invalid_trace = sorted(set(trace_content) - _SAFE_TRACE_CONTENT)
    if invalid_trace:
        raise ValueError(f"unknown campaign trace content: {', '.join(invalid_trace)}")
    evidence_scope = str(raw.get("evidence_scope") or "rows")
    if evidence_scope not in _SAFE_EVIDENCE_SCOPES:
        raise ValueError("campaign evidence_scope must be summary, rows, or traces")
    unsigned = ResearchCampaignSpecV1(
        schema_version=CAMPAIGN_SCHEMA_VERSION,
        id=campaign_id,
        revision=validate_id(raw.get("revision") or "v1", kind="campaign revision"),
        title=_bounded_text(raw.get("title") or campaign_id, "campaign title", 200),
        objective=_bounded_text(raw.get("objective"), "campaign objective", 4000),
        allowed_experiments=experiments,
        allowed_models=models,
        allowed_harnesses=harnesses,
        allowed_workloads=workloads,
        allowed_variants=variants,
        allowed_context_systems=context_systems,
        allowed_analyses=analyses,
        allowed_trace_content=trace_content,
        stages=stages,
        limits=limits,
        task_authoring=task_authoring_policy_from_dict(raw.get("task_authoring")),
        evidence_scope=evidence_scope,  # type: ignore[arg-type]
        require_clean_source=bool(raw.get("require_clean_source", True)),
    )
    digest = _artifact_digest(unsigned.to_dict(), "campaign_digest")
    supplied = str(raw.get("campaign_digest") or "")
    if supplied and supplied != digest:
        raise ValueError("campaign_digest does not match the campaign contract")
    return replace(unsigned, campaign_digest=digest)


def build_experiment_proposal(
    *,
    proposal_id: str,
    campaign_id: str,
    catalog_digest: str,
    stage_id: str,
    research_question: str,
    hypothesis: str,
    fixed_dimensions: Sequence[str],
    varied_dimensions: Sequence[str],
    measured_dimensions: Sequence[str],
    experiment_id: str,
    model: str,
    n_attempts: int,
    n_concurrent: int,
    preset_id: str | None = None,
    workloads: Sequence[str] = (),
    harnesses: Sequence[str] = (),
    context_systems: Sequence[str] = (),
    variants: Sequence[str] = (),
    n_tasks: int | None = None,
    trace_content: str = "full",
    task_suite_digest: str | None = None,
    analysis_ids: Sequence[str] = (),
    parent_outcome_id: str | None = None,
    decision_rationale: str = "",
) -> ExperimentProposalV1:
    raw = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "proposal_id": proposal_id,
        "campaign_id": campaign_id,
        "catalog_digest": catalog_digest,
        "stage_id": stage_id,
        "research_question": research_question,
        "hypothesis": hypothesis,
        "fixed_dimensions": list(fixed_dimensions),
        "varied_dimensions": list(varied_dimensions),
        "measured_dimensions": list(measured_dimensions),
        "experiment_id": experiment_id,
        "preset_id": preset_id,
        "workloads": list(workloads),
        "harnesses": list(harnesses),
        "context_systems": list(context_systems),
        "variants": list(variants),
        "model": model,
        "n_attempts": n_attempts,
        "n_tasks": n_tasks,
        "n_concurrent": n_concurrent,
        "trace_content": trace_content,
        "task_suite_digest": task_suite_digest,
        "analysis_ids": list(analysis_ids),
        "parent_outcome_id": parent_outcome_id,
        "decision_rationale": decision_rationale,
    }
    proposal = _proposal_from_dict(raw, require_digest=False)
    return replace(
        proposal,
        proposal_digest=_artifact_digest(proposal.to_dict(), "proposal_digest"),
    )


def experiment_proposal_from_dict(raw: Mapping[str, Any]) -> ExperimentProposalV1:
    return _proposal_from_dict(raw, require_digest=True)


def campaign_catalog_snapshot_from_dict(
    raw: Mapping[str, Any],
) -> CampaignCatalogSnapshotV1:
    fields = {
        "schema_version",
        "campaign_id",
        "policy_digest",
        "source_provenance",
        "experiments",
        "models",
        "harnesses",
        "context_systems",
        "analyses",
        "task_authoring",
        "component_digests",
        "catalog_digest",
    }
    _reject_unknown(raw, fields, "campaign catalog snapshot")
    value = CampaignCatalogSnapshotV1(
        schema_version=_schema(raw, "campaign catalog snapshot"),
        campaign_id=validate_id(raw.get("campaign_id") or "", kind="campaign id"),
        policy_digest=_required_digest(raw.get("policy_digest"), "policy_digest"),
        source_provenance=_mapping(raw.get("source_provenance"), "source provenance"),
        experiments=tuple(
            _mapping(item, "experiment catalog item")
            for item in _sequence(raw.get("experiments"), "experiments")
        ),
        models=tuple(
            _mapping(item, "model catalog item")
            for item in _sequence(raw.get("models"), "models")
        ),
        harnesses=_id_tuple(raw.get("harnesses"), "harness"),
        context_systems=tuple(
            _mapping(item, "context catalog item")
            for item in _sequence(raw.get("context_systems"), "context systems")
        ),
        analyses=tuple(
            _mapping(item, "analysis catalog item")
            for item in _sequence(raw.get("analyses"), "analyses")
        ),
        task_authoring=(
            _mapping(raw["task_authoring"], "task authoring catalog")
            if raw.get("task_authoring") is not None
            else None
        ),
        component_digests=_digest_mapping(
            raw.get("component_digests"), "component digests"
        ),
        catalog_digest=_required_digest(raw.get("catalog_digest"), "catalog_digest"),
    )
    _verify_artifact(value.to_dict(), "catalog_digest", "campaign catalog snapshot")
    return value


def plan_receipt_from_dict(raw: Mapping[str, Any]) -> PlanReceiptV1:
    return _plan_receipt_from_dict(raw)


def prepared_plan_from_dict(raw: Mapping[str, Any]) -> PreparedPlanV1:
    return _prepared_plan_from_dict(raw)


def admission_receipt_from_dict(raw: Mapping[str, Any]) -> AdmissionReceiptV1:
    return _admission_receipt_from_dict(raw)


def outcome_packet_from_dict(raw: Mapping[str, Any]) -> OutcomePacketV1:
    return _outcome_packet_from_dict(raw)


def campaign_event_from_dict(raw: Mapping[str, Any]) -> CampaignEventV1:
    return _campaign_event_from_dict(raw)


def campaign_error_from_dict(raw: Mapping[str, Any]) -> CampaignError:
    fields = {
        "schema_version",
        "code",
        "category",
        "safe_to_repeat",
        "message",
        "details",
        "error_digest",
    }
    _reject_unknown(raw, fields, "campaign error")
    _schema(raw, "campaign error")
    _verify_artifact(raw, "error_digest", "campaign error")
    return CampaignError(
        validate_id(raw.get("code") or "", kind="campaign error code"),
        _bounded_text(raw.get("message"), "campaign error message", 2000),
        category=str(raw.get("category") or ""),
        retryable=_strict_bool(
            raw.get("safe_to_repeat"), "campaign error safe_to_repeat"
        ),
        details=_mapping(raw.get("details"), "campaign error details"),
    )


def campaign_status_from_dict(raw: Mapping[str, Any]) -> CampaignStatusV1:
    fields = {
        "schema_version",
        "campaign_id",
        "subject_id",
        "state",
        "policy_digest",
        "active_runs",
        "runs",
        "admissions",
        "outcomes",
        "admitted_cells",
        "total_cost_usd",
        "accounted_cost_usd",
        "reserved_cost_usd",
        "remaining_cost_usd",
        "next_actions",
        "blockers",
        "status_digest",
    }
    _reject_unknown(raw, fields, "campaign status")
    value = CampaignStatusV1(
        schema_version=_schema(raw, "campaign status"),
        campaign_id=validate_id(raw.get("campaign_id") or "", kind="campaign id"),
        subject_id=validate_id(raw.get("subject_id") or "", kind="subject id"),
        state=_bounded_text(raw.get("state"), "campaign state", 100),
        policy_digest=_required_digest(raw.get("policy_digest"), "policy digest"),
        active_runs=_id_tuple(raw.get("active_runs"), "run", allow_empty=True),
        runs=tuple(
            _mapping(item, "campaign run")
            for item in _sequence(raw.get("runs"), "runs")
        ),
        admissions=_non_negative_int(raw.get("admissions"), "admissions"),
        outcomes=_non_negative_int(raw.get("outcomes"), "outcomes"),
        admitted_cells=_non_negative_int(raw.get("admitted_cells"), "admitted cells"),
        total_cost_usd=_non_negative_number(raw.get("total_cost_usd"), "total cost"),
        accounted_cost_usd=_non_negative_number(
            raw.get("accounted_cost_usd"), "accounted cost"
        ),
        reserved_cost_usd=_non_negative_number(
            raw.get("reserved_cost_usd"), "reserved cost"
        ),
        remaining_cost_usd=_non_negative_number(
            raw.get("remaining_cost_usd"), "remaining cost"
        ),
        next_actions=_text_tuple(
            raw.get("next_actions"), "next action", allow_empty=True
        ),
        blockers=_text_tuple(raw.get("blockers"), "blocker", allow_empty=True),
        status_digest=_required_digest(raw.get("status_digest"), "status digest"),
    )
    _verify_artifact(value.to_dict(), "status_digest", "campaign status")
    return value


def _proposal_from_dict(
    raw: Mapping[str, Any], *, require_digest: bool
) -> ExperimentProposalV1:
    fields = {
        "schema_version",
        "proposal_id",
        "campaign_id",
        "catalog_digest",
        "stage_id",
        "research_question",
        "hypothesis",
        "fixed_dimensions",
        "varied_dimensions",
        "measured_dimensions",
        "experiment_id",
        "preset_id",
        "workloads",
        "harnesses",
        "context_systems",
        "variants",
        "model",
        "n_attempts",
        "n_tasks",
        "n_concurrent",
        "trace_content",
        "task_suite_digest",
        "analysis_ids",
        "parent_outcome_id",
        "decision_rationale",
        "proposal_digest",
    }
    _reject_unknown(raw, fields, "experiment proposal")
    if int(raw.get("schema_version") or 0) != CAMPAIGN_SCHEMA_VERSION:
        raise ValueError("experiment proposal must use schema_version 1")
    proposal = ExperimentProposalV1(
        schema_version=CAMPAIGN_SCHEMA_VERSION,
        proposal_id=validate_id(raw.get("proposal_id") or "", kind="proposal id"),
        campaign_id=validate_id(raw.get("campaign_id") or "", kind="campaign id"),
        catalog_digest=_required_digest(raw.get("catalog_digest"), "catalog_digest"),
        stage_id=validate_id(raw.get("stage_id") or "", kind="campaign stage id"),
        research_question=_bounded_text(
            raw.get("research_question"), "research question", 4000
        ),
        hypothesis=_bounded_text(raw.get("hypothesis"), "hypothesis", 4000),
        fixed_dimensions=_dimension_tuple(raw.get("fixed_dimensions"), "fixed"),
        varied_dimensions=_dimension_tuple(raw.get("varied_dimensions"), "varied"),
        measured_dimensions=_dimension_tuple(
            raw.get("measured_dimensions"), "measured"
        ),
        experiment_id=validate_id(raw.get("experiment_id") or "", kind="experiment id"),
        preset_id=(
            validate_id(raw["preset_id"], kind="preset id")
            if raw.get("preset_id")
            else None
        ),
        workloads=_id_tuple(raw.get("workloads"), "workload", allow_empty=True),
        harnesses=_id_tuple(raw.get("harnesses"), "harness", allow_empty=True),
        context_systems=_id_tuple(
            raw.get("context_systems"), "context system", allow_empty=True
        ),
        variants=_id_tuple(raw.get("variants"), "variant", allow_empty=True),
        model=_bounded_text(raw.get("model"), "model", 300),
        n_attempts=_positive_int(raw.get("n_attempts"), "proposal n_attempts"),
        n_tasks=(
            _positive_int(raw.get("n_tasks"), "proposal n_tasks")
            if raw.get("n_tasks") is not None
            else None
        ),
        n_concurrent=_positive_int(raw.get("n_concurrent"), "proposal n_concurrent"),
        trace_content=_bounded_text(raw.get("trace_content"), "trace content", 32),
        task_suite_digest=(
            _required_digest(raw["task_suite_digest"], "task suite digest")
            if raw.get("task_suite_digest")
            else None
        ),
        analysis_ids=_id_tuple(raw.get("analysis_ids"), "analysis", allow_empty=True),
        parent_outcome_id=(
            validate_id(raw["parent_outcome_id"], kind="outcome id")
            if raw.get("parent_outcome_id")
            else None
        ),
        decision_rationale=(
            _bounded_text(raw.get("decision_rationale"), "decision rationale", 4000)
            if raw.get("decision_rationale")
            else ""
        ),
        proposal_digest=str(raw.get("proposal_digest") or ""),
    )
    digest = _artifact_digest(proposal.to_dict(), "proposal_digest")
    if require_digest and proposal.proposal_digest != digest:
        raise ValueError("proposal_digest does not match the experiment proposal")
    if proposal.proposal_digest and proposal.proposal_digest != digest:
        raise ValueError("proposal_digest does not match the experiment proposal")
    return replace(proposal, proposal_digest=digest)


class CampaignService:
    """Policy-governed orchestration over the canonical OperatorService."""

    def __init__(
        self,
        repo_root: Path | None = None,
        env_file: Path | None = None,
        *,
        operator: OperatorService | None = None,
    ) -> None:
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self.operator = operator or OperatorService(self.repo_root, env_file)
        self._store = CampaignStore(self.repo_root, CAMPAIGN_RUNTIME_DIR)

    def catalog(self, campaign_id: str) -> CampaignCatalogSnapshotV1:
        policy = self._policy(campaign_id)
        source = resolve_fugue_source_provenance(self.repo_root)
        self._require_source(policy, source)
        experiment_items = {item.id: item for item in list_experiments(self.repo_root)}
        missing_experiments = sorted(
            set(policy.allowed_experiments) - set(experiment_items)
        )
        if missing_experiments:
            raise CampaignError(
                "unregistered_component",
                "campaign allows experiments that are not registered",
                category="validation",
                details={"experiments": missing_experiments},
            )
        registered_analyses = {
            item["id"]: item for item in list_analyses(self.repo_root)
        }
        missing_analyses = sorted(
            set(policy.allowed_analyses) - set(registered_analyses)
        )
        if missing_analyses:
            raise CampaignError(
                "unregistered_component",
                "campaign allows analyses that are not registered",
                category="validation",
                details={"analyses": missing_analyses},
            )
        registered_context = {
            item.id: item for item in list_context_systems(self.repo_root)
        }
        missing_context = sorted(
            set(policy.allowed_context_systems) - set(registered_context)
        )
        if missing_context:
            raise CampaignError(
                "unregistered_component",
                "campaign allows context systems that are not registered",
                category="validation",
                details={"context_systems": missing_context},
            )

        component_digests: dict[str, str] = {}
        experiments: list[dict[str, Any]] = []
        registered_harnesses: set[str] = set()
        registered_workloads: set[str] = set()
        registered_variants: set[str] = set()
        for experiment_id in policy.allowed_experiments:
            item = experiment_items[experiment_id]
            experiment = get_experiment(experiment_id, self.repo_root)
            component_digests[f"experiment:{experiment_id}"] = item.sha256
            experiments.append(_safe_experiment_record(experiment, item.sha256))
            registered_harnesses.update(experiment.harnesses)
            registered_workloads.update(item.id for item in experiment.workloads)
            if not experiment.workloads:
                registered_workloads.add("harbor")
            registered_variants.update(item.id for item in experiment.variants)
        if policy.task_authoring is not None:
            registered_workloads.add("harbor")
        for kind, allowed_values, registered_values in (
            ("harness", policy.allowed_harnesses, registered_harnesses),
            ("workload", policy.allowed_workloads, registered_workloads),
            ("variant", policy.allowed_variants, registered_variants),
        ):
            missing = sorted(set(allowed_values) - registered_values)
            if missing:
                raise CampaignError(
                    "unregistered_component",
                    f"campaign allows {kind}s not registered by its experiments",
                    category="validation",
                    details={f"{kind}s": missing},
                )

        contexts: list[dict[str, Any]] = []
        for system_id in policy.allowed_context_systems:
            spec = registered_context[system_id]
            path = (
                self.repo_root / "configs/fugue/context-systems" / f"{system_id}.yaml"
            )
            digest = _sha256_path(path)
            component_digests[f"context_system:{system_id}"] = digest
            contexts.append(
                {
                    "id": spec.id,
                    "title": spec.title,
                    "description": spec.description,
                    "version": spec.version,
                    "capabilities": sorted(spec.capabilities),
                    "deliveries": sorted(spec.deliveries),
                    "support": spec.support,
                    "sha256": digest,
                }
            )

        analyses: list[dict[str, Any]] = []
        for analysis_id in policy.allowed_analyses:
            spec = get_analysis(analysis_id, self.repo_root)
            path = self.repo_root / "configs/fugue/analyses" / f"{analysis_id}.yaml"
            digest = _sha256_path(path)
            component_digests[f"analysis:{analysis_id}"] = digest
            analyses.append(
                {
                    "id": spec.id,
                    "title": spec.title,
                    "question": spec.question,
                    "group_by": list(spec.group_by),
                    "metrics": list(spec.metrics),
                    "sha256": digest,
                }
            )

        models: list[dict[str, Any]] = []
        for model in policy.allowed_models:
            try:
                route = resolve_model_route(model, self.operator.env)
            except ValueError as exc:
                raise CampaignError(
                    "unregistered_component",
                    f"campaign model route is invalid: {model}",
                    category="validation",
                    details={"error": str(exc)},
                ) from exc
            identity = model_route_identity(route)
            models.append(
                {
                    "id": model,
                    "provider": identity["provider"],
                    "model_id": identity["model_id"],
                    "display_model": identity["display_model"],
                    "tool_result_modalities": identity["tool_result_modalities"],
                }
            )

        task_authoring = None
        if policy.task_authoring is not None:
            try:
                task_profiles = load_task_profiles(self.repo_root)
                _validate_task_profile_policy(policy.task_authoring, task_profiles)
            except (FileNotFoundError, KeyError, ValueError) as exc:
                raise CampaignError(
                    "unregistered_component",
                    f"campaign task authoring profiles are invalid: {exc}",
                    category="validation",
                ) from exc
            component_digests["task_profiles"] = task_profiles.catalog_digest
            task_authoring = _safe_task_profile_catalog(
                policy.task_authoring, task_profiles
            )

        unsigned = CampaignCatalogSnapshotV1(
            schema_version=CAMPAIGN_SCHEMA_VERSION,
            campaign_id=policy.id,
            policy_digest=policy.campaign_digest,
            source_provenance=_json_value(source),
            experiments=tuple(experiments),
            models=tuple(models),
            harnesses=policy.allowed_harnesses,
            context_systems=tuple(contexts),
            analyses=tuple(analyses),
            task_authoring=task_authoring,
            component_digests=dict(sorted(component_digests.items())),
        )
        return replace(
            unsigned,
            catalog_digest=_artifact_digest(unsigned.to_dict(), "catalog_digest"),
        )

    def preview_task_suite(
        self,
        campaign_id: str,
        catalog_digest: str,
        draft: TaskSuiteDraftV1,
    ) -> TaskSuitePreviewV1:
        policy = self._policy(campaign_id)
        catalog = self.catalog(campaign_id)
        if catalog.catalog_digest != catalog_digest:
            raise CampaignError(
                "catalog_drift",
                "task draft catalog digest does not match the current catalog",
                category="validation",
            )
        authoring = self._task_authoring_policy(policy)
        self._stage(policy, draft.stage_id)
        normalized = task_suite_draft_from_dict(draft.to_dict(), require_digest=True)
        profiles = load_task_profiles(self.repo_root)
        return preview_task_suite(
            campaign_id=campaign_id,
            catalog_digest=catalog_digest,
            policy_digest=policy.campaign_digest,
            draft=normalized,
            policy=authoring,
            profiles=profiles,
            harnesses=policy.allowed_harnesses,
            repo_root=self.repo_root,
        )

    def lock_task_suite(
        self, preview: TaskSuitePreviewV1, operation_id: str
    ) -> TaskSuiteLockV1:
        operation_id = self._operation_id(operation_id)
        preview = task_suite_preview_from_dict(preview.to_dict())
        campaign_id = preview.campaign_id
        operation_input = stable_digest(
            {"action": "lock_task_suite", "preview_digest": preview.preview_digest}
        )
        with self._operation_lock(campaign_id, operation_id):
            existing = self._completed_operation(
                campaign_id, operation_id, "lock_task_suite", operation_input
            )
            if existing is not None:
                return task_suite_lock_from_dict(self._read_artifact(existing))
            policy = self._policy(campaign_id)
            catalog = self.catalog(campaign_id)
            if preview.policy_digest != policy.campaign_digest:
                raise CampaignError(
                    "policy_drift",
                    "task suite preview is bound to a different campaign policy",
                    category="policy",
                )
            if preview.catalog_digest != catalog.catalog_digest:
                raise CampaignError(
                    "catalog_drift",
                    "task suite preview is bound to a different catalog",
                    category="validation",
                )
            if not preview.eligible:
                raise CampaignError(
                    "task_suite_ineligible",
                    "task suite preview contains policy or capability failures",
                    category="policy",
                    details={"failures": list(preview.failures)},
                )
            self._task_authoring_policy(policy)
            self._initialize_campaign(policy)
            destination = task_suite_lock_dir(
                self.repo_root, campaign_id, preview.preview_digest
            )
            profiles = load_task_profiles(self.repo_root)
            if destination.is_dir():
                lock = task_suite_lock_from_dict(
                    _read_json_object(destination / "task-suite-lock.json")
                )
                if lock.preview_digest != preview.preview_digest:
                    raise CampaignError(
                        "artifact_conflict",
                        "task suite asset directory belongs to another preview",
                        category="evidence",
                    )
            else:
                try:
                    lock = materialize_task_suite_lock(
                        preview,
                        profiles=profiles,
                        repo_root=self.repo_root,
                        destination=destination,
                        harnesses=policy.allowed_harnesses,
                    )
                except Exception:
                    if destination.exists():
                        import shutil

                        shutil.rmtree(destination)
                    raise
            relative = self._write_receipt(
                "task-suites", lock.suite_digest, lock.to_dict()
            )
            self._record_operation(
                campaign_id,
                operation_id,
                "lock_task_suite",
                operation_input,
                relative,
                lock.suite_digest,
            )
            self._event(
                campaign_id,
                "task_suite_locked",
                operation_id=operation_id,
                artifact_type="TaskSuiteLockV1",
                artifact_digest=lock.suite_digest,
            )
            return lock

    def score_task_suite(
        self,
        run_id: str,
        task_suite_digest: str,
        scoring_revision: TaskScoringRevisionV1,
        operation_id: str,
    ) -> TaskEvaluationV1:
        operation_id = self._operation_id(operation_id)
        campaign_id, _ = self._resolve_campaign_subject(run_id)
        revision = scoring_revision_from_dict(scoring_revision.to_dict())
        operation_input = stable_digest(
            {
                "action": "score_task_suite",
                "run_id": run_id,
                "task_suite_digest": task_suite_digest,
                "revision_digest": revision.revision_digest,
            }
        )
        with self._operation_lock(campaign_id, operation_id):
            existing = self._completed_operation(
                campaign_id, operation_id, "score_task_suite", operation_input
            )
            if existing is not None:
                return task_evaluation_from_dict(self._read_artifact(existing))
            in_progress = self._operation(campaign_id, operation_id)
            if in_progress is not None:
                raise CampaignError(
                    "operation_incomplete",
                    "task scoring previously started but did not commit an artifact; "
                    "the operation is blocked to prevent repeating paid judge calls",
                    category="evidence",
                    details={"operation_id": operation_id},
                )
            policy = self._policy(campaign_id)
            lock = read_task_suite_lock(self.repo_root, campaign_id, task_suite_digest)
            self._require_run_task_suite(campaign_id, run_id, lock)
            run = self.operator.run_summary(run_id)
            if run.status not in _TERMINAL_RUN_STATES:
                raise CampaignError(
                    "run_not_terminal",
                    "task scoring requires a terminal run",
                    category="evidence",
                    retryable=True,
                )
            export_path = (
                self._campaign_dir(campaign_id)
                / "task-evaluation-exports"
                / f"{run_id}.jsonl"
            )
            exported = self.operator.export_run(
                run_id,
                out=export_path,
                fetch_weave=True,
                to_weave=False,
            )
            rows = _read_jsonl(exported.path)
            profiles = load_task_profiles(self.repo_root)
            call_estimate = task_evaluation_call_estimate(
                lock, rows, self.repo_root, profiles
            )
            paid_call_reserve = float(call_estimate["reserve_usd"])
            if paid_call_reserve > self._remaining_task_budget(campaign_id, policy):
                raise CampaignError(
                    "budget_exceeded",
                    "task evaluation paid-call reserve exceeds the remaining "
                    "campaign budget",
                    category="admission",
                    details={
                        "judge_calls": call_estimate["judge"],
                        "reserve_usd": paid_call_reserve,
                    },
                )
            self._write_operation(
                campaign_id,
                operation_id,
                {
                    "schema_version": CAMPAIGN_SCHEMA_VERSION,
                    "operation_id": operation_id,
                    "action": "score_task_suite",
                    "input_digest": operation_input,
                    "status": "scoring",
                    "judge_calls": call_estimate["judge"],
                    "paid_call_reserve_usd": paid_call_reserve,
                },
            )
            evaluation = evaluate_task_rows(
                campaign_id=campaign_id,
                run_id=run_id,
                lock=lock,
                revision=revision,
                rows=rows,
                profiles=profiles,
                repo_root=self.repo_root,
                env=self.operator.env,
            )
            if evaluation.accounted_cost_usd > self._remaining_task_budget(
                campaign_id, policy
            ):
                raise CampaignError(
                    "budget_exceeded",
                    "task evaluation cost exceeds the remaining campaign budget",
                    category="admission",
                )
            relative = self._write_receipt(
                "task-evaluations",
                evaluation.evaluation_digest,
                evaluation.to_dict(),
            )
            self._account_task_evaluation(campaign_id, policy, run_id, evaluation)
            self._record_operation(
                campaign_id,
                operation_id,
                "score_task_suite",
                operation_input,
                relative,
                evaluation.evaluation_digest,
            )
            self._event(
                campaign_id,
                "task_evaluation_scored",
                operation_id=operation_id,
                run_id=run_id,
                artifact_type="TaskEvaluationV1",
                artifact_digest=evaluation.evaluation_digest,
            )
            return evaluation

    def analyze_task_study(
        self,
        run_id: str,
        analysis_id: str,
        operation_id: str,
        *,
        evaluation_digest: str | None = None,
    ) -> TaskStudyAnalysisV1:
        operation_id = self._operation_id(operation_id)
        campaign_id, _ = self._resolve_campaign_subject(run_id)
        operation_input = stable_digest(
            {
                "action": "analyze_task_study",
                "run_id": run_id,
                "analysis_id": analysis_id,
                "evaluation_digest": evaluation_digest,
            }
        )
        with self._operation_lock(campaign_id, operation_id):
            existing = self._completed_operation(
                campaign_id, operation_id, "analyze_task_study", operation_input
            )
            if existing is not None:
                return task_study_analysis_from_dict(self._read_artifact(existing))
            evaluation = self._task_evaluation(campaign_id, run_id, evaluation_digest)
            lock = read_task_suite_lock(
                self.repo_root, campaign_id, evaluation.task_suite_digest
            )
            analysis = analyze_task_evaluation(
                analysis_id=analysis_id,
                lock=lock,
                evaluation=evaluation,
                repo_root=self.repo_root,
            )
            relative = self._write_receipt(
                "task-analyses", analysis.analysis_digest, analysis.to_dict()
            )
            self._record_operation(
                campaign_id,
                operation_id,
                "analyze_task_study",
                operation_input,
                relative,
                analysis.analysis_digest,
            )
            self._event(
                campaign_id,
                "task_study_analyzed",
                operation_id=operation_id,
                run_id=run_id,
                artifact_type="TaskStudyAnalysisV1",
                artifact_digest=analysis.analysis_digest,
            )
            return analysis

    def preview(self, proposal: ExperimentProposalV1) -> PlanReceiptV1:
        _verify_artifact(proposal.to_dict(), "proposal_digest", "experiment proposal")
        policy = self._policy(proposal.campaign_id)
        catalog = self.catalog(policy.id)
        self._validate_proposal(policy, catalog, proposal)
        request = self._request(proposal)
        experiment = self._proposal_experiment(proposal)
        try:
            plan = self.operator.resolve_run_plan(
                request,
                run_id="campaign-preview",
                experiment=experiment,
            )
        except Exception as exc:
            raise CampaignError(
                "plan_resolution_failed",
                f"campaign proposal could not be resolved: {exc}",
                category="validation",
            ) from exc
        self._validate_resolved_plan(policy, proposal, plan)
        job_index = _plan_job_index(plan)
        cells = tuple(
            _plan_cell_record(index, cell, job_index)
            for index, cell in enumerate(plan.cells)
        )
        expected_predictions = sum(int(item["expected_predictions"]) for item in cells)
        stage = self._stage(policy, proposal.stage_id)
        if len(cells) > stage.max_cells:
            raise CampaignError(
                "stage_cell_limit",
                f"proposal resolves to {len(cells)} cells; stage permits {stage.max_cells}",
                category="policy",
            )
        if len(cells) > policy.limits.max_cells_per_proposal:
            raise CampaignError(
                "proposal_cell_limit",
                "proposal exceeds the campaign cell limit",
                category="policy",
            )
        components = _plan_component_digests(
            self.repo_root, proposal, plan, catalog.component_digests
        )
        if proposal.task_suite_digest:
            lock = read_task_suite_lock(
                self.repo_root, proposal.campaign_id, proposal.task_suite_digest
            )
            components = {
                **components,
                "task_suite": lock.suite_digest,
                "task_definition": lock.task_definition_digest,
                "task_criteria": lock.criteria_digest,
                **{
                    f"task_component:{key}": value
                    for key, value in lock.component_digests.items()
                },
            }
        unsigned = PlanReceiptV1(
            schema_version=CAMPAIGN_SCHEMA_VERSION,
            campaign_id=policy.id,
            proposal_id=proposal.proposal_id,
            proposal_digest=proposal.proposal_digest,
            policy_digest=policy.campaign_digest,
            catalog_digest=catalog.catalog_digest,
            source_provenance=catalog.source_provenance,
            proposal=proposal.to_dict(),
            request=_safe_request(request),
            cells=cells,
            cell_count=len(cells),
            applicable_cells=sum(bool(item["applicable"]) for item in cells),
            expected_predictions=expected_predictions,
            max_concurrent=plan.max_workers,
            component_digests=components,
            qualification_requirements=stage.required_evidence,
        )
        return replace(
            unsigned,
            plan_digest=_artifact_digest(unsigned.to_dict(), "plan_digest"),
        )

    def prepare(self, plan_receipt: PlanReceiptV1, operation_id: str) -> PreparedPlanV1:
        operation_id = self._operation_id(operation_id)
        self._verify_plan_receipt(plan_receipt)
        campaign_id = plan_receipt.campaign_id
        operation_input = stable_digest(
            {"action": "prepare", "plan_digest": plan_receipt.plan_digest}
        )
        with self._operation_lock(campaign_id, operation_id):
            existing = self._completed_operation(
                campaign_id, operation_id, "prepare", operation_input
            )
            if existing is not None:
                receipt = _prepared_plan_from_dict(self._read_artifact(existing))
                self._event(
                    campaign_id,
                    "plan_prepared",
                    operation_id=operation_id,
                    proposal_id=receipt.proposal_id,
                    artifact_type="PreparedPlanV1",
                    artifact_digest=receipt.prepared_plan_digest,
                )
                return receipt
            current = self._revalidate_plan(plan_receipt)
            policy = self._policy(campaign_id)
            self._initialize_campaign(policy)
            self._write_operation(
                campaign_id,
                operation_id,
                {
                    "schema_version": CAMPAIGN_SCHEMA_VERSION,
                    "operation_id": operation_id,
                    "action": "prepare",
                    "input_digest": operation_input,
                    "status": "preparing",
                },
            )
            self._write_receipt("plans", current.plan_digest, current.to_dict())
            request = _request_from_safe(current.request)
            proposal = experiment_proposal_from_dict(current.proposal)
            experiment = self._proposal_experiment(proposal)
            secrets = secrets_from_env(self.operator.env)
            try:
                preparation = self.operator.prepare(request, experiment=experiment)
                checks = self.operator.preflight(
                    request, live=True, experiment=experiment
                )
                canonical = _canonical_preparation(preparation, self.repo_root)
                _require_preparation(canonical)
                failed_checks = [item for item in checks if not item.ok]
                if failed_checks:
                    detail = "; ".join(
                        f"{item.name}: {_safe_diagnostic(item.detail, secrets)}"
                        for item in failed_checks
                    )
                    raise CampaignError(
                        "preflight_failed",
                        f"campaign live preflight failed: {detail}",
                        category="preparation",
                        retryable=True,
                    )
            except CampaignError:
                raise
            except Exception as exc:
                raise CampaignError(
                    "preparation_failed",
                    "campaign preparation failed in OperatorService",
                    category="preparation",
                    retryable=True,
                    details={"exception_type": type(exc).__name__},
                ) from exc
            refreshed = self._revalidate_plan(current)
            if refreshed.component_digests != current.component_digests:
                raise CampaignError(
                    "plan_drift",
                    "registered components changed during preparation",
                    category="preparation",
                )
            preflight = tuple(
                {
                    "name": item.name,
                    "ok": item.ok,
                    "detail": _safe_diagnostic(item.detail, secrets),
                }
                for item in checks
            )
            route_locks = _prepared_route_locks(current.cells, self.operator.env)
            integration_locks = {
                key.removeprefix("integration:"): value
                for key, value in current.component_digests.items()
                if key.startswith("integration:")
            }
            unsigned = PreparedPlanV1(
                schema_version=CAMPAIGN_SCHEMA_VERSION,
                campaign_id=campaign_id,
                proposal_id=current.proposal_id,
                plan_digest=current.plan_digest,
                policy_digest=policy.campaign_digest,
                source_provenance=current.source_provenance,
                plan=current.to_dict(),
                preparation=canonical,
                preflight=preflight,
                component_digests=current.component_digests,
                route_locks=route_locks,
                integration_locks=integration_locks,
                prepared_at=_now(),
            )
            receipt = replace(
                unsigned,
                prepared_plan_digest=_artifact_digest(
                    unsigned.to_dict(), "prepared_plan_digest"
                ),
            )
            relative = self._write_receipt(
                "prepared", receipt.prepared_plan_digest, receipt.to_dict()
            )
            self._record_operation(
                campaign_id,
                operation_id,
                "prepare",
                operation_input,
                relative,
                receipt.prepared_plan_digest,
            )
            self._event(
                campaign_id,
                "plan_prepared",
                operation_id=operation_id,
                proposal_id=receipt.proposal_id,
                artifact_type="PreparedPlanV1",
                artifact_digest=receipt.prepared_plan_digest,
            )
            return receipt

    def admit(
        self, prepared_plan: PreparedPlanV1, operation_id: str
    ) -> AdmissionReceiptV1:
        operation_id = self._operation_id(operation_id)
        self._verify_prepared_plan(prepared_plan)
        campaign_id = prepared_plan.campaign_id
        operation_input = stable_digest(
            {
                "action": "admit",
                "prepared_plan_digest": prepared_plan.prepared_plan_digest,
            }
        )
        with (
            self._operation_lock(campaign_id, operation_id),
            self._campaign_lock(campaign_id),
        ):
            existing = self._completed_operation(
                campaign_id, operation_id, "admit", operation_input
            )
            if existing is not None:
                receipt = _admission_receipt_from_dict(self._read_artifact(existing))
                self._event(
                    campaign_id,
                    "plan_admitted",
                    operation_id=operation_id,
                    proposal_id=receipt.proposal_id,
                    admission_id=receipt.admission_id,
                    artifact_type="AdmissionReceiptV1",
                    artifact_digest=receipt.admission_digest,
                )
                return receipt
            plan = _plan_receipt_from_dict(prepared_plan.plan)
            policy = self._policy(campaign_id)
            self._require_policy_snapshot(policy)
            if prepared_plan.policy_digest != policy.campaign_digest:
                raise CampaignError(
                    "policy_drift",
                    "prepared plan is bound to a different campaign policy",
                    category="policy",
                )
            self._revalidate_plan(plan)
            self._revalidate_prepared_bindings(prepared_plan, plan)
            proposal = experiment_proposal_from_dict(plan.proposal)
            stage = self._stage(policy, proposal.stage_id)
            admission_id = stable_digest(
                {
                    "schema_version": CAMPAIGN_SCHEMA_VERSION,
                    "campaign_id": campaign_id,
                    "proposal_id": proposal.proposal_id,
                    "prepared_plan_digest": prepared_plan.prepared_plan_digest,
                    "operation_id": operation_id,
                }
            )
            ledger = self._ledger(campaign_id, policy)
            existing_admissions = [
                item
                for item in ledger["admissions"]
                if item.get("proposal_id") == proposal.proposal_id
            ]
            if (
                existing_admissions
                and existing_admissions[0].get("admission_id") == admission_id
            ):
                receipt = _admission_receipt_from_dict(
                    self._read_receipt(campaign_id, "admissions", admission_id)
                )
                relative = (
                    (
                        self._campaign_dir(campaign_id)
                        / "admissions"
                        / f"{admission_id}.json"
                    )
                    .relative_to(self.repo_root)
                    .as_posix()
                )
                self._record_operation(
                    campaign_id,
                    operation_id,
                    "admit",
                    operation_input,
                    relative,
                    receipt.admission_digest,
                )
                self._event(
                    campaign_id,
                    "plan_admitted",
                    operation_id=operation_id,
                    proposal_id=receipt.proposal_id,
                    admission_id=receipt.admission_id,
                    artifact_type="AdmissionReceiptV1",
                    artifact_digest=receipt.admission_digest,
                )
                return receipt
            if existing_admissions:
                raise CampaignError(
                    "proposal_already_admitted",
                    "a prepared proposal may be admitted only once",
                    category="admission",
                )
            stage_count = sum(
                item.get("stage_id") == stage.id for item in ledger["admissions"]
            )
            if stage_count >= stage.max_proposals:
                raise CampaignError(
                    "stage_proposal_limit",
                    "campaign stage has reached its proposal limit",
                    category="policy",
                )
            admitted_cells = sum(
                int(item["cell_count"]) for item in ledger["admissions"]
            )
            if admitted_cells + plan.cell_count > policy.limits.max_total_cells:
                raise CampaignError(
                    "campaign_cell_limit",
                    "campaign has reached its total admitted-cell limit",
                    category="policy",
                )
            parent = self._eligible_parent(policy, proposal, stage)
            prior_maximum = (
                float(parent["maximum_measured_cell_cost_usd"])
                if parent is not None
                and parent.get("maximum_measured_cell_cost_usd") is not None
                else None
            )
            reservation, _ = reserve_campaign_cost(
                cell_count=plan.cell_count,
                initial_cell_reserve_usd=policy.limits.initial_cell_reserve_usd,
                safety_margin=policy.limits.safety_margin,
                prior_maximum_cell_cost_usd=prior_maximum,
            )
            per_cell = reservation / plan.cell_count
            remaining = _remaining_budget(ledger, policy)
            if reservation > remaining + 1e-9:
                raise CampaignError(
                    "budget_exceeded",
                    f"campaign reservation ${reservation:.2f} exceeds remaining budget ${remaining:.2f}",
                    category="admission",
                )
            self._write_operation(
                campaign_id,
                operation_id,
                {
                    "schema_version": CAMPAIGN_SCHEMA_VERSION,
                    "operation_id": operation_id,
                    "action": "admit",
                    "input_digest": operation_input,
                    "status": "admitting",
                    "admission_id": admission_id,
                },
            )
            receipt_path = (
                self._campaign_dir(campaign_id) / "admissions" / f"{admission_id}.json"
            )
            if receipt_path.is_file():
                receipt = _admission_receipt_from_dict(_read_json_object(receipt_path))
            else:
                unsigned = AdmissionReceiptV1(
                    schema_version=CAMPAIGN_SCHEMA_VERSION,
                    admission_id=admission_id,
                    campaign_id=campaign_id,
                    proposal_id=proposal.proposal_id,
                    stage_id=proposal.stage_id,
                    prepared_plan_digest=prepared_plan.prepared_plan_digest,
                    policy_digest=policy.campaign_digest,
                    operation_id=operation_id,
                    parent_outcome_id=proposal.parent_outcome_id,
                    cell_count=plan.cell_count,
                    reserved_cell_cost_usd=per_cell,
                    reserved_cost_usd=reservation,
                    prepared_plan=prepared_plan.to_dict(),
                    admitted_at=_now(),
                )
                receipt = replace(
                    unsigned,
                    admission_digest=_artifact_digest(
                        unsigned.to_dict(), "admission_digest"
                    ),
                )
            relative = self._write_receipt(
                "admissions", receipt.admission_id, receipt.to_dict()
            )
            ledger["admissions"].append(
                {
                    "admission_id": receipt.admission_id,
                    "proposal_id": receipt.proposal_id,
                    "stage_id": receipt.stage_id,
                    "status": "admitted",
                    "cell_count": receipt.cell_count,
                    "reserved_cell_cost_usd": receipt.reserved_cell_cost_usd,
                    "reserved_cost_usd": receipt.reserved_cost_usd,
                    "actual_cost_usd": None,
                    "run_id": None,
                    "outcome_id": None,
                }
            )
            self._write_ledger(campaign_id, ledger)
            self._record_operation(
                campaign_id,
                operation_id,
                "admit",
                operation_input,
                relative,
                receipt.admission_digest,
            )
            self._event(
                campaign_id,
                "plan_admitted",
                operation_id=operation_id,
                proposal_id=receipt.proposal_id,
                admission_id=receipt.admission_id,
                artifact_type="AdmissionReceiptV1",
                artifact_digest=receipt.admission_digest,
            )
            return receipt

    def launch(
        self, admission_receipt: AdmissionReceiptV1, operation_id: str
    ) -> CampaignStatusV1:
        operation_id = self._operation_id(operation_id)
        self._verify_admission(admission_receipt)
        campaign_id = admission_receipt.campaign_id
        operation_input = stable_digest(
            {
                "action": "launch",
                "admission_digest": admission_receipt.admission_digest,
            }
        )
        with self._operation_lock(campaign_id, operation_id):
            operation = self._operation(campaign_id, operation_id)
            if operation is not None:
                self._require_operation_match(
                    operation, "launch", operation_input, operation_id
                )
                run_id = str(operation.get("run_id") or "")
                if not run_id:
                    raise CampaignError(
                        "operation_corrupt",
                        "launch operation is missing its run identity",
                        category="execution",
                    )
                try:
                    recovered_run = self.operator.run_summary(run_id)
                except FileNotFoundError:
                    if operation.get("status") == "failed":
                        raise CampaignError(
                            "launch_failed",
                            "campaign launch failed before a trustworthy run was created",
                            category="execution",
                        ) from None
                else:
                    policy = self._policy(campaign_id)
                    with self._campaign_lock(campaign_id):
                        ledger = self._ledger(campaign_id, policy)
                        admission = _ledger_admission(
                            ledger, admission_receipt.admission_id
                        )
                        admission["status"] = (
                            recovered_run.status
                            if recovered_run.status in _TERMINAL_RUN_STATES
                            else "running"
                        )
                        admission["run_id"] = run_id
                        self._write_ledger(campaign_id, ledger)
                        self._write_operation(
                            campaign_id,
                            operation_id,
                            {
                                "schema_version": CAMPAIGN_SCHEMA_VERSION,
                                "operation_id": operation_id,
                                "action": "launch",
                                "input_digest": operation_input,
                                "status": "completed",
                                "run_id": run_id,
                            },
                        )
                        self._event(
                            campaign_id,
                            "run_started",
                            operation_id=operation_id,
                            proposal_id=admission_receipt.proposal_id,
                            admission_id=admission_receipt.admission_id,
                            run_id=run_id,
                        )
                    return self.status(run_id)

            prepared = _prepared_plan_from_dict(admission_receipt.prepared_plan)
            plan = _plan_receipt_from_dict(prepared.plan)
            policy = self._policy(campaign_id)
            self._require_policy_snapshot(policy)
            if admission_receipt.policy_digest != policy.campaign_digest:
                raise CampaignError(
                    "policy_drift",
                    "admission is bound to a different campaign policy",
                    category="policy",
                )
            self._revalidate_plan(plan)
            self._revalidate_prepared_bindings(prepared, plan)
            request = _request_from_safe(plan.request)
            proposal = experiment_proposal_from_dict(plan.proposal)
            experiment = self._proposal_experiment(proposal)
            checks = self.operator.preflight(request, live=True, experiment=experiment)
            failed_checks = [item for item in checks if not item.ok]
            if failed_checks:
                secrets = secrets_from_env(self.operator.env)
                raise CampaignError(
                    "preflight_failed",
                    "campaign live preflight no longer passes",
                    category="execution",
                    retryable=True,
                    details={
                        "checks": [
                            {
                                "name": item.name,
                                "detail": _safe_diagnostic(item.detail, secrets),
                            }
                            for item in failed_checks
                        ]
                    },
                )

            with self._campaign_lock(campaign_id):
                ledger = self._ledger(campaign_id, policy)
                admission = _ledger_admission(ledger, admission_receipt.admission_id)
                if admission.get("status") not in {"admitted", "launching"}:
                    raise CampaignError(
                        "admission_not_launchable",
                        "admission is not in a launchable state",
                        category="execution",
                        details={"status": admission.get("status")},
                    )
                active = self._active_run_ids(ledger)
                if (
                    admission.get("run_id") not in active
                    and len(active) >= policy.limits.max_active_runs
                ):
                    raise CampaignError(
                        "active_run_limit",
                        "campaign already has the maximum number of active runs",
                        category="admission",
                        retryable=True,
                    )
                run_id = str(
                    (operation or {}).get("run_id")
                    or admission.get("run_id")
                    or new_run_id()
                )
                self._write_operation(
                    campaign_id,
                    operation_id,
                    {
                        "schema_version": CAMPAIGN_SCHEMA_VERSION,
                        "operation_id": operation_id,
                        "action": "launch",
                        "input_digest": operation_input,
                        "status": "launching",
                        "run_id": run_id,
                    },
                )
                admission["run_id"] = run_id
                admission["status"] = "launching"
                self._write_ledger(campaign_id, ledger)
                self._event(
                    campaign_id,
                    "run_launching",
                    operation_id=operation_id,
                    proposal_id=admission_receipt.proposal_id,
                    admission_id=admission_receipt.admission_id,
                    run_id=run_id,
                )

            try:
                self.operator.launch(request, experiment=experiment, run_id=run_id)
            except Exception as exc:
                with self._campaign_lock(campaign_id):
                    ledger = self._ledger(campaign_id, policy)
                    admission = _ledger_admission(
                        ledger, admission_receipt.admission_id
                    )
                    admission["status"] = "incident"
                    admission["incident_code"] = "launch_failed"
                    admission["incident_exception_type"] = type(exc).__name__
                    self._write_ledger(campaign_id, ledger)
                    self._write_operation(
                        campaign_id,
                        operation_id,
                        {
                            "schema_version": CAMPAIGN_SCHEMA_VERSION,
                            "operation_id": operation_id,
                            "action": "launch",
                            "input_digest": operation_input,
                            "status": "failed",
                            "run_id": run_id,
                            "error_code": "launch_failed",
                            "exception_type": type(exc).__name__,
                        },
                    )
                    self._event(
                        campaign_id,
                        "run_incident",
                        operation_id=operation_id,
                        proposal_id=admission_receipt.proposal_id,
                        admission_id=admission_receipt.admission_id,
                        run_id=run_id,
                        error=CampaignError(
                            "launch_failed",
                            "campaign launch failed before a trustworthy terminal state",
                            category="execution",
                            details={"exception_type": type(exc).__name__},
                        ),
                    )
                raise CampaignError(
                    "launch_failed",
                    "campaign launch failed before a trustworthy terminal state",
                    category="execution",
                    details={"exception_type": type(exc).__name__},
                ) from exc

            with self._campaign_lock(campaign_id):
                ledger = self._ledger(campaign_id, policy)
                admission = _ledger_admission(ledger, admission_receipt.admission_id)
                admission["status"] = "running"
                self._write_ledger(campaign_id, ledger)
                self._write_operation(
                    campaign_id,
                    operation_id,
                    {
                        "schema_version": CAMPAIGN_SCHEMA_VERSION,
                        "operation_id": operation_id,
                        "action": "launch",
                        "input_digest": operation_input,
                        "status": "completed",
                        "run_id": run_id,
                    },
                )
                self._event(
                    campaign_id,
                    "run_started",
                    operation_id=operation_id,
                    proposal_id=admission_receipt.proposal_id,
                    admission_id=admission_receipt.admission_id,
                    run_id=run_id,
                )
            return self.status(run_id)

    def status(self, campaign_or_run_id: str) -> CampaignStatusV1:
        campaign_id, subject_id = self._resolve_campaign_subject(campaign_or_run_id)
        policy = self._policy(campaign_id)
        ledger = self._ledger(campaign_id, policy)
        blockers: list[str] = []
        try:
            self._require_policy_snapshot(policy, allow_missing=True)
            self._require_source(
                policy, resolve_fugue_source_provenance(self.repo_root)
            )
        except CampaignError as exc:
            blockers.append(f"{exc.code}: {exc}")
        runs: list[dict[str, Any]] = []
        active: list[str] = []
        for admission in ledger["admissions"]:
            run_id = str(admission.get("run_id") or "")
            if not run_id:
                continue
            try:
                run = self.operator.run_summary(run_id)
                run_state = run.status
            except (FileNotFoundError, ValueError):
                run_state = str(admission.get("status") or "unknown")
            if run_state in {"starting", "running", "launching"}:
                active.append(run_id)
            runs.append(
                {
                    "run_id": run_id,
                    "proposal_id": admission.get("proposal_id"),
                    "stage_id": admission.get("stage_id"),
                    "status": run_state,
                    "outcome_id": admission.get("outcome_id"),
                }
            )
        reserved = _reserved_budget(ledger)
        accounted = float(ledger.get("accounted_cost_usd") or 0.0)
        remaining = max(0.0, policy.limits.total_cost_usd - accounted - reserved)
        if active:
            state = "running"
            next_actions = ("status", "events", "cancel")
        elif any(item.get("status") == "incident" for item in ledger["admissions"]):
            state = "blocked"
            next_actions = ("status", "events", "finalize")
            blockers.append("campaign has an unreconciled run incident")
        elif ledger["admissions"] and not all(
            item.get("outcome_id") for item in ledger["admissions"]
        ):
            state = "terminal"
            next_actions = ("status", "events", "finalize")
        elif ledger["admissions"]:
            state = "evidence_ready"
            next_actions = ("catalog", "preview", "prepare", "events")
        else:
            state = "ready"
            next_actions = ("catalog", "preview", "prepare")
        if blockers and state not in {"running", "terminal"}:
            state = "blocked"
        unsigned = CampaignStatusV1(
            schema_version=CAMPAIGN_SCHEMA_VERSION,
            campaign_id=campaign_id,
            subject_id=subject_id,
            state=state,
            policy_digest=policy.campaign_digest,
            active_runs=tuple(active),
            runs=tuple(runs),
            admissions=len(ledger["admissions"]),
            outcomes=sum(bool(item.get("outcome_id")) for item in ledger["admissions"]),
            admitted_cells=sum(
                int(item["cell_count"]) for item in ledger["admissions"]
            ),
            total_cost_usd=policy.limits.total_cost_usd,
            accounted_cost_usd=accounted,
            reserved_cost_usd=reserved,
            remaining_cost_usd=remaining,
            next_actions=next_actions,
            blockers=tuple(blockers),
        )
        return replace(
            unsigned,
            status_digest=_artifact_digest(unsigned.to_dict(), "status_digest"),
        )

    def events(
        self, campaign_id: str, after_sequence: int = 0
    ) -> tuple[CampaignEventV1, ...]:
        policy = self._policy(campaign_id)
        if after_sequence < 0:
            raise CampaignError(
                "invalid_cursor",
                "campaign event cursor must be non-negative",
                category="validation",
            )
        path = self._campaign_dir(policy.id) / "events.jsonl"
        if not path.is_file():
            return ()
        values: list[CampaignEventV1] = []
        previous_digest: str | None = None
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CampaignError(
                    "event_log_corrupt",
                    f"campaign event {number} is invalid JSON",
                    category="evidence",
                ) from exc
            event = _campaign_event_from_dict(raw)
            if event.sequence_number != number:
                raise CampaignError(
                    "event_log_sequence_invalid",
                    f"campaign event {number} is not contiguous",
                    category="evidence",
                )
            if event.previous_event_digest != previous_digest:
                raise CampaignError(
                    "event_log_chain_invalid",
                    f"campaign event {number} does not bind its predecessor",
                    category="evidence",
                )
            previous_digest = event.event_digest
            if event.sequence_number > after_sequence:
                values.append(event)
        return tuple(values)

    def cancel(self, run_id: str, operation_id: str, reason: str) -> CampaignStatusV1:
        operation_id = self._operation_id(operation_id)
        reason = _safe_diagnostic(
            _bounded_text(reason, "cancellation reason", 1000),
            secrets_from_env(self.operator.env),
        )
        campaign_id, _ = self._resolve_campaign_subject(run_id)
        operation_input = stable_digest(
            {"action": "cancel", "run_id": run_id, "reason": reason}
        )
        with (
            self._operation_lock(campaign_id, operation_id),
            self._run_lock(campaign_id, run_id),
        ):
            existing = self._operation(campaign_id, operation_id)
            recovered_status: str | None = None
            if existing is not None:
                self._require_operation_match(
                    existing, "cancel", operation_input, operation_id
                )
                if existing.get("status") == "completed":
                    policy = self._policy(campaign_id)
                    admission = _run_admission(
                        self._ledger(campaign_id, policy), run_id
                    )
                    self._event(
                        campaign_id,
                        "run_cancelled",
                        operation_id=operation_id,
                        proposal_id=str(admission["proposal_id"]),
                        admission_id=str(admission["admission_id"]),
                        run_id=run_id,
                    )
                    return self.status(run_id)
                if existing.get("status") == "cancelled":
                    recovered_status = str(existing.get("terminal_status") or "")
                else:
                    try:
                        current_run = self.operator.run_summary(run_id)
                    except (FileNotFoundError, ValueError):
                        pass
                    else:
                        if current_run.status in _TERMINAL_RUN_STATES:
                            recovered_status = current_run.status
            policy = self._policy(campaign_id)
            ledger = self._ledger(campaign_id, policy)
            admission = _run_admission(ledger, run_id)
            self._write_operation(
                campaign_id,
                operation_id,
                {
                    "schema_version": CAMPAIGN_SCHEMA_VERSION,
                    "operation_id": operation_id,
                    "action": "cancel",
                    "input_digest": operation_input,
                    "status": "cancelling",
                    "run_id": run_id,
                },
            )
            if recovered_status is None:
                try:
                    recovered_status = self.operator.supervisor.cancel(run_id).status
                except Exception as exc:
                    raise CampaignError(
                        "cancellation_failed",
                        "campaign cancellation did not reach a trustworthy terminal state",
                        category="execution",
                        retryable=True,
                        details={"exception_type": type(exc).__name__},
                    ) from exc
                self._write_operation(
                    campaign_id,
                    operation_id,
                    {
                        "schema_version": CAMPAIGN_SCHEMA_VERSION,
                        "operation_id": operation_id,
                        "action": "cancel",
                        "input_digest": operation_input,
                        "status": "cancelled",
                        "run_id": run_id,
                        "terminal_status": recovered_status,
                    },
                )
            with self._campaign_lock(campaign_id):
                ledger = self._ledger(campaign_id, policy)
                admission = _ledger_admission(ledger, str(admission["admission_id"]))
                admission["status"] = (
                    "cancelled_unreconciled"
                    if recovered_status == "cancelled"
                    else recovered_status
                )
                admission["cancellation_reason"] = reason
                self._write_ledger(campaign_id, ledger)
                self._write_operation(
                    campaign_id,
                    operation_id,
                    {
                        "schema_version": CAMPAIGN_SCHEMA_VERSION,
                        "operation_id": operation_id,
                        "action": "cancel",
                        "input_digest": operation_input,
                        "status": "completed",
                        "run_id": run_id,
                    },
                )
                self._event(
                    campaign_id,
                    "run_cancelled",
                    operation_id=operation_id,
                    proposal_id=str(admission["proposal_id"]),
                    admission_id=str(admission["admission_id"]),
                    run_id=run_id,
                )
            return self.status(run_id)

    def finalize(self, run_id: str, operation_id: str) -> OutcomePacketV1:
        operation_id = self._operation_id(operation_id)
        campaign_id, _ = self._resolve_campaign_subject(run_id)
        operation_input = stable_digest({"action": "finalize", "run_id": run_id})
        with (
            self._operation_lock(campaign_id, operation_id),
            self._run_lock(campaign_id, run_id),
        ):
            existing = self._completed_operation(
                campaign_id, operation_id, "finalize", operation_input
            )
            if existing is not None:
                outcome = _outcome_packet_from_dict(self._read_artifact(existing))
                self._event(
                    campaign_id,
                    "evidence_finalized" if outcome.eligible else "evidence_blocked",
                    operation_id=operation_id,
                    proposal_id=outcome.proposal_id,
                    admission_id=outcome.admission_id,
                    run_id=run_id,
                    artifact_type="OutcomePacketV1",
                    artifact_digest=outcome.outcome_digest,
                )
                return outcome
            policy = self._policy(campaign_id)
            self._require_policy_snapshot(policy)
            ledger = self._ledger(campaign_id, policy)
            admission_entry = _run_admission(ledger, run_id)
            existing_outcome_id = str(admission_entry.get("outcome_id") or "")
            if existing_outcome_id:
                outcome = _outcome_packet_from_dict(
                    self._read_receipt(campaign_id, "outcomes", existing_outcome_id)
                )
                relative = (
                    (
                        self._campaign_dir(campaign_id)
                        / "outcomes"
                        / f"{existing_outcome_id}.json"
                    )
                    .relative_to(self.repo_root)
                    .as_posix()
                )
                self._record_operation(
                    campaign_id,
                    operation_id,
                    "finalize",
                    operation_input,
                    relative,
                    outcome.outcome_digest,
                )
                self._event(
                    campaign_id,
                    "evidence_finalized" if outcome.eligible else "evidence_blocked",
                    operation_id=operation_id,
                    proposal_id=outcome.proposal_id,
                    admission_id=outcome.admission_id,
                    run_id=run_id,
                    artifact_type="OutcomePacketV1",
                    artifact_digest=outcome.outcome_digest,
                )
                return outcome
            admission = _admission_receipt_from_dict(
                self._read_receipt(
                    campaign_id,
                    "admissions",
                    str(admission_entry["admission_id"]),
                )
            )
            prepared = _prepared_plan_from_dict(admission.prepared_plan)
            plan = _plan_receipt_from_dict(prepared.plan)
            proposal = experiment_proposal_from_dict(plan.proposal)
            run = self.operator.run_summary(run_id)
            if run.status not in _TERMINAL_RUN_STATES:
                raise CampaignError(
                    "run_not_terminal",
                    "campaign run must be terminal before finalization",
                    category="evidence",
                    retryable=True,
                    details={"status": run.status},
                )
            self._write_operation(
                campaign_id,
                operation_id,
                {
                    "schema_version": CAMPAIGN_SCHEMA_VERSION,
                    "operation_id": operation_id,
                    "action": "finalize",
                    "input_digest": operation_input,
                    "status": "finalizing",
                    "run_id": run_id,
                },
            )
            export_path = (
                self._campaign_dir(campaign_id) / "exports" / f"{run_id}.jsonl"
            )
            try:
                exported = self.operator.export_run(
                    run_id,
                    out=export_path,
                    fetch_weave=any(
                        item.get("execution_kind") == "agent" for item in plan.cells
                    ),
                    to_weave=False,
                )
            except Exception as exc:
                raise CampaignError(
                    "export_failed",
                    "campaign evidence export could not construct trustworthy rows",
                    category="evidence",
                    retryable=True,
                    details={"exception_type": type(exc).__name__},
                ) from exc
            rows = _read_jsonl(exported.path)
            input_lock_path = (
                self.repo_root / ".fugue/runtime" / run_id / "input-lock.json"
            )
            input_lock = (
                _read_json_object(input_lock_path)
                if input_lock_path.is_file()
                else None
            )
            evaluation_lock_path = (
                self.repo_root / ".fugue/runtime" / run_id / "evaluation-assets.json"
            )
            evaluation_lock_digest: str | None = None
            evaluation_lock_failure: str | None = None
            try:
                evaluation_lock = read_evaluation_asset_lock(evaluation_lock_path)
                if evaluation_lock.run_id != run_id:
                    evaluation_lock_failure = (
                        "evaluation asset lock belongs to a different run"
                    )
                else:
                    evaluation_lock_digest = evaluation_lock.lock_sha256
            except (FileNotFoundError, ValueError, json.JSONDecodeError):
                evaluation_lock_failure = "evaluation asset lock is missing or invalid"
            eligibility_failures = _outcome_eligibility_failures(
                run=run,
                rows=rows,
                plan=plan,
                prepared=prepared,
                input_lock=input_lock,
                evaluation_lock_digest=evaluation_lock_digest,
            )
            if evaluation_lock_failure:
                eligibility_failures.append(evaluation_lock_failure)
            analysis_results: tuple[dict[str, Any], ...] = ()
            analysis_failures: list[str] = []
            if proposal.analysis_ids:
                try:
                    analysis_results = self._registered_analyses(
                        proposal.analysis_ids, run_id
                    )
                except Exception as exc:
                    analysis_failures.append(
                        f"registered analysis failed ({type(exc).__name__})"
                    )
            eligibility_failures.extend(analysis_failures)
            accounting = account_prediction_costs(
                rows,
                expected_cells=plan.expected_predictions,
                reserved_cell_cost_usd=admission.reserved_cell_cost_usd,
            )
            scope = policy.evidence_scope
            secrets = secrets_from_env(self.operator.env)
            row_refs = (
                tuple(_safe_prediction_row(row, secrets) for row in rows)
                if scope in {"rows", "traces"}
                else ()
            )
            evidence_refs = (
                tuple(
                    _safe_agent_evidence(row, secrets)
                    for row in rows
                    if row.get("execution_kind") == "agent"
                )
                if scope == "traces"
                else ()
            )
            input_lock_sha = (
                _sha256_path(input_lock_path) if input_lock_path.is_file() else None
            )
            snapshot_sha = (
                str(
                    input_lock.get("snapshot_sha256")
                    or input_lock.get("lock_sha256")
                    or ""
                )
                if input_lock
                else None
            )
            passed = sum(bool(row.get("pass")) for row in rows)
            failed = sum(
                row.get("status") == "failed" or row.get("pass") is False
                for row in rows
            )
            not_applicable = sum(row.get("status") == "not_applicable" for row in rows)
            limitations: list[str] = []
            if proposal.n_attempts == 1:
                limitations.append("one attempt per selected coordinate")
            if accounting.unmeasured_cells:
                limitations.append(
                    f"{accounting.unmeasured_cells} cell cost(s) were conservatively accounted"
                )
            if run.status in {"cancelled", "interrupted"}:
                limitations.append(f"run ended as {run.status}")
            if eligibility_failures:
                limitations.append("evidence is not eligible for stage progression")
            export_sha = _sha256_path(exported.path)
            outcome_id = stable_digest(
                {
                    "schema_version": CAMPAIGN_SCHEMA_VERSION,
                    "campaign_id": campaign_id,
                    "admission_id": admission.admission_id,
                    "run_id": run_id,
                    "export_sha256": export_sha,
                }
            )
            outcome_path = (
                self._campaign_dir(campaign_id) / "outcomes" / f"{outcome_id}.json"
            )
            if outcome_path.is_file():
                outcome = _outcome_packet_from_dict(_read_json_object(outcome_path))
                if (
                    outcome.run_id != run_id
                    or outcome.admission_id != admission.admission_id
                    or outcome.export_sha256 != export_sha
                ):
                    raise CampaignError(
                        "outcome_conflict",
                        "stored campaign outcome does not match the run export",
                        category="evidence",
                    )
            else:
                unsigned = OutcomePacketV1(
                    schema_version=CAMPAIGN_SCHEMA_VERSION,
                    outcome_id=outcome_id,
                    campaign_id=campaign_id,
                    proposal_id=proposal.proposal_id,
                    stage_id=proposal.stage_id,
                    admission_id=admission.admission_id,
                    run_id=run_id,
                    run_status=run.status,
                    expected_predictions=plan.expected_predictions,
                    observed_predictions=len(rows),
                    passed=passed,
                    failed=failed,
                    not_applicable=not_applicable,
                    eligible=not eligibility_failures,
                    eligibility_failures=tuple(eligibility_failures),
                    limitations=tuple(limitations),
                    observed_cost_usd=accounting.observed_cost_usd,
                    accounted_cost_usd=accounting.accounted_cost_usd,
                    measured_cost_cells=accounting.measured_cells,
                    unmeasured_cost_cells=accounting.unmeasured_cells,
                    maximum_measured_cell_cost_usd=(
                        accounting.maximum_measured_cell_cost_usd
                    ),
                    input_lock_sha256=input_lock_sha,
                    run_snapshot_sha256=snapshot_sha,
                    export_sha256=export_sha,
                    export_path=exported.path.relative_to(self.repo_root).as_posix(),
                    row_refs=row_refs,
                    evidence_refs=evidence_refs,
                    analysis_results=analysis_results,
                    metrics=_outcome_metrics(rows, passed),
                    finalized_at=_now(),
                )
                outcome = replace(
                    unsigned,
                    outcome_digest=_artifact_digest(
                        unsigned.to_dict(), "outcome_digest"
                    ),
                )
            with self._campaign_lock(campaign_id):
                ledger = self._ledger(campaign_id, policy)
                entry = _ledger_admission(ledger, admission.admission_id)
                if (
                    entry.get("outcome_id")
                    and entry["outcome_id"] != outcome.outcome_id
                ):
                    raise CampaignError(
                        "outcome_conflict",
                        "run was already finalized with different evidence",
                        category="evidence",
                    )
                relative = self._write_receipt(
                    "outcomes", outcome.outcome_id, outcome.to_dict()
                )
                entry["status"] = "evidence_ready" if outcome.eligible else "blocked"
                entry["outcome_id"] = outcome.outcome_id
                entry["actual_cost_usd"] = outcome.accounted_cost_usd
                ledger["accounted_cost_usd"] = sum(
                    float(item.get("actual_cost_usd") or 0.0)
                    for item in ledger["admissions"]
                    if item.get("outcome_id")
                )
                self._write_ledger(campaign_id, ledger)
                self._record_operation(
                    campaign_id,
                    operation_id,
                    "finalize",
                    operation_input,
                    relative,
                    outcome.outcome_digest,
                )
                self._event(
                    campaign_id,
                    "evidence_finalized" if outcome.eligible else "evidence_blocked",
                    operation_id=operation_id,
                    proposal_id=proposal.proposal_id,
                    admission_id=admission.admission_id,
                    run_id=run_id,
                    artifact_type="OutcomePacketV1",
                    artifact_digest=outcome.outcome_digest,
                )
            return outcome

    def _registered_analyses(
        self, analysis_ids: Sequence[str], run_id: str
    ) -> tuple[dict[str, Any], ...]:
        results: list[dict[str, Any]] = []
        for analysis_id in analysis_ids:
            spec = get_analysis(analysis_id, self.repo_root)
            local = replace(
                spec,
                filters={**spec.filters, "run_id": run_id},
                source="local",
                include_artifacts=False,
            )
            preview = self.operator.prepare_analysis(local)
            if preview.scope.rows < 1:
                raise ValueError(f"analysis {analysis_id} resolved no rows")
            results.append(
                {
                    "analysis_id": analysis_id,
                    "snapshot_id": preview.snapshot.id,
                    "snapshot_digest": preview.snapshot.digest,
                    "rows": preview.scope.rows,
                    "aggregates": _json_value(preview.aggregates),
                    "selection": (
                        preview.selection.to_dict()
                        if preview.selection is not None
                        else None
                    ),
                }
            )
        return tuple(results)

    def _policy(self, campaign_id: str) -> ResearchCampaignSpecV1:
        try:
            return get_campaign(campaign_id, self.repo_root)
        except (FileNotFoundError, ValueError) as exc:
            raise CampaignError(
                "invalid_campaign",
                str(exc),
                category="validation",
            ) from exc

    def _stage(
        self, policy: ResearchCampaignSpecV1, stage_id: str
    ) -> CampaignStagePolicyV1:
        match = next((item for item in policy.stages if item.id == stage_id), None)
        if match is None:
            raise CampaignError(
                "stage_not_allowed",
                f"campaign does not allow stage {stage_id!r}",
                category="policy",
            )
        return match

    def _require_source(
        self, policy: ResearchCampaignSpecV1, source: Mapping[str, Any]
    ) -> None:
        if policy.require_clean_source and (
            source.get("kind") != "git" or bool(source.get("dirty"))
        ):
            raise CampaignError(
                "source_not_clean",
                "campaign requires a clean Git source state",
                category="policy",
                retryable=True,
            )

    def _validate_proposal(
        self,
        policy: ResearchCampaignSpecV1,
        catalog: CampaignCatalogSnapshotV1,
        proposal: ExperimentProposalV1,
    ) -> None:
        if proposal.catalog_digest != catalog.catalog_digest:
            raise CampaignError(
                "catalog_drift",
                "proposal catalog digest does not match the current catalog",
                category="validation",
            )
        self._stage(policy, proposal.stage_id)
        _require_allowed(
            "experiment", [proposal.experiment_id], policy.allowed_experiments
        )
        _require_allowed("model", [proposal.model], policy.allowed_models)
        _require_allowed("harness", proposal.harnesses, policy.allowed_harnesses)
        _require_allowed("workload", proposal.workloads, policy.allowed_workloads)
        _require_allowed("variant", proposal.variants, policy.allowed_variants)
        _require_allowed(
            "context system",
            proposal.context_systems,
            policy.allowed_context_systems,
        )
        _require_allowed("analysis", proposal.analysis_ids, policy.allowed_analyses)
        _require_allowed(
            "trace content", [proposal.trace_content], policy.allowed_trace_content
        )
        if proposal.n_attempts > policy.limits.max_attempts_per_cell:
            raise CampaignError(
                "attempt_limit",
                "proposal exceeds the campaign attempt limit",
                category="policy",
            )
        if proposal.n_concurrent > policy.limits.max_concurrent:
            raise CampaignError(
                "concurrency_limit",
                "proposal exceeds the campaign concurrency limit",
                category="policy",
            )
        if proposal.parent_outcome_id and not proposal.decision_rationale:
            raise CampaignError(
                "missing_decision_rationale",
                "adaptive proposals must explain why the next experiment was selected",
                category="validation",
            )
        if proposal.task_suite_digest:
            authoring = self._task_authoring_policy(policy)
            lock = read_task_suite_lock(
                self.repo_root, policy.id, proposal.task_suite_digest
            )
            if lock.policy_digest != policy.campaign_digest:
                raise CampaignError(
                    "task_suite_policy_drift",
                    "task suite lock is bound to a different campaign policy",
                    category="policy",
                )
            if lock.catalog_digest != catalog.catalog_digest:
                raise CampaignError(
                    "task_suite_catalog_drift",
                    "task suite lock is bound to a different campaign catalog",
                    category="validation",
                )
            if lock.stage_id != proposal.stage_id:
                raise CampaignError(
                    "task_suite_stage_mismatch",
                    "task suite lock and proposal use different campaign stages",
                    category="policy",
                )
            if proposal.n_tasks not in {None, lock.task_count}:
                raise CampaignError(
                    "task_suite_truncation",
                    "authored task suites must run their exact locked task set",
                    category="validation",
                )
            if proposal.preset_id is not None:
                raise CampaignError(
                    "task_suite_preset_unsupported",
                    "authored task suites cannot inherit a registered preset",
                    category="validation",
                )
            if proposal.workloads not in {(), ("harbor",)}:
                raise CampaignError(
                    "task_suite_workload_unsupported",
                    "authored task suites execute through the Harbor workload",
                    category="validation",
                )
            if lock.parent_outcome_id != proposal.parent_outcome_id:
                if lock.parent_outcome_id is not None:
                    raise CampaignError(
                        "task_suite_parent_mismatch",
                        "adaptive task suite and experiment proposal have different parents",
                        category="policy",
                    )
            if "holdout" in lock.partitions and lock.parent_outcome_id is not None:
                raise CampaignError(
                    "adaptive_holdout_forbidden",
                    "holdout tasks must be locked before observing a parent outcome",
                    category="policy",
                )
            if lock.parent_outcome_id and not authoring.adaptive_discovery:
                raise CampaignError(
                    "adaptive_task_authoring_forbidden",
                    "campaign policy does not allow adaptive discovery tasks",
                    category="policy",
                )

    def _validate_resolved_plan(
        self,
        policy: ResearchCampaignSpecV1,
        proposal: ExperimentProposalV1,
        plan: ResolvedRunPlan,
    ) -> None:
        _require_allowed(
            "resolved harness",
            [cell.harness for cell in plan.cells],
            policy.allowed_harnesses,
        )
        _require_allowed(
            "resolved workload",
            [cell.workload_id for cell in plan.cells],
            policy.allowed_workloads,
        )
        _require_allowed(
            "resolved variant",
            [cell.variant_id for cell in plan.cells],
            policy.allowed_variants,
        )
        _require_allowed(
            "resolved context system",
            [cell.context_system_id for cell in plan.cells],
            policy.allowed_context_systems,
        )
        _require_allowed(
            "resolved model", [cell.model for cell in plan.cells], policy.allowed_models
        )
        if plan.max_workers != proposal.n_concurrent:
            raise CampaignError(
                "concurrency_drift",
                "resolved concurrency differs from the proposal",
                category="validation",
            )
        if any(cell.n_attempts != proposal.n_attempts for cell in plan.cells):
            raise CampaignError(
                "attempt_drift",
                "resolved attempts differ from the proposal",
                category="validation",
            )
        if not plan.cells:
            raise CampaignError(
                "empty_plan",
                "campaign proposal resolved no cells",
                category="validation",
            )
        if proposal.task_suite_digest:
            lock = read_task_suite_lock(
                self.repo_root, policy.id, proposal.task_suite_digest
            )
            observed_tasks = {cell.task_id for cell in plan.cells}
            if observed_tasks != set(lock.task_ids):
                raise CampaignError(
                    "task_suite_coordinate_drift",
                    "resolved plan tasks differ from the locked task suite",
                    category="validation",
                )

    def _proposal_experiment(
        self, proposal: ExperimentProposalV1
    ) -> ExperimentSpec | None:
        if not proposal.task_suite_digest:
            return None
        lock = read_task_suite_lock(
            self.repo_root, proposal.campaign_id, proposal.task_suite_digest
        )
        base = get_experiment(proposal.experiment_id, self.repo_root)
        return replace(
            base,
            manifest=Path(lock.manifest_path),
            run_name=f"{proposal.campaign_id}-{proposal.proposal_id}",
            tags=[*base.tags, f"task-suite:{lock.suite_digest}"],
            n_tasks=lock.task_count,
            workloads=[],
            presets=[],
            default_preset=None,
            evaluation_generation=None,
            integrations=[
                IntegrationSelection(id=integration_id)
                for integration_id in lock.integration_ids
            ],
        )

    def _request(self, proposal: ExperimentProposalV1) -> ExperimentRequest:
        task_lock = (
            read_task_suite_lock(
                self.repo_root, proposal.campaign_id, proposal.task_suite_digest
            )
            if proposal.task_suite_digest
            else None
        )
        return ExperimentRequest(
            experiment_id=proposal.experiment_id,
            preset=None if task_lock else proposal.preset_id,
            workloads=proposal.workloads,
            harnesses=proposal.harnesses,
            systems=proposal.context_systems,
            variants=proposal.variants,
            model=proposal.model,
            n_attempts=proposal.n_attempts,
            n_tasks=task_lock.task_count if task_lock else proposal.n_tasks,
            n_concurrent=proposal.n_concurrent,
            run_name=f"{proposal.campaign_id}-{proposal.proposal_id}",
            tags=(
                f"campaign:{proposal.campaign_id}",
                f"proposal:{proposal.proposal_id}",
                f"stage:{proposal.stage_id}",
            ),
            trace_content=proposal.trace_content,
            cohort_id=f"{proposal.campaign_id}:{proposal.proposal_id}",
        )

    def _task_authoring_policy(
        self, policy: ResearchCampaignSpecV1
    ) -> TaskAuthoringPolicyV1:
        if policy.task_authoring is None:
            raise CampaignError(
                "task_authoring_disabled",
                "campaign policy does not enable authored tasks",
                category="policy",
            )
        return policy.task_authoring

    def _require_run_task_suite(
        self,
        campaign_id: str,
        run_id: str,
        lock: TaskSuiteLockV1,
    ) -> None:
        policy = self._policy(campaign_id)
        ledger = self._ledger(campaign_id, policy)
        admission_entry = _run_admission(ledger, run_id)
        admission = _admission_receipt_from_dict(
            self._read_receipt(
                campaign_id,
                "admissions",
                str(admission_entry["admission_id"]),
            )
        )
        prepared = _prepared_plan_from_dict(admission.prepared_plan)
        plan = _plan_receipt_from_dict(prepared.plan)
        proposal = experiment_proposal_from_dict(plan.proposal)
        if proposal.task_suite_digest != lock.suite_digest:
            raise CampaignError(
                "task_suite_run_mismatch",
                "run was not planned from the requested task suite lock",
                category="evidence",
            )

    def _task_evaluation(
        self,
        campaign_id: str,
        run_id: str,
        evaluation_digest: str | None,
    ) -> TaskEvaluationV1:
        root = self._campaign_dir(campaign_id) / "task-evaluations"
        if evaluation_digest:
            return task_evaluation_from_dict(
                self._read_receipt(campaign_id, "task-evaluations", evaluation_digest)
            )
        matches: list[TaskEvaluationV1] = []
        if root.is_dir():
            for path in sorted(root.glob("*.json")):
                value = task_evaluation_from_dict(_read_json_object(path))
                if value.run_id == run_id:
                    matches.append(value)
        if len(matches) != 1:
            raise CampaignError(
                "task_evaluation_ambiguous",
                "specify an evaluation digest when a run has zero or multiple scoring revisions",
                category="validation",
            )
        return matches[0]

    def _remaining_task_budget(
        self, campaign_id: str, policy: ResearchCampaignSpecV1
    ) -> float:
        with self._campaign_lock(campaign_id):
            return _remaining_budget(self._ledger(campaign_id, policy), policy)

    def _account_task_evaluation(
        self,
        campaign_id: str,
        policy: ResearchCampaignSpecV1,
        run_id: str,
        evaluation: TaskEvaluationV1,
    ) -> None:
        with self._campaign_lock(campaign_id):
            ledger = self._ledger(campaign_id, policy)
            admission = _run_admission(ledger, run_id)
            evaluations = admission.setdefault("task_evaluations", [])
            if not isinstance(evaluations, list):
                raise CampaignError(
                    "ledger_corrupt",
                    "campaign task evaluations must be a list",
                    category="evidence",
                )
            existing = next(
                (
                    item
                    for item in evaluations
                    if item.get("evaluation_digest") == evaluation.evaluation_digest
                ),
                None,
            )
            if existing is None:
                evaluations.append(
                    {
                        "evaluation_digest": evaluation.evaluation_digest,
                        "accounted_cost_usd": evaluation.accounted_cost_usd,
                    }
                )
            agent_cost = sum(
                float(item.get("actual_cost_usd") or 0.0)
                for item in ledger["admissions"]
                if item.get("outcome_id")
            )
            evaluation_cost = sum(
                float(result.get("accounted_cost_usd") or 0.0)
                for item in ledger["admissions"]
                for result in item.get("task_evaluations") or []
                if isinstance(result, dict)
            )
            ledger["accounted_cost_usd"] = agent_cost + evaluation_cost
            self._write_ledger(campaign_id, ledger)

    def _verify_plan_receipt(self, receipt: PlanReceiptV1) -> None:
        _verify_artifact(receipt.to_dict(), "plan_digest", "plan receipt")
        proposal = experiment_proposal_from_dict(receipt.proposal)
        if proposal.campaign_id != receipt.campaign_id:
            raise CampaignError(
                "artifact_identity_mismatch",
                "plan campaign does not match its proposal",
                category="validation",
            )

    def _verify_prepared_plan(self, receipt: PreparedPlanV1) -> None:
        _verify_artifact(receipt.to_dict(), "prepared_plan_digest", "prepared plan")
        plan = _plan_receipt_from_dict(receipt.plan)
        if plan.plan_digest != receipt.plan_digest:
            raise CampaignError(
                "artifact_identity_mismatch",
                "prepared plan does not bind its plan receipt",
                category="validation",
            )
        if receipt.component_digests != plan.component_digests:
            raise CampaignError(
                "artifact_identity_mismatch",
                "prepared plan component locks do not match its plan receipt",
                category="validation",
            )

    def _revalidate_prepared_bindings(
        self, prepared: PreparedPlanV1, plan: PlanReceiptV1
    ) -> None:
        current_routes = _prepared_route_locks(plan.cells, self.operator.env)
        if current_routes != prepared.route_locks:
            raise CampaignError(
                "route_drift",
                "current model routes differ from the prepared route locks",
                category="preparation",
            )
        integrations = {
            key.removeprefix("integration:"): value
            for key, value in plan.component_digests.items()
            if key.startswith("integration:")
        }
        if integrations != prepared.integration_locks:
            raise CampaignError(
                "integration_drift",
                "current integration locks differ from the prepared plan",
                category="preparation",
            )

    def _verify_admission(self, receipt: AdmissionReceiptV1) -> None:
        _verify_artifact(receipt.to_dict(), "admission_digest", "admission receipt")
        prepared = _prepared_plan_from_dict(receipt.prepared_plan)
        if prepared.prepared_plan_digest != receipt.prepared_plan_digest:
            raise CampaignError(
                "artifact_identity_mismatch",
                "admission does not bind its prepared plan",
                category="validation",
            )

    def _revalidate_plan(self, receipt: PlanReceiptV1) -> PlanReceiptV1:
        self._verify_plan_receipt(receipt)
        current = self.preview(experiment_proposal_from_dict(receipt.proposal))
        if current.plan_digest != receipt.plan_digest:
            raise CampaignError(
                "plan_drift",
                "current resolved plan differs from the immutable plan receipt",
                category="validation",
            )
        return current

    def _eligible_parent(
        self,
        policy: ResearchCampaignSpecV1,
        proposal: ExperimentProposalV1,
        stage: CampaignStagePolicyV1,
    ) -> dict[str, Any] | None:
        if not stage.require_eligible_parent:
            if proposal.parent_outcome_id:
                return self._outcome(
                    campaign_id=policy.id, outcome_id=proposal.parent_outcome_id
                )
            return None
        if not proposal.parent_outcome_id:
            raise CampaignError(
                "parent_outcome_required",
                f"campaign stage {stage.id!r} requires an eligible parent outcome",
                category="admission",
            )
        parent = self._outcome(
            campaign_id=policy.id, outcome_id=proposal.parent_outcome_id
        )
        if not parent.get("eligible"):
            raise CampaignError(
                "parent_outcome_ineligible",
                "parent outcome is not eligible for stage progression",
                category="admission",
            )
        if parent.get("stage_id") not in stage.predecessors:
            raise CampaignError(
                "stage_predecessor_mismatch",
                "parent outcome does not come from an allowed predecessor stage",
                category="admission",
            )
        return parent

    def _outcome(self, *, campaign_id: str, outcome_id: str) -> dict[str, Any]:
        try:
            value = self._read_receipt(campaign_id, "outcomes", outcome_id)
        except FileNotFoundError as exc:
            raise CampaignError(
                "parent_outcome_missing",
                f"campaign outcome not found: {outcome_id}",
                category="admission",
            ) from exc
        _outcome_packet_from_dict(value)
        return value

    def _operation_id(self, value: str) -> str:
        try:
            return validate_id(value, kind="operation id")
        except ValueError as exc:
            raise CampaignError(
                "invalid_operation_id", str(exc), category="validation"
            ) from exc

    def _campaign_dir(self, campaign_id: str) -> Path:
        return self._store.campaign_dir(campaign_id)

    def _campaign_lock(self, campaign_id: str) -> FileLock:
        return self._store.campaign_lock(campaign_id)

    def _operation_lock(self, campaign_id: str, operation_id: str) -> FileLock:
        return self._store.operation_lock(campaign_id, operation_id)

    def _run_lock(self, campaign_id: str, run_id: str) -> FileLock:
        return self._store.run_lock(campaign_id, run_id)

    def _initialize_campaign(self, policy: ResearchCampaignSpecV1) -> None:
        with self._campaign_lock(policy.id):
            path = self._campaign_dir(policy.id) / "policy.json"
            ledger_path = self._campaign_dir(policy.id) / "ledger.json"
            if path.exists():
                stored = _read_json_object(path)
                stored_policy = campaign_spec_from_dict(stored, item_id=policy.id)
                if stored_policy.id != policy.id:
                    raise CampaignError(
                        "policy_identity_mismatch",
                        "stored campaign policy has a different identity",
                        category="policy",
                    )
                if stored_policy.campaign_digest != policy.campaign_digest:
                    ledger = (
                        _read_json_object(ledger_path)
                        if ledger_path.exists()
                        else _new_ledger(stored_policy)
                    )
                    _validate_ledger(ledger, stored_policy)
                    admissions = ledger.get("admissions")
                    assert isinstance(admissions, list)
                    if admissions or float(ledger.get("accounted_cost_usd") or 0) != 0:
                        raise CampaignError(
                            "policy_drift",
                            "checked-in campaign policy changed after first admission",
                            category="policy",
                        )
                    self._store.write_json(path, policy.to_dict())
                    self._write_ledger(policy.id, _new_ledger(policy))
            else:
                self._store.write_json(path, policy.to_dict())
            if not ledger_path.exists():
                self._write_ledger(policy.id, _new_ledger(policy))

    def _require_policy_snapshot(
        self, policy: ResearchCampaignSpecV1, *, allow_missing: bool = False
    ) -> None:
        path = self._campaign_dir(policy.id) / "policy.json"
        if not path.is_file():
            if allow_missing:
                return
            raise CampaignError(
                "campaign_not_initialized",
                "campaign must be prepared before admission",
                category="admission",
            )
        stored = _read_json_object(path)
        stored_policy = campaign_spec_from_dict(stored, item_id=policy.id)
        if (
            stored_policy.id != policy.id
            or stored_policy.campaign_digest != policy.campaign_digest
        ):
            raise CampaignError(
                "policy_drift",
                "checked-in campaign policy changed after campaign initialization",
                category="policy",
            )

    def _ledger(
        self, campaign_id: str, policy: ResearchCampaignSpecV1
    ) -> dict[str, Any]:
        path = self._campaign_dir(campaign_id) / "ledger.json"
        if not path.is_file():
            return _new_ledger(policy)
        ledger = _read_json_object(path)
        _validate_ledger(ledger, policy)
        return ledger

    def _write_ledger(self, campaign_id: str, ledger: Mapping[str, Any]) -> None:
        self._store.write_json(self._campaign_dir(campaign_id) / "ledger.json", ledger)

    def _write_receipt(self, kind: str, identity: str, value: Mapping[str, Any]) -> str:
        campaign_id = validate_id(value.get("campaign_id") or "", kind="campaign id")
        if kind not in _RECEIPT_KINDS:
            raise ValueError(f"unknown campaign receipt kind: {kind}")
        if not _DIGEST_RE.fullmatch(identity):
            raise ValueError(f"invalid campaign receipt identity: {identity}")
        path = self._campaign_dir(campaign_id) / kind / f"{identity}.json"
        if path.is_file():
            existing = _read_json_object(path)
            if existing != _json_value(value):
                raise CampaignError(
                    "artifact_conflict",
                    "campaign artifact already exists with different content",
                    category="evidence",
                )
        else:
            self._store.write_json(path, value)
        return path.relative_to(self.repo_root).as_posix()

    def _read_receipt(
        self, campaign_id: str, kind: str, identity: str
    ) -> dict[str, Any]:
        if kind not in _RECEIPT_KINDS:
            raise ValueError(f"unknown campaign receipt kind: {kind}")
        if not _DIGEST_RE.fullmatch(identity):
            raise ValueError(f"invalid campaign receipt identity: {identity}")
        return _read_json_object(
            self._campaign_dir(campaign_id) / kind / f"{identity}.json"
        )

    def _read_artifact(self, relative: str) -> dict[str, Any]:
        path = (self.repo_root / relative).resolve()
        root = (self.repo_root / CAMPAIGN_RUNTIME_DIR).resolve()
        if not path.is_relative_to(root):
            raise CampaignError(
                "artifact_path_unsafe",
                "operation artifact is outside campaign runtime storage",
                category="evidence",
            )
        return _read_json_object(path)

    def _operation(self, campaign_id: str, operation_id: str) -> dict[str, Any] | None:
        path = self._campaign_dir(campaign_id) / "operations" / f"{operation_id}.json"
        return _read_json_object(path) if path.is_file() else None

    def _completed_operation(
        self,
        campaign_id: str,
        operation_id: str,
        action: str,
        input_digest: str,
    ) -> str | None:
        operation = self._operation(campaign_id, operation_id)
        if operation is None:
            return None
        self._require_operation_match(operation, action, input_digest, operation_id)
        if operation.get("status") != "completed":
            return None
        artifact = str(operation.get("artifact_path") or "")
        if not artifact:
            raise CampaignError(
                "operation_corrupt",
                "completed operation is missing its artifact",
                category="evidence",
            )
        return artifact

    def _require_operation_match(
        self,
        operation: Mapping[str, Any],
        action: str,
        input_digest: str,
        operation_id: str,
    ) -> None:
        if (
            operation.get("action") != action
            or operation.get("input_digest") != input_digest
        ):
            raise CampaignError(
                "operation_conflict",
                f"operation id {operation_id!r} was already used for different input",
                category="validation",
            )

    def _record_operation(
        self,
        campaign_id: str,
        operation_id: str,
        action: str,
        input_digest: str,
        artifact_path: str,
        artifact_digest: str,
    ) -> None:
        self._write_operation(
            campaign_id,
            operation_id,
            {
                "schema_version": CAMPAIGN_SCHEMA_VERSION,
                "operation_id": operation_id,
                "action": action,
                "input_digest": input_digest,
                "status": "completed",
                "artifact_path": artifact_path,
                "artifact_digest": artifact_digest,
            },
        )

    def _write_operation(
        self, campaign_id: str, operation_id: str, value: Mapping[str, Any]
    ) -> None:
        if str(value.get("action")) not in _IDEMPOTENT_ACTIONS:
            raise ValueError("unknown campaign operation action")
        self._store.write_json(
            self._campaign_dir(campaign_id) / "operations" / f"{operation_id}.json",
            value,
        )

    def _event(
        self,
        campaign_id: str,
        event: str,
        *,
        operation_id: str | None = None,
        proposal_id: str | None = None,
        admission_id: str | None = None,
        run_id: str | None = None,
        artifact_type: str | None = None,
        artifact_digest: str | None = None,
        error: CampaignError | None = None,
    ) -> None:
        path = self._campaign_dir(campaign_id) / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        error_value = error.to_dict() if error else None
        event_id = stable_digest(
            {
                "schema_version": CAMPAIGN_SCHEMA_VERSION,
                "campaign_id": campaign_id,
                "event": event,
                "operation_id": operation_id,
                "proposal_id": proposal_id,
                "admission_id": admission_id,
                "run_id": run_id,
                "artifact_type": artifact_type,
                "artifact_digest": artifact_digest,
                "error_code": error.code if error else None,
            }
        )
        index_path = path.parent / "event-index" / f"{event_id}.json"
        with FileLock((path.parent / ".events.lock").as_posix()):
            recovering = False
            if index_path.is_file():
                marker = _read_json_object(index_path)
                if marker.get("event_id") != event_id:
                    raise CampaignError(
                        "event_index_corrupt",
                        "campaign event index does not match its identity",
                        category="evidence",
                    )
                if marker.get("status") == "completed":
                    return
                if marker.get("status") != "pending":
                    raise CampaignError(
                        "event_index_corrupt",
                        "campaign event index has an invalid state",
                        category="evidence",
                    )
                recovering = True
            else:
                self._store.write_json(
                    index_path,
                    {"event_id": event_id, "status": "pending"},
                )
            sequence = 1
            previous_digest: str | None = None
            if path.is_file():
                previous = _read_last_json_object(path)
                if previous is not None:
                    previous_event = _campaign_event_from_dict(previous)
                    if previous_event.event_id == event_id:
                        self._store.write_json(
                            index_path,
                            {
                                "event_id": event_id,
                                "status": "completed",
                                "sequence_number": previous_event.sequence_number,
                                "event_digest": previous_event.event_digest,
                            },
                        )
                        return
                    if recovering and _event_id_in_log(path, event_id):
                        self._store.write_json(
                            index_path,
                            {"event_id": event_id, "status": "completed"},
                        )
                        return
                    sequence = previous_event.sequence_number + 1
                    previous_digest = previous_event.event_digest
            unsigned = CampaignEventV1(
                schema_version=CAMPAIGN_SCHEMA_VERSION,
                sequence_number=sequence,
                event_id=event_id,
                campaign_id=campaign_id,
                event=event,
                recorded_at=_now(),
                operation_id=operation_id,
                proposal_id=proposal_id,
                admission_id=admission_id,
                run_id=run_id,
                artifact_type=artifact_type,
                artifact_digest=artifact_digest,
                error=error_value,
                previous_event_digest=previous_digest,
            )
            record = replace(
                unsigned,
                event_digest=_artifact_digest(unsigned.to_dict(), "event_digest"),
            ).to_dict()
            self._store.append_json(path, record)
            self._store.write_json(
                index_path,
                {
                    "event_id": event_id,
                    "status": "completed",
                    "sequence_number": sequence,
                    "event_digest": record["event_digest"],
                },
            )

    def _active_run_ids(self, ledger: Mapping[str, Any]) -> list[str]:
        active: list[str] = []
        for item in ledger.get("admissions") or []:
            run_id = str(item.get("run_id") or "")
            if not run_id:
                continue
            try:
                state = self.operator.run_summary(run_id).status
            except (FileNotFoundError, ValueError):
                state = str(item.get("status") or "")
            if state in {"launching", "starting", "running"}:
                active.append(run_id)
        return active

    def _resolve_campaign_subject(self, value: str) -> tuple[str, str]:
        value = validate_id(value, kind="campaign or run id")
        campaign_path = self.repo_root / CAMPAIGNS_DIR / f"{value}.yaml"
        if campaign_path.is_file():
            return value, value
        root = self.repo_root / CAMPAIGN_RUNTIME_DIR
        if root.is_dir():
            for path in sorted(root.glob("*/ledger.json")):
                ledger = _read_json_object(path)
                if any(
                    item.get("run_id") == value
                    for item in ledger.get("admissions") or []
                ):
                    return path.parent.name, value
        raise CampaignError(
            "campaign_subject_missing",
            f"campaign or campaign run not found: {value}",
            category="validation",
        )


def _stage_policy(raw: Any) -> CampaignStagePolicyV1:
    value = _mapping(raw, "campaign stage")
    _reject_unknown(
        value,
        {
            "id",
            "predecessors",
            "max_proposals",
            "max_cells",
            "require_eligible_parent",
            "required_evidence",
        },
        "campaign stage",
    )
    evidence = _text_tuple(
        value.get("required_evidence") or sorted(_EVIDENCE_REQUIREMENTS),
        "stage evidence requirement",
    )
    invalid = sorted(set(evidence) - _EVIDENCE_REQUIREMENTS)
    if invalid:
        raise ValueError(f"unknown stage evidence requirement(s): {', '.join(invalid)}")
    return CampaignStagePolicyV1(
        id=validate_id(value.get("id") or "", kind="campaign stage id"),
        predecessors=_id_tuple(
            value.get("predecessors"), "stage predecessor", allow_empty=True
        ),
        max_proposals=_positive_int(value.get("max_proposals"), "stage max_proposals"),
        max_cells=_positive_int(value.get("max_cells"), "stage max_cells"),
        require_eligible_parent=bool(value.get("require_eligible_parent", False)),
        required_evidence=evidence,
    )


def _validate_stages(stages: Sequence[CampaignStagePolicyV1]) -> None:
    if not stages:
        raise ValueError("campaign must declare at least one stage")
    ids = [item.id for item in stages]
    if len(set(ids)) != len(ids):
        raise ValueError("campaign stage ids must be unique")
    known = set(ids)
    for index, stage in enumerate(stages):
        unknown = sorted(set(stage.predecessors) - known)
        if unknown:
            raise ValueError(
                f"stage {stage.id} has unknown predecessor(s): {', '.join(unknown)}"
            )
        if stage.id in stage.predecessors:
            raise ValueError(f"stage {stage.id} cannot depend on itself")
        if index == 0 and stage.require_eligible_parent:
            raise ValueError("the first campaign stage cannot require a parent outcome")
        if stage.require_eligible_parent and not stage.predecessors:
            raise ValueError(
                f"stage {stage.id} requires a parent but declares no predecessors"
            )
    graph = {stage.id: set(stage.predecessors) for stage in stages}
    remaining = set(graph)
    resolved: set[str] = set()
    while remaining:
        ready = {item for item in remaining if graph[item] <= resolved}
        if not ready:
            raise ValueError("campaign stage graph contains a cycle")
        resolved.update(ready)
        remaining -= ready


def _safe_experiment_record(
    experiment: ExperimentSpec, config_digest: str
) -> dict[str, Any]:
    return {
        "id": experiment.id,
        "title": experiment.title,
        "description": experiment.description,
        "model": experiment.model,
        "harnesses": list(experiment.harnesses),
        "variants": [
            {
                "id": item.id,
                "label": item.label,
                "prompt_id": item.prompt_id,
                "skills": list(item.skills),
                "context_system_id": item.context.system_id,
                "context_delivery": item.context.delivery,
                "integrations": [value.id for value in item.integrations],
                "enabled": item.enabled,
            }
            for item in experiment.variants
        ],
        "workloads": [
            {
                "id": item.id,
                "runner": item.runner,
                "systems": list(item.systems),
                "variants": list(item.variants),
                "required_capabilities": list(item.required_capabilities),
            }
            for item in experiment.workloads
        ],
        "presets": [
            {
                "id": item.id,
                "workloads": list(item.workloads),
                "systems": list(item.systems),
                "harnesses": list(item.harnesses),
                "n_tasks": item.n_tasks,
                "n_attempts": item.n_attempts,
                "n_concurrent": item.n_concurrent,
                "selection_lock_required": item.selection_lock_required,
            }
            for item in experiment.presets
        ],
        "default_preset": experiment.default_preset,
        "trace_content": experiment.trace_content,
        "sha256": config_digest,
    }


def _safe_diagnostic(value: Any, secrets: Sequence[str]) -> str:
    """Return a bounded, single-line diagnostic with known secrets removed."""

    redacted = str(redact_value(str(value or ""), secrets=secrets))
    normalized = " ".join(redacted.replace("\x00", " ").split())
    return normalized[:1000]


def _prepared_route_locks(
    cells: Sequence[Mapping[str, Any]], env: Mapping[str, str]
) -> tuple[dict[str, Any], ...]:
    """Resolve secret-free, exact route receipts for every Agent candidate."""

    locks: dict[str, dict[str, Any]] = {}
    for cell in cells:
        if cell.get("execution_kind") != "agent":
            continue
        candidate_id = str(cell.get("candidate_id") or "")
        harness = str(cell.get("harness") or "")
        model = str(cell.get("model") or "")
        if not candidate_id or not harness or not model:
            raise CampaignError(
                "route_lock_invalid",
                "an Agent cell is missing its candidate, harness, or model identity",
                category="preparation",
            )
        try:
            route = resolve_model_route(model, env)
            route_identity = model_route_identity(route)
            transport = resolve_harness_model_route(route, harness)
        except ValueError as exc:
            raise CampaignError(
                "route_lock_invalid",
                "an Agent route could not be resolved for preparation",
                category="preparation",
                details={
                    "candidate_id": candidate_id,
                    "exception_type": type(exc).__name__,
                },
            ) from exc
        unsigned = {
            "candidate_id": candidate_id,
            "harness": harness,
            "model": model,
            "provider": route.provider,
            "model_id": route.model_id,
            "route_configuration_sha256": stable_digest(route_identity),
            "transport": _json_value(transport),
            "route_lock_sha256": "",
        }
        lock = {
            **unsigned,
            "route_lock_sha256": _artifact_digest(unsigned, "route_lock_sha256"),
        }
        previous = locks.setdefault(candidate_id, lock)
        if previous != lock:
            raise CampaignError(
                "route_lock_conflict",
                "one Agent candidate resolved to more than one model route",
                category="preparation",
                details={"candidate_id": candidate_id},
            )
    return tuple(locks[key] for key in sorted(locks))


def _route_lock_from_dict(raw: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "candidate_id",
        "harness",
        "model",
        "provider",
        "model_id",
        "route_configuration_sha256",
        "transport",
        "route_lock_sha256",
    }
    _reject_unknown(raw, fields, "route lock")
    transport = _mapping(raw.get("transport"), "route lock transport")
    _reject_unknown(
        transport,
        {
            "harness",
            "wire_protocol",
            "endpoint_kind",
            "upstream_host",
            "bridge_required",
        },
        "route lock transport",
    )
    value = {
        "candidate_id": _bounded_text(
            raw.get("candidate_id"), "route lock candidate id", 200
        ),
        "harness": validate_id(raw.get("harness") or "", kind="route lock harness"),
        "model": _bounded_text(raw.get("model"), "route lock model", 300),
        "provider": validate_id(raw.get("provider") or "", kind="route lock provider"),
        "model_id": _bounded_text(raw.get("model_id"), "route lock model id", 300),
        "route_configuration_sha256": _required_digest(
            raw.get("route_configuration_sha256"), "route configuration digest"
        ),
        "transport": {
            "harness": validate_id(
                transport.get("harness") or "", kind="route transport harness"
            ),
            "wire_protocol": _bounded_text(
                transport.get("wire_protocol"), "route wire protocol", 100
            ),
            "endpoint_kind": _bounded_text(
                transport.get("endpoint_kind"), "route endpoint kind", 100
            ),
            "upstream_host": _bounded_text(
                transport.get("upstream_host"), "route upstream host", 300
            ),
            "bridge_required": _strict_bool(
                transport.get("bridge_required"), "route bridge_required"
            ),
        },
        "route_lock_sha256": _required_digest(
            raw.get("route_lock_sha256"), "route lock digest"
        ),
    }
    _verify_artifact(value, "route_lock_sha256", "route lock")
    return value


def _plan_job_key(value: Any) -> tuple[Any, ...]:
    return (
        value.workload_id,
        value.task_id,
        value.harness,
        value.context_system_id,
        value.variant_id,
        value.trial_index,
        value.candidate_id,
    )


def _plan_job_index(plan: ResolvedRunPlan) -> dict[tuple[Any, ...], Any]:
    result: dict[tuple[Any, ...], Any] = {}
    for job in plan.jobs:
        key = _plan_job_key(job)
        if key in result:
            raise CampaignError(
                "duplicate_resolved_job",
                "resolved plan contains duplicate job coordinates",
                category="validation",
            )
        result[key] = job
    return result


def _plan_cell_record(
    position: int, cell: Any, job_index: Mapping[tuple[Any, ...], Any]
) -> dict[str, Any]:
    job = job_index.get(_plan_job_key(cell))
    task_count = (
        int((job.config.get("fugue") or {}).get("task_count") or 1) if job else 1
    )
    expected = 1 if cell.execution_kind == "agent" else task_count * cell.n_attempts
    coordinate = {
        "workload_id": cell.workload_id,
        "task_id": cell.task_id,
        "harness": cell.harness,
        "context_system_id": cell.context_system_id,
        "context_delivery": cell.context_delivery,
        "variant_id": cell.variant_id,
        "model_provider": cell.model_provider,
        "model": cell.model,
        "trial_index": cell.trial_index,
        "comparison_example_id": cell.comparison_example_id,
        "candidate_id": cell.candidate_id,
        "execution_fingerprint": cell.execution_fingerprint,
        "execution_kind": cell.execution_kind,
        "applicable": cell.applicable,
        "skip_reason": cell.skip_reason,
        "expected_predictions": expected,
    }
    return {
        "position": position,
        "coordinate_id": stable_digest(coordinate),
        **coordinate,
    }


def _plan_component_digests(
    repo_root: Path,
    proposal: ExperimentProposalV1,
    plan: ResolvedRunPlan,
    catalog_digests: Mapping[str, str],
) -> dict[str, str]:
    values: dict[str, str] = {}
    experiment_key = f"experiment:{proposal.experiment_id}"
    values[experiment_key] = catalog_digests[experiment_key]
    for analysis_id in proposal.analysis_ids:
        key = f"analysis:{analysis_id}"
        values[key] = catalog_digests[key]
    selected_variants = set(proposal.variants) or {
        item.variant_id for item in plan.cells
    }
    for variant in plan.experiment.variants:
        if variant.id not in selected_variants:
            continue
        if variant.prompt_id:
            values[f"prompt:{variant.prompt_id}"] = get_prompt(
                variant.prompt_id, repo_root
            ).sha256
        for skill_id in variant.skills:
            values[f"skill:{skill_id}"] = get_skill(skill_id, repo_root).sha256
        key = f"context_system:{variant.context.system_id}"
        values[key] = catalog_digests[key]
        for integration in variant.integrations:
            path = repo_root / "configs/fugue/integrations" / f"{integration.id}.yaml"
            values[f"integration:{integration.id}"] = _sha256_path(path)
    for integration in plan.experiment.integrations:
        path = repo_root / "configs/fugue/integrations" / f"{integration.id}.yaml"
        values[f"integration:{integration.id}"] = _sha256_path(path)
    for workload in plan.workloads:
        if workload.manifest:
            path = workload.manifest
            if not path.is_absolute():
                path = repo_root / path
            values[f"manifest:{workload.id}"] = _sha256_path(path)
        if workload.dataset:
            path = Path(workload.dataset)
            if not path.is_absolute():
                path = repo_root / path
            values[f"dataset:{workload.id}"] = _sha256_path(path)
    if plan.request.manifest:
        path = plan.request.manifest
        if not path.is_absolute():
            path = repo_root / path
        values["manifest:override"] = _sha256_path(path)
    if plan.request.selection_lock:
        path = plan.request.selection_lock
        if not path.is_absolute():
            path = repo_root / path
        values["selection_lock"] = _sha256_path(path)
    return dict(sorted(values.items()))


def _safe_request(request: ExperimentRequest) -> dict[str, Any]:
    if request.manifest is not None or request.jobs_dir is not None:
        raise CampaignError(
            "unsafe_request_field",
            "campaign requests may not contain manifest or jobs-directory overrides",
            category="validation",
        )
    return {
        "experiment_id": request.experiment_id,
        "preset": request.preset,
        "workloads": list(request.workloads),
        "harnesses": list(request.harnesses),
        "systems": list(request.systems),
        "variants": list(request.variants),
        "model": request.model,
        "n_attempts": request.n_attempts,
        "n_tasks": request.n_tasks,
        "n_concurrent": request.n_concurrent,
        "run_name": request.run_name,
        "tags": list(request.tags),
        "trace_content": request.trace_content,
        "cohort_id": request.cohort_id,
    }


def _request_from_safe(raw: Mapping[str, Any]) -> ExperimentRequest:
    _reject_unknown(
        raw,
        {
            "experiment_id",
            "preset",
            "workloads",
            "harnesses",
            "systems",
            "variants",
            "model",
            "n_attempts",
            "n_tasks",
            "n_concurrent",
            "run_name",
            "tags",
            "trace_content",
            "cohort_id",
        },
        "campaign request",
    )
    return ExperimentRequest(
        experiment_id=validate_id(raw.get("experiment_id") or "", kind="experiment id"),
        preset=str(raw["preset"]) if raw.get("preset") else None,
        workloads=_id_tuple(raw.get("workloads"), "workload", allow_empty=True),
        harnesses=_id_tuple(raw.get("harnesses"), "harness", allow_empty=True),
        systems=_id_tuple(raw.get("systems"), "context system", allow_empty=True),
        variants=_id_tuple(raw.get("variants"), "variant", allow_empty=True),
        model=str(raw["model"]) if raw.get("model") else None,
        n_attempts=int(raw["n_attempts"]) if raw.get("n_attempts") else None,
        n_tasks=int(raw["n_tasks"]) if raw.get("n_tasks") else None,
        n_concurrent=int(raw["n_concurrent"]) if raw.get("n_concurrent") else None,
        run_name=str(raw["run_name"]) if raw.get("run_name") else None,
        tags=_text_tuple(raw.get("tags"), "tag", allow_empty=True),
        trace_content=(str(raw["trace_content"]) if raw.get("trace_content") else None),
        cohort_id=str(raw["cohort_id"]) if raw.get("cohort_id") else None,
    )


def _canonical_preparation(
    preparation: SetupPreparation, repo_root: Path
) -> dict[str, Any]:
    evaluation_locks: list[dict[str, str]] = []
    for value in preparation.evaluation_asset_locks:
        path = Path(value)
        if not path.is_absolute():
            path = repo_root / path
        evaluation_locks.append({"id": path.name, "sha256": _sha256_path(path)})
    return {
        "context": [
            {
                "system_id": item.system_id,
                "task_id": item.task_id,
                "status": item.status,
                "cache_key": item.cache_key,
                "variant_id": item.variant_id,
                "config_digest": item.config_digest,
                "retrieval_mode": item.retrieval_mode,
            }
            for item in preparation.context
        ],
        "agent_runtimes": [
            {
                "harness": item.harness,
                "architecture": item.architecture,
                "status": item.status,
                "image": item.image,
                "image_id": item.image_id,
                "recipe_sha256": item.recipe_sha256,
            }
            for item in preparation.agent_runtimes
        ],
        "task_runtimes": [
            {
                "task_id": item.task_id,
                "architecture": item.architecture,
                "status": item.status,
                "image": item.image,
                "image_id": item.image_id,
                "recipe_sha256": item.recipe_sha256,
                "verification": _json_value(item.verification),
            }
            for item in preparation.task_runtimes
        ],
        "workload_datasets": [
            {
                "dataset_id": item.dataset_id,
                "status": item.status,
                "sample_count": item.sample_count,
                "sha256": item.sha256,
            }
            for item in preparation.workload_datasets
        ],
        "evaluation_asset_locks": sorted(
            evaluation_locks, key=lambda item: (item["id"], item["sha256"])
        ),
        "portable_context_runtime": (
            {
                "harness": preparation.portable_context_runtime.harness,
                "architecture": preparation.portable_context_runtime.architecture,
                "status": preparation.portable_context_runtime.status,
                "image": preparation.portable_context_runtime.image,
                "image_id": preparation.portable_context_runtime.image_id,
                "recipe_sha256": preparation.portable_context_runtime.recipe_sha256,
            }
            if preparation.portable_context_runtime is not None
            else None
        ),
    }


def _require_preparation(preparation: Mapping[str, Any]) -> None:
    skipped = [
        f"{item.get('system_id')}:{item.get('task_id')}"
        for item in preparation.get("context") or []
        if item.get("status") == "skipped"
    ]
    if skipped:
        raise CampaignError(
            "preparation_incomplete",
            "context preparation skipped required targets",
            category="preparation",
            details={"targets": skipped},
        )
    for runtime in preparation.get("task_runtimes") or []:
        verification = runtime.get("verification") or {}
        if (
            verification.get("base_failed") is not True
            or verification.get("gold_passed") is not True
        ):
            raise CampaignError(
                "task_runtime_unverified",
                f"task runtime {runtime.get('task_id')} lacks base-fail/gold-pass verification",
                category="preparation",
            )


def _plan_receipt_from_dict(raw: Mapping[str, Any]) -> PlanReceiptV1:
    fields = {
        "schema_version",
        "campaign_id",
        "proposal_id",
        "proposal_digest",
        "policy_digest",
        "catalog_digest",
        "source_provenance",
        "proposal",
        "request",
        "cells",
        "cell_count",
        "applicable_cells",
        "expected_predictions",
        "max_concurrent",
        "component_digests",
        "qualification_requirements",
        "plan_digest",
    }
    _reject_unknown(raw, fields, "plan receipt")
    receipt = PlanReceiptV1(
        schema_version=_schema(raw, "plan receipt"),
        campaign_id=validate_id(raw.get("campaign_id") or "", kind="campaign id"),
        proposal_id=validate_id(raw.get("proposal_id") or "", kind="proposal id"),
        proposal_digest=_required_digest(raw.get("proposal_digest"), "proposal_digest"),
        policy_digest=_required_digest(raw.get("policy_digest"), "policy_digest"),
        catalog_digest=_required_digest(raw.get("catalog_digest"), "catalog_digest"),
        source_provenance=_mapping(raw.get("source_provenance"), "source provenance"),
        proposal=_mapping(raw.get("proposal"), "proposal"),
        request=_mapping(raw.get("request"), "request"),
        cells=tuple(
            _mapping(item, "plan cell") for item in _sequence(raw.get("cells"), "cells")
        ),
        cell_count=_positive_int(raw.get("cell_count"), "plan cell_count"),
        applicable_cells=_non_negative_int(
            raw.get("applicable_cells"), "plan applicable_cells"
        ),
        expected_predictions=_positive_int(
            raw.get("expected_predictions"), "plan expected_predictions"
        ),
        max_concurrent=_positive_int(raw.get("max_concurrent"), "plan concurrency"),
        component_digests=_digest_mapping(
            raw.get("component_digests"), "component digests"
        ),
        qualification_requirements=_text_tuple(
            raw.get("qualification_requirements"), "qualification requirement"
        ),
        plan_digest=_required_digest(raw.get("plan_digest"), "plan_digest"),
    )
    _verify_artifact(receipt.to_dict(), "plan_digest", "plan receipt")
    experiment_proposal_from_dict(receipt.proposal)
    if receipt.cell_count != len(receipt.cells):
        raise ValueError("plan cell_count does not match cells")
    return receipt


def _prepared_plan_from_dict(raw: Mapping[str, Any]) -> PreparedPlanV1:
    fields = {
        "schema_version",
        "campaign_id",
        "proposal_id",
        "plan_digest",
        "policy_digest",
        "source_provenance",
        "plan",
        "preparation",
        "preflight",
        "component_digests",
        "route_locks",
        "integration_locks",
        "prepared_at",
        "prepared_plan_digest",
    }
    _reject_unknown(raw, fields, "prepared plan")
    receipt = PreparedPlanV1(
        schema_version=_schema(raw, "prepared plan"),
        campaign_id=validate_id(raw.get("campaign_id") or "", kind="campaign id"),
        proposal_id=validate_id(raw.get("proposal_id") or "", kind="proposal id"),
        plan_digest=_required_digest(raw.get("plan_digest"), "plan_digest"),
        policy_digest=_required_digest(raw.get("policy_digest"), "policy_digest"),
        source_provenance=_mapping(raw.get("source_provenance"), "source provenance"),
        plan=_mapping(raw.get("plan"), "plan"),
        preparation=_mapping(raw.get("preparation"), "preparation"),
        preflight=tuple(
            _mapping(item, "preflight check")
            for item in _sequence(raw.get("preflight"), "preflight")
        ),
        component_digests=_digest_mapping(
            raw.get("component_digests"), "component digests"
        ),
        route_locks=tuple(
            _route_lock_from_dict(_mapping(item, "route lock"))
            for item in _sequence(raw.get("route_locks"), "route locks")
        ),
        integration_locks=_digest_mapping(
            raw.get("integration_locks"), "integration locks"
        ),
        prepared_at=_bounded_text(raw.get("prepared_at"), "prepared_at", 100),
        prepared_plan_digest=_required_digest(
            raw.get("prepared_plan_digest"), "prepared_plan_digest"
        ),
    )
    _verify_artifact(receipt.to_dict(), "prepared_plan_digest", "prepared plan")
    _plan_receipt_from_dict(receipt.plan)
    return receipt


def _admission_receipt_from_dict(raw: Mapping[str, Any]) -> AdmissionReceiptV1:
    fields = {
        "schema_version",
        "admission_id",
        "campaign_id",
        "proposal_id",
        "stage_id",
        "prepared_plan_digest",
        "policy_digest",
        "operation_id",
        "parent_outcome_id",
        "cell_count",
        "reserved_cell_cost_usd",
        "reserved_cost_usd",
        "prepared_plan",
        "admitted_at",
        "admission_digest",
    }
    _reject_unknown(raw, fields, "admission receipt")
    receipt = AdmissionReceiptV1(
        schema_version=_schema(raw, "admission receipt"),
        admission_id=_required_digest(raw.get("admission_id"), "admission_id"),
        campaign_id=validate_id(raw.get("campaign_id") or "", kind="campaign id"),
        proposal_id=validate_id(raw.get("proposal_id") or "", kind="proposal id"),
        stage_id=validate_id(raw.get("stage_id") or "", kind="stage id"),
        prepared_plan_digest=_required_digest(
            raw.get("prepared_plan_digest"), "prepared_plan_digest"
        ),
        policy_digest=_required_digest(raw.get("policy_digest"), "policy_digest"),
        operation_id=validate_id(raw.get("operation_id") or "", kind="operation id"),
        parent_outcome_id=(
            validate_id(raw["parent_outcome_id"], kind="outcome id")
            if raw.get("parent_outcome_id")
            else None
        ),
        cell_count=_positive_int(raw.get("cell_count"), "admission cell_count"),
        reserved_cell_cost_usd=_non_negative_number(
            raw.get("reserved_cell_cost_usd"), "reserved cell cost"
        ),
        reserved_cost_usd=_non_negative_number(
            raw.get("reserved_cost_usd"), "reserved cost"
        ),
        prepared_plan=_mapping(raw.get("prepared_plan"), "prepared plan"),
        admitted_at=_bounded_text(raw.get("admitted_at"), "admitted_at", 100),
        admission_digest=_required_digest(
            raw.get("admission_digest"), "admission_digest"
        ),
    )
    _verify_artifact(receipt.to_dict(), "admission_digest", "admission receipt")
    _prepared_plan_from_dict(receipt.prepared_plan)
    return receipt


def _outcome_packet_from_dict(raw: Mapping[str, Any]) -> OutcomePacketV1:
    fields = {
        "schema_version",
        "outcome_id",
        "campaign_id",
        "proposal_id",
        "stage_id",
        "admission_id",
        "run_id",
        "run_status",
        "expected_predictions",
        "observed_predictions",
        "passed",
        "failed",
        "not_applicable",
        "eligible",
        "eligibility_failures",
        "limitations",
        "observed_cost_usd",
        "accounted_cost_usd",
        "measured_cost_cells",
        "unmeasured_cost_cells",
        "maximum_measured_cell_cost_usd",
        "input_lock_sha256",
        "run_snapshot_sha256",
        "export_sha256",
        "export_path",
        "row_refs",
        "evidence_refs",
        "analysis_results",
        "metrics",
        "finalized_at",
        "outcome_digest",
    }
    _reject_unknown(raw, fields, "outcome packet")
    outcome = OutcomePacketV1(
        schema_version=_schema(raw, "outcome packet"),
        outcome_id=_required_digest(raw.get("outcome_id"), "outcome_id"),
        campaign_id=validate_id(raw.get("campaign_id") or "", kind="campaign id"),
        proposal_id=validate_id(raw.get("proposal_id") or "", kind="proposal id"),
        stage_id=validate_id(raw.get("stage_id") or "", kind="stage id"),
        admission_id=_required_digest(raw.get("admission_id"), "admission_id"),
        run_id=validate_id(raw.get("run_id") or "", kind="run id"),
        run_status=_bounded_text(raw.get("run_status"), "run status", 50),
        expected_predictions=_positive_int(
            raw.get("expected_predictions"), "expected predictions"
        ),
        observed_predictions=_non_negative_int(
            raw.get("observed_predictions"), "observed predictions"
        ),
        passed=_non_negative_int(raw.get("passed"), "passed count"),
        failed=_non_negative_int(raw.get("failed"), "failed count"),
        not_applicable=_non_negative_int(
            raw.get("not_applicable"), "not applicable count"
        ),
        eligible=bool(raw.get("eligible")),
        eligibility_failures=_text_tuple(
            raw.get("eligibility_failures"), "eligibility failure", allow_empty=True
        ),
        limitations=_text_tuple(raw.get("limitations"), "limitation", allow_empty=True),
        observed_cost_usd=_non_negative_number(
            raw.get("observed_cost_usd"), "observed cost"
        ),
        accounted_cost_usd=_non_negative_number(
            raw.get("accounted_cost_usd"), "accounted cost"
        ),
        measured_cost_cells=_non_negative_int(
            raw.get("measured_cost_cells"), "measured cost cells"
        ),
        unmeasured_cost_cells=_non_negative_int(
            raw.get("unmeasured_cost_cells"), "unmeasured cost cells"
        ),
        maximum_measured_cell_cost_usd=(
            _non_negative_number(
                raw.get("maximum_measured_cell_cost_usd"), "maximum measured cost"
            )
            if raw.get("maximum_measured_cell_cost_usd") is not None
            else None
        ),
        input_lock_sha256=(
            _required_digest(raw["input_lock_sha256"], "input lock digest")
            if raw.get("input_lock_sha256")
            else None
        ),
        run_snapshot_sha256=(
            _required_digest(raw["run_snapshot_sha256"], "run snapshot digest")
            if raw.get("run_snapshot_sha256")
            else None
        ),
        export_sha256=_required_digest(raw.get("export_sha256"), "export digest"),
        export_path=_safe_runtime_path(raw.get("export_path")),
        row_refs=tuple(
            _mapping(item, "row reference")
            for item in _sequence(raw.get("row_refs"), "row references")
        ),
        evidence_refs=tuple(
            _mapping(item, "evidence reference")
            for item in _sequence(raw.get("evidence_refs"), "evidence references")
        ),
        analysis_results=tuple(
            _mapping(item, "analysis result")
            for item in _sequence(raw.get("analysis_results"), "analysis results")
        ),
        metrics=_mapping(raw.get("metrics"), "outcome metrics"),
        finalized_at=_bounded_text(raw.get("finalized_at"), "finalized_at", 100),
        outcome_digest=_required_digest(raw.get("outcome_digest"), "outcome_digest"),
    )
    _verify_artifact(outcome.to_dict(), "outcome_digest", "outcome packet")
    return outcome


def _campaign_event_from_dict(raw: Mapping[str, Any]) -> CampaignEventV1:
    fields = {
        "schema_version",
        "sequence_number",
        "event_id",
        "campaign_id",
        "event",
        "recorded_at",
        "operation_id",
        "proposal_id",
        "admission_id",
        "run_id",
        "artifact_type",
        "artifact_digest",
        "error",
        "previous_event_digest",
        "event_digest",
    }
    _reject_unknown(raw, fields, "campaign event")
    value = CampaignEventV1(
        schema_version=_schema(raw, "campaign event"),
        sequence_number=_positive_int(raw.get("sequence_number"), "event sequence"),
        event_id=_bounded_text(raw.get("event_id"), "event id", 100),
        campaign_id=validate_id(raw.get("campaign_id") or "", kind="campaign id"),
        event=_bounded_text(raw.get("event"), "event", 100),
        recorded_at=_bounded_text(raw.get("recorded_at"), "recorded_at", 100),
        operation_id=str(raw["operation_id"]) if raw.get("operation_id") else None,
        proposal_id=str(raw["proposal_id"]) if raw.get("proposal_id") else None,
        admission_id=str(raw["admission_id"]) if raw.get("admission_id") else None,
        run_id=str(raw["run_id"]) if raw.get("run_id") else None,
        artifact_type=str(raw["artifact_type"]) if raw.get("artifact_type") else None,
        artifact_digest=(
            _required_digest(raw["artifact_digest"], "artifact digest")
            if raw.get("artifact_digest")
            else None
        ),
        error=(
            campaign_error_from_dict(
                _mapping(raw.get("error"), "campaign event error")
            ).to_dict()
            if raw.get("error")
            else None
        ),
        previous_event_digest=(
            _required_digest(raw["previous_event_digest"], "previous event digest")
            if raw.get("previous_event_digest")
            else None
        ),
        event_digest=_required_digest(raw.get("event_digest"), "event digest"),
    )
    _verify_artifact(value.to_dict(), "event_digest", "campaign event")
    return value


def _new_ledger(policy: ResearchCampaignSpecV1) -> dict[str, Any]:
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_id": policy.id,
        "policy_digest": policy.campaign_digest,
        "accounted_cost_usd": 0.0,
        "admissions": [],
    }


def _validate_ledger(ledger: Mapping[str, Any], policy: ResearchCampaignSpecV1) -> None:
    _reject_unknown(
        ledger,
        {
            "schema_version",
            "campaign_id",
            "policy_digest",
            "accounted_cost_usd",
            "admissions",
        },
        "campaign ledger",
    )
    if int(ledger.get("schema_version") or 0) != CAMPAIGN_SCHEMA_VERSION:
        raise CampaignError(
            "ledger_version_unsupported",
            "campaign ledger must use schema_version 1",
            category="evidence",
        )
    if ledger.get("campaign_id") != policy.id:
        raise CampaignError(
            "ledger_identity_mismatch",
            "campaign ledger identity does not match its policy",
            category="evidence",
        )
    if ledger.get("policy_digest") != policy.campaign_digest:
        raise CampaignError(
            "policy_drift",
            "campaign ledger is bound to a different policy revision",
            category="policy",
        )
    if not isinstance(ledger.get("admissions"), list):
        raise CampaignError(
            "ledger_corrupt",
            "campaign ledger admissions must be a list",
            category="evidence",
        )
    _non_negative_number(ledger.get("accounted_cost_usd"), "accounted cost")


def _ledger_admission(ledger: Mapping[str, Any], admission_id: str) -> dict[str, Any]:
    matches = [
        item
        for item in ledger.get("admissions") or []
        if isinstance(item, dict) and item.get("admission_id") == admission_id
    ]
    if len(matches) != 1:
        raise CampaignError(
            "admission_missing",
            "campaign admission was not found exactly once in the ledger",
            category="admission",
        )
    return matches[0]


def _run_admission(ledger: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    matches = [
        item
        for item in ledger.get("admissions") or []
        if isinstance(item, dict) and item.get("run_id") == run_id
    ]
    if len(matches) != 1:
        raise CampaignError(
            "run_not_admitted",
            "run is not bound to exactly one campaign admission",
            category="admission",
        )
    return matches[0]


def _reserved_budget(ledger: Mapping[str, Any]) -> float:
    return sum(
        (
            float(item.get("reserved_cost_usd") or 0.0)
            for item in ledger.get("admissions") or []
            if item.get("outcome_id") is None or item.get("status") == "blocked"
        ),
        0.0,
    )


def _remaining_budget(
    ledger: Mapping[str, Any], policy: ResearchCampaignSpecV1
) -> float:
    return max(
        0.0,
        policy.limits.total_cost_usd
        - float(ledger.get("accounted_cost_usd") or 0.0)
        - _reserved_budget(ledger),
    )


def _require_allowed(kind: str, values: Sequence[str], allowed: Sequence[str]) -> None:
    unexpected = sorted(set(values) - set(allowed))
    if unexpected:
        raise CampaignError(
            "component_not_allowed",
            f"campaign does not allow {kind}(s): {', '.join(unexpected)}",
            category="policy",
            details={"kind": kind, "values": unexpected},
        )


def _artifact_digest(value: Mapping[str, Any], digest_field: str) -> str:
    unsigned = dict(_json_value(value))
    unsigned[digest_field] = ""
    return stable_digest(unsigned)


def _verify_artifact(value: Mapping[str, Any], digest_field: str, label: str) -> None:
    expected = str(value.get(digest_field) or "")
    if not _DIGEST_RE.fullmatch(expected):
        raise CampaignError(
            "artifact_digest_missing",
            f"{label} is missing a canonical digest",
            category="validation",
        )
    if expected != _artifact_digest(value, digest_field):
        raise CampaignError(
            "artifact_digest_mismatch",
            f"{label} digest does not match its content",
            category="validation",
        )


def _required_digest(value: Any, label: str) -> str:
    result = str(value or "")
    if not _DIGEST_RE.fullmatch(result):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return result


def _validate_task_profile_policy(policy: TaskAuthoringPolicyV1, profiles: Any) -> None:
    declared = {
        "environment": (
            {item.id for item in profiles.environments},
            policy.allowed_environment_profiles,
        ),
        "resource": (
            {item.id for item in profiles.resources},
            policy.allowed_resource_profiles,
        ),
        "interactor": (
            {item.id for item in profiles.interactors},
            policy.allowed_interactor_profiles,
        ),
        "judge": ({item.id for item in profiles.judges}, policy.allowed_judge_profiles),
        "scorer runtime": (
            {item.id for item in profiles.scorer_runtimes},
            policy.allowed_scorer_runtimes,
        ),
    }
    for label, (available, allowed) in declared.items():
        missing = sorted(set(allowed) - available)
        if missing:
            raise ValueError(
                f"campaign allows unregistered {label} profile(s): {', '.join(missing)}"
            )


def _safe_task_profile_catalog(
    policy: TaskAuthoringPolicyV1, profiles: Any
) -> dict[str, Any]:
    safe = profiles.safe_dict()
    allowed = {
        "environments": set(policy.allowed_environment_profiles),
        "resources": set(policy.allowed_resource_profiles),
        "interactors": set(policy.allowed_interactor_profiles),
        "judges": set(policy.allowed_judge_profiles),
        "scorer_runtimes": set(policy.allowed_scorer_runtimes),
    }
    for key, ids in allowed.items():
        safe[key] = [item for item in safe[key] if item["id"] in ids]
    safe["policy"] = policy.to_dict()
    return safe


def _strict_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _digest_mapping(value: Any, label: str) -> dict[str, str]:
    raw = _mapping(value, label)
    return {
        str(key): _required_digest(item, f"{label} {key}") for key, item in raw.items()
    }


def _safe_runtime_path(value: Any) -> str:
    path = str(value or "")
    candidate = Path(path)
    if (
        not path
        or candidate.is_absolute()
        or ".." in candidate.parts
        or not candidate.parts
        or candidate.parts[0] != ".fugue"
    ):
        raise ValueError("outcome export path must remain inside .fugue")
    return candidate.as_posix()


def _schema(raw: Mapping[str, Any], label: str) -> int:
    value = int(raw.get("schema_version") or 0)
    if value != CAMPAIGN_SCHEMA_VERSION:
        raise ValueError(f"{label} must use schema_version 1")
    return value


def _reject_unknown(raw: Mapping[str, Any], known: set[str], label: str) -> None:
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"unknown {label} field(s): {', '.join(unknown)}")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _sequence(value: Any, label: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list | tuple):
        raise ValueError(f"{label} must be a list")
    return list(value)


def _text_tuple(
    value: Any, label: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    result = tuple(_bounded_text(item, label, 1000) for item in _sequence(value, label))
    if not allow_empty and not result:
        raise ValueError(f"{label} must contain at least one value")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} values must be unique")
    return result


def _id_tuple(value: Any, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    result = tuple(
        validate_id(item, kind=f"{label} id") for item in _sequence(value, label)
    )
    if not allow_empty and not result:
        raise ValueError(f"campaign must allow at least one {label}")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} ids must be unique")
    return result


def _dimension_tuple(value: Any, label: str) -> tuple[str, ...]:
    return _text_tuple(value, f"{label} dimensions")


def _bounded_text(value: Any, label: str, limit: int) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{label} must be non-empty")
    if len(result) > limit:
        raise ValueError(f"{label} exceeds {limit} characters")
    if any(ord(character) < 32 and character not in "\n\t" for character in result):
        raise ValueError(f"{label} contains control characters")
    return result


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


def _positive_number(value: Any, label: str) -> float:
    result = float(value or 0)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _non_negative_number(value: Any, label: str) -> float:
    result = float(value or 0)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _at_least_one_number(value: Any, label: str) -> float:
    result = float(value or 0)
    if not math.isfinite(result) or result < 1:
        raise ValueError(f"{label} must be finite and at least one")
    return result


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _now() -> str:
    return datetime.now(UTC).isoformat()
