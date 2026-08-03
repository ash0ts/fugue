from __future__ import annotations

import ast
import base64
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import urllib.parse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml

from fugue.bench.analysis_contracts import (
    AlignedAnalysisV1,
    AlignedArmV1,
    AlignedAttemptSetV1,
    AlignedContrastV1,
    AlignedDimensionV1,
    DimensionRole,
    EvidenceDriftCheckV1,
    EvidenceTopologyV1,
    LockDescriptorV1,
    SupersededResultV1,
    TaskStratifiedSummaryV1,
    TaskValidityV1,
    aligned_analysis_from_dict,
    evidence_topology_from_dict,
    lock_descriptor_from_dict,
    superseded_result_from_dict,
    task_validity_from_dict,
)
from fugue.bench.candidates import attempt_id, attempt_identity, stable_digest
from fugue.bench.files import atomic_write_json, inspect_docker_image
from fugue.bench.library import ExperimentSpec, experiment_from_data, validate_id
from fugue.bench.operator import (
    ExperimentRequest,
    OperatorService,
    PreviewSummary,
)
from fugue.model_plane import (
    EvidenceDestinationV1,
    evidence_destination_environment,
    evidence_destination_from_dict,
    trace_project_environment,
    trace_project_slug,
)
from fugue.redaction import redact_text, redact_value, secrets_from_env
from fugue.research.approvals import ApprovalLedger
from fugue.research.store import StudyStore

COMPARISON_SCHEMA_VERSION = 2
COMPARISON_RESULT_SCHEMA_VERSION = 3
COMPARISON_READABLE_SCHEMA_VERSIONS = frozenset({1, 2, 3})
COMPARISON_RUNTIME_ROOT = Path(".fugue/runtime/comparisons")
COMPARISON_RESULT_ROOT = Path(".fugue/results/comparisons")
APPROVED_COMPARISON_LOCK_NAME = "approved-comparison.lock.json"
COMPARISON_INPUT_ROOT = Path(".fugue/runtime/comparison-inputs")
COMPARISON_PRIVATE_INPUT_ROOT = Path(".fugue/private/comparison-inputs")
COMPARISON_EVALUATOR_RUNTIME_ROOT = Path(".fugue/runtime/evaluator-runtimes")
DEFAULT_COMPARISON_JUDGE_TIMEOUT_SEC = 120
MAX_COMPARISON_JUDGE_TIMEOUT_SEC = 900
COMPARISON_JUDGE_RESPONSE_VALIDATOR_VERSION = 2
COMPARISON_JUDGE_TEXT_REQUESTED_MAX_CHARACTERS = 500
COMPARISON_JUDGE_RESPONSE_MAX_CHARACTERS = 16_000
COMPARISON_JUDGE_MAX_PROMPT_CHARACTERS = 48_000
COMPARISON_JUDGE_PASSING_LABELS = ("adequate", "strong", "exceptional")
COMPARISON_JUDGE_PASSING_SCORE = 0.5
COMPARISON_JUDGE_METRIC_POLICY = "balanced_accuracy_by_split_v1"
COMPARISON_JUDGE_MINIMUM_BALANCED_ACCURACY = 0.85
COMPARISON_JUDGE_BALANCED_ACCURACY_SCOPES = (
    "overall",
    "calibration",
    "holdout",
)
COMPARISON_JUDGE_CRITICAL_POLICY = (
    "Missing required evidence makes the qualitative judgment unavailable, and "
    "a qualitative label can never override a deterministic critical failure."
)
_COMPARISON_JUDGE_LABEL_BAND_ROWS = (
    (
        "unusable",
        "0.00 <= mean < 0.25",
        "No safe, actionable result, or a critical claim conflicts with the "
        "supplied evidence.",
    ),
    (
        "weak",
        "0.25 <= mean < 0.50",
        "Some relevant content, but it is materially generic, ungrounded, "
        "unsafe, or not reviewable.",
    ),
    (
        "adequate",
        "0.50 <= mean < 0.75",
        "Minimum acceptable with explicit limitations: plausible, bounded, and "
        "reviewable enough to use, but missing depth in repository evidence, "
        "review detail, or risk handling.",
    ),
    (
        "strong",
        "0.75 <= mean < 0.90",
        "Actionable, repository-grounded, reviewable, and appropriately "
        "calibrated, with only minor omissions.",
    ),
    (
        "exceptional",
        "0.90 <= mean <= 1.00",
        "Strong on every dimension and unusually clear about evidence, "
        "tradeoffs, verification, and limits.",
    ),
)

_HARNESS_AGENTS = {
    "hermes": "fugue.agents:FugueHermes",
    "openclaw": "fugue.agents:FugueOpenClaw",
    "claude-code": "fugue.agents:FugueClaudeCode",
    "codex": "fugue.agents:FugueCodex",
}
_READINESS = frozenset(
    {"ready", "needs_review", "blocked", "no_comparison_justified"}
)
_PUBLIC_TASK_FIELDS = frozenset(
    {
        "id",
        "input",
        "resources",
        "tags",
        "partition",
        "critical_dimensions",
    }
)
_PRIVATE_LABEL_FIELDS = frozenset(
    {
        "id",
        "expected",
        "base_output",
        "gold_output",
        "base_evidence",
        "gold_evidence",
    }
)
_PRIVATE_WORDS = frozenset(
    {"expected", "gold", "reference_answer", "private", "answer_key"}
)
_COMPARISON_BASE_IMAGE = (
    "python:3.12.10-slim-bookworm@"
    "sha256:fd95fa221297a88e1cf49c55ec1828edd7c5a428187e67b5d1805692d11588db"
)
_MCP_RELEASE_READ_ONLY_TOOLS = frozenset(
    {
        "compare_artifact_versions_tool",
        "compare_runs_tool",
        "count_weave_traces_tool",
        "diagnose_run_tool",
        "get_artifact_details_tool",
        "get_run_history_tool",
        "infer_trace_schema_tool",
        "list_artifact_versions_tool",
        "probe_project_tool",
        "query_wandb_tool",
        "query_weave_traces_tool",
        "resolve_trace_roots_tool",
        "summarize_evaluation_tool",
    }
)


def comparison_judge_label_bands() -> list[dict[str, str]]:
    """Return the canonical provider-visible meaning of judge score bands."""

    return [
        {"id": label, "score_band": score_band, "description": description}
        for label, score_band, description in _COMPARISON_JUDGE_LABEL_BAND_ROWS
    ]


def comparison_judge_public_rubric_contract(
    *,
    rubric: str,
    dimensions: Sequence[str],
) -> dict[str, Any]:
    """Build the one public rubric contract used by calibration and live judges."""

    normalized_rubric = str(rubric).strip()
    normalized_dimensions = [str(item).strip() for item in dimensions]
    if not normalized_rubric:
        raise ValueError("comparison judge public rubric must not be empty")
    if not normalized_dimensions or any(not item for item in normalized_dimensions):
        raise ValueError("comparison judge public rubric dimensions must not be empty")
    if len(set(normalized_dimensions)) != len(normalized_dimensions):
        raise ValueError("comparison judge public rubric dimensions must be unique")
    return {
        "text": normalized_rubric,
        "dimensions": normalized_dimensions,
        "label_bands": comparison_judge_label_bands(),
        "passing_labels": list(COMPARISON_JUDGE_PASSING_LABELS),
        "passing_score": COMPARISON_JUDGE_PASSING_SCORE,
        "critical_policy": COMPARISON_JUDGE_CRITICAL_POLICY,
    }


def comparison_judge_request_payload(
    *,
    public_task: Mapping[str, Any],
    response: Any,
    permitted_evidence: Mapping[str, Any],
    rubric: str,
    dimensions: Sequence[str],
) -> dict[str, Any]:
    """Return the normalized provider-visible request used in every judge mode."""

    return {
        "public_task": dict(public_task),
        "response": response,
        "permitted_evidence": dict(permitted_evidence),
        "rubric": comparison_judge_public_rubric_contract(
            rubric=rubric,
            dimensions=dimensions,
        ),
    }


@dataclass(frozen=True)
class ComparisonTasksetV1:
    tasks: str
    private_labels: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComparisonCandidateV1:
    label: str
    prompt_id: str | None = None
    skills: tuple[str, ...] = ()
    context: dict[str, Any] = field(
        default_factory=lambda: {"system_id": "none", "delivery": "portable"}
    )
    integrations: tuple[dict[str, Any], ...] = ()
    agent_kwargs: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self))

    def behavior(self) -> dict[str, Any]:
        value = self.to_dict()
        value.pop("label", None)
        return value


@dataclass(frozen=True)
class ComparisonPostTrialVerifierV1:
    type: Literal["node_test"]
    source: str
    runtime: str
    dimension: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComparisonEvaluatorV1:
    id: str
    type: Literal["deterministic", "llm_judge"]
    required: bool
    checks: tuple[str, ...] = ()
    scorer: str | None = None
    runtime: str | None = None
    verifier: ComparisonPostTrialVerifierV1 | None = None
    profile: str | None = None
    calibration: str | None = None
    rubric: str | None = None
    dimensions: tuple[str, ...] = ()
    dimension_roles: dict[str, DimensionRole] = field(default_factory=dict)
    evidence: tuple[str, ...] = ()
    timeout_sec: int | None = None
    reserve_cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass(frozen=True)
class ComparisonExecutionPolicyV1:
    model: str
    harnesses: tuple[str, ...]
    attempts: int
    concurrency: int
    max_cost_usd: float
    reserve_per_attempt_usd: float
    approval_required: bool
    trace_content: Literal["full", "metadata"]
    source_evidence_project: str | None = None
    source_evidence_destination: EvidenceDestinationV1 | None = None
    evidence_project: str | None = None
    evidence_destination: EvidenceDestinationV1 | None = None
    study_console_base_url: str | None = None
    research_id: str | None = None
    infrastructure_receipt: str | None = None
    source_lock: str | None = None
    evidence_lock: str | None = None
    source_conformance_receipt: str | None = None
    release_notes_lock: str | None = None
    mechanism_receipt: str | None = None
    prerequisite_result: str | None = None
    prerequisite_attestation: str | None = None
    prerequisite_comparison_id: str | None = None
    prerequisite_spec: str | None = None
    preparation_required: bool = False
    evidence_checkpoint_cells: int = 0
    scheduling_seed: str | None = None
    qualification_inputs: dict[str, str] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass
class _ComparisonRuntimeBudget:
    """Fail closed between approved comparison cells when spend drifts.

    A model-backed cell is an atomic unit and can itself exceed its reservation.
    This ledger prevents Fugue from launching another queued cell after accounted
    spend, missing cost evidence, or the minimum reservation for remaining cells
    makes the approved ceiling impossible to honor.
    """

    max_cost_usd: float
    reserve_per_attempt_usd: float
    total_cells: int
    cancellation_event: threading.Event = field(default_factory=threading.Event)
    accounted_cost_usd: float = 0.0
    failure_reason: str | None = None
    _receipts: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def observe(self, row: dict[str, Any]) -> None:
        attempt = str(row.get("attempt_id") or "").strip()
        with self._lock:
            if attempt and attempt in self._receipts:
                row["comparison_runtime_budget"] = dict(self._receipts[attempt])
                if self.failure_reason:
                    raise RuntimeError(self.failure_reason)
                return

            terminal_cells = len(self._receipts) + 1
            remaining_cells = max(0, self.total_cells - terminal_cells)
            cost = _row_number(row, "accounted_cost_usd")
            cost_source = "accounted_cost_usd" if cost is not None else None
            if cost is None:
                cost, cost_source = _observed_attempt_cost(row)
            reason: str | None = None
            if not attempt:
                reason = (
                    "comparison runtime budget could not bind terminal spend "
                    "to a stable attempt identity"
                )
            elif cost is None and (
                self.max_cost_usd == 0 and self.reserve_per_attempt_usd == 0
            ):
                cost = 0.0
                self.accounted_cost_usd += cost
            elif cost is None or cost < 0:
                reason = (
                    "comparison runtime cost evidence is unavailable for "
                    f"attempt {attempt}; remaining cells were cancelled"
                )
            else:
                self.accounted_cost_usd += cost

            reserved_remaining = self.reserve_per_attempt_usd * remaining_cells
            projected_minimum = self.accounted_cost_usd + reserved_remaining
            if reason is None and self.accounted_cost_usd > self.max_cost_usd + 1e-9:
                reason = (
                    "comparison accounted spend exceeded the approved ceiling "
                    f"(${self.accounted_cost_usd:.6f} > ${self.max_cost_usd:.6f}); "
                    "remaining cells were cancelled"
                )
            elif reason is None and projected_minimum > self.max_cost_usd + 1e-9:
                reason = (
                    "comparison accounted spend plus the locked reservation for "
                    "remaining cells exceeds the approved ceiling "
                    f"(${projected_minimum:.6f} > ${self.max_cost_usd:.6f}); "
                    "remaining cells were cancelled"
                )

            receipt = {
                "schema_version": 1,
                "status": "failed" if reason else "within_budget",
                "attempt_id": attempt or None,
                "cell_accounted_cost_usd": cost,
                "cell_accounted_cost_source": cost_source,
                "accounted_cost_usd": round(self.accounted_cost_usd, 9),
                "approved_max_cost_usd": self.max_cost_usd,
                "reserve_per_remaining_attempt_usd": self.reserve_per_attempt_usd,
                "terminal_cells": terminal_cells,
                "remaining_cells": remaining_cells,
                "reserved_remaining_cost_usd": round(reserved_remaining, 9),
                "projected_minimum_cost_usd": round(projected_minimum, 9),
                **({"reason": reason} if reason else {}),
            }
            if attempt:
                self._receipts[attempt] = receipt
            row["comparison_runtime_budget"] = dict(receipt)
            if reason:
                self.failure_reason = self.failure_reason or reason
                self.cancellation_event.set()
                raise RuntimeError(self.failure_reason)


DecisionStatus = Literal[
    "invalid",
    "blocked",
    "hold",
    "inconclusive",
    "ready_for_signoff",
    "go",
]
DecisionGateCategory = Literal[
    "integrity",
    "task",
    "infrastructure",
    "evidence",
    "efficiency",
    "privacy",
]


@dataclass(frozen=True)
class DecisionGatePolicyV1:
    id: str
    label: str
    category: DecisionGateCategory
    source: str
    operator: Literal["eq", "lte", "gte"]
    target: str | float | int | bool
    critical: bool = True

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass(frozen=True)
class DecisionPolicyV1:
    release_target: str
    candidate_sha: str
    minimum_evidence_grade: Literal["A", "B", "C"] = "A"
    human_signoff_required: bool = True
    gates: tuple[DecisionGatePolicyV1, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(_json_value(asdict(self)), preserve_false=True)


@dataclass(frozen=True)
class DecisionGateResultV1:
    id: str
    label: str
    category: DecisionGateCategory
    status: Literal["passed", "failed", "unavailable"]
    critical: bool
    actual: str | float | int | bool | None
    target: str | float | int | bool

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass(frozen=True)
class DecisionAttestationV1:
    signer: str
    signed_result_digest: str
    signed_at: str
    review_status: Literal["accepted_actionable", "rejected"] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecisionSummaryV1:
    status: DecisionStatus
    recommendation: str
    release_target: str | None
    candidate_sha: str | None
    evidence_grade: Literal["A", "B", "C", "invalid"]
    gates: tuple[DecisionGateResultV1, ...]
    critical_blockers: tuple[str, ...]
    limitations: tuple[str, ...]
    next_action: str
    human_signoff_required: bool
    attestation: DecisionAttestationV1 | None = None

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


BehavioralStatus = Literal[
    "invalid",
    "incomplete",
    "improved",
    "regressed",
    "mixed",
    "unchanged",
]
PairStatus = Literal["improved", "regressed", "mixed", "unchanged", "incomplete"]
DimensionStatus = Literal["improved", "regressed", "unchanged", "unavailable"]
EvidenceLinkStatus = Literal["resolved", "missing", "invalid"]
EvidenceLinkKind = Literal[
    "evaluation_root",
    "prediction_and_score",
    "prediction",
    "agent_root",
    "dataset",
]


@dataclass(frozen=True)
class AttemptEvidenceLinkV1:
    kind: EvidenceLinkKind
    status: EvidenceLinkStatus
    system: Literal["weave"] = "weave"
    ref: str | None = None
    url: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass(frozen=True)
class DimensionChangeV1:
    id: str
    status: DimensionStatus
    baseline: bool | None
    candidate: bool | None
    critical: bool

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass(frozen=True)
class DimensionChangeV2:
    id: str
    label: str
    status: DimensionStatus
    baseline: bool | None
    candidate: bool | None
    critical: bool
    role: DimensionRole
    baseline_explanation: str | None = None
    candidate_explanation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass(frozen=True)
class PairedAttemptV2:
    attempt_id: str
    identity: dict[str, Any]
    prediction_id: str | None
    passed: bool | None
    execution_status: str
    evaluation_status: str
    evidence_status: str
    cost_usd: float | None
    latency_sec: float | None
    input_tokens: float | None
    output_tokens: float | None
    tool_calls: int
    tools: tuple[str, ...]
    queried_projects: tuple[str, ...]
    scores: dict[str, Any]
    evidence_links: tuple[AttemptEvidenceLinkV1, ...]
    weave_agent_root_call_id: str | None = None
    otel_root_span_id: str | None = None
    execution_fingerprint: str | None = None
    runtime_lock_digest: str | None = None
    infrastructure: dict[str, Any] = field(default_factory=dict)
    judge_reviews: dict[str, JudgeReviewV1] = field(default_factory=dict)
    benchmark_outcome: Literal[
        "passed", "failed", "unscored", "not_applicable"
    ] | None = None
    runtime_outcome: Literal[
        "completed", "timed_out", "cancelled", "not_started", "not_applicable"
    ] | None = None

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(_json_value(asdict(self)), preserve_false=True)


@dataclass(frozen=True)
class JudgeReviewV1:
    """Safe, decision-independent presentation of one blind judge result."""

    label: Literal[
        "unusable",
        "weak",
        "adequate",
        "strong",
        "exceptional",
    ]
    reason: str
    missing_evidence: bool
    observed_cost_usd: float | None = None
    accounted_reserve_usd: float | None = None
    cost_status: Literal["observed", "unavailable"] = "unavailable"

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass(frozen=True)
class PairedAttemptV3:
    attempt_id: str
    identity: dict[str, Any]
    prediction_id: str | None
    passed: bool | None
    execution_status: str
    evaluation_status: str
    evidence_status: str
    cost_usd: float | None
    latency_sec: float | None
    input_tokens: float | None
    output_tokens: float | None
    tool_calls: int
    tools: tuple[str, ...]
    queried_projects: tuple[str, ...]
    scores: dict[str, Any]
    score_explanations: dict[str, str]
    judge_reviews: dict[str, JudgeReviewV1]
    sanitized_answer_excerpt: str | None
    actual_query_scope: tuple[str, ...]
    reported_project_identity: str | None
    evidence_links: tuple[AttemptEvidenceLinkV1, ...]
    weave_agent_root_call_id: str | None = None
    otel_root_span_id: str | None = None
    execution_fingerprint: str | None = None
    runtime_lock_digest: str | None = None
    infrastructure: dict[str, Any] = field(default_factory=dict)
    benchmark_outcome: Literal[
        "passed", "failed", "unscored", "not_applicable"
    ] | None = None
    runtime_outcome: Literal[
        "completed", "timed_out", "cancelled", "not_started", "not_applicable"
    ] | None = None

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(_json_value(asdict(self)), preserve_false=True)


@dataclass(frozen=True)
class PairedCaseV2:
    pair_id: str
    task_id: str
    harness: str
    attempt: int
    status: PairStatus
    dimension_changes: tuple[DimensionChangeV1, ...]
    baseline: PairedAttemptV2 | None
    candidate: PairedAttemptV2 | None
    task_label: str | None = None
    baseline_passed: bool | None = None
    candidate_passed: bool | None = None
    baseline_prediction_id: str | None = None
    candidate_prediction_id: str | None = None
    baseline_evaluation_call_id: str | None = None
    candidate_evaluation_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(_json_value(asdict(self)), preserve_false=True)


@dataclass(frozen=True)
class PairedCaseV3:
    pair_id: str
    task_id: str
    harness: str
    attempt: int
    status: PairStatus
    dimension_changes: tuple[DimensionChangeV2, ...]
    baseline: PairedAttemptV3 | None
    candidate: PairedAttemptV3 | None
    task_label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(_json_value(asdict(self)), preserve_false=True)


@dataclass(frozen=True)
class BehavioralSummaryV1:
    status: BehavioralStatus
    recommendation: str
    improved_pairs: int
    regressed_pairs: int
    mixed_pairs: int
    unchanged_pairs: int
    incomplete_pairs: int
    candidate_critical_failures: int
    critical_blockers: tuple[str, ...]
    supported_claim: str | None
    limitations: tuple[str, ...]
    next_action: str

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass(frozen=True)
class CandidateSourceRevisionV1:
    kind: str
    id: str
    version_identity: str
    runtime_digest: str
    lock_digest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass(frozen=True)
class ComparisonSpecV1:
    schema_version: int
    id: str
    question: str
    taskset: ComparisonTasksetV1
    baseline: ComparisonCandidateV1
    candidate: ComparisonCandidateV1
    changed: tuple[str, ...]
    evaluators: tuple[ComparisonEvaluatorV1, ...]
    execution: ComparisonExecutionPolicyV1
    decision_policy: DecisionPolicyV1 | None = None
    supersedes: tuple[SupersededResultV1, ...] = ()
    spec_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = _json_value(asdict(self))
        execution = value.get("execution")
        if isinstance(execution, dict):
            if not execution.get("environment"):
                execution.pop("environment", None)
            for field_name in (
                "evidence_project",
                "evidence_destination",
                "source_evidence_project",
                "source_evidence_destination",
                "study_console_base_url",
                "research_id",
                "infrastructure_receipt",
                "source_lock",
                "evidence_lock",
                "source_conformance_receipt",
                "release_notes_lock",
                "mechanism_receipt",
                "prerequisite_result",
                "prerequisite_attestation",
                "prerequisite_comparison_id",
                "prerequisite_spec",
            ):
                if execution.get(field_name) is None:
                    execution.pop(field_name, None)
            if execution.get("preparation_required") is False:
                execution.pop("preparation_required", None)
            if execution.get("evidence_checkpoint_cells") == 0:
                execution.pop("evidence_checkpoint_cells", None)
        for evaluator in value.get("evaluators") or ():
            if isinstance(evaluator, dict):
                if not evaluator.get("dimension_roles"):
                    evaluator.pop("dimension_roles", None)
                if evaluator.get("timeout_sec") is None:
                    evaluator.pop("timeout_sec", None)
        if value.get("decision_policy") is None:
            value.pop("decision_policy", None)
        if not value.get("supersedes"):
            value.pop("supersedes", None)
        return value


@dataclass(frozen=True)
class ComparisonReadinessV1:
    schema_version: int
    comparison_id: str
    question: str
    evidence_project: str | None
    task_count: int
    taskset_digest: str
    private_labels_digest: str
    actual_changes: tuple[str, ...]
    declared_changes: tuple[str, ...]
    base_failures: int
    gold_passes: int
    deterministic_evaluators: tuple[str, ...]
    judge_evaluators: tuple[str, ...]
    evaluator_digests: dict[str, str]
    attempts: int
    estimated_cells: int
    estimated_cost_usd: float
    status: Literal[
        "ready", "needs_review", "blocked", "no_comparison_justified"
    ]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    readiness_digest: str = ""
    runtime_lock_digests: dict[str, str] = field(default_factory=dict)
    qualification_input_digests: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComparisonPreviewV1:
    schema_version: int
    comparison: dict[str, Any]
    readiness: dict[str, Any]
    matrix: dict[str, Any]
    experiment: dict[str, Any]
    manifest: dict[str, Any]
    preview_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComparisonResultV1:
    schema_version: int
    comparison_id: str
    preview_digest: str
    source: str
    evidence_project: str | None
    rows: int
    baseline_passed: int
    candidate_passed: int
    improved: int
    regressed: int
    unchanged: int
    incomplete: int
    required_evaluations_incomplete: int
    deterministic_summary: dict[str, Any]
    judge_summary: dict[str, Any]
    mechanism_summary: dict[str, Any]
    operational_summary: dict[str, Any]
    evidence_links: tuple[dict[str, str], ...]
    paired_cases: tuple[dict[str, Any], ...]
    limitations: tuple[str, ...]
    result_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComparisonResultV2:
    schema_version: Literal[2]
    comparison_id: str
    preview_digest: str
    source: str
    evidence_project: str | None
    rows: int
    baseline_passed: int
    candidate_passed: int
    improved: int
    regressed: int
    mixed: int
    unchanged: int
    incomplete: int
    required_evaluations_incomplete: int
    deterministic_summary: dict[str, Any]
    judge_summary: dict[str, Any]
    mechanism_summary: dict[str, Any]
    operational_summary: dict[str, Any]
    evidence_links: tuple[dict[str, str], ...]
    paired_cases: tuple[PairedCaseV2, ...]
    limitations: tuple[str, ...]
    integrity: dict[str, Any]
    behavioral_summary: BehavioralSummaryV1
    decision_policy: dict[str, Any] | None
    decision: DecisionSummaryV1
    candidate_source_revisions: tuple[CandidateSourceRevisionV1, ...] = ()
    evidence_destination: dict[str, Any] = field(default_factory=dict)
    qualification_digest: str = ""
    result_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass(frozen=True)
class ComparisonResultV3:
    schema_version: Literal[3]
    comparison_id: str
    preview_digest: str
    source: str
    evidence_project: str | None
    rows: int
    baseline_passed: int
    candidate_passed: int
    improved: int
    regressed: int
    mixed: int
    unchanged: int
    incomplete: int
    required_evaluations_incomplete: int
    deterministic_summary: dict[str, Any]
    judge_summary: dict[str, Any]
    mechanism_summary: dict[str, Any]
    operational_summary: dict[str, Any]
    evidence_links: tuple[dict[str, str], ...]
    paired_cases: tuple[PairedCaseV3, ...]
    limitations: tuple[str, ...]
    integrity: dict[str, Any]
    behavioral_summary: BehavioralSummaryV1
    decision_policy: dict[str, Any] | None
    decision: DecisionSummaryV1
    evidence_topology: EvidenceTopologyV1
    aligned_analysis: AlignedAnalysisV1
    task_validity: tuple[TaskValidityV1, ...]
    release_note_coverage: tuple[dict[str, Any], ...]
    scorer_revisions: tuple[LockDescriptorV1, ...]
    runtime_locks: tuple[LockDescriptorV1, ...]
    cohort_lineage: dict[str, Any]
    supersedes: tuple[SupersededResultV1, ...] = ()
    candidate_source_revisions: tuple[CandidateSourceRevisionV1, ...] = ()
    evidence_destination: dict[str, Any] = field(default_factory=dict)
    qualification_digest: str = ""
    result_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != COMPARISON_RESULT_SCHEMA_VERSION:
            raise ValueError("unsupported ComparisonResultV3 schema version")
        if not self.paired_cases or not self.task_validity:
            raise ValueError(
                "ComparisonResultV3 requires nonzero paired cases and task validity"
            )
        if not self.scorer_revisions:
            raise ValueError("ComparisonResultV3 requires scorer revisions")
        if not self.runtime_locks:
            raise ValueError("ComparisonResultV3 requires runtime locks")
        _verify_cohort_lineage(self.cohort_lineage)
        if self.evidence_project != (
            self.evidence_topology.result_destination.project_slug
        ):
            raise ValueError(
                "ComparisonResultV3 evidence project disagrees with its topology"
            )

    def to_dict(self) -> dict[str, Any]:
        value = _json_value(asdict(self))
        value["evidence_topology"] = self.evidence_topology.to_dict()
        value["aligned_analysis"] = self.aligned_analysis.to_dict()
        value["task_validity"] = [
            item.to_dict() for item in self.task_validity
        ]
        value["scorer_revisions"] = [
            item.to_dict() for item in self.scorer_revisions
        ]
        value["runtime_locks"] = [
            item.to_dict() for item in self.runtime_locks
        ]
        value["supersedes"] = [item.to_dict() for item in self.supersedes]
        serialized = _drop_empty(value, preserve_false=True)
        serialized["deterministic_summary"] = dict(self.deterministic_summary)
        serialized["judge_summary"] = dict(self.judge_summary)
        serialized["mechanism_summary"] = dict(self.mechanism_summary)
        serialized["operational_summary"] = dict(self.operational_summary)
        serialized["evidence_links"] = [
            dict(item) for item in self.evidence_links
        ]
        serialized["release_note_coverage"] = [
            dict(item) for item in self.release_note_coverage
        ]
        serialized["supersedes"] = [
            item.to_dict() for item in self.supersedes
        ]
        return serialized


ComparisonResult = ComparisonResultV1 | ComparisonResultV2 | ComparisonResultV3
ComparisonPreviewV2 = ComparisonPreviewV1


@dataclass(frozen=True)
class _PairedAnalysis:
    v2: tuple[PairedCaseV2, ...]
    v3: tuple[PairedCaseV3, ...]
    baseline_passed: int
    candidate_passed: int
    improved: int
    regressed: int
    mixed: int
    unchanged: int
    incomplete: int


class ComparisonPublicationError(RuntimeError):
    """A declared Research projection failed at a governed publication boundary."""

    def __init__(
        self,
        *,
        stage: Literal["start", "result"],
        research_id: str,
        receipt_path: Path,
        error_type: str,
        result: ComparisonResult | None = None,
        result_path: Path | None = None,
        markdown_path: Path | None = None,
    ) -> None:
        super().__init__(
            f"Research {stage} publication is incomplete for {research_id} "
            f"({error_type}); see {receipt_path}"
        )
        self.stage = stage
        self.research_id = research_id
        self.receipt_path = receipt_path
        self.error_type = error_type
        self.result = result
        self.result_path = result_path
        self.markdown_path = markdown_path


@dataclass
class _DirectComparisonExecutionContext:
    spec: ComparisonSpecV1 | None = None
    environment: dict[str, str] = field(default_factory=dict)
    publication_path: Path | None = None
    start_projection: dict[str, Any] | None = None
    started: bool = False
    run_id: str | None = None
    run_status: str | None = None
    completed_cells: int = 0
    canonical_result_published: bool = False


def load_comparison(path: Path, *, repo_root: Path) -> ComparisonSpecV1:
    resolved = _safe_input_path(path, repo_root, "comparison")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("comparison YAML must be a mapping")
    return comparison_from_dict(raw, repo_root=repo_root, source=resolved.parent)


def comparison_from_dict(
    raw: Mapping[str, Any], *, repo_root: Path, source: Path | None = None
) -> ComparisonSpecV1:
    _reject_unknown(
        raw,
        {
            "schema_version",
            "id",
            "question",
            "taskset",
            "baseline",
            "candidate",
            "changed",
            "evaluators",
            "execution",
            "decision_policy",
            "supersedes",
            "spec_digest",
        },
        "comparison",
    )
    version = _schema(raw, "comparison")
    base = source or repo_root
    taskset_raw = _mapping(raw.get("taskset"), "taskset")
    _reject_unknown(taskset_raw, {"tasks", "private_labels"}, "taskset")
    taskset = ComparisonTasksetV1(
        tasks=_portable_input_path(taskset_raw.get("tasks"), base, repo_root, "tasks"),
        private_labels=_portable_input_path(
            taskset_raw.get("private_labels"),
            base,
            repo_root,
            "private labels",
        ),
    )
    parsed_evaluators = tuple(
        _evaluator(item)
        for item in _sequence(raw.get("evaluators"), "evaluators")
    )
    evaluators = tuple(
        replace(
            evaluator,
            scorer=(
                _portable_input_path(
                    evaluator.scorer,
                    base,
                    repo_root,
                    "deterministic scorer",
                )
                if evaluator.scorer
                else None
            ),
            verifier=(
                replace(
                    evaluator.verifier,
                    source=_portable_input_path(
                        evaluator.verifier.source,
                        base,
                        repo_root,
                        "host verifier",
                    ),
                )
                if evaluator.verifier
                else None
            ),
            calibration=(
                _portable_input_path(
                    evaluator.calibration,
                    base,
                    repo_root,
                    "judge calibration",
                )
                if evaluator.calibration
                else None
            ),
        )
        for evaluator in parsed_evaluators
    )
    if len({item.id for item in evaluators}) != len(evaluators):
        raise ValueError("comparison evaluator ids must be unique")
    if version >= 3:
        for evaluator in evaluators:
            if evaluator.dimensions and set(evaluator.dimension_roles) != set(
                evaluator.dimensions
            ):
                raise ValueError(
                    f"V3 evaluator {evaluator.id!r} must declare one typed role "
                    "for every dimension"
                )
    execution = _execution(
        raw.get("execution"),
        source=base,
        repo_root=repo_root,
    )
    if version >= 3 and (
        execution.evidence_project is None
        or execution.evidence_destination is None
        or execution.source_evidence_project is None
        or execution.source_evidence_destination is None
    ):
        raise ValueError(
            "V3 comparison requires exact source and result evidence destinations"
        )
    changed = _string_tuple(raw.get("changed"), "changed dimension")
    if len(set(changed)) != len(changed):
        raise ValueError("declared changed dimensions must be unique")
    supersedes = tuple(
        superseded_result_from_dict(
            _mapping(item, "superseded result")
        )
        for item in _sequence(
            raw.get("supersedes") or [],
            "superseded results",
            allow_empty=True,
        )
    )
    if supersedes and version < 3:
        raise ValueError("comparison supersession requires schema_version 3")
    if len({item.result_digest for item in supersedes}) != len(supersedes):
        raise ValueError("superseded result digests must be unique")
    unsigned = ComparisonSpecV1(
        schema_version=version,
        id=validate_id(str(raw.get("id") or ""), kind="comparison id"),
        question=_text(raw.get("question"), "comparison question", 1000),
        taskset=taskset,
        baseline=_candidate(raw.get("baseline"), "Baseline"),
        candidate=_candidate(raw.get("candidate"), "Candidate"),
        changed=changed,
        evaluators=evaluators,
        execution=execution,
        decision_policy=_decision_policy(raw.get("decision_policy")),
        supersedes=supersedes,
    )
    digest = _artifact_digest(unsigned.to_dict(), "spec_digest")
    supplied = str(raw.get("spec_digest") or "")
    if supplied and supplied != digest:
        raise ValueError("comparison spec digest does not match")
    return replace(unsigned, spec_digest=digest)


def _comparison_reserved_cost_per_attempt(spec: ComparisonSpecV1) -> float:
    return spec.execution.reserve_per_attempt_usd + sum(
        evaluator.reserve_cost_usd
        for evaluator in spec.evaluators
        if evaluator.type == "llm_judge"
    )


def check_comparison(
    spec: ComparisonSpecV1, *, repo_root: Path
) -> ComparisonReadinessV1:
    tasks = _load_public_tasks(repo_root / spec.taskset.tasks)
    labels = _load_private_labels(repo_root / spec.taskset.private_labels)
    actual_changes, blockers = _comparison_identity_issues(spec)
    warnings: list[str] = []
    infrastructure_receipt, infrastructure_blockers = (
        _infrastructure_readiness(spec, repo_root=repo_root)
    )
    if _local_behavior_study_has_independent_release_gates(spec):
        warnings.extend(
            "package-release gate does not block this local behavioral study: "
            + item
            for item in infrastructure_blockers
        )
    else:
        blockers.extend(infrastructure_blockers)
    local_source_digests, local_source_blockers = (
        _local_source_lock_readiness(spec, repo_root=repo_root)
    )
    declared_input_digests, declared_input_blockers = (
        _declared_qualification_input_readiness(spec, repo_root=repo_root)
    )
    qualification_input_digests, qualification_blockers = (
        _qualification_input_readiness(spec, repo_root=repo_root)
    )
    qualification_input_digests = {
        **local_source_digests,
        **declared_input_digests,
        **qualification_input_digests,
    }
    blockers.extend(local_source_blockers)
    blockers.extend(declared_input_blockers)
    blockers.extend(qualification_blockers)
    task_ids = tuple(str(item["id"]) for item in tasks)
    blockers.extend(_task_label_issues(task_ids, labels))
    asset_blockers, asset_warnings = _candidate_asset_readiness(
        spec, repo_root=repo_root
    )
    blockers.extend(asset_blockers)
    warnings.extend(asset_warnings)
    runtime_lock_digests, runtime_blockers = _runtime_readiness(
        spec, repo_root=repo_root
    )
    blockers.extend(runtime_blockers)
    preparation_digest, preparation_blockers = _preparation_receipt_readiness(
        spec,
        repo_root=repo_root,
        runtime_lock_digests=runtime_lock_digests,
        qualification_input_digests=qualification_input_digests,
    )
    blockers.extend(preparation_blockers)
    if preparation_digest:
        runtime_lock_digests = {
            **runtime_lock_digests,
            "comparison_preparation": preparation_digest,
        }
    deterministic = tuple(
        item.id for item in spec.evaluators if item.type == "deterministic"
    )
    judges = tuple(item.id for item in spec.evaluators if item.type == "llm_judge")
    if not deterministic:
        blockers.append("at least one deterministic evaluator is required")
    required_checks = {
        check for item in spec.evaluators if item.type == "deterministic" for check in item.checks
    }
    unsupported_checks = sorted(
        required_checks - {"answer_present", "expected_values"}
    )
    if unsupported_checks:
        blockers.append(
            "unsupported deterministic checks: " + ", ".join(unsupported_checks)
        )
    (
        base_failures,
        gold_passes,
        qualification_blockers,
        qualification_warnings,
    ) = (
        _qualification_results(
            tasks,
            labels,
            tuple(
                item
                for item in spec.evaluators
                if item.type == "deterministic"
            ),
            repo_root=repo_root,
        )
    )
    blockers.extend(qualification_blockers)
    warnings.extend(qualification_warnings)
    labels_by_id = {str(item["id"]): item for item in labels}
    if tasks and base_failures == 0 and all(
        "base_output" in labels_by_id.get(task_id, {}) for task_id in task_ids
    ):
        warnings.append("the baseline fixtures pass every task; the cohort is saturated")
    if spec.execution.attempts < 2:
        warnings.append(
            "one attempt cannot estimate ordinary run-to-run variation"
        )
    for judge in (item for item in spec.evaluators if item.type == "llm_judge"):
        issue = _judge_calibration_issue(judge, repo_root)
        if issue:
            (blockers if judge.required else warnings).append(issue)
        execution_issue = _judge_execution_calibration_issue(
            judge,
            repo_root=repo_root,
            approved_inputs=None,
        )
        if execution_issue and execution_issue != issue:
            warnings.append(
                "Agent execution is gated until " + execution_issue
            )
    estimated_cells = (
        len(tasks)
        * 2
        * len(spec.execution.harnesses)
        * spec.execution.attempts
    )
    estimated_cost = estimated_cells * _comparison_reserved_cost_per_attempt(spec)
    if estimated_cells < 1:
        blockers.append("comparison must resolve at least one attempt")
    if estimated_cost > spec.execution.max_cost_usd + 1e-9:
        blockers.append(
            f"estimated cost ${estimated_cost:.2f} exceeds the "
            f"${spec.execution.max_cost_usd:.2f} comparison limit"
        )
    if spec.execution.concurrency != 1:
        blockers.append(
            "runtime budget enforcement requires comparison concurrency=1"
        )
    if blockers:
        status = "blocked"
    elif tasks and base_failures == 0:
        status = "no_comparison_justified"
    elif warnings:
        status = "needs_review"
    else:
        status = "ready"
    unsigned = ComparisonReadinessV1(
        schema_version=COMPARISON_SCHEMA_VERSION,
        comparison_id=spec.id,
        question=spec.question,
        evidence_project=spec.execution.evidence_project,
        task_count=len(tasks),
        taskset_digest=_sha256_path(repo_root / spec.taskset.tasks),
        private_labels_digest=_sha256_path(
            repo_root / spec.taskset.private_labels
        ),
        actual_changes=actual_changes,
        declared_changes=spec.changed,
        base_failures=base_failures,
        gold_passes=gold_passes,
        deterministic_evaluators=deterministic,
        judge_evaluators=judges,
        evaluator_digests={
            item.id: _evaluator_digest(item, repo_root)
            for item in spec.evaluators
        }
        | (
            {
                "infrastructure_receipt": str(
                    infrastructure_receipt["receipt_digest"]
                )
            }
            if infrastructure_receipt is not None
            else {}
        ),
        attempts=spec.execution.attempts,
        estimated_cells=estimated_cells,
        estimated_cost_usd=round(estimated_cost, 6),
        status=status,  # type: ignore[arg-type]
        blockers=tuple(dict.fromkeys(blockers)),
        warnings=tuple(dict.fromkeys(warnings)),
        runtime_lock_digests=runtime_lock_digests,
        qualification_input_digests=qualification_input_digests,
    )
    return replace(
        unsigned,
        readiness_digest=_artifact_digest(
            unsigned.to_dict(), "readiness_digest"
        ),
    )


def _local_behavior_study_has_independent_release_gates(
    spec: ComparisonSpecV1,
) -> bool:
    """Keep package qualification from blocking a local behavior cohort.

    V3 local Docker comparisons can produce behavioral evidence while their
    governed package-release decision remains blocked by missing or incomplete
    infrastructure conformance. The infrastructure receipt, when usable, is
    still digest-bound into the preview and result. Decision analysis continues
    to require the declared release gates before it can produce ``go``.
    """
    return (
        spec.schema_version >= 3
        and spec.decision_policy is not None
        and str(spec.execution.environment.get("type") or "docker")
        == "docker"
    )


def _infrastructure_readiness(
    spec: ComparisonSpecV1, *, repo_root: Path
) -> tuple[dict[str, Any] | None, list[str]]:
    receipt_path = spec.execution.infrastructure_receipt
    if not receipt_path:
        blockers = (
            ["a governed release decision requires an infrastructure receipt"]
            if spec.decision_policy is not None
            else []
        )
        return None, blockers
    try:
        receipt = _load_infrastructure_receipt(
            repo_root / receipt_path,
            repo_root=repo_root,
        )
    except (FileNotFoundError, ValueError) as exc:
        return None, [f"infrastructure receipt is not usable: {exc}"]
    conformance = _mapping_or_empty(receipt.get("infrastructure_conformance"))
    if conformance.get("complete") is True:
        return receipt, []
    unavailable = ", ".join(
        str(item) for item in conformance.get("unavailable") or ()
    )
    failed = ", ".join(str(item) for item in conformance.get("failed") or ())
    detail = "; ".join(
        item
        for item in (
            f"failed: {failed}" if failed else "",
            f"unavailable: {unavailable}" if unavailable else "",
        )
        if item
    )
    blocker = "infrastructure conformance is incomplete"
    if detail:
        blocker += f" ({detail})"
    return receipt, [blocker]


def _bound_execution_infrastructure_receipt(
    spec: ComparisonSpecV1,
    *,
    readiness: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any] | None:
    """Resolve optional release evidence before any comparison cell can run."""
    receipt, blockers = _infrastructure_readiness(spec, repo_root=repo_root)
    evaluator_digests = _mapping_or_empty(readiness.get("evaluator_digests"))
    expected_digest = str(
        evaluator_digests.get("infrastructure_receipt") or ""
    )
    if receipt is None:
        if expected_digest:
            raise ValueError(
                "infrastructure receipt disappeared after preview; "
                "prepare and approve a new exact preview"
            )
        if blockers and not _local_behavior_study_has_independent_release_gates(
            spec
        ):
            raise ValueError(
                "infrastructure receipt is not execution-ready:\n- "
                + "\n- ".join(blockers)
            )
        return None
    observed_digest = str(receipt.get("receipt_digest") or "")
    if not expected_digest or observed_digest != expected_digest:
        raise ValueError(
            "infrastructure receipt changed after preview; "
            "prepare and approve a new exact preview"
        )
    return receipt


def _bound_v3_release_note_coverage(
    spec: ComparisonSpecV1,
    *,
    readiness: Mapping[str, Any],
    repo_root: Path,
) -> tuple[dict[str, Any], ...]:
    if spec.schema_version < 3 or not spec.execution.mechanism_receipt:
        return ()
    receipt_path = _safe_input_path(
        Path(spec.execution.mechanism_receipt),
        repo_root,
        "mechanism receipt",
    )
    receipt = _load_digest_receipt(receipt_path, "mechanism receipt")
    expected_inputs = _mapping_or_empty(
        readiness.get("qualification_input_digests")
    )
    if receipt.get("receipt_digest") != expected_inputs.get(
        "mechanism_receipt"
    ):
        raise ValueError(
            "mechanism receipt changed after preview; prepare and approve a "
            "new exact preview"
        )
    from fugue.bench.mcp_release_qualification import (
        release_note_coverage_v3,
    )

    task_ids = tuple(
        str(item["id"])
        for item in _load_public_tasks(repo_root / spec.taskset.tasks)
    )
    coverage = release_note_coverage_v3(
        receipt,
        task_ids=task_ids,
        dimension_ids=tuple(
            f"{evaluator.id}.{dimension}"
            for evaluator in spec.evaluators
            for dimension in evaluator.dimensions
        ),
    )
    if stable_digest([dict(item) for item in coverage]) != expected_inputs.get(
        "release_note_coverage"
    ):
        raise ValueError(
            "release-note coverage changed after preview; prepare and "
            "approve a new exact preview"
        )
    return coverage


def _verify_v3_source_drift(
    spec: ComparisonSpecV1,
    *,
    readiness: Mapping[str, Any],
    repo_root: Path,
    env: Mapping[str, str],
) -> EvidenceDriftCheckV1 | None:
    if spec.schema_version < 3:
        return None
    if spec.execution.source_lock:
        from fugue.bench.source_locks import verify_local_source_drift

        expected_digest = str(
            _mapping_or_empty(
                readiness.get("qualification_input_digests")
            ).get("source_lock")
            or ""
        )
        if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
            return EvidenceDriftCheckV1(
                status="unavailable",
                expected_digest="0" * 64,
                reason="the approved local source lock digest is unavailable",
            )
        return verify_local_source_drift(
            repo_root / spec.execution.source_lock,
            repo_root=repo_root,
            expected_source_project=str(
                spec.execution.source_evidence_project or ""
            ),
            expected_result_project=str(spec.execution.evidence_project or ""),
            expected_digest=expected_digest,
        )
    if not spec.execution.evidence_lock:
        return None
    evidence_path = _safe_input_path(
        Path(spec.execution.evidence_lock),
        repo_root,
        "evidence lock",
    )
    from fugue.bench.mcp_release_qualification import (
        verify_hosted_source_drift,
    )

    check = verify_hosted_source_drift(evidence_lock=evidence_path, env=env)
    expected_digest = str(
        _mapping_or_empty(
            readiness.get("qualification_input_digests")
        ).get("evidence_lock")
        or ""
    )
    if not expected_digest or check.expected_digest != expected_digest:
        raise ValueError(
            "source drift verification disagrees with the approved evidence "
            "lock"
        )
    return check


def _validate_source_conformance_binding(
    *,
    conformance: Mapping[str, Any],
    evidence: Mapping[str, Any],
    source_project: str,
    result_project: str,
    private_labels_path: Path,
) -> None:
    if (
        conformance.get("kind")
        != "mcp-release-hosted-source-conformance"
        or conformance.get("status") != "passed"
        or conformance.get("schema_version") != 2
    ):
        raise ValueError("source conformance receipt did not pass")
    if (
        conformance.get("source_project") != source_project
        or conformance.get("result_project") != result_project
    ):
        raise ValueError(
            "source conformance receipt project topology does not match"
        )
    if conformance.get("project_access") != {
        source_project: "PRIVATE",
        result_project: "PRIVATE",
    }:
        raise ValueError(
            "source conformance receipt does not prove private source and "
            "result projects"
        )
    expected_endpoints = {
        "api_base_url": "https://api.wandb.ai",
        "trace_base_url": "https://trace.wandb.ai",
    }
    expected_endpoint_binding = {
        **expected_endpoints,
        "endpoint_digest": stable_digest(expected_endpoints),
    }
    if conformance.get("endpoint_binding") != expected_endpoint_binding:
        raise ValueError(
            "source conformance receipt endpoint binding does not match "
            "the public-cloud topology"
        )
    if conformance.get("evidence_lock_digest") != evidence.get(
        "evidence_lock_digest"
    ):
        raise ValueError(
            "source conformance receipt evidence lock does not match"
        )
    if conformance.get("observed") != {
        "evaluation_roots": 2,
        "direct_children": 18,
        "predict_and_score_children": 16,
        "summarize_children": 2,
    }:
        raise ValueError(
            "source conformance receipt does not prove the locked "
            "18-child/16-prediction cohort"
        )
    locked_root_counts = {
        str(item.get("evaluation_call_id") or ""): {
            "Evaluation.predict_and_score": int(
                item.get("observed_predict_and_score_children") or 0
            ),
            "Evaluation.summarize": int(
                item.get("observed_summarize_children") or 0
            ),
        }
        for item in conformance.get("evaluation_roots") or ()
        if isinstance(item, Mapping)
        and str(item.get("evaluation_call_id") or "")
    }
    locked_root_ids = {
        str(item.get("call_id") or "")
        for item in _mapping(
            evidence.get("objects"),
            "evidence lock objects",
        ).get("evaluations")
        or ()
        if isinstance(item, Mapping)
        and str(item.get("call_id") or "")
    }
    private_by_id = {
        str(item["id"]): item
        for item in _load_private_labels(private_labels_path)
    }
    reconciliation = private_by_id.get(
        "maintainer-evaluation-reconciliation"
    )
    if reconciliation is None:
        return
    private_expected = _mapping(
        reconciliation.get("expected"),
        "reconciliation private expected values",
    )
    fixture_root_ids = set(
        _string_tuple(
            private_expected.get("evaluation_parent_ids") or (),
            "reconciliation Evaluation root",
        )
    )
    fixture_counts = _mapping(
        _mapping(
            private_expected.get("mechanism"),
            "reconciliation private mechanism",
        ).get("evaluation_parent_operation_counts"),
        "reconciliation private operation counts",
    )
    if (
        fixture_root_ids != locked_root_ids
        or fixture_counts != locked_root_counts
    ):
        raise ValueError(
            "reconciliation private truth disagrees with the approved "
            "evidence lock and source conformance receipt"
        )


def _validate_mechanism_reconciliation_binding(
    receipt: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any],
    baseline_id: str,
    candidate_id: str,
) -> None:
    findings = _mapping(
        receipt.get("findings"),
        "mechanism receipt findings",
    )
    if (
        findings.get("baseline_evaluation_rows_reconciled") is not False
        or findings.get("candidate_evaluation_rows_reconciled") is not True
    ):
        raise ValueError(
            "mechanism receipt does not reproduce the baseline "
            "reconciliation bug and repaired candidate"
        )
    root_ids = {
        str(item.get("call_id") or "")
        for item in _mapping(
            evidence.get("objects"),
            "evidence lock objects",
        ).get("evaluations")
        or ()
        if isinstance(item, Mapping)
        and str(item.get("call_id") or "")
    }
    candidates = {
        str(item.get("id") or ""): item
        for item in receipt.get("candidates") or ()
        if isinstance(item, Mapping)
    }
    for integration_id, repaired in (
        (baseline_id, False),
        (candidate_id, True),
    ):
        candidate = _mapping(
            candidates.get(integration_id),
            f"mechanism candidate {integration_id}",
        )
        rows = [
            _mapping(item, "mechanism reconciliation row")
            for item in candidate.get("evaluation_reconciliation") or ()
        ]
        if (
            len(rows) != 2
            or {
                str(item.get("evaluation_call_id") or "") for item in rows
            }
            != root_ids
            or any(
                int(item.get("locked_prediction_rows") or 0) != 8
                or int(
                    item.get("observed_predict_and_score_children") or 0
                )
                != 8
                or int(item.get("observed_summarize_children") or 0) != 1
                or item.get("trace_children_reconciled") is not True
                or item.get("prediction_rows_reconciled") is not repaired
                or int(item.get("tool_reported_total_predictions") or 0)
                != (8 if repaired else 9)
                for item in rows
            )
        ):
            raise ValueError(
                f"mechanism receipt {integration_id!r} does not prove the "
                "exact locked reconciliation behavior"
            )


def _validate_prerequisite_result_binding(
    spec: ComparisonSpecV1,
    *,
    repo_root: Path,
    source_lock_digest: str,
) -> dict[str, str]:
    if not spec.execution.prerequisite_result:
        return {}
    result_path = _safe_input_path(
        Path(spec.execution.prerequisite_result),
        repo_root,
        "prerequisite result",
    )
    attestation_path = _safe_input_path(
        Path(str(spec.execution.prerequisite_attestation)),
        repo_root,
        "prerequisite attestation",
    )
    result = read_comparison_result(result_path)
    if not isinstance(result, ComparisonResultV3):
        raise ValueError("prerequisite comparison result must be V3")
    if result.comparison_id != spec.execution.prerequisite_comparison_id:
        raise ValueError("prerequisite comparison identity does not match")
    prerequisite_spec = load_comparison(
        repo_root / str(spec.execution.prerequisite_spec),
        repo_root=repo_root,
    )
    if prerequisite_spec.id != result.comparison_id:
        raise ValueError(
            "prerequisite comparison spec does not match the result"
        )
    prerequisite_tasks = {
        str(item["id"])
        for item in _load_public_tasks(
            repo_root / prerequisite_spec.taskset.tasks
        )
    }
    if {item.task_id for item in result.task_validity} != prerequisite_tasks:
        raise ValueError(
            "prerequisite result tasks do not match its locked comparison spec"
        )
    expected_prerequisite_cells = (
        len(prerequisite_tasks)
        * 2
        * len(prerequisite_spec.execution.harnesses)
        * prerequisite_spec.execution.attempts
    )
    if result.rows != expected_prerequisite_cells:
        raise ValueError(
            "prerequisite result row count does not match its locked "
            "comparison matrix"
        )
    expected_lineage = _comparison_cohort_lineage(
        spec,
        repo_root=repo_root,
        source_lock_digest=source_lock_digest,
    )
    prerequisite_lineage = _comparison_cohort_lineage(
        prerequisite_spec,
        repo_root=repo_root,
        source_lock_digest=source_lock_digest,
    )
    if (
        _common_cohort_lineage(expected_lineage)
        != _common_cohort_lineage(prerequisite_lineage)
        or result.cohort_lineage != prerequisite_lineage
    ):
        raise ValueError(
            "prerequisite baseline, candidate, scorer, or execution lineage "
            "does not match the confirmation cohort"
        )
    if (
        result.behavioral_summary.status not in {"improved", "unchanged"}
        or result.behavioral_summary.candidate_critical_failures
        or result.behavioral_summary.critical_blockers
        or result.integrity.get("status") != "reconciled"
        or any(
            item.status in {"drifted", "invalid", "inconclusive"}
            for item in result.task_validity
        )
        or result.evidence_topology.pre_run_drift.status != "matched"
        or result.evidence_topology.post_run_drift.status != "matched"
        or result.evidence_topology.source_lock_digest != source_lock_digest
    ):
        raise ValueError(
            "prerequisite comparison is not valid, non-regressing, "
            "critical-dimension clean, reconciled, and source-locked"
        )
    candidate_sha = (
        spec.decision_policy.candidate_sha
        if spec.decision_policy is not None
        else ""
    )
    candidate_revisions = {
        item.id: item for item in result.candidate_source_revisions
    }
    for selection in spec.candidate.integrations:
        integration_id = str(selection["id"])
        lock_path = (
            repo_root
            / ".fugue/imports/mcp/locks"
            / f"{integration_id}.json"
        )
        lock = _load_json_object(
            _safe_input_path(
                lock_path,
                repo_root,
                f"prerequisite MCP lock {integration_id}",
            ),
            f"prerequisite MCP lock {integration_id}",
        )
        revision = candidate_revisions.get(integration_id)
        if (
            revision is None
            or revision.version_identity != lock.get("version_identity")
            or revision.runtime_digest != lock.get("runtime_digest")
            or (
                candidate_sha
                and revision.version_identity != f"git:{candidate_sha}"
            )
        ):
            raise ValueError(
                "prerequisite candidate identity does not match the "
                "confirmation candidate"
            )
    attestation = _load_digest_receipt(
        attestation_path,
        "prerequisite attestation",
    )
    _reject_unknown(
        attestation,
        {
            "schema_version",
            "kind",
            "comparison_id",
            "qualification_digest",
            "result_digest",
            "source_lock_digest",
            "runtime_locks_digest",
            "scorer_revisions_digest",
            "cohort_lineage_digest",
            "taskset_digest",
            "private_labels_digest",
            "review_status",
            "reviewed_by",
            "reviewed_at",
            "receipt_digest",
        },
        "prerequisite attestation",
    )
    expected_attestation = {
        "schema_version": 1,
        "kind": "comparison-prerequisite-attestation",
        "comparison_id": result.comparison_id,
        "qualification_digest": result.qualification_digest,
        "result_digest": result.result_digest,
        "source_lock_digest": source_lock_digest,
        "runtime_locks_digest": stable_digest(
            [item.to_dict() for item in result.runtime_locks]
        ),
        "scorer_revisions_digest": stable_digest(
            [item.to_dict() for item in result.scorer_revisions]
        ),
        "cohort_lineage_digest": str(
            result.cohort_lineage["lineage_digest"]
        ),
        "taskset_digest": _sha256_path(
            repo_root / prerequisite_spec.taskset.tasks
        ),
        "private_labels_digest": _sha256_path(
            repo_root / prerequisite_spec.taskset.private_labels
        ),
        "review_status": "accepted_valid_non_regressing_useful",
    }
    if any(
        attestation.get(key) != value
        for key, value in expected_attestation.items()
    ) or not str(attestation.get("reviewed_by") or "").strip() or not str(
        attestation.get("reviewed_at") or ""
    ).strip():
        raise ValueError(
            "prerequisite attestation does not sign the exact useful canary "
            "result"
        )
    return {
        "prerequisite_result": result.qualification_digest,
        "prerequisite_result_file": _sha256_path(result_path),
        "prerequisite_attestation": str(attestation["receipt_digest"]),
        "prerequisite_attestation_file": _sha256_path(attestation_path),
        "prerequisite_cohort_lineage": str(
            result.cohort_lineage["lineage_digest"]
        ),
    }


def authorize_comparison_followup(
    *,
    result_path: Path,
    followup_spec_path: Path,
    reviewed_by: str,
    reviewed_at: str,
    repo_root: Path,
) -> tuple[dict[str, Any], Path, Path]:
    """Bind one reviewed V3 result as the exact prerequisite for a follow-up.

    This is an operator action. It does not approve or run the follow-up. The
    generated receipt is accepted only after the same prerequisite validator
    used by comparison readiness has verified it against temporary immutable
    copies and again at the declared canonical paths.
    """

    root = repo_root.resolve()
    reviewer = reviewed_by.strip()
    timestamp = reviewed_at.strip()
    if not reviewer:
        raise ValueError("follow-up reviewer must not be empty")
    if not timestamp:
        raise ValueError("follow-up review time must not be empty")
    selected_result_path = _safe_input_path(
        result_path,
        root,
        "follow-up prerequisite result",
    )
    result = read_comparison_result(selected_result_path)
    if not isinstance(result, ComparisonResultV3):
        raise ValueError("follow-up prerequisite result must be V3")
    followup = load_comparison(followup_spec_path, repo_root=root)
    execution = followup.execution
    if not (
        execution.prerequisite_result
        and execution.prerequisite_attestation
        and execution.prerequisite_comparison_id
        and execution.prerequisite_spec
    ):
        raise ValueError(
            "follow-up comparison must declare its prerequisite result, "
            "attestation, comparison id, and spec"
        )
    prerequisite_spec_path = _safe_input_path(
        Path(execution.prerequisite_spec),
        root,
        "follow-up prerequisite comparison spec",
    )
    prerequisite_spec = load_comparison(
        prerequisite_spec_path,
        repo_root=root,
    )
    attestation = _comparison_prerequisite_attestation(
        result,
        prerequisite_spec=prerequisite_spec,
        reviewed_by=reviewer,
        reviewed_at=timestamp,
        repo_root=root,
    )
    temporary_root = root / COMPARISON_RUNTIME_ROOT / "followup-validation"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="authorize-",
        dir=temporary_root,
    ) as directory:
        temporary = Path(directory)
        temporary_result = temporary / "result.json"
        temporary_attestation = temporary / "attestation.json"
        atomic_write_json(temporary_result, result.to_dict())
        atomic_write_json(temporary_attestation, attestation)
        validation_spec = replace(
            followup,
            execution=replace(
                execution,
                prerequisite_result=temporary_result.relative_to(root).as_posix(),
                prerequisite_attestation=temporary_attestation.relative_to(
                    root
                ).as_posix(),
            ),
        )
        _validate_prerequisite_result_binding(
            validation_spec,
            repo_root=root,
            source_lock_digest=result.evidence_topology.source_lock_digest,
        )
    canonical_result = _safe_repository_output_path(
        Path(execution.prerequisite_result),
        root,
        "canonical prerequisite result",
    )
    canonical_attestation = _safe_repository_output_path(
        Path(execution.prerequisite_attestation),
        root,
        "canonical prerequisite attestation",
    )
    _write_consistent_json(
        canonical_result,
        result.to_dict(),
        label="canonical prerequisite result",
    )
    _write_consistent_json(
        canonical_attestation,
        attestation,
        label="canonical prerequisite attestation",
    )
    locked = _validate_prerequisite_result_binding(
        followup,
        repo_root=root,
        source_lock_digest=result.evidence_topology.source_lock_digest,
    )
    return (
        {
            "schema_version": 1,
            "kind": "comparison-followup-authorization",
            "comparison_id": result.comparison_id,
            "followup_comparison_id": followup.id,
            "qualification_digest": result.qualification_digest,
            "result_digest": result.result_digest,
            "reviewed_by": reviewer,
            "reviewed_at": timestamp,
            **locked,
        },
        canonical_result,
        canonical_attestation,
    )


def attest_comparison_decision(
    *,
    result_path: Path,
    signer: str,
    signed_at: str,
) -> ComparisonResultV3:
    """Attach an accepted-actionability sign-off to an exact V3 result.

    The unsigned qualification is archived beside the canonical result before
    the signed envelope replaces it. Re-analysis from final attempt rows must
    still reproduce the complete signed result.
    """

    result = read_comparison_result(result_path)
    if not isinstance(result, ComparisonResultV3):
        raise ValueError("release sign-off requires a ComparisonResultV3")
    if result.decision.status != "ready_for_signoff":
        raise ValueError(
            "release sign-off requires a result whose decision is "
            "ready_for_signoff"
        )
    selected_signer = signer.strip()
    timestamp = signed_at.strip()
    if not selected_signer:
        raise ValueError("release signer must not be empty")
    if not timestamp:
        raise ValueError("release sign-off time must not be empty")
    attestation = DecisionAttestationV1(
        signer=selected_signer,
        signed_result_digest=result.qualification_digest,
        signed_at=timestamp,
        review_status="accepted_actionable",
    )
    signed = replace(
        result,
        decision=_apply_decision_attestation(
            result.decision,
            attestation,
            qualification_digest=result.qualification_digest,
            require_actionability_review=True,
        ),
    )
    signed = replace(
        signed,
        result_digest=_comparison_result_digest(signed.to_dict()),
    )
    archive = result_path.with_name(
        f"result.unsigned-{result.qualification_digest[:16]}.json"
    )
    _write_consistent_json(
        archive,
        result.to_dict(),
        label="unsigned comparison result archive",
    )
    written, _ = write_comparison_result(
        signed,
        destination=result_path.parent,
    )
    reloaded = read_comparison_result(written)
    if not isinstance(reloaded, ComparisonResultV3):
        raise RuntimeError("signed comparison result did not remain V3")
    return reloaded


def _comparison_prerequisite_attestation(
    result: ComparisonResultV3,
    *,
    prerequisite_spec: ComparisonSpecV1,
    reviewed_by: str,
    reviewed_at: str,
    repo_root: Path,
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "kind": "comparison-prerequisite-attestation",
        "comparison_id": result.comparison_id,
        "qualification_digest": result.qualification_digest,
        "result_digest": result.result_digest,
        "source_lock_digest": result.evidence_topology.source_lock_digest,
        "runtime_locks_digest": stable_digest(
            [item.to_dict() for item in result.runtime_locks]
        ),
        "scorer_revisions_digest": stable_digest(
            [item.to_dict() for item in result.scorer_revisions]
        ),
        "cohort_lineage_digest": str(result.cohort_lineage["lineage_digest"]),
        "taskset_digest": _sha256_path(
            repo_root / prerequisite_spec.taskset.tasks
        ),
        "private_labels_digest": _sha256_path(
            repo_root / prerequisite_spec.taskset.private_labels
        ),
        "review_status": "accepted_valid_non_regressing_useful",
        "reviewed_by": reviewed_by,
        "reviewed_at": reviewed_at,
        "receipt_digest": "",
    }
    value["receipt_digest"] = stable_digest(value)
    return value


def _comparison_cohort_lineage(
    spec: ComparisonSpecV1,
    *,
    repo_root: Path,
    source_lock_digest: str,
) -> dict[str, Any]:
    from fugue.bench.sources import resolve_skill

    arms: dict[str, Any] = {}
    for arm, candidate in (
        ("baseline", spec.baseline),
        ("candidate", spec.candidate),
    ):
        revisions: list[dict[str, Any]] = []
        for skill_id in candidate.skills:
            skill = resolve_skill(skill_id, repo_root)
            revisions.append(
                {
                    "kind": "skill",
                    "id": skill.id,
                    "version_identity": (
                        f"git:{skill.resolved_commit}"
                        if skill.resolved_commit
                        else f"digest:{skill.digest}"
                    ),
                    "runtime_digest": skill.digest,
                }
            )
        for selection in candidate.integrations:
            integration_id = str(selection["id"])
            lock_path = (
                repo_root
                / ".fugue/imports/mcp/locks"
                / f"{integration_id}.json"
            )
            safe_lock_path = _safe_input_path(
                lock_path,
                repo_root,
                f"{arm} cohort integration lock {integration_id}",
            )
            lock = _load_json_object(
                safe_lock_path,
                f"{arm} cohort integration lock {integration_id}",
            )
            revisions.append(
                {
                    "kind": "integration",
                    "id": integration_id,
                    "version_identity": str(
                        lock.get("version_identity") or ""
                    ),
                    "runtime_digest": str(lock.get("runtime_digest") or ""),
                    "lock_digest": f"sha256:{_sha256_path(safe_lock_path)}",
                    "allowed_tools_digest": stable_digest(
                        sorted(str(item) for item in lock.get("allowed_tools") or ())
                    ),
                }
            )
        arms[arm] = {
            "behavior_digest": stable_digest(candidate.behavior()),
            "source_revisions": sorted(
                revisions,
                key=lambda item: (str(item.get("kind") or ""), str(item["id"])),
            ),
        }
    unsigned = {
        "schema_version": 1,
        "source_lock_digest": source_lock_digest,
        "taskset_digest": _sha256_path(
            repo_root / spec.taskset.tasks
        ),
        "private_labels_digest": _sha256_path(
            repo_root / spec.taskset.private_labels
        ),
        "arms": arms,
        "execution": {
            "model": spec.execution.model,
            "harnesses": list(spec.execution.harnesses),
            "trace_content": spec.execution.trace_content,
            "environment_digest": stable_digest(spec.execution.environment),
            "source_evidence_project": (
                spec.execution.source_evidence_project
            ),
            "source_evidence_destination": (
                spec.execution.source_evidence_destination.to_dict()
                if spec.execution.source_evidence_destination is not None
                else None
            ),
            "result_evidence_project": spec.execution.evidence_project,
            "result_evidence_destination": (
                spec.execution.evidence_destination.to_dict()
                if spec.execution.evidence_destination is not None
                else None
            ),
        },
        "scorer_digests": {
            evaluator.id: _evaluator_digest(evaluator, repo_root)
            for evaluator in spec.evaluators
        },
    }
    return {
        **unsigned,
        "lineage_digest": stable_digest(unsigned),
    }


def _verify_cohort_lineage(raw: Mapping[str, Any]) -> None:
    value = _mapping(raw, "comparison cohort lineage")
    _reject_unknown(
        value,
        {
            "schema_version",
            "source_lock_digest",
            "taskset_digest",
            "private_labels_digest",
            "arms",
            "execution",
            "scorer_digests",
            "lineage_digest",
        },
        "comparison cohort lineage",
    )
    if value.get("schema_version") != 1:
        raise ValueError("unsupported comparison cohort lineage schema")
    digest = str(value.get("lineage_digest") or "")
    unsigned = {
        key: value[key]
        for key in (
            "schema_version",
            "source_lock_digest",
            "taskset_digest",
            "private_labels_digest",
            "arms",
            "execution",
            "scorer_digests",
        )
    }
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or digest != stable_digest(
        unsigned
    ):
        raise ValueError("comparison cohort lineage digest does not match")
    source_lock_digest = str(value.get("source_lock_digest") or "")
    if source_lock_digest and not re.fullmatch(
        r"[0-9a-f]{64}", source_lock_digest
    ):
        raise ValueError(
            "comparison cohort lineage source_lock_digest must be an exact "
            "digest when present"
        )
    for key in ("taskset_digest", "private_labels_digest"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(value.get(key) or "")):
            raise ValueError(
                f"comparison cohort lineage {key} must be an exact digest"
            )


def _common_cohort_lineage(raw: Mapping[str, Any]) -> dict[str, Any]:
    value = _mapping(raw, "comparison cohort lineage")
    return {
        key: value[key]
        for key in (
            "schema_version",
            "source_lock_digest",
            "arms",
            "execution",
            "scorer_digests",
        )
    }


def _bind_locked_release_mcp_profiles(
    *,
    repo_root: Path,
    integration_ids: Sequence[str],
    receipt_candidates: Mapping[str, Any],
    digests: dict[str, str],
) -> None:
    for integration_id in integration_ids:
        lock_path = (
            repo_root
            / ".fugue/imports/mcp/locks"
            / f"{integration_id}.json"
        )
        lock = _load_json_object(
            _safe_input_path(
                lock_path,
                repo_root,
                f"MCP lock {integration_id}",
            ),
            f"MCP lock {integration_id}",
        )
        fixed_env = _mapping(
            lock.get("fixed_env"),
            f"MCP lock {integration_id} fixed environment",
        )
        allowed_tools = set(
            _string_tuple(
                lock.get("allowed_tools"),
                f"MCP lock {integration_id} allowed tool",
            )
        )
        manifest_tools = {
            str(item.get("name") or "")
            for item in lock.get("tool_manifest") or ()
            if isinstance(item, Mapping) and str(item.get("name") or "")
        }
        if (
            fixed_env.get("WANDB_MCP_READ_ONLY") != "true"
            or allowed_tools != _MCP_RELEASE_READ_ONLY_TOOLS
            or not _MCP_RELEASE_READ_ONLY_TOOLS <= manifest_tools
        ):
            raise ValueError(
                f"MCP lock {integration_id!r} is not the exact read-only "
                "release qualification profile"
            )
        observed = receipt_candidates.get(integration_id)
        if not isinstance(observed, Mapping):
            raise ValueError(
                f"mechanism receipt is missing candidate {integration_id!r}"
            )
        for field_name in (
            "version_identity",
            "runtime_digest",
            "tool_manifest_digest",
        ):
            if observed.get(field_name) != lock.get(field_name):
                raise ValueError(
                    f"mechanism receipt {integration_id!r} {field_name} "
                    "does not match its MCP lock"
                )
        if observed.get("initialized_manifest_matches_lock") is not True:
            raise ValueError(
                f"mechanism receipt {integration_id!r} tool manifest was "
                "not verified"
            )
        runtime_digest = str(lock.get("runtime_digest") or "")
        manifest_digest = str(lock.get("tool_manifest_digest") or "")
        for label, digest in (
            ("runtime", runtime_digest),
            ("tool manifest", manifest_digest),
        ):
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
                raise ValueError(
                    f"MCP lock {integration_id!r} {label} digest is invalid"
                )
        digests[f"mcp_lock:{integration_id}"] = _sha256_path(lock_path)
        digests[f"mcp_runtime:{integration_id}"] = (
            runtime_digest.removeprefix("sha256:")
        )
        digests[f"mcp_tool_manifest:{integration_id}"] = (
            manifest_digest.removeprefix("sha256:")
        )


def _validate_release_candidate_binding(
    spec: ComparisonSpecV1,
    *,
    release_notes: Mapping[str, Any],
    repo_root: Path,
) -> None:
    policy = spec.decision_policy
    if policy is None:
        raise ValueError(
            "MCP release qualification requires a governed decision policy"
        )
    candidate_sha = policy.candidate_sha
    if release_notes.get("commit") != candidate_sha:
        raise ValueError(
            "decision-policy candidate SHA does not match the release-notes lock"
        )
    candidate_integrations = tuple(
        str(selection["id"]) for selection in spec.candidate.integrations
    )
    if len(candidate_integrations) != 1:
        raise ValueError(
            "MCP release qualification requires exactly one candidate integration"
        )
    integration_id = candidate_integrations[0]
    lock_path = (
        repo_root
        / ".fugue/imports/mcp/locks"
        / f"{integration_id}.json"
    )
    lock = _load_json_object(
        _safe_input_path(
            lock_path,
            repo_root,
            f"MCP release candidate lock {integration_id}",
        ),
        f"MCP release candidate lock {integration_id}",
    )
    if lock.get("version_identity") != f"git:{candidate_sha}":
        raise ValueError(
            "decision-policy candidate SHA does not match the candidate MCP lock"
        )


def _local_source_lock_readiness(
    spec: ComparisonSpecV1,
    *,
    repo_root: Path,
) -> tuple[dict[str, str], list[str]]:
    """Bind immutable local task/source inputs for non-hosted V3 studies."""

    if not spec.execution.source_lock:
        if spec.schema_version >= 3 and not spec.execution.evidence_lock:
            return {}, [
                "V3 comparison requires a local source lock or hosted evidence lock"
            ]
        return {}, []
    if spec.execution.evidence_lock:
        return {}, [
            "comparison cannot declare local and hosted source locks together"
        ]
    try:
        from fugue.bench.source_locks import read_local_source_lock

        lock_path = _safe_input_path(
            Path(spec.execution.source_lock),
            repo_root,
            "local source lock",
        )
        lock = read_local_source_lock(
            lock_path,
            repo_root=repo_root,
            expected_source_project=str(
                spec.execution.source_evidence_project or ""
            ),
            expected_result_project=str(spec.execution.evidence_project or ""),
            verify_files=True,
        )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        return {}, [f"local source lock is not usable: {exc}"]
    return {
        "source_lock": str(lock["source_lock_digest"]),
        "source_lock_file": _sha256_path(lock_path),
    }, []


def _qualification_input_readiness(
    spec: ComparisonSpecV1,
    *,
    repo_root: Path,
) -> tuple[dict[str, str], list[str]]:
    """Verify optional MCP release evidence without treating it as release CI.

    These inputs qualify the mechanism exercised by a behavioral comparison.
    They are deliberately distinct from ``infrastructure_receipt``: a local
    Harbor study may bind exact hosted evidence and MCP probes while leaving
    the package-release infrastructure decision unevaluated.
    """
    declared = {
        "evidence_lock": spec.execution.evidence_lock,
        "source_conformance_receipt": (
            spec.execution.source_conformance_receipt
        ),
        "release_notes_lock": spec.execution.release_notes_lock,
        "mechanism_receipt": spec.execution.mechanism_receipt,
    }
    if not any(declared.values()):
        return {}, []
    if spec.schema_version >= 3 and (
        spec.execution.source_evidence_project is None
        or spec.execution.source_evidence_destination is None
    ):
        return {}, [
            "V3 mechanism qualification requires a distinct immutable "
            "source evidence destination"
        ]
    if (
        spec.schema_version >= 3
        and spec.execution.source_evidence_project
        == spec.execution.evidence_project
    ):
        return {}, [
            "MCP mechanism qualification requires source and result evidence "
            "projects to be different"
        ]
    missing = [name for name, value in declared.items() if not value]
    if missing:
        return {}, [
            "comparison mechanism qualification must declare "
            + ", ".join(sorted(declared))
            + " together"
        ]

    try:
        evidence_path = _safe_input_path(
            Path(str(declared["evidence_lock"])),
            repo_root,
            "evidence lock",
        )
        release_path = _safe_input_path(
            Path(str(declared["release_notes_lock"])),
            repo_root,
            "release-notes lock",
        )
        conformance_path = _safe_input_path(
            Path(str(declared["source_conformance_receipt"])),
            repo_root,
            "source conformance receipt",
        )
        receipt_path = _safe_input_path(
            Path(str(declared["mechanism_receipt"])),
            repo_root,
            "mechanism receipt",
        )
        evidence_raw = _load_json_object(evidence_path, "evidence lock")
        release_raw = _load_json_object(release_path, "release-notes lock")
        conformance = _load_digest_receipt(
            conformance_path,
            "source conformance receipt",
        )
        receipt = _load_digest_receipt(receipt_path, "mechanism receipt")

        from fugue.bench.mcp_release_qualification import (
            release_note_coverage_v3,
            tool_surface_coverage_v1,
            validate_evidence_lock,
            validate_release_notes_lock,
        )

        source_project = (
            spec.execution.source_evidence_project
            or spec.execution.evidence_project
            or ""
        )
        result_project = spec.execution.evidence_project or ""
        evidence = validate_evidence_lock(
            evidence_raw,
            expected_source_project=str(source_project),
            expected_result_project=str(result_project),
        )
        release = validate_release_notes_lock(release_raw)
        _validate_release_candidate_binding(
            spec,
            release_notes=release,
            repo_root=repo_root,
        )
        prerequisite_digests = _validate_prerequisite_result_binding(
            spec,
            repo_root=repo_root,
            source_lock_digest=str(evidence["evidence_lock_digest"]),
        )
        _validate_source_conformance_binding(
            conformance=conformance,
            evidence=evidence,
            source_project=str(source_project),
            result_project=str(result_project),
            private_labels_path=repo_root / spec.taskset.private_labels,
        )
        if receipt.get("kind") != "mcp-release-mechanism-qualification":
            raise ValueError("mechanism receipt kind does not match")
        if (
            receipt.get("project") != source_project
            or receipt.get("source_project") != source_project
        ):
            raise ValueError("mechanism receipt source project does not match")
        if receipt.get("result_project") != result_project:
            raise ValueError("mechanism receipt result project does not match")
        expected_endpoints = {
            "api_base_url": "https://api.wandb.ai",
            "trace_base_url": "https://trace.wandb.ai",
        }
        if receipt.get("endpoint_binding") != {
            **expected_endpoints,
            "endpoint_digest": stable_digest(expected_endpoints),
        }:
            raise ValueError(
                "mechanism receipt endpoint binding does not match the "
                "approved public-cloud topology"
            )
        if receipt.get("evidence_lock_digest") != evidence.get(
            "evidence_lock_digest"
        ):
            raise ValueError("mechanism receipt evidence lock does not match")
        if receipt.get("release_notes_lock") != release:
            raise ValueError("mechanism receipt release-notes lock does not match")

        receipt_candidates = {
            str(item.get("id") or ""): item
            for item in _sequence(
                receipt.get("candidates"),
                "mechanism receipt candidates",
            )
            if isinstance(item, Mapping)
        }
        integration_ids = tuple(
            dict.fromkeys(
                str(selection["id"])
                for candidate in (spec.baseline, spec.candidate)
                for selection in candidate.integrations
            )
        )
        if len(integration_ids) == 2:
            _validate_mechanism_reconciliation_binding(
                receipt,
                evidence=evidence,
                baseline_id=integration_ids[0],
                candidate_id=integration_ids[1],
            )
        digests = {
            "evidence_lock": str(evidence["evidence_lock_digest"]),
            "evidence_lock_file": _sha256_path(evidence_path),
            "source_conformance_receipt": str(
                conformance["receipt_digest"]
            ),
            "source_conformance_receipt_file": _sha256_path(
                conformance_path
            ),
            "release_notes_lock": _sha256_path(release_path),
            "mechanism_receipt": str(receipt["receipt_digest"]),
            **prerequisite_digests,
        }
        task_ids = tuple(
            str(item["id"])
            for item in _load_public_tasks(repo_root / spec.taskset.tasks)
        )
        coverage = release_note_coverage_v3(
            receipt,
            task_ids=task_ids,
            dimension_ids=tuple(
                f"{evaluator.id}.{dimension}"
                for evaluator in spec.evaluators
                for dimension in evaluator.dimensions
            ),
        )
        digests["release_note_coverage"] = stable_digest(
            [dict(item) for item in coverage]
        )
        tool_coverage = tool_surface_coverage_v1(
            receipt,
            task_ids=task_ids,
        )
        digests["mcp_tool_surface_coverage"] = str(tool_coverage["coverage_digest"])
        _bind_locked_release_mcp_profiles(
            repo_root=repo_root,
            integration_ids=integration_ids,
            receipt_candidates=receipt_candidates,
            digests=digests,
        )
        return dict(sorted(digests.items())), []
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        return {}, [f"mechanism qualification inputs are not usable: {exc}"]


def _declared_qualification_input_readiness(
    spec: ComparisonSpecV1,
    *,
    repo_root: Path,
) -> tuple[dict[str, str], list[str]]:
    if not spec.execution.qualification_inputs:
        return {}, []
    source_files: dict[str, str] = {}
    if spec.execution.source_lock:
        try:
            from fugue.bench.source_locks import read_local_source_lock

            source_lock = read_local_source_lock(
                repo_root / spec.execution.source_lock,
                repo_root=repo_root,
                expected_source_project=str(
                    spec.execution.source_evidence_project or ""
                ),
                expected_result_project=str(spec.execution.evidence_project or ""),
                verify_files=True,
            )
            source_files = {
                str(item["path"]): str(item["sha256"])
                for item in source_lock["files"]
            }
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
            return {}, [f"declared qualification source lock is unusable: {exc}"]
    digests: dict[str, str] = {}
    blockers: list[str] = []
    for name, relative in sorted(spec.execution.qualification_inputs.items()):
        try:
            path = _safe_input_path(
                Path(relative),
                repo_root,
                f"qualification input {name}",
            )
            digest = _sha256_path(path)
        except (FileNotFoundError, OSError, ValueError) as exc:
            blockers.append(f"qualification input {name!r} is unusable: {exc}")
            continue
        if source_files and source_files.get(relative) != digest:
            blockers.append(
                f"qualification input {name!r} is not bound by the local source lock"
            )
            continue
        digests[name] = digest
    return dict(sorted(digests.items())), blockers


def _candidate_asset_readiness(
    spec: ComparisonSpecV1, *, repo_root: Path
) -> tuple[list[str], list[str]]:
    from fugue.bench.integrations import load_integration
    from fugue.bench.sources import resolve_skill

    blockers: list[str] = []
    warnings: list[str] = []
    for candidate_name, candidate in (
        ("baseline", spec.baseline),
        ("candidate", spec.candidate),
    ):
        for skill_id in candidate.skills:
            try:
                resolve_skill(skill_id, repo_root)
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                blockers.append(
                    f"{candidate_name} Skill {skill_id!r} is not locked and usable: {exc}"
                )
        for selection in candidate.integrations:
            integration_id = str(selection["id"])
            try:
                integration = load_integration(integration_id, repo_root)
            except (FileNotFoundError, ValueError) as exc:
                blockers.append(
                    f"{candidate_name} integration {integration_id!r} is not locked "
                    f"and usable: {exc}"
                )
                continue
            if integration.support != "supported":
                warnings.append(
                    f"{candidate_name} integration {integration_id!r} is "
                    f"{integration.support}; its evidence is exploratory"
                )
    return blockers, warnings


def _runtime_readiness(
    spec: ComparisonSpecV1, *, repo_root: Path
) -> tuple[dict[str, str], list[str]]:
    environment_type = str(spec.execution.environment.get("type") or "docker")
    if environment_type == "docker" and spec.execution.preparation_required:
        return _local_runtime_readiness(spec, repo_root=repo_root)
    if environment_type != "wandb":
        return {}, []

    from fugue.bench.wandb_sandbox import wandb_execution_identity

    digests: dict[str, str] = {}
    blockers: list[str] = []
    for harness in spec.execution.harnesses:
        try:
            identity = wandb_execution_identity(
                spec.execution.environment,
                harness=harness,
                repo_root=repo_root,
            )
        except (FileNotFoundError, ValueError) as exc:
            blockers.append(
                f"W&B Serverless runtime for {harness!r} is not locked "
                f"and usable: {exc}"
            )
            continue
        if identity is None:
            blockers.append(
                f"W&B Serverless runtime for {harness!r} did not resolve"
            )
            continue
        digests[harness] = str(identity["lock_digest"])
    return digests, blockers


def _local_runtime_readiness(
    spec: ComparisonSpecV1,
    *,
    repo_root: Path,
) -> tuple[dict[str, str], list[str]]:
    """Verify the exact local Agent and task images selected by the compiler."""
    from fugue.bench.agent_runtime import (
        read_runtime_lock,
        runtime_ready,
        runtime_spec,
    )
    from fugue.bench.manifest import load_manifest
    from fugue.bench.task_runtime import (
        read_task_runtime_lock,
        task_architecture,
        task_runtime_lock_digest,
        task_runtime_ready,
    )

    try:
        experiment, raw_manifest, _public_rows = compile_comparison(
            spec,
            repo_root=repo_root,
        )
        manifest = load_manifest(
            repo_root / experiment.manifest,
            text=yaml.safe_dump(raw_manifest, sort_keys=False),
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        return {}, [f"local comparison runtime cannot be resolved: {exc}"]

    digests: dict[str, str] = {}
    blockers: list[str] = []
    evaluator_digests, evaluator_blockers = _evaluator_runtime_readiness(
        spec,
        repo_root=repo_root,
    )
    digests.update(evaluator_digests)
    blockers.extend(evaluator_blockers)
    architectures = sorted(
        {task_architecture(task) for task in manifest.tasks}
    )
    for harness in spec.execution.harnesses:
        if runtime_spec(harness) is None:
            continue
        for architecture in architectures:
            ready, detail = runtime_ready(harness, repo_root, architecture)
            lock = read_runtime_lock(harness, repo_root, architecture)
            key = f"agent:{harness}:{architecture}"
            if not ready or lock is None:
                blockers.append(
                    f"local {key} is not prepared and locked: {detail}"
                )
            else:
                digests[key] = stable_digest(lock)
    for task in manifest.tasks:
        architecture = task_architecture(task)
        ready, detail = task_runtime_ready(manifest, task, repo_root)
        lock = read_task_runtime_lock(manifest, task, repo_root)
        key = f"task:{task.id}:{architecture}"
        if not ready or lock is None:
            blockers.append(
                f"local {key} is not prepared and locked: {detail}"
            )
        else:
            try:
                digests[key] = task_runtime_lock_digest(
                    lock,
                    repo_root=repo_root,
                )
            except ValueError as exc:
                blockers.append(f"local {key} lock is not portable: {exc}")
    return dict(sorted(digests.items())), blockers


def _comparison_evaluator_runtime_profiles(
    spec: ComparisonSpecV1,
    *,
    repo_root: Path,
) -> tuple[Any, ...]:
    """Resolve every trusted scorer/verifier runtime used by one comparison."""
    from fugue.bench.task_authoring import load_task_profiles

    profile_ids = {
        runtime_id
        for evaluator in spec.evaluators
        for runtime_id in (
            evaluator.runtime if evaluator.scorer else None,
            evaluator.verifier.runtime if evaluator.verifier else None,
        )
        if runtime_id
    }
    if not profile_ids:
        return ()
    catalog = load_task_profiles(repo_root)
    return tuple(catalog.scorer_runtime(item) for item in sorted(profile_ids))


def _evaluator_runtime_lock_path(
    profile: Any,
    *,
    repo_root: Path,
) -> Path:
    platform = str(profile.platform).replace("/", "-")
    return (
        repo_root
        / COMPARISON_EVALUATOR_RUNTIME_ROOT
        / str(profile.id)
        / f"{profile.profile_digest}-{platform}.json"
    )


def _read_evaluator_runtime_lock(
    profile: Any,
    *,
    repo_root: Path,
    inspect_image: bool,
) -> dict[str, Any] | None:
    path = _evaluator_runtime_lock_path(profile, repo_root=repo_root)
    if not path.is_file() or path.is_symlink():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    required = {
        "schema_version": 1,
        "kind": "comparison_evaluator_runtime",
        "profile_id": profile.id,
        "profile_digest": profile.profile_digest,
        "image": profile.image,
        "platform": profile.platform,
        "command": list(profile.command),
    }
    if not isinstance(value, dict) or any(
        value.get(key) != expected for key, expected in required.items()
    ):
        return None
    unsigned = {key: item for key, item in value.items() if key != "lock_digest"}
    if value.get("lock_digest") != stable_digest(unsigned):
        return None
    image_id = str(value.get("image_id") or "")
    if not image_id.startswith("sha256:"):
        return None
    expected_os, expected_architecture = str(profile.platform).split("/", 1)
    if (
        value.get("os") != expected_os
        or value.get("architecture") != expected_architecture
    ):
        return None
    if inspect_image:
        try:
            inspected = inspect_docker_image(image_id)
        except (OSError, RuntimeError, subprocess.SubprocessError):
            return None
        if (
            inspected.get("Id") != image_id
            or inspected.get("Os") != expected_os
            or inspected.get("Architecture") != expected_architecture
        ):
            return None
    return value


def _evaluator_runtime_readiness(
    spec: ComparisonSpecV1,
    *,
    repo_root: Path,
) -> tuple[dict[str, str], list[str]]:
    digests: dict[str, str] = {}
    blockers: list[str] = []
    try:
        profiles = _comparison_evaluator_runtime_profiles(
            spec,
            repo_root=repo_root,
        )
    except (FileNotFoundError, ValueError) as exc:
        return {}, [f"comparison evaluator runtime cannot be resolved: {exc}"]
    for profile in profiles:
        lock = _read_evaluator_runtime_lock(
            profile,
            repo_root=repo_root,
            inspect_image=True,
        )
        key = f"evaluator:{profile.id}:{profile.platform}"
        if lock is None:
            blockers.append(
                f"local {key} is not prepared and locked; run "
                "`fugue compare SPEC --prepare`"
            )
        else:
            digests[key] = str(lock["lock_digest"])
    return dict(sorted(digests.items())), blockers


def _prepare_evaluator_runtimes(
    spec: ComparisonSpecV1,
    *,
    repo_root: Path,
) -> tuple[dict[str, str], ...]:
    """Materialize exact evaluator images in the trusted preparation boundary."""
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError("Docker is required to prepare evaluator runtimes")
    locks: list[dict[str, str]] = []
    for profile in _comparison_evaluator_runtime_profiles(
        spec,
        repo_root=repo_root,
    ):
        existing = _read_evaluator_runtime_lock(
            profile,
            repo_root=repo_root,
            inspect_image=True,
        )
        if existing is not None:
            locks.append(existing)
            continue
        completed = subprocess.run(
            [
                docker,
                "pull",
                "--platform",
                str(profile.platform),
                str(profile.image),
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=900,
            env={"PATH": os.environ.get("PATH", "")},
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"could not prepare evaluator runtime {profile.id}: "
                + (completed.stderr or completed.stdout or "docker pull failed").strip()
            )
        inspected = inspect_docker_image(str(profile.image))
        expected_os, expected_architecture = str(profile.platform).split("/", 1)
        if (
            inspected.get("Os") != expected_os
            or inspected.get("Architecture") != expected_architecture
        ):
            raise RuntimeError(
                f"evaluator runtime {profile.id} resolved "
                f"{inspected.get('Os')}/{inspected.get('Architecture')} instead of "
                f"{profile.platform}"
            )
        unsigned: dict[str, Any] = {
            "schema_version": 1,
            "kind": "comparison_evaluator_runtime",
            "profile_id": profile.id,
            "profile_digest": profile.profile_digest,
            "image": profile.image,
            "image_id": inspected.get("Id"),
            "platform": profile.platform,
            "os": inspected.get("Os"),
            "architecture": inspected.get("Architecture"),
            "command": list(profile.command),
        }
        lock = {**unsigned, "lock_digest": stable_digest(unsigned)}
        atomic_write_json(
            _evaluator_runtime_lock_path(profile, repo_root=repo_root),
            lock,
        )
        locks.append(lock)
    return tuple(locks)


def _comparison_preparation_receipt_path(
    spec: ComparisonSpecV1,
    *,
    repo_root: Path,
) -> Path:
    return (
        repo_root
        / COMPARISON_RUNTIME_ROOT
        / spec.spec_digest
        / "preparation.json"
    )


def _preparation_receipt_readiness(
    spec: ComparisonSpecV1,
    *,
    repo_root: Path,
    runtime_lock_digests: Mapping[str, str],
    qualification_input_digests: Mapping[str, str],
) -> tuple[str | None, list[str]]:
    if not spec.execution.preparation_required:
        return None, []
    path = _comparison_preparation_receipt_path(spec, repo_root=repo_root)
    try:
        receipt = _load_digest_receipt(path, "comparison preparation receipt")
        if receipt.get("kind") != "comparison_preparation":
            raise ValueError("comparison preparation receipt kind does not match")
        if receipt.get("spec_digest") != spec.spec_digest:
            raise ValueError("comparison preparation receipt spec does not match")
        if receipt.get("runtime_lock_digests") != dict(
            sorted(runtime_lock_digests.items())
        ):
            raise ValueError(
                "comparison preparation receipt runtime locks do not match"
            )
        if receipt.get("qualification_input_digests") != dict(
            sorted(qualification_input_digests.items())
        ):
            raise ValueError(
                "comparison preparation receipt qualification inputs do not match"
            )
        inputs = _mapping(
            receipt.get("frozen_inputs"),
            "comparison preparation frozen inputs",
        )
        if receipt.get("frozen_inputs_digest") != stable_digest(inputs):
            raise ValueError(
                "comparison preparation frozen input digest does not match"
            )
        _verify_frozen_comparison_inputs(inputs, repo_root=repo_root)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        return None, [
            "comparison preparation is missing or drifted; run "
            f"`fugue compare SPEC --prepare`: {exc}"
        ]
    return str(receipt["receipt_digest"]), []


def _comparison_identity_issues(
    spec: ComparisonSpecV1,
) -> tuple[tuple[str, ...], list[str]]:
    actual = tuple(
        sorted(_behavior_diff(spec.baseline.behavior(), spec.candidate.behavior()))
    )
    blockers: list[str] = []
    if not actual:
        blockers.append("baseline and candidate have identical behavior")
    if set(actual) != set(spec.changed):
        blockers.append(
            "declared candidate changes do not match the resolved behavior diff "
            f"(declared={list(spec.changed)}, actual={list(actual)})"
        )
    return actual, blockers


def _task_label_issues(
    task_ids: Sequence[str], labels: Sequence[Mapping[str, Any]]
) -> list[str]:
    label_ids = {str(item["id"]) for item in labels}
    task_set = set(task_ids)
    issues: list[str] = []
    missing = sorted(task_set - label_ids)
    extra = sorted(label_ids - task_set)
    if missing:
        issues.append("private labels are missing tasks: " + ", ".join(missing))
    if extra:
        issues.append("private labels reference unknown tasks: " + ", ".join(extra))
    return issues


def _qualification_results(
    tasks: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    evaluators: Sequence[ComparisonEvaluatorV1],
    *,
    repo_root: Path,
) -> tuple[int, int, list[str], list[str]]:
    values = {str(item["id"]): item for item in labels}
    blockers: list[str] = []
    warnings: list[str] = []
    base_failures = 0
    gold_passes = 0
    for task in tasks:
        task_id = str(task["id"])
        label = values.get(task_id)
        if label is None:
            continue
        if "base_output" not in label:
            warnings.append(f"{task_id}: missing base_output qualification fixture")
        else:
            try:
                base_passed, _, _ = _score_deterministic_output(
                    task=task,
                    output=label["base_output"],
                    expected=label.get("expected"),
                    evidence=_qualification_fixture_evidence(
                        label["base_output"],
                        declared=label.get("base_evidence"),
                    ),
                    evaluators=evaluators,
                    repo_root=repo_root,
                )
            except Exception as exc:
                blockers.append(
                    f"{task_id}: evaluator qualification failed: "
                    f"{type(exc).__name__}"
                )
            else:
                if not base_passed:
                    base_failures += 1
        if "gold_output" not in label:
            warnings.append(f"{task_id}: missing gold_output qualification fixture")
        else:
            try:
                gold_passed, _, _ = _score_deterministic_output(
                    task=task,
                    output=label["gold_output"],
                    expected=label.get("expected"),
                    evidence=_qualification_fixture_evidence(
                        label["gold_output"],
                        declared=label.get("gold_evidence"),
                    ),
                    evaluators=evaluators,
                    repo_root=repo_root,
                )
            except Exception as exc:
                blockers.append(
                    f"{task_id}: evaluator qualification failed: "
                    f"{type(exc).__name__}"
                )
            else:
                if gold_passed:
                    gold_passes += 1
                else:
                    blockers.append(
                        f"{task_id}: known-good output fails the evaluator"
                    )
    return base_failures, gold_passes, blockers, warnings


def _qualification_fixture_evidence(
    output: Any,
    *,
    declared: Any = None,
) -> dict[str, Any]:
    if declared is not None:
        return dict(_mapping(declared, "qualification fixture evidence"))
    if not isinstance(output, Mapping):
        return {}
    project = output.get("project")
    evidence = (
        {"queried_projects": [project]}
        if isinstance(project, str) and project
        else {}
    )
    projected_fields = output.get("projected_fields") or output.get(
        "selected_fields"
    )
    if isinstance(projected_fields, list):
        evidence["mcp_tool_calls"] = [
            {
                "tool": "query_wandb_tool",
                "queried_projects": [project] if isinstance(project, str) else [],
                "projected_fields": list(projected_fields),
                "limit": output.get("returned_count"),
            }
        ]
    return evidence


def preview_comparison(
    spec: ComparisonSpecV1,
    *,
    repo_root: Path,
    operator: OperatorService | None = None,
) -> ComparisonPreviewV1:
    readiness = check_comparison(spec, repo_root=repo_root)
    experiment, manifest, public_rows = compile_comparison(spec, repo_root=repo_root)
    manifest_path = Path(experiment.manifest)
    overlay = {
        manifest_path.as_posix(): yaml.safe_dump(manifest, sort_keys=False),
    }
    service = operator or OperatorService(repo_root)
    matrix = service.preview_experiment(
        experiment,
        request=ExperimentRequest(
            experiment_id=experiment.id,
            n_concurrent=spec.execution.concurrency,
        ),
        asset_overlay=overlay,
    )
    if matrix.cells != readiness.estimated_cells:
        raise RuntimeError(
            "comparison compiler and OperatorService resolved different planned cells"
        )
    unsigned = ComparisonPreviewV1(
        schema_version=COMPARISON_SCHEMA_VERSION,
        comparison=spec.to_dict(),
        readiness=readiness.to_dict(),
        matrix=_preview_dict(matrix),
        experiment=experiment.to_dict(),
        manifest=manifest,
    )
    return replace(
        unsigned,
        preview_digest=_artifact_digest(unsigned.to_dict(), "preview_digest"),
    )


def compile_comparison(
    spec: ComparisonSpecV1, *, repo_root: Path
) -> tuple[ExperimentSpec, dict[str, Any], list[dict[str, Any]]]:
    tasks = _load_public_tasks(repo_root / spec.taskset.tasks)
    public_rows = [
        _public_case(task, spec=spec, index=index, repo_root=repo_root)
        for index, task in enumerate(tasks)
    ]
    public_text = _jsonl(public_rows)
    taskset_digest = hashlib.sha256(public_text.encode()).hexdigest()
    evaluator_digests = {
        item.id: _evaluator_digest(item, repo_root) for item in spec.evaluators
    }
    runtime = COMPARISON_RUNTIME_ROOT / spec.spec_digest / stable_digest(
        {
            "public_cases": taskset_digest,
            "evaluators": evaluator_digests,
        }
    )
    source_path = runtime / "public-cases.jsonl"
    manifest_path = runtime / "manifest.yaml"
    dataset_path = Path(".fugue/cache/simple-task-datasets") / taskset_digest
    manifest = {
        "dataset": {
            "path": dataset_path.as_posix(),
            "materializer": "fugue.bench.task_authoring:AuthoredTaskMaterializer",
            "source": {
                "path": source_path.as_posix(),
                "sha256": hashlib.sha256(public_text.encode()).hexdigest(),
            },
        },
        "model": spec.execution.model,
        "k": 1,
        "n_concurrent": spec.execution.concurrency,
        "jobs_dir": f".fugue/runtime/jobs/{spec.id}",
        "harnesses": [
            {"name": name, "agent": _HARNESS_AGENTS[name]}
            for name in spec.execution.harnesses
        ],
        "tasks": [
            {
                "id": row["id"],
                "notes": row["instruction"][:500],
                "metadata": {
                    "source_index": index,
                    "task_authoring": {
                        "task_definition_digest": taskset_digest,
                        "criteria_digest": _sha256_path(
                            repo_root / spec.taskset.private_labels
                        ),
                        "scenario_id": "comparison",
                        "interaction": {
                            "type": "single_turn",
                            "max_user_turns": 0,
                            "max_agent_turns": 1,
                        },
                        "interaction_controller": row["interaction"],
                        "environment_profile_id": "artifact-python-v1",
                        "environment_kind": "artifact",
                        "profile_digests": {
                            f"comparison-evaluator:{evaluator_id}": digest
                            for evaluator_id, digest in evaluator_digests.items()
                        },
                        "harness_applicability": row["harness_applicability"],
                        "partition": row["partition"],
                        "tags": row["tags"],
                    },
                },
            }
            for index, row in enumerate(public_rows)
        ],
    }
    experiment = experiment_from_data(
        {
            "id": spec.id,
            "title": spec.question,
            "description": "Compiled Fugue Agent-change comparison.",
            "manifest": manifest_path.as_posix(),
            "model": spec.execution.model,
            "run_name": spec.id,
            "tags": ["comparison", "technical-preview"],
            "harnesses": list(spec.execution.harnesses),
            "variants": [
                _variant_dict("baseline", spec.baseline),
                _variant_dict("candidate", spec.candidate),
            ],
            "n_attempts": spec.execution.attempts,
            "n_concurrent": spec.execution.concurrency,
            "n_tasks": len(tasks),
            "jobs_dir": f".fugue/runtime/jobs/{spec.id}",
            "presets": [
                {
                    "id": "counterbalanced",
                    "scheduling_seed": spec.execution.scheduling_seed,
                }
            ]
            if spec.execution.scheduling_seed
            else [],
            "default_preset": (
                "counterbalanced" if spec.execution.scheduling_seed else None
            ),
            "trace_content": spec.execution.trace_content,
            "source_evidence_project": spec.execution.source_evidence_project,
            "source_evidence_destination": (
                asdict(spec.execution.source_evidence_destination)
                if spec.execution.source_evidence_destination is not None
                else None
            ),
            "evidence_project": spec.execution.evidence_project,
            "evidence_destination": (
                asdict(spec.execution.evidence_destination)
                if spec.execution.evidence_destination is not None
                else None
            ),
            "environment": spec.execution.environment,
            "agent_env": _drop_empty(
                {
                    "FUGUE_SOURCE_EVIDENCE_PROJECT": (
                        spec.execution.source_evidence_project
                    ),
                    "FUGUE_RESULT_EVIDENCE_PROJECT": (
                        spec.execution.evidence_project
                    ),
                    "FUGUE_WANDB_RESEARCH_ID": spec.execution.research_id,
                    "FUGUE_WANDB_STUDY_ID": spec.id,
                    "FUGUE_STUDY_CONSOLE_BACKLINK": (
                        _study_console_backlink(
                            spec.execution.study_console_base_url,
                            research_id=spec.execution.research_id,
                            study_id=spec.id,
                        )
                    ),
                }
            ),
            "research_view": {
                "observation": spec.question,
                "rationale": "Test one declared Agent-system change on aligned tasks.",
                "success_definition": "Pass the required deterministic evaluator.",
                "task_title": f"{len(tasks)} locked comparison tasks",
                "task_summary": "Public task inputs with host-only expected values.",
                "interaction_mode": "Single turn",
                "base_instruction_summary": "Use the common task instruction.",
                "treatment_summaries": {
                    "baseline": spec.baseline.label,
                    "candidate": spec.candidate.label,
                },
                "pass_rule": "All required deterministic checks must pass.",
                "scorers": [
                    _research_scorer(
                        item,
                        revision=evaluator_digests[item.id],
                    )
                    for item in spec.evaluators
                ],
            },
        }
    )
    return experiment, manifest, public_rows


def prepare_comparison(
    spec: ComparisonSpecV1,
    *,
    repo_root: Path,
    operator: OperatorService | None = None,
) -> tuple[dict[str, Any], ComparisonPreviewV1, Path]:
    """Materialize and lock one exact local comparison without running cells.

    Preparation is a trusted, no-spend boundary. It freezes source inputs and
    builds the exact task and Agent images before the final preview is
    computed. The resulting receipt intentionally does not contain approval
    state and cannot authorize execution.
    """
    environment_type = str(spec.execution.environment.get("type") or "docker")
    if environment_type != "docker":
        raise ValueError(
            "comparison-scoped preparation currently supports local Docker only"
        )
    if not spec.execution.preparation_required:
        raise ValueError(
            "comparison does not declare execution.preparation_required"
        )
    service = operator or OperatorService(repo_root)
    experiment, manifest, public_rows = compile_comparison(
        spec,
        repo_root=repo_root,
    )
    runtime_root = repo_root / Path(experiment.manifest).parent
    runtime_root.mkdir(parents=True, exist_ok=True)
    public_text = _jsonl(public_rows)
    _write_immutable_text(
        runtime_root / "public-cases.jsonl",
        public_text,
        expected_sha256=hashlib.sha256(public_text.encode()).hexdigest(),
        mode=0o444,
        label="prepared compiled public cases",
    )
    manifest_text = yaml.safe_dump(manifest, sort_keys=False)
    _write_immutable_text(
        runtime_root / "manifest.yaml",
        manifest_text,
        expected_sha256=hashlib.sha256(manifest_text.encode()).hexdigest(),
        mode=0o444,
        label="prepared comparison manifest",
    )

    # Trusted evaluators execute during readiness qualification. Materialize
    # their exact platform images before the provisional (read-only) preview;
    # pure preview never pulls or mutates runtime state.
    _prepare_evaluator_runtimes(spec, repo_root=repo_root)

    # A provisional preview is never approval-eligible. It exists solely to
    # bind and freeze the source-side input manifest before Docker builds.
    provisional = preview_comparison(
        spec,
        repo_root=repo_root,
        operator=service,
    )
    unexpected_blockers = [
        str(item)
        for item in provisional.readiness.get("blockers") or ()
        if not str(item).startswith("local agent:")
        and not str(item).startswith("local task:")
        and not str(item).startswith(
            "comparison preparation is missing or drifted"
        )
    ]
    if unexpected_blockers:
        raise ValueError(
            "comparison has non-preparation blockers:\n- "
            + "\n- ".join(unexpected_blockers)
        )
    if provisional.readiness["status"] == "no_comparison_justified":
        raise ValueError("comparison does not justify an execution cohort")
    _materialize_approved_comparison_inputs(
        spec,
        preview=provisional,
        public_rows=public_rows,
        repo_root=repo_root,
    )
    frozen_inputs = _approved_input_manifest(
        provisional,
        repo_root=repo_root,
        public_rows=public_rows,
    )
    service.prepare(
        ExperimentRequest(
            experiment_id=spec.id,
            n_concurrent=spec.execution.concurrency,
            n_attempts=spec.execution.attempts,
            n_tasks=len(public_rows),
        ),
        experiment=experiment,
    )
    runtime_lock_digests, runtime_blockers = _runtime_readiness(
        spec,
        repo_root=repo_root,
    )
    local_source_digests, local_source_blockers = (
        _local_source_lock_readiness(spec, repo_root=repo_root)
    )
    declared_input_digests, declared_input_blockers = (
        _declared_qualification_input_readiness(spec, repo_root=repo_root)
    )
    qualification_input_digests, qualification_blockers = (
        _qualification_input_readiness(spec, repo_root=repo_root)
    )
    qualification_input_digests = {
        **local_source_digests,
        **declared_input_digests,
        **qualification_input_digests,
    }
    preparation_blockers = [
        *runtime_blockers,
        *local_source_blockers,
        *declared_input_blockers,
        *qualification_blockers,
    ]
    if preparation_blockers:
        raise RuntimeError(
            "comparison preparation did not produce a usable lock set:\n- "
            + "\n- ".join(preparation_blockers)
        )
    unsigned_receipt = {
        "schema_version": 1,
        "kind": "comparison_preparation",
        "comparison_id": spec.id,
        "spec_digest": spec.spec_digest,
        "environment_type": environment_type,
        "runtime_lock_digests": dict(sorted(runtime_lock_digests.items())),
        "qualification_input_digests": dict(
            sorted(qualification_input_digests.items())
        ),
        "frozen_inputs": frozen_inputs,
        "frozen_inputs_digest": stable_digest(frozen_inputs),
        "receipt_digest": "",
    }
    receipt = {
        **unsigned_receipt,
        "receipt_digest": stable_digest(unsigned_receipt),
    }
    receipt_path = _comparison_preparation_receipt_path(
        spec,
        repo_root=repo_root,
    )
    atomic_write_json(receipt_path, receipt)

    final_preview = preview_comparison(
        spec,
        repo_root=repo_root,
        operator=service,
    )
    if final_preview.readiness["status"] not in {"ready", "needs_review"}:
        blockers = final_preview.readiness.get("blockers") or []
        raise RuntimeError(
            "comparison remains blocked after preparation:\n- "
            + "\n- ".join(str(item) for item in blockers)
        )
    _materialize_approved_comparison_inputs(
        spec,
        preview=final_preview,
        public_rows=public_rows,
        repo_root=repo_root,
    )
    stable_preview = preview_comparison(
        spec,
        repo_root=repo_root,
        operator=service,
    )
    if stable_preview.preview_digest != final_preview.preview_digest:
        raise RuntimeError("comparison preview changed while freezing prepared inputs")
    atomic_write_json(runtime_root / "prepared-preview.json", stable_preview.to_dict())
    return receipt, stable_preview, receipt_path


def materialize_comparison(
    preview: ComparisonPreviewV1,
    *,
    repo_root: Path,
    operator: OperatorService | None = None,
    approval_digest: str = "",
    approval_input_bindings: Mapping[str, str] | None = None,
) -> tuple[ExperimentSpec, ExperimentRequest]:
    _verify_artifact(preview.to_dict(), "preview_digest", "comparison preview")
    spec = comparison_from_dict(
        preview.comparison,
        repo_root=repo_root,
        source=repo_root,
    )
    current = preview_comparison(spec, repo_root=repo_root, operator=operator)
    if current.preview_digest != preview.preview_digest:
        raise ValueError("comparison inputs changed after preview")
    experiment, manifest, public_rows = compile_comparison(spec, repo_root=repo_root)
    if spec.execution.research_id:
        experiment = replace(
            experiment,
            agent_env={
                **experiment.agent_env,
                "FUGUE_WANDB_RESEARCH_ID": spec.execution.research_id,
                "FUGUE_WANDB_STUDY_ID": spec.id,
            },
        )
    root = repo_root / Path(experiment.manifest).parent
    root.mkdir(parents=True, exist_ok=True)
    _materialize_approved_comparison_inputs(
        spec,
        preview=preview,
        public_rows=public_rows,
        repo_root=repo_root,
    )
    _write_immutable_text(
        root / "public-cases.jsonl",
        _jsonl(public_rows),
        expected_sha256=str(
            _mapping(preview.manifest["dataset"], "preview dataset")
            .get("source", {})
            .get("sha256")
            or ""
        ),
        mode=0o444,
        label="approved compiled public cases",
    )
    _write_immutable_text(
        root / "manifest.yaml",
        yaml.safe_dump(manifest, sort_keys=False),
        expected_sha256=hashlib.sha256(
            yaml.safe_dump(preview.manifest, sort_keys=False).encode()
        ).hexdigest(),
        mode=0o444,
        label="approved comparison manifest",
    )
    _atomic_text(
        root / "comparison.yaml",
        yaml.safe_dump(spec.to_dict(), sort_keys=False),
    )
    atomic_write_json(root / "preview.json", preview.to_dict())
    approved_comparison = _approved_comparison_execution_lock(
        preview,
        approval_digest=approval_digest,
        approval_input_bindings=approval_input_bindings,
        repo_root=repo_root,
        public_rows=public_rows,
    )
    return experiment, ExperimentRequest(
        experiment_id=spec.id,
        n_concurrent=spec.execution.concurrency,
        n_attempts=spec.execution.attempts,
        n_tasks=int(preview.readiness["task_count"]),
        approved_comparison=approved_comparison,
    )


def _approved_input_manifest(
    preview: ComparisonPreviewV1,
    *,
    repo_root: Path,
    public_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    spec = comparison_from_dict(
        preview.comparison,
        repo_root=repo_root,
        source=repo_root,
    )
    readiness = _mapping(preview.readiness, "preview readiness")
    taskset_digest = _sha256_path(repo_root / spec.taskset.tasks)
    private_digest = _sha256_path(repo_root / spec.taskset.private_labels)
    if taskset_digest != str(readiness.get("taskset_digest") or ""):
        raise ValueError("public taskset changed after approval preview")
    if private_digest != str(readiness.get("private_labels_digest") or ""):
        raise ValueError("private labels changed after approval preview")
    expected_evaluators = _mapping(
        readiness.get("evaluator_digests"),
        "preview evaluator digests",
    )
    evaluator_artifacts: dict[str, dict[str, str]] = {}
    for evaluator in spec.evaluators:
        if _evaluator_digest(evaluator, repo_root) != str(
            expected_evaluators.get(evaluator.id) or ""
        ):
            raise ValueError(
                f"evaluator {evaluator.id!r} changed after approval preview"
            )
        artifacts: dict[str, str] = {}
        if evaluator.scorer:
            artifacts["scorer_sha256"] = _sha256_path(
                _safe_input_path(
                    Path(evaluator.scorer),
                    repo_root,
                    "deterministic scorer",
                )
            )
        if evaluator.verifier:
            artifacts["verifier_sha256"] = _sha256_path(
                _safe_input_path(
                    Path(evaluator.verifier.source),
                    repo_root,
                    "host verifier",
                )
            )
        if evaluator.calibration:
            artifacts["calibration_sha256"] = _sha256_path(
                _safe_input_path(
                    Path(evaluator.calibration),
                    repo_root,
                    "judge calibration",
                )
            )
        evaluator_artifacts[evaluator.id] = artifacts
    source = _mapping(
        _mapping(preview.manifest.get("dataset"), "preview dataset").get("source"),
        "preview dataset source",
    )
    compiled_digest = hashlib.sha256(_jsonl(public_rows).encode()).hexdigest()
    if compiled_digest != str(source.get("sha256") or ""):
        raise ValueError("compiled public cases changed after approval preview")
    resources = sorted(
        (
            {
                "locked_relative": str(attachment["locked_relative"]),
                "sha256": str(attachment["sha256"]),
            }
            for row in public_rows
            for attachment in row.get("attachments") or ()
            if isinstance(attachment, Mapping)
        ),
        key=lambda item: (item["locked_relative"], item["sha256"]),
    )
    return {
        "schema_version": 1,
        "public_tasks_sha256": taskset_digest,
        "compiled_public_cases_sha256": compiled_digest,
        "private_labels_sha256": private_digest,
        "evaluator_digests": dict(sorted(expected_evaluators.items())),
        "evaluator_artifacts": dict(sorted(evaluator_artifacts.items())),
        "task_resources": resources,
    }


def _materialize_approved_comparison_inputs(
    spec: ComparisonSpecV1,
    *,
    preview: ComparisonPreviewV1,
    public_rows: Sequence[Mapping[str, Any]],
    repo_root: Path,
) -> None:
    approved = _approved_input_manifest(
        preview,
        repo_root=repo_root,
        public_rows=public_rows,
    )
    public_digest = str(approved["public_tasks_sha256"])
    private_digest = str(approved["private_labels_sha256"])
    _write_immutable_bytes(
        _frozen_public_tasks_path(repo_root, public_digest),
        (repo_root / spec.taskset.tasks).read_bytes(),
        expected_sha256=public_digest,
        mode=0o444,
        label="approved public taskset",
    )
    _write_immutable_bytes(
        _frozen_private_labels_path(repo_root, private_digest),
        (repo_root / spec.taskset.private_labels).read_bytes(),
        expected_sha256=private_digest,
        mode=0o400,
        label="approved private labels",
    )
    for evaluator in spec.evaluators:
        artifacts = _mapping(
            _mapping(
                approved["evaluator_artifacts"],
                "approved evaluator artifacts",
            ).get(evaluator.id),
            f"approved evaluator {evaluator.id} artifacts",
        )
        if evaluator.scorer:
            digest = str(artifacts.get("scorer_sha256") or "")
            source = _safe_input_path(
                Path(evaluator.scorer),
                repo_root,
                "deterministic scorer",
            )
            _write_immutable_bytes(
                _frozen_evaluator_artifact_path(
                    repo_root,
                    digest,
                    kind="scorer",
                ),
                source.read_bytes(),
                expected_sha256=digest,
                mode=0o400,
                label=f"approved scorer {evaluator.id}",
            )
        if evaluator.verifier:
            digest = str(artifacts.get("verifier_sha256") or "")
            source = _safe_input_path(
                Path(evaluator.verifier.source),
                repo_root,
                "host verifier",
            )
            _write_immutable_bytes(
                _frozen_evaluator_artifact_path(
                    repo_root,
                    digest,
                    kind="verifier",
                ),
                source.read_bytes(),
                expected_sha256=digest,
                mode=0o400,
                label=f"approved host verifier {evaluator.id}",
            )
        if evaluator.calibration:
            digest = str(artifacts.get("calibration_sha256") or "")
            source = _safe_input_path(
                Path(evaluator.calibration),
                repo_root,
                "judge calibration",
            )
            _write_immutable_bytes(
                _frozen_evaluator_artifact_path(
                    repo_root,
                    digest,
                    kind="calibration",
                ),
                source.read_bytes(),
                expected_sha256=digest,
                mode=0o400,
                label=f"approved calibration {evaluator.id}",
            )
    tasks = _load_public_tasks(repo_root / spec.taskset.tasks)
    expected_resources = {
        (str(item["locked_relative"]), str(item["sha256"]))
        for item in approved["task_resources"]
        if isinstance(item, Mapping)
    }
    observed_resources: set[tuple[str, str]] = set()
    for task in tasks:
        for raw in task.get("resources") or ():
            relative = _safe_resource_relative_path(
                raw.get("path"),
                label=f"public task {task['id']} resource",
            )
            source = repo_root / relative
            digest = _sha256_path(source)
            target = _frozen_resource_path(repo_root, digest, source.name)
            frozen_relative = target.relative_to(repo_root).as_posix()
            observed_resources.add((frozen_relative, digest))
            _write_immutable_bytes(
                target,
                source.read_bytes(),
                expected_sha256=digest,
                mode=0o444,
                label=f"approved task resource {task['id']}",
            )
    if observed_resources != expected_resources:
        raise ValueError("approved task resources changed after preview")


def _frozen_public_tasks_path(repo_root: Path, digest: str) -> Path:
    return repo_root / COMPARISON_INPUT_ROOT / "tasksets" / f"{digest}.jsonl"


def _frozen_private_labels_path(repo_root: Path, digest: str) -> Path:
    return repo_root / COMPARISON_PRIVATE_INPUT_ROOT / "labels" / f"{digest}.jsonl"


def _frozen_evaluator_artifact_path(
    repo_root: Path,
    digest: str,
    *,
    kind: Literal["scorer", "verifier", "calibration"],
) -> Path:
    suffix = {"scorer": ".py", "verifier": ".cjs", "calibration": ".json"}[
        kind
    ]
    return repo_root / COMPARISON_PRIVATE_INPUT_ROOT / kind / f"{digest}{suffix}"


def _frozen_resource_path(repo_root: Path, digest: str, name: str) -> Path:
    return repo_root / COMPARISON_INPUT_ROOT / "resources" / digest / name


def _verify_frozen_comparison_inputs(
    inputs: Mapping[str, Any],
    *,
    repo_root: Path,
) -> None:
    checks = (
        (
            _frozen_public_tasks_path(
                repo_root,
                str(inputs.get("public_tasks_sha256") or ""),
            ),
            str(inputs.get("public_tasks_sha256") or ""),
            "prepared public taskset",
        ),
        (
            _frozen_private_labels_path(
                repo_root,
                str(inputs.get("private_labels_sha256") or ""),
            ),
            str(inputs.get("private_labels_sha256") or ""),
            "prepared private labels",
        ),
    )
    for path, digest, label in checks:
        if (
            not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not path.is_file()
            or path.is_symlink()
            or _sha256_path(path) != digest
        ):
            raise ValueError(f"{label} immutable copy changed")
    artifacts = _mapping(
        inputs.get("evaluator_artifacts"),
        "prepared evaluator artifacts",
    )
    for evaluator_id, raw in artifacts.items():
        values = _mapping(raw, f"prepared evaluator {evaluator_id} artifacts")
        for digest_field, kind in (
            ("scorer_sha256", "scorer"),
            ("verifier_sha256", "verifier"),
            ("calibration_sha256", "calibration"),
        ):
            digest = str(values.get(digest_field) or "")
            if not digest:
                continue
            path = _frozen_evaluator_artifact_path(
                repo_root,
                digest,
                kind=kind,  # type: ignore[arg-type]
            )
            if (
                not re.fullmatch(r"[0-9a-f]{64}", digest)
                or not path.is_file()
                or path.is_symlink()
                or _sha256_path(path) != digest
            ):
                raise ValueError(
                    f"prepared {kind} {evaluator_id!r} immutable copy changed"
                )
    for raw in _sequence(
        inputs.get("task_resources"),
        "prepared task resources",
        allow_empty=True,
    ):
        resource = _mapping(raw, "prepared task resource")
        relative = _safe_resource_relative_path(
            resource.get("locked_relative"),
            label="prepared task resource",
        )
        path = repo_root / relative
        digest = str(resource.get("sha256") or "")
        if (
            not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not path.is_file()
            or path.is_symlink()
            or _sha256_path(path) != digest
        ):
            raise ValueError(f"prepared task resource changed: {relative}")


def _approved_comparison_execution_lock(
    preview: ComparisonPreviewV1,
    *,
    approval_digest: str,
    approval_input_bindings: Mapping[str, str] | None,
    repo_root: Path,
    public_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _verify_artifact(preview.to_dict(), "preview_digest", "comparison preview")
    comparison = _mapping(preview.comparison, "preview comparison")
    comparison_execution = _mapping(
        comparison.get("execution"),
        "preview comparison execution",
    )
    readiness = _mapping(preview.readiness, "preview readiness")
    matrix = _mapping(preview.matrix, "preview matrix")
    evidence_destination = _evidence_destination(
        matrix.get("evidence_destination"),
        "preview evidence destination",
    )
    source_project = str(matrix.get("source_evidence_project") or "")
    source_destination = (
        _evidence_destination(
            matrix.get("source_evidence_destination"),
            "preview source evidence destination",
        )
        if source_project
        else None
    )
    cells: list[dict[str, Any]] = []
    coordinates: set[tuple[str, str, str, int]] = set()
    attempt_ids: set[str] = set()
    for raw in _sequence(matrix.get("matrix_cells"), "preview matrix cells"):
        item = _mapping(raw, "preview matrix cell")
        task_id = validate_id(str(item.get("task_id") or ""), kind="task id")
        variant_id = validate_id(
            str(item.get("variant_id") or ""), kind="variant id"
        )
        harness = validate_id(str(item.get("harness") or ""), kind="harness")
        trial_index = int(item.get("trial_index") or 0)
        if trial_index < 1:
            raise ValueError("preview matrix trial_index must be positive")
        candidate_id = str(item.get("candidate_id") or "")
        execution_fingerprint = str(item.get("execution_fingerprint") or "")
        if len(candidate_id) != 64 or len(execution_fingerprint) != 64:
            raise ValueError(
                "preview matrix candidate and execution identities must be exact digests"
            )
        identity = attempt_identity(
            task_id=task_id,
            arm=variant_id,
            harness=harness,
            attempt=trial_index,
            candidate=candidate_id,
            runtime=execution_fingerprint,
        )
        resolved_attempt_id = attempt_id(**identity)
        if str(item.get("attempt_id") or "") != resolved_attempt_id:
            raise ValueError(
                f"preview attempt identity disagrees for {task_id}/{variant_id}"
            )
        coordinate = (task_id, variant_id, harness, trial_index)
        if coordinate in coordinates:
            raise ValueError("preview matrix contains duplicate attempt coordinates")
        if resolved_attempt_id in attempt_ids:
            raise ValueError("preview matrix contains duplicate attempt identities")
        coordinates.add(coordinate)
        attempt_ids.add(resolved_attempt_id)
        cells.append(
            {
                "attempt_id": resolved_attempt_id,
                "attempt_identity": identity,
                "task_id": task_id,
                "variant_id": variant_id,
                "harness": harness,
                "trial_index": trial_index,
                "candidate_id": candidate_id,
                "execution_fingerprint": execution_fingerprint,
                "applicable": bool(item.get("applicable")),
                "skip_reason": str(item.get("reason") or ""),
                "integration_provenance_digest": stable_digest(
                    item.get("integration_provenance") or []
                ),
            }
        )
    cells.sort(key=lambda item: str(item["attempt_id"]))
    expected_count = int(matrix.get("cells") or 0)
    if expected_count != len(cells):
        raise ValueError("preview matrix cell count disagrees with its manifest")
    evaluator_digests = _mapping(
        readiness.get("evaluator_digests"), "preview evaluator digests"
    )
    runtime_lock_digests = _mapping(
        readiness.get("runtime_lock_digests") or {},
        "preview runtime lock digests",
    )
    preview_qualification_input_digests = _mapping(
        readiness.get("qualification_input_digests") or {},
        "preview qualification input digests",
    )
    normalized_approval_bindings = dict(
        sorted((approval_input_bindings or {}).items())
    )
    conflicts = {
        name
        for name, digest in normalized_approval_bindings.items()
        if name in preview_qualification_input_digests
        and preview_qualification_input_digests[name] != digest
    }
    if conflicts:
        raise ValueError(
            "approval input bindings conflict with preview qualification inputs: "
            + ", ".join(sorted(conflicts))
        )
    qualification_input_digests = {
        **preview_qualification_input_digests,
        **normalized_approval_bindings,
    }
    approved_inputs = _approved_input_manifest(
        preview,
        repo_root=repo_root,
        public_rows=public_rows,
    )
    integration_change = _integration_change_required(comparison)
    skill_change = _skill_change_required(comparison)
    source_change = integration_change or skill_change
    candidate_cells = [
        item
        for item in _sequence(matrix.get("matrix_cells"), "preview matrix cells")
        if isinstance(item, Mapping)
        and str(item.get("variant_id") or "") == "candidate"
    ]
    candidate_revisions = _consistent_source_revisions(
        candidate_cells,
        required=source_change,
        label="approved candidate",
    )
    candidate_ids = sorted(
        {str(item.get("candidate_id") or "") for item in candidate_cells}
    )
    if source_change and len(candidate_ids) != 1:
        raise ValueError(
            "source-changing comparison must resolve one candidate identity"
        )
    unsigned = {
        "schema_version": 1,
        "kind": "approved_comparison_execution",
        "comparison_id": validate_id(
            str(comparison.get("id") or ""), kind="comparison id"
        ),
        "preview_digest": preview.preview_digest,
        "approval_digest": approval_digest,
        "spec_digest": str(comparison.get("spec_digest") or ""),
        "taskset_digest": str(readiness.get("taskset_digest") or ""),
        "private_labels_digest": str(
            readiness.get("private_labels_digest") or ""
        ),
        "scorer_digests": dict(sorted(evaluator_digests.items())),
        "runtime_locks_required": bool(
            comparison_execution.get("preparation_required")
        ),
        "runtime_lock_digests": dict(sorted(runtime_lock_digests.items())),
        "approval_input_bindings": normalized_approval_bindings,
        "qualification_input_digests": dict(
            sorted(qualification_input_digests.items())
        ),
        "approved_inputs": approved_inputs,
        "approved_inputs_digest": stable_digest(approved_inputs),
        "source_evidence_project": source_project,
        "source_evidence_destination": source_destination,
        "source_lock_digest": str(
            qualification_input_digests.get("source_lock")
            or qualification_input_digests.get("evidence_lock")
            or (
                readiness.get("taskset_digest")
                if source_project and source_destination
                else ""
            )
            or ""
        ),
        "evidence_project": str(matrix.get("evidence_project") or ""),
        "evidence_destination": evidence_destination,
        "evidence_checkpoint_cells": int(
            comparison_execution.get("evidence_checkpoint_cells") or 0
        ),
        "scheduling_seed": str(
            comparison_execution.get("scheduling_seed") or ""
        ),
        "candidate_source_revisions_required": source_change,
        "candidate_source_revisions": [
            item.to_dict() for item in candidate_revisions
        ],
        "candidate_source_identity_digest": stable_digest(
            {
                "candidate_ids": candidate_ids,
                "source_revisions": [
                    item.to_dict() for item in candidate_revisions
                ],
            }
        ),
        "expected_cell_count": len(cells),
        "expected_cells_digest": stable_digest(cells),
        "expected_cells": cells,
    }
    unsigned["evidence_topology_identity"] = stable_digest(
        {
            "source_evidence_project": unsigned["source_evidence_project"],
            "source_evidence_destination": unsigned[
                "source_evidence_destination"
            ],
            "source_lock_digest": unsigned["source_lock_digest"],
            "result_evidence_project": unsigned["evidence_project"],
            "result_evidence_destination": unsigned["evidence_destination"],
        }
    )
    locked_spec = comparison_from_dict(
        comparison,
        repo_root=repo_root,
        source=repo_root,
    )
    unsigned["cohort_lineage"] = _comparison_cohort_lineage(
        locked_spec,
        repo_root=repo_root,
        source_lock_digest=str(unsigned["source_lock_digest"]),
    )
    for key in (
        "preview_digest",
        "spec_digest",
        "taskset_digest",
        "private_labels_digest",
        "expected_cells_digest",
    ):
        value = str(unsigned[key])
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"approved comparison {key} must be an exact digest")
    if approval_digest and (
        len(approval_digest) != 64
        or any(char not in "0123456789abcdef" for char in approval_digest)
    ):
        raise ValueError("approved comparison approval_digest must be an exact digest")
    if not unsigned["evidence_project"]:
        raise ValueError("approved comparison must lock an exact evidence project")
    locked = {**unsigned, "lock_digest": stable_digest(unsigned)}
    _verify_approved_comparison_execution_lock(locked)
    return locked


def claim_comparison_approval(
    preview: ComparisonPreviewV1,
    *,
    approval_digest: str,
    repo_root: Path,
    execution_instance_id: str,
) -> Any:
    execution_instance_id = validate_id(
        execution_instance_id,
        kind="comparison execution instance id",
    )
    readiness = ComparisonReadinessV1(**preview.readiness)
    if readiness.status not in {"ready", "needs_review"}:
        raise ValueError(
            f"comparison is {readiness.status}; blocked and unjustified "
            "comparisons may not run"
        )
    store = StudyStore(repo_root)
    return ApprovalLedger(store.path).claim(
        approval_digest=approval_digest,
        subject_kind="experiment",
        preview_digest=preview.preview_digest,
        subject_id=execution_instance_id,
        estimated_cells=readiness.estimated_cells,
        estimated_cost_usd=readiness.estimated_cost_usd,
    )


def analyze_comparison_rows(
    *,
    comparison_id: str,
    preview_digest: str,
    rows: Sequence[Mapping[str, Any]],
    source: str,
    expected_evidence_project: str | None = None,
    expected_source_evidence_project: str | None = None,
    approved_comparison: Mapping[str, Any] | None = None,
    decision_policy: DecisionPolicyV1 | Mapping[str, Any] | None = None,
    attestation: DecisionAttestationV1 | Mapping[str, Any] | None = None,
    result_schema_version: Literal[2, 3] = 2,
    study_intent: str = "candidate_comparison",
    evidence_topology: EvidenceTopologyV1 | Mapping[str, Any] | None = None,
    release_note_coverage: Sequence[Mapping[str, Any]] = (),
    supersedes: Sequence[SupersededResultV1 | Mapping[str, Any]] = (),
) -> ComparisonResultV2 | ComparisonResultV3:
    execution_lock = _resolve_approved_comparison_execution_lock(
        rows,
        supplied=approved_comparison,
    )
    normalized = [_bind_attempt_identity(row) for row in rows]
    if not normalized:
        raise ValueError("comparison result requires at least one attempt row")
    if result_schema_version >= 3:
        non_terminal = [
            str(
                row.get("status")
                or row.get("execution_status")
                or "unknown"
            )
            for row in normalized
            if _terminal_execution_status(row) is None
        ]
        if non_terminal:
            raise ValueError(
                "V3 comparison result requires only terminal attempt rows: "
                + ", ".join(sorted(set(non_terminal)))
            )
    if execution_lock is not None:
        _validate_approved_comparison_rows(
            normalized,
            source=source,
            approved=execution_lock,
        )
        locked_project = str(execution_lock.get("evidence_project") or "")
        if expected_evidence_project and expected_evidence_project != locked_project:
            raise ValueError(
                "analysis evidence project disagrees with the approved comparison "
                f"({expected_evidence_project!r} != {locked_project!r})"
            )
        expected_evidence_project = locked_project
        locked_source_project = str(
            execution_lock.get("source_evidence_project") or ""
        )
        if (
            expected_source_evidence_project
            and expected_source_evidence_project != locked_source_project
        ):
            raise ValueError(
                "analysis source evidence project disagrees with the approved "
                f"comparison ({expected_source_evidence_project!r} != "
                f"{locked_source_project!r})"
            )
        expected_source_evidence_project = (
            locked_source_project or expected_source_evidence_project
        )
    resolved_destination = _resolve_result_evidence_destination(
        normalized,
        execution_lock=execution_lock,
    )
    observed_evidence_projects = sorted(
        {
            str(row.get("trace_project") or "")
            for row in normalized
            if str(row.get("trace_project") or "")
        }
    )
    if expected_evidence_project:
        mismatched = sorted(
            {
                str(row.get("trace_project") or "unreported")
                for row in normalized
                if str(row.get("trace_project") or "")
                != expected_evidence_project
            }
        )
        if mismatched:
            raise ValueError(
                "comparison rows disagree with the locked evidence project "
                f"{expected_evidence_project}: {', '.join(mismatched)}"
            )
    elif len(observed_evidence_projects) > 1:
        raise ValueError(
            "comparison rows disagree on the evidence project: "
            + ", ".join(observed_evidence_projects)
        )
    policy = (
        decision_policy
        if isinstance(decision_policy, DecisionPolicyV1)
        else _decision_policy(decision_policy)
    )
    signed = _decision_attestation(attestation)
    attempt_ids = [str(row["attempt_id"]) for row in normalized]
    duplicate_attempt_ids = sorted(
        value for value, count in Counter(attempt_ids).items() if count > 1
    )
    paired = _analyze_aligned_pairs(
        normalized,
        result_schema_version=result_schema_version,
    )
    paired_cases_v2 = list(paired.v2)
    paired_cases_v3 = list(paired.v3)
    baseline_passed = paired.baseline_passed
    candidate_passed = paired.candidate_passed
    improved = paired.improved
    regressed = paired.regressed
    mixed = paired.mixed
    unchanged = paired.unchanged
    incomplete = paired.incomplete
    limitations = [
        "Task outcomes and authored evaluations must be interpreted separately.",
        "The result applies only to the locked taskset, candidates, and attempts.",
    ]
    if any(
        isinstance(judge, Mapping)
        and judge.get("status") in {"scored", "missing_evidence", "unavailable"}
        and (
            judge.get("cost_status") is not None
            or _row_number(judge, "accounted_reserve_usd") is not None
        )
        and judge.get("cost_status") != "observed"
        for row in normalized
        for judge in _mapping_or_empty(row.get("comparison_judges")).values()
    ):
        limitations.append(
            "The judge provider did not return observed dollar cost; Fugue "
            "reports its accounted reservation separately from Agent spend."
        )
    if incomplete:
        limitations.append("At least one aligned pair is incomplete.")
    required_incomplete = sum(
        1
        for row in normalized
        if row.get("comparison_required_evaluation_complete") is False
    )
    operational = _operational_summary(normalized)
    evidence_statuses = [_attempt_evidence_status(row) for row in normalized]
    unresolved_evidence = sum(
        status in {"missing", "invalid"} for status in evidence_statuses
    )
    invalid_evidence = sum(status == "invalid" for status in evidence_statuses)
    cross_project_attempts = sum(
        bool(
            _cross_project_queries(
                row,
                expected_source_evidence_project
                or expected_evidence_project,
            )
        )
        for row in normalized
    )
    harbor_conformance_failed_attempts = sum(
        _local_harbor_conformance_failed(row) for row in normalized
    )
    harbor_conformance_unavailable_attempts = sum(
        _local_harbor_conformance_unavailable(row) for row in normalized
    )
    local_privacy_failed_attempts = sum(
        _privacy_scan_status(row, "local_artifact_privacy_scan_status")
        == "failed"
        for row in normalized
    )
    hosted_privacy_failed_attempts = sum(
        _privacy_scan_status(row, "hosted_evidence_privacy_scan_status")
        == "failed"
        for row in normalized
    )
    local_privacy_unavailable_attempts = sum(
        _privacy_scan_status(row, "local_artifact_privacy_scan_status")
        in {"legacy", "unavailable", "not_applicable"}
        for row in normalized
    )
    hosted_privacy_unavailable_attempts = sum(
        _privacy_scan_status(row, "hosted_evidence_privacy_scan_status")
        in {"legacy", "unavailable", "not_applicable"}
        for row in normalized
    )
    integrity = {
        "status": (
            "invalid"
            if (
                duplicate_attempt_ids
                or invalid_evidence
                or cross_project_attempts
                or harbor_conformance_failed_attempts
                or local_privacy_failed_attempts
                or hosted_privacy_failed_attempts
            )
            else "reconciled"
        ),
        "row_count": len(normalized),
        "unique_attempts": len(set(attempt_ids)),
        "duplicate_attempt_ids": duplicate_attempt_ids,
        "unresolved_evidence_attempts": unresolved_evidence,
        "invalid_evidence_attempts": invalid_evidence,
        "cross_project_attempts": cross_project_attempts,
        "harbor_conformance_failed_attempts": (
            harbor_conformance_failed_attempts
        ),
        "harbor_conformance_unavailable_attempts": (
            harbor_conformance_unavailable_attempts
        ),
        "local_artifact_privacy_failed_attempts": (
            local_privacy_failed_attempts
        ),
        "hosted_evidence_privacy_failed_attempts": (
            hosted_privacy_failed_attempts
        ),
        "local_artifact_privacy_unavailable_attempts": (
            local_privacy_unavailable_attempts
        ),
        "hosted_evidence_privacy_unavailable_attempts": (
            hosted_privacy_unavailable_attempts
        ),
        "privacy_complete_attempts": sum(
            _privacy_scans_complete(row) for row in normalized
        ),
        "rows_digest": stable_digest(normalized),
        "recomputed": True,
        "approved_manifest_status": (
            "reconciled" if execution_lock is not None else "not_provided"
        ),
        "approved_manifest_digest": (
            str(execution_lock["lock_digest"])
            if execution_lock is not None
            else None
        ),
        "expected_cell_count": (
            int(execution_lock["expected_cell_count"])
            if execution_lock is not None
            else None
        ),
    }
    deterministic = _deterministic_summary(normalized)
    behavioral = _behavioral_summary(
        rows=normalized,
        integrity=integrity,
        improved=improved,
        regressed=regressed,
        mixed=mixed,
        unchanged=unchanged,
        incomplete=incomplete,
        required_incomplete=required_incomplete,
        unresolved_evidence=unresolved_evidence,
        cross_project_attempts=cross_project_attempts,
    )
    topology: EvidenceTopologyV1 | None = None
    task_validity: tuple[TaskValidityV1, ...] = ()
    aligned_analysis: AlignedAnalysisV1 | None = None
    resolved_release_note_coverage: tuple[dict[str, Any], ...] = ()
    if result_schema_version == 3:
        topology = _resolve_evidence_topology_v3(
            normalized,
            execution_lock=execution_lock,
            supplied=evidence_topology,
            result_destination=resolved_destination,
        )
        task_validity = _task_validity_v3(
            paired_cases_v3,
            topology=topology,
        )
        behavioral = _behavioral_summary_v3(
            behavioral,
            paired_cases=paired_cases_v3,
            task_validity=task_validity,
        )
        aligned_analysis = _aligned_analysis_v3(
            paired_cases_v3,
            rows=normalized,
            study_intent=study_intent,
            task_validity=task_validity,
        )
        resolved_release_note_coverage = (
            _resolve_release_note_coverage_v3(
                normalized,
                execution_lock=execution_lock,
                supplied=release_note_coverage,
            )
        )
    decision = _evaluate_decision(
        policy=policy,
        rows=normalized,
        deterministic=deterministic,
        operational=operational,
        improved=improved,
        regressed=regressed,
        incomplete=incomplete,
        required_incomplete=required_incomplete,
        integrity=integrity,
        attestation=signed,
        release_note_coverage=resolved_release_note_coverage,
    )
    if result_schema_version == 3 and topology is not None:
        decision = _apply_v3_decision_validity(
            decision,
            task_validity=task_validity,
            topology=topology,
            release_note_coverage=resolved_release_note_coverage,
        )
    common = {
        "comparison_id": validate_id(comparison_id, kind="comparison id"),
        "preview_digest": preview_digest,
        "source": source,
        "evidence_project": (
            expected_evidence_project
            or (
                observed_evidence_projects[0]
                if observed_evidence_projects
                else None
            )
        ),
        "rows": len(normalized),
        "baseline_passed": baseline_passed,
        "candidate_passed": candidate_passed,
        "improved": improved,
        "regressed": regressed,
        "mixed": mixed,
        "unchanged": unchanged,
        "incomplete": incomplete,
        "required_evaluations_incomplete": required_incomplete,
        "deterministic_summary": deterministic,
        "judge_summary": _judge_summary(normalized),
        "mechanism_summary": _mechanism_summary(normalized),
        "operational_summary": operational,
        "evidence_links": _comparison_evidence_links(normalized),
        "limitations": tuple(limitations),
        "integrity": integrity,
        "behavioral_summary": behavioral,
        "decision_policy": policy.to_dict() if policy else None,
        "decision": decision,
        "candidate_source_revisions": _candidate_source_revisions(normalized),
        "evidence_destination": resolved_destination,
    }
    if result_schema_version == 3:
        if topology is None or aligned_analysis is None:
            raise AssertionError("V3 comparison analysis was not materialized")
        unsigned: ComparisonResultV2 | ComparisonResultV3 = ComparisonResultV3(
            schema_version=COMPARISON_RESULT_SCHEMA_VERSION,
            paired_cases=tuple(paired_cases_v3),
            evidence_topology=topology,
            aligned_analysis=aligned_analysis,
            task_validity=task_validity,
            release_note_coverage=resolved_release_note_coverage,
            scorer_revisions=_scorer_revisions_v3(execution_lock),
            runtime_locks=_runtime_locks_v3(
                normalized,
                execution_lock=execution_lock,
            ),
            cohort_lineage=dict(
                _mapping(
                    execution_lock.get("cohort_lineage")
                    if execution_lock is not None
                    else None,
                    "approved comparison cohort lineage",
                )
            ),
            supersedes=tuple(
                item
                if isinstance(item, SupersededResultV1)
                else superseded_result_from_dict(item)
                for item in supersedes
            ),
            **common,
        )
    else:
        unsigned = ComparisonResultV2(
            schema_version=COMPARISON_SCHEMA_VERSION,
            paired_cases=tuple(paired_cases_v2),
            **common,
        )
    qualification_digest = _comparison_qualification_digest(unsigned.to_dict())
    qualified = replace(
        unsigned,
        decision=_apply_decision_attestation(
            unsigned.decision,
            signed,
            qualification_digest=qualification_digest,
            require_actionability_review=result_schema_version >= 3,
        ),
        qualification_digest=qualification_digest,
    )
    # Before sign-off, result_digest is the exact immutable digest the release
    # owner must attest. Once an attestation is attached, result_digest becomes
    # an envelope digest that also binds signer and signed_at without making the
    # signed qualification digest circular.
    result_digest = (
        qualification_digest
        if signed is None
        else _comparison_result_digest(qualified.to_dict())
    )
    return replace(qualified, result_digest=result_digest)


def _analyze_aligned_pairs(
    rows: Sequence[Mapping[str, Any]],
    *,
    result_schema_version: Literal[2, 3],
) -> _PairedAnalysis:
    grouped: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    baseline_passed = 0
    candidate_passed = 0
    for row in rows:
        variant = str(row.get("variant_id") or "")
        if variant not in {"baseline", "candidate"}:
            continue
        task = str(row.get("task_id") or row.get("task_name") or "")
        harness = str(row.get("harness") or "")
        attempt = int(row.get("trial_index") or 1)
        grouped.setdefault((task, harness, attempt), {})[variant] = dict(row)
        baseline_passed += variant == "baseline" and row.get("pass") is True
        candidate_passed += variant == "candidate" and row.get("pass") is True
    paired_cases_v2: list[PairedCaseV2] = []
    paired_cases_v3: list[PairedCaseV3] = []
    counts: Counter[str] = Counter()
    for (task, harness, attempt), pair in sorted(grouped.items()):
        base = pair.get("baseline")
        candidate = pair.get("candidate")
        dimension_changes_v2 = _paired_dimension_changes(base, candidate)
        dimension_changes_v3 = (
            _paired_dimension_changes_v3(base, candidate)
            if result_schema_version == 3
            else ()
        )
        if (
            base is None
            or candidate is None
            or base.get("wandb_serverless_eligible") is False
            or candidate.get("wandb_serverless_eligible") is False
            or base.get("comparison_required_evaluation_complete") is False
            or candidate.get("comparison_required_evaluation_complete") is False
            or any(
                item.status == "unavailable" for item in dimension_changes_v2
            )
        ):
            status: PairStatus = "incomplete"
        else:
            status = (
                _pair_status_v3(dimension_changes_v3)
                if result_schema_version == 3
                else _pair_status_v2(dimension_changes_v2)
            )
        counts[status] += 1
        pair_id = stable_digest(
            {
                "schema_version": 1,
                "task_id": task,
                "harness": harness,
                "attempt": attempt,
            }
        )
        task_label = (
            _row_text(candidate, "task_name")
            or _row_text(base, "task_name")
            or task
        )
        if result_schema_version == 3:
            paired_cases_v3.append(
                PairedCaseV3(
                    pair_id=pair_id,
                    task_id=task,
                    harness=harness,
                    attempt=attempt,
                    status=status,
                    dimension_changes=dimension_changes_v3,
                    baseline=_paired_attempt_view_v3(base),
                    candidate=_paired_attempt_view_v3(candidate),
                    task_label=task_label,
                )
            )
            continue
        paired_cases_v2.append(
            PairedCaseV2(
                pair_id=pair_id,
                task_id=task,
                harness=harness,
                attempt=attempt,
                status=status,
                dimension_changes=dimension_changes_v2,
                baseline=_paired_attempt_view(base),
                candidate=_paired_attempt_view(candidate),
                task_label=task_label,
                baseline_passed=base.get("pass") if base else None,
                candidate_passed=candidate.get("pass") if candidate else None,
                baseline_prediction_id=_row_text(base, "prediction_id"),
                candidate_prediction_id=_row_text(candidate, "prediction_id"),
                baseline_evaluation_call_id=_row_text(
                    base, "eval_predict_and_score_call_id"
                ),
                candidate_evaluation_call_id=_row_text(
                    candidate, "eval_predict_and_score_call_id"
                ),
            )
        )
    return _PairedAnalysis(
        v2=tuple(paired_cases_v2),
        v3=tuple(paired_cases_v3),
        baseline_passed=baseline_passed,
        candidate_passed=candidate_passed,
        improved=counts["improved"],
        regressed=counts["regressed"],
        mixed=counts["mixed"],
        unchanged=counts["unchanged"],
        incomplete=counts["incomplete"],
    )


def _resolve_result_evidence_destination(
    rows: Sequence[Mapping[str, Any]],
    *,
    execution_lock: Mapping[str, Any] | None,
) -> dict[str, Any]:
    locked = (
        _evidence_destination(
            execution_lock.get("evidence_destination"),
            "approved evidence destination",
        )
        if execution_lock is not None
        else None
    )
    observed: dict[str, dict[str, Any]] = {}
    for row in rows:
        receipt = row.get("trace_receipt")
        if receipt is None:
            if locked is not None:
                raise ValueError(
                    "approved comparison row is missing its evidence destination "
                    "receipt"
                )
            continue
        destination = _evidence_destination(
            receipt,
            "comparison row evidence destination",
        )
        observed[str(destination["destination_digest"])] = destination
        if locked is not None and destination != locked:
            raise ValueError(
                "comparison row evidence destination disagrees with the "
                "approved destination"
            )
    if len(observed) > 1:
        raise ValueError("comparison rows disagree on the evidence destination")
    return locked or next(iter(observed.values()), {})


def _resolve_evidence_topology_v3(
    rows: Sequence[Mapping[str, Any]],
    *,
    execution_lock: Mapping[str, Any] | None,
    supplied: EvidenceTopologyV1 | Mapping[str, Any] | None,
    result_destination: Mapping[str, Any],
) -> EvidenceTopologyV1:
    supplied_topology = (
        supplied
        if isinstance(supplied, EvidenceTopologyV1)
        else evidence_topology_from_dict(supplied)
        if isinstance(supplied, Mapping)
        else None
    )
    if execution_lock is not None:
        source_destination = evidence_destination_from_dict(
            _mapping(
                execution_lock.get("source_evidence_destination"),
                "approved source evidence destination",
            )
        )
        result = evidence_destination_from_dict(result_destination)
        source_lock_digest = str(
            execution_lock.get("source_lock_digest") or ""
        )
        pre = _consistent_drift_check(
            rows,
            field="source_pre_run_drift",
            expected_digest=source_lock_digest,
            missing_reason="pre-run source drift verification was not recorded",
        )
        post = _consistent_drift_check(
            rows,
            field="source_post_run_drift",
            expected_digest=source_lock_digest,
            missing_reason="post-run source drift verification was not recorded",
        )
        if int(execution_lock.get("evidence_checkpoint_cells") or 0):
            checkpoint = _consistent_drift_check(
                rows,
                field="source_checkpoint_drift",
                expected_digest=source_lock_digest,
                missing_reason=(
                    "first-cell source drift verification was not recorded"
                ),
            )
            if checkpoint.status != "matched":
                raise ValueError(
                    "approved V3 comparison first-cell source checkpoint "
                    f"is {checkpoint.status}"
                )
        topology = EvidenceTopologyV1(
            source_destination=source_destination,
            result_destination=result,
            source_lock_digest=source_lock_digest,
            pre_run_drift=pre,
            post_run_drift=post,
            execution_identity=str(
                execution_lock.get("evidence_topology_identity") or ""
            ),
        )
        if (
            supplied_topology is not None
            and supplied_topology.to_dict() != topology.to_dict()
        ):
            raise ValueError(
                "supplied V3 evidence topology disagrees with final attempt rows"
            )
    elif supplied_topology is not None:
        topology = supplied_topology
    else:
        raise ValueError(
            "ComparisonResultV3 requires an approved evidence topology"
        )
    if topology.result_destination.to_dict() != dict(result_destination):
        raise ValueError(
            "V3 evidence topology result destination disagrees with rows"
        )
    if execution_lock is not None:
        if topology.source_destination.project_slug != execution_lock.get(
            "source_evidence_project"
        ):
            raise ValueError(
                "V3 evidence topology source project disagrees with approval"
            )
        if topology.source_lock_digest != execution_lock.get(
            "source_lock_digest"
        ):
            raise ValueError(
                "V3 evidence topology source lock disagrees with approval"
            )
    return topology


def _consistent_drift_check(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    expected_digest: str,
    missing_reason: str,
) -> EvidenceDriftCheckV1:
    observed = [
        dict(value)
        for row in rows
        if isinstance((value := row.get(field)), Mapping)
    ]
    if not observed:
        return EvidenceDriftCheckV1(
            status="unavailable",
            expected_digest=expected_digest,
            reason=missing_reason,
        )
    if len(observed) != len(rows) or any(
        value != observed[0] for value in observed[1:]
    ):
        raise ValueError(f"comparison rows disagree on {field}")
    parsed = EvidenceDriftCheckV1(
        status=str(observed[0].get("status") or ""),  # type: ignore[arg-type]
        expected_digest=str(
            observed[0].get("expected_digest") or expected_digest
        ),
        observed_digest=(
            str(observed[0].get("observed_digest"))
            if observed[0].get("observed_digest")
            else None
        ),
        reason=(
            str(observed[0].get("reason"))
            if observed[0].get("reason")
            else None
        ),
    )
    if parsed.expected_digest != expected_digest:
        raise ValueError(f"{field} expected digest disagrees with approval")
    return parsed


def _task_validity_v3(
    pairs: Sequence[PairedCaseV3],
    *,
    topology: EvidenceTopologyV1,
) -> tuple[TaskValidityV1, ...]:
    by_task: dict[str, list[PairedCaseV3]] = {}
    for pair in pairs:
        by_task.setdefault(pair.task_id, []).append(pair)
    result: list[TaskValidityV1] = []
    topology_statuses = {
        topology.pre_run_drift.status,
        topology.post_run_drift.status,
    }
    for task_id, task_pairs in sorted(by_task.items()):
        discriminating = sorted(
            {
                change.id
                for pair in task_pairs
                for change in pair.dimension_changes
                if change.role == "outcome"
                and change.status in {"improved", "regressed"}
            }
        )
        shared_failures = sorted(
            {
                change.id
                for pair in task_pairs
                for change in pair.dimension_changes
                if change.role in {"outcome", "safety_gate"}
                and change.critical
                and change.baseline is False
                and change.candidate is False
            }
        )
        statuses = {
            _pair_task_validity(
                pair.baseline,
                pair.candidate,
                pair.dimension_changes,
            )
            for pair in task_pairs
        }
        pair_outcomes = {pair.status for pair in task_pairs}
        if "drifted" in topology_statuses:
            status = "drifted"
            reasons = ("the immutable source project drifted during execution",)
            blockers = (f"{task_id}: source evidence drifted",)
        elif "unavailable" in topology_statuses:
            status = "inconclusive"
            reasons = ("source drift verification is incomplete",)
            blockers = (f"{task_id}: source drift verification is unavailable",)
        elif "invalid" in statuses:
            status = "invalid"
            reasons = ("the task contract or deterministic output was invalid",)
            blockers = (f"{task_id}: task evidence is invalid",)
        elif "inconclusive" in statuses or "incomplete" in pair_outcomes or (
            len(pair_outcomes & {"improved", "regressed", "mixed"}) > 1
        ):
            status = "inconclusive"
            reasons = ("repeated aligned attempts did not agree",)
            blockers = (f"{task_id}: aligned attempts disagree",)
        elif statuses == {"non_discriminating"}:
            status = "non_discriminating"
            reasons = (
                (
                    "both revisions retained the same critical failure"
                    if shared_failures
                    else "the task produced stable non-regression evidence "
                    "without an outcome delta"
                ),
            )
            blockers = tuple(
                f"{task_id}: {dimension} failed in both arms"
                for dimension in shared_failures
            )
        else:
            status = "valid"
            reasons = (
                (
                    "the task produced a stable aligned behavioral contrast",
                )
                if discriminating
                else (
                    "the task produced stable non-regression evidence with "
                    "candidate critical dimensions passing",
                )
            )
            blockers = ()
        result.append(
            TaskValidityV1(
                task_id=task_id,
                status=status,  # type: ignore[arg-type]
                reasons=tuple(reasons),
                discriminating_dimensions=tuple(discriminating),
                blockers=tuple(blockers),
            )
        )
    return tuple(result)


def _behavioral_summary_v3(
    legacy: BehavioralSummaryV1,
    *,
    paired_cases: Sequence[PairedCaseV3],
    task_validity: Sequence[TaskValidityV1],
) -> BehavioralSummaryV1:
    if legacy.status in {"invalid", "incomplete"}:
        return legacy
    blockers = tuple(
        dict.fromkeys(
            [
                *(
                    blocker
                    for validity in task_validity
                    for blocker in validity.blockers
                ),
                *(
                    f"{pair.task_id}: {change.id} failed for the candidate"
                    for pair in paired_cases
                    for change in pair.dimension_changes
                    if change.role in {"outcome", "safety_gate"}
                    and change.critical
                    and change.candidate is False
                ),
            ]
        )
    )
    improved = sum(item.status == "improved" for item in paired_cases)
    regressed = sum(item.status == "regressed" for item in paired_cases)
    mixed = sum(item.status == "mixed" for item in paired_cases)
    unchanged = sum(item.status == "unchanged" for item in paired_cases)
    incomplete = sum(item.status == "incomplete" for item in paired_cases)
    candidate_critical_failures = sum(
        change.critical
        and change.role in {"outcome", "safety_gate"}
        and change.candidate is False
        for pair in paired_cases
        for change in pair.dimension_changes
    )
    if mixed or (improved and regressed):
        status: BehavioralStatus = "mixed"
        recommendation = (
            "MIXED — outcome improvements and regressions coexist."
        )
        claim = None
        next_action = "Inspect the named regressed dimensions before promotion."
    elif regressed:
        status = "regressed"
        recommendation = "REGRESSED — the candidate is worse on a locked outcome."
        claim = (
            f"The candidate regressed on {regressed} aligned outcome pair(s)."
        )
        next_action = "Do not promote the candidate; inspect the regressions."
    elif improved and not blockers:
        status = "improved"
        recommendation = (
            "IMPROVED — the candidate improved a locked outcome without "
            "a critical regression."
        )
        claim = (
            f"The candidate improved {improved} aligned outcome pair(s), "
            "regressed none, and passed every candidate critical dimension."
        )
        next_action = "Run the separately approved confirmation cohort."
    else:
        status = "unchanged"
        recommendation = (
            "UNCHANGED — no release-qualifying behavioral improvement was "
            "established."
        )
        claim = (
            f"No release-qualifying improvement was established across "
            f"{len(paired_cases)} aligned pair(s)."
        )
        next_action = (
            "Repair the named blockers or use harder pre-frozen tasks."
            if blockers
            else "Use harder pre-frozen tasks if the decision still matters."
        )
    return BehavioralSummaryV1(
        status=status,
        recommendation=recommendation,
        improved_pairs=improved,
        regressed_pairs=regressed,
        mixed_pairs=mixed,
        unchanged_pairs=unchanged,
        incomplete_pairs=incomplete,
        candidate_critical_failures=candidate_critical_failures,
        critical_blockers=blockers,
        supported_claim=claim,
        limitations=legacy.limitations,
        next_action=next_action,
    )


def _apply_v3_decision_validity(
    decision: DecisionSummaryV1,
    *,
    task_validity: Sequence[TaskValidityV1],
    topology: EvidenceTopologyV1,
    release_note_coverage: Sequence[Mapping[str, Any]],
) -> DecisionSummaryV1:
    blockers = [
        blocker
        for item in task_validity
        if item.status in {"drifted", "invalid", "inconclusive"}
        for blocker in item.blockers
    ]
    for label, drift in (
        ("pre-run", topology.pre_run_drift),
        ("post-run", topology.post_run_drift),
    ):
        if drift.status != "matched":
            blockers.append(
                f"{label} source drift check is {drift.status}"
            )
    unqualified_release_notes = [
        str(item.get("release_note") or "")
        for item in release_note_coverage
        if str(item.get("status") or "") == "unqualified"
    ]
    if decision.release_target and unqualified_release_notes:
        blockers.append(
            "unqualified release-note behavior(s): "
            + ", ".join(sorted(unqualified_release_notes))
        )
    if not blockers:
        return decision
    status: DecisionStatus = (
        "invalid"
        if any(item.status in {"drifted", "invalid"} for item in task_validity)
        or topology.pre_run_drift.status == "drifted"
        or topology.post_run_drift.status == "drifted"
        else "hold"
    )
    return replace(
        decision,
        status=status,
        recommendation=(
            "INVALID — source or task validity failed."
            if status == "invalid"
            else "HOLD — the behavioral study is not release-complete."
        ),
        critical_blockers=tuple(
            dict.fromkeys((*decision.critical_blockers, *blockers))
        ),
        next_action=(
            "Repair the named task/topology blockers and create a new "
            "immutable preview."
        ),
        attestation=None,
    )


def _aligned_analysis_v3(
    pairs: Sequence[PairedCaseV3],
    *,
    rows: Sequence[Mapping[str, Any]],
    study_intent: str,
    task_validity: Sequence[TaskValidityV1],
) -> AlignedAnalysisV1:
    dimensions: dict[str, AlignedDimensionV1] = {}
    for pair in pairs:
        for change in pair.dimension_changes:
            dimensions.setdefault(
                change.id,
                AlignedDimensionV1(
                    id=change.id,
                    label=change.label,
                    role=change.role,
                    critical=change.critical,
                ),
            )
    arms = []
    for arm_id, label in (("baseline", "Baseline"), ("candidate", "Candidate")):
        revisions = _consistent_source_revisions(
            [
                row
                for row in rows
                if str(row.get("variant_id") or "") == arm_id
            ],
            required=False,
            label=f"{arm_id} aligned analysis",
        )
        arms.append(
            AlignedArmV1(
                id=arm_id,
                label=label,
                source_revision=(
                    {"revisions": [item.to_dict() for item in revisions]}
                    if revisions
                    else None
                ),
            )
        )
    validities = {item.task_id: item for item in task_validity}
    task_summaries = []
    for task_id in sorted(validities):
        selected = [item for item in pairs if item.task_id == task_id]
        task_summaries.append(
            TaskStratifiedSummaryV1(
                task_id=task_id,
                validity=validities[task_id].status,
                pair_counts=dict(
                    Counter(item.status for item in selected)
                ),
                blockers=validities[task_id].blockers,
            )
        )
    return AlignedAnalysisV1(
        study_intent=study_intent,
        reference_arm="baseline",
        arms=tuple(arms),
        contrasts=(
            AlignedContrastV1(
                id="candidate-vs-baseline",
                reference_arm="baseline",
                treatment_arms=("candidate",),
                dimensions=tuple(dimensions[key] for key in sorted(dimensions)),
            ),
        ),
        aligned_attempts=tuple(
            AlignedAttemptSetV1(
                alignment_id=pair.pair_id,
                task_id=pair.task_id,
                task_label=pair.task_label,
                harness=pair.harness,
                attempt=pair.attempt,
                attempt_ids_by_arm={
                    "baseline": pair.baseline.attempt_id,
                    "candidate": pair.candidate.attempt_id,
                },
            )
            for pair in pairs
            if pair.baseline is not None and pair.candidate is not None
        ),
        task_summaries=tuple(task_summaries),
    )


def _scorer_revisions_v3(
    execution_lock: Mapping[str, Any] | None,
) -> tuple[LockDescriptorV1, ...]:
    if execution_lock is None:
        return ()
    scorers = _mapping_or_empty(execution_lock.get("scorer_digests"))
    return tuple(
        LockDescriptorV1(
            id=str(scorer_id),
            label=str(scorer_id).replace("-", " ").title(),
            digest=str(digest),
            details={"kind": "scorer"},
        )
        for scorer_id, digest in sorted(scorers.items())
    )


def _runtime_locks_v3(
    rows: Sequence[Mapping[str, Any]],
    *,
    execution_lock: Mapping[str, Any] | None = None,
) -> tuple[LockDescriptorV1, ...]:
    values: dict[str, dict[str, Any]] = {}
    for row in rows:
        digest = str(row.get("execution_fingerprint") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            continue
        key = (
            f"{row.get('task_id') or row.get('task_name') or 'task'}-"
            f"{row.get('variant_id') or 'arm'}-"
            f"{row.get('harness') or 'harness'}"
        )
        values.setdefault(
            key,
            {
                "digest": digest,
                "details": {
                    "task_id": str(
                        row.get("task_id") or row.get("task_name") or ""
                    ),
                    "variant_id": str(row.get("variant_id") or ""),
                    "harness": str(row.get("harness") or ""),
                    "backend": (
                        row.get("harbor_environment")
                        or _mapping_or_empty(
                            row.get("sandbox_runtime")
                        ).get("provider")
                        or "harbor-docker"
                    ),
                },
            },
        )
        if values[key]["digest"] != digest:
            raise ValueError(f"runtime execution identity drifted for {key}")
    for key, digest in sorted(
        _mapping_or_empty(
            execution_lock.get("runtime_lock_digests")
            if execution_lock is not None
            else None
        ).items()
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
            raise ValueError(f"approved runtime lock digest is invalid for {key}")
        values[f"approved-{key}"] = {
            "digest": str(digest),
            "details": {
                "kind": "approved_runtime_lock",
                "runtime_lock_id": str(key),
            },
        }
    return tuple(
        LockDescriptorV1(
            id=key,
            label=key.replace("-", " ").title(),
            digest=str(value["digest"]),
            details=dict(value["details"]),
        )
        for key, value in sorted(values.items())
    )


def _release_note_coverage_v3(
    raw: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    allowed_statuses = {
        "observed_delta",
        "already_on_main",
        "infrastructure_only",
        "unqualified",
        "not_applicable",
    }
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw, start=1):
        value = dict(item)
        _reject_unknown(
            value,
            {
                "release_note",
                "status",
                "task_ids",
                "dimensions",
                "infrastructure_gates",
                "rationale",
            },
            f"release-note coverage {index}",
        )
        release_note = validate_id(
            str(value.get("release_note") or ""),
            kind="release-note behavior",
        )
        if release_note in seen:
            raise ValueError(
                f"release-note behavior {release_note!r} was classified twice"
            )
        seen.add(release_note)
        status = str(value.get("status") or "")
        if status not in allowed_statuses:
            raise ValueError(
                f"release-note behavior {release_note!r} has unsupported status"
            )
        rationale = _text(
            value.get("rationale"),
            f"release-note behavior {release_note} rationale",
            2000,
        )
        result.append(
            {
                "release_note": release_note,
                "status": status,
                "task_ids": list(
                    _string_tuple(
                        value.get("task_ids") or [],
                        "release-note task id",
                        allow_empty=True,
                    )
                ),
                "dimensions": list(
                    _string_tuple(
                        value.get("dimensions") or [],
                        "release-note dimension",
                        allow_empty=True,
                    )
                ),
                "infrastructure_gates": list(
                    _string_tuple(
                        value.get("infrastructure_gates") or [],
                        "release-note infrastructure gate",
                        allow_empty=True,
                    )
                ),
                "rationale": rationale,
            }
        )
    return tuple(result)


def _resolve_release_note_coverage_v3(
    rows: Sequence[Mapping[str, Any]],
    *,
    execution_lock: Mapping[str, Any] | None,
    supplied: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    embedded = [
        value
        for row in rows
        if isinstance((value := row.get("release_note_coverage")), list)
    ]
    supplied_coverage = _release_note_coverage_v3(supplied)
    if embedded:
        if len(embedded) != len(rows) or any(
            value != embedded[0] for value in embedded[1:]
        ):
            raise ValueError(
                "comparison rows disagree on release-note coverage"
            )
        row_coverage = _release_note_coverage_v3(
            [
                _mapping(item, "row release-note coverage")
                for item in embedded[0]
            ]
        )
        if supplied_coverage and supplied_coverage != row_coverage:
            raise ValueError(
                "supplied release-note coverage disagrees with final "
                "attempt rows"
            )
        return row_coverage
    coverage_required = bool(
        execution_lock
        and "release_note_coverage"
        in _mapping_or_empty(
            execution_lock.get("qualification_input_digests")
        )
    )
    if coverage_required:
        raise ValueError(
            "approved V3 comparison rows are missing locked release-note "
            "coverage"
        )
    return supplied_coverage


def _resolve_approved_comparison_execution_lock(
    rows: Sequence[Mapping[str, Any]],
    *,
    supplied: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    embedded = [
        dict(value)
        for row in rows
        if isinstance((value := row.get("approved_comparison")), Mapping)
    ]
    references = [
        str(row.get("approved_comparison_lock_digest") or "") for row in rows
    ]
    if supplied is None and not embedded:
        if any(references):
            raise ValueError(
                "comparison rows contain approval references but no approved "
                "execution lock was supplied"
            )
        return None
    expected = dict(supplied) if supplied is not None else embedded[0]
    _verify_approved_comparison_execution_lock(expected)
    expected_digest = str(expected["lock_digest"])
    for row in rows:
        embedded_value = row.get("approved_comparison")
        reference = str(row.get("approved_comparison_lock_digest") or "")
        if isinstance(embedded_value, Mapping):
            if dict(embedded_value) != expected:
                raise ValueError(
                    "comparison rows disagree on the approved execution lock"
                )
            if reference and reference != expected_digest:
                raise ValueError(
                    "comparison row approval reference disagrees with its "
                    "embedded execution lock"
                )
        elif reference != expected_digest:
            raise ValueError(
                "every comparison row must carry the approved comparison "
                "execution lock or its exact lock digest"
            )
    return expected


def _verify_approved_comparison_execution_lock(
    approved: Mapping[str, Any],
) -> None:
    if approved.get("schema_version") != 1:
        raise ValueError("approved comparison execution lock schema_version must be 1")
    if approved.get("kind") != "approved_comparison_execution":
        raise ValueError("approved comparison execution lock kind is invalid")
    supplied_digest = str(approved.get("lock_digest") or "")
    unsigned = {key: value for key, value in approved.items() if key != "lock_digest"}
    if len(supplied_digest) != 64 or supplied_digest != stable_digest(unsigned):
        raise ValueError("approved comparison execution lock digest does not match")
    expected_cells = approved.get("expected_cells")
    if not isinstance(expected_cells, list) or not expected_cells:
        raise ValueError("approved comparison execution lock has no expected cells")
    if int(approved.get("expected_cell_count") or 0) != len(expected_cells):
        raise ValueError("approved comparison expected cell count does not match")
    if str(approved.get("expected_cells_digest") or "") != stable_digest(
        expected_cells
    ):
        raise ValueError("approved comparison expected cell manifest digest does not match")
    validate_id(
        str(approved.get("comparison_id") or ""),
        kind="approved comparison id",
    )
    for key in (
        "preview_digest",
        "spec_digest",
        "taskset_digest",
        "private_labels_digest",
    ):
        value = str(approved.get(key) or "")
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"approved comparison {key} must be an exact digest")
    approval_digest = str(approved.get("approval_digest") or "")
    if approval_digest and (
        len(approval_digest) != 64
        or any(char not in "0123456789abcdef" for char in approval_digest)
    ):
        raise ValueError("approved comparison approval_digest must be an exact digest")
    if not str(approved.get("evidence_project") or ""):
        raise ValueError("approved comparison evidence project must be exact")
    evidence_destination = _evidence_destination(
        approved.get("evidence_destination"),
        "approved evidence destination",
    )
    if evidence_destination["project_slug"] != approved.get("evidence_project"):
        raise ValueError(
            "approved evidence destination disagrees with its project lock"
        )
    source_project, source_destination_raw, source_lock_digest = (
        _verified_source_topology_lock(approved)
    )
    checkpoint_cells = _non_negative_int(
        approved.get("evidence_checkpoint_cells", 0),
        "approved evidence checkpoint cells",
    )
    if checkpoint_cells > len(expected_cells):
        raise ValueError(
            "approved evidence checkpoint exceeds the expected cell count"
        )
    scorer_digests = approved.get("scorer_digests")
    if not isinstance(scorer_digests, Mapping) or not scorer_digests:
        raise ValueError("approved comparison must lock scorer digests")
    for scorer_id, digest in scorer_digests.items():
        if (
            not str(scorer_id)
            or len(str(digest)) != 64
            or any(char not in "0123456789abcdef" for char in str(digest))
        ):
            raise ValueError("approved comparison scorer digest is invalid")
    _verify_approved_runtime_locks(approved)
    qualification_inputs = approved.get("qualification_input_digests") or {}
    if not isinstance(qualification_inputs, Mapping) or any(
        not str(name)
        or not re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", str(digest))
        for name, digest in qualification_inputs.items()
    ):
        raise ValueError(
            "approved comparison qualification input digest is invalid"
        )
    approval_bindings = approved.get("approval_input_bindings") or {}
    if not isinstance(approval_bindings, Mapping) or any(
        not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,99}", str(name))
        or not re.fullmatch(r"[0-9a-f]{64}", str(digest))
        or qualification_inputs.get(name) != digest
        for name, digest in approval_bindings.items()
    ):
        raise ValueError("approved comparison approval input binding is invalid")
    _verify_approved_source_contract(
        approved,
        expected_cells=expected_cells,
        scorer_digests=scorer_digests,
    )
    _verify_approved_topology_identity(
        approved,
        source_project=source_project,
        source_destination=source_destination_raw,
        source_lock_digest=source_lock_digest,
    )
    _verify_cohort_lineage(
        _mapping(
            approved.get("cohort_lineage"),
            "approved comparison cohort lineage",
        )
    )
    for item in expected_cells:
        _verify_approved_expected_cell(
            _mapping(item, "approved comparison expected cell")
        )


def _verify_approved_runtime_locks(approved: Mapping[str, Any]) -> None:
    required = approved.get("runtime_locks_required", False)
    if not isinstance(required, bool):
        raise ValueError("approved runtime lock requirement must be boolean")
    locks = approved.get("runtime_lock_digests") or {}
    if not isinstance(locks, Mapping) or any(
        not str(name)
        or not re.fullmatch(r"[0-9a-f]{64}", str(digest))
        for name, digest in locks.items()
    ):
        raise ValueError("approved comparison runtime lock digest is invalid")
    if required and not locks:
        raise ValueError("approved comparison requires exact runtime locks")


def _verified_source_topology_lock(
    approved: Mapping[str, Any],
) -> tuple[str, Any, str]:
    source_project = str(approved.get("source_evidence_project") or "")
    source_destination_raw = approved.get("source_evidence_destination")
    source_lock_digest = str(approved.get("source_lock_digest") or "")
    if not any((source_project, source_destination_raw, source_lock_digest)):
        return source_project, source_destination_raw, source_lock_digest
    if not all((source_project, source_destination_raw, source_lock_digest)):
        raise ValueError("approved source evidence topology is incomplete")
    source_destination = _evidence_destination(
        source_destination_raw,
        "approved source evidence destination",
    )
    if source_destination["project_slug"] != source_project:
        raise ValueError(
            "approved source evidence destination disagrees with its project"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", source_lock_digest):
        raise ValueError("approved source evidence lock digest is invalid")
    return source_project, source_destination_raw, source_lock_digest


def _verify_approved_topology_identity(
    approved: Mapping[str, Any],
    *,
    source_project: str,
    source_destination: Any,
    source_lock_digest: str,
) -> None:
    expected = stable_digest(
        {
            "source_evidence_project": source_project,
            "source_evidence_destination": source_destination,
            "source_lock_digest": source_lock_digest,
            "result_evidence_project": approved.get("evidence_project"),
            "result_evidence_destination": approved.get(
                "evidence_destination"
            ),
        }
    )
    if approved.get("evidence_topology_identity") != expected:
        raise ValueError("approved evidence topology identity does not match")


def _verify_approved_source_contract(
    approved: Mapping[str, Any],
    *,
    expected_cells: Sequence[Any],
    scorer_digests: Mapping[str, Any],
) -> None:
    approved_inputs = _mapping(
        approved.get("approved_inputs"),
        "approved comparison inputs",
    )
    if str(approved.get("approved_inputs_digest") or "") != stable_digest(
        approved_inputs
    ):
        raise ValueError("approved comparison input manifest digest does not match")
    if approved_inputs.get("public_tasks_sha256") != approved.get(
        "taskset_digest"
    ) or approved_inputs.get("private_labels_sha256") != approved.get(
        "private_labels_digest"
    ):
        raise ValueError("approved comparison input digests disagree")
    if approved_inputs.get("evaluator_digests") != dict(scorer_digests):
        raise ValueError("approved evaluator artifacts disagree with scorer digests")
    candidate_revisions = tuple(
        _candidate_source_revision(item)
        for item in _sequence(
            approved.get("candidate_source_revisions"),
            "approved candidate source revisions",
            allow_empty=True,
        )
    )
    revisions_required = approved.get("candidate_source_revisions_required")
    if not isinstance(revisions_required, bool):
        raise ValueError(
            "approved candidate source revision requirement must be boolean"
        )
    if revisions_required and not candidate_revisions:
        raise ValueError(
            "source-changing comparison has no approved candidate source revision"
        )
    candidate_ids = sorted(
        {
            str(item.get("candidate_id") or "")
            for item in expected_cells
            if isinstance(item, Mapping)
            and str(item.get("variant_id") or "") == "candidate"
        }
    )
    if revisions_required and len(candidate_ids) != 1:
        raise ValueError(
            "source-changing comparison must lock one candidate identity"
        )
    if str(approved.get("candidate_source_identity_digest") or "") != stable_digest(
        {
            "candidate_ids": candidate_ids,
            "source_revisions": [
                item.to_dict() for item in candidate_revisions
            ],
        }
    ):
        raise ValueError("approved candidate source identity digest does not match")


def _verify_approved_expected_cell(cell: Mapping[str, Any]) -> None:
    applicable = cell.get("applicable")
    skip_reason = cell.get("skip_reason")
    if not isinstance(applicable, bool) or not isinstance(skip_reason, str):
        raise ValueError("approved comparison applicability contract is incomplete")
    if applicable == bool(skip_reason):
        raise ValueError("approved comparison applicability and skip reason disagree")
    provenance_digest = str(cell.get("integration_provenance_digest") or "")
    if (
        len(provenance_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in provenance_digest
        )
    ):
        raise ValueError(
            "approved comparison integration provenance digest is invalid"
        )
    identity = attempt_identity(
        task_id=str(cell.get("task_id") or ""),
        arm=str(cell.get("variant_id") or ""),
        harness=str(cell.get("harness") or ""),
        attempt=int(cell.get("trial_index") or 0),
        candidate=str(cell.get("candidate_id") or ""),
        runtime=str(cell.get("execution_fingerprint") or ""),
    )
    if cell.get("attempt_identity") != identity:
        raise ValueError("approved comparison expected cell identity does not match")
    if str(cell.get("attempt_id") or "") != attempt_id(**identity):
        raise ValueError("approved comparison expected attempt id does not match")


def _verified_approved_inputs(
    approved: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    _verify_approved_comparison_execution_lock(approved)
    inputs = dict(
        _mapping(approved.get("approved_inputs"), "approved comparison inputs")
    )
    checks = (
        (
            _frozen_public_tasks_path(
                repo_root,
                str(inputs["public_tasks_sha256"]),
            ),
            str(inputs["public_tasks_sha256"]),
            "approved public taskset",
        ),
        (
            _frozen_private_labels_path(
                repo_root,
                str(inputs["private_labels_sha256"]),
            ),
            str(inputs["private_labels_sha256"]),
            "approved private labels",
        ),
    )
    for path, digest, label in checks:
        if not path.is_file() or path.is_symlink() or _sha256_path(path) != digest:
            raise ValueError(f"{label} immutable copy changed")
    artifacts = _mapping(
        inputs.get("evaluator_artifacts"),
        "approved evaluator artifacts",
    )
    for evaluator_id, raw in artifacts.items():
        values = _mapping(raw, f"approved evaluator {evaluator_id} artifacts")
        for digest_field, kind in (
            ("scorer_sha256", "scorer"),
            ("verifier_sha256", "verifier"),
            ("calibration_sha256", "calibration"),
        ):
            digest = str(values.get(digest_field) or "")
            if not digest:
                continue
            path = _frozen_evaluator_artifact_path(
                repo_root,
                digest,
                kind=kind,  # type: ignore[arg-type]
            )
            if (
                not path.is_file()
                or path.is_symlink()
                or _sha256_path(path) != digest
            ):
                raise ValueError(
                    f"approved {kind} {evaluator_id!r} immutable copy changed"
                )
    for raw in _sequence(
        inputs.get("task_resources"),
        "approved task resources",
        allow_empty=True,
    ):
        resource = _mapping(raw, "approved task resource")
        relative = _safe_resource_relative_path(
            resource.get("locked_relative"),
            label="approved task resource",
        )
        path = repo_root / relative
        digest = str(resource.get("sha256") or "")
        if (
            not path.is_file()
            or path.is_symlink()
            or _sha256_path(path) != digest
        ):
            raise ValueError("approved task resource immutable copy changed")
    return inputs


def _validate_approved_comparison_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: str,
    approved: Mapping[str, Any],
) -> None:
    expected_cells = [
        _mapping(item, "approved comparison expected cell")
        for item in _sequence(
            approved.get("expected_cells"),
            "approved comparison expected cells",
        )
    ]
    expected_by_attempt = {
        str(item.get("attempt_id") or ""): item for item in expected_cells
    }
    if len(expected_by_attempt) != len(expected_cells):
        raise ValueError("approved comparison manifest has duplicate attempt identities")
    expected_coordinates = [
        _attempt_coordinate(item, label="approved comparison expected cell")
        for item in expected_cells
    ]
    if len(set(expected_coordinates)) != len(expected_coordinates):
        raise ValueError("approved comparison manifest has duplicate coordinates")
    arm_bindings: dict[str, set[str]] = {}
    for item in expected_cells:
        arm_bindings.setdefault(str(item.get("variant_id") or ""), set()).add(
            str(item.get("candidate_id") or "")
        )
    drifted_arms = sorted(
        arm for arm, identities in arm_bindings.items() if len(identities) != 1
    )
    if drifted_arms:
        raise ValueError(
            "approved comparison manifest has per-arm identity drift: "
            + ", ".join(drifted_arms)
        )
    row_attempt_ids = [str(row.get("attempt_id") or "") for row in rows]
    if len(set(row_attempt_ids)) != len(row_attempt_ids):
        raise ValueError("comparison rows contain duplicate attempt identities")
    row_coordinates = [
        _attempt_coordinate(row, label="comparison row") for row in rows
    ]
    if len(set(row_coordinates)) != len(row_coordinates):
        raise ValueError("comparison rows contain duplicate attempt coordinates")
    if len(rows) != len(expected_cells):
        raise ValueError(
            "comparison row count disagrees with the approved cell manifest "
            f"(expected {len(expected_cells)}, got {len(rows)})"
        )
    if set(row_attempt_ids) != set(expected_by_attempt):
        missing = sorted(set(expected_by_attempt) - set(row_attempt_ids))
        unexpected = sorted(set(row_attempt_ids) - set(expected_by_attempt))
        raise ValueError(
            "comparison attempt identities disagree with the approved cell manifest "
            f"(missing={missing}, unexpected={unexpected})"
        )
    if not source:
        raise ValueError("comparison source run id must not be empty")
    row_run_ids = {str(row.get("run_id") or "") for row in rows}
    if row_run_ids != {source}:
        raise ValueError(
            "comparison row run ids disagree with the analyzed source run "
            f"{source!r}: {sorted(row_run_ids)}"
        )
    for row in rows:
        expected = expected_by_attempt[str(row["attempt_id"])]
        for key in (
            "task_id",
            "variant_id",
            "harness",
            "trial_index",
            "candidate_id",
            "execution_fingerprint",
            "attempt_identity",
            "applicable",
        ):
            observed = (
                row.get("task_id") or row.get("task_name")
                if key == "task_id"
                else row.get(key)
            )
            if observed != expected.get(key):
                raise ValueError(
                    "comparison row identity drifted from the approved cell "
                    f"{row['attempt_id']} at {key}"
                )
        if str(row.get("skip_reason") or "") != str(
            expected.get("skip_reason") or ""
        ):
            raise ValueError(
                "comparison row applicability drifted from the approved cell "
                f"{row['attempt_id']} at skip_reason"
            )
        if stable_digest(row.get("integration_provenance") or []) != str(
            expected.get("integration_provenance_digest") or ""
        ):
            raise ValueError(
                "comparison row integration provenance drifted from the "
                f"approved cell {row['attempt_id']}"
            )
    observed_revisions = _consistent_source_revisions(
        [
            row
            for row in rows
            if str(row.get("variant_id") or "") == "candidate"
        ],
        required=bool(approved.get("candidate_source_revisions_required")),
        label="comparison candidate",
    )
    approved_revisions = tuple(
        _candidate_source_revision(item)
        for item in _sequence(
            approved.get("candidate_source_revisions"),
            "approved candidate source revisions",
            allow_empty=True,
        )
    )
    if observed_revisions != approved_revisions:
        raise ValueError(
            "comparison candidate source revisions disagree with the approved "
            "candidate identity"
        )


def _attempt_coordinate(
    value: Mapping[str, Any],
    *,
    label: str,
) -> tuple[str, str, str, int]:
    task_id = str(value.get("task_id") or value.get("task_name") or "")
    variant_id = str(value.get("variant_id") or "")
    harness = str(value.get("harness") or "")
    attempt = int(value.get("trial_index") or value.get("attempt") or 0)
    if not task_id or not variant_id or not harness or attempt < 1:
        raise ValueError(f"{label} has incomplete attempt coordinates")
    return task_id, variant_id, harness, attempt


def write_comparison_result(
    result: ComparisonResult,
    *,
    destination: Path,
    approved_comparison: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "result.json"
    markdown_path = destination / "result.md"
    attempts_path = destination / "attempts.jsonl"
    if isinstance(result, ComparisonResultV2 | ComparisonResultV3):
        if not attempts_path.is_file():
            raise FileNotFoundError(
                "ComparisonResultV2/V3 requires final attempts.jsonl before "
                "result writing"
            )
        is_v3 = isinstance(result, ComparisonResultV3)
        approved_lock = _result_approved_comparison_lock(
            destination,
            supplied=approved_comparison,
        )
        recomputed = analyze_comparison_rows(
            comparison_id=result.comparison_id,
            preview_digest=result.preview_digest,
            rows=_read_jsonl(attempts_path, "comparison attempt rows"),
            source=result.source,
            expected_evidence_project=result.evidence_project,
            expected_source_evidence_project=(
                result.evidence_topology.source_destination.project_slug
                if is_v3
                else None
            ),
            approved_comparison=approved_lock,
            decision_policy=result.decision_policy,
            attestation=result.decision.attestation,
            result_schema_version=3 if is_v3 else 2,
            study_intent=(
                result.aligned_analysis.study_intent
                if is_v3
                else "candidate_comparison"
            ),
            evidence_topology=(
                result.evidence_topology if is_v3 else None
            ),
            release_note_coverage=(
                result.release_note_coverage if is_v3 else ()
            ),
            supersedes=result.supersedes if is_v3 else (),
        )
        if recomputed.to_dict() != result.to_dict():
            raise RuntimeError(
                "comparison result disagrees with the final exported attempt rows"
            )
    atomic_write_json(json_path, result.to_dict())
    reloaded = read_comparison_result(json_path)
    if reloaded.to_dict() != result.to_dict():
        raise RuntimeError("persisted comparison result does not round-trip exactly")
    _atomic_text(markdown_path, _result_markdown(result))
    artifacts = {
        path.name: _sha256_path(path)
        for path in (
            json_path,
            markdown_path,
            destination / "attempts.jsonl",
            destination / APPROVED_COMPARISON_LOCK_NAME,
        )
        if path.is_file()
    }
    atomic_write_json(
        destination / "reproduction.json",
        {
            "schema_version": 1,
            "comparison_id": result.comparison_id,
            "preview_digest": result.preview_digest,
            "result_digest": result.result_digest,
            "source": result.source,
            "artifacts": artifacts,
            "private_labels_included": False,
            "commands": {
                "inspect": f"uv run fugue result {result.comparison_id}",
                "replay": (
                    "uv run fugue demo source-use"
                    if result.source == "bundled-replay"
                    else None
                ),
            },
            "limitations": [
                "Private labels are intentionally excluded.",
                "A live rerun requires the original locked private labels, "
                "component locks, runtime locks, and exact-preview approval.",
            ],
        },
    )
    return json_path, markdown_path


def _result_approved_comparison_lock(
    destination: Path,
    *,
    supplied: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Resolve one full lock for content-addressed attempt-row references."""

    path = destination / APPROVED_COMPARISON_LOCK_NAME
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(
            "approved comparison result lock must be a regular, non-symlink file"
        )
    if supplied is not None:
        value = dict(supplied)
        _verify_approved_comparison_execution_lock(value)
        _write_consistent_json(
            path,
            value,
            label="approved comparison result lock",
        )
        persisted = _read_result_approved_comparison_lock(path)
        if persisted != value:
            raise ValueError(
                "persisted approved comparison result lock disagrees with the "
                "supplied execution lock"
            )
        return persisted
    if not path.is_file():
        return None

    return _read_result_approved_comparison_lock(path)


def _read_result_approved_comparison_lock(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("approved comparison result lock is unreadable") from exc
    if not isinstance(raw, dict):
        raise ValueError("approved comparison result lock must be a mapping")
    _verify_approved_comparison_execution_lock(raw)
    return raw


def read_comparison_result(path: Path) -> ComparisonResult:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("comparison result must be a mapping")
    version = raw.get("schema_version")
    if version == 1:
        allowed = {item.name for item in ComparisonResultV1.__dataclass_fields__.values()}
        _reject_unknown(raw, allowed, "comparison result")
        value = dict(raw)
        value["evidence_links"] = tuple(value.get("evidence_links") or ())
        value["paired_cases"] = tuple(value.get("paired_cases") or ())
        value["limitations"] = tuple(value.get("limitations") or ())
        result = ComparisonResultV1(**value)
    elif version == 2:
        allowed = {item.name for item in ComparisonResultV2.__dataclass_fields__.values()}
        _reject_unknown(raw, allowed, "comparison result")
        has_qualification_digest = bool(raw.get("qualification_digest"))
        value = dict(raw)
        value["evidence_links"] = tuple(value.get("evidence_links") or ())
        value["paired_cases"] = tuple(
            _paired_case(item) for item in value.get("paired_cases") or ()
        )
        value["limitations"] = tuple(value.get("limitations") or ())
        value["behavioral_summary"] = _behavioral_summary_from_dict(
            value.get("behavioral_summary")
        )
        value["decision"] = _decision_summary(value.get("decision"))
        value["candidate_source_revisions"] = tuple(
            _candidate_source_revision(item)
            for item in value.get("candidate_source_revisions") or ()
        )
        value.setdefault("evidence_project", None)
        value.setdefault("decision_policy", None)
        if value["decision_policy"] is not None:
            parsed_policy = _decision_policy(value["decision_policy"])
            value["decision_policy"] = (
                parsed_policy.to_dict() if parsed_policy is not None else None
            )
        value.setdefault("deterministic_summary", {})
        value.setdefault("judge_summary", {})
        value.setdefault("mechanism_summary", {})
        value.setdefault("operational_summary", {})
        value.setdefault("evidence_destination", {})
        if value["evidence_destination"]:
            value["evidence_destination"] = _evidence_destination(
                value["evidence_destination"],
                "comparison result evidence destination",
            )
        value.setdefault("qualification_digest", "")
        result = ComparisonResultV2(**value)
    elif version == 3:
        allowed = {
            item.name for item in ComparisonResultV3.__dataclass_fields__.values()
        }
        _reject_unknown(raw, allowed, "comparison result")
        required = {
            "behavioral_summary",
            "decision",
            "evidence_destination",
            "evidence_topology",
            "aligned_analysis",
            "task_validity",
            "release_note_coverage",
            "scorer_revisions",
            "runtime_locks",
            "supersedes",
            "qualification_digest",
            "result_digest",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValueError(
                "ComparisonResultV3 is missing required field(s): "
                + ", ".join(missing)
            )
        has_qualification_digest = True
        value = dict(raw)
        value["evidence_links"] = tuple(value.get("evidence_links") or ())
        value["paired_cases"] = tuple(
            _paired_case_v3(item) for item in value.get("paired_cases") or ()
        )
        value["limitations"] = tuple(value.get("limitations") or ())
        value["behavioral_summary"] = _behavioral_summary_from_dict(
            value.get("behavioral_summary")
        )
        value["decision"] = _decision_summary(value.get("decision"))
        value["candidate_source_revisions"] = tuple(
            _candidate_source_revision(item)
            for item in value.get("candidate_source_revisions") or ()
        )
        value["evidence_destination"] = _evidence_destination(
            value.get("evidence_destination"),
            "comparison result evidence destination",
        )
        value["evidence_topology"] = evidence_topology_from_dict(
            _mapping(value.get("evidence_topology"), "evidence topology")
        )
        value["aligned_analysis"] = aligned_analysis_from_dict(
            _mapping(value.get("aligned_analysis"), "aligned analysis")
        )
        value["task_validity"] = tuple(
            task_validity_from_dict(
                _mapping(item, "task validity")
            )
            for item in value.get("task_validity") or ()
        )
        value["release_note_coverage"] = _release_note_coverage_v3(
            tuple(
                _mapping(item, "release-note coverage")
                for item in value.get("release_note_coverage") or ()
            )
        )
        value["scorer_revisions"] = tuple(
            lock_descriptor_from_dict(
                _mapping(item, "scorer revision")
            )
            for item in value.get("scorer_revisions") or ()
        )
        value["runtime_locks"] = tuple(
            lock_descriptor_from_dict(
                _mapping(item, "runtime lock")
            )
            for item in value.get("runtime_locks") or ()
        )
        value["cohort_lineage"] = dict(
            _mapping(
                value.get("cohort_lineage"),
                "comparison cohort lineage",
            )
        )
        value["supersedes"] = tuple(
            superseded_result_from_dict(
                _mapping(item, "superseded result")
            )
            for item in value.get("supersedes") or ()
        )
        value.setdefault("evidence_project", None)
        value.setdefault("decision_policy", None)
        if value["decision_policy"] is not None:
            parsed_policy = _decision_policy(value["decision_policy"])
            value["decision_policy"] = (
                parsed_policy.to_dict() if parsed_policy is not None else None
            )
        result = ComparisonResultV3(**value)
    else:
        raise ValueError("comparison result schema_version must be 1, 2, or 3")
    if isinstance(result, ComparisonResultV2 | ComparisonResultV3):
        _verify_v2_result_integrity(
            result,
            has_qualification_digest=has_qualification_digest,
        )
    elif result.result_digest != _legacy_comparison_result_digest(result.to_dict()):
        raise ValueError("comparison result digest does not match")
    return result


def _bind_attempt_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(row)
    task_id = str(value.get("task_id") or value.get("task_name") or "")
    variant = str(value.get("variant_id") or "")
    harness = str(value.get("harness") or "")
    attempt = int(value.get("trial_index") or value.get("attempt") or 1)
    candidate_identity = str(
        value.get("candidate_digest")
        or value.get("candidate_lock_digest")
        or value.get("candidate_id")
        or variant
    )
    runtime_identity = str(
        value.get("execution_fingerprint")
        or value.get("runtime_lock_digest")
        or value.get("runtime_digest")
        or "unreported"
    )
    identity = attempt_identity(
        task_id=task_id,
        arm=variant,
        harness=harness,
        attempt=attempt,
        candidate=candidate_identity,
        runtime=runtime_identity,
    )
    supplied_identity = value.get("attempt_identity")
    if supplied_identity is not None:
        if not isinstance(supplied_identity, Mapping) or dict(
            supplied_identity
        ) != identity:
            raise ValueError(
                f"supplied attempt identity disagrees with locked coordinates "
                f"for {task_id}"
            )
    resolved_attempt_id = attempt_id(**identity)
    supplied = str(value.get("attempt_id") or "")
    if supplied and supplied != resolved_attempt_id:
        raise ValueError(
            f"attempt identity disagrees with locked coordinates for {task_id}"
        )
    value["attempt_id"] = resolved_attempt_id
    value["attempt_identity"] = identity
    return value


def _row_text(row: Mapping[str, Any] | None, key: str) -> str | None:
    if row is None:
        return None
    value = str(row.get(key) or "")
    return value or None


def _terminal_execution_status(
    row: Mapping[str, Any],
) -> Literal[
    "completed",
    "failed",
    "cancelled",
    "interrupted",
    "not_applicable",
] | None:
    status = str(
        row.get("status")
        or row.get("execution_status")
        or ""
    ).lower()
    if status in {"passed", "success", "succeeded", "completed"}:
        return "completed"
    if status in {
        "failed",
        "error",
        "infrastructure_failed",
        "timed_out",
        "timeout",
    }:
        return "failed"
    if status in {"cancelled", "interrupted", "not_applicable"}:
        return status  # type: ignore[return-value]
    return None


_BENCHMARK_OUTCOMES = frozenset(
    {"passed", "failed", "unscored", "not_applicable"}
)
_RUNTIME_OUTCOMES = frozenset(
    {"completed", "timed_out", "cancelled", "not_started", "not_applicable"}
)


def _explicit_row_outcome(
    row: Mapping[str, Any],
    key: str,
    allowed: frozenset[str],
) -> str | None:
    raw = row.get(key)
    if raw is None or raw == "":
        return None
    if not isinstance(raw, str) or raw not in allowed:
        raise ValueError(f"comparison row has invalid {key}: {raw!r}")
    return raw


def _agent_timeout_outcome(row: Mapping[str, Any]) -> bool:
    """Recognize exact Agent timeout evidence without conflating infra errors.

    New exports carry both governed outcome fields. Historical rows may lack
    them, so replay accepts only the exact Harbor exception class or an exact
    terminal structured Agent-timeout event. Any partially populated or
    contradictory new outcome pair fails closed as a timeout classification.
    """

    benchmark_outcome = _explicit_row_outcome(
        row, "benchmark_outcome", _BENCHMARK_OUTCOMES
    )
    runtime_outcome = _explicit_row_outcome(
        row, "runtime_outcome", _RUNTIME_OUTCOMES
    )
    if benchmark_outcome is not None or runtime_outcome is not None:
        return (
            benchmark_outcome == "failed"
            and runtime_outcome == "timed_out"
        )
    if row.get("exception_class") == "AgentTimeoutError":
        return True
    return any(
        isinstance(event, Mapping)
        and event.get("terminal") is True
        and event.get("origin") == "agent"
        and event.get("kind") == "agent_timeout"
        for event in row.get("error_events") or ()
    )


def _row_benchmark_outcome(
    row: Mapping[str, Any],
) -> Literal["passed", "failed", "unscored", "not_applicable"] | None:
    explicit = _explicit_row_outcome(
        row, "benchmark_outcome", _BENCHMARK_OUTCOMES
    )
    if explicit is not None:
        return explicit  # type: ignore[return-value]
    return "failed" if _agent_timeout_outcome(row) else None


def _row_runtime_outcome(
    row: Mapping[str, Any],
) -> Literal[
    "completed", "timed_out", "cancelled", "not_started", "not_applicable"
] | None:
    explicit = _explicit_row_outcome(row, "runtime_outcome", _RUNTIME_OUTCOMES)
    if explicit is not None:
        return explicit  # type: ignore[return-value]
    return "timed_out" if _agent_timeout_outcome(row) else None


def _paired_dimension_changes(
    baseline: Mapping[str, Any] | None,
    candidate: Mapping[str, Any] | None,
) -> tuple[DimensionChangeV1, ...]:
    if baseline is None or candidate is None:
        return ()
    baseline_scores = _mapping_or_empty(
        baseline.get("comparison_deterministic_scores")
    )
    candidate_scores = _mapping_or_empty(
        candidate.get("comparison_deterministic_scores")
    )
    baseline_critical = _mapping_or_empty(
        baseline.get("comparison_deterministic_criticality")
    )
    candidate_critical = _mapping_or_empty(
        candidate.get("comparison_deterministic_criticality")
    )
    dimensions = sorted(set(baseline_scores) | set(candidate_scores))
    changes: list[DimensionChangeV1] = []
    for dimension in dimensions:
        baseline_value = _bool_score(baseline_scores.get(dimension))
        candidate_value = _bool_score(candidate_scores.get(dimension))
        if baseline_value is None or candidate_value is None:
            status: DimensionStatus = "unavailable"
        elif baseline_value is False and candidate_value is True:
            status = "improved"
        elif baseline_value is True and candidate_value is False:
            status = "regressed"
        else:
            status = "unchanged"
        changes.append(
            DimensionChangeV1(
                id=str(dimension),
                status=status,
                baseline=baseline_value,
                candidate=candidate_value,
                critical=bool(
                    baseline_critical.get(dimension)
                    or candidate_critical.get(dimension)
                ),
            )
        )
    if changes:
        return tuple(changes)
    baseline_pass = _bool_score(baseline.get("pass"))
    candidate_pass = _bool_score(candidate.get("pass"))
    if baseline_pass is None or candidate_pass is None:
        status = "unavailable"
    elif baseline_pass is False and candidate_pass is True:
        status = "improved"
    elif baseline_pass is True and candidate_pass is False:
        status = "regressed"
    else:
        status = "unchanged"
    return (
        DimensionChangeV1(
            id="task_pass",
            status=status,
            baseline=baseline_pass,
            candidate=candidate_pass,
            critical=True,
        ),
    )


def _pair_status_v2(
    changes: Sequence[DimensionChangeV1],
) -> PairStatus:
    improved = any(item.status == "improved" for item in changes)
    regressed = any(item.status == "regressed" for item in changes)
    candidate_critical_failure = any(
        item.critical and item.candidate is False for item in changes
    )
    if improved and regressed:
        return "mixed"
    if regressed:
        return "regressed"
    # A critical dimension that failed in both arms is a blocker, not a
    # directional regression. Keep the observed dimension-level improvement,
    # but do not call the pair improved until every candidate critical
    # dimension passes. The behavioral summary names the exact blocker.
    if improved and candidate_critical_failure:
        return "unchanged"
    if improved:
        return "improved"
    return "unchanged"


def _pair_status_v3(
    changes: Sequence[DimensionChangeV2],
) -> PairStatus:
    behavioral = tuple(
        item
        for item in changes
        if item.role == "outcome"
    )
    improved = any(item.status == "improved" for item in behavioral)
    regressed = any(item.status == "regressed" for item in behavioral)
    candidate_critical_failure = any(
        item.critical and item.candidate is False for item in changes
    )
    if improved and regressed:
        return "mixed"
    if regressed:
        return "regressed"
    if improved and candidate_critical_failure:
        return "unchanged"
    if improved:
        return "improved"
    return "unchanged"


def _pair_task_validity(
    baseline: Mapping[str, Any] | None,
    candidate: Mapping[str, Any] | None,
    changes: Sequence[DimensionChangeV2],
) -> Literal[
    "valid",
    "non_discriminating",
    "drifted",
    "invalid",
    "inconclusive",
]:
    if baseline is None or candidate is None or any(
        item.status == "unavailable" for item in changes
    ):
        return "inconclusive"
    behavioral = tuple(
        item
        for item in changes
        if item.role == "outcome"
    )
    discriminating = any(
        item.status in {"improved", "regressed"} for item in behavioral
    )
    if not discriminating:
        return "non_discriminating"
    return "valid"


def _bool_score(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, Mapping):
        passed = value.get("passed")
        return passed if isinstance(passed, bool) else None
    return None


def _paired_attempt_view(
    row: Mapping[str, Any] | None,
) -> PairedAttemptV2 | None:
    if row is None:
        return None
    tool_names, tool_call_count = _observed_tool_activity(row)
    input_tokens, output_tokens = _row_token_usage(row)
    infrastructure = _drop_empty(
        {
            "backend": (
                row.get("harbor_environment")
                or _mapping_or_empty(row.get("sandbox_runtime")).get("provider")
                or _mapping_or_empty(row.get("sandbox_runtime")).get("type")
                or ("harbor-docker" if row.get("harbor_config") else None)
            ),
            "image_digest": (
                row.get("runtime_image_digest")
                or _mapping_or_empty(row.get("sandbox_runtime")).get("image_digest")
            ),
            "policy_attestation_digest": (
                row.get("sandbox_attestation_digest")
                or row.get("harbor_policy_attestation_digest")
            ),
            "policy_attestation_verified": row.get(
                "harbor_policy_attestation_verified"
            ),
            "conformance_receipt_digest": row.get(
                "harbor_conformance_receipt_digest"
            ),
            "conformance_status": row.get("harbor_conformance_status"),
            "infrastructure_conformance_complete": row.get(
                "infrastructure_conformance_complete"
            ),
            "infrastructure_receipt_digest": row.get(
                "infrastructure_receipt_digest"
            ),
            "infrastructure_gate_statuses": dict(
                _mapping_or_empty(row.get("infrastructure_gate_statuses"))
            ),
            "decision_facts": dict(
                _mapping_or_empty(row.get("decision_facts"))
            ),
            "host_verifier_receipts": dict(
                _mapping_or_empty(
                    row.get("comparison_host_verifier_receipts")
                )
            ),
            "privacy_contract_version": row.get("privacy_contract_version"),
            "local_artifact_privacy_scan_status": row.get(
                "local_artifact_privacy_scan_status"
            ),
            "local_artifact_privacy_scan_digest": row.get(
                "local_artifact_privacy_scan_digest"
            ),
            "local_artifact_privacy_match_count": row.get(
                "local_artifact_privacy_match_count"
            ),
            "hosted_evidence_privacy_scan_status": row.get(
                "hosted_evidence_privacy_scan_status"
            ),
            "hosted_evidence_privacy_scan_digest": row.get(
                "hosted_evidence_privacy_scan_digest"
            ),
            "hosted_evidence_privacy_match_count": row.get(
                "hosted_evidence_privacy_match_count"
            ),
            "privacy_complete": _privacy_scans_complete(row),
            "credential_leak": row.get("credential_leak"),
            "private_label_leak": row.get("private_label_leak"),
            "legacy_privacy_scan_complete": (
                row.get("privacy_scan_complete")
                if row.get("privacy_contract_version") != 2
                else None
            ),
            "private_label_boundary_verified": row.get(
                "private_label_boundary_verified"
            ),
            "cleanup_verified": (
                row.get("harbor_cleanup_verified")
                if row.get("harbor_cleanup_verified") is not None
                else row.get("sandbox_cleanup_verified")
            ),
            "orphaned": row.get("orphaned_sandbox"),
        },
        preserve_false=True,
    )
    return PairedAttemptV2(
        attempt_id=str(row.get("attempt_id") or ""),
        identity=dict(_mapping_or_empty(row.get("attempt_identity"))),
        prediction_id=_row_text(row, "prediction_id"),
        passed=_bool_score(row.get("pass")),
        execution_status=(
            _terminal_execution_status(row)
            or str(
                row.get("status")
                or row.get("execution_status")
                or "unknown"
            )
        ),
        evaluation_status=str(
            row.get("comparison_evaluation_status") or "unknown"
        ),
        evidence_status=_attempt_evidence_status(row),
        cost_usd=_observed_attempt_cost(row)[0],
        latency_sec=_row_latency_sec(row),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tool_calls=tool_call_count,
        tools=tuple(tool_names),
        queried_projects=tuple(sorted(_queried_projects(row))),
        scores=dict(_mapping_or_empty(row.get("comparison_deterministic_scores"))),
        evidence_links=_attempt_evidence_links(row),
        weave_agent_root_call_id=(
            _row_text(row, "weave_agent_root_call_id")
            or _row_text(row, "native_agent_root_call_id")
        ),
        otel_root_span_id=(
            _row_text(row, "otel_root_span_id")
            or _row_text(row, "native_root_span_id")
            or _row_text(row, "root_span_id")
        ),
        execution_fingerprint=_row_text(row, "execution_fingerprint"),
        runtime_lock_digest=(
            _row_text(row, "runtime_lock_digest")
            or _row_text(row, "runtime_digest")
        ),
        infrastructure=infrastructure,
        judge_reviews=_judge_reviews(row),
        benchmark_outcome=_row_benchmark_outcome(row),
        runtime_outcome=_row_runtime_outcome(row),
    )


def _paired_attempt_view_v3(
    row: Mapping[str, Any] | None,
) -> PairedAttemptV3 | None:
    legacy = _paired_attempt_view(row)
    if legacy is None or row is None:
        return None
    deterministic_scores = dict(
        _mapping_or_empty(row.get("comparison_deterministic_scores"))
    )
    judge_scores = {
        f"comparison.judge.{dimension}": value
        for dimension, value in _mapping_or_empty(
            row.get("comparison_judge_scores")
        ).items()
        if isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    }
    scores = {**deterministic_scores, **judge_scores}
    return PairedAttemptV3(
        attempt_id=legacy.attempt_id,
        identity=legacy.identity,
        prediction_id=legacy.prediction_id,
        passed=legacy.passed,
        execution_status=legacy.execution_status,
        evaluation_status=legacy.evaluation_status,
        evidence_status=legacy.evidence_status,
        cost_usd=legacy.cost_usd,
        latency_sec=legacy.latency_sec,
        input_tokens=legacy.input_tokens,
        output_tokens=legacy.output_tokens,
        tool_calls=legacy.tool_calls,
        tools=legacy.tools,
        queried_projects=legacy.queried_projects,
        scores=scores,
        score_explanations={
            dimension: (
                "Blind judge score; no rationale or private truth is published."
                if dimension.startswith("comparison.judge.")
                else _safe_score_explanation(
                    dimension,
                    _bool_score(value),
                    row=row,
                )
            )
            for dimension, value in scores.items()
        },
        judge_reviews=_judge_reviews(row),
        sanitized_answer_excerpt=_sanitized_answer_excerpt(row),
        actual_query_scope=tuple(sorted(_queried_projects(row))),
        reported_project_identity=_reported_project_identity(row),
        evidence_links=legacy.evidence_links,
        weave_agent_root_call_id=legacy.weave_agent_root_call_id,
        otel_root_span_id=legacy.otel_root_span_id,
        execution_fingerprint=legacy.execution_fingerprint,
        runtime_lock_digest=legacy.runtime_lock_digest,
        infrastructure=legacy.infrastructure,
        benchmark_outcome=legacy.benchmark_outcome,
        runtime_outcome=legacy.runtime_outcome,
    )


def _judge_reviews(row: Mapping[str, Any]) -> dict[str, JudgeReviewV1]:
    """Project only bounded, redacted judge summaries into public results."""

    reviews: dict[str, JudgeReviewV1] = {}
    for judge_id, raw in sorted(
        _mapping_or_empty(row.get("comparison_judges")).items()
    ):
        if not isinstance(raw, Mapping) or raw.get("status") not in {
            "scored",
            "missing_evidence",
        }:
            continue
        scores = [
            float(value)
            for value in _mapping_or_empty(raw.get("scores")).values()
            if isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ]
        if not scores:
            continue
        reason = str(
            raw.get("overall_assessment") or raw.get("rationale") or ""
        ).strip()
        reason = " ".join(
            str(
                redact_value(reason, secrets=secrets_from_env(os.environ))
            ).split()
        )[:500]
        if not reason:
            continue
        missing_evidence = raw.get("status") == "missing_evidence" or bool(
            raw.get("missing_evidence", False)
        )
        observed_cost = _row_number(raw, "observed_cost_usd")
        accounted_reserve = _row_number(raw, "accounted_reserve_usd")
        cost_status = str(raw.get("cost_status") or "unavailable")
        if cost_status not in {"observed", "unavailable"}:
            continue
        if cost_status == "observed" and observed_cost is None:
            continue
        if cost_status == "unavailable" and observed_cost is not None:
            continue
        reviews[str(judge_id)] = JudgeReviewV1(
            label=(
                "unusable"
                if missing_evidence
                else _judge_review_label(sum(scores) / len(scores))
            ),
            reason=reason,
            missing_evidence=missing_evidence,
            observed_cost_usd=observed_cost,
            accounted_reserve_usd=accounted_reserve,
            cost_status=cost_status,  # type: ignore[arg-type]
        )
    return reviews


def _judge_review_label(score: float) -> Literal[
    "unusable",
    "weak",
    "adequate",
    "strong",
    "exceptional",
]:
    if score < 0.25:
        return "unusable"
    if score < 0.5:
        return "weak"
    if score < 0.75:
        return "adequate"
    if score < 0.9:
        return "strong"
    return "exceptional"


def _paired_dimension_changes_v3(
    baseline: Mapping[str, Any] | None,
    candidate: Mapping[str, Any] | None,
) -> tuple[DimensionChangeV2, ...]:
    return tuple(
        DimensionChangeV2(
            id=item.id,
            label=item.id.rsplit(".", 1)[-1].replace("_", " ").title(),
            status=item.status,
            baseline=item.baseline,
            candidate=item.candidate,
            critical=item.critical,
            role=_locked_dimension_role(
                item.id,
                baseline=baseline,
                candidate=candidate,
            ),
            baseline_explanation=(
                _safe_score_explanation(
                    item.id,
                    item.baseline,
                    row=baseline,
                )
                if baseline is not None
                else None
            ),
            candidate_explanation=(
                _safe_score_explanation(
                    item.id,
                    item.candidate,
                    row=candidate,
                )
                if candidate is not None
                else None
            ),
        )
        for item in _paired_dimension_changes(baseline, candidate)
    )


def _locked_dimension_role(
    dimension: str,
    *,
    baseline: Mapping[str, Any] | None,
    candidate: Mapping[str, Any] | None,
) -> DimensionRole:
    observed: set[str] = set()
    for row in (baseline, candidate):
        if row is None:
            continue
        roles = _mapping_or_empty(row.get("comparison_dimension_roles"))
        raw = roles.get(dimension)
        if isinstance(raw, str):
            observed.add(raw)
    allowed = {
        "outcome",
        "mechanism",
        "safety_gate",
        "infrastructure",
        "efficiency",
    }
    if len(observed) != 1 or not observed <= allowed:
        raise ValueError(
            f"V3 comparison dimension {dimension!r} lacks one consistent "
            "locked role"
        )
    return next(iter(observed))  # type: ignore[return-value]


def _safe_score_explanation(
    dimension: str,
    passed: bool | None,
    *,
    row: Mapping[str, Any] | None,
) -> str:
    label = dimension.rsplit(".", 1)[-1].replace("_", " ")
    if passed is None:
        return f"{label}: the required evidence was unavailable."
    if dimension.endswith(
        ("locked_project_scope", "locked_source_scope")
    ) and row is not None:
        actual = sorted(_queried_projects(row))
        reported = _reported_project_identity(row)
        if passed:
            return (
                f"locked project scope passed; actual MCP reads stayed in "
                f"{', '.join(actual) if actual else 'the locked scope'}."
            )
        if actual and reported:
            return (
                "locked project scope failed in the serialized answer; "
                f"actual MCP scope was {', '.join(actual)}, while the answer "
                f"reported {reported}."
            )
        return (
            "locked project scope failed; the answer or normalized MCP evidence "
            "did not prove the exact locked source project."
        )
    return (
        f"{label} {'passed' if passed else 'failed'} under the pinned "
        "deterministic scorer."
    )


def _sanitized_answer_excerpt(row: Mapping[str, Any]) -> str | None:
    if not _privacy_scans_complete(row):
        return None
    raw = row.get("agent_response")
    if raw is None:
        raw = row.get("final_output")
    if raw is None:
        raw = row.get("answer")
    if raw is None:
        return None
    structured = _extract_structured_result(raw)
    secrets = secrets_from_env(os.environ)
    if isinstance(structured, Mapping | list | tuple):
        text = json.dumps(
            redact_value(structured, secrets=secrets),
            sort_keys=True,
            separators=(",", ":"),
        )
    else:
        text = redact_text(str(raw), secrets=secrets)
    text = " ".join(text.strip().split())
    if not text:
        return None
    return text.encode()[:1000].decode("utf-8", errors="ignore").rstrip()


def _reported_project_identity(row: Mapping[str, Any]) -> str | None:
    raw = row.get("agent_response")
    if raw is None:
        raw = row.get("final_output")
    if isinstance(raw, str):
        raw = _extract_structured_result(raw)
    if not isinstance(raw, Mapping):
        return None
    value = raw.get("source_project")
    if value is None:
        value = raw.get("project")
    return str(value).strip() or None if value is not None else None


def _attempt_evidence_links(
    row: Mapping[str, Any],
) -> tuple[AttemptEvidenceLinkV1, ...]:
    project = str(row.get("trace_project") or "")
    result: list[AttemptEvidenceLinkV1] = []
    receipt_raw = row.get("trace_receipt")
    try:
        receipt = (
            _evidence_destination(
                receipt_raw,
                "attempt evidence destination",
            )
            if isinstance(receipt_raw, Mapping)
            else {}
        )
    except ValueError:
        receipt = {}
    app_base_url = str(receipt.get("app_base_url") or "")
    destination_valid = bool(
        project
        and receipt.get("project_slug") == project
        and _safe_application_base_url(
            app_base_url,
            label="attempt evidence application origin",
        )
    )

    def add_call(
        kind: EvidenceLinkKind,
        call_id: str,
        stable_ref: str,
        *,
        relationship_ok: bool = True,
        missing_reason: str,
    ) -> None:
        if not call_id or not stable_ref:
            result.append(
                AttemptEvidenceLinkV1(
                    kind=kind,
                    status="missing",
                    reason=missing_reason,
                )
            )
            return
        expected_ref = _weave_call_ref(project, call_id)
        if (
            not destination_valid
            or not expected_ref
            or stable_ref != expected_ref
        ):
            result.append(
                AttemptEvidenceLinkV1(
                    kind=kind,
                    status="invalid",
                    ref=stable_ref,
                    reason=(
                        "evidence Call identity or application origin does "
                        "not match the locked destination"
                    ),
                )
            )
            return
        uri = _weave_call_url(
            project,
            call_id,
            app_base_url=app_base_url,
        )
        if not relationship_ok:
            result.append(
                AttemptEvidenceLinkV1(
                    kind=kind,
                    status="invalid",
                    ref=stable_ref,
                    url=uri,
                    reason="evidence relationship did not reconcile",
                )
            )
            return
        result.append(
            AttemptEvidenceLinkV1(
                kind=kind,
                status="resolved",
                ref=stable_ref,
                url=uri,
            )
        )

    evaluation_call_id = str(
        row.get("weave_evaluation_root_call_id")
        or row.get("evaluation_root_call_id")
        or ""
    )
    add_call(
        "evaluation_root",
        evaluation_call_id,
        str(row.get("weave_evaluation_root_ref") or ""),
        relationship_ok=bool(
            row.get("evaluation_root_object_verified") is True
            and row.get("evaluation_root_dataset_relationship_verified") is True
            and row.get("evaluation_root_prediction_relationship_verified") is True
            and row.get("evaluation_prediction_graph_verified") is True
        ),
        missing_reason="Evaluation root reference is unavailable",
    )
    dataset_ref = str(
        row.get("weave_dataset_ref")
        or row.get("weave_dataset_id")
        or row.get("dataset_id")
        or ""
    )
    dataset_relationship_ok = bool(
        row.get("dataset_version_object_verified") is True
        and row.get("evaluation_root_dataset_relationship_verified") is True
        and row.get("evaluation_prediction_graph_verified") is True
    )
    dataset_url = _weave_object_url_from_ref(
        project,
        dataset_ref,
        app_base_url=app_base_url,
    )
    if not dataset_ref:
        result.append(
            AttemptEvidenceLinkV1(
                kind="dataset",
                status="missing",
                reason="Dataset reference is unavailable",
            )
        )
    elif not destination_valid or not dataset_url:
        result.append(
            AttemptEvidenceLinkV1(
                kind="dataset",
                status="invalid",
                ref=dataset_ref,
                reason=(
                    "Dataset identity or application origin does not match "
                    "the locked destination"
                ),
            )
        )
    elif not dataset_relationship_ok:
        result.append(
            AttemptEvidenceLinkV1(
                kind="dataset",
                status="invalid",
                ref=dataset_ref,
                url=dataset_url,
                reason="evidence relationship did not reconcile",
            )
        )
    else:
        result.append(
            AttemptEvidenceLinkV1(
                kind="dataset",
                status="resolved",
                ref=dataset_ref,
                url=dataset_url,
            )
        )
    for kind, id_fields, ref_fields in (
        (
            "prediction_and_score",
            ("eval_predict_and_score_call_id",),
            ("eval_predict_and_score_ref",),
        ),
        (
            "prediction",
            ("weave_prediction_call_id", "prediction_call_id"),
            ("weave_prediction_ref",),
        ),
        (
            "agent_root",
            ("weave_agent_root_call_id", "native_agent_root_call_id"),
            ("weave_agent_root_ref",),
        ),
    ):
        ref = next(
            (str(row.get(field) or "") for field in id_fields if row.get(field)),
            "",
        )
        stable_ref = next(
            (
                str(row.get(ref_field) or "")
                for ref_field in ref_fields
                if row.get(ref_field)
            ),
            "",
        )
        relationship_ok = (
            str(row.get("trace_link_status") or "") == "linked"
            and row.get("agent_graph_verified") is True
            if kind == "agent_root"
            else (
                row.get("eval_predict_and_score_object_verified") is True
                and row.get("evaluation_root_prediction_relationship_verified")
                is True
                and row.get("evaluation_prediction_graph_verified") is True
                if kind == "prediction_and_score"
                else row.get("weave_prediction_object_verified") is True
                and row.get("prediction_child_relationship_verified") is True
                and row.get("evaluation_prediction_graph_verified") is True
            )
        )
        add_call(
            kind,  # type: ignore[arg-type]
            ref,
            stable_ref,
            relationship_ok=relationship_ok,
            missing_reason=(
                "Verified Agent root Call is unavailable"
                if kind == "agent_root"
                else f"{kind.replace('_', ' ').title()} Call is unavailable"
            ),
        )
    return tuple(result)


def _is_local_harbor_row(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("harbor_config")
        or str(row.get("harbor_environment") or "").startswith("local_harbor")
    )


def _local_harbor_conformance_failed(row: Mapping[str, Any]) -> bool:
    if not _is_local_harbor_row(row):
        return False
    return bool(
        row.get("harbor_conformance_status") == "failed"
        or row.get("harbor_policy_attestation_verified") is False
        or _privacy_scans_failed(row)
        or row.get("private_label_boundary_verified") is False
        or row.get("sandbox_cleanup_verified") is False
        or row.get("orphaned_sandbox") is True
    )


def _local_harbor_conformance_unavailable(row: Mapping[str, Any]) -> bool:
    if not _is_local_harbor_row(row) or _local_harbor_conformance_failed(row):
        return False
    return bool(
        row.get("harbor_conformance_status") != "passed"
        or row.get("harbor_policy_attestation_verified") is not True
        or not _privacy_scans_complete(row)
        or row.get("private_label_boundary_verified") is not True
        or row.get("sandbox_cleanup_verified") is not True
        or row.get("orphaned_sandbox") is not False
    )


def _privacy_scan_status(
    row: Mapping[str, Any],
    field: str,
) -> str:
    if row.get("privacy_contract_version") != 2:
        return "legacy"
    status = str(row.get(field) or "")
    return (
        status
        if status in {"passed", "failed", "unavailable", "not_applicable"}
        else "unavailable"
    )


def _privacy_scans_complete(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("privacy_contract_version") == 2
        and _privacy_scan_status(
            row, "local_artifact_privacy_scan_status"
        )
        == "passed"
        and _privacy_scan_status(
            row, "hosted_evidence_privacy_scan_status"
        )
        == "passed"
        and row.get("private_label_boundary_verified") is True
    )


def _privacy_scans_failed(row: Mapping[str, Any]) -> bool:
    if row.get("privacy_contract_version") != 2:
        return False
    return bool(
        _privacy_scan_status(row, "local_artifact_privacy_scan_status")
        == "failed"
        or _privacy_scan_status(
            row, "hosted_evidence_privacy_scan_status"
        )
        == "failed"
    )


def _privacy_scan_evidence_available(row: Mapping[str, Any]) -> bool:
    if row.get("privacy_contract_version") != 2:
        return False
    return bool(
        _privacy_scan_status(row, "local_artifact_privacy_scan_status")
        in {"passed", "failed"}
        and _privacy_scan_status(
            row, "hosted_evidence_privacy_scan_status"
        )
        in {"passed", "failed"}
        and isinstance(row.get("private_label_boundary_verified"), bool)
    )


def _attempt_evidence_status(row: Mapping[str, Any]) -> str:
    links = _attempt_evidence_links(row)
    if not str(row.get("trace_project") or ""):
        return "not_applicable"
    statuses = {item.status for item in links}
    if statuses == {"resolved"} and len(links) == 5:
        return "reconciled"
    return "invalid" if "invalid" in statuses else "missing"


def _queried_projects(row: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    for key in ("queried_projects", "mcp_queried_projects", "project_reads"):
        raw = row.get(key) or ()
        if isinstance(raw, str):
            result.add(raw)
        elif isinstance(raw, Sequence):
            result.update(str(item) for item in raw if str(item))
    return result


def _cross_project_queries(
    row: Mapping[str, Any], expected_project: str | None
) -> tuple[str, ...]:
    if not expected_project:
        return ()
    return tuple(sorted(_queried_projects(row) - {expected_project}))


def _observed_tool_activity(row: Mapping[str, Any]) -> tuple[list[str], int]:
    local_names: list[str] = []
    local_count = 0
    for item in row.get("tool_calls") or ():
        if isinstance(item, str):
            local_names.append(item)
            local_count += 1
        elif isinstance(item, Mapping):
            name = str(item.get("name") or item.get("tool_name") or "")
            if name:
                local_names.append(name)
            local_count += 1
    normalized_mcp_count = 0
    for item in row.get("mcp_tool_calls") or ():
        if not isinstance(item, Mapping):
            continue
        name = str(
            item.get("tool")
            or item.get("name")
            or item.get("tool_name")
            or ""
        )
        if name:
            local_names.append(name)
        normalized_mcp_count += 1
    explicit = row.get("mcp_tool_names") or ()
    if isinstance(explicit, str):
        local_names.append(explicit)
    elif isinstance(explicit, Sequence):
        local_names.extend(str(item) for item in explicit if str(item))
    traced = row.get("weave_tool_names")
    traced_count = 0
    traced_names: list[str] = []
    if isinstance(traced, Mapping):
        for raw_name, raw_count in traced.items():
            name = str(raw_name)
            count = (
                max(raw_count, 0)
                if isinstance(raw_count, int) and not isinstance(raw_count, bool)
                else 0
            )
            if name and count:
                traced_names.append(name)
            traced_count += count
    declared_count = row.get("tool_call_count")
    if isinstance(declared_count, int) and not isinstance(declared_count, bool):
        local_count = max(local_count, declared_count)
    # The normalized MCP receipt is attempt-scoped. Native Weave summaries may
    # include cumulative or non-MCP activity, so they are only a fallback when
    # an attempt has no normalized MCP calls.
    if normalized_mcp_count:
        return sorted(set(local_names)), normalized_mcp_count
    # A Harbor trajectory is attempt-local evidence of general Agent tool
    # activity. Keep it distinct from normalized MCP receipts above: it can
    # recover missing or cumulative native trace summaries, but it must never
    # be presented as proof that an MCP mechanism was used.
    if row.get("agent_trajectory_tool_activity_status") == "available":
        trajectory_count = row.get("agent_trajectory_tool_call_count")
        trajectory_names = row.get("agent_trajectory_tool_names") or {}
        if (
            isinstance(trajectory_count, int)
            and not isinstance(trajectory_count, bool)
            and trajectory_count >= 0
            and isinstance(trajectory_names, Mapping)
        ):
            return (
                sorted(
                    str(name)
                    for name, count in trajectory_names.items()
                    if str(name)
                    and isinstance(count, int)
                    and not isinstance(count, bool)
                    and count > 0
                ),
                trajectory_count,
            )
    if traced_names:
        return sorted(set(traced_names)), traced_count
    return sorted(set(local_names)), local_count


def _row_number(row: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _number_or_none(row.get(key))
        if value is not None:
            return value
    return None


def _observed_attempt_cost(
    row: Mapping[str, Any],
) -> tuple[float | None, str | None]:
    """Return one normalized observed cost without inventing missing usage.

    The in-cell adapter is the primary source. A timed-out Agent may not emit
    that terminal field even though the reconciled Weave usage collector did.
    Only an explicitly available Weave usage/cost receipt is accepted as the
    fallback; an unqualified numeric trace attribute remains unavailable.
    """

    for field_name in (
        "cost_usd",
        "observed_cost_usd",
        "total_cost_usd",
        "agent_cost_usd",
    ):
        value = _number_or_none(row.get(field_name))
        if value is not None:
            return value, field_name
    if (
        row.get("weave_usage_status") == "available"
        and row.get("weave_cost_status") == "available"
    ):
        value = _number_or_none(row.get("weave_total_cost_usd"))
        if value is not None:
            return value, "weave_total_cost_usd"
    return None, None


def _accounted_attempt_cost(row: Mapping[str, Any]) -> float | None:
    accounted = _row_number(row, "accounted_cost_usd")
    return accounted if accounted is not None else _observed_attempt_cost(row)[0]


def _row_latency_sec(row: Mapping[str, Any]) -> float | None:
    direct = _row_number(row, "latency_sec", "wall_time_sec")
    if direct is not None:
        return direct
    milliseconds = _row_number(row, "latency_ms")
    return round(milliseconds / 1000, 6) if milliseconds is not None else None


def _row_token_usage(
    row: Mapping[str, Any],
) -> tuple[float | None, float | None]:
    usage = _mapping_or_empty(row.get("usage"))
    sources: list[tuple[Any, Any]] = [
        (usage.get("input_tokens"), usage.get("output_tokens")),
        (row.get("input_tokens"), row.get("output_tokens")),
    ]
    if row.get("local_usage_status") == "available":
        sources.append((row.get("n_input_tokens"), row.get("n_output_tokens")))
    if row.get("weave_usage_status") == "available":
        sources.append((row.get("weave_input_tokens"), row.get("weave_output_tokens")))
    for raw_input, raw_output in sources:
        input_tokens = _number_or_none(raw_input)
        output_tokens = _number_or_none(raw_output)
        if (
            input_tokens is not None
            and output_tokens is not None
            and input_tokens >= 0
            and output_tokens >= 0
        ):
            return input_tokens, output_tokens
    return None, None


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _decision_attestation(
    raw: DecisionAttestationV1 | Mapping[str, Any] | None,
) -> DecisionAttestationV1 | None:
    if raw is None or isinstance(raw, DecisionAttestationV1):
        return raw
    value = _mapping(raw, "decision attestation")
    _reject_unknown(
        value,
        {"signer", "signed_result_digest", "signed_at", "review_status"},
        "decision attestation",
    )
    review_status = str(value.get("review_status") or "") or None
    if review_status not in {None, "accepted_actionable", "rejected"}:
        raise ValueError("unsupported decision attestation review status")
    return DecisionAttestationV1(
        signer=_text(value.get("signer"), "attestation signer", 300),
        signed_result_digest=_text(
            value.get("signed_result_digest"),
            "signed result digest",
            200,
        ),
        signed_at=_text(value.get("signed_at"), "attestation time", 100),
        review_status=review_status,  # type: ignore[arg-type]
    )


def _decision_summary(raw: Any) -> DecisionSummaryV1:
    value = _mapping(raw, "decision summary")
    _reject_unknown(
        value,
        {
            "status",
            "recommendation",
            "release_target",
            "candidate_sha",
            "evidence_grade",
            "gates",
            "critical_blockers",
            "limitations",
            "next_action",
            "human_signoff_required",
            "attestation",
        },
        "decision summary",
    )
    attestation = _decision_attestation(value.get("attestation"))
    status = str(value.get("status") or "")
    if status not in {
        "invalid",
        "blocked",
        "hold",
        "inconclusive",
        "ready_for_signoff",
        "go",
    }:
        raise ValueError("unsupported decision status")
    evidence_grade = str(value.get("evidence_grade") or "")
    if evidence_grade not in {"A", "B", "C", "invalid"}:
        raise ValueError("unsupported decision evidence grade")
    gates: list[DecisionGateResultV1] = []
    gate_ids: set[str] = set()
    for item in _sequence(
        value.get("gates") or [],
        "decision gate results",
        allow_empty=True,
    ):
        gate = _mapping(item, "decision gate result")
        _reject_unknown(
            gate,
            {
                "id",
                "label",
                "category",
                "status",
                "critical",
                "actual",
                "target",
            },
            "decision gate result",
        )
        gate_id = validate_id(
            str(gate.get("id") or ""),
            kind="decision gate id",
        )
        if gate_id in gate_ids:
            raise ValueError(f"duplicate decision gate id: {gate_id}")
        gate_ids.add(gate_id)
        category = str(gate.get("category") or "")
        if category not in {
            "integrity",
            "task",
            "infrastructure",
            "evidence",
            "efficiency",
            "privacy",
        }:
            raise ValueError("unsupported decision gate category")
        gate_status = str(gate.get("status") or "")
        if gate_status not in {"passed", "failed", "unavailable"}:
            raise ValueError("unsupported decision gate status")
        if not isinstance(gate.get("critical"), bool):
            raise ValueError("decision gate critical flag must be boolean")
        actual = gate.get("actual")
        target = gate.get("target")
        allowed_scalar = (str, int, float, bool)
        if actual is not None and not isinstance(actual, allowed_scalar):
            raise ValueError("decision gate actual must be a scalar or null")
        if not isinstance(target, allowed_scalar):
            raise ValueError("decision gate target must be a scalar")
        gates.append(
            DecisionGateResultV1(
                id=gate_id,
                label=_text(
                    gate.get("label"),
                    "decision gate label",
                    500,
                ),
                category=category,  # type: ignore[arg-type]
                status=gate_status,  # type: ignore[arg-type]
                critical=bool(gate["critical"]),
                actual=actual,
                target=target,
            )
        )
    human_signoff_required = value.get("human_signoff_required", True)
    if not isinstance(human_signoff_required, bool):
        raise ValueError("decision human_signoff_required must be boolean")
    return DecisionSummaryV1(
        status=status,  # type: ignore[arg-type]
        recommendation=_text(
            value.get("recommendation"), "decision recommendation", 1000
        ),
        release_target=str(value.get("release_target") or "") or None,
        candidate_sha=str(value.get("candidate_sha") or "") or None,
        evidence_grade=evidence_grade,  # type: ignore[arg-type]
        gates=tuple(gates),
        critical_blockers=tuple(
            str(item) for item in value.get("critical_blockers") or ()
        ),
        limitations=tuple(str(item) for item in value.get("limitations") or ()),
        next_action=_text(value.get("next_action"), "decision next action", 1000),
        human_signoff_required=human_signoff_required,
        attestation=attestation,
    )


def _attempt_evidence_link(raw: Any) -> AttemptEvidenceLinkV1:
    value = _mapping(raw, "attempt evidence link")
    _reject_unknown(
        value,
        {"kind", "status", "system", "ref", "url", "reason"},
        "attempt evidence link",
    )
    kind = str(value.get("kind") or "")
    if kind not in {
        "evaluation_root",
        "prediction_and_score",
        "prediction",
        "agent_root",
        "dataset",
    }:
        raise ValueError(f"unknown attempt evidence link kind: {kind}")
    status = str(value.get("status") or "")
    if status not in {"resolved", "missing", "invalid"}:
        raise ValueError(f"unknown attempt evidence link status: {status}")
    system = str(value.get("system") or "weave")
    if system != "weave":
        raise ValueError("attempt evidence link system must be weave")
    ref = _optional_text(value.get("ref"), "attempt evidence ref", 2_000)
    url = _optional_text(value.get("url"), "attempt evidence URL", 2_000)
    reason = _optional_text(value.get("reason"), "attempt evidence reason", 1_000)
    if status == "resolved":
        if not ref or not url:
            raise ValueError(
                "resolved attempt evidence requires a stable ref and URL"
            )
        if reason:
            raise ValueError(
                "resolved attempt evidence cannot carry an unresolved reason"
            )
    elif not reason:
        raise ValueError("unresolved attempt evidence requires a reason")
    return AttemptEvidenceLinkV1(
        kind=kind,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        system="weave",
        ref=ref,
        url=url,
        reason=reason,
    )


def _dimension_change(raw: Any) -> DimensionChangeV1:
    value = _mapping(raw, "dimension change")
    _reject_unknown(
        value,
        {"id", "status", "baseline", "candidate", "critical"},
        "dimension change",
    )
    status = str(value.get("status") or "")
    if status not in {"improved", "regressed", "unchanged", "unavailable"}:
        raise ValueError(f"unknown dimension change status: {status}")
    return DimensionChangeV1(
        id=_text(value.get("id"), "dimension id", 300),
        status=status,  # type: ignore[arg-type]
        baseline=(
            value.get("baseline")
            if isinstance(value.get("baseline"), bool)
            else None
        ),
        candidate=(
            value.get("candidate")
            if isinstance(value.get("candidate"), bool)
            else None
        ),
        critical=bool(value.get("critical")),
    )


def _dimension_change_v3(raw: Any) -> DimensionChangeV2:
    value = _mapping(raw, "V3 dimension change")
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
        "V3 dimension change",
    )
    status = str(value.get("status") or "")
    if status not in {"improved", "regressed", "unchanged", "unavailable"}:
        raise ValueError(f"unknown V3 dimension change status: {status}")
    role = str(value.get("role") or "")
    if role not in {
        "outcome",
        "mechanism",
        "safety_gate",
        "infrastructure",
        "efficiency",
    }:
        raise ValueError(f"unknown V3 dimension role: {role}")
    critical = value.get("critical")
    if not isinstance(critical, bool):
        raise ValueError("V3 dimension change critical must be boolean")
    return DimensionChangeV2(
        id=_text(value.get("id"), "V3 dimension id", 300),
        label=_text(value.get("label"), "V3 dimension label", 300),
        status=status,  # type: ignore[arg-type]
        baseline=(
            value.get("baseline")
            if isinstance(value.get("baseline"), bool)
            else None
        ),
        candidate=(
            value.get("candidate")
            if isinstance(value.get("candidate"), bool)
            else None
        ),
        critical=critical,
        role=role,  # type: ignore[arg-type]
        baseline_explanation=_optional_text(
            value.get("baseline_explanation"),
            "baseline score explanation",
            2000,
        ),
        candidate_explanation=_optional_text(
            value.get("candidate_explanation"),
            "candidate score explanation",
            2000,
        ),
    )


def _paired_outcome(
    raw: Any,
    *,
    label: str,
    allowed: frozenset[str],
) -> str | None:
    value = _optional_text(raw, label, 100)
    if value is not None and value not in allowed:
        raise ValueError(f"unknown {label}: {value}")
    return value


def _paired_attempt(raw: Any) -> PairedAttemptV2 | None:
    if raw is None:
        return None
    value = _mapping(raw, "paired attempt")
    allowed = {
        item.name for item in PairedAttemptV2.__dataclass_fields__.values()
    }
    _reject_unknown(value, allowed, "paired attempt")
    return PairedAttemptV2(
        attempt_id=_text(value.get("attempt_id"), "attempt id", 200),
        identity=dict(_mapping(value.get("identity"), "attempt identity")),
        prediction_id=str(value.get("prediction_id") or "") or None,
        passed=(
            value.get("passed") if isinstance(value.get("passed"), bool) else None
        ),
        execution_status=_text(
            value.get("execution_status"), "execution status", 100
        ),
        evaluation_status=_text(
            value.get("evaluation_status"), "evaluation status", 100
        ),
        evidence_status=_text(
            value.get("evidence_status"), "evidence status", 100
        ),
        cost_usd=_number_or_none(value.get("cost_usd")),
        latency_sec=_number_or_none(value.get("latency_sec")),
        input_tokens=_number_or_none(value.get("input_tokens")),
        output_tokens=_number_or_none(value.get("output_tokens")),
        tool_calls=_non_negative_int(value.get("tool_calls", 0), "tool calls"),
        tools=tuple(str(item) for item in value.get("tools") or ()),
        queried_projects=tuple(
            str(item) for item in value.get("queried_projects") or ()
        ),
        scores=dict(_mapping_or_empty(value.get("scores"))),
        evidence_links=tuple(
            _attempt_evidence_link(item)
            for item in value.get("evidence_links") or ()
        ),
        weave_agent_root_call_id=(
            str(value.get("weave_agent_root_call_id") or "") or None
        ),
        otel_root_span_id=str(value.get("otel_root_span_id") or "") or None,
        execution_fingerprint=(
            str(value.get("execution_fingerprint") or "") or None
        ),
        runtime_lock_digest=(
            str(value.get("runtime_lock_digest") or "") or None
        ),
        infrastructure=dict(_mapping_or_empty(value.get("infrastructure"))),
        judge_reviews={
            str(key): _judge_review(item)
            for key, item in _mapping_or_empty(
                value.get("judge_reviews")
            ).items()
        },
        benchmark_outcome=_paired_outcome(
            value.get("benchmark_outcome"),
            label="paired benchmark outcome",
            allowed=_BENCHMARK_OUTCOMES,
        ),  # type: ignore[arg-type]
        runtime_outcome=_paired_outcome(
            value.get("runtime_outcome"),
            label="paired runtime outcome",
            allowed=_RUNTIME_OUTCOMES,
        ),  # type: ignore[arg-type]
    )


def _paired_attempt_v3(raw: Any) -> PairedAttemptV3 | None:
    if raw is None:
        return None
    value = _mapping(raw, "V3 paired attempt")
    allowed = {
        item.name for item in PairedAttemptV3.__dataclass_fields__.values()
    }
    _reject_unknown(value, allowed, "V3 paired attempt")
    score_explanations = _mapping(
        value.get("score_explanations"), "V3 score explanations"
    )
    judge_reviews = _mapping_or_empty(value.get("judge_reviews"))
    return PairedAttemptV3(
        attempt_id=_text(value.get("attempt_id"), "V3 attempt id", 200),
        identity=dict(_mapping(value.get("identity"), "V3 attempt identity")),
        prediction_id=str(value.get("prediction_id") or "") or None,
        passed=(
            value.get("passed") if isinstance(value.get("passed"), bool) else None
        ),
        execution_status=_text(
            value.get("execution_status"), "V3 execution status", 100
        ),
        evaluation_status=_text(
            value.get("evaluation_status"), "V3 evaluation status", 100
        ),
        evidence_status=_text(
            value.get("evidence_status"), "V3 evidence status", 100
        ),
        cost_usd=_number_or_none(value.get("cost_usd")),
        latency_sec=_number_or_none(value.get("latency_sec")),
        input_tokens=_number_or_none(value.get("input_tokens")),
        output_tokens=_number_or_none(value.get("output_tokens")),
        tool_calls=_non_negative_int(value.get("tool_calls", 0), "tool calls"),
        tools=_string_tuple(
            value.get("tools") or [], "V3 tool", allow_empty=True
        ),
        queried_projects=_string_tuple(
            value.get("queried_projects") or [],
            "V3 queried project",
            allow_empty=True,
        ),
        scores=dict(_mapping_or_empty(value.get("scores"))),
        score_explanations={
            str(key): _text(item, "V3 score explanation", 2000)
            for key, item in score_explanations.items()
        },
        judge_reviews={
            str(key): _judge_review(item)
            for key, item in judge_reviews.items()
        },
        sanitized_answer_excerpt=_optional_text(
            value.get("sanitized_answer_excerpt"),
            "sanitized answer excerpt",
            1000,
        ),
        actual_query_scope=_string_tuple(
            value.get("actual_query_scope") or [],
            "V3 actual query scope",
            allow_empty=True,
        ),
        reported_project_identity=_optional_text(
            value.get("reported_project_identity"),
            "reported project identity",
            300,
        ),
        evidence_links=tuple(
            _attempt_evidence_link(item)
            for item in value.get("evidence_links") or ()
        ),
        weave_agent_root_call_id=(
            str(value.get("weave_agent_root_call_id") or "") or None
        ),
        otel_root_span_id=str(value.get("otel_root_span_id") or "") or None,
        execution_fingerprint=(
            str(value.get("execution_fingerprint") or "") or None
        ),
        runtime_lock_digest=(
            str(value.get("runtime_lock_digest") or "") or None
        ),
        infrastructure=dict(_mapping_or_empty(value.get("infrastructure"))),
        benchmark_outcome=_paired_outcome(
            value.get("benchmark_outcome"),
            label="V3 paired benchmark outcome",
            allowed=_BENCHMARK_OUTCOMES,
        ),  # type: ignore[arg-type]
        runtime_outcome=_paired_outcome(
            value.get("runtime_outcome"),
            label="V3 paired runtime outcome",
            allowed=_RUNTIME_OUTCOMES,
        ),  # type: ignore[arg-type]
    )


def _judge_review(raw: Any) -> JudgeReviewV1:
    value = _mapping(raw, "V3 judge review")
    _reject_unknown(
        value,
        {
            "label",
            "reason",
            "missing_evidence",
            "observed_cost_usd",
            "accounted_reserve_usd",
            "cost_status",
        },
        "V3 judge review",
    )
    label = str(value.get("label") or "")
    if label not in {
        "unusable",
        "weak",
        "adequate",
        "strong",
        "exceptional",
    }:
        raise ValueError(f"unknown V3 judge review label: {label}")
    missing_evidence = value.get("missing_evidence")
    if not isinstance(missing_evidence, bool):
        raise ValueError("V3 judge review missing_evidence must be boolean")
    observed_cost = _number_or_none(value.get("observed_cost_usd"))
    accounted_reserve = _number_or_none(value.get("accounted_reserve_usd"))
    cost_status = str(value.get("cost_status") or "unavailable")
    if cost_status not in {"observed", "unavailable"}:
        raise ValueError("V3 judge review cost_status is unsupported")
    if observed_cost is not None and observed_cost < 0:
        raise ValueError("V3 judge review observed cost must be non-negative")
    if accounted_reserve is not None and accounted_reserve < 0:
        raise ValueError("V3 judge review accounted reserve must be non-negative")
    if (cost_status == "observed") != (observed_cost is not None):
        raise ValueError("V3 judge review cost evidence is inconsistent")
    return JudgeReviewV1(
        label=label,  # type: ignore[arg-type]
        reason=_text(value.get("reason"), "V3 judge review reason", 500),
        missing_evidence=missing_evidence,
        observed_cost_usd=observed_cost,
        accounted_reserve_usd=accounted_reserve,
        cost_status=cost_status,  # type: ignore[arg-type]
    )


def _paired_case(raw: Any) -> PairedCaseV2:
    value = _mapping(raw, "paired case")
    allowed = {item.name for item in PairedCaseV2.__dataclass_fields__.values()}
    _reject_unknown(value, allowed, "paired case")
    status = str(value.get("status") or "")
    if status not in {"improved", "regressed", "mixed", "unchanged", "incomplete"}:
        raise ValueError(f"unknown paired case status: {status}")
    return PairedCaseV2(
        pair_id=_text(value.get("pair_id"), "pair id", 200),
        task_id=_text(value.get("task_id"), "paired task id", 300),
        harness=_text(value.get("harness"), "paired harness", 100),
        attempt=_positive_int(value.get("attempt"), "paired attempt"),
        status=status,  # type: ignore[arg-type]
        dimension_changes=tuple(
            _dimension_change(item)
            for item in value.get("dimension_changes") or ()
        ),
        baseline=_paired_attempt(value.get("baseline")),
        candidate=_paired_attempt(value.get("candidate")),
        task_label=str(value.get("task_label") or "") or None,
        baseline_passed=(
            value.get("baseline_passed")
            if isinstance(value.get("baseline_passed"), bool)
            else None
        ),
        candidate_passed=(
            value.get("candidate_passed")
            if isinstance(value.get("candidate_passed"), bool)
            else None
        ),
        baseline_prediction_id=(
            str(value.get("baseline_prediction_id") or "") or None
        ),
        candidate_prediction_id=(
            str(value.get("candidate_prediction_id") or "") or None
        ),
        baseline_evaluation_call_id=(
            str(value.get("baseline_evaluation_call_id") or "") or None
        ),
        candidate_evaluation_call_id=(
            str(value.get("candidate_evaluation_call_id") or "") or None
        ),
    )


def _paired_case_v3(raw: Any) -> PairedCaseV3:
    value = _mapping(raw, "V3 paired case")
    allowed = {item.name for item in PairedCaseV3.__dataclass_fields__.values()}
    _reject_unknown(value, allowed, "V3 paired case")
    status = str(value.get("status") or "")
    if status not in {"improved", "regressed", "mixed", "unchanged", "incomplete"}:
        raise ValueError(f"unknown V3 paired case status: {status}")
    return PairedCaseV3(
        pair_id=_text(value.get("pair_id"), "V3 pair id", 200),
        task_id=_text(value.get("task_id"), "V3 paired task id", 300),
        harness=_text(value.get("harness"), "V3 paired harness", 100),
        attempt=_positive_int(value.get("attempt"), "V3 paired attempt"),
        status=status,  # type: ignore[arg-type]
        dimension_changes=tuple(
            _dimension_change_v3(item)
            for item in value.get("dimension_changes") or ()
        ),
        baseline=_paired_attempt_v3(value.get("baseline")),
        candidate=_paired_attempt_v3(value.get("candidate")),
        task_label=str(value.get("task_label") or "") or None,
    )


def _behavioral_summary_from_dict(raw: Any) -> BehavioralSummaryV1:
    value = _mapping(raw, "behavioral summary")
    allowed = {
        item.name for item in BehavioralSummaryV1.__dataclass_fields__.values()
    }
    _reject_unknown(value, allowed, "behavioral summary")
    status = str(value.get("status") or "")
    if status not in {
        "invalid",
        "incomplete",
        "improved",
        "regressed",
        "mixed",
        "unchanged",
    }:
        raise ValueError(f"unknown behavioral status: {status}")
    return BehavioralSummaryV1(
        status=status,  # type: ignore[arg-type]
        recommendation=_text(
            value.get("recommendation"), "behavioral recommendation", 1000
        ),
        improved_pairs=_non_negative_int(
            value.get("improved_pairs", 0), "improved pairs"
        ),
        regressed_pairs=_non_negative_int(
            value.get("regressed_pairs", 0), "regressed pairs"
        ),
        mixed_pairs=_non_negative_int(value.get("mixed_pairs", 0), "mixed pairs"),
        unchanged_pairs=_non_negative_int(
            value.get("unchanged_pairs", 0), "unchanged pairs"
        ),
        incomplete_pairs=_non_negative_int(
            value.get("incomplete_pairs", 0), "incomplete pairs"
        ),
        candidate_critical_failures=_non_negative_int(
            value.get("candidate_critical_failures", 0),
            "candidate critical failures",
        ),
        critical_blockers=tuple(
            str(item) for item in value.get("critical_blockers") or ()
        ),
        supported_claim=str(value.get("supported_claim") or "") or None,
        limitations=tuple(str(item) for item in value.get("limitations") or ()),
        next_action=_text(value.get("next_action"), "behavioral next action", 1000),
    )


def _candidate_source_revision(raw: Any) -> CandidateSourceRevisionV1:
    value = _mapping(raw, "candidate source revision")
    allowed = {
        item.name
        for item in CandidateSourceRevisionV1.__dataclass_fields__.values()
    }
    _reject_unknown(value, allowed, "candidate source revision")
    kind = validate_id(
        _text(value.get("kind"), "candidate source revision kind", 100),
        kind="candidate source revision kind",
    )
    source_id = validate_id(
        _text(value.get("id"), "candidate source revision id", 300),
        kind="candidate source revision id",
    )
    version_identity = _text(
        value.get("version_identity"),
        "candidate source version identity",
        500,
    )
    runtime_digest = _text(
        value.get("runtime_digest"),
        "candidate source runtime digest",
        200,
    )
    if not runtime_digest.startswith("sha256:"):
        raise ValueError("candidate source runtime digest must be sha256-qualified")
    lock_digest = str(value.get("lock_digest") or "") or None
    if lock_digest is not None and not lock_digest.startswith("sha256:"):
        raise ValueError("candidate source lock digest must be sha256-qualified")
    return CandidateSourceRevisionV1(
        kind=kind,
        id=source_id,
        version_identity=version_identity,
        runtime_digest=runtime_digest,
        lock_digest=lock_digest,
    )


def _candidate_source_revisions(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[CandidateSourceRevisionV1, ...]:
    candidate_rows = [
        row
        for row in rows
        if str(row.get("variant_id") or "") == "candidate"
    ]
    return _consistent_source_revisions(
        candidate_rows,
        required=False,
        label="candidate result",
    )


def _integration_change_required(comparison: Mapping[str, Any]) -> bool:
    return any(
        str(value) == "integrations"
        or str(value).startswith("integrations.")
        for value in comparison.get("changed") or ()
    )


def _skill_change_required(comparison: Mapping[str, Any]) -> bool:
    return any(
        str(value) == "skills" or str(value).startswith("skills.")
        for value in comparison.get("changed") or ()
    )


def _consistent_source_revisions(
    cells: Sequence[Mapping[str, Any]],
    *,
    required: bool,
    label: str,
) -> tuple[CandidateSourceRevisionV1, ...]:
    observed: list[tuple[CandidateSourceRevisionV1, ...]] = []
    for cell in cells:
        integration_revisions = _source_revisions_from_provenance(
            cell.get("integration_provenance"),
            required=required,
            label=label,
        )
        skill_revisions = _skill_source_revisions_from_provenance(
            cell.get("skill_provenance"),
            required=required,
            label=label,
        )
        observed.append(
            tuple(
                sorted(
                    (*integration_revisions, *skill_revisions),
                    key=lambda item: (
                        item.kind,
                        item.id,
                        item.version_identity,
                        item.runtime_digest,
                        item.lock_digest or "",
                    ),
                )
            )
        )
    if not observed:
        if required:
            raise ValueError(f"{label} has no source-revision-bearing cells")
        return ()
    first = observed[0]
    if any(value != first for value in observed[1:]):
        raise ValueError(f"{label} source revisions drift across attempts")
    if required and not first:
        raise ValueError(f"{label} has no candidate source revisions")
    return first


def _source_revisions_from_provenance(
    raw: Any,
    *,
    required: bool,
    label: str,
) -> tuple[CandidateSourceRevisionV1, ...]:
    provenance = raw or ()
    if isinstance(provenance, Mapping):
        provenance = (provenance,)
    if not isinstance(provenance, Sequence) or isinstance(
        provenance,
        str | bytes,
    ):
        raise ValueError(f"{label} integration provenance must be an array")
    revisions: dict[
        tuple[str, str, str, str, str | None],
        CandidateSourceRevisionV1,
    ] = {}
    missing: list[str] = []
    for raw_item in provenance:
        if not isinstance(raw_item, Mapping):
            raise ValueError(f"{label} integration provenance must contain objects")
        fields = ("kind", "id", "version_identity", "runtime_digest")
        if any(not str(raw_item.get(field) or "") for field in fields):
            if required:
                missing.append(str(raw_item.get("id") or "unknown"))
            continue
        revision = _candidate_source_revision(
            {
                field: raw_item[field]
                for field in (*fields, "lock_digest")
                if raw_item.get(field) is not None
            }
        )
        key = (
            revision.kind,
            revision.id,
            revision.version_identity,
            revision.runtime_digest,
            revision.lock_digest,
        )
        revisions[key] = revision
    if missing:
        raise ValueError(
            f"{label} integration provenance lacks immutable source revisions: "
            + ", ".join(sorted(missing))
        )
    return tuple(
        revisions[key]
        for key in sorted(
            revisions,
            key=lambda value: tuple(item or "" for item in value),
        )
    )


def _skill_source_revisions_from_provenance(
    raw: Any,
    *,
    required: bool,
    label: str,
) -> tuple[CandidateSourceRevisionV1, ...]:
    provenance = raw or ()
    if isinstance(provenance, Mapping):
        provenance = (provenance,)
    if not isinstance(provenance, Sequence) or isinstance(
        provenance,
        str | bytes,
    ):
        raise ValueError(f"{label} Skill provenance must be an array")
    revisions: dict[
        tuple[str, str, str, str, str | None],
        CandidateSourceRevisionV1,
    ] = {}
    missing: list[str] = []
    for raw_item in provenance:
        if not isinstance(raw_item, Mapping):
            raise ValueError(f"{label} Skill provenance must contain objects")
        source_id = str(raw_item.get("id") or "")
        commit = str(raw_item.get("resolved_commit") or "")
        digest = str(raw_item.get("digest") or "")
        if not source_id or not digest:
            if required:
                missing.append(source_id or "unknown")
            continue
        revision = _candidate_source_revision(
            {
                "kind": "skill",
                "id": source_id,
                "version_identity": (
                    f"git:{commit}" if commit else f"digest:{digest}"
                ),
                "runtime_digest": digest,
            }
        )
        key = (
            revision.kind,
            revision.id,
            revision.version_identity,
            revision.runtime_digest,
            revision.lock_digest,
        )
        revisions[key] = revision
    if missing:
        raise ValueError(
            f"{label} Skill provenance lacks immutable source revisions: "
            + ", ".join(sorted(missing))
        )
    return tuple(
        revisions[key]
        for key in sorted(
            revisions,
            key=lambda value: tuple(item or "" for item in value),
        )
    )


def _evaluate_decision(
    *,
    policy: DecisionPolicyV1 | None,
    rows: Sequence[Mapping[str, Any]],
    deterministic: Mapping[str, Any],
    operational: Mapping[str, Any],
    improved: int,
    regressed: int,
    incomplete: int,
    required_incomplete: int,
    integrity: Mapping[str, Any],
    attestation: DecisionAttestationV1 | None,
    release_note_coverage: Sequence[Mapping[str, Any]] = (),
) -> DecisionSummaryV1:
    evidence_grade = _evidence_grade(integrity, rows)
    if integrity.get("status") == "invalid":
        return DecisionSummaryV1(
            status="invalid",
            recommendation="INVALID — do not use this result for a release decision.",
            release_target=policy.release_target if policy else None,
            candidate_sha=policy.candidate_sha if policy else None,
            evidence_grade=evidence_grade,
            gates=(),
            critical_blockers=("result integrity is invalid",),
            limitations=(
                "Task and operational observations remain mechanism evidence only.",
            ),
            next_action="Repair result integrity and create a new Study identity.",
            human_signoff_required=(
                policy.human_signoff_required if policy else True
            ),
            attestation=attestation,
        )
    if policy is None:
        return DecisionSummaryV1(
            status="inconclusive",
            recommendation=(
                "Package release not evaluated by this Study."
            ),
            release_target=None,
            candidate_sha=None,
            evidence_grade=evidence_grade,
            gates=(),
            critical_blockers=("decision policy is unavailable",),
            limitations=(
                "Task and operational observations are evidence, not a release decision.",
            ),
            next_action=(
                "Use this Study only for its bounded behavioral verdict; run the "
                "separate governed release qualification for a package decision."
            ),
            human_signoff_required=True,
            attestation=None,
        )
    facts = _decision_facts(
        rows=rows,
        deterministic=deterministic,
        operational=operational,
        improved=improved,
        regressed=regressed,
        incomplete=incomplete,
        required_incomplete=required_incomplete,
        integrity=integrity,
        evidence_grade=evidence_grade,
    )
    policies = list(policy.gates)
    implicit = _implicit_decision_gate_policies()
    policies = _canonical_decision_gate_policies(
        policies,
        implicit=implicit,
        release_note_coverage=release_note_coverage,
    )
    gate_results: list[DecisionGateResultV1] = []
    for gate in policies:
        actual = facts.get(gate.source)
        gate_results.append(
            DecisionGateResultV1(
                id=gate.id,
                label=gate.label,
                category=gate.category,
                status=_gate_status(actual, gate.operator, gate.target),
                critical=gate.critical,
                actual=actual,
                target=gate.target,
            )
        )
    grade_passed = _grade_rank(evidence_grade) >= _grade_rank(
        policy.minimum_evidence_grade
    )
    gate_results.append(
        DecisionGateResultV1(
            id="evidence-grade",
            label="Minimum evidence grade",
            category="evidence",
            status="passed" if grade_passed else "failed",
            critical=True,
            actual=evidence_grade,
            target=policy.minimum_evidence_grade,
        )
    )
    blockers = tuple(
        gate.label
        for gate in gate_results
        if gate.critical and gate.status != "passed"
    )
    if any(gate.status == "unavailable" for gate in gate_results if gate.critical):
        status: DecisionStatus = "blocked"
        recommendation = "BLOCKED — required release evidence is unavailable."
        next_action = "Collect the missing required evidence and rerun qualification."
    elif blockers:
        status = "hold"
        recommendation = "HOLD — one or more critical release gates failed."
        next_action = "Address the critical blockers, then approve a new immutable preview."
    elif policy.human_signoff_required:
        status = "ready_for_signoff"
        recommendation = (
            "READY FOR SIGN-OFF — automated gates passed; this is not yet GO."
        )
        next_action = "Have the release owner sign the immutable result digest."
    else:
        status = "go"
        recommendation = "GO — all governed gates passed."
        next_action = "Proceed through the separate package release procedure."
    return DecisionSummaryV1(
        status=status,
        recommendation=recommendation,
        release_target=policy.release_target,
        candidate_sha=policy.candidate_sha,
        evidence_grade=evidence_grade,
        gates=tuple(gate_results),
        critical_blockers=blockers,
        limitations=(
            "The decision applies only to the exact candidate, runtime, taskset, scorers, and project in the approved preview.",
            "Managed-service and Helm release gates are not covered.",
        ),
        next_action=next_action,
        human_signoff_required=policy.human_signoff_required,
        attestation=attestation,
    )


def _implicit_decision_gate_policies() -> tuple[DecisionGatePolicyV1, ...]:
    return (
        DecisionGatePolicyV1(
            id="result-integrity",
            label="Result rows and summary reconcile",
            category="integrity",
            source="integrity.valid",
            operator="eq",
            target=True,
        ),
        DecisionGatePolicyV1(
            id="unique-attempts",
            label="Every terminal attempt identity is unique",
            category="integrity",
            source="attempts.duplicates",
            operator="eq",
            target=0,
        ),
        DecisionGatePolicyV1(
            id="critical-regressions",
            label="No critical task regression",
            category="task",
            source="task.critical_regressions",
            operator="eq",
            target=0,
        ),
        DecisionGatePolicyV1(
            id="required-evaluations",
            label="Required evaluations are complete",
            category="evidence",
            source="evidence.required_incomplete",
            operator="eq",
            target=0,
        ),
        DecisionGatePolicyV1(
            id="cross-project-scope",
            label="Every query stays in the locked project",
            category="privacy",
            source="evidence.cross_project_attempts",
            operator="eq",
            target=0,
        ),
        DecisionGatePolicyV1(
            id="infrastructure",
            label="Infrastructure conformance has no failures",
            category="infrastructure",
            source="infrastructure.failures",
            operator="eq",
            target=0,
        ),
        DecisionGatePolicyV1(
            id="credentials-and-private-labels",
            label="No credential or private-label leak",
            category="privacy",
            source="privacy.leaks",
            operator="eq",
            target=0,
        ),
        DecisionGatePolicyV1(
            id="local-artifact-privacy",
            label="Local run artifacts pass privacy scanning",
            category="privacy",
            source="privacy.local_artifacts_passed",
            operator="eq",
            target=True,
        ),
        DecisionGatePolicyV1(
            id="hosted-evidence-privacy",
            label="Hosted evidence passes post-publication privacy scanning",
            category="privacy",
            source="privacy.hosted_evidence_passed",
            operator="eq",
            target=True,
        ),
        DecisionGatePolicyV1(
            id="sandbox-cleanup",
            label="No orphaned sandbox",
            category="infrastructure",
            source="cleanup.orphans",
            operator="eq",
            target=0,
        ),
    )


def _canonical_decision_gate_policies(
    explicit: Sequence[DecisionGatePolicyV1],
    *,
    implicit: Sequence[DecisionGatePolicyV1],
    release_note_coverage: Sequence[Mapping[str, Any]],
) -> list[DecisionGatePolicyV1]:
    policies = list(explicit)
    ids = {gate.id: gate for gate in policies}
    sources = {gate.source: gate for gate in policies}
    if len(ids) != len(policies) or len(sources) != len(policies):
        raise ValueError(
            "decision policy gate ids and sources must each be unique"
        )

    for gate in implicit:
        existing = sources.get(gate.source)
        if existing is not None:
            if (
                existing.category != gate.category
                or existing.operator != gate.operator
                or existing.target != gate.target
                or existing.critical is not True
            ):
                raise ValueError(
                    f"implicit decision source {gate.source!r} cannot be "
                    "weakened or redefined"
                )
            continue
        if gate.id in ids:
            raise ValueError(
                f"decision gate id {gate.id!r} conflicts with another source"
            )
        policies.append(gate)
        ids[gate.id] = gate
        sources[gate.source] = gate

    release_notes_by_gate: dict[str, set[str]] = {}
    for item in release_note_coverage:
        if str(item.get("status") or "") == "not_applicable":
            continue
        release_note = str(item.get("release_note") or "")
        for gate_id in _string_tuple(
            item.get("infrastructure_gates") or [],
            "release-note infrastructure gate",
            allow_empty=True,
        ):
            release_notes_by_gate.setdefault(gate_id, set()).add(
                release_note
            )
    for gate_id, release_notes in sorted(release_notes_by_gate.items()):
        source = f"infrastructure.gate.{gate_id}"
        existing = sources.get(source)
        if existing is not None:
            if (
                existing.category != "infrastructure"
                or existing.operator != "eq"
                or existing.target is not True
                or existing.critical is not True
            ):
                raise ValueError(
                    f"release-note infrastructure source {source!r} must be "
                    "a critical infrastructure eq/true gate"
                )
            continue
        policy_id = f"release-note-infrastructure-{gate_id}"
        if policy_id in ids:
            raise ValueError(
                f"decision gate id {policy_id!r} conflicts with another source"
            )
        gate = DecisionGatePolicyV1(
            id=policy_id,
            label=(
                f"Release-note infrastructure gate {gate_id} passes "
                f"({', '.join(sorted(release_notes))})"
            ),
            category="infrastructure",
            source=source,
            operator="eq",
            target=True,
            critical=True,
        )
        policies.append(gate)
        ids[gate.id] = gate
        sources[gate.source] = gate
    return policies


def _decision_facts(
    *,
    rows: Sequence[Mapping[str, Any]],
    deterministic: Mapping[str, Any],
    operational: Mapping[str, Any],
    improved: int,
    regressed: int,
    incomplete: int,
    required_incomplete: int,
    integrity: Mapping[str, Any],
    evidence_grade: str,
) -> dict[str, str | float | int | bool | None]:
    candidate = _mapping_or_empty(deterministic.get("candidate"))
    critical_failures = _critical_dimension_failures(rows, "candidate")
    critical_regressions = _critical_regressions(rows)
    row_privacy_leaks = sum(
        int(bool(row.get("credential_leak") or row.get("private_label_leak")))
        for row in rows
    )
    privacy_leaks = (
        row_privacy_leaks
        + max(
            (
                int(row.get("local_artifact_privacy_match_count") or 0)
                for row in rows
            ),
            default=0,
        )
        + max(
            (
                int(row.get("hosted_evidence_privacy_match_count") or 0)
                for row in rows
            ),
            default=0,
        )
    )
    orphans = sum(
        int(
            row.get("sandbox_deleted") is False
            or row.get("orphaned_sandbox") is True
        )
        for row in rows
    )
    infrastructure_complete = bool(rows) and all(
        row.get("infrastructure_conformance_complete") is True for row in rows
    )
    privacy_evidence_available = bool(rows) and all(
        _privacy_scan_evidence_available(row) for row in rows
    )
    cleanup_complete = bool(rows) and all(
        row.get("sandbox_cleanup_verified") is True for row in rows
    )
    facts: dict[str, str | float | int | bool | None] = {
        "integrity.valid": integrity.get("status") == "reconciled",
        "attempts.duplicates": len(integrity.get("duplicate_attempt_ids") or ()),
        "matrix.rows": len(rows),
        "matrix.terminal_rows": sum(
            _terminal_execution_status(row) is not None for row in rows
        ),
        "matrix.aligned_pairs": improved + regressed + max(0, len(rows) // 2 - improved - regressed - incomplete),
        "task.candidate_passed": int(candidate.get("passed") or 0),
        "task.candidate_evaluated": int(candidate.get("evaluated") or 0),
        "task.candidate_critical_failures": critical_failures,
        "task.critical_regressions": critical_regressions,
        "task.incomplete_pairs": incomplete,
        "evidence.required_incomplete": required_incomplete,
        "evidence.unresolved_attempts": int(
            integrity.get("unresolved_evidence_attempts") or 0
        ),
        "evidence.cross_project_attempts": int(
            integrity.get("cross_project_attempts") or 0
        ),
        "integrity.source_project_writes": _source_project_write_count(
            rows
        ),
        "evidence.grade": evidence_grade,
        "infrastructure.failures": (
            int(operational.get("infrastructure_failures") or 0)
            if infrastructure_complete
            else None
        ),
        "privacy.leaks": (
            privacy_leaks if privacy_evidence_available else None
        ),
        "privacy.local_artifacts_passed": _privacy_component_fact(
            rows,
            "local_artifact_privacy_scan_status",
        ),
        "privacy.hosted_evidence_passed": _privacy_component_fact(
            rows,
            "hosted_evidence_privacy_scan_status",
        ),
        "cleanup.orphans": orphans if cleanup_complete else None,
    }
    facts.update(_efficiency_regressions(rows))
    facts.update(_locked_decision_facts(rows))
    facts.update(_infrastructure_gate_facts(rows))
    return facts


def _source_project_write_count(
    rows: Sequence[Mapping[str, Any]],
) -> int | None:
    if not rows:
        return None
    checks: list[Mapping[str, Any]] = []
    for row in rows:
        for field_name in (
            "source_pre_run_drift",
            "source_checkpoint_drift",
            "source_post_run_drift",
        ):
            value = row.get(field_name)
            if not isinstance(value, Mapping):
                return None
            checks.append(value)
    statuses = {str(item.get("status") or "") for item in checks}
    if "drifted" in statuses:
        return 1
    if statuses == {"matched"}:
        expected = {
            str(item.get("expected_digest") or "") for item in checks
        }
        observed = {
            str(item.get("observed_digest") or "") for item in checks
        }
        if len(expected) == 1 and observed == expected:
            return 0
    return None


def _privacy_component_fact(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> bool | None:
    if not rows:
        return None
    statuses = [_privacy_scan_status(row, field) for row in rows]
    if any(status == "failed" for status in statuses):
        return False
    if all(status == "passed" for status in statuses):
        return True
    return None


def _infrastructure_gate_facts(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, bool | None]:
    if not rows:
        return {}
    digests = {
        str(row.get("infrastructure_receipt_digest") or "") for row in rows
    }
    if len(digests) != 1 or not next(iter(digests)):
        return {}
    mappings = [
        _mapping_or_empty(row.get("infrastructure_gate_statuses"))
        for row in rows
    ]
    gate_ids = (
        set.intersection(*(set(value) for value in mappings))
        if mappings
        else set()
    )
    facts: dict[str, bool | None] = {}
    for gate_id in sorted(gate_ids):
        statuses = {str(value.get(gate_id) or "") for value in mappings}
        if len(statuses) != 1:
            continue
        status = next(iter(statuses))
        facts[f"infrastructure.gate.{gate_id}"] = (
            True
            if status == "passed"
            else False
            if status == "failed"
            else None
        )
    return facts


def _critical_dimension_failures(
    rows: Sequence[Mapping[str, Any]], variant: str
) -> int:
    failures = 0
    for row in rows:
        if str(row.get("variant_id") or "") != variant:
            continue
        critical = row.get("comparison_deterministic_criticality") or {}
        scores = _mapping_or_empty(row.get("comparison_deterministic_scores"))
        if isinstance(critical, Mapping):
            failures += sum(
                not _dimension_passed(scores.get(str(name)))
                for name, required in critical.items()
                if required is True
            )
    return failures


def _critical_dimension_blockers(
    rows: Sequence[Mapping[str, Any]], variant: str
) -> tuple[str, ...]:
    blockers: list[str] = []
    for row in rows:
        if str(row.get("variant_id") or "") != variant:
            continue
        task_id = str(row.get("task_id") or row.get("task_name") or "unknown-task")
        critical = row.get("comparison_deterministic_criticality") or {}
        scores = _mapping_or_empty(row.get("comparison_deterministic_scores"))
        if not isinstance(critical, Mapping):
            continue
        for raw_name, required in critical.items():
            name = str(raw_name)
            if required is True and not _dimension_passed(scores.get(name)):
                blockers.append(f"{task_id}: {name} failed for the {variant}")
    return tuple(dict.fromkeys(blockers))


def _critical_regressions(rows: Sequence[Mapping[str, Any]]) -> int:
    pairs: dict[tuple[str, str, int], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        variant = str(row.get("variant_id") or "")
        if variant not in {"baseline", "candidate"}:
            continue
        key = (
            str(row.get("task_id") or row.get("task_name") or ""),
            str(row.get("harness") or ""),
            int(row.get("trial_index") or 1),
        )
        pairs.setdefault(key, {})[variant] = row
    regressions = 0
    for pair in pairs.values():
        baseline = pair.get("baseline")
        candidate = pair.get("candidate")
        if baseline is None or candidate is None:
            continue
        critical = candidate.get("comparison_deterministic_criticality") or {}
        baseline_scores = _mapping_or_empty(
            baseline.get("comparison_deterministic_scores")
        )
        candidate_scores = _mapping_or_empty(
            candidate.get("comparison_deterministic_scores")
        )
        if isinstance(critical, Mapping):
            regressions += sum(
                _dimension_passed(baseline_scores.get(str(name)))
                and not _dimension_passed(candidate_scores.get(str(name)))
                for name, required in critical.items()
                if required is True
            )
    return regressions


def _behavioral_summary(
    *,
    rows: Sequence[Mapping[str, Any]],
    integrity: Mapping[str, Any],
    improved: int,
    regressed: int,
    mixed: int,
    unchanged: int,
    incomplete: int,
    required_incomplete: int,
    unresolved_evidence: int,
    cross_project_attempts: int,
) -> BehavioralSummaryV1:
    candidate_critical_failures = _critical_dimension_failures(rows, "candidate")
    candidate_critical_blockers = _critical_dimension_blockers(rows, "candidate")
    local_conformance_failed = bool(
        integrity.get("harbor_conformance_failed_attempts")
    )
    local_conformance_unavailable = bool(
        integrity.get("harbor_conformance_unavailable_attempts")
    )
    common_limitations = (
        "This is behavioral evidence for the exact locked candidates, taskset, model, harness, attempts, and execution fingerprint.",
        "A local Harbor result does not qualify W&B Serverless or authorize a package release.",
    )
    if (
        integrity.get("status") == "invalid"
        or cross_project_attempts
        or local_conformance_failed
    ):
        blockers = (
            ("result integrity is invalid",)
            if integrity.get("status") == "invalid"
            else ()
        ) + (
            ("one or more attempts queried outside the locked evidence project",)
            if cross_project_attempts
            else ()
        ) + (
            ("local Harbor privacy, policy, or cleanup conformance failed",)
            if local_conformance_failed
            else ()
        )
        return BehavioralSummaryV1(
            status="invalid",
            recommendation="INVALID — do not use this result as behavioral evidence.",
            improved_pairs=improved,
            regressed_pairs=regressed,
            mixed_pairs=mixed,
            unchanged_pairs=unchanged,
            incomplete_pairs=incomplete,
            candidate_critical_failures=candidate_critical_failures,
            critical_blockers=blockers,
            supported_claim=None,
            limitations=common_limitations,
            next_action="Repair result integrity and create a new Study identity.",
        )
    if (
        incomplete
        or required_incomplete
        or unresolved_evidence
        or local_conformance_unavailable
    ):
        blockers = tuple(
            item
            for item in (
                (
                    f"{incomplete} aligned pair(s) are incomplete"
                    if incomplete
                    else ""
                ),
                (
                    f"{required_incomplete} required evaluation(s) are incomplete"
                    if required_incomplete
                    else ""
                ),
                (
                    f"{unresolved_evidence} attempt(s) have unresolved evidence links"
                    if unresolved_evidence
                    else ""
                ),
                (
                    "local Harbor privacy, policy, or cleanup evidence is unavailable"
                    if local_conformance_unavailable
                    else ""
                ),
            )
            if item
        )
        return BehavioralSummaryV1(
            status="incomplete",
            recommendation="INCOMPLETE — required behavioral evidence is unavailable.",
            improved_pairs=improved,
            regressed_pairs=regressed,
            mixed_pairs=mixed,
            unchanged_pairs=unchanged,
            incomplete_pairs=incomplete,
            candidate_critical_failures=candidate_critical_failures,
            critical_blockers=blockers,
            supported_claim=None,
            limitations=common_limitations,
            next_action="Resolve the missing attempts or evidence links and rerun.",
        )
    if mixed or (improved and regressed):
        return BehavioralSummaryV1(
            status="mixed",
            recommendation="MIXED — outcome improvements and regressions coexist.",
            improved_pairs=improved,
            regressed_pairs=regressed,
            mixed_pairs=mixed,
            unchanged_pairs=unchanged,
            incomplete_pairs=incomplete,
            candidate_critical_failures=candidate_critical_failures,
            critical_blockers=candidate_critical_blockers,
            supported_claim=None,
            limitations=common_limitations,
            next_action="Inspect the discordant dimensions before selecting the candidate.",
        )
    if regressed:
        return BehavioralSummaryV1(
            status="regressed",
            recommendation="REGRESSED — the candidate is worse on the locked comparison.",
            improved_pairs=improved,
            regressed_pairs=regressed,
            mixed_pairs=mixed,
            unchanged_pairs=unchanged,
            incomplete_pairs=incomplete,
            candidate_critical_failures=candidate_critical_failures,
            critical_blockers=("one or more aligned pairs regressed",),
            supported_claim=(
                f"The candidate regressed on {regressed} aligned pair(s) "
                "under the locked execution."
            ),
            limitations=common_limitations,
            next_action="Do not promote this candidate; inspect the regressed attempts.",
        )
    if improved and not candidate_critical_failures:
        return BehavioralSummaryV1(
            status="improved",
            recommendation="IMPROVED — the candidate is better on the locked comparison.",
            improved_pairs=improved,
            regressed_pairs=regressed,
            mixed_pairs=mixed,
            unchanged_pairs=unchanged,
            incomplete_pairs=incomplete,
            candidate_critical_failures=candidate_critical_failures,
            critical_blockers=(),
            supported_claim=(
                f"The candidate improved {improved} aligned pair(s), regressed "
                "none, and passed every candidate critical dimension under "
                "the locked execution."
            ),
            limitations=common_limitations,
            next_action="Use a separately approved confirmation cohort before promotion.",
        )
    blocked = bool(candidate_critical_blockers)
    return BehavioralSummaryV1(
        status="unchanged",
        recommendation=(
            "UNCHANGED — no release-qualifying behavioral improvement was "
            "established."
            if blocked
            else "UNCHANGED — no behavioral difference was detected."
        ),
        improved_pairs=improved,
        regressed_pairs=regressed,
        mixed_pairs=mixed,
        unchanged_pairs=unchanged,
        incomplete_pairs=incomplete,
        candidate_critical_failures=candidate_critical_failures,
        critical_blockers=candidate_critical_blockers,
        supported_claim=(
            None
            if blocked
            else f"No behavioral difference was detected across {unchanged} aligned pair(s)."
        ),
        limitations=common_limitations,
        next_action=(
            "Repair the named candidate blockers before promotion."
            if blocked
            else "Use harder pre-frozen tasks if the decision still matters."
        ),
    )


def _efficiency_regressions(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for label, extractor in (
        ("cost", lambda row: _observed_attempt_cost(row)[0]),
        ("latency", _row_latency_sec),
        ("tool_calls", lambda row: float(_observed_tool_activity(row)[1])),
    ):
        by_variant = {
            variant: [
                value
                for row in rows
                if str(row.get("variant_id") or "") == variant
                and (value := extractor(row)) is not None
            ]
            for variant in ("baseline", "candidate")
        }
        baseline = by_variant["baseline"]
        candidate = by_variant["candidate"]
        if not baseline or not candidate or sum(baseline) == 0:
            result[f"efficiency.{label}_regression_pct"] = None
        else:
            result[f"efficiency.{label}_regression_pct"] = round(
                ((sum(candidate) / len(candidate)) / (sum(baseline) / len(baseline)) - 1)
                * 100,
                6,
            )
        paired_regressions: list[float] = []
        pairs: dict[tuple[str, str, int], dict[str, float]] = {}
        for row in rows:
            variant = str(row.get("variant_id") or "")
            value = extractor(row)
            if variant not in {"baseline", "candidate"} or value is None:
                continue
            key = (
                str(row.get("task_id") or row.get("task_name") or ""),
                str(row.get("harness") or ""),
                int(row.get("trial_index") or 1),
            )
            pairs.setdefault(key, {})[variant] = value
        for pair in pairs.values():
            baseline_value = pair.get("baseline")
            candidate_value = pair.get("candidate")
            if baseline_value and candidate_value is not None:
                paired_regressions.append(
                    (candidate_value / baseline_value - 1) * 100
                )
        result[f"efficiency.max_task_{label}_regression_pct"] = (
            round(max(paired_regressions), 6) if paired_regressions else None
        )
    return result


def _locked_decision_facts(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, str | float | int | bool | None]:
    values: dict[str, set[str]] = {}
    decoded: dict[tuple[str, str], str | float | int | bool | None] = {}
    for row in rows:
        facts = row.get("decision_facts")
        if not isinstance(facts, Mapping):
            continue
        for key, value in facts.items():
            name = str(key)
            if not name.startswith(
                ("infrastructure.", "privacy.", "cleanup.", "evidence.")
            ):
                continue
            if value is not None and not isinstance(value, str | int | float | bool):
                continue
            encoded = json.dumps(value, sort_keys=True)
            values.setdefault(name, set()).add(encoded)
            decoded[(name, encoded)] = value
    return {
        name: (
            decoded[(name, next(iter(observed)))]
            if len(observed) == 1
            else None
        )
        for name, observed in values.items()
    }


def _evidence_grade(
    integrity: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> Literal["A", "B", "C", "invalid"]:
    if integrity.get("status") == "invalid":
        return "invalid"
    if int(integrity.get("cross_project_attempts") or 0):
        return "C"
    live_rows = [row for row in rows if str(row.get("trace_project") or "")]
    missing = int(integrity.get("unresolved_evidence_attempts") or 0)
    if live_rows and missing == 0:
        return "A"
    if missing == 0:
        return "B"
    return "C"


def _grade_rank(value: str) -> int:
    return {"invalid": 0, "C": 1, "B": 2, "A": 3}.get(value, 0)


def _gate_status(
    actual: str | float | int | bool | None,
    operator: str,
    target: str | float | int | bool,
) -> Literal["passed", "failed", "unavailable"]:
    if actual is None:
        return "unavailable"
    if operator == "eq":
        passed = actual == target
    elif (
        operator in {"lte", "gte"}
        and isinstance(actual, int | float)
        and not isinstance(actual, bool)
        and isinstance(target, int | float)
        and not isinstance(target, bool)
    ):
        passed = actual <= target if operator == "lte" else actual >= target
    else:
        return "unavailable"
    return "passed" if passed else "failed"


def _apply_decision_attestation(
    decision: DecisionSummaryV1,
    attestation: DecisionAttestationV1 | None,
    *,
    qualification_digest: str,
    require_actionability_review: bool,
) -> DecisionSummaryV1:
    if attestation is None:
        return replace(decision, attestation=None)
    if (
        decision.status != "ready_for_signoff"
        or attestation.signed_result_digest != qualification_digest
        or (
            require_actionability_review
            and attestation.review_status != "accepted_actionable"
        )
    ):
        return replace(
            decision,
            status="invalid",
            recommendation="INVALID — the release attestation does not match the qualified result.",
            critical_blockers=tuple(
                dict.fromkeys(
                    (*decision.critical_blockers, "release attestation mismatch")
                )
            ),
            next_action=(
                "Review the exact maintainer memos and sign the immutable "
                "qualification digest with accepted_actionable status."
            ),
            attestation=attestation,
        )
    return replace(
        decision,
        status="go",
        recommendation="GO — all governed gates passed and the release owner signed.",
        next_action="Proceed through the separate package release procedure.",
        attestation=attestation,
    )


def _comparison_result_digest(raw: Mapping[str, Any]) -> str:
    """Digest the complete post-attestation result envelope."""

    value = _json_value(dict(raw))
    value.pop("result_digest", None)
    return stable_digest(value)


def _comparison_qualification_digest(raw: Mapping[str, Any]) -> str:
    """Digest the immutable qualification state a release owner signs."""

    value = _json_value(dict(raw))
    value.pop("result_digest", None)
    value.pop("qualification_digest", None)
    decision = value.get("decision")
    if isinstance(decision, dict):
        decision.pop("attestation", None)
        if decision.get("status") == "go":
            decision["status"] = "ready_for_signoff"
            decision["recommendation"] = (
                "READY FOR SIGN-OFF — automated gates passed; this is not yet GO."
            )
            decision["next_action"] = (
                "Have the release owner sign the immutable result digest."
            )
    return stable_digest(value)


def _legacy_comparison_result_digest(raw: Mapping[str, Any]) -> str:
    """Read pre-envelope V1/V2 results without trusting a legacy GO."""

    return _comparison_qualification_digest(raw)


def _verify_v2_result_integrity(
    result: ComparisonResultV2 | ComparisonResultV3,
    *,
    has_qualification_digest: bool,
) -> None:
    if isinstance(result, ComparisonResultV3):
        _verify_v3_result_shape(result)
    attestation = result.decision.attestation
    if not has_qualification_digest:
        if result.decision.status == "go" or attestation is not None:
            raise ValueError(
                "legacy ComparisonResultV2 cannot carry a trusted release attestation"
            )
        if result.result_digest != _legacy_comparison_result_digest(
            result.to_dict()
        ):
            raise ValueError("comparison result digest does not match")
        return
    if not re.fullmatch(r"[0-9a-f]{64}", result.qualification_digest):
        raise ValueError("comparison qualification digest is invalid")
    expected_qualification = _comparison_qualification_digest(result.to_dict())
    if result.qualification_digest != expected_qualification:
        raise ValueError("comparison qualification digest does not match")
    if attestation is None:
        if result.decision.status == "go":
            raise ValueError("GO comparison result is missing release attestation")
        if result.result_digest != result.qualification_digest:
            raise ValueError(
                "unsigned comparison result must expose its qualification digest"
            )
        return
    expected_envelope = _comparison_result_digest(result.to_dict())
    if result.result_digest != expected_envelope:
        raise ValueError("attested comparison result envelope digest does not match")
    if result.decision.status == "go":
        if not re.fullmatch(r"[0-9a-f]{64}", attestation.signed_result_digest):
            raise ValueError("GO release attestation digest is invalid")
        if attestation.signed_result_digest != result.qualification_digest:
            raise ValueError(
                "GO release attestation does not sign the qualification digest"
            )


def _v3_canonical_attempt_rows(  # noqa: C901 - one bounded audit checks every V3 edge
    result: ComparisonResultV3,
) -> list[dict[str, Any]]:
    """Rebuild the safe decision-bearing row surface from canonical V3 pairs."""

    rows: list[dict[str, Any]] = []
    project = result.evidence_topology.result_destination.project_slug
    source_project = result.evidence_topology.source_destination.project_slug
    app_base_url = result.evidence_topology.result_destination.app_base_url
    aligned_by_id = {
        item.alignment_id: item
        for item in result.aligned_analysis.aligned_attempts
    }
    if len(aligned_by_id) != len(result.aligned_analysis.aligned_attempts):
        raise ValueError("ComparisonResultV3 aligned attempt IDs must be unique")
    aligned_dimensions: dict[str, AlignedDimensionV1] = {}
    for contrast in result.aligned_analysis.contrasts:
        for dimension in contrast.dimensions:
            previous = aligned_dimensions.setdefault(dimension.id, dimension)
            if previous != dimension:
                raise ValueError(
                    "ComparisonResultV3 aligned dimension contracts disagree"
                )

    for pair in result.paired_cases:
        expected_pair_id = stable_digest(
            {
                "schema_version": 1,
                "task_id": pair.task_id,
                "harness": pair.harness,
                "attempt": pair.attempt,
            }
        )
        if pair.pair_id != expected_pair_id:
            raise ValueError(
                "ComparisonResultV3 pair identity disagrees with its coordinates"
            )
        complete = pair.baseline is not None and pair.candidate is not None
        aligned = aligned_by_id.get(pair.pair_id)
        if complete:
            if (
                aligned is None
                or aligned.task_id != pair.task_id
                or aligned.harness != pair.harness
                or aligned.attempt != pair.attempt
            ):
                raise ValueError(
                    "ComparisonResultV3 aligned attempt coordinates disagree "
                    "with their pair"
                )
        elif aligned is not None:
            raise ValueError(
                "ComparisonResultV3 incomplete pair cannot claim aligned attempts"
            )

        score_dimensions: set[str] = set()
        if complete:
            assert pair.baseline is not None and pair.candidate is not None
            score_dimensions = {
                dimension
                for dimension in (
                    set(pair.baseline.scores) | set(pair.candidate.scores)
                )
                if not dimension.startswith("comparison.judge.")
            }
            if not score_dimensions:
                raise ValueError(
                    "ComparisonResultV3 complete pairs require deterministic scores"
                )
            if {item.id for item in pair.dimension_changes} != score_dimensions:
                raise ValueError(
                    "ComparisonResultV3 dimension changes do not cover the "
                    "canonical attempt scores"
                )
        for change in pair.dimension_changes:
            aligned_dimension = aligned_dimensions.get(change.id)
            if (
                aligned_dimension is None
                or aligned_dimension.label != change.label
                or aligned_dimension.role != change.role
                or aligned_dimension.critical != change.critical
            ):
                raise ValueError(
                    "ComparisonResultV3 paired dimension contract disagrees "
                    "with aligned analysis"
                )
            if pair.baseline is not None and change.baseline != _bool_score(
                pair.baseline.scores.get(change.id)
            ):
                raise ValueError(
                    "ComparisonResultV3 baseline dimension value disagrees "
                    "with its attempt score"
                )
            if pair.candidate is not None and change.candidate != _bool_score(
                pair.candidate.scores.get(change.id)
            ):
                raise ValueError(
                    "ComparisonResultV3 candidate dimension value disagrees "
                    "with its attempt score"
                )

        for arm, attempt in (
            ("baseline", pair.baseline),
            ("candidate", pair.candidate),
        ):
            if attempt is None:
                continue
            raw_identity = dict(attempt.identity)
            if set(raw_identity) != {
                "task_id",
                "arm",
                "harness",
                "attempt",
                "candidate",
                "runtime",
            }:
                raise ValueError(
                    "ComparisonResultV3 attempt identity is not canonical"
                )
            try:
                canonical_identity = attempt_identity(
                    task_id=str(raw_identity["task_id"]),
                    arm=str(raw_identity["arm"]),
                    harness=str(raw_identity["harness"]),
                    attempt=(
                        raw_identity["attempt"]
                        if type(raw_identity["attempt"]) is int
                        else 0
                    ),
                    candidate=str(raw_identity["candidate"]),
                    runtime=str(raw_identity["runtime"]),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "ComparisonResultV3 attempt identity is invalid"
                ) from exc
            if (
                raw_identity != canonical_identity
                or canonical_identity["task_id"] != pair.task_id
                or canonical_identity["arm"] != arm
                or canonical_identity["harness"] != pair.harness
                or canonical_identity["attempt"] != pair.attempt
                or attempt.attempt_id != attempt_id(**canonical_identity)
            ):
                raise ValueError(
                    "ComparisonResultV3 attempt identity disagrees with its "
                    "pair coordinates"
                )
            locked_runtime = (
                attempt.execution_fingerprint
                or attempt.runtime_lock_digest
            )
            if (
                locked_runtime is not None
                and canonical_identity["runtime"] != locked_runtime
            ):
                raise ValueError(
                    "ComparisonResultV3 attempt runtime identity disagrees "
                    "with its execution lock"
                )
            if complete and (
                aligned is None
                or aligned.attempt_ids_by_arm.get(arm) != attempt.attempt_id
            ):
                raise ValueError(
                    "ComparisonResultV3 aligned attempt identity disagrees "
                    "with its paired attempt"
                )

            links_by_kind = {item.kind: item for item in attempt.evidence_links}
            expected_link_kinds = {
                "evaluation_root",
                "prediction_and_score",
                "prediction",
                "agent_root",
                "dataset",
            }
            if (
                len(attempt.evidence_links) != 5
                or set(links_by_kind) != expected_link_kinds
            ):
                raise ValueError(
                    "ComparisonResultV3 attempts require exactly five unique "
                    "evidence links"
                )
            link_statuses = {item.status for item in attempt.evidence_links}
            expected_evidence_status = (
                "reconciled"
                if link_statuses == {"resolved"}
                else "invalid"
                if "invalid" in link_statuses
                else "missing"
            )
            if attempt.evidence_status != expected_evidence_status:
                raise ValueError(
                    "ComparisonResultV3 attempt evidence status disagrees "
                    "with its five-link chain"
                )

            resolved_call_ids: dict[str, str] = {}
            for kind, link in links_by_kind.items():
                if link.status != "resolved":
                    continue
                assert link.ref is not None and link.url is not None
                if kind == "dataset":
                    expected_url = _weave_object_url_from_ref(
                        project,
                        link.ref,
                        app_base_url=app_base_url,
                    )
                    if not expected_url or link.url != expected_url:
                        raise ValueError(
                            "ComparisonResultV3 Dataset link disagrees with "
                            "the result topology"
                        )
                    continue
                prefix = f"weave:///{project}/call/"
                if not link.ref.startswith(prefix):
                    raise ValueError(
                        "ComparisonResultV3 Call ref disagrees with the result "
                        "topology"
                    )
                call_id = link.ref.removeprefix(prefix)
                if (
                    not call_id
                    or link.url
                    != _weave_call_url(
                        project,
                        call_id,
                        app_base_url=app_base_url,
                    )
                ):
                    raise ValueError(
                        "ComparisonResultV3 Call link disagrees with its "
                        "stable Weave ref"
                    )
                resolved_call_ids[kind] = call_id
            agent_call_id = resolved_call_ids.get("agent_root")
            if (
                agent_call_id is not None
                and attempt.weave_agent_root_call_id != agent_call_id
            ):
                raise ValueError(
                    "ComparisonResultV3 Agent root ID disagrees with its "
                    "verified Weave link"
                )
            if attempt.otel_root_span_id and attempt.otel_root_span_id in {
                *resolved_call_ids.values(),
                *(item.ref or "" for item in attempt.evidence_links),
            }:
                raise ValueError(
                    "ComparisonResultV3 OTel span ID cannot be a Weave Call ID"
                )
            if attempt.actual_query_scope != attempt.queried_projects:
                raise ValueError(
                    "ComparisonResultV3 actual query scope disagrees with "
                    "normalized queried projects"
                )
            if set(attempt.score_explanations) != set(attempt.scores):
                raise ValueError(
                    "ComparisonResultV3 score explanations must cover every "
                    "deterministic score"
                )
            for judge_id, review in attempt.judge_reviews.items():
                if review.cost_status == "observed":
                    if (
                        review.observed_cost_usd is None
                        or review.observed_cost_usd < 0
                    ):
                        raise ValueError(
                            "ComparisonResultV3 observed judge cost is invalid"
                        )
                elif review.observed_cost_usd is not None:
                    raise ValueError(
                        "ComparisonResultV3 unavailable judge cost cannot carry "
                        "an observed value"
                    )
                if (
                    review.accounted_reserve_usd is not None
                    and review.accounted_reserve_usd < 0
                ):
                    raise ValueError(
                        "ComparisonResultV3 judge reserve must be non-negative"
                    )
                prefix = f"comparison.judge.{judge_id}."
                judge_scores = [
                    float(value)
                    for dimension, value in attempt.scores.items()
                    if dimension.startswith(prefix)
                    and isinstance(value, int | float)
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                ]
                if review.missing_evidence:
                    if judge_scores or review.label != "unusable":
                        raise ValueError(
                            "ComparisonResultV3 missing-evidence judge review "
                            "must be unusable and excluded from numeric summaries"
                        )
                    continue
                if not judge_scores:
                    raise ValueError(
                        "ComparisonResultV3 judge review lacks its locked "
                        "numeric judge dimensions"
                    )
                if review.label != _judge_review_label(
                    sum(judge_scores) / len(judge_scores)
                ):
                    raise ValueError(
                        "ComparisonResultV3 judge review label disagrees with "
                        "its locked numeric dimensions"
                    )

            infrastructure = dict(attempt.infrastructure)
            backend = str(infrastructure.get("backend") or "")
            row = {
                "variant_id": arm,
                "task_id": pair.task_id,
                "harness": pair.harness,
                "trial_index": pair.attempt,
                "attempt_id": attempt.attempt_id,
                "attempt_identity": canonical_identity,
                "pass": attempt.passed,
                "status": attempt.execution_status,
                "comparison_evaluation_status": attempt.evaluation_status,
                "comparison_required_evaluation_complete": (
                    attempt.evaluation_status == "completed"
                ),
                "comparison_deterministic_scores": dict(attempt.scores),
                "comparison_deterministic_criticality": {
                    item.id: item.critical for item in pair.dimension_changes
                },
                "trace_project": project,
                "queried_projects": list(attempt.queried_projects),
                "cost_usd": attempt.cost_usd,
                "latency_sec": attempt.latency_sec,
                "tool_call_count": attempt.tool_calls,
                "execution_fingerprint": attempt.execution_fingerprint,
                "runtime_lock_digest": attempt.runtime_lock_digest,
                "infrastructure_conformance_complete": infrastructure.get(
                    "infrastructure_conformance_complete"
                ),
                "infrastructure_receipt_digest": infrastructure.get(
                    "infrastructure_receipt_digest"
                ),
                "infrastructure_gate_statuses": dict(
                    _mapping_or_empty(
                        infrastructure.get("infrastructure_gate_statuses")
                    )
                ),
                "decision_facts": dict(
                    _mapping_or_empty(infrastructure.get("decision_facts"))
                ),
                "privacy_contract_version": infrastructure.get(
                    "privacy_contract_version"
                ),
                "local_artifact_privacy_scan_status": infrastructure.get(
                    "local_artifact_privacy_scan_status"
                ),
                "local_artifact_privacy_match_count": infrastructure.get(
                    "local_artifact_privacy_match_count"
                ),
                "hosted_evidence_privacy_scan_status": infrastructure.get(
                    "hosted_evidence_privacy_scan_status"
                ),
                "hosted_evidence_privacy_match_count": infrastructure.get(
                    "hosted_evidence_privacy_match_count"
                ),
                "credential_leak": infrastructure.get("credential_leak"),
                "private_label_leak": infrastructure.get("private_label_leak"),
                "private_label_boundary_verified": infrastructure.get(
                    "private_label_boundary_verified"
                ),
                "sandbox_cleanup_verified": infrastructure.get(
                    "cleanup_verified"
                ),
                "sandbox_deleted": infrastructure.get("cleanup_verified"),
                "orphaned_sandbox": infrastructure.get("orphaned"),
                "harbor_environment": backend,
                "harbor_config": (
                    {"environment": "docker"}
                    if backend.startswith(("local_harbor", "harbor-docker"))
                    else None
                ),
                "harbor_conformance_status": infrastructure.get(
                    "conformance_status"
                ),
                "harbor_policy_attestation_verified": infrastructure.get(
                    "policy_attestation_verified"
                ),
                "source_pre_run_drift": (
                    result.evidence_topology.pre_run_drift.to_dict()
                ),
                "source_post_run_drift": (
                    result.evidence_topology.post_run_drift.to_dict()
                ),
            }
            rows.append(row)

    if set(aligned_by_id) != {
        pair.pair_id
        for pair in result.paired_cases
        if pair.baseline is not None and pair.candidate is not None
    }:
        raise ValueError(
            "ComparisonResultV3 aligned attempts do not match complete pairs"
        )
    if any(
        project_id != source_project
        for row in rows
        for project_id in _queried_projects(row)
    ):
        # The result remains readable as invalid audit evidence. The exact
        # count is reconciled below and suppresses behavioral/release claims.
        pass
    return rows


def _v3_semantic_integrity(
    result: ComparisonResultV3,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    attempt_ids = [str(row["attempt_id"]) for row in rows]
    duplicate_attempt_ids = sorted(
        value for value, count in Counter(attempt_ids).items() if count > 1
    )
    evidence_statuses = [
        str(
            next(
                attempt.evidence_status
                for pair in result.paired_cases
                for attempt in (pair.baseline, pair.candidate)
                if attempt is not None
                and attempt.attempt_id == row["attempt_id"]
            )
        )
        for row in rows
    ]
    cross_project_attempts = sum(
        bool(
            _cross_project_queries(
                row,
                result.evidence_topology.source_destination.project_slug,
            )
        )
        for row in rows
    )
    harbor_failed = sum(_local_harbor_conformance_failed(row) for row in rows)
    harbor_unavailable = sum(
        _local_harbor_conformance_unavailable(row) for row in rows
    )
    local_privacy_failed = sum(
        _privacy_scan_status(row, "local_artifact_privacy_scan_status")
        == "failed"
        for row in rows
    )
    hosted_privacy_failed = sum(
        _privacy_scan_status(row, "hosted_evidence_privacy_scan_status")
        == "failed"
        for row in rows
    )
    local_privacy_unavailable = sum(
        _privacy_scan_status(row, "local_artifact_privacy_scan_status")
        in {"legacy", "unavailable", "not_applicable"}
        for row in rows
    )
    hosted_privacy_unavailable = sum(
        _privacy_scan_status(row, "hosted_evidence_privacy_scan_status")
        in {"legacy", "unavailable", "not_applicable"}
        for row in rows
    )
    invalid_evidence = sum(status == "invalid" for status in evidence_statuses)
    return {
        "status": (
            "invalid"
            if (
                duplicate_attempt_ids
                or invalid_evidence
                or cross_project_attempts
                or harbor_failed
                or local_privacy_failed
                or hosted_privacy_failed
            )
            else "reconciled"
        ),
        "row_count": len(rows),
        "unique_attempts": len(set(attempt_ids)),
        "duplicate_attempt_ids": duplicate_attempt_ids,
        "unresolved_evidence_attempts": sum(
            status in {"missing", "invalid"} for status in evidence_statuses
        ),
        "invalid_evidence_attempts": invalid_evidence,
        "cross_project_attempts": cross_project_attempts,
        "harbor_conformance_failed_attempts": harbor_failed,
        "harbor_conformance_unavailable_attempts": harbor_unavailable,
        "local_artifact_privacy_failed_attempts": local_privacy_failed,
        "hosted_evidence_privacy_failed_attempts": hosted_privacy_failed,
        "local_artifact_privacy_unavailable_attempts": (
            local_privacy_unavailable
        ),
        "hosted_evidence_privacy_unavailable_attempts": (
            hosted_privacy_unavailable
        ),
        "privacy_complete_attempts": sum(
            _privacy_scans_complete(row) for row in rows
        ),
    }


def _v3_decision_facts(
    result: ComparisonResultV3,
    rows: Sequence[Mapping[str, Any]],
    *,
    integrity: Mapping[str, Any],
) -> dict[str, str | float | int | bool | None]:
    grade = _evidence_grade(integrity, rows)
    facts = _decision_facts(
        rows=rows,
        deterministic=result.deterministic_summary,
        operational={
            **result.operational_summary,
            "infrastructure_failures": sum(
                str(row.get("status") or "") == "failed" for row in rows
            ),
        },
        improved=result.improved,
        regressed=result.regressed,
        incomplete=result.incomplete,
        required_incomplete=result.required_evaluations_incomplete,
        integrity=integrity,
        evidence_grade=grade,
    )
    source_statuses = {
        result.evidence_topology.pre_run_drift.status,
        result.evidence_topology.post_run_drift.status,
    }
    facts["integrity.source_project_writes"] = (
        1
        if "drifted" in source_statuses
        else 0
        if source_statuses == {"matched"}
        else None
    )
    return facts


def _verify_v3_result_shape(result: ComparisonResultV3) -> None:
    if result.evidence_project != (
        result.evidence_topology.result_destination.project_slug
    ):
        raise ValueError(
            "ComparisonResultV3 evidence project disagrees with its topology"
        )
    lineage = result.cohort_lineage
    if lineage.get("source_lock_digest") != (
        result.evidence_topology.source_lock_digest
    ):
        raise ValueError(
            "ComparisonResultV3 cohort lineage source lock disagrees with "
            "its evidence topology"
        )
    lineage_scorers = _mapping(
        lineage.get("scorer_digests"),
        "comparison cohort scorer digests",
    )
    if lineage_scorers != {
        item.id: item.digest for item in result.scorer_revisions
    }:
        raise ValueError(
            "ComparisonResultV3 cohort lineage scorer revisions disagree"
        )
    lineage_execution = _mapping(
        lineage.get("execution"),
        "comparison cohort execution",
    )
    if (
        lineage_execution.get("source_evidence_project")
        != result.evidence_topology.source_destination.project_slug
        or lineage_execution.get("result_evidence_project")
        != result.evidence_topology.result_destination.project_slug
    ):
        raise ValueError(
            "ComparisonResultV3 cohort lineage project topology disagrees"
        )
    canonical_rows = _v3_canonical_attempt_rows(result)
    semantic_integrity = _v3_semantic_integrity(result, canonical_rows)
    for key, expected in semantic_integrity.items():
        if result.integrity.get(key) != expected:
            raise ValueError(
                "ComparisonResultV3 integrity field "
                f"{key!r} disagrees with canonical attempts"
            )
    expected_runtime_locks = _runtime_locks_v3(canonical_rows)
    observed_attempt_locks = tuple(
        item
        for item in result.runtime_locks
        if item.details.get("kind") != "approved_runtime_lock"
    )
    if tuple(item.to_dict() for item in observed_attempt_locks) != tuple(
        item.to_dict() for item in expected_runtime_locks
    ):
        raise ValueError(
            "ComparisonResultV3 runtime locks disagree with canonical attempts"
        )
    aligned_harnesses = {
        item.harness for item in result.aligned_analysis.aligned_attempts
    }
    if set(lineage_execution.get("harnesses") or ()) != aligned_harnesses:
        raise ValueError(
            "ComparisonResultV3 cohort lineage harnesses disagree with "
            "aligned attempts"
        )
    lineage_arms = _mapping(lineage.get("arms"), "comparison cohort arms")
    for arm in result.aligned_analysis.arms:
        expected_arm = _mapping(
            lineage_arms.get(arm.id),
            f"comparison cohort arm {arm.id}",
        )
        expected_revisions = {
            (
                str(item.get("id") or ""),
                str(item.get("version_identity") or ""),
                str(item.get("runtime_digest") or ""),
            )
            for item in expected_arm.get("source_revisions") or ()
            if isinstance(item, Mapping)
        }
        source_revision = (
            arm.source_revision
            if isinstance(arm.source_revision, Mapping)
            else {}
        )
        observed_revisions = {
            (
                str(item.get("id") or ""),
                str(item.get("version_identity") or ""),
                str(item.get("runtime_digest") or ""),
            )
            for item in source_revision.get("revisions") or ()
            if isinstance(item, Mapping)
        }
        if expected_revisions != observed_revisions:
            raise ValueError(
                f"ComparisonResultV3 cohort lineage {arm.id} source "
                "revisions disagree"
            )
    validity_by_task = {item.task_id: item.status for item in result.task_validity}
    if len(validity_by_task) != len(result.task_validity):
        raise ValueError("ComparisonResultV3 task validity IDs must be unique")
    paired_tasks = {item.task_id for item in result.paired_cases}
    if paired_tasks != set(validity_by_task):
        raise ValueError(
            "ComparisonResultV3 task validity does not cover every paired task"
        )
    attempt_ids = {
        attempt.attempt_id
        for pair in result.paired_cases
        for attempt in (pair.baseline, pair.candidate)
        if attempt is not None
    }
    terminal_states = {
        "completed",
        "failed",
        "cancelled",
        "interrupted",
        "not_applicable",
    }
    attempt_statuses = [
        attempt.execution_status
        for pair in result.paired_cases
        for attempt in (pair.baseline, pair.candidate)
        if attempt is not None
    ]
    if any(status not in terminal_states for status in attempt_statuses):
        raise ValueError(
            "ComparisonResultV3 paired attempts must all be terminal"
        )
    if _mapping_or_empty(
        result.operational_summary.get("execution_states")
    ) != dict(sorted(Counter(attempt_statuses).items())):
        raise ValueError(
            "ComparisonResultV3 terminal execution-state totals disagree "
            "with paired attempts"
        )
    aligned_attempt_ids = {
        attempt_id
        for aligned in result.aligned_analysis.aligned_attempts
        for attempt_id in aligned.attempt_ids_by_arm.values()
    }
    complete_pair_attempt_ids = {
        attempt.attempt_id
        for pair in result.paired_cases
        if pair.baseline is not None and pair.candidate is not None
        for attempt in (pair.baseline, pair.candidate)
    }
    if aligned_attempt_ids != complete_pair_attempt_ids:
        raise ValueError(
            "ComparisonResultV3 aligned analysis does not cover every complete "
            "paired attempt"
        )
    if result.rows != len(attempt_ids):
        raise ValueError(
            "ComparisonResultV3 row count disagrees with unique paired attempts"
        )
    paired_dimensions = {
        change.id
        for pair in result.paired_cases
        for change in pair.dimension_changes
    }
    for item in result.release_note_coverage:
        unknown_tasks = set(item.get("task_ids") or ()) - paired_tasks
        unknown_dimensions = (
            set(item.get("dimensions") or ()) - paired_dimensions
        )
        if unknown_tasks or unknown_dimensions:
            raise ValueError(
                "ComparisonResultV3 release-note coverage references "
                "unknown task or dimension evidence"
            )
    _verify_v3_derived_analysis(
        result,
        canonical_rows=canonical_rows,
        semantic_integrity=semantic_integrity,
    )


def _verify_v3_derived_analysis(
    result: ComparisonResultV3,
    *,
    canonical_rows: Sequence[Mapping[str, Any]],
    semantic_integrity: Mapping[str, Any],
) -> None:
    (
        pair_counts,
        baseline_passed,
        candidate_passed,
        candidate_critical_failures,
    ) = _verify_v3_pairs(result.paired_cases)
    expected_counts = {
        "improved": result.improved,
        "regressed": result.regressed,
        "mixed": result.mixed,
        "unchanged": result.unchanged,
        "incomplete": result.incomplete,
    }
    if any(
        pair_counts[status] != count
        for status, count in expected_counts.items()
    ):
        raise ValueError(
            "ComparisonResultV3 aggregate pair counts disagree with paired cases"
        )
    if (
        result.baseline_passed != baseline_passed
        or result.candidate_passed != candidate_passed
    ):
        raise ValueError(
            "ComparisonResultV3 task pass totals disagree with paired attempts"
        )
    recomputed_validity = _task_validity_v3(
        result.paired_cases,
        topology=result.evidence_topology,
    )
    if tuple(item.to_dict() for item in result.task_validity) != tuple(
        item.to_dict() for item in recomputed_validity
    ):
        raise ValueError(
            "ComparisonResultV3 task validity disagrees with paired evidence"
        )
    expected_task_summaries = tuple(
        TaskStratifiedSummaryV1(
            task_id=validity.task_id,
            validity=validity.status,
            pair_counts=dict(
                Counter(
                    pair.status
                    for pair in result.paired_cases
                    if pair.task_id == validity.task_id
                )
            ),
            blockers=validity.blockers,
        ).to_dict()
        for validity in recomputed_validity
    )
    if tuple(
        item.to_dict() for item in result.aligned_analysis.task_summaries
    ) != expected_task_summaries:
        raise ValueError(
            "ComparisonResultV3 aligned task summaries disagree with paired cases"
        )
    _verify_v3_behavioral_summary(
        result,
        pair_counts=pair_counts,
        candidate_critical_failures=candidate_critical_failures,
        task_validity=recomputed_validity,
    )
    reconstructed_rows = [
        {
            "variant_id": variant,
            "pass": attempt.passed,
            "comparison_evaluation_status": attempt.evaluation_status,
            "comparison_deterministic_scores": {
                dimension: value
                for dimension, value in attempt.scores.items()
                if not dimension.startswith("comparison.judge.")
            },
        }
        for pair in result.paired_cases
        for variant, attempt in (
            ("baseline", pair.baseline),
            ("candidate", pair.candidate),
        )
        if attempt is not None
    ]
    if _deterministic_summary(reconstructed_rows) != result.deterministic_summary:
        raise ValueError(
            "ComparisonResultV3 deterministic summary disagrees with paired "
            "attempt scores"
        )
    _verify_v3_decision_gates(
        result,
        canonical_rows=canonical_rows,
        semantic_integrity=semantic_integrity,
    )


def _verify_v3_pairs(
    pairs: Sequence[PairedCaseV3],
) -> tuple[Counter[str], int, int, int]:
    pair_counts: Counter[str] = Counter()
    baseline_passed = 0
    candidate_passed = 0
    candidate_critical_failures = 0
    for pair in pairs:
        dimension_ids: set[str] = set()
        for change in pair.dimension_changes:
            if change.id in dimension_ids:
                raise ValueError(
                    "ComparisonResultV3 paired dimension IDs must be unique"
                )
            dimension_ids.add(change.id)
            if change.baseline is None or change.candidate is None:
                expected_status = "unavailable"
            elif change.baseline is False and change.candidate is True:
                expected_status = "improved"
            elif change.baseline is True and change.candidate is False:
                expected_status = "regressed"
            else:
                expected_status = "unchanged"
            if change.status != expected_status:
                raise ValueError(
                    "ComparisonResultV3 dimension status disagrees with its "
                    f"baseline/candidate values for {pair.task_id}:{change.id}"
                )
            candidate_critical_failures += bool(
                change.critical
                and change.role in {"outcome", "safety_gate"}
                and change.candidate is False
            )
        if (
            pair.baseline is None
            or pair.candidate is None
            or any(
                change.status == "unavailable"
                for change in pair.dimension_changes
            )
        ):
            expected_pair_status = "incomplete"
        else:
            expected_pair_status = _pair_status_v3(pair.dimension_changes)
        if pair.status != expected_pair_status:
            raise ValueError(
                "ComparisonResultV3 pair status disagrees with its locked "
                f"dimension changes for {pair.task_id}"
            )
        pair_counts[pair.status] += 1
        baseline_passed += bool(
            pair.baseline is not None and pair.baseline.passed is True
        )
        candidate_passed += bool(
            pair.candidate is not None and pair.candidate.passed is True
        )
    return (
        pair_counts,
        baseline_passed,
        candidate_passed,
        candidate_critical_failures,
    )


def _verify_v3_behavioral_summary(
    result: ComparisonResultV3,
    *,
    pair_counts: Counter[str],
    candidate_critical_failures: int,
    task_validity: Sequence[TaskValidityV1],
) -> None:
    behavioral = result.behavioral_summary
    if (
        behavioral.improved_pairs != pair_counts["improved"]
        or behavioral.regressed_pairs != pair_counts["regressed"]
        or behavioral.mixed_pairs != pair_counts["mixed"]
        or behavioral.unchanged_pairs != pair_counts["unchanged"]
        or behavioral.incomplete_pairs != pair_counts["incomplete"]
        or behavioral.candidate_critical_failures
        != candidate_critical_failures
    ):
        raise ValueError(
            "ComparisonResultV3 behavioral totals disagree with paired cases"
        )
    validity_blockers = tuple(
        dict.fromkeys(
            blocker
            for validity in task_validity
            for blocker in validity.blockers
        )
    )
    paired_blockers = tuple(
        dict.fromkeys(
            f"{pair.task_id}: {change.id} failed for the candidate"
            for pair in result.paired_cases
            for change in pair.dimension_changes
            if change.role in {"outcome", "safety_gate"}
            and change.critical
            and change.candidate is False
        )
    )
    expected_blockers = tuple(
        dict.fromkeys((*validity_blockers, *paired_blockers))
    )
    if result.integrity.get("status") == "invalid":
        expected_behavioral_status = "invalid"
    elif (
        pair_counts["incomplete"]
        or result.required_evaluations_incomplete
        or int(result.integrity.get("unresolved_evidence_attempts") or 0)
        or int(
            result.integrity.get(
                "harbor_conformance_unavailable_attempts"
            )
            or 0
        )
    ):
        expected_behavioral_status = "incomplete"
    elif pair_counts["mixed"] or (
        pair_counts["improved"] and pair_counts["regressed"]
    ):
        expected_behavioral_status = "mixed"
    elif pair_counts["regressed"]:
        expected_behavioral_status = "regressed"
    elif pair_counts["improved"] and not expected_blockers:
        expected_behavioral_status = "improved"
    else:
        expected_behavioral_status = "unchanged"
    if behavioral.status != expected_behavioral_status:
        raise ValueError(
            "ComparisonResultV3 behavioral status disagrees with paired evidence"
        )
    if expected_behavioral_status not in {"invalid", "incomplete"} and (
        behavioral.critical_blockers != expected_blockers
    ):
        raise ValueError(
            "ComparisonResultV3 behavioral blockers disagree with paired evidence"
        )


def _verify_v3_decision_gates(
    result: ComparisonResultV3,
    *,
    canonical_rows: Sequence[Mapping[str, Any]],
    semantic_integrity: Mapping[str, Any],
) -> None:
    decision = result.decision
    policy = _decision_policy(result.decision_policy)
    expected_grade = _evidence_grade(semantic_integrity, canonical_rows)
    if decision.evidence_grade != expected_grade:
        raise ValueError(
            "ComparisonResultV3 decision evidence grade disagrees with "
            "canonical attempts"
        )
    if policy is None:
        if (
            decision.release_target is not None
            or decision.candidate_sha is not None
            or decision.human_signoff_required is not True
        ):
            raise ValueError(
                "ComparisonResultV3 decision identity disagrees with the "
                "absence of a policy"
            )
    elif (
        decision.release_target != policy.release_target
        or decision.candidate_sha != policy.candidate_sha
        or decision.human_signoff_required
        != policy.human_signoff_required
    ):
        raise ValueError(
            "ComparisonResultV3 decision identity disagrees with its "
            "governed policy"
        )
    if decision.status == "go" and (
        decision.attestation is None
        or decision.attestation.review_status != "accepted_actionable"
        or decision.attestation.signed_result_digest
        != result.qualification_digest
    ):
        raise ValueError(
            "ComparisonResultV3 GO requires an accepted actionability "
            "attestation for the exact qualification digest"
        )
    if result.integrity.get("status") == "invalid":
        if decision.status != "invalid" or decision.gates:
            raise ValueError(
                "ComparisonResultV3 invalid integrity must suppress decision gates"
            )
        return
    if policy is None:
        if decision.status != "inconclusive" or decision.gates:
            raise ValueError(
                "ComparisonResultV3 without a policy cannot claim a release "
                "decision"
        )
        return
    facts = _v3_decision_facts(
        result,
        canonical_rows,
        integrity=semantic_integrity,
    )
    policies = _canonical_decision_gate_policies(
        list(policy.gates),
        implicit=_implicit_decision_gate_policies(),
        release_note_coverage=result.release_note_coverage,
    )
    by_id = {item.id: item for item in decision.gates}
    expected_ids = {item.id for item in policies} | {"evidence-grade"}
    if set(by_id) != expected_ids:
        raise ValueError(
            "ComparisonResultV3 decision gates do not match the governed policy"
        )
    for gate in policies:
        observed = by_id[gate.id]
        if (
            observed.label != gate.label
            or observed.category != gate.category
            or observed.critical != gate.critical
            or observed.target != gate.target
            or observed.actual != facts.get(gate.source)
            or observed.status
            != _gate_status(
                facts.get(gate.source),
                gate.operator,
                gate.target,
            )
        ):
            raise ValueError(
                f"ComparisonResultV3 decision gate {gate.id!r} disagrees "
                "with its governed policy"
            )
    grade = by_id["evidence-grade"]
    grade_passed = _grade_rank(expected_grade) >= _grade_rank(
        policy.minimum_evidence_grade
    )
    if (
        grade.label != "Minimum evidence grade"
        or grade.category != "evidence"
        or grade.critical is not True
        or grade.actual != expected_grade
        or grade.target != policy.minimum_evidence_grade
        or grade.status != ("passed" if grade_passed else "failed")
    ):
        raise ValueError(
            "ComparisonResultV3 evidence-grade gate disagrees with the policy"
        )
    expected_blockers = tuple(
        item.label
        for item in decision.gates
        if item.critical and item.status != "passed"
    )
    validity_blockers = tuple(
        blocker
        for item in result.task_validity
        if item.status in {"drifted", "invalid", "inconclusive"}
        for blocker in item.blockers
    )
    topology_blockers = tuple(
        f"{label} source drift check is {drift.status}"
        for label, drift in (
            ("pre-run", result.evidence_topology.pre_run_drift),
            ("post-run", result.evidence_topology.post_run_drift),
        )
        if drift.status != "matched"
    )
    release_note_blockers = (
        (
            "unqualified release-note behavior(s): "
            + ", ".join(
                sorted(
                    str(item.get("release_note") or "")
                    for item in result.release_note_coverage
                    if str(item.get("status") or "") == "unqualified"
                )
            ),
        )
        if any(
            str(item.get("status") or "") == "unqualified"
            for item in result.release_note_coverage
        )
        else ()
    )
    expected_blockers = tuple(
        dict.fromkeys(
            (
                *expected_blockers,
                *validity_blockers,
                *topology_blockers,
                *release_note_blockers,
            )
        )
    )
    if decision.critical_blockers != expected_blockers:
        raise ValueError(
            "ComparisonResultV3 decision blockers disagree with governed gates"
        )
    if validity_blockers or topology_blockers or release_note_blockers:
        expected_status: DecisionStatus = (
            "invalid"
            if any(
                item.status in {"drifted", "invalid"}
                for item in result.task_validity
            )
            or result.evidence_topology.pre_run_drift.status == "drifted"
            or result.evidence_topology.post_run_drift.status == "drifted"
            else "hold"
        )
    elif any(
        item.status == "unavailable"
        for item in decision.gates
        if item.critical
    ):
        expected_status = "blocked"
    elif expected_blockers:
        expected_status = "hold"
    elif policy.human_signoff_required:
        expected_status = (
            "go" if decision.attestation is not None else "ready_for_signoff"
        )
    else:
        expected_status = "go"
    if decision.status != expected_status:
        raise ValueError(
            "ComparisonResultV3 decision status disagrees with governed gates"
        )


def scaffold_comparison(destination: Path, *, force: bool = False) -> Path:
    root = destination.resolve()
    if root.exists() and any(root.iterdir()) and not force:
        raise FileExistsError(
            f"refusing to overwrite non-empty comparison directory: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    skill_root = (
        root / "configs" / "fugue" / "skills" / "verify-current-source"
    )
    skill_root.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        root / "tasks.jsonl",
        json.dumps(
            {
                "id": "policy-limit",
                "input": {
                    "question": (
                        "Find the current expense limit and return JSON with "
                        "amount and source."
                    )
                },
                "tags": ["source-use"],
                "partition": "holdout",
            },
            sort_keys=True,
        )
        + "\n",
    )
    _atomic_text(
        root / "private-labels.jsonl",
        json.dumps(
            {
                "id": "policy-limit",
                "expected": {"amount": 125, "source": "expense-policy-v4.md"},
                "base_output": {
                    "amount": 100,
                    "source": "expense-policy-v3.md",
                },
                "gold_output": {
                    "amount": 125,
                    "source": "expense-policy-v4.md",
                },
            },
            sort_keys=True,
        )
        + "\n",
    )
    _atomic_text(
        skill_root / "SKILL.md",
        (
            "---\n"
            "name: verify-current-source\n"
            "description: Inspect and cite the current authoritative source.\n"
            "---\n\n"
            "# Verify current source\n\n"
            "Open the authoritative source before answering. Prefer a current, "
            "effective document over a draft or superseded revision, and cite "
            "the exact filename used.\n"
        ),
    )
    config = {
        "schema_version": 1,
        "id": "source-use",
        "question": "Does verifying the current source improve evidence use?",
        "taskset": {
            "tasks": "tasks.jsonl",
            "private_labels": "private-labels.jsonl",
        },
        "baseline": {"label": "Current Agent"},
        "candidate": {
            "label": "Current Agent + source verification Skill",
            "skills": ["verify-current-source"],
        },
        "changed": ["skills"],
        "evaluators": [
            {
                "id": "fact-and-source",
                "type": "deterministic",
                "required": True,
                "checks": ["answer_present", "expected_values"],
            }
        ],
        "execution": {
            "model": "wandb/zai-org/GLM-5.2",
            "harnesses": ["codex"],
            "attempts": 2,
            "concurrency": 1,
            "max_cost_usd": 40,
            "reserve_per_attempt_usd": 10,
            "approval_required": True,
            "trace_content": "full",
        },
    }
    _atomic_text(
        root / "comparison.yaml",
        yaml.safe_dump(config, sort_keys=False),
    )
    _atomic_text(
        root / "README.md",
        (
            "# Fugue Agent-change comparison\n\n"
            "Run from the repository root:\n\n"
            "```bash\n"
            f"uv run fugue check {root.as_posix()}/comparison.yaml\n"
            "```\n"
        ),
    )
    return root / "comparison.yaml"


def _approved_evaluator_artifact_digests(
    approved_inputs: Mapping[str, Any] | None,
    evaluators: Sequence[ComparisonEvaluatorV1],
    *,
    artifact: Literal["scorer", "verifier"],
) -> dict[str, str] | None:
    if approved_inputs is None:
        return None
    artifacts = _mapping(
        approved_inputs["evaluator_artifacts"],
        "approved evaluator artifacts",
    )
    enabled = (
        (lambda evaluator: bool(evaluator.scorer))
        if artifact == "scorer"
        else (lambda evaluator: bool(evaluator.verifier))
    )
    return {
        evaluator.id: str(
            _mapping(
                artifacts.get(evaluator.id),
                f"approved evaluator {evaluator.id} artifacts",
            ).get(f"{artifact}_sha256")
            or ""
        )
        for evaluator in evaluators
        if enabled(evaluator)
    }


def _bind_host_verifier_receipts(
    row: dict[str, Any], receipts: Mapping[str, Mapping[str, Any]]
) -> None:
    if receipts:
        attempt = str(row.get("attempt_id") or "")
        task_id = str(row.get("task_id") or "")
        bound: dict[str, dict[str, Any]] = {}
        for evaluator_id, raw in receipts.items():
            value = dict(raw)
            _verify_bound_host_verifier_receipt(
                value,
                evaluator_id=str(evaluator_id),
                attempt_id=attempt,
                task_id=task_id,
            )
            bound[str(evaluator_id)] = value
        row["comparison_host_verifier_receipts"] = bound


def _verify_bound_host_verifier_receipt(
    value: Mapping[str, Any],
    *,
    evaluator_id: str,
    attempt_id: str,
    task_id: str,
) -> None:
    required = {
        "schema_version",
        "kind",
        "evaluator_id",
        "task_id",
        "attempt_id",
        "status",
        "failure_kind",
        "runtime",
        "command",
        "exit_code",
        "test_count",
        "pass_count",
        "fail_count",
        "output_sha256",
        "base_archive_sha256",
        "public_test_sha256",
        "submitted_artifact_sha256",
        "final_tree_sha256",
        "verifier_source_sha256",
        "runtime_profile_id",
        "runtime_profile_digest",
        "runtime_image",
        "runtime_platform",
        "runtime_image_id",
        "runtime_lock_digest",
        "runtime_cleanup",
        "receipt_digest",
    }
    if set(value) != required:
        raise ValueError("bound host verifier receipt fields do not match")
    unsigned = {key: item for key, item in value.items() if key != "receipt_digest"}
    if value.get("receipt_digest") != stable_digest(unsigned):
        raise ValueError("host verifier receipt digest does not match")
    if (
        value.get("schema_version") != 2
        or value.get("kind") != "post_trial_verifier_receipt"
        or value.get("evaluator_id") != evaluator_id
        or value.get("attempt_id") != attempt_id
        or value.get("task_id") != task_id
    ):
        raise ValueError("host verifier receipt belongs to another attempt")
    for name in (
        "output_sha256",
        "base_archive_sha256",
        "public_test_sha256",
        "submitted_artifact_sha256",
        "final_tree_sha256",
        "verifier_source_sha256",
        "runtime_profile_digest",
        "runtime_lock_digest",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(value.get(name) or "")):
            raise ValueError(f"host verifier receipt {name} is invalid")
    cleanup = _mapping(value.get("runtime_cleanup"), "verifier runtime cleanup")
    cleanup_unsigned = {
        key: item for key, item in cleanup.items() if key != "receipt_digest"
    }
    if (
        cleanup.get("schema_version") != 1
        or cleanup.get("status") != "verified_absent"
        or cleanup.get("receipt_digest") != stable_digest(cleanup_unsigned)
    ):
        raise ValueError("host verifier runtime cleanup receipt is invalid")


def score_comparison_rows(
    spec: ComparisonSpecV1,
    rows: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path,
    env: Mapping[str, str] | None = None,
    approved_comparison: Mapping[str, Any] | None = None,
    judge_request: Callable[..., tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]
    | None = None,
) -> list[dict[str, Any]]:
    approved_inputs = (
        _verified_approved_inputs(approved_comparison, repo_root=repo_root)
        if approved_comparison
        else None
    )
    public_tasks = {
        str(item["id"]): item
        for item in _load_public_tasks(
            _frozen_public_tasks_path(
                repo_root,
                str(approved_inputs["public_tasks_sha256"]),
            )
            if approved_inputs is not None
            else repo_root / spec.taskset.tasks
        )
    }
    labels = {
        str(item["id"]): item
        for item in _load_private_labels(
            _frozen_private_labels_path(
                repo_root,
                str(approved_inputs["private_labels_sha256"]),
            )
            if approved_inputs is not None
            else repo_root / spec.taskset.private_labels
        )
    }
    deterministic = tuple(
        evaluator
        for evaluator in spec.evaluators
        if evaluator.type == "deterministic"
    )
    scorer_source_digests = _approved_evaluator_artifact_digests(
        approved_inputs,
        deterministic,
        artifact="scorer",
    )
    verifier_source_digests = _approved_evaluator_artifact_digests(
        approved_inputs,
        deterministic,
        artifact="verifier",
    )
    scored: list[dict[str, Any]] = []
    for source in rows:
        row = _bind_attempt_identity(source)
        task_id = str(row.get("task_id") or row.get("task_name") or "")
        output = (
            row.get("final_output")
            if row.get("final_output") is not None
            else row.get("answer")
        )
        label = labels.get(task_id)
        row.setdefault("benchmark_pass", row.get("pass"))
        if label is None:
            row["pass"] = None
            row["comparison_evaluation_status"] = "unavailable"
            row["comparison_evaluation_reason"] = "private task label is unavailable"
            row["comparison_required_evaluation_complete"] = False
        else:
            try:
                passed, dimensions, verifier_receipts = _score_deterministic_output(
                    task=public_tasks.get(task_id, {}),
                    output=output,
                    expected=label["expected"],
                    evidence=_custom_scorer_evidence(row),
                    evaluators=deterministic,
                    repo_root=repo_root,
                    scorer_source_digests=scorer_source_digests,
                    verifier_source_digests=verifier_source_digests,
                )
            except Exception as exc:
                row["pass"] = None
                row["comparison_evaluation_status"] = "unavailable"
                row["comparison_evaluation_reason"] = (
                    "deterministic evaluation failed: "
                    f"{type(exc).__name__}"
                )
                row["comparison_required_evaluation_complete"] = False
            else:
                row["pass"] = passed
                row["comparison_evaluation_status"] = "scored"
                row["comparison_deterministic_scores"] = dimensions
                _bind_host_verifier_receipts(row, verifier_receipts)
                row["comparison_dimension_roles"] = {
                    f"{evaluator.id}.{dimension}": role
                    for evaluator in deterministic
                    for dimension, role in evaluator.dimension_roles.items()
                }
                critical_dimensions = tuple(
                    str(item)
                    for item in public_tasks.get(task_id, {}).get(
                        "critical_dimensions", ()
                    )
                )
                row["comparison_deterministic_criticality"] = {
                    name: True for name in critical_dimensions
                }
                row["comparison_mechanism"] = _comparison_mechanism(
                    row,
                    expected=label["expected"],
                    passed=passed,
                    expected_skill_ids=(
                        spec.baseline.skills
                        if str(row.get("variant_id") or "") == "baseline"
                        else spec.candidate.skills
                        if str(row.get("variant_id") or "") == "candidate"
                        else ()
                    ),
                )
                row["comparison_required_evaluation_complete"] = True
        judge_results: dict[str, Any] = {}
        judge_scores: dict[str, float] = {}
        judge_accounted_reserve = 0.0
        for judge in (
            evaluator
            for evaluator in spec.evaluators
            if evaluator.type == "llm_judge"
        ):
            judge_reserve_accounted = False
            qualification = _comparison_judge_qualification(
                judge,
                repo_root=repo_root,
                approved_inputs=approved_inputs,
            )
            if env is None:
                judge_results[judge.id] = {
                    "status": "unavailable",
                    "reason": "judge execution was not requested",
                    "qualification": qualification,
                }
                if judge.required:
                    row["comparison_required_evaluation_complete"] = False
                continue
            calibration_issue = _judge_execution_calibration_issue(
                judge,
                repo_root=repo_root,
                approved_inputs=approved_inputs,
            )
            if calibration_issue:
                judge_results[judge.id] = {
                    "status": "unavailable",
                    "reason": calibration_issue,
                    "qualification": qualification,
                }
                if judge.required:
                    row["comparison_required_evaluation_complete"] = False
                continue
            failure_stage = "input_privacy"
            request_policy: dict[str, Any] | None = None
            try:
                (
                    judge_provider_payload,
                    judge_input_privacy,
                ) = _prepare_comparison_judge_input(
                    evaluator=judge,
                    public_task=public_tasks.get(task_id, {}),
                    row=row,
                    env=env,
                )
                failure_stage = "provider_request"
                request_policy = _comparison_judge_request_policy(judge, env)
                judge_accounted_reserve += judge.reserve_cost_usd
                judge_reserve_accounted = True
                payload, usage, receipt = _invoke_comparison_judge_request(
                    judge_request=judge_request,
                    evaluator=judge,
                    public_task=public_tasks.get(task_id, {}),
                    row=row,
                    env=env,
                    prepared_payload=judge_provider_payload,
                    input_transform_receipt=judge_input_privacy,
                )
                failure_stage = "output_privacy"
                judge_output_privacy = _comparison_judge_output_privacy_receipt(
                    payload=payload,
                    usage=usage,
                    env=env,
                )
                failure_stage = "rubric_validation"
                parsed = _validate_comparison_judge_payload(judge, payload)
                observed_judge_cost = _row_number(
                    usage,
                    "cost_usd",
                    "observed_cost_usd",
                )
                judge_cost_evidence = {
                    "observed_cost_usd": observed_judge_cost,
                    "accounted_reserve_usd": judge.reserve_cost_usd,
                    "cost_status": (
                        "observed"
                        if observed_judge_cost is not None
                        else "unavailable"
                    ),
                }
                route_receipt = {
                    **dict(receipt),
                    "judge_input_privacy": judge_input_privacy,
                    "judge_output_privacy": judge_output_privacy,
                }
                if parsed["missing_evidence"]:
                    judge_results[judge.id] = {
                        "status": "missing_evidence",
                        **parsed,
                        "usage": usage,
                        "route_receipt": route_receipt,
                        "qualification": qualification,
                        **judge_cost_evidence,
                    }
                    if judge.required:
                        row["comparison_required_evaluation_complete"] = False
                else:
                    for dimension, value in parsed["scores"].items():
                        judge_scores[f"{judge.id}.{dimension}"] = value
                    judge_results[judge.id] = {
                        "status": "scored",
                        **parsed,
                        "usage": usage,
                        "route_receipt": route_receipt,
                        "qualification": qualification,
                        **judge_cost_evidence,
                    }
            except Exception as exc:
                failure = _comparison_judge_failure_metadata(
                    exc,
                    fallback_stage=failure_stage,
                )
                if request_policy is not None:
                    failure["request_policy"] = request_policy
                judge_results[judge.id] = {
                    "status": "unavailable",
                    "reason": (
                        "judge evaluation failed: "
                        f"{failure['exception_type']}"
                    ),
                    "failure": failure,
                    "qualification": qualification,
                    **(
                        {
                            "observed_cost_usd": None,
                            "accounted_reserve_usd": judge.reserve_cost_usd,
                            "cost_status": "unavailable",
                        }
                        if judge_reserve_accounted
                        else {}
                    ),
                }
                if judge.required:
                    row["comparison_required_evaluation_complete"] = False
        if judge_results:
            row["comparison_judges"] = judge_results
            row["comparison_judge_status"] = (
                "scored"
                if all(value["status"] == "scored" for value in judge_results.values())
                else "unavailable"
            )
        if judge_scores:
            row["comparison_judge_scores"] = judge_scores
        if judge_accounted_reserve:
            row["comparison_judge_accounted_cost_usd"] = judge_accounted_reserve
            base_cost = _accounted_attempt_cost(row)
            if base_cost is not None:
                row["accounted_cost_usd"] = base_cost + judge_accounted_reserve
        scored.append(row)
    return scored


def _comparison_judge_response_schema(
    dimensions: Sequence[str],
) -> dict[str, Any]:
    """Return the exact provider-visible shape for one blind judge response.

    Anthropic does not enforce string-length or numeric-range JSON Schema
    constraints. Those remain local fail-closed checks; descriptions communicate
    the requested concise form to the provider.
    """

    requested = COMPARISON_JUDGE_TEXT_REQUESTED_MAX_CHARACTERS
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "scores",
            "overall_assessment",
            "uncertainty",
            "missing_evidence",
            "rationale",
        ],
        "properties": {
            "scores": {
                "type": "object",
                "additionalProperties": False,
                "required": list(dimensions),
                "properties": {
                    dimension: {"type": "number"} for dimension in dimensions
                },
            },
            "overall_assessment": {
                "type": "string",
                "description": (
                    "Brief evidence-bounded assessment; requested maximum "
                    f"{requested} characters."
                ),
            },
            "uncertainty": {"type": "number"},
            "missing_evidence": {"type": "boolean"},
            "rationale": {
                "type": "string",
                "description": (
                    "Brief evidence-bounded rationale; requested maximum "
                    f"{requested} characters."
                ),
            },
        },
    }


def _request_comparison_judge(
    *,
    evaluator: ComparisonEvaluatorV1,
    public_task: Mapping[str, Any],
    row: Mapping[str, Any],
    env: Mapping[str, str],
    prepared_payload: Mapping[str, Any] | None = None,
    input_transform_receipt: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from fugue.bench.evaluations import (
        _post_judge,
    )
    from fugue.model_plane import (
        model_route_identity,
        provider_api_key,
        provider_api_key_env,
        resolve_model_route,
    )

    if not evaluator.profile or not evaluator.rubric:
        raise ValueError("comparison judge is missing its profile or public rubric")
    route = resolve_model_route(evaluator.profile, env)
    request_policy = _comparison_judge_request_policy(
        evaluator,
        env,
        route=route,
    )
    api_key = provider_api_key(route, env)
    if not api_key:
        raise RuntimeError(
            f"{provider_api_key_env(route)} is required for comparison judging"
        )
    expected_payload, expected_transform_receipt = (
        _prepare_comparison_judge_input(
            evaluator=evaluator,
            public_task=public_task,
            row=row,
            env=env,
        )
    )
    if prepared_payload is not None and dict(prepared_payload) != expected_payload:
        raise ValueError(
            "prepared judge payload does not match deterministic sanitization"
        )
    if (
        input_transform_receipt is not None
        and dict(input_transform_receipt) != expected_transform_receipt
    ):
        raise ValueError("judge input transform receipt does not match")
    payload = expected_payload
    payload_sha256 = stable_digest(payload)
    transform_receipt_digest = str(
        expected_transform_receipt["receipt_digest"]
    )
    if payload_sha256 != expected_transform_receipt["provider_payload_sha256"]:
        raise ValueError("judge provider payload digest does not match")
    if transform_receipt_digest != stable_digest(
        {
            key: value
            for key, value in expected_transform_receipt.items()
            if key != "receipt_digest"
        }
    ):
        raise ValueError("judge input transform receipt digest does not match")
    prompt_prefix = (
        "Blindly evaluate one Agent attempt. You do not know whether it came from "
        "the baseline or candidate. Use only the supplied public task, final "
        "response, permitted evidence, and rubric. Return one JSON object with: "
        "scores (one 0..1 number per dimension), overall_assessment (brief text), "
        "uncertainty (0..1), missing_evidence (boolean), and rationale (at most "
        "500 characters). Judge evidence applicability against the requested "
        "artifact type. Path lists are audited activity, not proof of correctness. "
        "Use each path status: available means collection succeeded; unavailable "
        "means the list is not evidence. An available empty changed_paths list is "
        "expected for a plan or recommendation and is not, by itself, missing "
        "evidence. Do not require implementation proof that the public task did "
        "not request. Do not return "
        "hidden reasoning or a chain of thought.\n\n"
    )
    prompt = prompt_prefix + json.dumps(payload, sort_keys=True, default=str)
    if len(prompt) > COMPARISON_JUDGE_MAX_PROMPT_CHARACTERS:
        raise ValueError("comparison judge prompt exceeds the locked bound")
    import httpx

    timeout_sec = int(request_policy["timeout_sec"])
    # HTTPX issues one request by default. Do not install a retrying transport:
    # a timed-out judge remains unavailable instead of creating duplicate spend.
    response_schema = _comparison_judge_response_schema(evaluator.dimensions)
    with httpx.Client(timeout=timeout_sec) as client:
        if route.messages_base_url:
            response, usage = _post_judge(
                client,
                route,
                api_key,
                env,
                prompt,
                response_schema=response_schema,
            )
        else:
            response, usage = _post_judge(
                client,
                route,
                api_key,
                env,
                prompt,
            )
    return (
        response,
        usage,
        {
            "schema_version": 1,
            "role": "blind_comparison_judge",
            "judge_id": evaluator.id,
            "profile": evaluator.profile,
            "route": model_route_identity(route, env),
            "rubric_digest": _judge_contract_digest(evaluator),
            "response_schema_digest": stable_digest(response_schema),
            "response_request_mode": (
                "anthropic_json_schema_no_thinking_v2"
                if route.messages_base_url
                else "provider_json_object_v1"
            ),
            "response_validator_version": (
                COMPARISON_JUDGE_RESPONSE_VALIDATOR_VERSION
            ),
            "public_rubric_contract_digest": stable_digest(payload["rubric"]),
            "provider_payload_sha256": payload_sha256,
            "input_transform_receipt_digest": transform_receipt_digest,
            "requested_text_max_characters": (
                COMPARISON_JUDGE_TEXT_REQUESTED_MAX_CHARACTERS
            ),
            "response_max_characters": COMPARISON_JUDGE_RESPONSE_MAX_CHARACTERS,
            "maximum_prompt_characters": COMPARISON_JUDGE_MAX_PROMPT_CHARACTERS,
            "request_policy": request_policy,
            "blind_fields": [
                "baseline_or_candidate",
                "candidate_revision",
                "variant_id",
                "treatment",
                "harness",
                "model",
                "deterministic_scores",
                "private_expected_values",
                "receipts",
                "internal_ids",
            ],
            "usage": usage,
        },
    )


def _validate_comparison_judge_request_binding(
    *,
    receipt: Mapping[str, Any],
    input_transform_receipt: Mapping[str, Any],
    required: bool,
) -> None:
    if not required:
        return
    if (
        receipt.get("provider_payload_sha256")
        != input_transform_receipt["provider_payload_sha256"]
        or receipt.get("input_transform_receipt_digest")
        != input_transform_receipt["receipt_digest"]
    ):
        raise ValueError("judge provider request did not bind the prepared input")


def _invoke_comparison_judge_request(
    *,
    judge_request: Callable[
        ...,
        tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    ]
    | None,
    evaluator: ComparisonEvaluatorV1,
    public_task: Mapping[str, Any],
    row: Mapping[str, Any],
    env: Mapping[str, str],
    prepared_payload: Mapping[str, Any],
    input_transform_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if judge_request is not None:
        # Preserve the established injection seam for offline/unit requesters.
        return judge_request(
            evaluator=evaluator,
            public_task=public_task,
            row=row,
            env=env,
        )
    response = _request_comparison_judge(
        evaluator=evaluator,
        public_task=public_task,
        row=row,
        env=env,
        prepared_payload=prepared_payload,
        input_transform_receipt=input_transform_receipt,
    )
    _validate_comparison_judge_request_binding(
        receipt=response[2],
        input_transform_receipt=input_transform_receipt,
        required=True,
    )
    return response


def _comparison_judge_request_policy(
    evaluator: ComparisonEvaluatorV1,
    env: Mapping[str, str],
    *,
    route: Any | None = None,
) -> dict[str, Any]:
    from fugue.bench.evaluations import (
        JUDGE_JSON_MAX_OUTPUT_TOKENS,
        JUDGE_JSON_MAX_RESPONSE_CHARACTERS,
        JUDGE_JSON_REQUEST_POLICY_SCHEMA_VERSION,
        bounded_judge_request_options,
    )
    from fugue.model_plane import resolve_model_route

    resolved_route = route or resolve_model_route(evaluator.profile, env)
    return {
        "schema_version": JUDGE_JSON_REQUEST_POLICY_SCHEMA_VERSION,
        "timeout_sec": (
            evaluator.timeout_sec or DEFAULT_COMPARISON_JUDGE_TIMEOUT_SEC
        ),
        "max_output_tokens": JUDGE_JSON_MAX_OUTPUT_TOKENS,
        "max_response_characters": JUDGE_JSON_MAX_RESPONSE_CHARACTERS,
        "structured_assistant_options": bounded_judge_request_options(resolved_route),
        "automatic_retries": 0,
    }


def _comparison_judge_failure_metadata(
    exc: Exception,
    *,
    fallback_stage: str,
) -> dict[str, Any]:
    """Return safe failure facts without persisting model output or private labels."""

    import httpx

    from fugue.bench.evaluations import JudgeResponseError

    if isinstance(exc, JudgeResponseError):
        return {
            "schema_version": 1,
            "stage": exc.stage,
            "code": exc.code,
            "message": exc.safe_message,
            "exception_type": "ValueError",
            "response_sha256": exc.response_sha256,
            "response_characters": exc.response_characters,
            "usage": dict(exc.usage),
        }
    if fallback_stage == "input_privacy":
        code = "input_privacy_rejected"
        message = "judge input failed the privacy gate"
    elif fallback_stage == "rubric_validation":
        code = "invalid_rubric_payload"
        message = "judge response failed strict rubric validation"
    elif isinstance(exc, httpx.TimeoutException):
        code = "provider_timeout"
        message = "judge provider request timed out"
    elif isinstance(exc, httpx.HTTPError):
        code = "provider_http_error"
        message = "judge provider request failed"
    else:
        code = "provider_request_failed"
        message = "judge provider request failed"
    return {
        "schema_version": 1,
        "stage": fallback_stage,
        "code": code,
        "message": message,
        "exception_type": type(exc).__name__,
    }


def _comparison_judge_evidence(
    row: Mapping[str, Any], fields: Sequence[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field_name in fields:
        if field_name == "tool_names":
            names: set[str] = set()
            for key in ("tool_calls", "mcp_tool_calls"):
                for item in row.get(key) or ():
                    if isinstance(item, str) and item:
                        names.add(item)
                    elif isinstance(item, Mapping):
                        name = str(
                            item.get("name")
                            or item.get("tool")
                            or item.get("tool_name")
                            or ""
                        )
                        if name:
                            names.add(name)
            explicit = row.get("mcp_tool_names") or ()
            if isinstance(explicit, str) and explicit:
                names.add(explicit)
            elif isinstance(explicit, Sequence):
                names.update(str(item) for item in explicit if str(item))
            result[field_name] = sorted(names)
        elif field_name in {
            "artifact_paths",
            "retrieved_paths",
            "inspected_paths",
            "changed_paths",
        }:
            values = row.get(field_name)
            status_name = f"{field_name}_status"
            status = row.get(status_name)
            if field_name in {"inspected_paths", "changed_paths"}:
                normalized_status = (
                    str(status)
                    if status in {"available", "unavailable"}
                    else "unavailable"
                )
                result[status_name] = normalized_status
                if normalized_status != "available":
                    result[field_name] = []
                    continue
            if isinstance(values, list):
                result[field_name] = [str(value)[:500] for value in values[:100]]
    return result


def _comparison_judge_payload(
    *,
    evaluator: ComparisonEvaluatorV1,
    public_task: Mapping[str, Any],
    row: Mapping[str, Any],
) -> dict[str, Any]:
    return comparison_judge_request_payload(
        public_task={
            "input": public_task.get("input"),
            "tags": public_task.get("tags") or [],
        },
        response=_comparison_output(row),
        permitted_evidence=_comparison_judge_evidence(
            row,
            evaluator.evidence,
        ),
        rubric=str(evaluator.rubric or ""),
        dimensions=evaluator.dimensions,
    )


def _comparison_judge_contains_exact_secret(
    value: Any,
    *,
    secrets: Sequence[str],
) -> bool:
    if isinstance(value, str):
        return any(secret and secret in value for secret in secrets)
    if isinstance(value, Mapping):
        return any(
            _comparison_judge_contains_exact_secret(name, secrets=secrets)
            or _comparison_judge_contains_exact_secret(item, secrets=secrets)
            for name, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return any(
            _comparison_judge_contains_exact_secret(item, secrets=secrets)
            for item in value
        )
    return False


def _comparison_judge_transformed_value_count(source: Any, target: Any) -> int:
    if source == target:
        return 0
    if isinstance(source, Mapping) and isinstance(target, Mapping):
        if set(source) != set(target):
            raise ValueError("judge input sanitization changed payload fields")
        return sum(
            _comparison_judge_transformed_value_count(source[key], target[key])
            for key in source
        )
    if (
        isinstance(source, Sequence)
        and not isinstance(source, str | bytes)
        and isinstance(target, Sequence)
        and not isinstance(target, str | bytes)
    ):
        if len(source) != len(target):
            raise ValueError("judge input sanitization changed payload length")
        return sum(
            _comparison_judge_transformed_value_count(before, after)
            for before, after in zip(source, target, strict=True)
        )
    if type(source) is not type(target):
        raise ValueError("judge input sanitization changed payload type")
    return 1


def _prepare_comparison_judge_input(
    *,
    evaluator: ComparisonEvaluatorV1,
    public_task: Mapping[str, Any],
    row: Mapping[str, Any],
    env: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if row.get("credential_leak") is True or row.get("private_label_leak") is True:
        raise ValueError(
            "judge input is ineligible because a privacy leak was detected"
        )
    payload = _comparison_judge_payload(
        evaluator=evaluator,
        public_task=public_task,
        row=row,
    )
    configured_secrets = secrets_from_env(env)
    if _comparison_judge_contains_exact_secret(
        payload,
        secrets=configured_secrets,
    ):
        raise ValueError("judge input contains an exact configured credential")
    sanitized = redact_value(payload, secrets=())
    if not isinstance(sanitized, dict):
        raise ValueError("judge input sanitization produced an invalid payload")
    if redact_value(sanitized, secrets=()) != sanitized:
        raise ValueError("judge input sanitization is not idempotent")
    if _comparison_judge_contains_exact_secret(
        sanitized,
        secrets=configured_secrets,
    ):
        raise ValueError("judge input sanitization retained a configured credential")
    transformed_value_count = _comparison_judge_transformed_value_count(
        payload,
        sanitized,
    )
    source_payload_sha256 = stable_digest(payload)
    provider_payload_sha256 = stable_digest(sanitized)
    unsigned_receipt = {
        "schema_version": 1,
        "status": "passed",
        "contract": "fugue-judge-input-sanitization-v1",
        "transform": "generic-credential-placeholder-redaction-v1",
        "source_payload_sha256": source_payload_sha256,
        "provider_payload_sha256": provider_payload_sha256,
        # Retain the established field as the exact provider-visible digest.
        "payload_sha256": provider_payload_sha256,
        "transformed": transformed_value_count > 0,
        "transformed_value_count": transformed_value_count,
        "exact_configured_secret_scan": "passed",
        "preexisting_leak_scan": "passed",
    }
    return sanitized, {
        **unsigned_receipt,
        "receipt_digest": stable_digest(unsigned_receipt),
    }


def _comparison_judge_output_privacy_receipt(
    *,
    payload: Mapping[str, Any],
    usage: Mapping[str, Any],
    env: Mapping[str, str],
) -> dict[str, Any]:
    canonical_payload = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if redact_value(payload, secrets=secrets_from_env(env)) != payload:
        from fugue.bench.evaluations import JudgeResponseError

        raise JudgeResponseError(
            stage="output_privacy",
            code="output_privacy_rejected",
            message="judge response failed the post-provider privacy scan",
            response_sha256=hashlib.sha256(canonical_payload.encode()).hexdigest(),
            response_characters=len(canonical_payload),
            usage=usage,
        )
    return {
        "schema_version": 1,
        "status": "passed",
        "contract": "fugue-redaction-v1",
        "payload_sha256": hashlib.sha256(canonical_payload.encode()).hexdigest(),
    }


def _comparison_output(row: Mapping[str, Any]) -> Any:
    for key in ("final_output", "answer", "agent_response"):
        if row.get(key) is not None:
            return row[key]
    return None


def _comparison_trial_output(row: Mapping[str, Any]) -> Any:
    raw_trial_dir = row.get("trial_dir")
    if isinstance(raw_trial_dir, str) and raw_trial_dir:
        trial_dir = Path(raw_trial_dir)
        if trial_dir.is_dir():
            answers = sorted(trial_dir.rglob("fugue-answer.md"))
            if answers:
                try:
                    value = answers[0].read_text(
                        encoding="utf-8", errors="replace"
                    ).strip()
                except OSError:
                    pass
                else:
                    if value:
                        return value[:16_000]
    return _comparison_output(row)


def _validate_comparison_judge_payload(
    evaluator: ComparisonEvaluatorV1, payload: Mapping[str, Any]
) -> dict[str, Any]:
    expected_fields = {
        "scores",
        "overall_assessment",
        "uncertainty",
        "missing_evidence",
        "rationale",
    }
    unknown = sorted(set(payload) - expected_fields)
    missing = sorted(expected_fields - set(payload))
    if unknown or missing:
        raise ValueError(
            "judge response fields do not match: "
            f"unknown={unknown}, missing={missing}"
        )
    expected = set(evaluator.dimensions)
    raw_scores = payload.get("scores")
    if not isinstance(raw_scores, Mapping) or set(raw_scores) != expected:
        raise ValueError("judge scores do not match the locked rubric dimensions")
    scores: dict[str, float] = {}
    for dimension, raw in raw_scores.items():
        if (
            not isinstance(raw, int | float)
            or isinstance(raw, bool)
            or not math.isfinite(float(raw))
            or not 0 <= float(raw) <= 1
        ):
            raise ValueError(f"judge score {dimension!r} must be between zero and one")
        scores[str(dimension)] = float(raw)
    raw_assessment = payload.get("overall_assessment")
    raw_rationale = payload.get("rationale")
    if not isinstance(raw_assessment, str):
        raise ValueError("judge overall_assessment must be text")
    if not isinstance(raw_rationale, str):
        raise ValueError("judge rationale must be text")
    assessment = raw_assessment.strip()
    rationale = raw_rationale.strip()
    uncertainty = payload.get("uncertainty")
    missing_evidence = payload.get("missing_evidence")
    hard_limit = COMPARISON_JUDGE_RESPONSE_MAX_CHARACTERS
    if not assessment or len(assessment) > hard_limit:
        raise ValueError(
            f"judge overall_assessment must be 1..{hard_limit} characters"
        )
    if not rationale or len(rationale) > hard_limit:
        raise ValueError(f"judge rationale must be 1..{hard_limit} characters")
    if (
        not isinstance(uncertainty, int | float)
        or isinstance(uncertainty, bool)
        or not 0 <= float(uncertainty) <= 1
    ):
        raise ValueError("judge uncertainty must be between zero and one")
    if not isinstance(missing_evidence, bool):
        raise ValueError("judge missing_evidence must be boolean")
    return {
        "scores": scores,
        "overall_assessment": assessment,
        "uncertainty": float(uncertainty),
        "rationale": rationale,
        "missing_evidence": missing_evidence,
    }


def _comparison_mechanism(
    row: Mapping[str, Any],
    *,
    expected: Any,
    passed: bool,
    expected_skill_ids: tuple[str, ...],
) -> dict[str, str]:
    skill_applicable = bool(expected_skill_ids)
    assigned = {
        str(value)
        for value in (
            row.get("skills_assigned") or row.get("skill_ids") or []
        )
    }
    registered = {str(value) for value in row.get("skills_registered") or []}
    registration_status = str(row.get("skill_registration_status") or "")
    invocation = row.get("skill_invocation_evidence") or {}
    invocation_status = (
        str(invocation.get("status") or "")
        if isinstance(invocation, Mapping)
        else ""
    )
    invoked = (
        {str(value) for value in invocation.get("skills_invoked") or []}
        if isinstance(invocation, Mapping)
        else set()
    )
    expected_skills = set(expected_skill_ids)
    source = (
        str(expected.get("source") or expected.get("source_document") or "")
        if isinstance(expected, Mapping)
        else ""
    )
    returned_paths = _row_paths(
        row,
        "context_result_paths",
        "retrieved_paths",
        "search_result_paths",
    )
    opened_paths = _row_paths(
        row,
        "inspected_paths",
        "opened_paths",
        "context_result_opened_paths",
    )
    source_returned = _path_observed(source, returned_paths)
    source_opened = _path_observed(source, opened_paths)
    output = (
        row.get("final_output")
        if row.get("final_output") is not None
        else row.get("answer")
    )
    parsed = json.loads(output) if isinstance(output, str) and _is_json(output) else output
    source_used = bool(
        source
        and source_opened
        and isinstance(parsed, Mapping)
        and (
            parsed.get("source") == source
            or parsed.get("source_document") == source
        )
    )
    return {
        "skill_assigned": _mechanism_state(
            applicable=skill_applicable,
            available=bool(assigned) or not skill_applicable,
            reached=expected_skills <= assigned,
        ),
        "skill_registered": _mechanism_state(
            applicable=skill_applicable,
            available=registration_status
            not in {"", "unavailable"}
            or bool(registered),
            reached=(
                registration_status == "registered"
                and (not registered or expected_skills <= registered)
            ),
        ),
        "skill_invoked": _mechanism_state(
            applicable=skill_applicable,
            available=invocation_status
            not in {"", "unavailable"},
            reached=invocation_status == "observed"
            and expected_skills <= invoked,
        ),
        "relevant_source_returned": _mechanism_state(
            applicable=bool(returned_paths),
            available=bool(returned_paths),
            reached=source_returned,
        ),
        "relevant_source_opened": _mechanism_state(
            applicable=bool(source),
            available=bool(opened_paths),
            reached=source_opened,
        ),
        "relevant_source_used": _mechanism_state(
            applicable=bool(source),
            available=bool(opened_paths) and output is not None,
            reached=source_used,
        ),
        "task_passed": "observed" if passed else "not_observed",
    }


def _mechanism_state(
    *, applicable: bool, available: bool, reached: bool
) -> str:
    if not applicable:
        return "not_applicable"
    if not available:
        return "unavailable"
    return "observed" if reached else "not_observed"


def _row_paths(row: Mapping[str, Any], *keys: str) -> set[str]:
    result: set[str] = set()
    for key in keys:
        value = row.get(key) or []
        if isinstance(value, str):
            result.add(value)
        elif isinstance(value, list | tuple | set):
            result.update(str(item) for item in value)
    return result


def _path_observed(expected: str, observed: set[str]) -> bool:
    return bool(
        expected
        and any(
            value == expected or PurePosixPath(value).name == PurePosixPath(expected).name
            for value in observed
        )
    )


def _deterministic_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in ("baseline", "candidate"):
        selected = [
            row for row in rows if str(row.get("variant_id") or "") == variant
        ]
        dimensions = sorted(
            {
                str(key)
                for row in selected
                for key in (
                    row.get("comparison_deterministic_scores") or {}
                )
            }
        )
        result[variant] = {
            "passed": sum(row.get("pass") is True for row in selected),
            "evaluated": sum(
                row.get("comparison_evaluation_status") == "scored"
                for row in selected
            ),
            "dimensions": {
                dimension: {
                    "passed": sum(
                        _dimension_passed(
                            (row.get("comparison_deterministic_scores") or {}).get(
                                dimension
                            )
                        )
                        for row in selected
                    ),
                    "evaluated": sum(
                        dimension
                        in (row.get("comparison_deterministic_scores") or {})
                        for row in selected
                    ),
                    "mean": _numeric_summary(
                        [
                            (
                                float(value)
                                if isinstance(value, bool)
                                else value
                            )
                            for row in selected
                            if (
                                value := (
                                    row.get("comparison_deterministic_scores") or {}
                                ).get(dimension)
                            )
                            is not None
                        ]
                    )["mean"],
                }
                for dimension in dimensions
            },
        }
    return result


def _dimension_passed(value: Any) -> bool:
    return value is True or (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and float(value) == 1.0
    )


def _judge_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    qualifications: dict[str, dict[str, Any]] = {}
    for row in rows:
        for judge_id, raw in _mapping_or_empty(row.get("comparison_judges")).items():
            value = _mapping_or_empty(raw)
            qualification = value.get("qualification")
            if not isinstance(qualification, Mapping):
                continue
            normalized = dict(qualification)
            existing = qualifications.get(str(judge_id))
            if existing is not None and existing != normalized:
                raise ValueError(
                    f"judge {judge_id!r} qualification changed across attempts"
                )
            qualifications[str(judge_id)] = normalized
    judges = [qualifications[judge_id] for judge_id in sorted(qualifications)]
    claim_status = (
        "calibrated"
        if judges
        and all(
            _mapping_or_empty(item.get("calibration")).get("passed") is True
            for item in judges
        )
        else "advisory_uncalibrated"
    )
    scored = [
        row
        for row in rows
        if isinstance(row.get("comparison_judge_scores"), Mapping)
    ]
    if not scored:
        if any(row.get("comparison_judge_status") for row in rows):
            return {
                "status": "unavailable",
                "claim_status": claim_status,
                "judges": judges,
                "by_variant": {"baseline": {}, "candidate": {}},
                "unavailable_attempts": sum(
                    row.get("comparison_judge_status") == "unavailable"
                    for row in rows
                ),
            }
        return {
            "status": "not_used",
            "claim_status": "not_applicable",
            "judges": [],
            "by_variant": {"baseline": {}, "candidate": {}},
            "unavailable_attempts": 0,
        }
    dimensions = sorted(
        {
            str(key)
            for row in scored
            for key in (row.get("comparison_judge_scores") or {})
        }
    )
    by_variant: dict[str, Any] = {}
    for variant in ("baseline", "candidate"):
        selected = [
            row for row in scored if str(row.get("variant_id") or "") == variant
        ]
        by_variant[variant] = {
            dimension: _numeric_summary(
                [
                    (row.get("comparison_judge_scores") or {}).get(dimension)
                    for row in selected
                ]
            )
            for dimension in dimensions
        }
    return {
        "status": "scored",
        "claim_status": claim_status,
        "judges": judges,
        "by_variant": by_variant,
        "unavailable_attempts": sum(
            row.get("comparison_judge_status") == "unavailable"
            for row in rows
        ),
    }


def _numeric_summary(values: Sequence[Any]) -> dict[str, Any]:
    numeric = [
        float(value)
        for value in values
        if isinstance(value, int | float) and not isinstance(value, bool)
    ]
    return {
        "evaluated": len(numeric),
        "mean": round(sum(numeric) / len(numeric), 6) if numeric else None,
    }


def _mechanism_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stages = sorted(
        {
            str(key)
            for row in rows
            for key in (row.get("comparison_mechanism") or {})
        }
    )
    return {
        stage: {
            variant: {
                "observed": sum(
                    (row.get("comparison_mechanism") or {}).get(stage)
                    == "observed"
                    for row in rows
                    if str(row.get("variant_id") or "") == variant
                ),
                "applicable": sum(
                    (row.get("comparison_mechanism") or {}).get(stage)
                    not in {None, "not_applicable"}
                    for row in rows
                    if str(row.get("variant_id") or "") == variant
                ),
                "unavailable": sum(
                    (row.get("comparison_mechanism") or {}).get(stage)
                    == "unavailable"
                    for row in rows
                    if str(row.get("variant_id") or "") == variant
                ),
            }
            for variant in ("baseline", "candidate")
        }
        for stage in stages
    }


def _comparison_evidence_links(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], ...]:
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    labels = {
        "evaluation_root": "Evaluation root",
        "prediction_and_score": "Prediction and score",
        "prediction": "Prediction",
        "agent_root": "Agent root",
        "dataset": "Dataset",
    }
    for row in rows:
        project = str(row.get("trace_project") or "")
        task = str(row.get("task_id") or row.get("task_name") or "task")
        variant = str(row.get("variant_id") or "candidate")
        overview = _weave_project_url(project)
        if overview and overview not in seen:
            seen.add(overview)
            result.append({"label": "Weave project", "url": overview})
        for link in _attempt_evidence_links(row):
            if link.status != "resolved":
                continue
            value = str(link.url or "")
            kind = str(link.kind)
            if value in seen:
                continue
            seen.add(value)
            result.append(
                {
                    "label": f"{labels.get(kind, kind)} — {task} — {variant}",
                    "url": value,
                }
            )
    return tuple(result)


def _weave_project_url(project: str) -> str:
    parts = project.split("/")
    if len(parts) != 2:
        return ""
    try:
        entity, project_id = (
            validate_id(parts[0], kind="W&B entity"),
            validate_id(parts[1], kind="W&B project"),
        )
    except ValueError:
        return ""
    return f"https://wandb.ai/{entity}/{project_id}/weave"


def _weave_call_url(
    project: str,
    call_id: str,
    *,
    app_base_url: str = "https://wandb.ai",
) -> str:
    overview = _weave_project_url(project)
    if not overview or not call_id or len(call_id) > 200:
        return ""
    entity, project_id = project.split("/", 1)
    return (
        f"{app_base_url.rstrip('/')}/{entity}/{project_id}/weave/calls/"
        f"{urllib.parse.quote(call_id, safe='')}"
    )


def _weave_call_ref(project: str, call_id: str) -> str:
    if project.count("/") != 1 or not call_id or len(call_id) > 200:
        return ""
    entity, project_id = project.split("/", 1)
    if not entity or not project_id:
        return ""
    return f"weave:///{entity}/{project_id}/call/{call_id}"


def _weave_object_url_from_ref(
    project: str,
    ref: str,
    *,
    app_base_url: str,
) -> str:
    if not _safe_application_base_url(
        app_base_url,
        label="Weave application origin",
    ):
        return ""
    if not ref or project.count("/") != 1:
        return ""
    parsed = urllib.parse.urlsplit(ref)
    parts = parsed.path.strip("/").split("/")
    if (
        parsed.scheme != "weave"
        or len(parts) != 4
        or "/".join(parts[:2]) != project
        or parts[2] != "object"
        or ":" not in parts[3]
    ):
        return ""
    name, digest = parts[3].rsplit(":", 1)
    if not name or not digest:
        return ""
    entity, project_id = project.split("/", 1)
    return (
        f"{app_base_url.rstrip('/')}/"
        f"{urllib.parse.quote(entity, safe='')}/"
        f"{urllib.parse.quote(project_id, safe='')}/weave/objects/"
        f"{urllib.parse.quote(name, safe='')}/versions/"
        f"{urllib.parse.quote(digest, safe='')}"
    )


def _safe_application_base_url(value: Any, *, label: str) -> str | None:
    text = str(value or "").strip().rstrip("/")
    if not text:
        return None
    parsed = urllib.parse.urlsplit(text)
    safe_scheme = parsed.scheme == "https" or (
        parsed.scheme == "http"
        and parsed.hostname in {"127.0.0.1", "localhost"}
    )
    if (
        not safe_scheme
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            f"{label} must be credential-free HTTPS or loopback HTTP"
        )
    return text


def _study_console_backlink(
    base_url: str | None,
    *,
    research_id: str | None,
    study_id: str,
) -> str | None:
    if base_url is None or research_id is None:
        return None
    return (
        f"{base_url}/?research_id="
        f"{urllib.parse.quote(research_id, safe='')}&study_id="
        f"{urllib.parse.quote(study_id, safe='')}"
    )


def _operational_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    execution: dict[str, int] = {}
    evidence: dict[str, int] = {}
    observed_cost = 0.0
    cost_rows = 0
    accounted_cost = 0.0
    accounted_cost_rows = 0
    latency_ms = 0.0
    latency_rows = 0
    input_tokens = 0
    output_tokens = 0
    usage_rows = 0
    infrastructure_failures = 0
    agent_timeouts = 0
    wandb_rows = 0
    wandb_eligible = 0
    evidence_projects: set[str] = set()
    mcp_tool_usage: dict[str, dict[str, int]] = {}
    agent_trajectory_tool_activity: dict[str, dict[str, Any]] = {}
    for row in rows:
        trace_project = str(row.get("trace_project") or "")
        if _weave_project_url(trace_project):
            evidence_projects.add(trace_project)
        status = _terminal_execution_status(row) or str(
            row.get("status")
            or row.get("execution_status")
            or "unknown"
        )
        execution[status] = execution.get(status, 0) + 1
        evidence_status = str(
            row.get("trace_link_status")
            or row.get("evidence_status")
            or "unknown"
        )
        evidence[evidence_status] = evidence.get(evidence_status, 0) + 1
        if _agent_timeout_outcome(row):
            agent_timeouts += 1
        elif (
            status in {"failed", "error", "infrastructure_failed"}
            or _row_runtime_outcome(row) == "timed_out"
            or row.get("exception_class")
        ):
            infrastructure_failures += 1
        if "wandb_serverless_eligible" in row:
            wandb_rows += 1
            wandb_eligible += row.get("wandb_serverless_eligible") is True
        variant = str(row.get("variant_id") or "unknown")
        if row.get("agent_trajectory_tool_activity_status") == "available":
            trajectory_count = row.get("agent_trajectory_tool_call_count")
            trajectory_names = row.get("agent_trajectory_tool_names") or {}
            if (
                isinstance(trajectory_count, int)
                and not isinstance(trajectory_count, bool)
                and trajectory_count >= 0
                and isinstance(trajectory_names, Mapping)
            ):
                activity = agent_trajectory_tool_activity.setdefault(
                    variant,
                    {"calls": 0, "rows": 0, "tools": {}},
                )
                activity["calls"] += trajectory_count
                activity["rows"] += 1
                activity_tools = activity["tools"]
                for raw_name, raw_count in trajectory_names.items():
                    if (
                        not isinstance(raw_count, int)
                        or isinstance(raw_count, bool)
                        or raw_count <= 0
                    ):
                        continue
                    name = str(raw_name)
                    if name:
                        activity_tools[name] = (
                            activity_tools.get(name, 0) + raw_count
                        )
        tool_counts = row.get("weave_tool_names") or {}
        if isinstance(tool_counts, Mapping):
            for raw_name, raw_count in tool_counts.items():
                name = str(raw_name)
                if not name.startswith("mcp__"):
                    continue
                public_name = name.split("__", 2)[-1]
                count = (
                    int(raw_count)
                    if isinstance(raw_count, int) and not isinstance(raw_count, bool)
                    else 0
                )
                usage = mcp_tool_usage.setdefault(variant, {})
                usage[public_name] = usage.get(public_name, 0) + count
        cost = _observed_attempt_cost(row)[0]
        if isinstance(cost, int | float) and not isinstance(cost, bool):
            observed_cost += float(cost)
            cost_rows += 1
        row_accounted_cost = _row_number(row, "accounted_cost_usd")
        if row_accounted_cost is not None:
            accounted_cost += row_accounted_cost
            accounted_cost_rows += 1
        latency = _row_latency_sec(row)
        if latency is not None:
            latency_ms += latency * 1000
            latency_rows += 1
        row_input, row_output = _row_token_usage(row)
        if row_input is not None and row_output is not None:
            input_tokens += int(row_input)
            output_tokens += int(row_output)
            usage_rows += 1
    result = {
        "execution_states": dict(sorted(execution.items())),
        "evidence_states": dict(sorted(evidence.items())),
        "infrastructure_failures": infrastructure_failures,
        "observed_cost_usd": round(observed_cost, 6) if cost_rows else None,
        "cost_rows": cost_rows,
        "accounted_cost_usd": (
            round(accounted_cost, 6) if accounted_cost_rows else None
        ),
        "accounted_cost_rows": accounted_cost_rows,
        "latency_ms": round(latency_ms, 3) if latency_rows else None,
        "latency_rows": latency_rows,
        "input_tokens": input_tokens if usage_rows else None,
        "output_tokens": output_tokens if usage_rows else None,
        "usage_rows": usage_rows,
        "evidence_projects": sorted(evidence_projects),
        "mcp_tool_usage": {
            variant: dict(sorted(tool_counts.items()))
            for variant, tool_counts in sorted(mcp_tool_usage.items())
        },
        "agent_trajectory_tool_activity": {
            variant: {
                "calls": activity["calls"],
                "rows": activity["rows"],
                "tools": dict(sorted(activity["tools"].items())),
            }
            for variant, activity in sorted(
                agent_trajectory_tool_activity.items()
            )
        },
    }
    if wandb_rows:
        result["wandb_serverless"] = {
            "rows": wandb_rows,
            "eligible": wandb_eligible,
            "ineligible": wandb_rows - wandb_eligible,
        }
    if agent_timeouts:
        result["agent_timeouts"] = agent_timeouts
    return result


def _comparison_scorer_names(spec: ComparisonSpecV1) -> tuple[str, ...]:
    names: set[str] = set()
    for evaluator in spec.evaluators:
        if evaluator.type == "llm_judge":
            names.update(
                f"comparison.judge.{evaluator.id}.{dimension}"
                for dimension in evaluator.dimensions
            )
        elif evaluator.scorer:
            names.update(
                f"comparison.deterministic.{evaluator.id}.{dimension}"
                for dimension in evaluator.dimensions
            )
        else:
            names.update(
                f"comparison.deterministic.{check}"
                for check in evaluator.checks
            )
    return tuple(sorted(names))


def _project_comparison_start(
    *,
    spec: ComparisonSpecV1,
    preview: ComparisonPreviewV1,
    repo_root: Path,
    destination: Path,
    publish_research: bool,
) -> tuple[Path, dict[str, Any] | None]:
    publication_path = destination / "research-publication.json"
    if not publish_research or not spec.execution.research_id:
        return publication_path, None
    try:
        from fugue.research.comparisons import project_direct_comparison_start

        projection = project_direct_comparison_start(
            repo_root,
            spec.execution.research_id,
            preview,
        )
    except Exception as exc:
        failed = {
            "schema_version": 1,
            "research_id": spec.execution.research_id,
            "publication_complete": False,
            "status": "publication_incomplete",
            "stage": "start",
            "error_type": type(exc).__name__,
        }
        atomic_write_json(publication_path, failed)
        raise ComparisonPublicationError(
            stage="start",
            research_id=spec.execution.research_id,
            receipt_path=publication_path,
            error_type=type(exc).__name__,
        ) from None
    started = {
        **projection,
        "publication_complete": False,
        "status": "started",
        "stage": "start",
    }
    atomic_write_json(publication_path, started)
    return publication_path, started


def _project_direct_comparison_failure(
    *,
    context: _DirectComparisonExecutionContext,
    preview: ComparisonPreviewV1,
    repo_root: Path,
    error: BaseException,
) -> None:
    spec = context.spec
    publication_path = context.publication_path
    if spec is None or publication_path is None or not spec.execution.research_id:
        return
    run_status = context.run_status
    completed_cells = context.completed_cells
    if context.run_id and run_status is None:
        from fugue.bench.execution import read_run_manifest

        manifest = read_run_manifest(
            repo_root / ".fugue" / "runtime" / context.run_id
        ) or {}
        run_status = str(manifest.get("status") or "") or None
        completed_cells = sum(
            int(manifest.get(key) or 0)
            for key in ("passed_cells", "failed_cells")
        )
    terminal_status = (
        "cancelled"
        if run_status in {"cancelled", "interrupted"}
        or isinstance(error, (KeyboardInterrupt, SystemExit))
        else "failed"
    )
    try:
        from fugue.research.comparisons import (
            project_direct_comparison_failure,
        )

        projected = project_direct_comparison_failure(
            repo_root,
            spec.execution.research_id,
            preview,
            error=error,
            env=context.environment,
            run_id=context.run_id,
            status=terminal_status,
            completed_cells=completed_cells,
        )
        receipt = {
            **dict(context.start_projection or {}),
            **projected,
            "publication_complete": True,
            "status": terminal_status,
            "stage": "failure",
        }
    except Exception as projection_error:
        receipt = {
            **dict(context.start_projection or {}),
            "schema_version": 1,
            "research_id": spec.execution.research_id,
            "comparison_id": spec.id,
            "preview_digest": preview.preview_digest,
            "run_id": context.run_id,
            "publication_complete": False,
            "status": "publication_incomplete",
            "stage": "failure",
            "error_type": type(projection_error).__name__,
        }
    atomic_write_json(publication_path, receipt)


def _score_and_bind_exported_comparison_rows(
    *,
    spec: ComparisonSpecV1,
    rows: Sequence[Mapping[str, Any]],
    repo_root: Path,
    env: Mapping[str, str],
    approved_comparison: Mapping[str, Any],
    source_pre_run_drift: EvidenceDriftCheckV1 | None,
    source_checkpoint_drift: EvidenceDriftCheckV1 | None,
    source_post_run_drift: EvidenceDriftCheckV1 | None,
    release_note_coverage: Sequence[Mapping[str, Any]],
    infrastructure_receipt: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for raw_row in rows:
        row = dict(raw_row)
        if row.get("comparison_evaluation_status") not in {
            "scored",
            "unavailable",
        }:
            row = score_comparison_rows(
                spec,
                [row],
                repo_root=repo_root,
                env=env,
                approved_comparison=approved_comparison,
            )[0]
        scored.append(row)
    for row in scored:
        for field_name, check in (
            ("source_pre_run_drift", source_pre_run_drift),
            ("source_checkpoint_drift", source_checkpoint_drift),
            ("source_post_run_drift", source_post_run_drift),
        ):
            if check is not None:
                row[field_name] = check.to_dict()
        if release_note_coverage:
            row["release_note_coverage"] = [
                dict(item) for item in release_note_coverage
            ]
    if infrastructure_receipt is not None:
        conformance = _mapping_or_empty(
            infrastructure_receipt.get("infrastructure_conformance")
        )
        gate_statuses = {
            str(item["id"]): str(item["status"])
            for item in conformance.get("gates") or ()
            if isinstance(item, Mapping)
        }
        for row in scored:
            row["infrastructure_conformance_complete"] = (
                conformance.get("complete") is True
            )
            row["infrastructure_gate_statuses"] = gate_statuses
            row["infrastructure_receipt_digest"] = str(
                infrastructure_receipt["receipt_digest"]
            )
    return scored


def _publish_comparison_result(
    *,
    spec: ComparisonSpecV1,
    result: ComparisonResult,
    json_path: Path,
    markdown_path: Path,
    destination: Path,
    publication_path: Path,
    projection: Mapping[str, Any] | None,
    publish_research: bool,
    repo_root: Path,
) -> None:
    publication_error: ComparisonPublicationError | None = None
    terminal_projection = dict(projection or {})
    if publish_research and spec.execution.research_id:
        try:
            from fugue.research.comparisons import (
                project_direct_comparison_result,
            )

            projected = project_direct_comparison_result(
                repo_root,
                spec.execution.research_id,
                result,
                json_path,
            )
            terminal_projection = {
                **terminal_projection,
                **projected,
                "publication_complete": True,
                "status": "projected",
                "stage": "result",
            }
        except Exception as exc:
            terminal_projection = {
                **terminal_projection,
                "schema_version": 1,
                "research_id": spec.execution.research_id,
                "publication_complete": False,
                "status": "publication_incomplete",
                "stage": "result",
                "error_type": type(exc).__name__,
                "result_digest": result.result_digest,
                "result": json_path.relative_to(repo_root).as_posix(),
            }
            publication_error = ComparisonPublicationError(
                stage="result",
                research_id=spec.execution.research_id,
                receipt_path=publication_path,
                error_type=type(exc).__name__,
                result=result,
                result_path=json_path,
                markdown_path=markdown_path,
            )
        atomic_write_json(publication_path, terminal_projection)
    atomic_write_json(
        repo_root / COMPARISON_RESULT_ROOT / "latest.json",
        {
            "comparison_id": spec.id,
            "preview_digest": result.preview_digest,
            "evidence_project": spec.execution.evidence_project,
            "result": json_path.relative_to(repo_root).as_posix(),
            "markdown": markdown_path.relative_to(repo_root).as_posix(),
            **(
                {
                    "research_publication": {
                        "status": str(
                            terminal_projection.get("status")
                            or "not_declared"
                        ),
                        "complete": bool(
                            terminal_projection.get(
                                "publication_complete"
                            )
                        ),
                        "receipt": publication_path.relative_to(
                            repo_root
                        ).as_posix(),
                    }
                }
                if spec.execution.research_id
                else {}
            ),
        },
    )
    if publication_error is not None:
        raise publication_error


def _comparison_run_exportability_issue(run_summary: Any) -> str | None:
    """Reject orchestration failures while retaining terminal task failures.

    A failed Agent task is comparative evidence and must reach canonical
    scoring. Cancellation, interruption, pending work, or failed evidence
    publication are run-level failures and cannot be analyzed as a completed
    comparison.
    """

    if run_summary is None:
        return None
    if run_summary.cancelled:
        return f"status={run_summary.status}, cancelled_cells={run_summary.cancelled}"
    if run_summary.interrupted:
        return (
            f"status={run_summary.status}, "
            f"interrupted_cells={run_summary.interrupted}"
        )
    if run_summary.pending:
        return f"status={run_summary.status}, pending_cells={run_summary.pending}"
    if run_summary.evaluation_failures:
        return (
            f"status={run_summary.status}, "
            f"evaluation_failures={len(run_summary.evaluation_failures)}"
        )
    if run_summary.status == "passed":
        return None
    if run_summary.observability_status not in {None, "passed"}:
        return (
            f"status={run_summary.status}, "
            f"observability_status={run_summary.observability_status}"
        )
    if run_summary.status == "failed" and run_summary.failed:
        return None
    return f"status={run_summary.status}, no terminal task failure to analyze"


def execute_comparison(
    preview: ComparisonPreviewV1,
    *,
    approval_digest: str,
    repo_root: Path,
    env_file: Path | None = None,
    fetch_weave: bool = True,
    publish_research: bool = True,
    run_id: str | None = None,
) -> tuple[ComparisonResult, Path, Path]:
    """Execute one approved comparison.

    Direct CLI execution owns Research publication by default. The governed
    Research worker disables this path because its control service already
    owns the start, failure, and terminal-result projections.
    """
    from fugue.bench.execution import new_run_id

    selected_run_id = validate_id(
        run_id or new_run_id(),
        kind="comparison execution instance id",
    )
    context = _DirectComparisonExecutionContext(run_id=selected_run_id)
    try:
        return _execute_comparison(
            preview,
            approval_digest=approval_digest,
            repo_root=repo_root,
            env_file=env_file,
            fetch_weave=fetch_weave,
            publish_research=publish_research,
            projection_context=context,
            run_id=selected_run_id,
        )
    except BaseException as exc:
        if context.started and not context.canonical_result_published:
            try:
                _project_direct_comparison_failure(
                    context=context,
                    preview=preview,
                    repo_root=repo_root,
                    error=exc,
                )
            except Exception as projection_error:  # pragma: no cover - disk failure
                exc.add_note(
                    "direct comparison failure projection also failed: "
                    f"{type(projection_error).__name__}"
                )
        raise


def _execute_comparison(
    preview: ComparisonPreviewV1,
    *,
    approval_digest: str,
    repo_root: Path,
    env_file: Path | None,
    fetch_weave: bool,
    publish_research: bool,
    projection_context: _DirectComparisonExecutionContext,
    run_id: str,
) -> tuple[ComparisonResult, Path, Path]:
    _verify_artifact(preview.to_dict(), "preview_digest", "comparison preview")
    spec = comparison_from_dict(
        preview.comparison,
        repo_root=repo_root,
        source=repo_root,
    )
    projection_context.spec = spec
    service = OperatorService(repo_root, env_file)
    projection_context.environment = dict(service.env)
    current = preview_comparison(
        spec,
        repo_root=repo_root,
        operator=service,
    )
    if current.preview_digest != preview.preview_digest:
        raise ValueError(
            "comparison preparation or inputs changed after preview; "
            "prepare and approve a new exact preview"
        )
    if current.readiness["status"] not in {"ready", "needs_review"}:
        raise ValueError(
            "comparison is no longer execution-ready; prepare and preview it again"
        )
    infrastructure_receipt = _bound_execution_infrastructure_receipt(
        spec,
        readiness=current.readiness,
        repo_root=repo_root,
    )
    release_note_coverage = _bound_v3_release_note_coverage(
        spec,
        readiness=current.readiness,
        repo_root=repo_root,
    )
    source_pre_run_drift = _verify_v3_source_drift(
        spec,
        readiness=current.readiness,
        repo_root=repo_root,
        env=service.env,
    )
    if (
        source_pre_run_drift is not None
        and source_pre_run_drift.status != "matched"
    ):
        raise RuntimeError(
            "immutable source evidence did not match before execution; no "
            "comparison cells were launched"
        )
    claimed_approval = None
    if spec.execution.approval_required:
        if not approval_digest:
            raise ValueError("comparison execution requires an approval digest")
        claimed_approval = claim_comparison_approval(
            preview,
            approval_digest=approval_digest,
            repo_root=repo_root,
            execution_instance_id=run_id,
        )
    experiment, request = materialize_comparison(
        preview,
        repo_root=repo_root,
        operator=service,
        approval_digest=approval_digest,
        approval_input_bindings=(
            claimed_approval.input_bindings if claimed_approval is not None else {}
        ),
    )
    destination = repo_root / COMPARISON_RESULT_ROOT / preview.preview_digest
    research_publication_path, research_projection = (
        _project_comparison_start(
            spec=spec,
            preview=preview,
            repo_root=repo_root,
            destination=destination,
            publish_research=publish_research,
        )
    )
    projection_context.publication_path = research_publication_path
    projection_context.start_projection = (
        dict(research_projection) if research_projection is not None else None
    )
    projection_context.started = bool(
        publish_research and spec.execution.research_id and research_projection
    )
    approved_execution_lock = _result_approved_comparison_lock(
        destination,
        supplied=request.approved_comparison,
    )
    if approved_execution_lock is None:  # pragma: no cover - supplied above
        raise RuntimeError("approved comparison result lock was not persisted")

    approved_inputs = _verified_approved_inputs(
        approved_execution_lock,
        repo_root=repo_root,
    )
    _require_judge_execution_calibrations(
        spec,
        repo_root=repo_root,
        approved_inputs=approved_inputs,
    )
    evaluated_cells = 0
    source_checkpoint_drift: EvidenceDriftCheckV1 | None = None
    runtime_budget = _ComparisonRuntimeBudget(
        max_cost_usd=spec.execution.max_cost_usd,
        reserve_per_attempt_usd=_comparison_reserved_cost_per_attempt(spec),
        total_cells=int(current.readiness["estimated_cells"]),
    )

    def evaluate_attempt(row: dict[str, Any]) -> None:
        nonlocal evaluated_cells, source_checkpoint_drift
        try:
            evaluation_row = dict(row)
            evaluation_row["final_output"] = _comparison_trial_output(row)
            scored = score_comparison_rows(
                spec,
                [evaluation_row],
                repo_root=repo_root,
                env=service.env,
                approved_comparison=approved_execution_lock,
            )[0]
            scored.pop("final_output", None)
            row.update(scored)
            _require_checkpoint_judges(
                spec,
                row,
                checkpoint_index=evaluated_cells,
            )
            if source_pre_run_drift is not None:
                row["source_pre_run_drift"] = (
                    source_pre_run_drift.to_dict()
                )
            evaluated_cells += 1
            if (
                source_pre_run_drift is not None
                and source_checkpoint_drift is None
                and evaluated_cells
                >= max(1, spec.execution.evidence_checkpoint_cells)
            ):
                source_checkpoint_drift = _verify_v3_source_drift(
                    spec,
                    readiness=current.readiness,
                    repo_root=repo_root,
                    env=service.env,
                )
                if (
                    source_checkpoint_drift is None
                    or source_checkpoint_drift.status != "matched"
                ):
                    raise RuntimeError(
                        "immutable source evidence changed at the first-cell "
                        "checkpoint; remaining cells were cancelled"
                    )
            if source_checkpoint_drift is not None:
                row["source_checkpoint_drift"] = (
                    source_checkpoint_drift.to_dict()
                )
            runtime_budget.observe(row)
        except Exception:
            # Host evaluator exceptions are normalized into evidence rows by the
            # live publisher. The shared event is what stops queued cells.
            runtime_budget.cancellation_event.set()
            raise

    run_summary = service.execute_run(
        request,
        run_id=run_id,
        experiment=experiment,
        cancellation_event=runtime_budget.cancellation_event,
        cancellation_origin="internal",
        host_evaluator=evaluate_attempt,
        host_scorer_names=_comparison_scorer_names(spec),
    )
    if run_summary is not None:
        projection_context.run_status = str(run_summary.status or "") or None
        projection_context.completed_cells = int(run_summary.passed) + int(
            run_summary.failed
        )
    if runtime_budget.failure_reason:
        raise RuntimeError(runtime_budget.failure_reason)
    run_issue = _comparison_run_exportability_issue(run_summary)
    if run_issue is not None:
        raise RuntimeError(
            "comparison execution did not pass its required cell/evidence "
            f"gates (run={run_id}, {run_issue})"
        )
    source_post_run_drift = _verify_v3_source_drift(
        spec,
        readiness=current.readiness,
        repo_root=repo_root,
        env=service.env,
    )
    export_path = (
        repo_root / COMPARISON_RESULT_ROOT / preview.preview_digest / "attempts.jsonl"
    )
    summary = service.export_run(
        run_id,
        out=export_path,
        fetch_weave=fetch_weave,
        to_weave=False,
    )
    rows = _read_jsonl(summary.path, "comparison attempt rows")
    _apply_harbor_conformance(rows, repo_root=repo_root, run_id=run_id)
    scored = _score_and_bind_exported_comparison_rows(
        spec=spec,
        rows=rows,
        repo_root=repo_root,
        env=service.env,
        approved_comparison=approved_execution_lock,
        source_pre_run_drift=source_pre_run_drift,
        source_checkpoint_drift=source_checkpoint_drift,
        source_post_run_drift=source_post_run_drift,
        release_note_coverage=release_note_coverage,
        infrastructure_receipt=infrastructure_receipt,
    )
    draft_result = analyze_comparison_rows(
        comparison_id=spec.id,
        preview_digest=preview.preview_digest,
        rows=scored,
        source=run_id,
        expected_evidence_project=trace_project_slug(
            _comparison_evidence_environment(spec, service.env)
        ),
        approved_comparison=approved_execution_lock,
        decision_policy=spec.decision_policy,
        expected_source_evidence_project=(
            spec.execution.source_evidence_project
        ),
        result_schema_version=3 if spec.schema_version >= 3 else 2,
        study_intent=(
            "mcp_release_qualification"
            if spec.decision_policy is not None
            else "candidate_comparison"
        ),
        release_note_coverage=release_note_coverage,
        supersedes=spec.supersedes,
    )
    publication_payloads: dict[str, Any] = {
        "result": draft_result.to_dict(),
    }
    if publish_research and spec.execution.research_id:
        from fugue.research.experiment_views import (
            build_comparison_evaluation_view,
        )

        publication_payloads["study"] = build_comparison_evaluation_view(
            draft_result.to_dict(),
            result_ref=(
                destination.resolve() / "result.json"
            ).relative_to(repo_root.resolve()).as_posix(),
        ).to_dict()
    from fugue.bench.run_conformance import (
        write_hosted_evidence_privacy_receipt,
    )

    hosted_privacy = write_hosted_evidence_privacy_receipt(
        repo_root=repo_root,
        run_id=run_id,
        rows=scored,
        env=service.env,
        evidence_project=trace_project_slug(
            _comparison_evidence_environment(spec, service.env)
        ),
        private_labels_path=_frozen_private_labels_path(
            repo_root,
            str(approved_inputs["private_labels_sha256"]),
        ),
        publication_payloads=publication_payloads,
        fetch_hosted=fetch_weave,
    )
    _apply_hosted_evidence_privacy(
        scored,
        repo_root=repo_root,
        run_id=run_id,
        receipt_digest=hosted_privacy.sha256,
    )
    _atomic_text(summary.path, _jsonl(scored))
    result = analyze_comparison_rows(
        comparison_id=spec.id,
        preview_digest=preview.preview_digest,
        rows=scored,
        source=run_id,
        expected_evidence_project=trace_project_slug(
            _comparison_evidence_environment(spec, service.env)
        ),
        approved_comparison=approved_execution_lock,
        decision_policy=spec.decision_policy,
        expected_source_evidence_project=(
            spec.execution.source_evidence_project
        ),
        result_schema_version=3 if spec.schema_version >= 3 else 2,
        study_intent=(
            "mcp_release_qualification"
            if spec.decision_policy is not None
            else "candidate_comparison"
        ),
        release_note_coverage=release_note_coverage,
        supersedes=spec.supersedes,
    )
    json_path, markdown_path = write_comparison_result(
        result,
        destination=destination,
    )
    _publish_comparison_result(
        spec=spec,
        result=result,
        json_path=json_path,
        markdown_path=markdown_path,
        destination=destination,
        publication_path=research_publication_path,
        projection=research_projection,
        publish_research=publish_research,
        repo_root=repo_root,
    )
    projection_context.canonical_result_published = True
    return result, json_path, markdown_path


def _require_checkpoint_judges(
    spec: ComparisonSpecV1,
    row: dict[str, Any],
    *,
    checkpoint_index: int,
) -> None:
    """Record whether configured judges scored at the guarded launch boundary."""

    if checkpoint_index >= spec.execution.evidence_checkpoint_cells:
        return
    judge_ids = tuple(
        evaluator.id
        for evaluator in spec.evaluators
        if evaluator.type == "llm_judge"
    )
    if not judge_ids:
        return
    results = _mapping_or_empty(row.get("comparison_judges"))
    unavailable = [
        judge_id
        for judge_id in judge_ids
        if _mapping_or_empty(results.get(judge_id)).get("status") != "scored"
    ]
    if unavailable:
        row["comparison_judge_checkpoint_unavailable"] = unavailable
        required_unavailable = [
            evaluator.id
            for evaluator in spec.evaluators
            if evaluator.type == "llm_judge"
            and evaluator.required
            and evaluator.id in unavailable
        ]
        if required_unavailable:
            row["comparison_judge_checkpoint_status"] = "failed"
            row["comparison_judge_checkpoint_required_unavailable"] = (
                required_unavailable
            )
            raise RuntimeError(
                "first-cell required judge checkpoint did not score: "
                + ", ".join(required_unavailable)
            )
        row["comparison_judge_checkpoint_status"] = "advisory_unavailable"
        return
    row["comparison_judge_checkpoint_status"] = "passed"


def _apply_harbor_conformance(
    rows: Sequence[dict[str, Any]],
    *,
    repo_root: Path,
    run_id: str,
) -> None:
    if not any(row.get("harbor_config") for row in rows):
        return
    try:
        from fugue.bench.run_conformance import (
            read_harbor_run_conformance_receipt,
        )

        receipt = read_harbor_run_conformance_receipt(
            repo_root=repo_root,
            run_id=run_id,
        )
    except (FileNotFoundError, ValueError):
        for row in rows:
            row["harbor_conformance_status"] = "unavailable"
        return
    if receipt.get("backend") != "local_harbor_docker":
        return
    identity = _mapping_or_empty(receipt.get("execution_identity"))
    receipt_version = int(receipt.get("schema_version") or 1)
    local_privacy_scan = _mapping_or_empty(
        receipt.get("local_artifact_privacy_scan")
        if receipt_version == 2
        else receipt.get("secret_value_scan")
    )
    private_boundary = _mapping_or_empty(receipt.get("private_label_boundary"))
    cleanup = _mapping_or_empty(receipt.get("docker_cleanup"))
    matches = cleanup.get("matched_containers") or ()
    for row in rows:
        row.update(
            {
                "harbor_environment": "local_harbor_docker",
                "harbor_conformance_status": str(
                    receipt.get("status") or "unavailable"
                ),
                "harbor_conformance_receipt_digest": str(
                    receipt.get("receipt_sha256") or ""
                ),
                "harbor_policy_attestation_verified": _status_boolean(
                    identity.get("status")
                ),
                "private_label_boundary_verified": _status_boolean(
                    private_boundary.get("status")
                ),
                "sandbox_cleanup_verified": _status_boolean(
                    cleanup.get("status")
                ),
                "orphaned_sandbox": bool(matches),
            }
        )
        if receipt_version == 2:
            row.update(
                {
                    "privacy_contract_version": 2,
                    "local_artifact_privacy_scan_status": str(
                        local_privacy_scan.get("status") or "unavailable"
                    ),
                    "local_artifact_privacy_scan_digest": stable_digest(
                        local_privacy_scan
                    ),
                    "local_artifact_privacy_match_count": sum(
                        int(item.get("match_count") or 0)
                        for item in local_privacy_scan.get(
                            "files_with_matches"
                        )
                        or ()
                        if isinstance(item, Mapping)
                    ),
                }
            )
        else:
            # V1 receipts remain readable as historical evidence, but their
            # aggregate privacy flag cannot qualify a new V2 result.
            row.update(
                {
                    "privacy_contract_version": 1,
                    "legacy_privacy_scan_complete": _status_boolean(
                        local_privacy_scan.get("status")
                    ),
                }
            )


def _apply_hosted_evidence_privacy(
    rows: Sequence[dict[str, Any]],
    *,
    repo_root: Path,
    run_id: str,
    receipt_digest: str,
) -> None:
    try:
        from fugue.bench.run_conformance import (
            PRIVACY_CONTRACT_VERSION,
            read_hosted_evidence_privacy_receipt,
        )

        receipt = read_hosted_evidence_privacy_receipt(
            repo_root=repo_root,
            run_id=run_id,
        )
    except (FileNotFoundError, ValueError):
        for row in rows:
            if row.get("privacy_contract_version") == 2:
                row["hosted_evidence_privacy_scan_status"] = "unavailable"
        return
    digest = str(receipt.get("receipt_sha256") or "")
    digest_verified = bool(digest and digest == receipt_digest)
    status = (
        str(receipt.get("status") or "unavailable")
        if digest_verified
        else "unavailable"
    )
    match_count = sum(
        int(receipt.get(field) or 0)
        for field in (
            "secret_match_count",
            "private_corpus_match_count",
            "private_structure_match_count",
        )
    )
    for row in rows:
        if row.get("privacy_contract_version") != PRIVACY_CONTRACT_VERSION:
            continue
        row.update(
            {
                "hosted_evidence_privacy_scan_status": status,
                "hosted_evidence_privacy_scan_digest": (
                    digest if digest_verified else None
                ),
                "hosted_evidence_privacy_match_count": match_count,
            }
        )


def _status_boolean(value: Any) -> bool | None:
    status = str(value or "")
    if status == "passed":
        return True
    if status == "failed":
        return False
    return None


def _candidate(raw: Any, default_label: str) -> ComparisonCandidateV1:
    value = _mapping(raw, "candidate")
    _reject_unknown(
        value,
        {
            "label",
            "prompt_id",
            "skills",
            "context",
            "integrations",
            "agent_kwargs",
            "environment",
        },
        "candidate",
    )
    skills = _string_tuple(value.get("skills") or [], "skill", allow_empty=True)
    for item in skills:
        validate_id(item, kind="skill id")
    if len(set(skills)) != len(skills):
        raise ValueError("candidate skills must be unique")
    context = dict(
        value.get("context")
        or {"system_id": "none", "delivery": "portable"}
    )
    _reject_unknown(context, {"system_id", "delivery", "config"}, "candidate context")
    if context.get("delivery") not in {"portable", "native_mcp"}:
        raise ValueError("candidate context delivery must be portable or native_mcp")
    integrations = tuple(
        _integration(value, index)
        for index, value in enumerate(
            _sequence(value.get("integrations") or [], "integrations", allow_empty=True),
            start=1,
        )
    )
    if len({item["id"] for item in integrations}) != len(integrations):
        raise ValueError("candidate integrations must be unique")
    return ComparisonCandidateV1(
        label=_text(value.get("label") or default_label, "candidate label", 200),
        prompt_id=(
            validate_id(str(value["prompt_id"]), kind="prompt id")
            if value.get("prompt_id")
            else None
        ),
        skills=skills,
        context=context,
        integrations=integrations,
        agent_kwargs=dict(_mapping(value.get("agent_kwargs") or {}, "agent kwargs")),
        environment=dict(_mapping(value.get("environment") or {}, "environment")),
    )


def _integration(raw: Any, index: int) -> dict[str, Any]:
    if isinstance(raw, str):
        return {"id": validate_id(raw, kind="integration id")}
    value = _mapping(raw, f"integration {index}")
    _reject_unknown(value, {"id", "config"}, f"integration {index}")
    return _drop_empty(
        {
            "id": validate_id(str(value.get("id") or ""), kind="integration id"),
            "config": dict(_mapping(value.get("config") or {}, "integration config")),
        }
    )


def _post_trial_verifier(
    raw: Any,
    *,
    evaluator_type: str,
    scorer: str | None,
    dimensions: Sequence[str],
) -> ComparisonPostTrialVerifierV1 | None:
    if raw is None:
        return None
    value = _mapping(raw, "post-trial verifier")
    _reject_unknown(
        value,
        {"type", "source", "runtime", "dimension"},
        "post-trial verifier",
    )
    if evaluator_type != "deterministic":
        raise ValueError("only deterministic evaluators may use a host verifier")
    if not scorer:
        raise ValueError("host verifier requires a custom deterministic scorer")
    verifier_type = str(value.get("type") or "")
    if verifier_type != "node_test":
        raise ValueError("post-trial verifier type must be node_test")
    source = str(value.get("source") or "")
    if not source:
        raise ValueError("post-trial verifier source is required")
    runtime = validate_id(
        str(value.get("runtime") or ""), kind="verifier runtime id"
    )
    dimension = str(value.get("dimension") or "")
    if dimension not in dimensions:
        raise ValueError(
            "host verifier dimension must be one declared evaluator dimension"
        )
    return ComparisonPostTrialVerifierV1(
        type="node_test",
        source=source,
        runtime=runtime,
        dimension=dimension,
    )


def _evaluator(raw: Any) -> ComparisonEvaluatorV1:
    value = _mapping(raw, "evaluator")
    _reject_unknown(
        value,
        {
            "id",
            "type",
            "required",
            "checks",
            "scorer",
            "runtime",
            "verifier",
            "profile",
            "calibration",
            "rubric",
            "dimensions",
            "dimension_roles",
            "evidence",
            "timeout_sec",
            "reserve_cost_usd",
        },
        "evaluator",
    )
    evaluator_type = str(value.get("type") or "")
    if evaluator_type not in {"deterministic", "llm_judge"}:
        raise ValueError("evaluator type must be deterministic or llm_judge")
    checks = _string_tuple(
        value.get("checks") or [],
        "evaluator check",
        allow_empty=True,
    )
    profile = str(value.get("profile") or "") or None
    scorer = str(value.get("scorer") or "") or None
    runtime = str(value.get("runtime") or "") or None
    calibration = str(value.get("calibration") or "") or None
    rubric = str(value.get("rubric") or "").strip() or None
    dimensions = _string_tuple(
        value.get("dimensions") or [], "judge dimension", allow_empty=True
    )
    raw_dimension_roles = _mapping(
        value.get("dimension_roles") or {},
        "evaluator dimension roles",
    )
    allowed_roles = {
        "outcome",
        "mechanism",
        "safety_gate",
        "infrastructure",
        "efficiency",
    }
    dimension_roles: dict[str, DimensionRole] = {}
    for dimension, raw_role in raw_dimension_roles.items():
        name = str(dimension)
        role = str(raw_role)
        if role not in allowed_roles:
            raise ValueError(
                f"evaluator dimension {name!r} has unsupported role {role!r}"
            )
        dimension_roles[name] = role  # type: ignore[assignment]
    if set(dimension_roles) - set(dimensions):
        raise ValueError(
            "evaluator dimension roles may reference only declared dimensions"
        )
    evidence = _string_tuple(
        value.get("evidence") or [], "judge evidence", allow_empty=True
    )
    if evaluator_type == "llm_judge":
        timeout_sec = None
        if "timeout_sec" in value:
            timeout_sec = _positive_int(
                value["timeout_sec"],
                "LLM judge timeout_sec",
            )
            if timeout_sec > MAX_COMPARISON_JUDGE_TIMEOUT_SEC:
                raise ValueError(
                    "LLM judge timeout_sec must be no greater than "
                    f"{MAX_COMPARISON_JUDGE_TIMEOUT_SEC}"
                )
    else:
        if "timeout_sec" in value:
            raise ValueError("timeout_sec is supported only for LLM judge evaluators")
        timeout_sec = None
    if evaluator_type == "deterministic" and bool(checks) == bool(scorer):
        raise ValueError(
            "deterministic evaluator requires exactly one of checks or scorer"
        )
    if evaluator_type == "deterministic" and scorer and not runtime:
        runtime = "python312-sandbox-v1"
    if evaluator_type == "deterministic" and runtime and not scorer:
        raise ValueError("deterministic evaluator runtime requires scorer")
    verifier = _post_trial_verifier(
        value.get("verifier"),
        evaluator_type=evaluator_type,
        scorer=scorer,
        dimensions=dimensions,
    )
    if evaluator_type == "deterministic" and scorer:
        validate_id(str(runtime), kind="scorer runtime id")
        if not dimensions:
            raise ValueError("custom deterministic scorer requires dimensions")
    if evaluator_type == "llm_judge" and not profile:
        raise ValueError("LLM judge evaluator requires a profile")
    if evaluator_type == "llm_judge" and not rubric:
        raise ValueError("LLM judge evaluator requires a public rubric")
    if evaluator_type == "llm_judge" and not dimensions:
        raise ValueError("LLM judge evaluator requires dimensions")
    unsupported_evidence = sorted(
        set(evidence)
        - {
            "tool_names",
            "artifact_paths",
            "retrieved_paths",
            "inspected_paths",
            "changed_paths",
        }
    )
    if unsupported_evidence:
        raise ValueError(
            "unsupported blind-judge evidence fields: "
            + ", ".join(unsupported_evidence)
        )
    return ComparisonEvaluatorV1(
        id=validate_id(str(value.get("id") or ""), kind="evaluator id"),
        type=evaluator_type,  # type: ignore[arg-type]
        required=bool(value.get("required", True)),
        checks=checks,
        scorer=scorer,
        runtime=runtime,
        verifier=verifier,
        profile=profile,
        calibration=calibration,
        rubric=rubric,
        dimensions=dimensions,
        dimension_roles=dimension_roles,
        evidence=evidence,
        timeout_sec=timeout_sec,
        reserve_cost_usd=_non_negative_number(
            value.get("reserve_cost_usd", 0), "judge reserve"
        ),
    )


def _execution(
    raw: Any,
    *,
    source: Path,
    repo_root: Path,
) -> ComparisonExecutionPolicyV1:
    value = _mapping(raw, "execution policy")
    _reject_unknown(
        value,
        {
            "model",
            "harnesses",
            "attempts",
            "concurrency",
            "max_cost_usd",
            "reserve_per_attempt_usd",
            "approval_required",
            "trace_content",
            "source_evidence_project",
            "source_evidence_destination",
            "evidence_project",
            "evidence_destination",
            "study_console_base_url",
            "research_id",
            "infrastructure_receipt",
            "source_lock",
            "evidence_lock",
            "source_conformance_receipt",
            "release_notes_lock",
            "mechanism_receipt",
            "prerequisite_result",
            "prerequisite_attestation",
            "prerequisite_comparison_id",
            "prerequisite_spec",
            "preparation_required",
            "evidence_checkpoint_cells",
            "scheduling_seed",
            "qualification_inputs",
            "environment",
        },
        "execution policy",
    )
    harnesses = _string_tuple(value.get("harnesses"), "harness")
    unknown = sorted(set(harnesses) - set(_HARNESS_AGENTS))
    if unknown:
        raise ValueError("unsupported harnesses: " + ", ".join(unknown))
    trace_content = str(value.get("trace_content") or "full")
    if trace_content not in {"full", "metadata"}:
        raise ValueError("trace_content must be full or metadata")
    concurrency = _positive_int(value.get("concurrency", 1), "concurrency")
    evidence_checkpoint_cells = _non_negative_int(
        value.get("evidence_checkpoint_cells", 0),
        "evidence checkpoint cells",
    )
    if evidence_checkpoint_cells and concurrency != 1:
        raise ValueError(
            "evidence checkpoint cells require comparison concurrency 1"
        )
    evidence_project = _evidence_project(value.get("evidence_project"))
    evidence_destination = _declared_evidence_destination(
        value.get("evidence_destination"),
        evidence_project=evidence_project,
    )
    source_evidence_project = _evidence_project(
        value.get("source_evidence_project"),
        label="source_evidence_project",
    )
    source_evidence_destination = _declared_evidence_destination(
        value.get("source_evidence_destination"),
        evidence_project=source_evidence_project,
        label="source_evidence_destination",
    )
    if (source_evidence_project is None) != (
        source_evidence_destination is None
    ):
        raise ValueError(
            "source evidence project and destination must be declared together"
        )
    prerequisite_values = (
        value.get("prerequisite_result"),
        value.get("prerequisite_attestation"),
        value.get("prerequisite_comparison_id"),
        value.get("prerequisite_spec"),
    )
    if any(prerequisite_values) and not all(prerequisite_values):
        raise ValueError(
            "prerequisite result, attestation, and comparison id must be "
            "declared together"
        )
    qualification_inputs: dict[str, str] = {}
    for raw_name, raw_path in _mapping(
        value.get("qualification_inputs") or {},
        "execution qualification inputs",
    ).items():
        name = str(raw_name)
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,99}", name):
            raise ValueError("qualification input names must be stable identifiers")
        qualification_inputs[name] = _portable_input_path(
            raw_path,
            source,
            repo_root,
            f"qualification input {name}",
        )
    return ComparisonExecutionPolicyV1(
        model=_text(value.get("model"), "execution model", 300),
        harnesses=harnesses,
        attempts=_positive_int(value.get("attempts", 1), "attempts"),
        concurrency=concurrency,
        max_cost_usd=_non_negative_number(
            value.get("max_cost_usd", 0), "maximum cost"
        ),
        reserve_per_attempt_usd=_non_negative_number(
            value.get("reserve_per_attempt_usd", 0),
            "attempt reserve",
        ),
        approval_required=bool(value.get("approval_required", True)),
        trace_content=trace_content,  # type: ignore[arg-type]
        source_evidence_project=source_evidence_project,
        source_evidence_destination=source_evidence_destination,
        evidence_project=evidence_project,
        evidence_destination=evidence_destination,
        study_console_base_url=_safe_application_base_url(
            value.get("study_console_base_url"),
            label="study_console_base_url",
        ),
        research_id=(
            validate_id(str(value["research_id"]), kind="research id")
            if value.get("research_id")
            else None
        ),
        infrastructure_receipt=(
            _portable_input_path(
                value.get("infrastructure_receipt"),
                source,
                repo_root,
                "infrastructure receipt",
            )
            if value.get("infrastructure_receipt")
            else None
        ),
        source_lock=(
            _portable_input_path(
                value.get("source_lock"),
                source,
                repo_root,
                "local source lock",
            )
            if value.get("source_lock")
            else None
        ),
        evidence_lock=(
            _portable_input_path(
                value.get("evidence_lock"),
                source,
                repo_root,
                "evidence lock",
            )
            if value.get("evidence_lock")
            else None
        ),
        source_conformance_receipt=(
            _portable_input_path(
                value.get("source_conformance_receipt"),
                source,
                repo_root,
                "source conformance receipt",
            )
            if value.get("source_conformance_receipt")
            else None
        ),
        release_notes_lock=(
            _portable_input_path(
                value.get("release_notes_lock"),
                source,
                repo_root,
                "release-notes lock",
            )
            if value.get("release_notes_lock")
            else None
        ),
        mechanism_receipt=(
            _portable_input_path(
                value.get("mechanism_receipt"),
                source,
                repo_root,
                "mechanism receipt",
            )
            if value.get("mechanism_receipt")
            else None
        ),
        prerequisite_result=(
            _portable_input_path(
                value.get("prerequisite_result"),
                source,
                repo_root,
                "prerequisite result",
            )
            if value.get("prerequisite_result")
            else None
        ),
        prerequisite_attestation=(
            _portable_input_path(
                value.get("prerequisite_attestation"),
                source,
                repo_root,
                "prerequisite attestation",
            )
            if value.get("prerequisite_attestation")
            else None
        ),
        prerequisite_comparison_id=(
            validate_id(
                str(value["prerequisite_comparison_id"]),
                kind="prerequisite comparison id",
            )
            if value.get("prerequisite_comparison_id")
            else None
        ),
        prerequisite_spec=(
            _portable_input_path(
                value.get("prerequisite_spec"),
                source,
                repo_root,
                "prerequisite comparison spec",
            )
            if value.get("prerequisite_spec")
            else None
        ),
        preparation_required=bool(value.get("preparation_required", False)),
        evidence_checkpoint_cells=evidence_checkpoint_cells,
        scheduling_seed=_optional_text(
            value.get("scheduling_seed"),
            "comparison scheduling seed",
            300,
        ),
        qualification_inputs=dict(sorted(qualification_inputs.items())),
        environment=dict(
            _mapping(value.get("environment") or {}, "execution environment")
        ),
    )


def _decision_policy(raw: Any) -> DecisionPolicyV1 | None:
    if raw is None:
        return None
    value = _mapping(raw, "decision policy")
    _reject_unknown(
        value,
        {
            "release_target",
            "candidate_sha",
            "minimum_evidence_grade",
            "human_signoff_required",
            "gates",
        },
        "decision policy",
    )
    grade = str(value.get("minimum_evidence_grade") or "A").upper()
    if grade not in {"A", "B", "C"}:
        raise ValueError("minimum evidence grade must be A, B, or C")
    gates: list[DecisionGatePolicyV1] = []
    for index, raw_gate in enumerate(
        _sequence(value.get("gates") or [], "decision gates", allow_empty=True),
        start=1,
    ):
        gate = _mapping(raw_gate, f"decision gate {index}")
        _reject_unknown(
            gate,
            {
                "id",
                "label",
                "category",
                "source",
                "operator",
                "target",
                "critical",
            },
            f"decision gate {index}",
        )
        category = str(gate.get("category") or "")
        if category not in {
            "integrity",
            "task",
            "infrastructure",
            "evidence",
            "efficiency",
            "privacy",
        }:
            raise ValueError(f"decision gate {index} has an unsupported category")
        operator = str(gate.get("operator") or "")
        if operator not in {"eq", "lte", "gte"}:
            raise ValueError(f"decision gate {index} has an unsupported operator")
        target = gate.get("target")
        if (
            not isinstance(target, str | int | float | bool)
            or isinstance(target, float)
            and not math.isfinite(target)
        ):
            raise ValueError(f"decision gate {index} target must be scalar")
        gates.append(
            DecisionGatePolicyV1(
                id=validate_id(
                    str(gate.get("id") or ""), kind="decision gate id"
                ),
                label=_text(
                    gate.get("label") or gate.get("id"),
                    "decision gate label",
                    300,
                ),
                category=category,  # type: ignore[arg-type]
                source=_text(
                    gate.get("source"), "decision gate source", 300
                ),
                operator=operator,  # type: ignore[arg-type]
                target=target,
                critical=bool(gate.get("critical", True)),
            )
        )
    if len({gate.id for gate in gates}) != len(gates):
        raise ValueError("decision gate ids must be unique")
    return DecisionPolicyV1(
        release_target=_text(
            value.get("release_target"), "release target", 300
        ),
        candidate_sha=_commit_sha(value.get("candidate_sha")),
        minimum_evidence_grade=grade,  # type: ignore[arg-type]
        human_signoff_required=bool(
            value.get("human_signoff_required", True)
        ),
        gates=tuple(gates),
    )


def _public_case(
    task: Mapping[str, Any],
    *,
    spec: ComparisonSpecV1,
    index: int,
    repo_root: Path,
) -> dict[str, Any]:
    task_id = str(task["id"])
    input_value = task["input"]
    instruction = (
        str(input_value["question"])
        if isinstance(input_value, dict)
        and isinstance(input_value.get("question"), str)
        else json.dumps(input_value, indent=2, sort_keys=True)
    )
    applicability = {
        harness: {"applicable": True, "reason": None}
        for harness in spec.execution.harnesses
    }
    interaction = {
        "type": "single_turn",
        "profile_id": None,
        "scripted_turns": [],
        "directions": [],
        "max_user_turns": 0,
        "max_agent_turns": 1,
        "timeout_sec": 900,
    }
    interaction["controller_digest"] = stable_digest(interaction)
    return {
        "schema_version": 1,
        "id": task_id,
        "title": task_id.replace("-", " ").title(),
        "instruction": instruction,
        "attachments": _task_attachments(task, repo_root),
        "environment": {
            "profile_id": "artifact-python-v1",
            "profile_digest": stable_digest(
                {
                    "id": "artifact-python-v1",
                    "image": _COMPARISON_BASE_IMAGE,
                }
            ),
            "kind": "artifact",
            "base_image": _COMPARISON_BASE_IMAGE,
            "cpus": 2,
            "memory_mb": 4096,
            "storage_mb": 10240,
            "repository": None,
            "integration_ids": sorted(
                {
                    str(item["id"])
                    for candidate in (spec.baseline, spec.candidate)
                    for item in candidate.integrations
                }
            ),
        },
        "interaction": interaction,
        "harness_applicability": applicability,
        "profile_digests": {},
        "scenario_id": "comparison",
        "tags": list(task.get("tags") or []),
        "partition": str(task.get("partition") or "holdout"),
        "source_index": index,
        "task_definition_digest": stable_digest(
            {
                "comparison_id": spec.id,
                "task_id": task_id,
                "public_task": dict(task),
            }
        ),
    }


def _task_attachments(
    task: Mapping[str, Any], repo_root: Path
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(task.get("resources") or [], start=1):
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"public task {task['id']} resource {index} must be an object"
            )
        _reject_unknown(
            raw,
            {"path", "target"},
            f"public task {task['id']} resource {index}",
        )
        relative = _safe_resource_relative_path(
            raw.get("path"),
            label=f"public task {task['id']} resource {index}",
        )
        source = repo_root / relative
        if source.is_symlink() or not source.is_file():
            raise ValueError(
                f"public task {task['id']} resource is not a regular file: {relative}"
            )
        target = str(raw.get("target") or "")
        target_path = PurePosixPath(target)
        allowed_root = PurePosixPath("/workspace/resources")
        if (
            not target_path.is_absolute()
            or any(part in {"", ".", ".."} for part in target_path.parts)
            or target_path.parts[: len(allowed_root.parts)] != allowed_root.parts
        ):
            raise ValueError(
                f"public task {task['id']} resource target must be under "
                "/workspace/resources"
            )
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        result.append(
            {
                "locked_relative": _frozen_resource_path(
                    repo_root,
                    digest,
                    source.name,
                )
                .relative_to(repo_root)
                .as_posix(),
                "sha256": digest,
                "target": target_path.as_posix(),
            }
        )
    return result


def _safe_resource_relative_path(value: Any, *, label: str) -> str:
    text = str(value or "")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} path must be a safe repository-relative file")
    return path.as_posix()


def _variant_dict(variant_id: str, value: ComparisonCandidateV1) -> dict[str, Any]:
    return {
        "id": variant_id,
        "label": value.label,
        "prompt_id": value.prompt_id,
        "skills": list(value.skills),
        "context": value.context,
        "integrations": list(value.integrations),
        "agent_kwargs": value.agent_kwargs,
        "environment": value.environment,
    }


def _research_scorer(
    value: ComparisonEvaluatorV1,
    *,
    revision: str | None = None,
) -> dict[str, Any]:
    if value.type == "deterministic":
        if value.scorer:
            return {
                "id": value.id,
                "label": value.id.replace("-", " ").title(),
                "kind": "deterministic",
                "description": "Pinned custom scorer executed in an isolated runtime.",
                "required": value.required,
                "aggregation": "Every returned dimension must pass.",
                "revision": revision,
                "evidence_inputs": [
                    "Public task",
                    "Agent output",
                    "Permitted normalized evidence",
                    "Host-only expected values",
                ],
                "dimensions": [
                    {
                        "id": dimension.replace("_", "-"),
                        "label": dimension.replace("_", " ").title(),
                        "source_key": (
                            f"comparison.deterministic.{value.id}.{dimension}"
                        ),
                        "target": True,
                        "primary": index == 0,
                    }
                    for index, dimension in enumerate(value.dimensions)
                ],
            }
        return {
            "id": value.id,
            "label": value.id.replace("-", " ").title(),
            "kind": "deterministic",
            "description": "Deterministic checks compiled from the comparison evaluator.",
            "required": value.required,
            "aggregation": "Every required check must pass.",
            "revision": revision,
            "evidence_inputs": ["Agent output", "Host-only expected values"],
            "dimensions": [
                {
                    "id": check.replace("_", "-"),
                    "label": check.replace("_", " ").title(),
                    "source_key": check,
                    "target": True,
                    "primary": check == "expected_values",
                }
                for check in value.checks
            ],
        }
    return {
        "id": value.id,
        "label": value.id.replace("-", " ").title(),
        "kind": "llm_judge",
        "description": "Calibrated blind qualitative review.",
        "required": value.required,
        "revision": revision,
        "model": value.profile,
        "rubric_summary": "Use only permitted evidence and remain calibrated about coverage.",
        "dimensions": [
            {
                "id": dimension.replace("_", "-"),
                "label": dimension.replace("_", " ").title(),
                "source_key": dimension,
                "primary": dimension == "evidence_grounding",
            }
            for dimension in value.dimensions
        ],
        "blind_fields": [
            "harness",
            "model",
            "variant_id",
            "context_system_id",
            "candidate_id",
            "treatment",
        ],
        "evidence_inputs": ["Agent output", *value.evidence],
    }


def _load_public_tasks(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path, "public taskset")
    ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        _reject_unknown(row, _PUBLIC_TASK_FIELDS, f"public task {index}")
        task_id = validate_id(str(row.get("id") or ""), kind="task id")
        if task_id in ids:
            raise ValueError(f"duplicate public task id: {task_id}")
        ids.add(task_id)
        if "input" not in row:
            raise ValueError(f"public task {task_id} requires input")
        leaked = sorted(
            key for key in row if key.lower() in _PRIVATE_WORDS
        )
        if leaked:
            raise ValueError(
                f"public task {task_id} contains private field(s): "
                + ", ".join(leaked)
            )
        partition = str(row.get("partition") or "holdout")
        if partition not in {"qualification", "discovery", "holdout"}:
            raise ValueError(f"public task {task_id} has invalid partition")
        tags = row.get("tags") or []
        if not isinstance(tags, list) or not all(isinstance(item, str) for item in tags):
            raise ValueError(f"public task {task_id} tags must be strings")
        critical_dimensions = row.get("critical_dimensions") or []
        if not isinstance(critical_dimensions, list) or not all(
            isinstance(item, str) and item.strip()
            for item in critical_dimensions
        ):
            raise ValueError(
                f"public task {task_id} critical_dimensions must be strings"
            )
        if len(critical_dimensions) != len(set(critical_dimensions)):
            raise ValueError(
                f"public task {task_id} critical_dimensions must be unique"
            )
        resources = row.get("resources") or []
        if not isinstance(resources, list):
            raise ValueError(f"public task {task_id} resources must be an array")
        for resource_index, resource in enumerate(resources, start=1):
            if not isinstance(resource, dict):
                raise ValueError(
                    f"public task {task_id} resource {resource_index} must be an object"
                )
            _reject_unknown(
                resource,
                {"path", "target"},
                f"public task {task_id} resource {resource_index}",
            )
            _safe_resource_relative_path(
                resource.get("path"),
                label=f"public task {task_id} resource {resource_index}",
            )
        row["id"] = task_id
        row["partition"] = partition
        row["tags"] = tags
        if critical_dimensions:
            row["critical_dimensions"] = critical_dimensions
        else:
            row.pop("critical_dimensions", None)
        row["resources"] = resources
    return rows


def _load_private_labels(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path, "private labels")
    ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        _reject_unknown(row, _PRIVATE_LABEL_FIELDS, f"private label {index}")
        task_id = validate_id(str(row.get("id") or ""), kind="task id")
        if task_id in ids:
            raise ValueError(f"duplicate private label id: {task_id}")
        if "expected" not in row:
            raise ValueError(f"private label {task_id} requires expected values")
        ids.add(task_id)
        row["id"] = task_id
    return rows


def _evaluator_digest(
    evaluator: ComparisonEvaluatorV1, repo_root: Path
) -> str:
    from fugue.bench.task_authoring import load_task_profiles

    value = evaluator.to_dict()
    profiles = (
        load_task_profiles(repo_root)
        if evaluator.scorer or evaluator.verifier
        else None
    )
    if evaluator.scorer:
        value["scorer_sha256"] = _sha256_path(
            _safe_input_path(
                Path(evaluator.scorer), repo_root, "deterministic scorer"
            )
        )
        assert profiles is not None
        value["scorer_runtime_profile_digest"] = profiles.scorer_runtime(
            str(evaluator.runtime)
        ).profile_digest
    if evaluator.verifier:
        value["verifier_sha256"] = _sha256_path(
            _safe_input_path(
                Path(evaluator.verifier.source), repo_root, "host verifier"
            )
        )
        assert profiles is not None
        value["verifier_runtime_profile_digest"] = profiles.scorer_runtime(
            evaluator.verifier.runtime
        ).profile_digest
    if evaluator.calibration:
        value["calibration_sha256"] = _sha256_path(
            _safe_input_path(
                Path(evaluator.calibration), repo_root, "judge calibration"
            )
        )
    if evaluator.type == "llm_judge":
        value["response_contract"] = {
            "schema_digest": stable_digest(
                _comparison_judge_response_schema(evaluator.dimensions)
            ),
            "public_rubric_contract_digest": stable_digest(
                comparison_judge_public_rubric_contract(
                    rubric=str(evaluator.rubric or ""),
                    dimensions=evaluator.dimensions,
                )
            ),
            "validator_version": COMPARISON_JUDGE_RESPONSE_VALIDATOR_VERSION,
            "requested_text_max_characters": (
                COMPARISON_JUDGE_TEXT_REQUESTED_MAX_CHARACTERS
            ),
            "response_max_characters": COMPARISON_JUDGE_RESPONSE_MAX_CHARACTERS,
            "maximum_prompt_characters": COMPARISON_JUDGE_MAX_PROMPT_CHARACTERS,
        }
    return stable_digest(value)


def _score_deterministic_output(
    *,
    task: Mapping[str, Any],
    output: Any,
    expected: Any,
    evidence: Mapping[str, Any],
    evaluators: Sequence[ComparisonEvaluatorV1],
    repo_root: Path,
    scorer_source_digests: Mapping[str, str] | None = None,
    verifier_source_digests: Mapping[str, str] | None = None,
) -> tuple[
    bool,
    dict[str, bool | float],
    dict[str, dict[str, Any]],
]:
    scores: dict[str, bool | float] = {}
    verifier_receipts: dict[str, dict[str, Any]] = {}
    evaluator_passes: list[bool] = []
    for evaluator in evaluators:
        evaluator_evidence = dict(evidence)
        verifier_passed: bool | None = None
        if evaluator.verifier:
            verifier_payload = _run_custom_verifier(
                evaluator,
                task=task,
                output=_extract_structured_result(output),
                expected=expected,
                evidence=evidence,
                repo_root=repo_root,
                approved_source_digest=(
                    str(verifier_source_digests.get(evaluator.id) or "")
                    if verifier_source_digests is not None
                    else None
                ),
            )
            verifier_passed = float(verifier_payload["score"]) == 1.0
            receipt = dict(
                _mapping(
                    verifier_payload.get("details"),
                    "host verifier receipt",
                )
            )
            verifier_receipts[evaluator.id] = receipt
            evaluator_evidence["host_verifier_receipt"] = receipt
        if evaluator.scorer:
            payload = _run_custom_scorer(
                evaluator,
                task=task,
                output=_extract_structured_result(output),
                evidence=evaluator_evidence,
                expected=expected,
                repo_root=repo_root,
                approved_source_digest=(
                    str(scorer_source_digests.get(evaluator.id) or "")
                    if scorer_source_digests is not None
                    else None
                ),
            )
            details = payload["details"]
            if not isinstance(details, Mapping) or not details:
                raise ValueError("custom scorer must return at least one dimension")
            if set(details) != set(evaluator.dimensions):
                raise ValueError(
                    "custom scorer output does not match its declared dimensions"
                )
            normalized_details = dict(details)
            if evaluator.verifier:
                verifier_dimension = evaluator.verifier.dimension
                scorer_value = normalized_details.get(verifier_dimension)
                if _bool_score(scorer_value) is not verifier_passed:
                    raise ValueError(
                        "custom scorer and host verifier dimension disagree"
                    )
            for name, value in normalized_details.items():
                dimension = str(name)
                if (
                    not 1 <= len(dimension) <= 100
                    or not dimension[0].isalnum()
                    or any(
                        not (character.isalnum() or character in {"_", "-"})
                        for character in dimension
                    )
                ):
                    raise ValueError(
                        "scorer dimension names must use letters, numbers, _ or -"
                    )
                if isinstance(value, bool):
                    normalized: bool | float = value
                elif (
                    isinstance(value, int | float)
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and 0 <= float(value) <= 1
                ):
                    normalized = float(value)
                else:
                    raise ValueError(
                        f"scorer dimension {dimension!r} must be bool or 0..1"
                    )
                scores[f"{evaluator.id}.{dimension}"] = normalized
            evaluator_passes.append(
                float(payload["score"]) == 1.0
                and (verifier_passed is None or verifier_passed)
            )
            continue
        check_scores = {
            "answer_present": bool(
                output is not None
                and (not isinstance(output, str) or bool(output.strip()))
            ),
            "expected_values": _contains_expected(
                _extract_structured_result(output),
                expected,
            ),
        }
        selected = {
            check: check_scores[check]
            for check in evaluator.checks
        }
        scores.update(selected)
        evaluator_passes.append(all(selected.values()))
    return (
        bool(evaluator_passes) and all(evaluator_passes),
        scores,
        verifier_receipts,
    )


def _run_custom_scorer(
    evaluator: ComparisonEvaluatorV1,
    *,
    task: Mapping[str, Any],
    output: Any,
    evidence: Mapping[str, Any],
    expected: Any,
    repo_root: Path,
    approved_source_digest: str | None = None,
) -> dict[str, Any]:
    from fugue.bench.task_authoring import (
        TaskAuthoringLimitsV1,
        load_task_profiles,
        run_inline_scorer,
    )

    if not evaluator.scorer or not evaluator.runtime:
        raise ValueError("custom scorer is missing its source or runtime")
    path = (
        _frozen_evaluator_artifact_path(
            repo_root,
            approved_source_digest,
            kind="scorer",
        )
        if approved_source_digest
        else _safe_input_path(
            Path(evaluator.scorer), repo_root, "deterministic scorer"
        )
    )
    if not path.is_file():
        raise FileNotFoundError(f"approved deterministic scorer not found: {path}")
    if approved_source_digest and _sha256_path(path) != approved_source_digest:
        raise ValueError("approved deterministic scorer immutable copy changed")
    source = path.read_text(encoding="utf-8")
    _validate_custom_scorer_source(source)
    wrapper = (
        source.rstrip()
        + "\n\n"
        + """
if __name__ == "__main__":
    import json
    import math
    import sys

    with open(sys.argv[1], encoding="utf-8") as handle:
        _fugue_payload = json.load(handle)
    _fugue_reference = _fugue_payload["reference"]
    _fugue_evidence = dict(_fugue_payload["evidence"])
    _fugue_evidence["expected"] = _fugue_reference["expected"]
    _fugue_result = score(
        _fugue_reference["task"],
        _fugue_reference["output"],
        _fugue_evidence,
    )
    if not isinstance(_fugue_result, dict) or not _fugue_result:
        raise ValueError("score() must return a non-empty object")
    _fugue_values = []
    for _fugue_name, _fugue_value in _fugue_result.items():
        if not isinstance(_fugue_name, str) or not _fugue_name:
            raise ValueError("score dimension names must be non-empty strings")
        if isinstance(_fugue_value, bool):
            _fugue_values.append(1.0 if _fugue_value else 0.0)
        elif (
            isinstance(_fugue_value, (int, float))
            and not isinstance(_fugue_value, bool)
            and math.isfinite(float(_fugue_value))
            and 0 <= float(_fugue_value) <= 1
        ):
            _fugue_values.append(float(_fugue_value))
        else:
            raise ValueError("score dimensions must be bool or numbers in 0..1")
    print(json.dumps({
        "score": min(_fugue_values),
        "reason": "custom deterministic scorer",
        "details": _fugue_result,
    }, sort_keys=True))
"""
    )
    profiles = load_task_profiles(repo_root)
    profile = profiles.scorer_runtime(evaluator.runtime)
    limits = TaskAuthoringLimitsV1(
        max_tasks=1,
        max_scenarios=1,
        max_prompt_bytes=1,
        max_authored_asset_bytes=1,
        max_user_turns=1,
        max_agent_turns=1,
        max_interactor_calls=0,
        max_judge_calls=0,
        scorer_timeout_sec=30,
        scorer_memory_mb=256,
        scorer_cpus=1.0,
        scorer_output_bytes=64_000,
    )
    return run_inline_scorer(
        source=wrapper,
        evidence=dict(evidence),
        reference={
            "task": dict(task),
            "output": output,
            "expected": expected,
        },
        profile=profile,
        limits=limits,
    )


def _run_custom_verifier(
    evaluator: ComparisonEvaluatorV1,
    *,
    task: Mapping[str, Any],
    output: Any,
    expected: Any,
    evidence: Mapping[str, Any],
    repo_root: Path,
    approved_source_digest: str | None = None,
) -> dict[str, Any]:
    """Run a frozen post-trial verifier without exposing its oracle to the Agent."""

    from fugue.bench.task_authoring import (
        TaskAuthoringLimitsV1,
        load_task_profiles,
        run_inline_scorer,
    )

    if not evaluator.verifier:
        raise ValueError("custom host verifier is missing its source or runtime")
    path = (
        _frozen_evaluator_artifact_path(
            repo_root,
            approved_source_digest,
            kind="verifier",
        )
        if approved_source_digest
        else _safe_input_path(
            Path(evaluator.verifier.source), repo_root, "host verifier"
        )
    )
    if not path.is_file():
        raise FileNotFoundError(f"approved host verifier not found: {path}")
    if approved_source_digest and _sha256_path(path) != approved_source_digest:
        raise ValueError("approved host verifier immutable copy changed")
    source = path.read_text(encoding="utf-8")
    if not source.strip() or len(source.encode()) > 64_000 or "\x00" in source:
        raise ValueError("host verifier source must be non-empty and at most 64 KiB")
    profiles = load_task_profiles(repo_root)
    profile = profiles.scorer_runtime(evaluator.verifier.runtime)
    runtime_lock = _read_evaluator_runtime_lock(
        profile,
        repo_root=repo_root,
        inspect_image=True,
    )
    if runtime_lock is None:
        raise RuntimeError(
            f"host verifier runtime {profile.id!r} is not prepared and locked"
        )
    archive, archive_digest = _verifier_base_archive(
        task,
        expected=expected,
        repo_root=repo_root,
    )
    limits = TaskAuthoringLimitsV1(
        max_tasks=1,
        max_scenarios=1,
        max_prompt_bytes=1,
        max_authored_asset_bytes=1,
        max_user_turns=1,
        max_agent_turns=1,
        max_interactor_calls=0,
        max_judge_calls=0,
        scorer_timeout_sec=30,
        scorer_memory_mb=256,
        scorer_cpus=1.0,
        scorer_output_bytes=64_000,
    )
    payload = run_inline_scorer(
        source=source,
        evidence={},
        reference={
            "task": dict(task),
            "output": output,
            "expected": {
                **dict(_mapping(expected, "host verifier expected contract")),
                "base_archive_base64": base64.b64encode(archive).decode("ascii"),
            },
        },
        profile=profile,
        limits=limits,
    )
    receipt = _host_verifier_receipt(
        payload,
        evaluator=evaluator,
        task=task,
        output=output,
        evidence=evidence,
        expected=expected,
        archive_digest=archive_digest,
        verifier_source_digest=_sha256_path(path),
        runtime_profile=profile,
        runtime_lock=runtime_lock,
    )
    return {**payload, "details": receipt}


def _verifier_base_archive(
    task: Mapping[str, Any],
    *,
    expected: Any,
    repo_root: Path,
) -> tuple[bytes, str]:
    contract = _mapping(expected, "host verifier expected contract")
    digest = str(contract.get("base_archive_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("host verifier base archive digest is unavailable")
    resources = [
        _mapping(item, "host verifier task resource")
        for item in _sequence(
            task.get("resources") or (),
            "host verifier task resources",
            allow_empty=True,
        )
    ]
    candidates: list[Path] = []
    for resource in resources:
        relative = _safe_resource_relative_path(
            resource.get("path"),
            label="host verifier task resource",
        )
        original = repo_root / relative
        candidates.extend(
            (
                _frozen_resource_path(repo_root, digest, original.name),
                original,
            )
        )
    for path in candidates:
        if (
            path.is_file()
            and not path.is_symlink()
            and _sha256_path(path) == digest
        ):
            return path.read_bytes(), digest
    raise ValueError("host verifier exact frozen base archive is unavailable")


def _host_verifier_receipt(
    payload: Mapping[str, Any],
    *,
    evaluator: ComparisonEvaluatorV1,
    task: Mapping[str, Any],
    output: Any,
    evidence: Mapping[str, Any],
    expected: Any,
    archive_digest: str,
    verifier_source_digest: str,
    runtime_profile: Any,
    runtime_lock: Mapping[str, Any],
) -> dict[str, Any]:
    runtime_cleanup = _verified_runtime_cleanup_receipt(payload)
    details = _mapping(payload.get("details"), "host verifier details")
    expected_fields = {
        "schema_version",
        "status",
        "failure_kind",
        "runtime",
        "command",
        "exit_code",
        "test_count",
        "pass_count",
        "fail_count",
        "output_sha256",
        "base_archive_sha256",
        "public_test_sha256",
        "submitted_artifact_sha256",
        "final_tree_sha256",
    }
    if set(details) != expected_fields:
        raise ValueError("host verifier receipt fields do not match the V2 contract")
    if details.get("schema_version") != 2:
        raise ValueError("host verifier receipt schema is unsupported")
    status = str(details.get("status") or "")
    if status not in {"passed", "failed"}:
        raise ValueError("host verifier status must be passed or failed")
    runtime = str(details.get("runtime") or "")
    command = details.get("command")
    if not runtime or len(runtime) > 200:
        raise ValueError("host verifier runtime identity is invalid")
    if command != ["node", "--test", "tests/task.test.mjs"]:
        raise ValueError("host verifier command is not the locked Node test command")
    exit_code = details.get("exit_code")
    test_count = details.get("test_count")
    pass_count = details.get("pass_count")
    fail_count = details.get("fail_count")
    if any(
        type(item) is not int
        for item in (exit_code, test_count, pass_count, fail_count)
    ):
        raise ValueError("host verifier receipt counts must be integers")
    if (
        test_count != 1
        or pass_count < 0
        or fail_count < 0
        or pass_count + fail_count != test_count
    ):
        raise ValueError("host verifier receipt counts do not reconcile")
    output_sha256 = str(details.get("output_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", output_sha256):
        raise ValueError("host verifier output digest is invalid")
    score_passed = float(payload.get("score") or 0) == 1.0
    failure_kind = details.get("failure_kind")
    if failure_kind is not None and (
        not isinstance(failure_kind, str) or not failure_kind
    ):
        raise ValueError("host verifier failure kind is invalid")
    if status == "passed":
        valid = (
            failure_kind is None
            and exit_code == 0
            and fail_count == 0
            and pass_count == test_count
        )
    else:
        valid = (
            isinstance(failure_kind, str)
            and bool(failure_kind)
            and exit_code in {1, 2, 124}
            and fail_count > 0
            and pass_count < test_count
        )
    if not valid or score_passed != (status == "passed"):
        raise ValueError("host verifier result and receipt disagree")
    contract = _mapping(expected, "host verifier expected contract")
    public_test_digest = str(contract.get("public_test_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", public_test_digest):
        raise ValueError("host verifier public test digest is unavailable")
    if details.get("base_archive_sha256") != archive_digest:
        raise ValueError("host verifier base archive receipt disagrees")
    if details.get("public_test_sha256") != public_test_digest:
        raise ValueError("host verifier public test receipt disagrees")
    submitted_digest = stable_digest(output)
    if details.get("submitted_artifact_sha256") != submitted_digest:
        raise ValueError("host verifier submitted artifact receipt disagrees")
    for name in ("final_tree_sha256", "output_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(details.get(name) or "")):
            raise ValueError(f"host verifier {name} is invalid")
    task_id = validate_id(str(task.get("id") or ""), kind="task id")
    attempt = str(evidence.get("attempt_id") or "")
    live_task_id = str(evidence.get("task_id") or "")
    if live_task_id:
        if live_task_id != task_id or not re.fullmatch(r"[0-9a-f]{64}", attempt):
            raise ValueError("host verifier evidence identity is invalid")
    else:
        attempt = stable_digest(
            {
                "kind": "qualification_fixture",
                "task_id": task_id,
                "submitted_artifact_sha256": submitted_digest,
            }
        )
    unsigned = {
        "schema_version": 2,
        "kind": "post_trial_verifier_receipt",
        "evaluator_id": evaluator.id,
        "task_id": task_id,
        "attempt_id": attempt,
        "status": status,
        "failure_kind": failure_kind,
        "runtime": runtime,
        "command": command,
        "exit_code": exit_code,
        "test_count": test_count,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "output_sha256": output_sha256,
        "base_archive_sha256": archive_digest,
        "public_test_sha256": public_test_digest,
        "submitted_artifact_sha256": submitted_digest,
        "final_tree_sha256": details["final_tree_sha256"],
        "verifier_source_sha256": verifier_source_digest,
        "runtime_profile_id": runtime_profile.id,
        "runtime_profile_digest": runtime_profile.profile_digest,
        "runtime_image": runtime_profile.image,
        "runtime_platform": runtime_profile.platform,
        "runtime_image_id": runtime_lock["image_id"],
        "runtime_lock_digest": runtime_lock["lock_digest"],
        "runtime_cleanup": dict(runtime_cleanup),
    }
    return {**unsigned, "receipt_digest": stable_digest(unsigned)}


def _verified_runtime_cleanup_receipt(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = dict(
        _mapping(
            payload.get("fugue_runtime_receipt"),
            "host verifier runtime cleanup receipt",
        )
    )
    unsigned = {
        key: value for key, value in receipt.items() if key != "receipt_digest"
    }
    if (
        receipt.get("schema_version") != 1
        or receipt.get("status") != "verified_absent"
        or receipt.get("receipt_digest") != stable_digest(unsigned)
    ):
        raise ValueError("host verifier runtime cleanup receipt is invalid")
    return receipt


def _validate_custom_scorer_source(source: str) -> None:
    if len(source.encode()) > 32_768:
        raise ValueError("custom scorer source exceeds 32 KiB")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError("custom scorer is not valid Python") from exc
    definitions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "score"
    ]
    if len(definitions) != 1:
        raise ValueError("custom scorer must define exactly one score function")
    function = definitions[0]
    if (
        len(function.args.args) != 3
        or function.args.vararg is not None
        or function.args.kwarg is not None
        or function.args.kwonlyargs
        or function.args.defaults
    ):
        raise ValueError("score must have the signature score(task, output, evidence)")
    if [argument.arg for argument in function.args.args] != [
        "task",
        "output",
        "evidence",
    ]:
        raise ValueError("score must have the signature score(task, output, evidence)")


def _custom_scorer_evidence(row: Mapping[str, Any]) -> dict[str, Any]:
    permitted = (
        "attempt_id",
        "task_id",
        "variant_id",
        "harness",
        "trial_index",
        "candidate_id",
        "execution_fingerprint",
        "artifacts",
        "artifact_paths",
        "changed_paths",
        "inspected_paths",
        "opened_paths",
        "retrieved_paths",
        "tool_calls",
        "mcp_tool_names",
        "mcp_tool_calls",
        "queried_projects",
        "mcp_queried_projects",
        "trace_summary",
    )
    return {
        key: row[key]
        for key in permitted
        if key in row
    }


def _contains_expected(output: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(output, dict) and all(
            key in output and _contains_expected(output[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(output, list) and all(
            any(_contains_expected(candidate, item) for candidate in output)
            for item in expected
        )
    return output == expected


def _is_json(value: str) -> bool:
    try:
        json.loads(value)
    except json.JSONDecodeError:
        return False
    return True


def _extract_structured_result(output: Any) -> Any:
    if not isinstance(output, str):
        return output
    text = output.strip()
    if not text:
        return output
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    fenced: list[Any] = []
    index = 0
    while True:
        start = text.find("```", index)
        if start < 0:
            break
        line_end = text.find("\n", start + 3)
        if line_end < 0:
            break
        fence_end = text.find("```", line_end + 1)
        if fence_end < 0:
            break
        language = text[start + 3 : line_end].strip().lower()
        body = text[line_end + 1 : fence_end].strip()
        if language in {"", "json"}:
            try:
                parsed, end = decoder.raw_decode(body)
            except json.JSONDecodeError:
                pass
            else:
                if not body[end:].strip():
                    fenced.append(parsed)
        index = fence_end + 3
    if len(fenced) != 1:
        return output
    return fenced[0]


def _judge_calibration_issue(
    judge: ComparisonEvaluatorV1, repo_root: Path
) -> str | None:
    if not judge.calibration:
        return f"judge {judge.id} has no reviewed calibration result"
    path = _safe_input_path(Path(judge.calibration), repo_root, "judge calibration")
    value = json.loads(path.read_text(encoding="utf-8"))
    return _judge_calibration_value_issue(judge, value)


def _judge_calibration_artifact(
    judge: ComparisonEvaluatorV1,
    *,
    repo_root: Path,
    approved_inputs: Mapping[str, Any] | None,
) -> tuple[Any, str | None]:
    if not judge.calibration:
        return None, None
    report_sha256: str | None = None
    try:
        if approved_inputs is None:
            path = _safe_input_path(
                Path(judge.calibration),
                repo_root,
                "judge calibration",
            )
            report_sha256 = _sha256_path(path)
        else:
            artifacts = _mapping(
                _mapping(
                    approved_inputs["evaluator_artifacts"],
                    "approved evaluator artifacts",
                ).get(judge.id),
                f"approved evaluator {judge.id} artifacts",
            )
            report_sha256 = str(artifacts.get("calibration_sha256") or "") or None
            path = _frozen_evaluator_artifact_path(
                repo_root,
                str(report_sha256 or ""),
                kind="calibration",
            )
        return json.loads(path.read_text(encoding="utf-8")), report_sha256
    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return None, report_sha256


def _judge_calibration_value_issue(
    judge: ComparisonEvaluatorV1,
    value: Any,
) -> str | None:
    if not isinstance(value, dict):
        return f"judge {judge.id} calibration must be a mapping"
    if value.get("schema_version") != 1:
        return f"judge {judge.id} calibration schema is unsupported"
    if value.get("review_status") != "adjudicated":
        return f"judge {judge.id} calibration is not adjudicated"
    if int(value.get("reviewers_per_example") or 0) < 2:
        return f"judge {judge.id} calibration was not double-reviewed"
    if value.get("disagreements_adjudicated") is not True:
        return f"judge {judge.id} calibration disagreements are unresolved"
    if value.get("judge_profile") != judge.profile:
        return f"judge {judge.id} calibration profile does not match"
    if value.get("rubric_digest") != _judge_contract_digest(judge):
        return f"judge {judge.id} calibration rubric does not match"
    cases_digest = str(value.get("cases_digest") or "")
    if len(cases_digest) != 64 or any(
        character not in "0123456789abcdef" for character in cases_digest
    ):
        return f"judge {judge.id} calibration case digest is unavailable"
    examples = int(value.get("examples") or 0)
    calibration_examples = int(value.get("calibration_examples") or 0)
    holdout_examples = int(value.get("holdout_examples") or 0)
    true_positive = float(value.get("true_positive_rate") or 0)
    true_negative = float(value.get("true_negative_rate") or 0)
    calibration_true_positive = float(
        value.get("calibration_true_positive_rate") or 0
    )
    calibration_true_negative = float(
        value.get("calibration_true_negative_rate") or 0
    )
    holdout_true_positive = float(value.get("holdout_true_positive_rate") or 0)
    holdout_true_negative = float(value.get("holdout_true_negative_rate") or 0)
    critical_false_passes = int(value.get("critical_false_passes") or 0)
    if examples < 48:
        return f"judge {judge.id} calibration has fewer than 48 examples"
    if calibration_examples < 36 or holdout_examples < 12:
        return f"judge {judge.id} calibration lacks the 36/12 split"
    balanced_fields = (
        "balanced_accuracy",
        "calibration_balanced_accuracy",
        "holdout_balanced_accuracy",
    )
    supplied_balanced_fields = [field for field in balanced_fields if field in value]
    if supplied_balanced_fields:
        if len(supplied_balanced_fields) != len(balanced_fields):
            return f"judge {judge.id} calibration balanced accuracy is incomplete"
        if any(
            float(value.get(field) or 0)
            < COMPARISON_JUDGE_MINIMUM_BALANCED_ACCURACY
            for field in balanced_fields
        ):
            return (
                f"judge {judge.id} calibration or holdout is below 0.85 "
                "balanced accuracy"
            )
    elif true_positive < 0.85 or true_negative < 0.85:
        # Historical V1 calibration artifacts gated TPR and TNR separately.
        return f"judge {judge.id} calibration is below 0.85 TPR/TNR"
    elif (
        calibration_true_positive < 0.85
        or calibration_true_negative < 0.85
        or holdout_true_positive < 0.85
        or holdout_true_negative < 0.85
    ):
        return f"judge {judge.id} calibration or holdout is below 0.85 TPR/TNR"
    if critical_false_passes:
        return f"judge {judge.id} has critical false passes"
    if value.get("passed") is not True:
        return f"judge {judge.id} calibration did not pass"
    return None


def _judge_execution_calibration_issue(  # noqa: C901 - strict artifact audit.
    judge: ComparisonEvaluatorV1,
    *,
    repo_root: Path,
    approved_inputs: Mapping[str, Any] | None,
) -> str | None:
    """Require a locked calibration result before any paid judge or Agent cell.

    A human-adjudicated calibration remains the qualification standard. An
    explicitly declared synthetic gate may enable an advisory same-family judge,
    but it never upgrades that judge to a human-qualified outcome claim.
    """

    value, _ = _judge_calibration_artifact(
        judge,
        repo_root=repo_root,
        approved_inputs=approved_inputs,
    )
    human_issue = _judge_calibration_value_issue(judge, value)
    if human_issue is None:
        return None
    if not isinstance(value, Mapping):
        return human_issue
    gate = value.get("execution_gate")
    if not isinstance(gate, Mapping):
        return human_issue
    expected_fields = {
        "kind",
        "required_before_agent_trials",
        "result_path",
        "preview_digest",
        "model",
        "cases_artifact_path",
        "cases_artifact_sha256",
        "cases_digest",
        "rubric_digest",
        "runner_artifact_path",
        "runner_artifact_sha256",
        "validator_artifact_path",
        "validator_artifact_sha256",
        "passing_labels",
        "score_threshold",
        "critical_policy",
        "metric_policy",
        "minimum_balanced_accuracy",
        "balanced_accuracy_scopes",
        "response_schema_digest",
        "response_request_mode",
        "response_validator_version",
        "examples",
        "calibration_examples",
        "holdout_examples",
        "maximum_cost_usd",
        "maximum_critical_false_passes",
        "campaign_allocation_usd",
        "prior_failed_requests",
        "prior_accounted_reserve_usd",
    }
    ledger_fields = {
        "prior_runs_ledger_path",
        "prior_runs_ledger_sha256",
        "prior_runs_ledger_digest",
        "prior_runs",
    }
    allowed_field_sets = {
        frozenset(expected_fields),
        frozenset(expected_fields | ledger_fields),
    }
    if frozenset(gate) not in allowed_field_sets:
        return f"judge {judge.id} synthetic execution gate fields do not match"
    if gate.get("kind") != "synthetic_blinded_advisory_v1":
        return f"judge {judge.id} synthetic execution gate kind is unsupported"
    if gate.get("required_before_agent_trials") is not True:
        return f"judge {judge.id} synthetic execution gate is not mandatory"
    if gate.get("model") != judge.profile:
        return f"judge {judge.id} synthetic execution gate profile does not match"
    if gate.get("rubric_digest") != _judge_contract_digest(judge):
        return f"judge {judge.id} synthetic execution gate rubric does not match"
    if gate.get("cases_digest") != value.get("cases_digest"):
        return f"judge {judge.id} synthetic execution gate cases do not match"
    binding_issue = _synthetic_calibration_binding_issue(
        judge,
        gate=gate,
        repo_root=repo_root,
    )
    if binding_issue:
        return binding_issue
    preview_digest = str(gate.get("preview_digest") or "")
    if not _is_sha256(preview_digest):
        return f"judge {judge.id} synthetic execution preview digest is invalid"
    if (
        int(gate.get("examples") or 0) != 48
        or int(gate.get("calibration_examples") or 0) != 36
        or int(gate.get("holdout_examples") or 0) != 12
    ):
        return f"judge {judge.id} synthetic execution gate has the wrong cohort"
    if gate.get("metric_policy") != COMPARISON_JUDGE_METRIC_POLICY:
        return f"judge {judge.id} synthetic metric policy drifted"
    if list(gate.get("balanced_accuracy_scopes") or []) != list(
        COMPARISON_JUDGE_BALANCED_ACCURACY_SCOPES
    ):
        return f"judge {judge.id} synthetic balanced-accuracy scopes drifted"
    minimum_balanced_accuracy = float(
        gate.get("minimum_balanced_accuracy") or 0
    )
    if minimum_balanced_accuracy < COMPARISON_JUDGE_MINIMUM_BALANCED_ACCURACY:
        return f"judge {judge.id} synthetic execution thresholds are too weak"
    if gate.get("maximum_critical_false_passes") != 0:
        return f"judge {judge.id} synthetic critical-failure threshold is unsafe"
    if list(gate.get("passing_labels") or []) != list(
        COMPARISON_JUDGE_PASSING_LABELS
    ):
        return f"judge {judge.id} synthetic passing-label policy drifted"
    if gate.get("score_threshold") != COMPARISON_JUDGE_PASSING_SCORE:
        return f"judge {judge.id} synthetic passing-score threshold drifted"
    if gate.get("critical_policy") != COMPARISON_JUDGE_CRITICAL_POLICY:
        return f"judge {judge.id} synthetic critical policy drifted"
    raw_result_path = str(gate.get("result_path") or "")
    result_path = Path(raw_result_path)
    if (
        result_path.is_absolute()
        or ".." in result_path.parts
        or result_path.parts[:3] != (".fugue", "runtime", "community-skill-upgrades")
    ):
        return f"judge {judge.id} synthetic result path is outside governed runtime"
    try:
        path = _safe_input_path(
            result_path,
            repo_root,
            "judge synthetic calibration result",
        )
        result = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return f"judge {judge.id} synthetic calibration result is unavailable"
    return _synthetic_calibration_result_issue(
        judge,
        result=result,
        gate=gate,
        repo_root=repo_root,
    )


def _is_sha256(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _synthetic_calibration_binding_issue(
    judge: ComparisonEvaluatorV1,
    *,
    gate: Mapping[str, Any],
    repo_root: Path,
) -> str | None:
    if gate.get("response_request_mode") != (
        "anthropic_json_schema_no_thinking_v2"
    ):
        return f"judge {judge.id} synthetic response request mode is unsupported"
    if gate.get("response_validator_version") != 2:
        return f"judge {judge.id} synthetic response validator is unsupported"
    if gate.get("response_schema_digest") != _synthetic_response_schema_digest(
        judge.dimensions
    ):
        return f"judge {judge.id} synthetic response schema digest does not match"
    raw_runner_path = str(gate.get("runner_artifact_path") or "")
    if raw_runner_path != (
        "examples/comparisons/community-skill-upgrades/"
        "run_synthetic_calibration.py"
    ):
        return f"judge {judge.id} synthetic calibration runner path is invalid"
    runner_sha256 = str(gate.get("runner_artifact_sha256") or "")
    if not _is_sha256(runner_sha256):
        return f"judge {judge.id} synthetic calibration runner digest is invalid"
    try:
        runner_path = _safe_input_path(
            Path(raw_runner_path),
            repo_root,
            "judge synthetic calibration runner",
        )
        if _sha256_path(runner_path) != runner_sha256:
            return f"judge {judge.id} synthetic calibration runner drifted"
    except (FileNotFoundError, ValueError):
        return f"judge {judge.id} synthetic calibration runner is unavailable"
    raw_validator_path = str(gate.get("validator_artifact_path") or "")
    if raw_validator_path != (
        "examples/comparisons/community-skill-upgrades/validate_campaign.py"
    ):
        return f"judge {judge.id} synthetic calibration validator path is invalid"
    validator_sha256 = str(gate.get("validator_artifact_sha256") or "")
    if not _is_sha256(validator_sha256):
        return f"judge {judge.id} synthetic calibration validator digest is invalid"
    try:
        validator_path = _safe_input_path(
            Path(raw_validator_path),
            repo_root,
            "judge synthetic calibration metrics validator",
        )
        if _sha256_path(validator_path) != validator_sha256:
            return f"judge {judge.id} synthetic calibration validator drifted"
    except (FileNotFoundError, ValueError):
        return f"judge {judge.id} synthetic calibration validator is unavailable"
    if str(gate.get("preview_digest") or "") != (
        _synthetic_calibration_preview_digest(gate)
    ):
        return f"judge {judge.id} synthetic execution preview digest is invalid"
    if "prior_runs" in gate:
        ledger_issue = _synthetic_prior_runs_ledger_issue(gate, repo_root=repo_root)
        if ledger_issue:
            return f"judge {judge.id} {ledger_issue}"
    allocation = gate.get("campaign_allocation_usd")
    prior_failures = gate.get("prior_failed_requests")
    prior_reserve = gate.get("prior_accounted_reserve_usd")
    run_cap = gate.get("maximum_cost_usd")
    if (
        not isinstance(allocation, int | float)
        or isinstance(allocation, bool)
        or float(allocation) <= 0
        or not isinstance(prior_failures, int)
        or isinstance(prior_failures, bool)
        or prior_failures < 0
        or not isinstance(prior_reserve, int | float)
        or isinstance(prior_reserve, bool)
        or float(prior_reserve) < 0
        or not isinstance(run_cap, int | float)
        or isinstance(run_cap, bool)
        or float(run_cap) <= 0
        or round(float(prior_reserve) + float(run_cap), 6)
        != round(float(allocation), 6)
    ):
        return f"judge {judge.id} synthetic execution budget is invalid"
    return None


def _synthetic_prior_runs_ledger_issue(
    gate: Mapping[str, Any], *, repo_root: Path
) -> str | None:
    raw_path = str(gate.get("prior_runs_ledger_path") or "")
    relative = Path(raw_path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix()
        != "examples/comparisons/community-skill-upgrades/calibration-prior-runs.json"
    ):
        return "synthetic prior-runs ledger path is invalid"
    expected_sha256 = str(gate.get("prior_runs_ledger_sha256") or "")
    expected_digest = str(gate.get("prior_runs_ledger_digest") or "")
    if not _is_sha256(expected_sha256) or not _is_sha256(expected_digest):
        return "synthetic prior-runs ledger digest is invalid"
    try:
        path = _safe_input_path(relative, repo_root, "synthetic prior-runs ledger")
        if _sha256_path(path) != expected_sha256:
            return "synthetic prior-runs ledger bytes drifted"
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return "synthetic prior-runs ledger is unavailable"
    if not isinstance(value, Mapping) or stable_digest(value) != expected_digest:
        return "synthetic prior-runs ledger content drifted"
    runs = value.get("runs")
    prior_runs = gate.get("prior_runs")
    if (
        value.get("kind") != "synthetic_calibration_prior_runs_ledger"
        or not isinstance(runs, list)
        or not isinstance(prior_runs, int)
        or isinstance(prior_runs, bool)
        or prior_runs != len(runs)
        or prior_runs != gate.get("prior_failed_requests")
        or value.get("prior_runs") != prior_runs
        or float(value.get("total_accounted_reserve_usd") or -1)
        != float(gate.get("prior_accounted_reserve_usd") or -2)
        or round(
            float(value.get("total_accounted_reserve_usd") or 0)
            + float(value.get("remaining_budget_usd") or 0),
            6,
        )
        != round(float(gate.get("campaign_allocation_usd") or 0), 6)
        or float(value.get("remaining_budget_usd") or 0)
        != float(gate.get("maximum_cost_usd") or -1)
    ):
        return "synthetic prior-runs ledger accounting is invalid"
    return None


def _synthetic_response_schema_digest(dimensions: Sequence[str]) -> str:
    return stable_digest(
        {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "scores",
                "overall_assessment",
                "uncertainty",
                "missing_evidence",
                "rationale",
            ],
            "properties": {
                "scores": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(dimensions),
                    "properties": {
                        dimension: {"type": "number"}
                        for dimension in dimensions
                    },
                },
                "overall_assessment": {
                    "type": "string",
                    "description": (
                        "Brief evidence-bounded assessment; requested maximum "
                        "500 characters."
                    ),
                },
                "uncertainty": {"type": "number"},
                "missing_evidence": {"type": "boolean"},
                "rationale": {
                    "type": "string",
                    "description": (
                        "Brief evidence-bounded rationale; requested maximum "
                        "500 characters."
                    ),
                },
            },
        }
    )


def _synthetic_calibration_preview_digest(gate: Mapping[str, Any]) -> str:
    value = {
        "schema_version": 1,
        "kind": "synthetic_judge_calibration_preview",
        "id": "community-skill-judge-calibration-v1",
        "model": gate["model"],
        "cases_artifact_path": gate["cases_artifact_path"],
        "cases_artifact_sha256": gate["cases_artifact_sha256"],
        "cases_digest": gate["cases_digest"],
        "rubric_digest": gate["rubric_digest"],
        "runner_artifact_path": gate["runner_artifact_path"],
        "runner_artifact_sha256": gate["runner_artifact_sha256"],
        "validator_artifact_path": gate["validator_artifact_path"],
        "validator_artifact_sha256": gate["validator_artifact_sha256"],
        "passing_labels": gate["passing_labels"],
        "score_threshold": gate["score_threshold"],
        "critical_policy": gate["critical_policy"],
        "metric_policy": gate["metric_policy"],
        "minimum_balanced_accuracy": gate["minimum_balanced_accuracy"],
        "balanced_accuracy_scopes": gate["balanced_accuracy_scopes"],
        "response_schema_digest": gate["response_schema_digest"],
        "response_request_mode": gate["response_request_mode"],
        "response_validator_version": gate["response_validator_version"],
        "requests": 48,
        "maximum_prompt_characters": 12_000,
        "maximum_output_tokens_per_request": 1_200,
        "automatic_retries": 0,
        "campaign_allocation_usd": gate["campaign_allocation_usd"],
        "prior_failed_requests": gate["prior_failed_requests"],
        "prior_accounted_reserve_usd": gate[
            "prior_accounted_reserve_usd"
        ],
        "maximum_cost_usd": gate["maximum_cost_usd"],
        "serialized_input_fields": [
            "public_task",
            "response",
            "permitted_evidence",
            "rubric",
        ],
        "human_calibration_satisfied": False,
    }
    if "prior_runs" in gate:
        value.update(
            {
                "prior_runs_ledger_path": gate["prior_runs_ledger_path"],
                "prior_runs_ledger_sha256": gate["prior_runs_ledger_sha256"],
                "prior_runs_ledger_digest": gate["prior_runs_ledger_digest"],
                "prior_runs": gate["prior_runs"],
            }
        )
    return stable_digest(value)


def _frozen_synthetic_calibration_cases(
    gate: Mapping[str, Any],
    *,
    repo_root: Path,
) -> list[dict[str, Any]]:
    raw_path = str(gate.get("cases_artifact_path") or "")
    relative = Path(raw_path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.parts[:3]
        != ("examples", "comparisons", "community-skill-upgrades")
    ):
        raise ValueError("synthetic calibration cases path is outside the campaign")
    expected_sha256 = str(gate.get("cases_artifact_sha256") or "")
    if not _is_sha256(expected_sha256):
        raise ValueError("synthetic calibration cases artifact digest is invalid")
    path = _safe_input_path(relative, repo_root, "synthetic calibration cases")
    if _sha256_path(path) != expected_sha256:
        raise ValueError("synthetic calibration cases artifact drifted")
    rows = _read_jsonl(path, "synthetic calibration cases")
    if stable_digest(rows) != gate.get("cases_digest"):
        raise ValueError("synthetic calibration cases content drifted")
    if len(rows) != int(gate["examples"]):
        raise ValueError("synthetic calibration cases have the wrong cohort")
    ids: set[str] = set()
    split_counts = Counter()
    for row in rows:
        case_id = str(row.get("id") or "")
        split = str(row.get("split") or "")
        reference = row.get("authored_reference")
        if (
            not case_id
            or case_id in ids
            or split not in {"calibration", "holdout"}
            or not isinstance(reference, Mapping)
            or reference.get("label") not in {"pass", "fail"}
            or not isinstance(reference.get("critical_false_pass"), bool)
        ):
            raise ValueError("synthetic calibration case contract is invalid")
        ids.add(case_id)
        split_counts[split] += 1
    if split_counts != {
        "calibration": int(gate["calibration_examples"]),
        "holdout": int(gate["holdout_examples"]),
    }:
        raise ValueError("synthetic calibration split disagrees")
    return rows


def _synthetic_classification_rates(
    actual: Sequence[str],
    predicted: Sequence[str],
) -> dict[str, float]:
    positives = actual.count("pass")
    negatives = actual.count("fail")
    if positives == 0 or negatives == 0:
        raise ValueError("synthetic calibration split is not balanced")
    true_positives = sum(
        left == right == "pass"
        for left, right in zip(actual, predicted, strict=True)
    )
    true_negatives = sum(
        left == right == "fail"
        for left, right in zip(actual, predicted, strict=True)
    )
    return {
        "true_positive_rate": true_positives / positives,
        "true_negative_rate": true_negatives / negatives,
        "balanced_accuracy": (
            (true_positives / positives) + (true_negatives / negatives)
        )
        / 2,
    }


def _recompute_synthetic_calibration_metrics(
    judge: ComparisonEvaluatorV1,
    *,
    cases: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(results) != len(cases):
        raise ValueError("synthetic calibration rows are incomplete")
    actual: list[str] = []
    predicted: list[str] = []
    splits: list[str] = []
    critical_false_passes = 0
    for case, result in zip(cases, results, strict=True):
        if not isinstance(result, Mapping):
            raise ValueError("synthetic calibration result row is invalid")
        if (
            result.get("case_id") != case.get("id")
            or result.get("repository_id") != case.get("repository_id")
            or result.get("split") != case.get("split")
        ):
            raise ValueError("synthetic calibration case identities disagree")
        scores = result.get("dimension_scores")
        if not isinstance(scores, Mapping) or set(scores) != set(judge.dimensions):
            raise ValueError("synthetic calibration judge dimensions disagree")
        normalized_scores: list[float] = []
        for dimension in judge.dimensions:
            value = scores[dimension]
            if (
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0 <= float(value) <= 1
            ):
                raise ValueError("synthetic calibration judge score is invalid")
            normalized_scores.append(float(value))
        label = str(result.get("overall_label") or "")
        if label != _judge_review_label(
            sum(normalized_scores) / len(normalized_scores)
        ):
            raise ValueError("synthetic calibration anchored label disagrees")
        missing_evidence = result.get("missing_evidence")
        if not isinstance(missing_evidence, bool):
            raise ValueError("synthetic calibration missing-evidence flag is invalid")
        predicted_label = (
            "pass"
            if label in COMPARISON_JUDGE_PASSING_LABELS and not missing_evidence
            else "fail"
        )
        reference = _mapping(case.get("authored_reference"), "authored reference")
        actual_label = str(reference["label"])
        actual.append(actual_label)
        predicted.append(predicted_label)
        splits.append(str(case["split"]))
        if (
            reference["critical_false_pass"] is True
            and actual_label == "fail"
            and predicted_label == "pass"
        ):
            critical_false_passes += 1
    rates = _synthetic_classification_rates(actual, predicted)
    split_rates = {
        split: _synthetic_classification_rates(
            [
                label
                for label, item_split in zip(actual, splits, strict=True)
                if item_split == split
            ],
            [
                label
                for label, item_split in zip(predicted, splits, strict=True)
                if item_split == split
            ],
        )
        for split in ("calibration", "holdout")
    }
    passed = bool(
        rates["balanced_accuracy"]
        >= COMPARISON_JUDGE_MINIMUM_BALANCED_ACCURACY
        and all(
            item["balanced_accuracy"]
            >= COMPARISON_JUDGE_MINIMUM_BALANCED_ACCURACY
            for item in split_rates.values()
        )
        and critical_false_passes == 0
    )
    return {
        "examples": len(results),
        "true_positive_rate": round(rates["true_positive_rate"], 6),
        "true_negative_rate": round(rates["true_negative_rate"], 6),
        "calibration_true_positive_rate": round(
            split_rates["calibration"]["true_positive_rate"], 6
        ),
        "calibration_true_negative_rate": round(
            split_rates["calibration"]["true_negative_rate"], 6
        ),
        "holdout_true_positive_rate": round(
            split_rates["holdout"]["true_positive_rate"], 6
        ),
        "holdout_true_negative_rate": round(
            split_rates["holdout"]["true_negative_rate"], 6
        ),
        "critical_false_passes": critical_false_passes,
        "balanced_accuracy": round(rates["balanced_accuracy"], 6),
        "calibration_balanced_accuracy": round(
            split_rates["calibration"]["balanced_accuracy"], 6
        ),
        "holdout_balanced_accuracy": round(
            split_rates["holdout"]["balanced_accuracy"], 6
        ),
        "synthetic_thresholds_passed": passed,
    }


def _synthetic_calibration_result_issue(  # noqa: C901 - one bounded receipt audit
    judge: ComparisonEvaluatorV1,
    *,
    result: Any,
    gate: Mapping[str, Any],
    repo_root: Path,
) -> str | None:
    if not isinstance(result, Mapping):
        return f"judge {judge.id} synthetic calibration result must be a mapping"
    supplied_digest = str(result.get("result_digest") or "")
    unsigned = dict(result)
    unsigned.pop("result_digest", None)
    if not _is_sha256(supplied_digest) or supplied_digest != stable_digest(unsigned):
        return f"judge {judge.id} synthetic calibration result digest disagrees"
    if result.get("kind") != "synthetic_gold_diagnostic":
        return f"judge {judge.id} synthetic calibration result kind is unsupported"
    if result.get("status") != "completed":
        return f"judge {judge.id} synthetic calibration did not complete"
    if result.get("model") != gate.get("model"):
        return f"judge {judge.id} synthetic calibration model drifted"
    for field_name in (
        "cases_artifact_path",
        "cases_artifact_sha256",
        "cases_digest",
        "rubric_digest",
        "runner_artifact_path",
        "runner_artifact_sha256",
        "response_schema_digest",
        "response_request_mode",
        "response_validator_version",
        "campaign_allocation_usd",
        "prior_failed_requests",
        "prior_accounted_reserve_usd",
        "preview_digest",
    ):
        if result.get(field_name) != gate.get(field_name):
            return f"judge {judge.id} synthetic calibration {field_name} drifted"
    approval_digest = str(result.get("approval_digest") or "")
    if not _is_sha256(approval_digest):
        return f"judge {judge.id} synthetic calibration approval is unavailable"
    try:
        ApprovalLedger(StudyStore(repo_root).path).verify_claim(
            approval_digest=approval_digest,
            subject_kind="experiment",
            preview_digest=str(gate["preview_digest"]),
            subject_id=f"calibration-{str(gate['preview_digest'])[:20]}",
            required_cells=int(gate["examples"]),
            required_cost_usd=float(gate["maximum_cost_usd"]),
        )
    except Exception:
        return f"judge {judge.id} synthetic calibration approval provenance is invalid"
    if (
        int(result.get("requested_cases") or 0) != int(gate["examples"])
        or int(result.get("completed_cases") or 0) != int(gate["examples"])
    ):
        return f"judge {judge.id} synthetic calibration row count disagrees"
    results = result.get("results")
    if not isinstance(results, list) or len(results) != int(gate["examples"]):
        return f"judge {judge.id} synthetic calibration rows are incomplete"
    try:
        cases = _frozen_synthetic_calibration_cases(gate, repo_root=repo_root)
        recomputed_metrics = _recompute_synthetic_calibration_metrics(
            judge,
            cases=cases,
            results=results,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return f"judge {judge.id} synthetic calibration rows do not match frozen cases"
    metrics = result.get("synthetic_metrics")
    if not isinstance(metrics, Mapping):
        return f"judge {judge.id} synthetic calibration metrics are unavailable"
    if dict(metrics) != recomputed_metrics:
        return f"judge {judge.id} synthetic calibration metrics disagree"
    minimum_balanced_accuracy = float(gate["minimum_balanced_accuracy"])
    for key in (
        "balanced_accuracy",
        "calibration_balanced_accuracy",
        "holdout_balanced_accuracy",
    ):
        if float(metrics.get(key) or 0) < minimum_balanced_accuracy:
            return f"judge {judge.id} synthetic calibration is below {key} threshold"
    if int(metrics.get("critical_false_passes") or 0) > int(
        gate["maximum_critical_false_passes"]
    ):
        return f"judge {judge.id} synthetic calibration has critical false passes"
    if metrics.get("synthetic_thresholds_passed") is not True:
        return f"judge {judge.id} synthetic calibration did not pass"
    if (
        float(result.get("budget_ceiling_usd") or 0)
        != float(gate["maximum_cost_usd"])
        or result.get("observed_cost_usd") is not None
        or float(result.get("accounted_cost_usd") or 0)
        != float(gate["maximum_cost_usd"])
        or float(result.get("cumulative_accounted_cost_usd") or 0)
        != float(gate["campaign_allocation_usd"])
        or round(
            float(result.get("accounted_cost_usd") or 0)
            + float(result.get("prior_accounted_reserve_usd") or 0),
            6,
        )
        != float(result.get("cumulative_accounted_cost_usd") or 0)
    ):
        return f"judge {judge.id} synthetic calibration cost contract disagrees"
    receipt = result.get("receipt")
    if not isinstance(receipt, Mapping):
        return f"judge {judge.id} synthetic calibration receipt is unavailable"
    receipt_unsigned = dict(receipt)
    receipt_digest = str(receipt_unsigned.pop("receipt_digest", "") or "")
    if not _is_sha256(receipt_digest) or receipt_digest != stable_digest(receipt_unsigned):
        return f"judge {judge.id} synthetic calibration receipt digest disagrees"
    if (
        receipt.get("preview_digest") != gate.get("preview_digest")
        or receipt.get("approval_digest") != approval_digest
        or receipt.get("secret_value_scan") != "passed"
        or receipt.get("cases_artifact_path")
        != gate.get("cases_artifact_path")
        or receipt.get("cases_artifact_sha256")
        != gate.get("cases_artifact_sha256")
        or receipt.get("cases_digest") != gate.get("cases_digest")
        or receipt.get("rubric_digest") != gate.get("rubric_digest")
        or receipt.get("runner_artifact_path")
        != gate.get("runner_artifact_path")
        or receipt.get("runner_artifact_sha256")
        != gate.get("runner_artifact_sha256")
        or receipt.get("response_schema_digest")
        != gate.get("response_schema_digest")
        or receipt.get("response_request_mode")
        != gate.get("response_request_mode")
        or receipt.get("response_validator_version")
        != gate.get("response_validator_version")
        or receipt.get("campaign_allocation_usd")
        != gate.get("campaign_allocation_usd")
        or receipt.get("prior_failed_requests")
        != gate.get("prior_failed_requests")
        or receipt.get("prior_accounted_reserve_usd")
        != gate.get("prior_accounted_reserve_usd")
        or float(receipt.get("run_accounted_cost_usd") or 0)
        != float(gate["maximum_cost_usd"])
        or float(receipt.get("cumulative_accounted_cost_usd") or 0)
        != float(gate["campaign_allocation_usd"])
        or int(receipt.get("request_count") or 0) != int(gate["examples"])
        or int(receipt.get("maximum_requests") or 0) != int(gate["examples"])
    ):
        return f"judge {judge.id} synthetic calibration receipt is not qualified"
    return None


def _require_judge_execution_calibrations(
    spec: ComparisonSpecV1,
    *,
    repo_root: Path,
    approved_inputs: Mapping[str, Any] | None,
) -> None:
    issues = [
        issue
        for judge in spec.evaluators
        if judge.type == "llm_judge"
        for issue in (
            _judge_execution_calibration_issue(
                judge,
                repo_root=repo_root,
                approved_inputs=approved_inputs,
            ),
        )
        if issue
    ]
    if issues:
        raise RuntimeError(
            "judge calibration must pass before Agent trials: " + "; ".join(issues)
        )


def _comparison_judge_qualification(
    judge: ComparisonEvaluatorV1,
    *,
    repo_root: Path,
    approved_inputs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    contract_digest = _judge_contract_digest(judge)
    value, report_sha256 = _judge_calibration_artifact(
        judge,
        repo_root=repo_root,
        approved_inputs=approved_inputs,
    )
    issue = _judge_calibration_value_issue(judge, value)
    review_status = (
        str(value.get("review_status") or "") if isinstance(value, Mapping) else ""
    )
    status = (
        review_status
        if review_status in {"pending_human_review", "adjudicated"}
        else ("missing" if not judge.calibration else "invalid")
    )
    passed = issue is None
    cases_digest = (
        str(value.get("cases_digest") or "") if isinstance(value, Mapping) else ""
    )
    execution_issue = _judge_execution_calibration_issue(
        judge,
        repo_root=repo_root,
        approved_inputs=approved_inputs,
    )
    return {
        "judge_id": judge.id,
        "profile": str(judge.profile or ""),
        "contract_digest": contract_digest,
        "dimensions": list(judge.dimensions),
        "calibration": {
            "status": status,
            "report_sha256": report_sha256,
            "cases_digest": cases_digest or None,
            "passed": passed,
        },
        "execution_calibration": {
            "status": "passed" if execution_issue is None else "blocked",
            **({"reason": execution_issue} if execution_issue else {}),
            "claim_scope": (
                "human_qualified" if passed else "advisory_same_family"
            ),
        },
    }


def _judge_contract_digest(judge: ComparisonEvaluatorV1) -> str:
    return stable_digest(
        {
            "schema_version": 1,
            "judge_id": judge.id,
            "profile": judge.profile,
            "rubric": judge.rubric,
            "dimensions": list(judge.dimensions),
            "evidence": list(judge.evidence),
        }
    )


def _behavior_diff(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], prefix: str = ""
) -> set[str]:
    result: set[str] = set()
    for key in sorted(set(baseline) | set(candidate)):
        path = f"{prefix}.{key}" if prefix else str(key)
        left = baseline.get(key)
        right = candidate.get(key)
        if isinstance(left, dict) and isinstance(right, dict):
            result.update(_behavior_diff(left, right, path))
        elif left != right:
            result.add(path)
    return result


def _preview_dict(value: PreviewSummary) -> dict[str, Any]:
    return {
        "cells": value.cells,
        "applicable_cells": value.applicable_cells,
        "estimated_trials": value.estimated_trials,
        "harnesses": list(value.harnesses),
        "variants": list(value.variants),
        "systems": list(value.systems),
        "workloads": list(value.workloads),
        "source_evidence_project": value.source_evidence_project,
        "source_evidence_destination": value.source_evidence_destination,
        "evidence_project": value.evidence_project,
        "evidence_destination": value.evidence_destination,
        "matrix_cells": [asdict(item) for item in value.matrix_cells],
    }


def _result_markdown(result: ComparisonResult) -> str:
    invalid_behavior = (
        isinstance(result, ComparisonResultV2 | ComparisonResultV3)
        and result.behavioral_summary.status == "invalid"
    )
    mechanism = "".join(
        (
            f"- {stage.replace('_', ' ').title()}: "
            f"baseline {values['baseline']['observed']}/"
            f"{values['baseline']['applicable']}; "
            f"candidate {values['candidate']['observed']}/"
            f"{values['candidate']['applicable']}\n"
        )
        for stage, values in result.mechanism_summary.items()
    )
    judge = _judge_result_markdown(result)
    pair_rows = (
        (
            item.task_id,
            item.harness,
            item.attempt,
            item.baseline_passed,
            item.candidate_passed,
            item.status,
        )
        if isinstance(item, PairedCaseV2)
        else (
            item.task_id,
            item.harness,
            item.attempt,
            item.baseline.passed if item.baseline is not None else None,
            item.candidate.passed if item.candidate is not None else None,
            item.status,
        )
        if isinstance(item, PairedCaseV3)
        else (
            str(item["task_id"]),
            str(item["harness"]),
            int(item["attempt"]),
            item.get("baseline_passed"),
            item.get("candidate_passed"),
            str(item["status"]),
        )
        for item in result.paired_cases
    )
    pairs = (
        "Aligned outcome claims are suppressed because behavioral evidence is invalid.\n"
        if invalid_behavior
        else (
            "| Task | Harness | Attempt | Baseline | Candidate | Change |\n"
            "| --- | --- | ---: | --- | --- | --- |\n"
            + "".join(
                f"| {task} | {harness} | {attempt} | {_pass_label(baseline)} | "
                f"{_pass_label(candidate)} | {status} |\n"
                for task, harness, attempt, baseline, candidate, status in pair_rows
            )
        )
    )
    evidence = (
        "- Attempt navigation is suppressed because behavioral evidence is invalid.\n"
        if invalid_behavior
        else "".join(
            f"- [{item['label']}]({item['url']})\n"
            for item in result.evidence_links
        )
    )
    projects = result.evidence_project or ", ".join(
        result.operational_summary.get("evidence_projects") or ()
    )
    mcp_tools = result.operational_summary.get("mcp_tool_usage") or {}
    mcp_usage = "".join(
        f"- {variant}: "
        + (
            ", ".join(f"`{name}` × {count}" for name, count in tools.items())
            if tools
            else "no observed MCP calls"
        )
        + "\n"
        for variant, tools in mcp_tools.items()
    )
    candidate_sources = (
        ", ".join(
            f"`{item.id}` at `{item.version_identity}` "
            f"(runtime `{item.runtime_digest}`)"
            for item in result.candidate_source_revisions
        )
        if isinstance(result, ComparisonResultV2 | ComparisonResultV3)
        and result.candidate_source_revisions
        else "unavailable"
    )
    release_candidate = (
        f"`{result.decision.candidate_sha}`"
        if isinstance(result, ComparisonResultV2 | ComparisonResultV3)
        and result.decision.candidate_sha
        else "not applicable (no package release policy)"
    )
    decision = (
        (
            f"- Behavioral verdict: "
            f"**{result.behavioral_summary.status.upper()}**\n"
            f"- Behavioral recommendation: "
            f"{result.behavioral_summary.recommendation}\n"
            f"- Release decision: **{result.decision.status.upper()}**\n"
            f"- Evidence grade: **{result.decision.evidence_grade}**\n"
            f"- Candidate source revisions: {candidate_sources}\n"
            f"- Governed release candidate SHA: {release_candidate}\n"
            f"- Recommendation: {result.decision.recommendation}\n"
        )
        if isinstance(result, ComparisonResultV2 | ComparisonResultV3)
        else "- Release decision: unavailable in V1 result\n"
    )
    return (
        f"# {result.comparison_id}\n\n"
        "## Decision summary\n\n"
        + decision
        + f"- Rows: {result.rows}\n"
        f"- Baseline passed: {result.baseline_passed}\n"
        f"- Candidate passed: {result.candidate_passed}\n"
        f"- Improved pairs: {result.improved}\n"
        f"- Regressed pairs: {result.regressed}\n"
        f"- Mixed pairs: "
        f"{result.mixed if isinstance(result, ComparisonResultV2 | ComparisonResultV3) else 0}\n"
        f"- Unchanged pairs: {result.unchanged}\n"
        f"- Incomplete pairs: {result.incomplete}\n\n"
        f"- Required evaluations incomplete: "
        f"{result.required_evaluations_incomplete}\n"
        f"- Evidence project: {projects or 'unavailable'}\n\n"
        "## Aligned cases\n\n"
        + pairs
        + "\n"
        "## Operational health\n\n"
        f"- Infrastructure failures: "
        f"{result.operational_summary['infrastructure_failures']}\n"
        f"- Agent timeouts: "
        f"{result.operational_summary.get('agent_timeouts', 0)}\n"
        f"- Execution states: "
        f"`{json.dumps(result.operational_summary['execution_states'], sort_keys=True)}`\n"
        f"- Evidence states: "
        f"`{json.dumps(result.operational_summary['evidence_states'], sort_keys=True)}`\n"
        f"- Observed cost: "
        f"{result.operational_summary['observed_cost_usd'] if result.operational_summary['observed_cost_usd'] is not None else 'unavailable'}\n"
        f"- Accounted cost (including judge reserves): "
        f"{result.operational_summary.get('accounted_cost_usd') if result.operational_summary.get('accounted_cost_usd') is not None else 'unavailable'}\n\n"
        "## Mechanism evidence\n\n"
        + (mechanism or "No mechanism evidence was available.\n")
        + "\n### MCP tool use\n\n"
        + (mcp_usage or "No MCP tool-use evidence was available.\n")
        + "\n## Blind judge\n\n"
        + judge
        + "\n## Open the evidence\n\n"
        + (evidence or "No safe evidence links were available.\n")
        + "\n"
        "## Limitations\n\n"
        + "".join(f"- {item}\n" for item in result.limitations)
    )


def _judge_result_markdown(result: ComparisonResult) -> str:
    """Render bounded judge evidence without turning it into a release gate."""

    if result.judge_summary.get("status") == "not_used":
        return "No blind judge was used.\n"
    reviews: list[tuple[str, str, JudgeReviewV1]] = []
    if isinstance(result, ComparisonResultV2 | ComparisonResultV3):
        for pair in result.paired_cases:
            for arm, attempt in (
                ("baseline", pair.baseline),
                ("candidate", pair.candidate),
            ):
                if attempt is None:
                    continue
                for review in attempt.judge_reviews.values():
                    reviews.append((pair.task_id, arm, review))
    if not reviews:
        return (
            "Blind-judge dimensions are available in `result.json`; no safe "
            "anchored review was publishable.\n\n"
            "Judge evidence is advisory. Deterministic correctness and safety "
            "gates remain authoritative.\n"
        )
    rows = "".join(
        "| "
        + " | ".join(
            (
                _markdown_cell(task_id),
                arm,
                review.label,
                "missing" if review.missing_evidence else "complete",
                _judge_cost_label(review),
                _markdown_cell(review.reason),
            )
        )
        + " |\n"
        for task_id, arm, review in reviews
    )
    return (
        "Judge evidence is advisory because the Agent and judge use the same "
        "model family. Deterministic correctness and safety gates remain "
        "authoritative.\n\n"
        "| Task | Arm | Usefulness | Evidence | Judge cost | Reason |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        + rows
    )


def _judge_cost_label(review: JudgeReviewV1) -> str:
    reserve = (
        f"${review.accounted_reserve_usd:.2f} reserved"
        if review.accounted_reserve_usd is not None
        else "reserve unavailable"
    )
    if review.cost_status == "observed" and review.observed_cost_usd is not None:
        return f"${review.observed_cost_usd:.4f} observed; {reserve}"
    return f"observed unavailable; {reserve}"


def _markdown_cell(value: str) -> str:
    return " ".join(value.replace("|", "\\|").split())


def _pass_label(value: Any) -> str:
    return "pass" if value is True else "fail" if value is False else "incomplete"


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{label} must contain at least one row")
    return rows


def _portable_input_path(
    value: Any, source: Path, repo_root: Path, label: str
) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} path is required")
    path = Path(text)
    resolved = (source / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        relative = resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must be inside the repository") from exc
    if ".." in relative.parts:
        raise ValueError(f"{label} path is unsafe")
    return relative.as_posix()


def _safe_input_path(path: Path, repo_root: Path, label: str) -> Path:
    resolved = path if path.is_absolute() else repo_root / path
    resolved = resolved.resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must be inside the repository") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _safe_repository_output_path(
    path: Path,
    repo_root: Path,
    label: str,
) -> Path:
    selected = path if path.is_absolute() else repo_root / path
    if selected.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {selected}")
    resolved = selected.resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must be inside the repository") from exc
    if resolved.exists() and not resolved.is_file():
        raise ValueError(f"{label} must be a regular file: {resolved}")
    return resolved


def _write_consistent_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} is not valid JSON: {path}") from exc
        if existing != _json_value(value):
            raise ValueError(
                f"{label} already exists with different immutable content: {path}"
            )
        return
    atomic_write_json(path, value)


def _load_infrastructure_receipt(
    path: Path,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    selected = _safe_input_path(path, repo_root, "infrastructure receipt")
    try:
        raw = json.loads(selected.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("infrastructure receipt is not valid JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("infrastructure receipt must be an object")
    supplied = str(raw.get("receipt_digest") or "")
    unsigned = dict(raw)
    unsigned["receipt_digest"] = ""
    if len(supplied) != 64 or stable_digest(unsigned) != supplied:
        raise ValueError("infrastructure receipt digest does not match")
    conformance = _mapping_or_empty(raw.get("infrastructure_conformance"))
    gates = conformance.get("gates")
    if not isinstance(gates, list) or not gates:
        raise ValueError("infrastructure receipt has no conformance gates")
    gate_ids: list[str] = []
    statuses: list[str] = []
    for item in gates:
        if not isinstance(item, Mapping):
            raise ValueError("infrastructure conformance gate is invalid")
        gate_id = str(item.get("id") or "")
        status = str(item.get("status") or "")
        if not gate_id or status not in {"passed", "failed", "unavailable"}:
            raise ValueError("infrastructure conformance gate is invalid")
        gate_ids.append(gate_id)
        statuses.append(status)
    if len(gate_ids) != len(set(gate_ids)):
        raise ValueError("infrastructure conformance gate ids must be unique")
    complete = not any(status != "passed" for status in statuses)
    if bool(conformance.get("complete")) != complete:
        raise ValueError("infrastructure conformance completeness disagrees")
    return raw


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be an object")
    return raw


def _load_digest_receipt(path: Path, label: str) -> dict[str, Any]:
    raw = _load_json_object(path, label)
    supplied = str(raw.get("receipt_digest") or "")
    unsigned = dict(raw)
    unsigned["receipt_digest"] = ""
    if len(supplied) != 64 or stable_digest(unsigned) != supplied:
        raise ValueError(f"{label} digest does not match")
    return raw


def _artifact_digest(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned[field] = ""
    return stable_digest(unsigned)


def _verify_artifact(value: Mapping[str, Any], field: str, label: str) -> None:
    supplied = str(value.get(field) or "")
    if len(supplied) != 64 or _artifact_digest(value, field) != supplied:
        raise ValueError(f"{label} digest does not match")


def _sha256_path(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )


def _json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _write_immutable_text(
    path: Path,
    value: str,
    *,
    expected_sha256: str,
    mode: int,
    label: str,
) -> None:
    _write_immutable_bytes(
        path,
        value.encode(),
        expected_sha256=expected_sha256,
        mode=mode,
        label=label,
    )


def _write_immutable_bytes(
    path: Path,
    value: bytes,
    *,
    expected_sha256: str,
    mode: int,
    label: str,
) -> None:
    if (
        len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
        or hashlib.sha256(value).hexdigest() != expected_sha256
    ):
        raise ValueError(f"{label} digest does not match approved content")
    if path.is_symlink():
        raise ValueError(f"{label} may not be a symlink")
    if path.exists():
        if not path.is_file() or _sha256_path(path) != expected_sha256:
            raise ValueError(f"{label} immutable copy changed")
        path.chmod(mode)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(value)
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _schema(raw: Mapping[str, Any], label: str) -> int:
    version = raw.get("schema_version")
    if version not in COMPARISON_READABLE_SCHEMA_VERSIONS:
        raise ValueError(
            f"{label} schema_version must be one of "
            f"{sorted(COMPARISON_READABLE_SCHEMA_VERSIONS)}"
        )
    return int(version)


def _commit_sha(value: Any) -> str:
    sha = str(value or "").strip().lower()
    if len(sha) != 40 or any(character not in "0123456789abcdef" for character in sha):
        raise ValueError("candidate SHA must be an exact 40-character commit SHA")
    return sha


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(
    value: Any, label: str, *, allow_empty: bool = False
) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    if not value and not allow_empty:
        raise ValueError(f"{label} must not be empty")
    return value


def _string_tuple(
    value: Any, label: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    values = _sequence(value, label, allow_empty=allow_empty)
    if not all(isinstance(item, str) and item.strip() for item in values):
        raise ValueError(f"{label} values must be non-empty strings")
    return tuple(str(item).strip() for item in values)


def _text(value: Any, label: str, limit: int) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is required")
    if len(text.encode()) > limit:
        raise ValueError(f"{label} exceeds {limit} bytes")
    return text


def _optional_text(value: Any, label: str, limit: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text.encode()) > limit:
        raise ValueError(f"{label} exceeds {limit} bytes")
    return text


def _evidence_project(
    value: Any,
    *,
    label: str = "evidence_project",
) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    parts = text.split("/")
    if len(parts) != 2:
        raise ValueError(
            f"execution {label} must be an exact W&B entity/project slug"
        )
    for index, part in enumerate(parts):
        validate_id(
            part,
            kind=(
                "execution evidence entity"
                if index == 0
                else f"execution {label}"
            ),
        )
    return text


def _declared_evidence_destination(
    value: Any,
    *,
    evidence_project: str | None,
    label: str = "evidence_destination",
) -> EvidenceDestinationV1 | None:
    if value in (None, {}):
        return None
    raw = _mapping(value, f"execution {label}")
    destination = evidence_destination_from_dict(raw)
    if evidence_project is not None and destination.project_slug != evidence_project:
        raise ValueError(
            f"execution {label} must match its project"
        )
    return destination


def _comparison_evidence_environment(
    spec: ComparisonSpecV1,
    env: Mapping[str, str],
) -> dict[str, str]:
    destination = spec.execution.evidence_destination
    if destination is not None:
        return evidence_destination_environment(destination, env)
    return trace_project_environment(spec.execution.evidence_project, env)


def _evidence_destination(value: Any, label: str) -> dict[str, Any]:
    raw = _mapping(value, label)
    _reject_unknown(
        raw,
        {
            "schema_version",
            "entity",
            "project",
            "project_slug",
            "api_base_url",
            "trace_base_url",
            "app_base_url",
            "destination_digest",
        },
        label,
    )
    destination = EvidenceDestinationV1(
        schema_version=int(raw.get("schema_version") or 0),
        entity=str(raw.get("entity") or ""),
        project=str(raw.get("project") or ""),
        api_base_url=str(raw.get("api_base_url") or ""),
        trace_base_url=str(raw.get("trace_base_url") or ""),
        app_base_url=str(raw.get("app_base_url") or ""),
    )
    canonical = destination.to_dict()
    if raw != canonical:
        raise ValueError(f"{label} does not match its canonical identity")
    return canonical


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _non_negative_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be a non-negative number")
    return result


def _reject_unknown(
    value: Mapping[str, Any], allowed: set[str] | frozenset[str], label: str
) -> None:
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ValueError(f"unknown {label} field(s): {', '.join(unknown)}")


def _drop_empty(
    value: dict[str, Any], *, preserve_false: bool = False
) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if item not in (None, "", [], (), {})
        and (preserve_false or item is not False)
    }
