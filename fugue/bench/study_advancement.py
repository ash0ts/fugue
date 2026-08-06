from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fugue.bench.candidates import stable_digest
from fugue.bench.comparison import ComparisonResultV3
from fugue.bench.files import atomic_write_json

AdvancementStatus = Literal[
    "advance_holdout",
    "run_no_skill_diagnostic",
    "stop_critical_regression",
    "stop_task_scorer_repair",
    "stop_invalid",
    "inconclusive",
]

_DIGEST = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class HoldoutExposureAuditV1:
    """Host-owned proof that sealed holdouts were selected without outcome peeking."""

    schema_version: Literal[1]
    kind: Literal["holdout_exposure_audit"]
    study_id: str
    holdout_suite_digest: str
    selected_task_ids: tuple[str, ...]
    searched_project_refs: tuple[str, ...]
    searched_call_count: int
    queried_fields: tuple[str, ...]
    projection_digest: str
    prior_evidence_digest: str
    historical_exposure_receipt_digest: str
    pool_fingerprint_digest: str
    project_coverage_digest: str
    matched_task_ids: tuple[str, ...]
    replacements: tuple[dict[str, str], ...]
    outcome_data_consulted: Literal[False]
    status: Literal["clear", "replaced_exposed"]
    audited_at: str
    expires_at: str
    audit_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != "holdout_exposure_audit":
            raise ValueError("unsupported holdout exposure audit schema")
        _text(self.study_id, "holdout audit study id")
        _digest(self.holdout_suite_digest, "holdout suite digest")
        _unique_text(self.selected_task_ids, "selected holdout task ids", required=True)
        _unique_text(self.searched_project_refs, "searched project refs", required=True)
        _unique_text(self.queried_fields, "holdout audit queried fields", required=True)
        if isinstance(self.searched_call_count, bool) or self.searched_call_count < 0:
            raise ValueError("searched_call_count must be a nonnegative integer")
        _digest(self.projection_digest, "holdout audit projection digest")
        _digest(self.prior_evidence_digest, "prior holdout evidence digest")
        _digest(
            self.historical_exposure_receipt_digest,
            "historical holdout exposure receipt digest",
        )
        _digest(self.pool_fingerprint_digest, "holdout pool fingerprint digest")
        _digest(self.project_coverage_digest, "holdout project coverage digest")
        if self.prior_evidence_digest != self.historical_exposure_receipt_digest:
            raise ValueError("holdout audit prior evidence is not its historical receipt")
        _unique_text(self.matched_task_ids, "matched holdout task ids")
        if self.outcome_data_consulted is not False:
            raise ValueError("holdout selection may not consult treatment outcomes")
        replacement_sources: set[str] = set()
        replacement_targets: set[str] = set()
        for item in self.replacements:
            if set(item) != {"exposed_task_id", "reserve_task_id", "behavior_family"}:
                raise ValueError("holdout replacement fields do not match")
            exposed = _text(item["exposed_task_id"], "exposed holdout task id")
            reserve = _text(item["reserve_task_id"], "reserve holdout task id")
            _text(item["behavior_family"], "holdout replacement behavior family")
            if (
                exposed == reserve
                or exposed in replacement_sources
                or reserve in replacement_targets
            ):
                raise ValueError(
                    "holdout replacements must be unique and non-reflexive"
                )
            replacement_sources.add(exposed)
            replacement_targets.add(reserve)
        if set(self.matched_task_ids) != replacement_sources:
            raise ValueError(
                "every exposed holdout must have exactly one reviewed replacement"
            )
        expected_status = "replaced_exposed" if self.replacements else "clear"
        if self.status != expected_status:
            raise ValueError("holdout exposure status disagrees with replacements")
        audited = _instant(self.audited_at, "holdout audit audited_at")
        expires = _instant(self.expires_at, "holdout audit expires_at")
        if expires <= audited:
            raise ValueError("holdout exposure audit must expire after it was created")
        computed = stable_digest(self.unsigned_dict())
        if self.audit_digest and self.audit_digest != computed:
            raise ValueError("holdout exposure audit digest does not match")
        if not self.audit_digest:
            object.__setattr__(self, "audit_digest", computed)

    def unsigned_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("audit_digest")
        value["selected_task_ids"] = list(self.selected_task_ids)
        value["searched_project_refs"] = list(self.searched_project_refs)
        value["queried_fields"] = list(self.queried_fields)
        value["matched_task_ids"] = list(self.matched_task_ids)
        value["replacements"] = [dict(item) for item in self.replacements]
        return value

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "audit_digest": self.audit_digest}


@dataclass(frozen=True)
class StudyAdvancementDecisionV1:
    """Immutable, deterministic development-to-holdout decision."""

    schema_version: Literal[1]
    kind: Literal["study_advancement_decision"]
    study_id: str
    development_result_digest: str
    development_qualification_digest: str
    preview_digest: str
    status: AdvancementStatus
    repeated_improvements: tuple[str, ...]
    repeated_regressions: tuple[str, ...]
    critical_blockers: tuple[str, ...]
    mechanism_gate: Literal["passed", "failed", "unavailable"]
    holdout_suite_digest: str | None
    holdout_exposure_audit_digest: str | None
    next_action: str
    decision_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != "study_advancement_decision":
            raise ValueError("unsupported Study advancement decision schema")
        _text(self.study_id, "advancement study id")
        for value, label in (
            (self.development_result_digest, "development result digest"),
            (self.development_qualification_digest, "development qualification digest"),
            (self.preview_digest, "development preview digest"),
        ):
            _digest(value, label)
        _unique_text(self.repeated_improvements, "repeated improvements")
        _unique_text(self.repeated_regressions, "repeated regressions")
        _unique_text(self.critical_blockers, "critical blockers")
        _text(self.next_action, "advancement next action")
        if (self.holdout_suite_digest is None) != (
            self.holdout_exposure_audit_digest is None
        ):
            raise ValueError("holdout suite and exposure audit must be bound together")
        if self.holdout_suite_digest is not None:
            _digest(self.holdout_suite_digest, "holdout suite digest")
            _digest(self.holdout_exposure_audit_digest, "holdout exposure audit digest")
        if self.status == "advance_holdout" and self.holdout_suite_digest is None:
            raise ValueError(
                "holdout advancement requires a frozen, exposure-audited suite"
            )
        computed = stable_digest(self.unsigned_dict())
        if self.decision_digest and self.decision_digest != computed:
            raise ValueError("Study advancement decision digest does not match")
        if not self.decision_digest:
            object.__setattr__(self, "decision_digest", computed)

    def unsigned_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("decision_digest")
        for name in (
            "repeated_improvements",
            "repeated_regressions",
            "critical_blockers",
        ):
            value[name] = list(getattr(self, name))
        return value

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "decision_digest": self.decision_digest}


def build_holdout_exposure_audit(
    *,
    study_id: str,
    holdout_suite_digest: str,
    selected_task_ids: Sequence[str],
    searched_project_refs: Sequence[str],
    project_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    prior_evidence_digest: str,
    historical_exposure_receipt_digest: str,
    pool_fingerprint_digest: str,
    project_coverage_digest: str,
    audited_at: str,
    expires_at: str,
    queried_fields: Sequence[str],
    matched_task_ids: Sequence[str] = (),
    replacements: Sequence[Mapping[str, str]] = (),
) -> HoldoutExposureAuditV1:
    projects = tuple(sorted(searched_project_refs))
    if set(project_rows) != set(projects):
        raise ValueError("holdout audit rows do not cover the searched projects")
    safe_rows = {
        project: [dict(row) for row in project_rows[project]] for project in projects
    }
    return HoldoutExposureAuditV1(
        schema_version=1,
        kind="holdout_exposure_audit",
        study_id=study_id,
        holdout_suite_digest=holdout_suite_digest,
        selected_task_ids=tuple(sorted(selected_task_ids)),
        searched_project_refs=projects,
        searched_call_count=sum(len(rows) for rows in safe_rows.values()),
        queried_fields=tuple(sorted(queried_fields)),
        projection_digest=stable_digest(safe_rows),
        prior_evidence_digest=prior_evidence_digest,
        historical_exposure_receipt_digest=historical_exposure_receipt_digest,
        pool_fingerprint_digest=pool_fingerprint_digest,
        project_coverage_digest=project_coverage_digest,
        matched_task_ids=tuple(sorted(matched_task_ids)),
        replacements=tuple(
            dict(item)
            for item in sorted(replacements, key=lambda row: row["exposed_task_id"])
        ),
        outcome_data_consulted=False,
        status="replaced_exposed" if replacements else "clear",
        audited_at=audited_at,
        expires_at=expires_at,
    )


def verify_fresh_holdout_exposure_audit(
    audit: HoldoutExposureAuditV1,
    *,
    study_id: str,
    holdout_suite_digest: str,
    now: datetime | None = None,
) -> None:
    if audit.study_id != study_id:
        raise ValueError("holdout exposure audit belongs to another Study")
    if audit.holdout_suite_digest != holdout_suite_digest:
        raise ValueError("holdout exposure audit belongs to another suite")
    instant = (now or datetime.now(UTC)).astimezone(UTC)
    if instant >= _instant(audit.expires_at, "holdout audit expires_at"):
        raise ValueError("holdout exposure audit expired before preview")
    if audit.matched_task_ids and not audit.replacements:
        raise ValueError("exposed holdouts have no reviewed replacements")


def build_study_advancement_decision(
    result: ComparisonResultV3,
    *,
    holdout_audit: HoldoutExposureAuditV1 | None = None,
) -> StudyAdvancementDecisionV1:
    """Apply the preregistered repeatability/safety/mechanism gates."""

    if not isinstance(result, ComparisonResultV3):
        raise ValueError("Study advancement requires ComparisonResultV3")
    grouped: dict[tuple[str, str, str], dict[int, str]] = defaultdict(dict)
    shared_failures_by_attempt: dict[tuple[str, str], set[int]] = defaultdict(set)
    critical_candidate_regressions: set[str] = set()
    safety_failures: set[str] = set()
    for pair in result.paired_cases:
        for change in pair.dimension_changes:
            coordinate = f"{pair.task_id}:{change.id}"
            grouped[(pair.task_id, change.id, change.role)][pair.attempt] = (
                change.status
            )
            if change.role in {"outcome", "safety_gate"} and change.critical:
                if change.baseline is True and change.candidate is False:
                    critical_candidate_regressions.add(coordinate)
                if change.baseline is False and change.candidate is False:
                    shared_failures_by_attempt[(pair.task_id, change.id)].add(
                        pair.attempt
                    )
            if change.role == "safety_gate" and change.candidate is not True:
                safety_failures.add(coordinate)

    repeated_improvements = _repeated(grouped, role="outcome", status="improved")
    repeated_regressions = tuple(
        sorted(
            set(_repeated(grouped, role="outcome", status="regressed"))
            | set(_repeated(grouped, role="safety_gate", status="regressed"))
        )
    )
    critical_shared_failures = {
        f"{task_id}:{dimension}"
        for (task_id, dimension), attempts in shared_failures_by_attempt.items()
        if {0, 1} <= attempts
    }
    mechanism_gate = _mechanism_gate(result.mechanism_summary)
    invalid = (
        str(result.integrity.get("status") or "") != "reconciled"
        or result.decision.evidence_grade == "invalid"
        or result.behavioral_summary.status in {"invalid", "incomplete"}
        or any(
            item.status in {"drifted", "invalid", "inconclusive"}
            for item in result.task_validity
        )
    )
    blockers = tuple(
        sorted(
            set(result.behavioral_summary.critical_blockers)
            | critical_candidate_regressions
            | critical_shared_failures
            | safety_failures
        )
    )
    if invalid:
        status: AdvancementStatus = "stop_invalid"
        next_action = (
            "Repair evidence or task validity and create a new Study identity."
        )
    elif critical_candidate_regressions or repeated_regressions:
        status = "stop_critical_regression"
        next_action = "Stop the lane and inspect the named candidate regression."
    elif critical_shared_failures:
        status = "stop_task_scorer_repair"
        next_action = (
            "Repair the task or scorer before evaluating another Skill revision."
        )
    elif safety_failures or mechanism_gate != "passed":
        status = "inconclusive"
        next_action = (
            "Resolve the candidate safety or observed-Skill-use gate before holdout."
        )
    elif not repeated_improvements and all(
        item.status in {"valid", "non_discriminating"} for item in result.task_validity
    ):
        status = "run_no_skill_diagnostic"
        next_action = "Run the preregistered candidate-versus-no-Skill diagnostic; do not open holdout."
    elif holdout_audit is None:
        status = "inconclusive"
        next_action = "Freeze and exposure-audit the sealed holdout before advancement."
    else:
        if holdout_audit.study_id != result.comparison_id:
            raise ValueError("holdout exposure audit belongs to another Study")
        verify_fresh_holdout_exposure_audit(
            holdout_audit,
            study_id=result.comparison_id,
            holdout_suite_digest=holdout_audit.holdout_suite_digest,
        )
        status = "advance_holdout"
        next_action = (
            "Request a new immutable preview and approval for the sealed holdout."
        )

    return StudyAdvancementDecisionV1(
        schema_version=1,
        kind="study_advancement_decision",
        study_id=result.comparison_id,
        development_result_digest=result.result_digest,
        development_qualification_digest=result.qualification_digest,
        preview_digest=result.preview_digest,
        status=status,
        repeated_improvements=repeated_improvements,
        repeated_regressions=repeated_regressions,
        critical_blockers=blockers,
        mechanism_gate=mechanism_gate,
        holdout_suite_digest=(
            holdout_audit.holdout_suite_digest if holdout_audit else None
        ),
        holdout_exposure_audit_digest=(
            holdout_audit.audit_digest if holdout_audit else None
        ),
        next_action=next_action,
    )


def write_study_advancement_decision(
    path: Path, decision: StudyAdvancementDecisionV1
) -> None:
    atomic_write_json(path, decision.to_dict())


def read_study_advancement_decision(path: Path) -> StudyAdvancementDecisionV1:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("Study advancement decision must be an object")
    expected = set(StudyAdvancementDecisionV1.__dataclass_fields__)
    if set(value) != expected:
        raise ValueError("Study advancement decision fields do not match")
    normalized = dict(value)
    for name in ("repeated_improvements", "repeated_regressions", "critical_blockers"):
        raw = normalized[name]
        if not isinstance(raw, list):
            raise ValueError(f"{name} must be an array")
        normalized[name] = tuple(str(item) for item in raw)
    return StudyAdvancementDecisionV1(**normalized)  # type: ignore[arg-type]


def write_holdout_exposure_audit(path: Path, audit: HoldoutExposureAuditV1) -> None:
    atomic_write_json(path, audit.to_dict())


def read_holdout_exposure_audit(path: Path) -> HoldoutExposureAuditV1:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("holdout exposure audit must be an object")
    expected = set(HoldoutExposureAuditV1.__dataclass_fields__)
    if set(value) != expected:
        raise ValueError("holdout exposure audit fields do not match")
    normalized = dict(value)
    for name in (
        "selected_task_ids",
        "searched_project_refs",
        "queried_fields",
        "matched_task_ids",
    ):
        raw = normalized[name]
        if not isinstance(raw, list):
            raise ValueError(f"{name} must be an array")
        normalized[name] = tuple(str(item) for item in raw)
    replacements = normalized["replacements"]
    if not isinstance(replacements, list) or not all(
        isinstance(item, Mapping) for item in replacements
    ):
        raise ValueError("holdout replacements must be an array of objects")
    normalized["replacements"] = tuple(dict(item) for item in replacements)
    return HoldoutExposureAuditV1(**normalized)  # type: ignore[arg-type]


def _repeated(
    grouped: Mapping[tuple[str, str, str], Mapping[int, str]],
    *,
    role: str,
    status: str,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            f"{task_id}:{dimension}"
            for (task_id, dimension, dimension_role), attempts in grouped.items()
            if dimension_role == role
            and attempts.get(0) == status
            and attempts.get(1) == status
        )
    )


def _mechanism_gate(
    summary: Mapping[str, Any],
) -> Literal["passed", "failed", "unavailable"]:
    required = ["skill_assigned", "skill_registered", "skill_opened"]
    states: list[str] = []
    for stage in required:
        raw = summary.get(stage)
        if not isinstance(raw, Mapping):
            return "unavailable"
        for arm in ("baseline", "candidate"):
            values = raw.get(arm)
            if not isinstance(values, Mapping):
                return "unavailable"
            applicable = values.get("applicable")
            observed = values.get("observed")
            unavailable = values.get("unavailable")
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (applicable, observed, unavailable)
            ):
                return "unavailable"
            states.append(
                "passed"
                if applicable > 0 and observed == applicable and unavailable == 0
                else "failed"
            )
    relevant = summary.get("relevant_skill_file_opened")
    if relevant is not None:
        if not isinstance(relevant, Mapping):
            return "unavailable"
        for arm in ("baseline", "candidate"):
            values = relevant.get(arm)
            if not isinstance(values, Mapping):
                return "unavailable"
            applicable = values.get("applicable")
            observed = values.get("observed")
            unavailable = values.get("unavailable")
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (applicable, observed, unavailable)
            ):
                return "unavailable"
            if applicable > 0:
                states.append(
                    "passed"
                    if observed == applicable and unavailable == 0
                    else "failed"
                )
    return "passed" if set(states) == {"passed"} else "failed"


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 2_000:
        raise ValueError(f"{label} must be nonempty bounded text")
    return value


def _unique_text(values: Sequence[str], label: str, *, required: bool = False) -> None:
    if required and not values:
        raise ValueError(f"{label} cannot be empty")
    if tuple(values) != tuple(sorted(values)) or len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique and sorted")
    for value in values:
        _text(value, label)


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value


def _instant(value: Any, label: str) -> datetime:
    text = _text(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 instant") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed.astimezone(UTC)
