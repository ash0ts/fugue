from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any, Literal

from fugue.bench.context import RetrievalHit, RetrievalQuery
from fugue.bench.files import atomic_write_json


@dataclass(frozen=True)
class SelectionPolicy:
    selection_unit: Literal["candidate", "variant"] = "candidate"
    baseline_variant_id: str | None = None
    required_examples: int | None = None
    required_harnesses: tuple[str, ...] = ()
    require_agent_links: bool = False
    require_registration: bool = False
    require_skill_invocation: bool = False
    required_any_tool_names: tuple[str, ...] = ()
    metric: str = "pass_rate"
    confidence: float = 0.95
    noninferiority_margin: float = 0.05
    require_complete_grid: bool = True
    bootstrap_samples: int = 2_000
    tie_breakers: tuple[str, ...] = (
        "cost_per_success",
        "median_wall_time_sec",
        "recoverable_error_rate",
    )
    incumbent_candidate_id: str | None = None
    minimum_paired_pass_rate_delta: float | None = None
    minimum_pass_rate_improvement: float = 0.05
    minimum_cost_improvement: float = 0.15
    minimum_latency_improvement: float = 0.15


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    eligible: bool
    reasons: tuple[str, ...]
    trials: int
    examples: int
    pass_rate: float | None
    cost_per_success: float | None
    median_wall_time_sec: float | None
    recoverable_error_rate: float | None
    paired_pass_rate_delta: float | None = None
    localization_recall_at_10: float | None = None
    localization_mrr: float | None = None
    delta_to_best: float | None = None
    confidence_low: float | None = None
    confidence_high: float | None = None
    competitive: bool = False


@dataclass(frozen=True)
class CandidateSelection:
    policy: SelectionPolicy
    best_candidate_id: str | None
    selected_candidate_id: str | None
    incumbent_candidate_id: str | None
    decision: str
    reason: str
    candidates: tuple[CandidateScore, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "selection_unit": self.policy.selection_unit,
            "best_selection_id": self.best_candidate_id,
            "selected_selection_id": self.selected_candidate_id,
        }


def factorial_difference_in_differences(
    selection: CandidateSelection,
    *,
    factor_a_id: str,
    factor_b_id: str,
    combined_id: str,
) -> dict[str, float | str]:
    """Derive a 2x2 interaction from already paired baseline deltas."""
    scores = {score.candidate_id: score for score in selection.candidates}
    requested = (factor_a_id, factor_b_id, combined_id)
    missing = [item for item in requested if item not in scores]
    if missing:
        raise ValueError("factorial contrast is missing arm(s): " + ", ".join(missing))
    invalid = [
        item
        for item in requested
        if not scores[item].eligible or scores[item].paired_pass_rate_delta is None
    ]
    if invalid:
        raise ValueError(
            "factorial contrast requires eligible paired arms: " + ", ".join(invalid)
        )
    factor_a = float(scores[factor_a_id].paired_pass_rate_delta)
    factor_b = float(scores[factor_b_id].paired_pass_rate_delta)
    combined = float(scores[combined_id].paired_pass_rate_delta)
    return {
        "factor_a_id": factor_a_id,
        "factor_b_id": factor_b_id,
        "combined_id": combined_id,
        "factor_a_delta": factor_a,
        "factor_b_delta": factor_b,
        "combined_delta": combined,
        "interaction": combined - factor_a - factor_b,
    }


@dataclass(frozen=True)
class TreatmentSelectionLockV1:
    schema_version: int
    source_commit: str
    calibration_snapshot_sha256: str
    discovery_snapshot_sha256: str
    rankings: tuple[dict[str, Any], ...]
    selected_variants: tuple[str, ...]
    lock_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["rankings"] = list(self.rankings)
        value["selected_variants"] = list(self.selected_variants)
        return value


@dataclass(frozen=True)
class InterventionSelectionLockV1:
    schema_version: int
    kind: Literal["intervention_selection"]
    experiment_id: str
    source_commit: str
    source_tree: str
    source_dirty_digest: str
    failure_lock_sha256: str
    discovery_suite_sha256: str
    holdout_suite_sha256: str
    analysis_snapshot_sha256: str
    discovery_run_snapshot_sha256s: tuple[str, ...]
    comparison_example_ids: tuple[str, ...]
    discovery_variant_ids: tuple[str, ...]
    baseline_variant_id: str
    selected_variant_id: str
    rankings: tuple[dict[str, Any], ...]
    decision: Literal["recommend", "promote"]
    rationale: str
    failure_locked_at: str
    suites_frozen_at: str
    discovery_completed_at: str
    selection_locked_at: str
    lock_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["discovery_run_snapshot_sha256s"] = list(
            self.discovery_run_snapshot_sha256s
        )
        value["comparison_example_ids"] = list(self.comparison_example_ids)
        value["discovery_variant_ids"] = list(self.discovery_variant_ids)
        value["rankings"] = list(self.rankings)
        return value


def select_candidate_configuration(
    rows: Iterable[dict[str, Any]],
    policy: SelectionPolicy,
    *,
    seed: str,
) -> CandidateSelection:
    """Select a candidate from normalized trials without model-authored arithmetic."""
    values = [dict(row) for row in rows]
    include_harness = policy.selection_unit == "variant"
    baseline_rows = [
        row
        for row in values
        if policy.baseline_variant_id
        and str(row.get("variant_id") or "") == policy.baseline_variant_id
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in values:
        selection_id = str(row.get(f"{policy.selection_unit}_id") or "").strip()
        if selection_id and selection_id != policy.baseline_variant_id:
            grouped.setdefault(selection_id, []).append(row)
    expected = {
        _trial_coordinate(row, include_harness=include_harness)
        for rows_for_candidate in grouped.values()
        for row in rows_for_candidate
        if _trial_coordinate(row, include_harness=include_harness) is not None
    }
    candidate_grids = {
        candidate_id: {
            coordinate
            for row in candidate_rows
            if (coordinate := _trial_coordinate(row, include_harness=include_harness))
            is not None
        }
        for candidate_id, candidate_rows in grouped.items()
    }
    grids_match = len({frozenset(value) for value in candidate_grids.values()}) <= 1
    scores: list[CandidateScore] = []
    eligible_rows: dict[str, list[dict[str, Any]]] = {}
    for candidate_id, candidate_rows in sorted(grouped.items()):
        reasons: list[str] = []
        coordinates = [
            _trial_coordinate(row, include_harness=include_harness)
            for row in candidate_rows
        ]
        present = {item for item in coordinates if item is not None}
        if any(item is None for item in coordinates):
            reasons.append("missing comparison_example_id, harness, or trial_index")
        if len(present) != len(coordinates):
            reasons.append("duplicate candidate/example/trial row")
        if policy.require_complete_grid and present != expected:
            reasons.append("incomplete comparison grid")
        if policy.require_complete_grid and not grids_match:
            reasons.append("candidates do not share one comparison grid")
        if any(row.get("pass") is None for row in candidate_rows):
            reasons.append("missing deterministic outcome")
        examples = {item[0] for item in present}
        if policy.required_examples and len(examples) != policy.required_examples:
            reasons.append(f"expected {policy.required_examples} comparison examples")
        if policy.required_harnesses:
            harness_counts = {
                harness: sum(row.get("harness") == harness for row in candidate_rows)
                for harness in policy.required_harnesses
            }
            if len(set(harness_counts.values())) != 1 or not all(
                harness_counts.values()
            ):
                reasons.append("harness assignment is not balanced")
        if policy.require_agent_links and any(
            row.get("trace_link_status") != "linked" for row in candidate_rows
        ):
            reasons.append("missing or ambiguous Agent link")
        if policy.require_registration and any(
            row.get("context_registration_status") not in {"registered", "static"}
            for row in candidate_rows
        ):
            reasons.append("context registration is incomplete")
        reasons.extend(_mechanism_reasons(candidate_rows, policy))
        paired_delta = _paired_baseline_delta(candidate_rows, baseline_rows)
        if policy.baseline_variant_id and paired_delta is None:
            reasons.append("missing same-task/harness baseline")
        if (
            policy.minimum_paired_pass_rate_delta is not None
            and (
                paired_delta is None
                or paired_delta < policy.minimum_paired_pass_rate_delta
            )
        ):
            reasons.append(
                "paired pass-rate delta is below the preregistered minimum"
            )
        eligible = not reasons
        if eligible:
            eligible_rows[candidate_id] = candidate_rows
        successes = sum(row.get("pass") is True for row in candidate_rows)
        costs = [row.get("cost_usd") for row in candidate_rows]
        latencies = [row.get("wall_time_sec") for row in candidate_rows]
        error_rows = [
            row
            for row in candidate_rows
            if float(
                row.get("weave_tool_error_count")
                or row.get("recoverable_tool_errors")
                or 0
            )
            > 0
        ]
        scores.append(
            CandidateScore(
                candidate_id=candidate_id,
                eligible=eligible,
                reasons=tuple(reasons),
                trials=len(candidate_rows),
                examples=len(examples),
                pass_rate=(successes / len(candidate_rows) if candidate_rows else None),
                cost_per_success=(
                    sum(float(value) for value in costs) / successes
                    if successes and costs and all(value is not None for value in costs)
                    else None
                ),
                median_wall_time_sec=(
                    median(float(value) for value in latencies)
                    if latencies and all(value is not None for value in latencies)
                    else None
                ),
                recoverable_error_rate=(
                    len(error_rows) / len(candidate_rows) if candidate_rows else None
                ),
                paired_pass_rate_delta=paired_delta,
                localization_recall_at_10=_mean_metric(
                    candidate_rows,
                    "localization_recall_at_10",
                    "retrieval_recall_at_10",
                    "recall_at_10",
                ),
                localization_mrr=_mean_metric(
                    candidate_rows, "localization_mrr", "retrieval_mrr", "mrr"
                ),
            )
        )
    eligible_scores = [score for score in scores if score.eligible]
    if not eligible_scores:
        return CandidateSelection(
            policy=policy,
            best_candidate_id=None,
            selected_candidate_id=None,
            incumbent_candidate_id=policy.incumbent_candidate_id,
            decision="blocked",
            reason="no candidate has a complete, uniquely scored comparison grid",
            candidates=tuple(scores),
        )
    best = sorted(
        eligible_scores,
        key=lambda item: (
            -(
                item.paired_pass_rate_delta
                if item.paired_pass_rate_delta is not None
                else item.pass_rate or 0.0
            ),
            item.candidate_id,
        ),
    )[0]
    enriched: list[CandidateScore] = []
    for score in scores:
        if not score.eligible:
            enriched.append(score)
            continue
        low, high = _paired_delta_interval(
            eligible_rows[score.candidate_id],
            eligible_rows[best.candidate_id],
            confidence=policy.confidence,
            samples=policy.bootstrap_samples,
            seed=f"{seed}:{score.candidate_id}:{best.candidate_id}",
        )
        enriched.append(
            CandidateScore(
                **{
                    **asdict(score),
                    "delta_to_best": (score.pass_rate or 0.0) - (best.pass_rate or 0.0),
                    "confidence_low": low,
                    "confidence_high": high,
                    "competitive": low >= -policy.noninferiority_margin,
                }
            )
        )
    competitive = [score for score in enriched if score.competitive]
    selected = sorted(
        competitive,
        key=lambda item: _selection_tie_key(item, policy.tie_breakers),
    )[0]
    decision, reason = _promotion_decision(
        selected,
        enriched,
        eligible_rows,
        policy,
        seed=seed,
    )
    return CandidateSelection(
        policy=policy,
        best_candidate_id=best.candidate_id,
        selected_candidate_id=selected.candidate_id,
        incumbent_candidate_id=policy.incumbent_candidate_id,
        decision=decision,
        reason=reason,
        candidates=tuple(enriched),
    )


def build_treatment_selection_lock(
    *,
    source_commit: str,
    calibration_snapshot_sha256: str,
    discovery_snapshot_sha256: str,
    rankings: Iterable[Mapping[str, Any]],
    selected_variants: Iterable[str],
) -> TreatmentSelectionLockV1:
    selected = tuple(str(value) for value in selected_variants)
    if len(selected) != 3 or len(set(selected)) != 3 or "none" in selected:
        raise ValueError("treatment selection lock requires three unique treatments")
    for label, digest in (
        ("calibration snapshot", calibration_snapshot_sha256),
        ("discovery snapshot", discovery_snapshot_sha256),
    ):
        if not _sha256(digest):
            raise ValueError(f"{label} must be a SHA-256 digest")
    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise ValueError("treatment selection source commit must be a full Git commit")
    normalized = tuple(dict(value) for value in rankings)
    ranked_ids = [str(value.get("variant_id") or "") for value in normalized]
    if not normalized or len(ranked_ids) != len(set(ranked_ids)) or "" in ranked_ids:
        raise ValueError("treatment rankings require unique variant_id values")
    if not set(selected) <= set(ranked_ids):
        raise ValueError("selected treatments must appear in the complete rankings")
    base = TreatmentSelectionLockV1(
        schema_version=1,
        source_commit=source_commit,
        calibration_snapshot_sha256=calibration_snapshot_sha256,
        discovery_snapshot_sha256=discovery_snapshot_sha256,
        rankings=normalized,
        selected_variants=selected,
    )
    digest = _digest(base.to_dict())
    return TreatmentSelectionLockV1(**{**asdict(base), "lock_sha256": digest})


def write_treatment_selection_lock(path: Path, lock: TreatmentSelectionLockV1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = lock.to_dict()
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError(f"treatment selection lock already differs: {path}")
        return path
    return atomic_write_json(path, payload)


def read_treatment_selection_lock(path: Path) -> TreatmentSelectionLockV1:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version") or 0) != 1:
        raise ValueError("unsupported treatment selection lock schema")
    expected = str(payload.get("lock_sha256") or "")
    if not expected or _digest({**payload, "lock_sha256": ""}) != expected:
        raise ValueError("treatment selection lock digest does not match its content")
    return TreatmentSelectionLockV1(
        schema_version=1,
        source_commit=str(payload.get("source_commit") or ""),
        calibration_snapshot_sha256=str(
            payload.get("calibration_snapshot_sha256") or ""
        ),
        discovery_snapshot_sha256=str(payload.get("discovery_snapshot_sha256") or ""),
        rankings=tuple(dict(value) for value in payload.get("rankings") or ()),
        selected_variants=tuple(
            str(value) for value in payload.get("selected_variants") or ()
        ),
        lock_sha256=expected,
    )


def build_intervention_selection_lock(  # noqa: C901 - one strict lock contract
    *,
    experiment_id: str,
    source_commit: str,
    source_tree: str,
    source_dirty_digest: str,
    failure_lock_sha256: str,
    discovery_suite_sha256: str,
    holdout_suite_sha256: str,
    analysis_snapshot_sha256: str,
    discovery_run_snapshot_sha256s: Iterable[str],
    comparison_example_ids: Iterable[str],
    discovery_variant_ids: Iterable[str],
    baseline_variant_id: str,
    selected_variant_id: str,
    rankings: Iterable[Mapping[str, Any]],
    decision: str,
    rationale: str,
    failure_locked_at: str,
    suites_frozen_at: str,
    discovery_completed_at: str,
    selection_locked_at: str,
) -> InterventionSelectionLockV1:
    experiment = str(experiment_id).strip()
    baseline = str(baseline_variant_id).strip()
    selected = str(selected_variant_id).strip()
    if not experiment:
        raise ValueError("intervention selection lock requires an experiment id")
    if not baseline or not selected or baseline == selected:
        raise ValueError(
            "intervention selection lock requires distinct baseline and selected variants"
        )
    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise ValueError("intervention selection source commit must be a full Git commit")
    if len(source_tree) != 40 or any(
        character not in "0123456789abcdef" for character in source_tree
    ):
        raise ValueError("intervention selection source tree must be a full Git tree")
    if source_dirty_digest:
        raise ValueError(
            "intervention selection requires a clean source tree; "
            "dirty worktrees cannot be qualified"
        )
    for value, label in (
        (failure_lock_sha256, "failure lock"),
        (discovery_suite_sha256, "discovery suite"),
        (holdout_suite_sha256, "holdout suite"),
    ):
        if not _sha256(value):
            raise ValueError(f"{label} must be a SHA-256 digest")
    if discovery_suite_sha256 == holdout_suite_sha256:
        raise ValueError("discovery and holdout suites must be independently frozen")
    if not _sha256(analysis_snapshot_sha256):
        raise ValueError("analysis snapshot must be a SHA-256 digest")
    run_snapshots = tuple(
        dict.fromkeys(str(value) for value in discovery_run_snapshot_sha256s)
    )
    if not run_snapshots or any(not _sha256(value) for value in run_snapshots):
        raise ValueError(
            "intervention selection requires nonzero SHA-256 run snapshots"
        )
    examples = tuple(dict.fromkeys(str(value) for value in comparison_example_ids))
    if not examples or any(not _sha256(value) for value in examples):
        raise ValueError(
            "intervention selection requires nonzero comparison example digests"
        )
    variants = tuple(
        dict.fromkeys(str(value).strip() for value in discovery_variant_ids)
    )
    if len(variants) < 2 or any(not value for value in variants):
        raise ValueError(
            "intervention selection requires every discovery arm identity"
        )
    normalized = tuple(dict(value) for value in rankings)
    ranked_ids = [str(value.get("variant_id") or "") for value in normalized]
    if not normalized or len(ranked_ids) != len(set(ranked_ids)) or "" in ranked_ids:
        raise ValueError("intervention rankings require unique variant_id values")
    if not {baseline, selected} <= set(ranked_ids):
        raise ValueError(
            "baseline and selected variants must appear in the complete rankings"
        )
    if set(variants) != set(ranked_ids):
        raise ValueError(
            "intervention rankings must bind every predeclared discovery arm"
        )
    for ranking in normalized:
        if not _sha256(str(ranking.get("candidate_digest") or "")):
            raise ValueError(
                "every intervention ranking must bind an exact candidate digest"
            )
    if decision not in {"recommend", "promote"}:
        raise ValueError("only a qualified intervention may be locked for holdout")
    reason = str(rationale).strip()
    if not reason:
        raise ValueError("intervention selection lock requires a rationale")
    failure_time = _utc_timestamp(failure_locked_at, "failure lock time")
    suites_time = _utc_timestamp(suites_frozen_at, "suite freeze time")
    discovery_time = _utc_timestamp(
        discovery_completed_at,
        "discovery completion time",
    )
    selection_time = _utc_timestamp(selection_locked_at, "selection lock time")
    parsed_times = tuple(datetime.fromisoformat(value) for value in (
        failure_time,
        suites_time,
        discovery_time,
        selection_time,
    ))
    if not (
        parsed_times[0] <= parsed_times[1]
        < parsed_times[2]
        <= parsed_times[3]
    ):
        raise ValueError(
            "intervention chronology must lock the failure, freeze both suites, "
            "complete discovery, then lock selection"
        )
    base = InterventionSelectionLockV1(
        schema_version=1,
        kind="intervention_selection",
        experiment_id=experiment,
        source_commit=source_commit,
        source_tree=source_tree,
        source_dirty_digest="",
        failure_lock_sha256=failure_lock_sha256,
        discovery_suite_sha256=discovery_suite_sha256,
        holdout_suite_sha256=holdout_suite_sha256,
        analysis_snapshot_sha256=analysis_snapshot_sha256,
        discovery_run_snapshot_sha256s=run_snapshots,
        comparison_example_ids=examples,
        discovery_variant_ids=variants,
        baseline_variant_id=baseline,
        selected_variant_id=selected,
        rankings=normalized,
        decision=decision,  # type: ignore[arg-type]
        rationale=reason,
        failure_locked_at=failure_time,
        suites_frozen_at=suites_time,
        discovery_completed_at=discovery_time,
        selection_locked_at=selection_time,
    )
    digest = _digest(base.to_dict())
    return InterventionSelectionLockV1(
        **{**asdict(base), "lock_sha256": digest}
    )


def write_intervention_selection_lock(
    path: Path, lock: InterventionSelectionLockV1
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = lock.to_dict()
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError(f"intervention selection lock already differs: {path}")
        return path
    return atomic_write_json(path, payload)


def read_intervention_selection_lock(path: Path) -> InterventionSelectionLockV1:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        int(payload.get("schema_version") or 0) != 1
        or payload.get("kind") != "intervention_selection"
    ):
        raise ValueError("unsupported intervention selection lock schema")
    expected = str(payload.get("lock_sha256") or "")
    if not expected or _digest({**payload, "lock_sha256": ""}) != expected:
        raise ValueError("intervention selection lock digest does not match its content")
    rebuilt = build_intervention_selection_lock(
        experiment_id=str(payload.get("experiment_id") or ""),
        source_commit=str(payload.get("source_commit") or ""),
        source_tree=str(payload.get("source_tree") or ""),
        source_dirty_digest=str(payload.get("source_dirty_digest") or ""),
        failure_lock_sha256=str(payload.get("failure_lock_sha256") or ""),
        discovery_suite_sha256=str(
            payload.get("discovery_suite_sha256") or ""
        ),
        holdout_suite_sha256=str(payload.get("holdout_suite_sha256") or ""),
        analysis_snapshot_sha256=str(
            payload.get("analysis_snapshot_sha256") or ""
        ),
        discovery_run_snapshot_sha256s=tuple(
            str(value)
            for value in payload.get("discovery_run_snapshot_sha256s") or ()
        ),
        comparison_example_ids=tuple(
            str(value) for value in payload.get("comparison_example_ids") or ()
        ),
        discovery_variant_ids=tuple(
            str(value) for value in payload.get("discovery_variant_ids") or ()
        ),
        baseline_variant_id=str(payload.get("baseline_variant_id") or ""),
        selected_variant_id=str(payload.get("selected_variant_id") or ""),
        rankings=tuple(dict(value) for value in payload.get("rankings") or ()),
        decision=str(payload.get("decision") or ""),
        rationale=str(payload.get("rationale") or ""),
        failure_locked_at=str(payload.get("failure_locked_at") or ""),
        suites_frozen_at=str(payload.get("suites_frozen_at") or ""),
        discovery_completed_at=str(
            payload.get("discovery_completed_at") or ""
        ),
        selection_locked_at=str(payload.get("selection_locked_at") or ""),
    )
    if rebuilt.lock_sha256 != expected:
        raise ValueError("intervention selection lock normalization changed its digest")
    return rebuilt


def _utc_timestamp(value: str, label: str) -> str:
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed.astimezone(UTC).isoformat()


def _sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _assigned_skills_invoked(row: Mapping[str, Any]) -> bool:
    assigned = {
        str(value)
        for value in row.get("skills_assigned") or row.get("skill_ids") or ()
    }
    if not assigned:
        return True
    evidence = row.get("skill_invocation_evidence")
    if not isinstance(evidence, Mapping) or evidence.get("status") != "observed":
        return False
    invoked = {str(value) for value in evidence.get("skills_invoked") or ()}
    return assigned <= invoked


def _mechanism_reasons(
    rows: list[dict[str, Any]], policy: SelectionPolicy
) -> list[str]:
    reasons: list[str] = []
    if policy.require_skill_invocation and any(
        not _assigned_skills_invoked(row)
        for row in rows
        if row.get("skill_ids") or row.get("skills_assigned")
    ):
        reasons.append("assigned Skill invocation is missing or unproven")
    if policy.required_any_tool_names and any(
        not (set(_tool_names(row)) & set(policy.required_any_tool_names))
        for row in rows
    ):
        reasons.append("required MCP tool invocation is missing or unproven")
    return reasons


def _tool_names(row: Mapping[str, Any]) -> tuple[str, ...]:
    value = row.get("weave_tool_names")
    if isinstance(value, Mapping):
        return tuple(str(name) for name, count in value.items() if float(count or 0) > 0)
    if isinstance(value, (list, tuple, set)):
        return tuple(str(name) for name in value)
    return ()


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _promotion_decision(
    selected: CandidateScore,
    scores: list[CandidateScore],
    rows: dict[str, list[dict[str, Any]]],
    policy: SelectionPolicy,
    *,
    seed: str,
) -> tuple[str, str]:
    incumbent_id = policy.incumbent_candidate_id
    if not incumbent_id:
        return (
            "recommend",
            "no incumbent was supplied; propose the selected candidate for review",
        )
    incumbent = next(
        (
            item
            for item in scores
            if item.candidate_id == incumbent_id and item.eligible
        ),
        None,
    )
    if incumbent is None:
        return (
            "blocked",
            "the incumbent is absent or ineligible in this comparison scope",
        )
    if incumbent.candidate_id == selected.candidate_id:
        return "no_promotion", "the incumbent remains the selected candidate"
    low, _ = _paired_delta_interval(
        rows[selected.candidate_id],
        rows[incumbent.candidate_id],
        confidence=policy.confidence,
        samples=policy.bootstrap_samples,
        seed=f"{seed}:{selected.candidate_id}:{incumbent.candidate_id}:incumbent",
    )
    if low < -policy.noninferiority_margin:
        return (
            "no_promotion",
            "the selected candidate is not quality non-inferior to the incumbent",
        )
    pass_improvement = (selected.pass_rate or 0.0) - (incumbent.pass_rate or 0.0)
    cost_improvement = _relative_improvement(
        incumbent.cost_per_success, selected.cost_per_success
    )
    latency_improvement = _relative_improvement(
        incumbent.median_wall_time_sec, selected.median_wall_time_sec
    )
    if (
        pass_improvement >= policy.minimum_pass_rate_improvement
        or cost_improvement >= policy.minimum_cost_improvement
        or latency_improvement >= policy.minimum_latency_improvement
    ):
        return (
            "promote",
            "quality is non-inferior and an explicit improvement threshold was met",
        )
    return (
        "no_promotion",
        "quality is non-inferior, but no promotion improvement threshold was met",
    )


def _paired_delta_interval(
    candidate_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    *,
    confidence: float,
    samples: int,
    seed: str,
) -> tuple[float, float]:
    candidate = _example_outcomes(candidate_rows)
    baseline = _example_outcomes(baseline_rows)
    examples = sorted(set(candidate) & set(baseline))
    if not examples:
        return 0.0, 0.0
    if candidate == baseline:
        return 0.0, 0.0
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        selected = [rng.choice(examples) for _ in examples]
        deltas.append(
            sum(candidate[item] - baseline[item] for item in selected) / len(selected)
        )
    alpha = (1.0 - confidence) / 2.0
    return _percentile(sorted(deltas), alpha), _percentile(sorted(deltas), 1.0 - alpha)


def _example_outcomes(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        example_id = str(row["comparison_example_id"])
        grouped.setdefault(example_id, []).append(float(row.get("pass") is True))
    return {
        example_id: sum(values) / len(values) for example_id, values in grouped.items()
    }


def _trial_coordinate(
    row: dict[str, Any], *, include_harness: bool
) -> tuple[str, int] | tuple[str, str, int] | None:
    example_id = str(row.get("comparison_example_id") or "").strip()
    harness = str(row.get("harness") or "").strip()
    try:
        trial_index = int(row.get("trial_index"))
    except (TypeError, ValueError):
        return None
    if not example_id or trial_index < 1:
        return None
    if include_harness:
        return (example_id, harness, trial_index) if harness else None
    return example_id, trial_index


def _paired_baseline_delta(
    candidate_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]
) -> float | None:
    if not baseline_rows:
        return None
    baseline: dict[tuple[str, str, int], bool] = {}
    for row in baseline_rows:
        key = _task_harness_trial(row)
        if key is None or key in baseline:
            return None
        baseline[key] = row.get("pass") is True
    deltas: list[float] = []
    for row in candidate_rows:
        key = _task_harness_trial(row)
        if key is None or key not in baseline:
            return None
        deltas.append(float(row.get("pass") is True) - float(baseline[key]))
    return sum(deltas) / len(deltas) if deltas else None


def _task_harness_trial(row: Mapping[str, Any]) -> tuple[str, str, int] | None:
    task_id = str(row.get("task_name") or "")
    harness = str(row.get("harness") or "")
    try:
        trial_index = int(row.get("trial_index"))
    except (TypeError, ValueError):
        return None
    if not task_id or not harness or trial_index < 1:
        return None
    return task_id, harness, trial_index


def _mean_metric(rows: list[dict[str, Any]], *names: str) -> float | None:
    values: list[float] = []
    for row in rows:
        value = next(
            (row.get(name) for name in names if row.get(name) is not None), None
        )
        if value is not None:
            values.append(float(value))
    return sum(values) / len(values) if values else None


def _selection_tie_key(
    score: CandidateScore, tie_breakers: tuple[str, ...]
) -> tuple[Any, ...]:
    values: list[Any] = []
    for name in tie_breakers:
        value = getattr(score, name)
        values.extend(
            (
                value is None,
                -value
                if name
                in {
                    "localization_recall_at_10",
                    "localization_mrr",
                }
                and value is not None
                else value
                if value is not None
                else math.inf,
            )
        )
    return (*values, score.candidate_id)


def _relative_improvement(before: float | None, after: float | None) -> float:
    if before is None or after is None or before <= 0:
        return -math.inf
    return (before - after) / before


def score_retrieval(
    query: RetrievalQuery, hits: list[RetrievalHit]
) -> dict[str, float | int | None]:
    expected = {_normalize_path(path) for path in query.expected_paths}
    raw_ranked = [_normalize_path(hit.path) for hit in hits]
    ranked = list(dict.fromkeys(path for path in raw_ranked if path))
    if not expected:
        return {
            "mrr": None,
            "ndcg_at_10": None,
            "first_relevant_rank": None,
            "empty": int(not hits),
            **{
                f"{metric}_at_{k}": None
                for k in (1, 5, 10, 20)
                for metric in ("recall", "precision")
            },
        }
    relevant_ranks = [
        rank for rank, path in enumerate(ranked, start=1) if path in expected
    ]
    first_rank = min(relevant_ranks) if relevant_ranks else None
    metrics: dict[str, float | int | None] = {
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "ndcg_at_10": _ndcg(ranked[:10], expected),
        "first_relevant_rank": first_rank,
        "empty": int(not hits),
        "raw_result_count": len(hits),
        "unique_result_count": len(ranked),
    }
    for k in (1, 5, 10, 20):
        selected = ranked[:k]
        found = sum(1 for path in selected if path in expected)
        metrics[f"recall_at_{k}"] = found / len(expected) if expected else None
        metrics[f"precision_at_{k}"] = found / len(selected) if selected else 0.0
    return metrics


def score_evidence_paths(
    expected_paths: Iterable[str], observed_paths: Iterable[str]
) -> dict[str, float | None]:
    expected = {_normalize_path(path) for path in expected_paths}
    observed = {_normalize_path(path) for path in observed_paths}
    if not expected:
        return {"evidence_recall": None, "evidence_precision": None}
    overlap = expected & observed
    return {
        "evidence_recall": len(overlap) / len(expected),
        "evidence_precision": len(overlap) / len(observed) if observed else 0.0,
    }


def score_fact_recall(
    expected_facts: Iterable[str], observed_text: Iterable[str]
) -> dict[str, float | None]:
    expected = [item.strip().lower() for item in expected_facts if item.strip()]
    corpus = "\n".join(observed_text).lower()
    if not expected:
        return {"fact_recall": None}
    recalled = sum(1 for fact in expected if fact in corpus)
    return {"fact_recall": recalled / len(expected)}


def latency_summary(values: Iterable[float]) -> dict[str, float | None]:
    selected = sorted(float(value) for value in values)
    if not selected:
        return {"p50_ms": None, "p95_ms": None}
    return {
        "p50_ms": median(selected),
        "p95_ms": _percentile(selected, 0.95),
    }


def _normalize_path(path: str) -> str:
    return str(path).strip().removeprefix("./").replace("\\", "/")


def _ndcg(ranked: list[str], expected: set[str]) -> float | None:
    if not expected:
        return None
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, path in enumerate(ranked, start=1)
        if path in expected
    )
    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(len(expected), len(ranked)) + 1)
    )
    return dcg / ideal if ideal else 0.0


def _percentile(values: list[float], fraction: float) -> float:
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[lower]
    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight
