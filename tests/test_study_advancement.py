from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from fugue.bench.comparison import ComparisonResultV3, DimensionChangeV2, PairedCaseV3
from fugue.bench.study_advancement import (
    build_holdout_exposure_audit,
    build_study_advancement_decision,
    read_study_advancement_decision,
    write_study_advancement_decision,
)


def _change(
    dimension: str,
    *,
    status: str,
    baseline: bool,
    candidate: bool,
    role: str = "outcome",
    critical: bool = True,
) -> DimensionChangeV2:
    return DimensionChangeV2(
        id=dimension,
        label=dimension,
        status=status,  # type: ignore[arg-type]
        baseline=baseline,
        candidate=candidate,
        critical=critical,
        role=role,  # type: ignore[arg-type]
    )


def _result(
    changes: tuple[DimensionChangeV2, ...], *, integrity: str = "reconciled"
) -> ComparisonResultV3:
    result = object.__new__(ComparisonResultV3)
    pairs = tuple(
        PairedCaseV3(
            pair_id=f"pair-{task}-{attempt}",
            task_id=task,
            harness="claude-code",
            attempt=attempt,
            status="improved"
            if any(item.status == "improved" for item in changes)
            else "unchanged",
            dimension_changes=changes,
            baseline=None,
            candidate=None,
        )
        for task in ("task-a", "task-b", "task-c", "task-d")
        for attempt in (0, 1)
    )
    mechanism = {
        name: {
            "candidate": {"observed": 8, "applicable": 8, "unavailable": 0},
            "baseline": {"observed": 8, "applicable": 8, "unavailable": 0},
        }
        for name in ("skill_assigned", "skill_registered", "skill_opened")
    }
    values = {
        "comparison_id": "skill-study",
        "result_digest": "a" * 64,
        "qualification_digest": "b" * 64,
        "preview_digest": "c" * 64,
        "paired_cases": pairs,
        "mechanism_summary": mechanism,
        "integrity": {"status": integrity},
        "decision": SimpleNamespace(evidence_grade="A"),
        "behavioral_summary": SimpleNamespace(
            status="improved",
            critical_blockers=(),
        ),
        "task_validity": tuple(SimpleNamespace(status="valid") for _task in range(4)),
    }
    for name, value in values.items():
        object.__setattr__(result, name, value)
    return result


def _audit():
    return build_holdout_exposure_audit(
        study_id="skill-study",
        holdout_suite_digest="d" * 64,
        selected_task_ids=("holdout-a", "holdout-b", "holdout-c", "holdout-d"),
        searched_project_refs=("wandb/project-a", "wandb/project-b"),
        project_rows={"wandb/project-a": (), "wandb/project-b": ()},
        prior_evidence_digest="e" * 64,
        historical_exposure_receipt_digest="e" * 64,
        pool_fingerprint_digest="f" * 64,
        project_coverage_digest="1" * 64,
        audited_at="2099-01-01T00:00:00+00:00",
        expires_at="2099-01-02T00:00:00+00:00",
        queried_fields=("attributes.fugue.task_id",),
    )


def test_repeated_improvement_advances_only_with_prefrozen_audit(tmp_path) -> None:
    result = _result(
        (
            _change("outcome", status="improved", baseline=False, candidate=True),
            _change(
                "safety",
                status="unchanged",
                baseline=True,
                candidate=True,
                role="safety_gate",
            ),
        )
    )
    blocked = build_study_advancement_decision(result)
    assert blocked.status == "inconclusive"

    decision = build_study_advancement_decision(result, holdout_audit=_audit())
    assert decision.status == "advance_holdout"
    assert decision.repeated_improvements == (
        "task-a:outcome",
        "task-b:outcome",
        "task-c:outcome",
        "task-d:outcome",
    )
    path = tmp_path / "decision.json"
    write_study_advancement_decision(path, decision)
    assert read_study_advancement_decision(path) == decision


def test_advancement_requires_opening_but_not_native_invocation() -> None:
    result = _result(
        (_change("outcome", status="improved", baseline=False, candidate=True),)
    )
    object.__setattr__(
        result,
        "mechanism_summary",
        {
            **result.mechanism_summary,
            "skill_native_invoked": {
                "candidate": {
                    "observed": 0,
                    "applicable": 8,
                    "unavailable": 8,
                }
            },
        },
    )

    decision = build_study_advancement_decision(result, holdout_audit=_audit())

    assert decision.mechanism_gate == "passed"
    assert decision.status == "advance_holdout"


def test_declared_relevant_skill_file_must_be_opened() -> None:
    result = _result(
        (_change("outcome", status="improved", baseline=False, candidate=True),)
    )
    object.__setattr__(
        result,
        "mechanism_summary",
        {
            **result.mechanism_summary,
            "relevant_skill_file_opened": {
                "baseline": {
                    "observed": 8,
                    "applicable": 8,
                    "unavailable": 0,
                },
                "candidate": {
                    "observed": 7,
                    "applicable": 8,
                    "unavailable": 0,
                }
            },
        },
    )

    decision = build_study_advancement_decision(result, holdout_audit=_audit())

    assert decision.mechanism_gate == "failed"
    assert decision.status == "inconclusive"


def test_baseline_skill_opening_is_required_for_a_fair_comparison() -> None:
    result = _result(
        (_change("outcome", status="improved", baseline=False, candidate=True),)
    )
    mechanism = dict(result.mechanism_summary)
    opened = dict(mechanism["skill_opened"])
    opened["baseline"] = {"observed": 7, "applicable": 8, "unavailable": 0}
    mechanism["skill_opened"] = opened
    object.__setattr__(result, "mechanism_summary", mechanism)

    decision = build_study_advancement_decision(result, holdout_audit=_audit())

    assert decision.mechanism_gate == "failed"
    assert decision.status == "inconclusive"


def test_critical_regression_stops_immediately() -> None:
    result = _result(
        (
            _change("outcome", status="regressed", baseline=True, candidate=False),
            _change(
                "safety",
                status="unchanged",
                baseline=True,
                candidate=True,
                role="safety_gate",
            ),
        )
    )
    decision = build_study_advancement_decision(result, holdout_audit=_audit())
    assert decision.status == "stop_critical_regression"
    assert decision.repeated_regressions


def test_shared_critical_failure_requires_task_or_scorer_repair() -> None:
    result = _result(
        (
            _change("contract", status="unchanged", baseline=False, candidate=False),
            _change(
                "safety",
                status="unchanged",
                baseline=True,
                candidate=True,
                role="safety_gate",
            ),
        )
    )
    decision = build_study_advancement_decision(result)
    assert decision.status == "stop_task_scorer_repair"
    assert any("contract" in blocker for blocker in decision.critical_blockers)


def test_non_discriminating_lane_routes_to_no_skill_diagnostic() -> None:
    result = _result(
        (
            _change("outcome", status="unchanged", baseline=True, candidate=True),
            _change(
                "safety",
                status="unchanged",
                baseline=True,
                candidate=True,
                role="safety_gate",
            ),
        )
    )
    object.__setattr__(
        result,
        "task_validity",
        tuple(SimpleNamespace(status="non_discriminating") for _task in range(4)),
    )
    decision = build_study_advancement_decision(result)
    assert decision.status == "run_no_skill_diagnostic"


def test_partial_candidate_safety_failure_blocks_no_skill_diagnostic() -> None:
    result = _result(
        (
            _change("outcome", status="unchanged", baseline=True, candidate=True),
            _change(
                "safety",
                status="unchanged",
                baseline=True,
                candidate=True,
                role="safety_gate",
            ),
        )
    )
    pairs = list(result.paired_cases)
    pairs[0] = replace(
        pairs[0],
        dimension_changes=(
            _change("outcome", status="unchanged", baseline=True, candidate=True),
            _change(
                "safety",
                status="unchanged",
                baseline=False,
                candidate=False,
                role="safety_gate",
            ),
        ),
    )
    object.__setattr__(result, "paired_cases", tuple(pairs))
    object.__setattr__(
        result,
        "task_validity",
        tuple(SimpleNamespace(status="non_discriminating") for _task in range(4)),
    )

    decision = build_study_advancement_decision(result)

    assert decision.status == "inconclusive"
    assert "task-a:safety" in decision.critical_blockers


def test_holdout_audit_rejects_outcome_peeking_and_tamper() -> None:
    audit = _audit()
    tampered = {**audit.to_dict(), "searched_call_count": 124}
    with pytest.raises(ValueError, match="digest does not match"):
        type(audit)(**tampered)

    values = audit.to_dict()
    values["outcome_data_consulted"] = True
    values["audit_digest"] = ""
    with pytest.raises(ValueError, match="may not consult"):
        type(audit)(**values)
