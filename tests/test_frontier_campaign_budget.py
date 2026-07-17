from __future__ import annotations

import pytest

from tools.frontier_campaign import (
    admit_cohort,
    complete_cohort,
    reconcile_failed_cohort,
    record_incident,
    validate_canary,
)


def _row(
    index: int,
    *,
    cost: float | None = 2.0,
    weave_cost: float | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "prediction_schema_version": 1,
        "prediction_id": f"prediction-{index}",
        "execution_kind": "agent",
        "model": "wandb/zai-org/GLM-5.2",
        "status": "passed",
        "trace_link_status": "verified",
        "weave_turn_count": 1,
        "root_span_id": f"root-{index}",
        "weave_root_span_ids": [f"root-{index}"],
        "observed_conversation_id": f"conversation-{index}",
        "weave_conversation_ids": [f"conversation-{index}"],
        "cost_usd": cost,
        "weave_total_cost_usd": weave_cost,
    }


def test_admission_conservatively_prices_unmeasured_canary_cells() -> None:
    canary = validate_canary(
        [_row(1, cost=2.0), _row(2, cost=None)],
        expected_predictions=2,
        model="wandb/zai-org/GLM-5.2",
    )
    ledger = admit_cohort(
        {"schema_version": 1, "cap_usd": 2000.0, "cohorts": []},
        cohort_id="glm",
        model="wandb/zai-org/GLM-5.2",
        canary=canary,
        cohort_predictions=32,
        safety_margin=1.5,
    )

    assert canary["accounted_cost_usd"] == 4.0
    assert ledger["accounted_cost_usd"] == 100.0
    assert ledger["remaining_budget_usd"] == 1900.0


def test_admission_refuses_missing_cost_and_budget_overflow() -> None:
    with pytest.raises(ValueError, match="without any measured"):
        validate_canary(
            [_row(1, cost=None)],
            expected_predictions=1,
            model="wandb/zai-org/GLM-5.2",
        )
    canary = validate_canary(
        [_row(1, cost=50.0)],
        expected_predictions=1,
        model="wandb/zai-org/GLM-5.2",
    )
    with pytest.raises(ValueError, match="exceeds remaining"):
        admit_cohort(
            {"schema_version": 1, "cap_usd": 2000.0, "cohorts": []},
            cohort_id="too-expensive",
            model="wandb/zai-org/GLM-5.2",
            canary=canary,
            cohort_predictions=32,
            safety_margin=1.5,
        )


def test_admission_uses_the_larger_measured_local_or_weave_cost() -> None:
    canary = validate_canary(
        [
            _row(1, cost=1.0, weave_cost=4.0),
            _row(2, cost=3.0, weave_cost=2.0),
            _row(3, cost=None, weave_cost=None),
        ],
        expected_predictions=3,
        model="wandb/zai-org/GLM-5.2",
    )

    assert canary["measured_predictions"] == 2
    assert canary["observed_cost_usd"] == 7.0
    assert canary["maximum_measured_cell_cost_usd"] == 4.0
    assert canary["accounted_cost_usd"] == 11.0


def test_completion_reconciles_actual_cost_without_fabricating_public_values() -> None:
    canary = validate_canary(
        [_row(1, cost=1.0)],
        expected_predictions=1,
        model="wandb/zai-org/GLM-5.2",
    )
    admitted = admit_cohort(
        {"schema_version": 1, "cap_usd": 2000.0, "cohorts": []},
        cohort_id="glm",
        model="wandb/zai-org/GLM-5.2",
        canary=canary,
        cohort_predictions=2,
        safety_margin=1.5,
    )

    completed = complete_cohort(
        admitted, cohort_id="glm", rows=[_row(1, cost=3.0), _row(2, cost=None)]
    )

    [entry] = completed["cohorts"]
    assert entry["status"] == "completed"
    assert entry["actual_cost_usd"] == 6.0
    assert entry["accounted_cost_usd"] == 7.0
    assert completed["remaining_budget_usd"] == 1993.0


def test_canary_requires_exact_agent_contract() -> None:
    row = _row(1)
    row["weave_root_span_ids"] = ["root-1", "root-2"]
    with pytest.raises(ValueError, match="exactly one Agent root"):
        validate_canary(
            [row],
            expected_predictions=1,
            model="wandb/zai-org/GLM-5.2",
        )


def test_canary_allows_nested_continuations_below_one_agent_root() -> None:
    row = _row(1)
    row["weave_turn_count"] = 3

    canary = validate_canary(
        [row],
        expected_predictions=1,
        model="wandb/zai-org/GLM-5.2",
    )

    assert canary["predictions"] == 1


def test_completion_keeps_terminal_harness_failures_as_evidence() -> None:
    canary = validate_canary(
        [_row(1, cost=1.0)],
        expected_predictions=1,
        model="wandb/zai-org/GLM-5.2",
    )
    admitted = admit_cohort(
        {"schema_version": 1, "cap_usd": 2000.0, "cohorts": []},
        cohort_id="glm",
        model="wandb/zai-org/GLM-5.2",
        canary=canary,
        cohort_predictions=1,
        safety_margin=1.5,
    )
    row = _row(1, cost=3.0)
    row["weave_turn_count"] = 3
    row["adapter_outcome"] = {"execution": {"state": "failed"}}

    completed = complete_cohort(admitted, cohort_id="glm", rows=[row])

    assert completed["cohorts"][0]["status"] == "completed"


def test_canary_rejects_harness_failure_even_when_the_row_is_terminal() -> None:
    row = _row(1)
    row["adapter_outcome"] = {"execution": {"state": "failed"}}

    with pytest.raises(ValueError, match="completed adapter execution"):
        validate_canary(
            [row],
            expected_predictions=1,
            model="wandb/zai-org/GLM-5.2",
        )


def test_failed_cohort_reconciles_spend_without_claiming_completion() -> None:
    canary = validate_canary(
        [_row(1, cost=1.0)],
        expected_predictions=1,
        model="wandb/zai-org/GLM-5.2",
    )
    admitted = admit_cohort(
        {"schema_version": 1, "cap_usd": 2000.0, "cohorts": []},
        cohort_id="glm",
        model="wandb/zai-org/GLM-5.2",
        canary=canary,
        cohort_predictions=2,
        safety_margin=1.5,
    )
    incomplete = [_row(1, cost=3.0), _row(2, cost=None)]
    incomplete[1]["trace_link_status"] = "missing"
    incomplete[1]["weave_root_span_ids"] = ["root-a", "root-b"]

    reconciled = reconcile_failed_cohort(
        admitted,
        cohort_id="glm",
        rows=incomplete,
        reason="one prediction emitted two Agent roots",
    )

    [entry] = reconciled["cohorts"]
    assert entry["status"] == "failed"
    assert entry["failure_reason"] == "one prediction emitted two Agent roots"
    assert entry["actual_cost_usd"] == 6.0
    assert entry["accounted_cost_usd"] == 7.0
    assert reconciled["remaining_budget_usd"] == 1993.0


def test_failed_cohort_accounts_for_unpublished_inflight_predictions() -> None:
    canary = validate_canary(
        [_row(1, cost=1.0)],
        expected_predictions=1,
        model="wandb/zai-org/GLM-5.2",
    )
    admitted = admit_cohort(
        {"schema_version": 1, "cap_usd": 100.0, "cohorts": []},
        cohort_id="partial",
        model="wandb/zai-org/GLM-5.2",
        canary=canary,
        cohort_predictions=4,
        safety_margin=1.5,
    )

    reconciled = reconcile_failed_cohort(
        admitted,
        cohort_id="partial",
        rows=[_row(1, cost=3.0)],
        unpublished_accounted_cost_usd=6.0,
        reason="three cells were cancelled before publication",
    )

    [entry] = reconciled["cohorts"]
    assert entry["status"] == "failed"
    assert entry["observed_predictions"] == 1
    assert entry["unpublished_accounted_cost_usd"] == 6.0
    assert entry["actual_cost_usd"] == 9.0
    assert entry["accounted_cost_usd"] == 10.0


def test_completion_records_actual_overspend_for_future_admission() -> None:
    canary = validate_canary(
        [_row(1, cost=1.0)],
        expected_predictions=1,
        model="wandb/zai-org/GLM-5.2",
    )
    admitted = admit_cohort(
        {"schema_version": 1, "cap_usd": 5.0, "cohorts": []},
        cohort_id="glm",
        model="wandb/zai-org/GLM-5.2",
        canary=canary,
        cohort_predictions=1,
        safety_margin=1.0,
    )

    completed = complete_cohort(
        admitted,
        cohort_id="glm",
        rows=[_row(1, cost=10.0)],
    )

    assert completed["accounted_cost_usd"] == 11.0
    assert completed["remaining_budget_usd"] == -6.0


def test_budget_incident_reserves_unreconciled_attempt_spend() -> None:
    ledger = record_incident(
        {"schema_version": 1, "cap_usd": 20.0, "cohorts": []},
        incident_id="cancelled-canary",
        model="anthropic/claude-sonnet-5",
        accounted_cost_usd=7.5,
        reason="the run was cancelled before publication closed",
    )

    assert ledger["accounted_cost_usd"] == 7.5
    assert ledger["remaining_budget_usd"] == 12.5
    assert ledger["cohorts"] == [
        {
            "cohort_id": "cancelled-canary",
            "model": "anthropic/claude-sonnet-5",
            "status": "incident",
            "failure_reason": "the run was cancelled before publication closed",
            "actual_cost_usd": None,
            "accounted_cost_usd": 7.5,
        }
    ]


@pytest.mark.parametrize("cost", [0.0, -1.0, float("inf"), float("nan")])
def test_budget_incident_rejects_invalid_cost(cost: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        record_incident(
            {"schema_version": 1, "cap_usd": 20.0, "cohorts": []},
            incident_id="bad-cost",
            model="anthropic/claude-sonnet-5",
            accounted_cost_usd=cost,
            reason="invalid",
        )
