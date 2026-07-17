from __future__ import annotations

import pytest

from tools.frontier_campaign import admit_cohort, complete_cohort, validate_canary


def _row(index: int, *, cost: float | None = 2.0) -> dict[str, object]:
    return {
        "schema_version": 1,
        "prediction_schema_version": 1,
        "prediction_id": f"prediction-{index}",
        "execution_kind": "agent",
        "model": "wandb/zai-org/GLM-5.2",
        "status": "passed",
        "trace_link_status": "verified",
        "weave_agent_root_count": 1,
        "weave_conversation_count": 1,
        "cost_usd": cost,
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
    row["weave_agent_root_count"] = 2
    with pytest.raises(ValueError, match="exactly one Agent root"):
        validate_canary(
            [row],
            expected_predictions=1,
            model="wandb/zai-org/GLM-5.2",
        )
