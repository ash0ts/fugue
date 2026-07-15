from __future__ import annotations

from fugue.bench.scoring import SelectionPolicy, select_candidate_configuration


def _rows(candidate: str, *, cost: float | None, wall: float = 2.0):
    return [
        {
            "candidate_id": candidate,
            "comparison_example_id": f"example-{index}",
            "trial_index": 1,
            "pass": True,
            "cost_usd": cost,
            "wall_time_sec": wall,
            "weave_tool_error_count": 0,
        }
        for index in range(1, 5)
    ]


def test_quality_first_selection_uses_measured_cost_after_quality_gate():
    rows = [*_rows("expensive", cost=1.0), *_rows("cheap", cost=0.25)]

    selection = select_candidate_configuration(
        rows,
        SelectionPolicy(bootstrap_samples=200),
        seed="snapshot",
    )

    assert selection.best_candidate_id == "cheap"
    assert selection.selected_candidate_id == "cheap"
    assert selection.decision == "recommend"
    assert all(candidate.competitive for candidate in selection.candidates)


def test_missing_cost_is_not_treated_as_zero():
    rows = [*_rows("unknown-cost", cost=None), *_rows("measured", cost=0.5)]

    selection = select_candidate_configuration(
        rows,
        SelectionPolicy(bootstrap_samples=200),
        seed="snapshot",
    )

    assert selection.selected_candidate_id == "measured"
    unknown = next(
        item for item in selection.candidates if item.candidate_id == "unknown-cost"
    )
    assert unknown.cost_per_success is None


def test_duplicate_or_incomplete_candidate_grid_is_ineligible():
    rows = [*_rows("complete", cost=0.5), *_rows("broken", cost=0.1)[:-1]]
    rows.append(dict(rows[-1]))

    selection = select_candidate_configuration(
        rows,
        SelectionPolicy(bootstrap_samples=200),
        seed="snapshot",
    )

    broken = next(item for item in selection.candidates if item.candidate_id == "broken")
    assert not broken.eligible
    assert "duplicate candidate/example/trial row" in broken.reasons
    assert "incomplete comparison grid" in broken.reasons
    assert selection.selected_candidate_id is None
    assert selection.decision == "blocked"
    complete = next(
        item for item in selection.candidates if item.candidate_id == "complete"
    )
    assert "candidates do not share one comparison grid" in complete.reasons


def test_incumbent_requires_explicit_improvement_for_promotion():
    same = [*_rows("incumbent", cost=0.5), *_rows("candidate", cost=0.49)]
    improved = [*_rows("incumbent", cost=0.5), *_rows("candidate", cost=0.3)]
    policy = SelectionPolicy(
        bootstrap_samples=200,
        incumbent_candidate_id="incumbent",
    )

    unchanged = select_candidate_configuration(same, policy, seed="same")
    promoted = select_candidate_configuration(improved, policy, seed="improved")

    assert unchanged.decision == "no_promotion"
    assert promoted.decision == "promote"
