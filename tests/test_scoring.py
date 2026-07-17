from __future__ import annotations

import json
from pathlib import Path

import pytest

from fugue.bench.scoring import (
    SelectionPolicy,
    build_treatment_selection_lock,
    factorial_difference_in_differences,
    read_treatment_selection_lock,
    select_candidate_configuration,
    write_treatment_selection_lock,
)


def _rows(candidate: str, *, cost: float | None, wall: float = 2.0):
    return [
        {
            "candidate_id": candidate,
            "comparison_example_id": f"example-{index}",
            "harness": "codex",
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

    broken = next(
        item for item in selection.candidates if item.candidate_id == "broken"
    )
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


def test_variant_selection_pairs_each_latin_square_row_to_its_baseline() -> None:
    harnesses = ("hermes", "openclaw", "claude-code", "codex")
    rows = []
    for index, harness in enumerate(harnesses, start=1):
        common = {
            "task_name": f"task-{index}",
            "harness": harness,
            "comparison_example_id": f"example-{index}",
            "trial_index": 1,
            "trace_link_status": "linked",
            "context_registration_status": "registered",
            "cost_usd": 1.0,
            "wall_time_sec": 2.0,
        }
        rows.extend(
            (
                {**common, "variant_id": "none", "pass": False},
                {
                    **common,
                    "variant_id": "vector",
                    "pass": True,
                    "localization_recall_at_10": 1.0,
                    "localization_mrr": 1.0,
                },
                {**common, "variant_id": "bm25", "pass": index == 1},
            )
        )

    selection = select_candidate_configuration(
        rows,
        SelectionPolicy(
            selection_unit="variant",
            baseline_variant_id="none",
            required_examples=4,
            required_harnesses=harnesses,
            require_agent_links=True,
            require_registration=True,
            tie_breakers=(
                "localization_recall_at_10",
                "localization_mrr",
                "recoverable_error_rate",
                "cost_per_success",
            ),
            bootstrap_samples=200,
        ),
        seed="latin-square",
    )

    assert selection.selected_candidate_id == "vector"
    vector = next(
        item for item in selection.candidates if item.candidate_id == "vector"
    )
    assert vector.paired_pass_rate_delta == 1.0
    assert selection.to_dict()["selection_unit"] == "variant"


def test_cross_harness_factorial_contrast_uses_paired_variant_deltas() -> None:
    rows = []
    outcomes = {
        "baseline": (False, False),
        "memory-only": (True, False),
        "policy-only": (False, True),
        "memory-policy": (True, True),
    }
    for harness_index, harness in enumerate(("claude-code", "codex")):
        for variant_id, passes in outcomes.items():
            rows.append(
                {
                    "variant_id": variant_id,
                    "comparison_example_id": "example-a",
                    "task_name": "task-a",
                    "harness": harness,
                    "trial_index": 1,
                    "pass": passes[harness_index],
                    "trace_link_status": "linked",
                    "cost_usd": 1.0,
                    "wall_time_sec": 1.0,
                }
            )
    selection = select_candidate_configuration(
        rows,
        SelectionPolicy(
            selection_unit="variant",
            baseline_variant_id="baseline",
            required_examples=1,
            required_harnesses=("claude-code", "codex"),
            require_agent_links=True,
            bootstrap_samples=200,
        ),
        seed="factorial",
    )

    contrast = factorial_difference_in_differences(
        selection,
        factor_a_id="memory-only",
        factor_b_id="policy-only",
        combined_id="memory-policy",
    )

    assert contrast["factor_a_delta"] == 0.5
    assert contrast["factor_b_delta"] == 0.5
    assert contrast["combined_delta"] == 1.0
    assert contrast["interaction"] == 0.0


def test_treatment_selection_lock_is_immutable_and_digest_verified(
    tmp_path: Path,
) -> None:
    lock = build_treatment_selection_lock(
        source_commit="a" * 40,
        calibration_snapshot_sha256="b" * 64,
        discovery_snapshot_sha256="c" * 64,
        rankings=tuple({"variant_id": value} for value in ("a", "b", "c", "d")),
        selected_variants=("a", "b", "c"),
    )
    path = write_treatment_selection_lock(tmp_path / "selection.json", lock)

    assert read_treatment_selection_lock(path) == lock
    payload = json.loads(path.read_text())
    payload["selected_variants"] = ["b", "c", "d"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest"):
        read_treatment_selection_lock(path)
