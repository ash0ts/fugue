from __future__ import annotations

import json
from pathlib import Path

import pytest

from fugue.bench.candidates import stable_digest
from fugue.bench.intervention_provenance import (
    build_intervention_component_lock,
)
from fugue.bench.scoring import (
    SelectionPolicy,
    build_intervention_selection_lock,
    build_treatment_selection_lock,
    factorial_difference_in_differences,
    read_intervention_selection_lock,
    read_treatment_selection_lock,
    select_candidate_configuration,
    write_intervention_selection_lock,
    write_treatment_selection_lock,
)


def _selected_components():
    return (
        build_intervention_component_lock(
            kind="skill",
            component_id="loop-intervention-skill",
            lock_digest="7" * 64,
            repository="https://github.com/wandb/fugue",
            source_commit="8" * 40,
            source_tree="9" * 40,
        ),
        build_intervention_component_lock(
            kind="mcp",
            component_id="loop-intervention-mcp",
            lock_digest="a" * 64,
            repository="https://github.com/wandb/wandb-mcp-server",
            source_commit="b" * 40,
            source_tree="c" * 40,
            release_target="wandb-mcp-server Python package 0.4",
            superseded_release_candidate_sha="d" * 40,
            release_requalification_required=True,
        ),
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


def test_intervention_selection_lock_binds_candidates_and_discovery(
    tmp_path: Path,
) -> None:
    rankings = (
        {"variant_id": "production", "candidate_digest": "a" * 64},
        {"variant_id": "skill-only", "candidate_digest": "b" * 64},
        {"variant_id": "mcp-only", "candidate_digest": "c" * 64},
        {"variant_id": "combined", "candidate_digest": "d" * 64},
    )
    lock = build_intervention_selection_lock(
        experiment_id="claude-loop-skill-mcp",
        source_commit="e" * 40,
        source_tree="9" * 40,
        source_dirty_digest="",
        failure_lock_sha256="4" * 64,
        discovery_suite_sha256="5" * 64,
        holdout_suite_sha256="6" * 64,
        analysis_snapshot_sha256="f" * 64,
        discovery_run_snapshot_sha256s=("1" * 64,),
        comparison_example_ids=("2" * 64, "3" * 64),
        discovery_variant_ids=(
            "production",
            "skill-only",
            "mcp-only",
            "combined",
        ),
        baseline_variant_id="production",
        selected_variant_id="combined",
        selected_components=_selected_components(),
        rankings=rankings,
        decision="recommend",
        rationale="combined is the only arm with a preregistered deterministic gain",
        failure_locked_at="2026-07-30T10:00:00Z",
        suites_frozen_at="2026-07-30T10:05:00Z",
        discovery_completed_at="2026-07-30T11:00:00Z",
        selection_locked_at="2026-07-30T11:05:00Z",
    )
    path = write_intervention_selection_lock(tmp_path / "selection.json", lock)

    assert read_intervention_selection_lock(path) == lock
    assert lock.failure_lock_sha256 == "4" * 64
    assert lock.discovery_suite_sha256 == "5" * 64
    assert lock.holdout_suite_sha256 == "6" * 64
    assert set(lock.discovery_variant_ids) == {
        "production",
        "skill-only",
        "mcp-only",
        "combined",
    }
    payload = json.loads(path.read_text())
    payload["rankings"][3]["candidate_digest"] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest"):
        read_intervention_selection_lock(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value.__setitem__(
                "discovery_variant_ids",
                ["production", "skill-only", "combined"],
            ),
            "every predeclared discovery arm",
        ),
        (
            lambda value: value.__setitem__(
                "suites_frozen_at",
                "2026-07-30T11:01:00+00:00",
            ),
            "intervention chronology",
        ),
        (
            lambda value: value.__setitem__(
                "holdout_suite_sha256",
                value["discovery_suite_sha256"],
            ),
            "independently frozen",
        ),
    ],
)
def test_intervention_selection_lock_rejects_incomplete_prefreeze_contract(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    rankings = (
        {"variant_id": "production", "candidate_digest": "a" * 64},
        {"variant_id": "skill-only", "candidate_digest": "b" * 64},
        {"variant_id": "mcp-only", "candidate_digest": "c" * 64},
        {"variant_id": "combined", "candidate_digest": "d" * 64},
    )
    lock = build_intervention_selection_lock(
        experiment_id="claude-loop-skill-mcp",
        source_commit="e" * 40,
        source_tree="9" * 40,
        source_dirty_digest="",
        failure_lock_sha256="4" * 64,
        discovery_suite_sha256="5" * 64,
        holdout_suite_sha256="6" * 64,
        analysis_snapshot_sha256="f" * 64,
        discovery_run_snapshot_sha256s=("1" * 64,),
        comparison_example_ids=("2" * 64, "3" * 64),
        discovery_variant_ids=(
            "production",
            "skill-only",
            "mcp-only",
            "combined",
        ),
        baseline_variant_id="production",
        selected_variant_id="combined",
        selected_components=_selected_components(),
        rankings=rankings,
        decision="recommend",
        rationale="combined met the preregistered gate",
        failure_locked_at="2026-07-30T10:00:00Z",
        suites_frozen_at="2026-07-30T10:05:00Z",
        discovery_completed_at="2026-07-30T11:00:00Z",
        selection_locked_at="2026-07-30T11:05:00Z",
    )
    payload = lock.to_dict()
    mutate(payload)
    payload["lock_sha256"] = stable_digest(
        {**payload, "lock_sha256": ""}
    )
    path = tmp_path / f"{message.replace(' ', '-')}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        read_intervention_selection_lock(path)


def test_selection_policy_rejects_arms_without_required_paired_gain() -> None:
    rows = []
    outcomes = {
        "production": (False, True),
        "skill-only": (False, True),
        "mcp-only": (True, True),
        "combined": (True, True),
    }
    for variant, passes in outcomes.items():
        for index, passed in enumerate(passes, start=1):
            skill_ids = (
                [f"wandb-evidence-{variant}"]
                if variant in {"skill-only", "combined"}
                else []
            )
            rows.append(
                {
                    "variant_id": variant,
                    "comparison_example_id": f"example-{index}",
                    "task_name": f"task-{index}",
                    "harness": "claude-code",
                    "trial_index": 1,
                    "pass": passed,
                    "cost_usd": 1.0,
                    "wall_time_sec": 1.0,
                    "skill_ids": skill_ids,
                    "skill_invocation_evidence": {
                        "status": "observed",
                        "skills_invoked": skill_ids,
                    },
                    "weave_tool_names": {"summarize_evaluation_tool": 1},
                }
            )
    selection = select_candidate_configuration(
        rows,
        SelectionPolicy(
            selection_unit="variant",
            baseline_variant_id="production",
            required_examples=2,
            required_harnesses=("claude-code",),
            require_skill_invocation=True,
            required_any_tool_names=("summarize_evaluation_tool",),
            minimum_paired_pass_rate_delta=0.5,
        ),
        seed="skill-mcp",
    )

    scores = {item.candidate_id: item for item in selection.candidates}
    assert scores["skill-only"].eligible is False
    assert scores["mcp-only"].eligible is True
    assert scores["combined"].eligible is True
