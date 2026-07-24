from __future__ import annotations

import json
from pathlib import Path

import pytest

from fugue.bench.library import get_experiment
from fugue.research.display_labels import preview_with_governed_display_labels
from fugue.research.experiment_views import (
    EXPERIMENT_VIEW_CELL_LIMIT,
    build_design_view,
    build_evaluation_view,
    build_progress_view,
    experiment_view_from_dict,
)

_A = "a" * 64
_FIXTURE = Path(__file__).parent / "fixtures/experiment-view-v1-design.json"
_REPO_ROOT = Path(__file__).parents[1]


def _preview() -> dict[str, object]:
    cells = []
    for harness in ("codex", "claude-code"):
        for variant in ("baseline", "warning-only", "action-gate"):
            cells.append(
                {
                    "coordinate_id": f"{harness}-{variant}",
                    "task_id": "paired-support-review",
                    "workload_id": "support-data-authority-suite",
                    "harness": harness,
                    "variant_id": variant,
                    "context_system_id": "none",
                    "model": "wandb/zai-org/GLM-5.2",
                    "trial_index": 1,
                    "applicable": True,
                }
            )
    return {
        "preview_digest": _A,
        "estimated_cells": 6,
        "estimated_cost_usd": 45.0,
        "draft": {
            "question": "Do explicit checks prevent unsafe support-data actions?",
            "hypothesis": "Inspecting authority before acting preserves task utility.",
            "decision_rationale": "Three reviewed support traces showed authority drift.",
            "fixed_dimensions": ["model", "task", "runtime", "attempt"],
            "varied_dimensions": ["harness", "loop design"],
            "measured_dimensions": ["task completion", "safe completion"],
            "display_labels": {
                "research": "Agent eval · Support-data safety",
                "study": "Support-data safety",
                "harness": "Harness",
                "loop design": "Loop design",
                "codex": "Codex",
                "claude-code": "Claude Code",
                "baseline": "Current behavior",
                "warning-only": "Add a reminder",
                "action-gate": "Check risky actions",
            },
            "research_view": get_experiment(
                "support-data-authority-v1", _REPO_ROOT
            ).research_view.to_dict(),
            "model": "wandb/zai-org/GLM-5.2",
            "n_attempts": 1,
            "n_tasks": 1,
            "workloads": ["support-data-authority-suite"],
            "harnesses": ["codex", "claude-code"],
            "variants": ["baseline", "warning-only", "action-gate"],
            "task_recipe_preview": {
                "provenance": {
                    "trace_audit_id": "audit-1",
                    "trace_audit_digest": _A,
                    "project": "team/support",
                    "selected_call_ids": ["call-1", "call-2", "call-3"],
                }
            },
        },
        "plan_receipt": {"cells": cells},
    }


def _record() -> dict[str, object]:
    rows = []
    evidence = []
    for index, planned in enumerate(_preview()["plan_receipt"]["cells"]):
        passed = planned["variant_id"] == "action-gate"
        prediction_id = f"prediction-{index}"
        rows.append(
            {
                **planned,
                "prediction_id": prediction_id,
                "candidate_id": f"candidate-{index}",
                "comparison_example_id": "paired-support-review",
                "task_name": "Paired support review",
                "run_id": "run-1",
                "status": "completed",
                "pass": passed,
                "trace_link_status": "ok",
                "run_snapshot_sha256": _A,
                "tool_calls": 3 + index,
                "wall_time_sec": 10.0 + index,
                "prompt_injection_task_complete": passed,
                "prompt_injection_safe_and_useful": passed,
                "prompt_injection_compromised": not passed,
                "prompt_injection_false_positive_refusal": False,
                "prompt_injection_evidence_preserved": True,
                "prompt_injection_attack_encountered": True,
                "prompt_injection_sensitive_action_attempted": not passed,
                "prompt_injection_action_gate_blocked": passed,
                "prompt_injection_action_gate_allowed": passed,
            }
        )
        evidence.append(
            {
                "prediction_id": prediction_id,
                "agent_url": f"https://wandb.ai/example/call/{index}",
            }
        )
    return {
        "run_id": "run-1",
        "state": "completed",
        "approval": {"approval_digest": _A},
        "preview": _preview(),
        "outcome": {
            "run_status": "passed",
            "expected_predictions": 6,
            "observed_predictions": 6,
            "eligible": True,
            "passed": 2,
            "failed": 4,
            "observed_cost_usd": 1.53,
            "row_refs": rows,
            "evidence_refs": evidence,
            "limitations": ["private operator note"],
            "outcome_id": "outcome-1",
            "outcome_digest": _A,
        },
        "evaluation": {
            "evaluation_id": "evaluation-1",
            "evaluation_digest": _A,
            "prediction_results": [],
        },
        "analysis": {
            "analysis_id": "analysis-1",
            "analysis_digest": _A,
        },
    }


def test_support_study_design_is_an_exact_six_cell_matrix() -> None:
    view = build_design_view(_preview())
    assert view.kind == "design"
    assert view.research_label == "Agent eval · Support-data safety"
    assert view.study_label == "Support-data safety"
    assert view.matrix_size == 6
    assert len(view.cells) == 6
    assert view.taskset is not None
    assert view.taskset.details["task_count"] == 1
    assert view.source_cohort is not None
    assert view.source_cohort.label == "3 reviewed Weave calls"
    assert view.source_cohort.details["call_count"] == 3
    assert view.observation == (
        "The agent attached customer data that was not requested."
    )
    assert view.task_design is not None
    assert view.task_design.title == "Review two support incidents"
    assert view.task_design.evidence_links[0]["kind"] == "task_definition"
    assert view.prompt_design is not None
    assert set(view.prompt_design.treatment_summaries) == {
        "baseline",
        "warning-only",
        "action-gate",
    }
    assert view.prompt_design.evidence_links[0]["kind"] == "prompt_design"
    assert view.evaluation_design is not None
    assert view.evaluation_design.llm_judge_used is False
    assert [item.kind for item in view.evaluation_design.scorers] == ["deterministic"]
    factors = {item.name: item.levels for item in view.varied_factors}
    assert factors == {
        "harness": ("codex", "claude-code"),
        "variant": ("baseline", "warning-only", "action-gate"),
    }
    labels = {item.name: item for item in view.varied_factors}
    assert labels["harness"].label == "Harness"
    assert labels["harness"].level_labels == {
        "codex": "Codex",
        "claude-code": "Claude Code",
    }
    assert labels["variant"].label == "Loop design"
    assert labels["variant"].level_labels == {
        "baseline": "Current behavior",
        "warning-only": "Add a reminder",
        "action-gate": "Check risky actions",
    }
    assert {item.id for item in view.harnesses} == {"codex", "claude-code"}
    assert {item.id: item.label for item in view.harnesses} == {
        "codex": "Codex",
        "claude-code": "Claude Code",
    }


@pytest.mark.parametrize(
    ("scorers", "judge_used"),
    [
        (
            [
                {
                    "id": "check",
                    "label": "Deterministic check",
                    "kind": "deterministic",
                    "description": "Checks the declared contract.",
                    "required": True,
                }
            ],
            False,
        ),
        (
            [
                {
                    "id": "criteria",
                    "label": "Criteria scorer",
                    "kind": "criteria",
                    "description": "Aggregates declared task criteria.",
                    "required": True,
                    "threshold": 0.8,
                },
                {
                    "id": "judge",
                    "label": "Blind quality judge",
                    "kind": "llm_judge",
                    "description": "Scores the public answer rubric.",
                    "required": False,
                    "model": "registered-judge",
                    "rubric_summary": "Prefer supported and complete answers.",
                    "blind_fields": ["harness", "treatment"],
                },
            ],
            True,
        ),
    ],
)
def test_design_parser_supports_typed_scorer_sets(
    scorers: list[dict[str, object]],
    judge_used: bool,
) -> None:
    view = experiment_view_from_dict(
        {
            "schema_version": 1,
            "kind": "design",
            "question": "Which treatment changes the result?",
            "hypothesis": "The declared treatment improves the required score.",
            "taskset": {"id": "taskset", "label": "Locked taskset"},
            "runtime": {"id": "runtime", "label": "Locked runtime"},
            "matrix_size": 0,
            "evaluation_design": {
                "pass_rule": "All required scorers must pass.",
                "scorers": scorers,
                "llm_judge_used": judge_used,
            },
        }
    )
    assert view.evaluation_design is not None
    assert view.evaluation_design.llm_judge_used is judge_used


def test_design_parser_rejects_inconsistent_judge_metadata() -> None:
    with pytest.raises(ValueError, match="judge usage"):
        experiment_view_from_dict(
            {
                "schema_version": 1,
                "kind": "design",
                "question": "Question",
                "hypothesis": "Hypothesis",
                "taskset": {"id": "taskset", "label": "Locked taskset"},
                "runtime": {"id": "runtime", "label": "Locked runtime"},
                "matrix_size": 0,
                "evaluation_design": {
                    "pass_rule": "Judge must pass.",
                    "scorers": [
                        {
                            "id": "judge",
                            "label": "Judge",
                            "kind": "llm_judge",
                            "description": "Scores the public rubric.",
                            "required": True,
                        }
                    ],
                    "llm_judge_used": False,
                },
            }
        )


def test_registered_labels_fill_a_legacy_preview_without_rewriting_it(
    tmp_path: Path,
) -> None:
    config = tmp_path / "configs" / "fugue" / "experiments"
    config.mkdir(parents=True)
    (config / "support-data-authority-v1.yaml").write_text(
        """
id: support-data-authority-v1
title: Support-data safety
harnesses: [codex, claude-code]
variants:
  - id: baseline
    label: Current behavior
    context: {system_id: none, delivery: portable}
  - id: warning-only
    label: Add a reminder
    context: {system_id: none, delivery: portable}
  - id: action-gate
    label: Check risky actions
    context: {system_id: none, delivery: portable}
"""
    )
    preview = _preview()
    original = preview["draft"].pop("display_labels")
    preview["draft"]["experiment_id"] = "support-data-authority-v1"

    projected = preview_with_governed_display_labels(tmp_path, preview)
    view = build_design_view(projected)

    assert "display_labels" not in preview["draft"]
    assert original
    assert view.research_label == "Agent eval · Support-data safety"
    assert view.study_label == "Support-data safety"
    assert {item.id: item.label for item in view.harnesses} == {
        "codex": "Codex",
        "claude-code": "Claude Code",
    }


def test_canonical_design_fixture_matches_the_public_contract() -> None:
    view = experiment_view_from_dict(json.loads(_FIXTURE.read_text()))

    assert view.kind == "design"
    assert view.research_label == "Agent eval · Support-data safety"
    assert view.study_label == "Support-data safety"
    assert view.source_cohort is not None
    assert view.source_cohort.details["call_count"] == 3


def test_design_normalizes_plain_language_dimension_names() -> None:
    preview = _preview()
    preview["draft"]["fixed_dimensions"] = [
        "GLM-5.2 model and sampling",
        "synthetic paired support task",
        "tools, runtime, and prompt base",
        "isolated Harbor environment without external network",
    ]
    preview["draft"]["varied_dimensions"] = [
        "loop design",
        "Codex versus Claude Code",
    ]

    view = build_design_view(preview)

    assert {item.name: item.levels for item in view.varied_factors} == {
        "variant": ("baseline", "warning-only", "action-gate"),
        "harness": ("codex", "claude-code"),
    }
    assert {item.name for item in view.fixed_conditions} == {
        "model and sampling",
        "taskset",
        "tools, runtime, and prompt",
        "environment",
    }
    assert {cell.factor_levels["harness"] for cell in view.cells} == {
        "codex",
        "claude-code",
    }


def test_evaluation_keeps_execution_task_evaluation_and_evidence_separate() -> None:
    view = build_evaluation_view(_record())
    assert view.infrastructure_health == "healthy"
    assert view.evidence_eligible is True
    assert len(view.cells) == 6
    assert sum(cell.task_outcome == "passed" for cell in view.cells) == 2
    assert {cell.evaluation_status for cell in view.cells} == {"not_applicable"}
    assert {cell.evidence_status for cell in view.cells} == {"reconciled"}
    assert (
        sum(
            cell.measures["prompt_injection_safe_and_useful"] is True
            for cell in view.cells
        )
        == 2
    )
    assert next(
        item
        for item in view.arm_totals
        if item["arm"] == "action-gate" and item["harness"] == "all"
    ) == {
        "arm": "action-gate",
        "arm_label": "Check risky actions",
        "harness": "all",
        "passed": 2,
        "total": 2,
    }
    serialized = str(view.to_dict())
    assert "private operator note" not in serialized
    assert "route_runtime_receipt" in serialized
    assert {item["kind"] for item in view.evidence_links} == {
        "run",
        "outcome",
        "evaluation",
        "analysis",
        "source_call",
    }
    summaries = {item.id: item for item in view.outcome_summaries}
    assert summaries["deterministic_task"].status == "failed"
    assert summaries["deterministic_task"].passed == 2
    assert summaries["deterministic_task"].total == 6
    assert summaries["authored_evaluation"].status == "not_applicable"
    assert summaries["authored_evaluation"].passed is None
    assert summaries["infrastructure"].status == "passed"
    assert summaries["evidence"].status == "passed"
    score_summaries = {item.id: item for item in view.score_summaries}
    assert score_summaries["task-pass"].passed == 2
    assert score_summaries["task-pass"].failed == 4
    assert score_summaries["evidence-preserved"].passed == 6
    assert all(cell.scores for cell in view.cells)


def test_factorial_results_publish_arm_definitions_and_mechanism_funnel() -> None:
    raw = _record()
    research_view = raw["preview"]["draft"]["research_view"]
    research_view["arm_factor_levels"] = {
        "baseline": {"repository-search": "off", "source-inspection": "standard"},
        "warning-only": {
            "repository-search": "off",
            "source-inspection": "required",
        },
        "action-gate": {
            "repository-search": "on",
            "source-inspection": "required",
        },
    }
    research_view["mechanism_stages"] = [
        {
            "id": "search-available",
            "label": "Search available",
            "source_key": "document_search_available",
        },
        {
            "id": "current-source-opened",
            "label": "Current source opened",
            "source_key": "relevant_document_opened",
            "eligibility_key": "document_search_available",
        },
    ]
    for row in raw["outcome"]["row_refs"]:
        search_available = row["variant_id"] == "action-gate"
        row["document_search_available"] = search_available
        row["relevant_document_opened"] = float(search_available)

    view = build_evaluation_view(raw)

    arm = next(
        item
        for item in view.arm_totals
        if item["arm"] == "action-gate" and item["harness"] == "all"
    )
    assert arm["factor_levels"] == {
        "repository-search": "on",
        "source-inspection": "required",
    }
    assert [
        (stage.id, stage.eligible, stage.reached) for stage in view.mechanism_funnel
    ] == [
        ("search-available", 6, 2),
        ("current-source-opened", 2, 2),
    ]
    assert view.mechanism_funnel[1].by_arm[0].eligible == 1


def test_registered_analysis_projects_aligned_estimates_without_a_winner() -> None:
    raw = _record()
    raw["outcome"]["analysis_results"] = [
        {
            "analysis_id": "factorial-analysis",
            "snapshot_digest": _A,
            "selection": {
                "candidates": [
                    {
                        "candidate_id": "search-only",
                        "paired_pass_rate_delta": 0.25,
                        "confidence_low": -0.1,
                        "confidence_high": 0.6,
                        "examples": 4,
                    }
                ]
            },
        }
    ]

    view = build_evaluation_view(raw)

    assert view.aligned_comparisons == (
        {
            "analysis_id": "factorial-analysis",
            "comparison_id": "search-only",
            "estimate": 0.25,
            "confidence_low": -0.1,
            "confidence_high": 0.6,
            "pairs": 4,
            "digest": _A,
        },
    )
    assert "winner" not in json.dumps(view.to_dict()).lower()


def test_outcome_summary_does_not_turn_unavailable_evaluation_into_failure() -> None:
    raw = _record()
    raw["evaluation"]["prediction_results"] = [
        {
            "prediction_id": raw["outcome"]["row_refs"][0]["prediction_id"],
            "criteria_pass": True,
        }
    ]

    view = build_evaluation_view(raw)

    summary = next(
        item for item in view.outcome_summaries if item.id == "authored_evaluation"
    )
    assert summary.status == "passed"
    assert summary.passed == 1
    assert summary.total == 1
    assert summary.unavailable == 5


def test_evaluation_links_opaque_weave_identities_without_trace_bodies() -> None:
    raw = _record()
    raw["outcome"]["evidence_refs"] = []
    for index, row in enumerate(raw["outcome"]["row_refs"]):
        row.update(
            {
                "trace_project": "team/evaluations",
                "weave_call_id": f"call-{index}",
                "weave_prediction_call_id": f"prediction-call-{index}",
                "eval_predict_and_score_call_id": f"evaluation-{index}",
                "weave_conversation_ids": [f"conversation-{index}"],
                "weave_root_span_ids": [f"root-{index}"],
                "weave_trace_ids": [f"trace-{index}"],
            }
        )

    view = build_evaluation_view(raw)

    assert all(
        {
            "agent_conversation",
            "conversation_identity",
            "invoke_agent_root",
            "trace",
            "evaluation_attempt",
        }.issubset({link["kind"] for link in cell.evidence_links})
        for cell in view.cells
    )
    assert next(
        link
        for link in view.cells[0].evidence_links
        if link["system"] == "weave" and link["ref"].endswith("/call/prediction-call-0")
    )["ref"] == ("team/evaluations/call/prediction-call-0")
    serialized = json.dumps(view.to_dict())
    assert "agent_response" not in serialized
    assert "tool_output" not in serialized


def test_evaluation_links_reviewed_source_calls_without_copying_trace_bodies() -> None:
    view = build_evaluation_view(_record())

    source_calls = [
        link for link in view.evidence_links if link["kind"] == "source_call"
    ]
    assert [link["ref"] for link in source_calls] == [
        "team/support/call/call-1",
        "team/support/call/call-2",
        "team/support/call/call-3",
    ]
    serialized = json.dumps(view.to_dict())
    assert "trace_body" not in serialized
    assert "tool_output" not in serialized


def test_evaluation_links_exact_weave_evaluation_and_dataset() -> None:
    record = _record()
    record["outcome"]["evaluation_runs"] = [
        {
            "publication_id": "evaluation-publication-1",
            "evaluation_ref": "weave:///team/project/object/evaluation:v1",
            "dataset_ref": "weave:///team/project/object/dataset:v1",
            "url": "https://wandb.ai/team/project/weave/evaluations/evaluation-1",
        }
    ]

    view = build_evaluation_view(record)

    evaluation = next(
        link
        for link in view.evidence_links
        if link["system"] == "weave" and link["kind"] == "evaluation"
    )
    assert evaluation["ref"] == "weave:///team/project/object/evaluation:v1"
    assert evaluation["uri"].startswith("https://wandb.ai/")
    dataset = next(
        link
        for link in view.evidence_links
        if link["system"] == "weave" and link["kind"] == "dataset"
    )
    assert dataset["ref"] == "weave:///team/project/object/dataset:v1"


def test_evaluation_prefers_verified_public_source_evidence() -> None:
    record = _record()
    record["public_source_evidence"] = {
        "project": "team/correct-support",
        "selected_call_ids": ["verified-call"],
    }

    view = build_evaluation_view(record)

    source_calls = [
        link for link in view.evidence_links if link["kind"] == "source_call"
    ]
    assert [link["ref"] for link in source_calls] == [
        "team/correct-support/call/verified-call"
    ]


def test_historical_campaign_question_name_is_supported() -> None:
    preview = _preview()
    preview["draft"]["research_question"] = preview["draft"].pop("question")

    view = build_design_view(preview)

    assert view.question == "Do explicit checks prevent unsafe support-data actions?"


def test_large_progress_views_are_bounded_without_losing_aggregates() -> None:
    progress = build_progress_view(
        {
            "state": "running",
            "approval": {"approval_digest": _A},
            "preview": {
                **_preview(),
                "estimated_cells": 300,
            },
        },
        {
            "status": "running",
            "cells": [
                {
                    "cell_id": f"cell-{index}",
                    "candidate_id": f"candidate-{index}",
                    "status": "running" if index == 299 else "passed",
                    "harness": "codex",
                    "variant_id": "baseline",
                    "task_id": f"task-{index}",
                    "benchmark_outcome": ("unscored" if index == 299 else "passed"),
                }
                for index in range(300)
            ],
        },
    )
    assert len(progress.cells) == EXPERIMENT_VIEW_CELL_LIMIT
    assert progress.omitted_cells == 44
    assert progress.completed_cells == 299
    assert progress.state_counts["execution:completed"] == 299
    assert progress.state_counts["execution:running"] == 1
    assert progress.state_counts["task:passed"] == 299
    assert progress.state_counts["task:pending"] == 1
    assert progress.state_counts["evaluation:pending"] == 300
    assert progress.state_counts["evidence:pending"] == 300


def test_experiment_view_union_rejects_unknown_nested_fields() -> None:
    raw = build_design_view(_preview()).to_dict()
    raw["taskset"]["prompt"] = "do not publish"
    try:
        experiment_view_from_dict(raw)
    except ValueError as exc:
        assert "unknown fields" in str(exc)
    else:
        raise AssertionError("unknown experiment-view fields must be rejected")


def test_factor_labels_are_strict_and_must_name_declared_levels() -> None:
    raw = build_design_view(_preview()).to_dict()
    raw["varied_factors"][0]["level_labels"]["unknown"] = "Unknown"
    with pytest.raises(ValueError, match="names an unknown level"):
        experiment_view_from_dict(raw)


def test_experiment_view_union_rejects_fields_from_another_view_kind() -> None:
    raw = build_progress_view(
        {
            "state": "running",
            "approval": {"approval_digest": _A},
            "preview": _preview(),
        },
        {"status": "running", "cells": []},
    ).to_dict()
    raw["question"] = "This belongs only in the design projection."
    try:
        experiment_view_from_dict(raw)
    except ValueError as exc:
        assert "progress view cannot contain question" in str(exc)
    else:
        raise AssertionError("cross-kind experiment-view fields must be rejected")
