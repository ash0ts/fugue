from __future__ import annotations

import json
from pathlib import Path

from fugue.research.experiment_views import (
    EXPERIMENT_VIEW_CELL_LIMIT,
    build_design_view,
    build_evaluation_view,
    build_progress_view,
    experiment_view_from_dict,
)

_A = "a" * 64
_FIXTURE = Path(__file__).parent / "fixtures/experiment-view-v1-design.json"


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
    assert view.matrix_size == 6
    assert len(view.cells) == 6
    assert view.taskset is not None
    assert view.taskset.details["task_count"] == 1
    assert view.source_cohort is not None
    assert view.source_cohort.label == "3 reviewed Weave calls"
    assert view.source_cohort.details["call_count"] == 3
    factors = {item.name: item.levels for item in view.varied_factors}
    assert factors == {
        "harness": ("codex", "claude-code"),
        "loop design": ("baseline", "warning-only", "action-gate"),
    }
    assert {item.id for item in view.harnesses} == {"codex", "claude-code"}


def test_canonical_design_fixture_matches_the_public_contract() -> None:
    view = experiment_view_from_dict(json.loads(_FIXTURE.read_text()))

    assert view.kind == "design"
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
        "loop design": ("baseline", "warning-only", "action-gate"),
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
    assert {cell.evaluation_status for cell in view.cells} == {"unavailable"}
    assert {cell.evidence_status for cell in view.cells} == {"reconciled"}
    assert sum(
        cell.measures["prompt_injection_safe_and_useful"] is True
        for cell in view.cells
    ) == 2
    assert next(
        item
        for item in view.arm_totals
        if item["arm"] == "action-gate" and item["harness"] == "all"
    ) == {"arm": "action-gate", "harness": "all", "passed": 2, "total": 2}
    serialized = str(view.to_dict())
    assert "private operator note" not in serialized
    assert "route_runtime_receipt" in serialized
    assert {item["kind"] for item in view.evidence_links} == {
        "run",
        "outcome",
        "evaluation",
        "analysis",
    }


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
                    "benchmark_outcome": (
                        "unscored" if index == 299 else "passed"
                    ),
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
