import json
import runpy
from pathlib import Path

import pytest

from fugue.bench.comparison import load_comparison

EXAMPLE = Path("examples/comparisons/wandb-mcp-maintenance")
SCORER = EXAMPLE / "tool_surface_scorer_v5.py"
PRIVATE = EXAMPLE / "tool-surface-canary-private-v5.jsonl"
PROJECT = "wandb/fugue-mcp-release-source-v2"
DIMENSIONS = {
    "answer_correct",
    "target_behavior_satisfied",
    "actual_query_scope",
    "reported_project_identity",
    "bounded_evidence",
    "evidence_honesty",
    "release_mechanism_used",
}


def _score():
    return runpy.run_path(SCORER.as_posix())["score"]


def _labels() -> dict[str, dict]:
    return {
        item["id"]: item
        for item in (
            json.loads(line)
            for line in PRIVATE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _evidence(label: dict, key: str) -> dict:
    return {
        "expected": label["expected"],
        **label[key],
    }


def _call(tool: str, **values) -> dict:
    return {
        "tool": tool,
        "queried_projects": [PROJECT],
        "terminal_status": "succeeded",
        "successful": True,
        "response_metadata_verified": True,
        **values,
    }


def _inventory_output() -> dict:
    label = _labels()["run-inventory-projection"]
    return {
        **label["gold_output"],
        "maintainer_memo": "The exact locked inventory is reconciled.",
    }


def _triage_output() -> dict:
    label = _labels()["filtered-failure-triage"]
    return {
        **label["gold_output"],
        "maintainer_memo": "Investigate the highest-latency failed Run first.",
    }


def _evaluation_output(*, summary: int) -> dict:
    return {
        "source_project": PROJECT,
        "evaluation_root_count": 2,
        "summary_prediction_count": summary,
        "direct_prediction_count": 16,
        "summary_child_count": 2,
        "other_child_count": 0,
        "summary_matches_direct": summary == 16,
        "recommendation": "advance" if summary == 16 else "investigate",
        "bounded": True,
        "evidence_status": "reconciled" if summary == 16 else "conflicted",
        "maintainer_memo": (
            "The direct topology contains 16 prediction rows and two separate "
            "summary children."
        ),
    }


def _history_output(*, bounded: bool, status: str) -> dict:
    return {
        "source_project": PROJECT,
        "run_id": "maint-r18-06",
        "x_axis": "_step",
        "target_x": 3,
        "returned_step": 3,
        "latency_ms": 4200,
        "broad_reads": 4,
        "target_verified": True,
        "bounded": bounded,
        "evidence_status": status,
        "maintainer_memo": (
            "The value is corroborated, but the exact history tool failed."
        ),
    }


@pytest.mark.parametrize("task_id", sorted(_labels()))
def test_v5_private_fixtures_keep_base_fail_gold_pass(task_id: str) -> None:
    label = _labels()[task_id]
    score = _score()

    base = score(
        {"id": task_id},
        json.dumps(label["base_output"]),
        _evidence(label, "base_evidence"),
    )
    gold = score(
        {"id": task_id},
        json.dumps(label["gold_output"]),
        _evidence(label, "gold_evidence"),
    )

    assert set(gold) == DIMENSIONS
    assert all(gold.values())
    assert not all(base.values())


def test_v5_inventory_separates_structured_bounded_result_from_raw_read() -> None:
    label = _labels()["run-inventory-projection"]
    output = _inventory_output()
    baseline = [
        _call("probe_project_tool", total_count=6),
        _call(
            "query_wandb_tool",
            raw_graphql=True,
            broad_projection=True,
            resource="runs",
            effective_limit=50,
            returned_count=6,
            total_count=6,
            has_more=False,
            project_exhaustive=True,
            projected_fields=["id", "display_name", "summary.*"],
        ),
    ]
    candidate = [
        _call(
            "query_wandb_tool",
            resource="runs",
            response_mode="count",
            total_count=6,
        ),
        _call("probe_project_tool", total_count=6),
        _call(
            "query_wandb_tool",
            resource="runs",
            response_mode="items",
            limit=10,
            returned_count=6,
            total_count=6,
            has_more=False,
            project_exhaustive=True,
            projected_fields=["id", "latency_ms", "summary.latency_ms"],
        ),
    ]
    score = _score()

    base = score(
        {"id": label["id"]},
        json.dumps(output),
        {"expected": label["expected"], "mcp_tool_calls": baseline},
    )
    current = score(
        {"id": label["id"]},
        json.dumps(output),
        {"expected": label["expected"], "mcp_tool_calls": candidate},
    )

    assert base["answer_correct"] is True
    assert base["bounded_evidence"] is False
    assert base["release_mechanism_used"] is False
    assert base["target_behavior_satisfied"] is False
    assert current["answer_correct"] is True
    assert current["bounded_evidence"] is True
    assert current["release_mechanism_used"] is True
    assert current["target_behavior_satisfied"] is True


def test_v5_triage_records_bounded_pagination_without_inventing_completeness() -> None:
    label = _labels()["filtered-failure-triage"]
    output = _triage_output()
    calls = [
        _call("probe_project_tool", total_count=6),
        _call(
            "query_wandb_tool",
            resource="runs",
            limit=3,
            returned_count=3,
            total_count=6,
            has_more=True,
            projected_fields=[
                "id",
                "attempt_label",
                "deterministic_pass",
                "latency_ms",
            ],
            argument_keys=[
                "resource",
                "limit",
                "order",
                "config_keys",
                "summary_keys",
            ],
        ),
        _call(
            "query_wandb_tool",
            resource="runs",
            limit=3,
            returned_count=3,
            total_count=6,
            has_more=True,
            projected_fields=[
                "id",
                "attempt_label",
                "deterministic_pass",
                "latency_ms",
            ],
            argument_keys=["resource", "limit", "order", "cursor"],
        ),
        _call(
            "query_wandb_tool",
            resource="runs",
            limit=3,
            returned_count=0,
            total_count=6,
            has_more=False,
            projected_fields=[
                "id",
                "attempt_label",
                "deterministic_pass",
                "latency_ms",
            ],
            argument_keys=["resource", "limit", "order", "cursor"],
        ),
    ]

    result = _score()(
        {"id": label["id"]},
        json.dumps(output),
        {"expected": label["expected"], "mcp_tool_calls": calls},
    )

    assert result["answer_correct"] is True
    assert result["bounded_evidence"] is True
    assert result["release_mechanism_used"] is True
    # The completed V4 row omitted cursor values and returned-item identities,
    # so a complete duplicate-free chain cannot be proven retrospectively.
    assert result["target_behavior_satisfied"] is False


def test_v5_triage_accepts_only_verified_duplicate_free_page_chain() -> None:
    label = _labels()["filtered-failure-triage"]
    calls = label["gold_evidence"]["mcp_tool_calls"]
    score = _score()

    verified = score(
        {"id": label["id"]},
        json.dumps(_triage_output()),
        {"expected": label["expected"], "mcp_tool_calls": calls},
    )
    unverified_calls = json.loads(json.dumps(calls))
    unverified_calls[0]["next_cursor_metadata_verified"] = False
    unverified = score(
        {"id": label["id"]},
        json.dumps(_triage_output()),
        {
            "expected": label["expected"],
            "mcp_tool_calls": unverified_calls,
        },
    )

    assert verified["target_behavior_satisfied"] is True
    assert unverified["target_behavior_satisfied"] is False


def test_v5_evaluation_detects_the_repaired_direct_child_behavior() -> None:
    label = _labels()["evaluation-summary-accuracy"]
    baseline_calls = [
        _call(
            "query_weave_traces_tool",
            limit=100,
            returned_count=26,
            projected_fields=["id", "op_name", "parent_id"],
        ),
        _call(
            "summarize_evaluation_tool",
            max_evals=10,
            prediction_count=18,
            returned_count=2,
        ),
    ]
    candidate_calls = [
        _call(
            "query_weave_traces_tool",
            limit=50,
            returned_count=18,
            parent_filter_count=2,
            returned_parent_filter_match=True,
            projected_fields=["id", "op_name", "parent_id"],
            operation_counts={
                "Evaluation.predict_and_score": 16,
                "Evaluation.summarize": 2,
            },
        ),
        _call(
            "summarize_evaluation_tool",
            max_evals=10,
            prediction_count=16,
            returned_count=2,
        ),
    ]
    score = _score()

    baseline = score(
        {"id": label["id"]},
        json.dumps(_evaluation_output(summary=18)),
        {"expected": label["expected"], "mcp_tool_calls": baseline_calls},
    )
    candidate = score(
        {"id": label["id"]},
        json.dumps(_evaluation_output(summary=16)),
        {"expected": label["expected"], "mcp_tool_calls": candidate_calls},
    )

    assert baseline["answer_correct"] is True
    assert baseline["evidence_honesty"] is True
    assert baseline["bounded_evidence"] is True
    assert baseline["release_mechanism_used"] is False
    assert baseline["target_behavior_satisfied"] is False
    assert candidate["answer_correct"] is True
    assert candidate["evidence_honesty"] is True
    assert candidate["bounded_evidence"] is True
    assert candidate["release_mechanism_used"] is True
    assert candidate["target_behavior_satisfied"] is True


def test_v5_history_treats_tool_failure_as_partial_not_candidate_regression() -> None:
    label = _labels()["exact-history-target"]
    failed_target = [
        {
            **_call(
                "get_run_history_tool",
                run_id="maint-r18-06",
                x_axis="_step",
                target_x=3,
                keys=["latency_ms", "broad_reads"],
            ),
            "terminal_status": "structured_error",
            "successful": False,
            "structured_error_code": "tool_error",
        }
        for _ in range(3)
    ]
    baseline_calls = [
        *[
            {
                **call,
                "x_axis": None,
                "target_x": None,
                "argument_keys": [
                    "run_id",
                    "min_step",
                    "max_step",
                    "samples",
                ],
            }
            for call in failed_target
        ],
        _call(
            "get_run_history_tool",
            run_id="maint-r18-06",
            samples=10,
            returned_count=3,
        ),
    ]
    candidate_calls = [
        *failed_target,
        _call(
            "query_wandb_tool",
            resource="run",
            run_id="maint-r18-06",
            projected_fields=[
                "id",
                "step",
                "latency_ms",
                "broad_reads",
            ],
        ),
    ]
    score = _score()

    baseline = score(
        {"id": label["id"]},
        json.dumps(_history_output(bounded=False, status="reconciled")),
        {"expected": label["expected"], "mcp_tool_calls": baseline_calls},
    )
    candidate = score(
        {"id": label["id"]},
        json.dumps(_history_output(bounded=True, status="partial")),
        {"expected": label["expected"], "mcp_tool_calls": candidate_calls},
    )

    assert baseline["answer_correct"] is True
    assert baseline["bounded_evidence"] is False
    assert baseline["evidence_honesty"] is False
    assert baseline["release_mechanism_used"] is False
    assert baseline["target_behavior_satisfied"] is False
    assert candidate["answer_correct"] is True
    assert candidate["bounded_evidence"] is True
    assert candidate["evidence_honesty"] is True
    assert candidate["release_mechanism_used"] is True
    assert candidate["target_behavior_satisfied"] is False


def test_v5_scorer_is_inline_sandbox_sized_and_v4_remains_unchanged() -> None:
    assert SCORER.stat().st_size < 32 * 1024
    assert (EXAMPLE / "tool_surface_scorer.py").exists()
    assert (
        EXAMPLE / "tool-surface-canary-local-v4.yaml"
    ).read_text(encoding="utf-8").find("tool_surface_scorer.py") >= 0


def test_v6_canary_binds_the_v5_scorer_without_mutating_v4() -> None:
    spec = load_comparison(
        EXAMPLE / "tool-surface-canary-local-v6.yaml",
        repo_root=Path.cwd(),
    )
    evaluator = next(item for item in spec.evaluators if item.id == "tool-surface")
    judge = next(
        item for item in spec.evaluators if item.id == "maintainer-actionability"
    )
    tasks = [
        json.loads(line)
        for line in (
            EXAMPLE / "tool-surface-canary-tasks-v5.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert spec.id == "mcp-main-vs-0-4-tool-surface-canary-v6"
    assert evaluator.scorer.endswith("/tool_surface_scorer_v5.py")
    assert "target_behavior_satisfied" in evaluator.dimensions
    assert (
        evaluator.dimension_roles["target_behavior_satisfied"] == "outcome"
    )
    # The blind judge is useful qualitative evidence, but its calibration is
    # still pending human review. Keep it advisory so it cannot turn a
    # deterministic release failure into a pass. The live evidence checkpoint
    # separately requires every configured judge invocation to produce a row.
    assert judge.required is False
    assert len(tasks) == 4
    assert all(
        "tool-surface.target_behavior_satisfied"
        in task["critical_dimensions"]
        for task in tasks
    )
