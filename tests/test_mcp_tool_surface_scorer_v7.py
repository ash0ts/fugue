import json
import runpy
from pathlib import Path

import pytest

from fugue.bench.comparison import load_comparison

EXAMPLE = Path("examples/comparisons/wandb-mcp-maintenance")
SCORER = EXAMPLE / "tool_surface_scorer_v7.py"
PRIVATE = EXAMPLE / "tool-surface-confirmation-private-v8.jsonl"
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


def _call(tool: str, **values) -> dict:
    return {
        "tool": tool,
        "queried_projects": [PROJECT],
        "terminal_status": "succeeded",
        "successful": True,
        "response_metadata_verified": True,
        **values,
    }


def _evidence(label: dict, calls: list[dict]) -> dict:
    return {"expected": label["expected"], "mcp_tool_calls": calls}


def _evaluation_output(summary: int) -> dict:
    matches = summary == 16
    return {
        "source_project": PROJECT,
        "evaluation_root_count": 2,
        "summary_prediction_count": summary,
        "direct_prediction_count": 16,
        "summary_child_count": 2,
        "other_child_count": 0,
        "summary_matches_direct": matches,
        "recommendation": "advance" if matches else "investigate",
        "bounded": True,
        "evidence_status": "reconciled",
        "maintainer_memo": "Reconcile the summary with both direct topologies.",
    }


@pytest.mark.parametrize("task_id", sorted(_labels()))
def test_v7_private_fixtures_keep_base_fail_gold_pass(task_id: str) -> None:
    label = _labels()[task_id]
    score = _score()

    base = score(
        {"id": task_id},
        json.dumps(label["base_output"]),
        _evidence(label, label["base_evidence"]["mcp_tool_calls"]),
    )
    gold = score(
        {"id": task_id},
        json.dumps(label["gold_output"]),
        _evidence(label, label["gold_evidence"]["mcp_tool_calls"]),
    )

    assert set(gold) == DIMENSIONS
    assert all(gold.values())
    assert not all(base.values())


def test_v7_inventory_accepts_one_combined_count_and_items_response() -> None:
    label = _labels()["run-inventory-projection"]
    combined = _call(
        "query_wandb_tool",
        resource="runs",
        limit=50,
        returned_count=6,
        total_count=6,
        has_more=False,
        project_exhaustive=True,
        projected_fields=["id", "attempt_label", "latency_ms"],
        truncation_applied=False,
    )

    result = _score()(
        {"id": label["id"]},
        json.dumps(label["gold_output"]),
        _evidence(label, [combined]),
    )

    assert result["answer_correct"] is True
    assert result["bounded_evidence"] is True
    assert result["release_mechanism_used"] is True
    assert result["target_behavior_satisfied"] is True


def test_v7_inventory_keeps_partial_projection_as_mechanism_only() -> None:
    label = _labels()["run-inventory-projection"]
    partial_projection = _call(
        "query_wandb_tool",
        resource="runs",
        limit=50,
        returned_count=6,
        total_count=6,
        has_more=False,
        project_exhaustive=True,
        projected_fields=["id", "latency_ms"],
        truncation_applied=False,
    )

    result = _score()(
        {"id": label["id"]},
        json.dumps(label["gold_output"]),
        _evidence(label, [partial_projection]),
    )

    assert result["answer_correct"] is True
    assert result["bounded_evidence"] is True
    assert result["release_mechanism_used"] is True
    assert result["target_behavior_satisfied"] is False


def test_v7_evaluation_accepts_bounded_trace_id_reconciliation() -> None:
    label = _labels()["evaluation-summary-accuracy"]
    calls = [
        _call(
            "query_weave_traces_tool",
            limit=100,
            returned_count=26,
            returned_trace_ids_count=1,
            returned_trace_ids_digest="1" * 64,
            projected_fields=["id", "op_name", "parent_id", "trace_id"],
            truncation_applied=False,
        ),
        _call(
            "query_weave_traces_tool",
            limit=100,
            returned_count=26,
            returned_trace_ids_count=1,
            returned_trace_ids_digest="2" * 64,
            projected_fields=["id", "op_name", "parent_id", "trace_id"],
            truncation_applied=False,
        ),
        _call(
            "summarize_evaluation_tool",
            max_evals=10,
            returned_count=2,
            prediction_count=16,
        ),
    ]

    candidate = _score()(
        {"id": label["id"]},
        json.dumps(_evaluation_output(16)),
        _evidence(label, calls),
    )
    baseline_calls = json.loads(json.dumps(calls))
    baseline_calls[-1]["prediction_count"] = 18
    baseline = _score()(
        {"id": label["id"]},
        json.dumps(_evaluation_output(18)),
        _evidence(label, baseline_calls),
    )

    assert candidate["answer_correct"] is True
    assert candidate["bounded_evidence"] is True
    assert candidate["release_mechanism_used"] is True
    assert candidate["target_behavior_satisfied"] is True
    assert baseline["answer_correct"] is True
    assert baseline["evidence_honesty"] is True
    assert baseline["release_mechanism_used"] is True
    assert baseline["target_behavior_satisfied"] is False


def test_v7_pagination_records_verified_chain_but_rejects_empty_probe() -> None:
    label = _labels()["filtered-failure-triage"]
    output = label["gold_output"]
    direct = label["gold_evidence"]["mcp_tool_calls"]
    direct_result = _score()(
        {"id": label["id"]},
        json.dumps(output),
        _evidence(label, direct),
    )

    with_empty_probe = json.loads(json.dumps(direct))
    second = with_empty_probe[-1]
    second["has_more"] = True
    second["next_cursor_present"] = True
    second["next_cursor_metadata_verified"] = True
    second["next_cursor_digest"] = "3" * 64
    with_empty_probe.append(
        _call(
            "query_wandb_tool",
            resource="runs",
            limit=3,
            total_count=6,
            returned_count=0,
            has_more=False,
            projected_fields=[
                "attempt_label",
                "deterministic_pass",
                "latency_ms",
            ],
            argument_keys=["resource", "limit", "order", "cursor"],
            cursor_present=True,
            cursor_metadata_verified=True,
            cursor_digest="3" * 64,
            next_cursor_present=False,
            next_cursor_metadata_verified=True,
            pagination_metadata_verified=True,
            returned_object_id_hashes=[],
            returned_object_id_count=0,
            returned_object_id_metadata_verified=True,
            returned_object_ids_unique=True,
        )
    )
    probe_result = _score()(
        {"id": label["id"]},
        json.dumps(output),
        _evidence(label, with_empty_probe),
    )

    assert direct_result["target_behavior_satisfied"] is True
    assert probe_result["answer_correct"] is True
    assert probe_result["bounded_evidence"] is True
    assert probe_result["release_mechanism_used"] is True
    assert probe_result["target_behavior_satisfied"] is False


def test_v7_history_keeps_fallback_outcome_separate_from_failed_tool() -> None:
    label = _labels()["exact-history-target"]
    output = {
        **label["gold_output"],
        "evidence_status": "partial",
        "maintainer_memo": "The exact history tool failed; this is partial.",
    }
    calls = [
        {
            **_call(
                "get_run_history_tool",
                run_id="maint-r18-06",
                x_axis="_step",
                target_x=3,
                keys=["latency_ms", "broad_reads"],
            ),
            "successful": False,
            "terminal_status": "structured_error",
            "structured_error_code": "tool_error",
        },
        _call(
            "query_wandb_tool",
            resource="run",
            run_id="maint-r18-06",
            projected_fields=["id"],
        ),
    ]

    result = _score()(
        {"id": label["id"]},
        json.dumps(output),
        _evidence(label, calls),
    )

    assert result["answer_correct"] is True
    assert result["bounded_evidence"] is True
    assert result["evidence_honesty"] is True
    assert result["release_mechanism_used"] is True
    assert result["target_behavior_satisfied"] is False


def test_v10_confirmation_uses_v7_scorer_and_mechanism_is_noncritical() -> None:
    spec = load_comparison(
        EXAMPLE / "tool-surface-confirmation-local-v10.yaml",
        repo_root=Path.cwd(),
    )
    evaluator = next(item for item in spec.evaluators if item.id == "tool-surface")
    tasks = [
        json.loads(line)
        for line in (
            EXAMPLE / "tool-surface-confirmation-tasks-v8.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert spec.id == "mcp-main-vs-0-4-tool-surface-confirmation-v10"
    assert spec.execution.attempts == 2
    assert spec.execution.concurrency == 1
    assert spec.execution.evidence_checkpoint_cells == 1
    assert spec.execution.max_cost_usd == 10
    assert evaluator.scorer.endswith("/tool_surface_scorer_v7.py")
    assert (
        evaluator.dimension_roles["target_behavior_satisfied"] == "mechanism"
    )
    assert evaluator.dimension_roles["release_mechanism_used"] == "mechanism"
    assert len(tasks) == 4
    assert all(
        "tool-surface.target_behavior_satisfied"
        not in task["critical_dimensions"]
        for task in tasks
    )
    assert all(len(task["critical_dimensions"]) == 5 for task in tasks)
    assert (
        spec.supersedes[0].result_digest
        == "cd083539eb06f1253fd663110b551696cc8fe08a7a10dfb164c3c9e8b1c6de12"
    )


def test_v7_scorer_is_bound_to_v10_and_preserves_v6_inputs() -> None:
    assert SCORER.stat().st_size < 32 * 1024
    assert "tool_surface_scorer_v5.py" in (
        EXAMPLE / "tool-surface-canary-local-v6.yaml"
    ).read_text(encoding="utf-8")
    assert "tool_surface_scorer_v7.py" in (
        EXAMPLE / "tool-surface-confirmation-local-v10.yaml"
    ).read_text(encoding="utf-8")
