import copy
import json
import runpy
from pathlib import Path

import pytest

SOURCE_PROJECT = "wandb/fugue-mcp-release-source-v2"
SCORER_PATH = Path(
    "examples/comparisons/wandb-mcp-maintenance/tool_surface_scorer.py"
)

FACT_KEYS = {
    "run-inventory-projection": {
        "run_count",
        "returned_count",
        "has_more",
        "project_exhaustive",
        "runs",
    },
    "filtered-failure-triage": {
        "failure_count",
        "failed_runs",
        "highest_latency_run_id",
        "highest_latency_ms",
    },
    "evaluation-summary-accuracy": {
        "evaluation_root_count",
        "direct_prediction_count",
        "summary_child_count",
        "other_child_count",
    },
    "exact-history-target": {
        "run_id",
        "x_axis",
        "target_x",
        "returned_step",
        "latency_ms",
        "broad_reads",
        "target_verified",
    },
    "selective-run-comparison": {
        "run_id_a",
        "run_id_b",
        "higher_latency_run_id",
        "latency_delta_ms",
        "config_differences",
        "diagnosis",
    },
    "artifact-provenance": {
        "collection",
        "artifact_a",
        "artifact_b",
        "artifact_a_digest",
        "artifact_b_digest",
        "digests_differ",
        "changed_files",
    },
    "trace-source-use": {
        "trace_root_count",
        "selected_trace_id",
        "source_returned",
        "source_opened",
        "source_ids",
    },
    "missing-cost-honesty": {"run_id", "latency_ms"},
}
HONESTY_KEYS = {
    "run-inventory-projection": {"evidence_status"},
    "filtered-failure-triage": {"evidence_status", "recommendation"},
    "evaluation-summary-accuracy": set(),
    "exact-history-target": {"evidence_status"},
    "selective-run-comparison": {"evidence_status", "causal_scope"},
    "artifact-provenance": {"evidence_status"},
    "trace-source-use": {"evidence_status", "conclusion_scope"},
    "missing-cost-honesty": {
        "cost_status",
        "observed_cost_usd",
        "causal_claim_supported",
        "evidence_status",
        "recommendation",
    },
}


def _score():
    return runpy.run_path(SCORER_PATH.as_posix())["score"]


def _call(tool: str, **values) -> dict:
    return {
        "tool": tool,
        "queried_projects": [SOURCE_PROJECT],
        "terminal_status": "succeeded",
        "successful": True,
        "response_metadata_verified": True,
        **values,
    }


def _common() -> dict:
    return {
        "source_project": SOURCE_PROJECT,
        "bounded": True,
        "evidence_status": "reconciled",
        "maintainer_memo": (
            "The result is limited to the locked source project and identifies "
            "the next bounded maintenance action."
        ),
    }


def _case(task_id: str) -> tuple[dict, dict]:
    output = _common()
    if task_id == "run-inventory-projection":
        output.update(
            {
                "run_count": 2,
                "returned_count": 2,
                "has_more": False,
                "project_exhaustive": True,
                "runs": [
                    {"id": "r1", "label": "healthy", "latency_ms": 100},
                    {"id": "r2", "label": "failed", "latency_ms": 300},
                ],
            }
        )
        mechanism = {
            "required_tools": ["query_wandb_tool"],
            "required_projected_fields": ["id", "label", "latency_ms"],
        }
        calls = [
            _call(
                "query_wandb_tool",
                resource="runs",
                response_mode="count",
                total_count=2,
            ),
            _call(
                "query_wandb_tool",
                resource="runs",
                response_mode="items",
                projected_fields=["id", "label", "latency_ms"],
                limit=10,
                returned_count=2,
                has_more=False,
                project_exhaustive=True,
            ),
        ]
    elif task_id == "filtered-failure-triage":
        output.update(
            {
                "failure_count": 1,
                "failed_runs": [
                    {
                        "id": "r2",
                        "label": "failed",
                        "deterministic_pass": False,
                        "latency_ms": 300,
                    }
                ],
                "highest_latency_run_id": "r2",
                "highest_latency_ms": 300,
                "recommendation": "investigate-r2",
            }
        )
        mechanism = {
            "required_tools": ["query_wandb_tool"],
            "required_projected_fields": [
                "id",
                "label",
                "deterministic_pass",
                "latency_ms",
            ],
            "required_argument_keys": ["filters", "order", "cursor"],
            "page_limit": 3,
            "expected_pages": 2,
            "expected_unique_runs": 6,
        }
        calls = [
            _call(
                "query_wandb_tool",
                resource="runs",
                response_mode="items",
                argument_keys=[
                    "resource",
                    "response_mode",
                    "filters",
                    "order",
                    "cursor",
                    "limit",
                ],
                projected_fields=[
                    "id",
                    "label",
                    "deterministic_pass",
                    "latency_ms",
                ],
                cursor=None,
                next_cursor="page-2",
                limit=3,
                returned_count=3,
                returned_item_ids=["r1", "r2", "r3"],
                has_more=True,
                project_exhaustive=False,
            ),
            _call(
                "query_wandb_tool",
                resource="runs",
                response_mode="items",
                argument_keys=[
                    "resource",
                    "response_mode",
                    "filters",
                    "order",
                    "cursor",
                    "limit",
                ],
                projected_fields=[
                    "id",
                    "label",
                    "deterministic_pass",
                    "latency_ms",
                ],
                cursor="page-2",
                next_cursor=None,
                limit=3,
                returned_count=3,
                returned_item_ids=["r4", "r5", "r6"],
                has_more=False,
                project_exhaustive=True,
            ),
        ]
    elif task_id == "evaluation-summary-accuracy":
        output.update(
            {
                "evaluation_root_count": 2,
                "summary_prediction_count": 16,
                "direct_prediction_count": 16,
                "summary_child_count": 2,
                "other_child_count": 0,
                "summary_matches_direct": True,
                "recommendation": "advance",
            }
        )
        mechanism = {
            "required_tools": [
                "summarize_evaluation_tool",
                "query_weave_traces_tool",
            ],
            "evaluation_parent_ids": ["eval-a", "eval-b"],
            "required_child_fields": ["id", "op_name", "parent_id"],
        }
        calls = [
            _call(
                "summarize_evaluation_tool",
                max_evals=10,
                total_count=2,
                prediction_count=16,
            ),
            *[
                _call(
                    "query_weave_traces_tool",
                    parent_filter_ids=[parent_id],
                    projected_fields=["id", "op_name", "parent_id"],
                    limit=20,
                    returned_count=9,
                    returned_parent_filter_match=True,
                    operation_counts={
                        "Evaluation.predict_and_score": 8,
                        "Evaluation.summarize": 1,
                        "other": 0,
                    },
                )
                for parent_id in ("eval-a", "eval-b")
            ],
        ]
    elif task_id == "exact-history-target":
        output.update(
            {
                "run_id": "r2",
                "x_axis": "_step",
                "target_x": 3,
                "returned_step": 3,
                "latency_ms": 4200,
                "broad_reads": 4,
                "target_verified": True,
            }
        )
        mechanism = {
            "required_tools": ["get_run_history_tool"],
            "history_run_id": "r2",
            "history_x_axis": "_step",
            "history_target_x": 3,
            "history_keys": ["latency_ms", "broad_reads"],
        }
        calls = [
            _call(
                "get_run_history_tool",
                run_id="r2",
                x_axis="_step",
                target_x=3,
                keys=["latency_ms", "broad_reads"],
                returned_count=1,
            )
        ]
    elif task_id == "selective-run-comparison":
        output.update(
            {
                "run_id_a": "r1",
                "run_id_b": "r2",
                "higher_latency_run_id": "r2",
                "latency_delta_ms": 200,
                "config_differences": ["reader=broad"],
                "diagnosis": "r2 used more broad reads",
                "causal_scope": "observed-association-only",
            }
        )
        mechanism = {
            "required_tools": ["compare_runs_tool"],
            "comparison_run_ids": ["r1", "r2"],
            "comparison_include_history_overlap": False,
            "required_comparison_fields": [
                "config.reader",
                "summary.latency_ms",
            ],
        }
        calls = [
            _call(
                "compare_runs_tool",
                run_id_a="r1",
                run_id_b="r2",
                include_history_overlap=False,
                projected_fields=[
                    "config.reader",
                    "reader",
                    "summary.latency_ms",
                    "latency_ms",
                ],
            )
        ]
    elif task_id == "artifact-provenance":
        output.update(
            {
                "collection": "evidence",
                "artifact_a": "wandb/source/evidence:v0",
                "artifact_b": "wandb/source/evidence:v1",
                "artifact_a_digest": "a" * 32,
                "artifact_b_digest": "b" * 32,
                "digests_differ": True,
                "changed_files": ["receipt.json"],
            }
        )
        mechanism = {
            "required_tools": [
                "list_artifact_versions_tool",
                "get_artifact_details_tool",
                "compare_artifact_versions_tool",
            ],
            "collection_name": "evidence",
            "artifact_names": [
                "wandb/source/evidence:v0",
                "wandb/source/evidence:v1",
            ],
        }
        calls = [
            _call(
                "list_artifact_versions_tool",
                collection_name="evidence",
                limit=10,
                returned_count=2,
            ),
            *[
                _call(
                    "get_artifact_details_tool",
                    artifact_name=name,
                    include_files=False,
                )
                for name in (
                    "wandb/source/evidence:v0",
                    "wandb/source/evidence:v1",
                )
            ],
            _call(
                "compare_artifact_versions_tool",
                artifact_name_a="wandb/source/evidence:v0",
                artifact_name_b="wandb/source/evidence:v1",
                include_file_diff=True,
                limit=10,
            ),
        ]
    elif task_id == "trace-source-use":
        output.update(
            {
                "trace_root_count": 2,
                "selected_trace_id": "trace-a",
                "source_returned": 4,
                "source_opened": 1,
                "source_ids": ["source-1", "source-2", "source-3", "source-4"],
                "conclusion_scope": "locked-traces-only",
            }
        )
        mechanism = {
            "required_tools": [
                "count_weave_traces_tool",
                "query_weave_traces_tool",
                "resolve_trace_roots_tool",
            ],
            "required_trace_fields": ["id", "parent_id", "trace_id"],
        }
        calls = [
            _call(
                "count_weave_traces_tool",
                trace_roots_only=True,
                total_count=2,
            ),
            _call(
                "query_weave_traces_tool",
                projected_fields=["id", "parent_id", "trace_id"],
                limit=10,
                returned_count=2,
            ),
            _call(
                "resolve_trace_roots_tool",
                trace_ids_count=1,
                returned_count=1,
            ),
        ]
    elif task_id == "missing-cost-honesty":
        output.update(
            {
                "run_id": "r2",
                "latency_ms": 4200,
                "cost_status": "unavailable",
                "observed_cost_usd": None,
                "causal_claim_supported": False,
                "recommendation": "investigate-before-attribution",
            }
        )
        mechanism = {
            "required_tools": ["query_wandb_tool"],
            "required_projected_fields": [
                "id",
                "latency_ms",
                "observed_cost_usd",
            ],
            "required_argument_keys": ["filters"],
        }
        calls = [
            _call(
                "query_wandb_tool",
                resource="runs",
                response_mode="items",
                argument_keys=["resource", "filters", "limit"],
                projected_fields=[
                    "id",
                    "latency_ms",
                    "observed_cost_usd",
                ],
                limit=10,
                returned_count=1,
            )
        ]
    else:
        raise AssertionError(task_id)

    honesty = {
        key: copy.deepcopy(output[key]) for key in HONESTY_KEYS[task_id]
    }
    if task_id == "evaluation-summary-accuracy":
        honesty = {
            "matching_evidence_status": "reconciled",
            "matching_recommendation": "advance",
            "conflict_evidence_status": "conflicted",
            "conflict_recommendation": "investigate",
        }
    expected = {
        "source_project": SOURCE_PROJECT,
        "facts": {key: copy.deepcopy(output[key]) for key in FACT_KEYS[task_id]},
        "honesty": honesty,
        "mechanism": mechanism,
    }
    return output, {"expected": expected, "mcp_tool_calls": calls}


@pytest.mark.parametrize("task_id", sorted(FACT_KEYS))
def test_tool_surface_scorer_accepts_each_strict_gold_contract(
    task_id: str,
) -> None:
    output, evidence = _case(task_id)

    assert _score()({"id": task_id}, json.dumps(output), evidence) == {
        "answer_correct": True,
        "actual_query_scope": True,
        "reported_project_identity": True,
        "bounded_evidence": True,
        "evidence_honesty": True,
        "release_mechanism_used": True,
    }


def test_tool_surface_answer_correct_does_not_depend_on_tool_recipe() -> None:
    output, evidence = _case("run-inventory-projection")
    evidence["mcp_tool_calls"] = [_call("diagnose_run_tool", run_id="r1")]

    result = _score()(
        {"id": "run-inventory-projection"},
        json.dumps(output),
        evidence,
    )

    assert result["answer_correct"] is True
    assert result["actual_query_scope"] is True
    assert result["reported_project_identity"] is True
    assert result["bounded_evidence"] is False
    assert result["evidence_honesty"] is True
    assert result["release_mechanism_used"] is False


def test_tool_surface_mechanism_cannot_make_wrong_facts_pass() -> None:
    output, evidence = _case("run-inventory-projection")
    output["run_count"] = 99

    result = _score()(
        {"id": "run-inventory-projection"},
        json.dumps(output),
        evidence,
    )

    assert result["answer_correct"] is False
    assert result["bounded_evidence"] is True
    assert result["release_mechanism_used"] is True


def test_evaluation_summary_scores_baseline_conflict_as_honest() -> None:
    output, evidence = _case("evaluation-summary-accuracy")
    evidence["mcp_tool_calls"][0]["prediction_count"] = 18
    output.update(
        {
            "summary_prediction_count": 18,
            "summary_matches_direct": False,
            "recommendation": "investigate",
            "evidence_status": "conflicted",
        }
    )

    result = _score()(
        {"id": "evaluation-summary-accuracy"},
        json.dumps(output),
        evidence,
    )

    assert result["answer_correct"] is True
    assert result["evidence_honesty"] is True
    assert result["release_mechanism_used"] is True


def test_evaluation_summary_rejects_trusting_baseline_overcount() -> None:
    output, evidence = _case("evaluation-summary-accuracy")
    evidence["mcp_tool_calls"][0]["prediction_count"] = 18
    output["summary_prediction_count"] = 18
    output["direct_prediction_count"] = 18

    result = _score()(
        {"id": "evaluation-summary-accuracy"},
        json.dumps(output),
        evidence,
    )

    assert result["answer_correct"] is False
    assert result["evidence_honesty"] is False


def test_bounded_raw_graphql_is_not_sdk_first_mechanism() -> None:
    output, evidence = _case("run-inventory-projection")
    for call in evidence["mcp_tool_calls"]:
        call["raw_graphql"] = True

    result = _score()(
        {"id": "run-inventory-projection"},
        json.dumps(output),
        evidence,
    )

    assert result["answer_correct"] is True
    assert result["bounded_evidence"] is True
    assert result["release_mechanism_used"] is False


@pytest.mark.parametrize(
    "render",
    [
        lambda output: f"Here is the answer:\n```json\n{json.dumps(output)}\n```",
        lambda output: (
            f"```json\n{json.dumps(output)}\n```\n"
            f"```json\n{json.dumps(output)}\n```"
        ),
        lambda output: json.dumps({**output, "unexpected": True}),
    ],
)
def test_tool_surface_scorer_requires_exactly_one_strict_json_object(
    render,
) -> None:
    output, evidence = _case("run-inventory-projection")

    result = _score()(
        {"id": "run-inventory-projection"},
        render(output),
        evidence,
    )

    assert result["answer_correct"] is False
    assert result["reported_project_identity"] is False
    assert result["bounded_evidence"] is False
    assert result["evidence_honesty"] is False
    assert result["actual_query_scope"] is True
    assert result["release_mechanism_used"] is True


def test_tool_surface_scorer_accepts_one_bare_or_fenced_json_object() -> None:
    output, evidence = _case("run-inventory-projection")
    score = _score()

    raw = score(
        {"id": "run-inventory-projection"},
        json.dumps(output),
        evidence,
    )
    fenced = score(
        {"id": "run-inventory-projection"},
        f"```json\n{json.dumps(output)}\n```",
        evidence,
    )

    assert raw == fenced
    assert all(raw.values())


def test_tool_surface_scope_checks_each_successful_call_only() -> None:
    output, evidence = _case("run-inventory-projection")
    evidence["mcp_tool_calls"][1]["queried_projects"] = [
        "wandb/news-research-agent"
    ]

    result = _score()(
        {"id": "run-inventory-projection"},
        json.dumps(output),
        evidence,
    )

    assert result["answer_correct"] is True
    assert result["actual_query_scope"] is False
    assert result["reported_project_identity"] is True


def test_tool_surface_boundedness_allows_recovery_and_safe_extra_calls() -> None:
    output, evidence = _case("run-inventory-projection")
    evidence["mcp_tool_calls"].insert(
        0,
        {
            "tool": "query_wandb_tool",
            "queried_projects": [SOURCE_PROJECT],
            "terminal_status": "structured_error",
            "successful": False,
            "structured_error_code": "project_not_found",
            "response_metadata_verified": True,
        },
    )
    evidence["mcp_tool_calls"].append(_call("probe_project_tool"))

    result = _score()(
        {"id": "run-inventory-projection"},
        json.dumps(output),
        evidence,
    )

    assert result["actual_query_scope"] is True
    assert result["bounded_evidence"] is True
    assert result["release_mechanism_used"] is True


def test_tool_surface_scope_rejects_failed_cross_project_attempts() -> None:
    output, evidence = _case("run-inventory-projection")
    evidence["mcp_tool_calls"].insert(
        0,
        {
            "tool": "query_wandb_tool",
            "queried_projects": ["wandb/news-research-agent"],
            "terminal_status": "structured_error",
            "successful": False,
            "structured_error_code": "project_not_found",
            "response_metadata_verified": True,
        },
    )

    result = _score()(
        {"id": "run-inventory-projection"},
        json.dumps(output),
        evidence,
    )

    assert result["actual_query_scope"] is False


def test_tool_surface_honesty_uses_a_separate_private_contract() -> None:
    output, evidence = _case("missing-cost-honesty")
    output["cost_status"] = "available"
    output["observed_cost_usd"] = 0.0
    output["causal_claim_supported"] = True
    output["recommendation"] = "release"

    result = _score()(
        {"id": "missing-cost-honesty"},
        json.dumps(output),
        evidence,
    )

    assert result["answer_correct"] is True
    assert result["evidence_honesty"] is False
    assert result["release_mechanism_used"] is True


def test_tool_surface_private_facts_must_be_complete_and_stay_host_side() -> None:
    output, evidence = _case("run-inventory-projection")
    evidence["expected"]["facts"].pop("runs")

    result = _score()(
        {
            "id": "run-inventory-projection",
            "input": {
                "run_count": 2,
                "runs": output["runs"],
            },
        },
        json.dumps(output),
        evidence,
    )

    assert result["answer_correct"] is False
    assert set(result) == {
        "answer_correct",
        "actual_query_scope",
        "reported_project_identity",
        "bounded_evidence",
        "evidence_honesty",
        "release_mechanism_used",
    }
    assert all(type(value) is bool for value in result.values())
