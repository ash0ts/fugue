from __future__ import annotations

import json
from pathlib import Path

import pytest

import fugue.research.experiment_views as experiment_views_module
from fugue.bench.candidates import attempt_id
from fugue.bench.library import get_experiment
from fugue.bench.task_presentation import task_presentation_from_public_case
from fugue.research.display_labels import preview_with_governed_display_labels
from fugue.research.experiment_views import (
    EXPERIMENT_VIEW_CELL_LIMIT,
    ExperimentViewV1,
    ExperimentViewV3,
    build_comparison_evaluation_view,
    build_design_view,
    build_evaluation_view,
    build_progress_view,
    experiment_view_from_dict,
)

_A = "a" * 64
_FIXTURE = Path(__file__).parent / "fixtures/experiment-view-v1-design.json"
_V3_STUDY_CONSOLE_GOLDEN = (
    Path(__file__).parent / "fixtures/experiment-view-v3-study-console-golden.json"
)
_REPO_ROOT = Path(__file__).parents[1]


def _current_v3_result_payload() -> dict[str, object]:
    payload = json.loads(_V3_STUDY_CONSOLE_GOLDEN.read_text(encoding="utf-8"))
    presentation = task_presentation_from_public_case(
        task_id="task-1",
        public_case={
            "title": "Reconcile direct Evaluation children",
            "prompt": (
                "Inspect the two named Evaluation roots. Report the exact direct "
                "child counts and explain whether maintainers can trust them."
            ),
            "required_output": "A concise maintainer decision with exact counts.",
            "public_acceptance_criteria": [
                "Report one count for each named Evaluation root.",
                "Separate observed evidence from unsupported causes.",
            ],
            "scenario": "mcp-release-maintenance",
            "tags": ["mcp", "evaluation"],
            "partition": "checkpoint",
        },
    )
    assert presentation is not None
    for arm in ("baseline", "candidate"):
        attempt = payload["paired_cases"][0][arm]
        task_passed = bool(attempt["passed"])
        attempt.update(
            {
                "arm_label": "MCP main" if arm == "baseline" else "MCP 0.4",
                "treatment_summary": (
                    "Use the locked baseline MCP runtime."
                    if arm == "baseline"
                    else "Use the locked staging/0.4 MCP runtime."
                ),
                "task_presentation": presentation.to_dict(),
                "task_result": {
                    "schema_version": 1,
                    "task_passed": task_passed,
                    "outcome_summary": (
                        "The answer met every required check."
                        if task_passed
                        else "The answer did not report the required exact counts."
                    ),
                    "failed_required_checks": (
                        []
                        if task_passed
                        else [
                            {
                                "id": "exact-counts",
                                "label": "Report both exact direct-child counts",
                                "explanation": (
                                    "The answer omitted one required count."
                                ),
                                "critical": True,
                            }
                        ]
                    ),
                    "answer_digest": ("b" if arm == "baseline" else "c") * 64,
                    "agent_execution_status": "completed",
                    "evidence_integrity_status": "verified",
                },
            }
        )
    payload.update(
        {
            "rows": 2,
            "source": "current-result-run",
            "result_digest": "7" * 64,
            "qualification_digest": "8" * 64,
            "incomplete": 0,
            "required_evaluations_incomplete": 0,
            "operational_summary": {
                "execution_states": {"completed": 2},
                "infrastructure_failures": 0,
            },
            "integrity": {
                "status": "reconciled",
                "unresolved_evidence_attempts": 0,
                "harbor_conformance_failed_attempts": 0,
            },
        }
    )
    return payload


def test_local_comparison_rows_use_a_non_clickable_fugue_identity() -> None:
    links = experiment_views_module._comparison_evidence_links(
        (),
        result_ref=None,
        result_source="local-fugue-run-1",
        result_digest="a" * 64,
    )

    assert links == (
        {
            "system": "fugue",
            "kind": "comparison_rows",
            "ref": "local-fugue-run-1",
        },
    )
    assert "uri" not in links[0]


def test_reader_preserves_a_declared_hosted_comparison_rows_link() -> None:
    run_url = "https://wandb.ai/wandb/fugue-results/runs/hosted-run-1"
    links = experiment_views_module._evidence_links(
        (
            {
                "system": "wandb",
                "kind": "comparison_rows",
                "ref": "hosted-run-1",
                "uri": run_url,
            },
        ),
    )

    assert links == (
        {
            "system": "wandb",
            "kind": "comparison_rows",
            "ref": "hosted-run-1",
            "uri": run_url,
        },
    )


def test_wandb_run_label_does_not_promote_local_rows_to_hosted_evidence() -> None:
    run_url = "https://wandb.ai/wandb/fugue-results/runs/unverified-run-1"
    links = experiment_views_module._comparison_evidence_links(
        ({"label": "W&B Run", "url": run_url},),
        result_ref=None,
        result_source="local-fugue-run-1",
        result_digest="a" * 64,
    )

    assert links == (
        {
            "system": "wandb",
            "kind": "w&b_run",
            "ref": run_url,
            "uri": run_url,
        },
        {
            "system": "fugue",
            "kind": "comparison_rows",
            "ref": "local-fugue-run-1",
        },
    )


def test_legacy_local_rows_link_remains_readable_without_a_fake_deep_link() -> None:
    links = experiment_views_module._evidence_links(
        (
            {
                "system": "wandb",
                "kind": "comparison_rows",
                "ref": "legacy-local-run-1",
            },
        )
    )

    assert links == (
        {
            "system": "fugue",
            "kind": "comparison_rows",
            "ref": "legacy-local-run-1",
        },
    )


def test_comparison_rows_reject_a_non_run_wandb_deep_link() -> None:
    with pytest.raises(ValueError, match="canonical W&B Run URL"):
        experiment_views_module._evidence_links(
            (
                {
                    "system": "wandb",
                    "kind": "comparison_rows",
                    "ref": "not-a-run",
                    "uri": (
                        "https://wandb.ai/wandb/fugue-results/weave/calls/not-a-run"
                    ),
                },
            )
        )


def test_comparison_rows_reject_a_wandb_run_id_that_disagrees_with_its_url() -> None:
    with pytest.raises(ValueError, match="Run ID must match"):
        experiment_views_module._evidence_links(
            (
                {
                    "system": "wandb",
                    "kind": "comparison_rows",
                    "ref": "different-run",
                    "uri": "https://wandb.ai/wandb/project/runs/declared-run",
                },
            ),
        )


def test_v3_study_console_wire_golden_is_byte_structure_stable() -> None:
    payload = json.loads(_V3_STUDY_CONSOLE_GOLDEN.read_text(encoding="utf-8"))

    parsed = experiment_view_from_dict(payload)
    normalized = json.loads(json.dumps(parsed.to_dict()))

    assert normalized == payload


def test_v3_study_console_wire_keeps_pre_status_results_readable() -> None:
    payload = json.loads(_V3_STUDY_CONSOLE_GOLDEN.read_text(encoding="utf-8"))
    for arm in ("baseline", "candidate"):
        attempt = payload["paired_cases"][0][arm]
        attempt.pop("cost_reconciliation_status")
        attempt.pop("latency_reconciliation_status")
        attempt.pop("usage_reconciliation_status")

    parsed = experiment_view_from_dict(payload)
    normalized = json.loads(json.dumps(parsed.to_dict()))

    assert normalized == payload


def test_reviewed_legacy_v3_view_keeps_bare_audit_refs_readable() -> None:
    payload = json.loads(_V3_STUDY_CONSOLE_GOLDEN.read_text(encoding="utf-8"))
    payload.update(
        {
            "result_digest": (
                "e062f5b392a36d9ebd97adc3ab58b6e253cdd9dd943381342d51d76303bbcf38"
            ),
            "qualification_digest": (
                "e062f5b392a36d9ebd97adc3ab58b6e253cdd9dd943381342d51d76303bbcf38"
            ),
            "preview_digest": (
                "26e97037dd3b146d610cc5a95fc487df0cb407aeb04f3c1cd01e074b14d5d803"
            ),
        }
    )
    for arm in ("baseline", "candidate"):
        for link in payload["paired_cases"][0][arm]["evidence_links"]:
            link["ref"] = link["ref"].rsplit("/", 1)[-1]

    parsed = experiment_view_from_dict(payload)

    assert parsed.result_digest == payload["result_digest"]
    assert parsed.paired_cases[0]["candidate"]["evidence_links"][0]["ref"] == (
        "candidate-evaluation_root"
    )


def test_current_v3_result_projects_human_readable_attempt_contract() -> None:
    view = build_comparison_evaluation_view(_current_v3_result_payload())

    assert isinstance(view, ExperimentViewV3)
    pair = view.paired_cases[0]
    assert pair["task_label"] == "Bounded answer"
    assert pair["baseline"]["arm_label"] == "MCP main"
    assert pair["candidate"]["arm_label"] == "MCP 0.4"
    assert pair["candidate"]["treatment_summary"] == (
        "Use the locked staging/0.4 MCP runtime."
    )
    assert pair["candidate"]["task_presentation"]["title"] == (
        "Reconcile direct Evaluation children"
    )
    assert pair["candidate"]["task_presentation"]["public_prompt"] == [
        {
            "order": 1,
            "text": (
                "Inspect the two named Evaluation roots. Report the exact direct "
                "child counts and explain whether maintainers can trust them."
            ),
        }
    ]
    assert pair["baseline"]["task_result"]["task_passed"] is False
    assert pair["baseline"]["task_result"]["failed_required_checks"] == [
        {
            "id": "exact-counts",
            "label": "Report both exact direct-child counts",
            "explanation": "The answer omitted one required count.",
            "critical": True,
        }
    ]
    assert pair["candidate"]["task_result"]["task_passed"] is True
    assert pair["candidate"]["task_result"]["failed_required_checks"] == []
    reparsed = experiment_view_from_dict(view.to_dict())
    assert reparsed.to_dict() == view.to_dict()


def test_v3_result_accepts_timed_out_as_terminal_behavioral_evidence() -> None:
    payload = _current_v3_result_payload()
    baseline = payload["paired_cases"][0]["baseline"]
    baseline["execution_status"] = "timed_out"
    baseline["task_result"]["agent_execution_status"] = "timed_out"
    payload["operational_summary"]["execution_states"] = {
        "completed": 1,
        "timed_out": 1,
    }

    view = build_comparison_evaluation_view(payload)

    assert isinstance(view, ExperimentViewV3)
    assert view.completed_cells == 2
    assert view.state_counts == {"completed": 1, "timed_out": 1}
    pair = view.paired_cases[0]
    assert pair["baseline"]["execution_status"] == "timed_out"
    assert pair["baseline"]["task_result"]["task_passed"] is False
    assert pair["dimension_changes"][0]["status"] == "improved"


def test_v3_not_started_pair_is_terminal_but_has_no_behavioral_claim() -> None:
    payload = _current_v3_result_payload()
    payload["incomplete"] = 1
    payload["operational_summary"] = {
        "execution_states": {"not_started": 2},
        "infrastructure_failures": 2,
    }
    payload["behavioral_summary"].update(
        {
            "status": "incomplete",
            "recommendation": "INCOMPLETE — repair execution before comparison.",
            "improved_pairs": 0,
            "regressed_pairs": 0,
            "mixed_pairs": 0,
            "unchanged_pairs": 0,
            "incomplete_pairs": 1,
            "candidate_critical_failures": 0,
            "critical_blockers": ["task-1: Agent execution did not start."],
            "supported_claim": None,
            "next_action": "Repair execution and resume the same logical cells.",
        }
    )
    payload["judge_summary"] = {
        "status": "not_used",
        "claim_status": "not_applicable",
        "judges": [],
        "by_variant": {"baseline": {}, "candidate": {}},
        "unavailable_attempts": 0,
    }
    pair = payload["paired_cases"][0]
    pair["status"] = "incomplete"
    pair["dimension_changes"] = []
    for arm in ("baseline", "candidate"):
        attempt = pair[arm]
        attempt["execution_status"] = "not_started"
        attempt["evaluation_status"] = "unavailable"
        attempt["cost_reconciliation_status"] = "unavailable"
        attempt["latency_reconciliation_status"] = "unavailable"
        attempt["usage_reconciliation_status"] = "unavailable"
        attempt["tool_calls"] = 0
        attempt["tools"] = []
        attempt["queried_projects"] = []
        attempt["actual_query_scope"] = []
        for field_name in (
            "passed",
            "cost_usd",
            "latency_sec",
            "input_tokens",
            "output_tokens",
            "scores",
            "score_explanations",
            "sanitized_answer_excerpt",
            "reported_project_identity",
            "task_result",
        ):
            attempt.pop(field_name, None)

    view = build_comparison_evaluation_view(payload)

    assert isinstance(view, ExperimentViewV3)
    assert view.completed_cells == 2
    assert view.state_counts == {"not_started": 2}
    assert view.evidence_eligible is False
    assert view.judge_summary["status"] == "not_used"
    projected_pair = view.paired_cases[0]
    assert projected_pair["status"] == "incomplete"
    assert projected_pair["dimension_changes"] == ()
    for arm in ("baseline", "candidate"):
        attempt = projected_pair[arm]
        assert attempt["execution_status"] == "not_started"
        assert attempt["arm_label"]
        assert attempt["treatment_summary"]
        assert attempt["task_presentation"]["task_id"] == "task-1"
        assert "passed" not in attempt
        assert attempt["scores"] == {}
        assert "task_result" not in attempt


def test_v3_nonbehavioral_pair_rejects_behavioral_dimensions() -> None:
    payload = _current_v3_result_payload()
    pair = payload["paired_cases"][0]
    pair["status"] = "incomplete"
    pair["baseline"]["execution_status"] = "not_started"
    for field_name in (
        "passed",
        "scores",
        "score_explanations",
        "sanitized_answer_excerpt",
        "reported_project_identity",
        "task_result",
    ):
        pair["baseline"].pop(field_name, None)

    with pytest.raises(
        ValueError,
        match="incomplete nonbehavioral pair cannot contain dimensions",
    ):
        build_comparison_evaluation_view(payload)


def test_v3_fatal_integrity_terminal_overrides_completed_behavior() -> None:
    payload = _current_v3_result_payload()
    pair = payload["paired_cases"][0]
    baseline = pair["baseline"]
    baseline["infrastructure"]["terminal_kind"] = "evidence_failure"

    with pytest.raises(
        ValueError,
        match="nonbehavioral attempt cannot contain task results",
    ):
        build_comparison_evaluation_view(payload)

    for field_name in (
        "passed",
        "scores",
        "score_explanations",
        "score_details",
        "judge_reviews",
        "sanitized_answer_excerpt",
        "reported_project_identity",
        "task_result",
    ):
        baseline.pop(field_name, None)
    pair["status"] = "incomplete"

    with pytest.raises(
        ValueError,
        match="incomplete nonbehavioral pair cannot contain dimensions",
    ):
        build_comparison_evaluation_view(payload)


def test_v3_study_console_wire_rejects_unknown_reconciliation_status() -> None:
    payload = json.loads(_V3_STUDY_CONSOLE_GOLDEN.read_text(encoding="utf-8"))
    payload["paired_cases"][0]["baseline"]["cost_reconciliation_status"] = "available"

    with pytest.raises(ValueError, match="must be one of"):
        experiment_view_from_dict(payload)


@pytest.mark.parametrize(
    ("status_field", "measurement_fields", "message"),
    [
        (
            "cost_reconciliation_status",
            ("cost_usd",),
            "resolved cost reconciliation requires cost_usd",
        ),
        (
            "latency_reconciliation_status",
            ("latency_sec",),
            "resolved latency reconciliation requires latency_sec",
        ),
        (
            "usage_reconciliation_status",
            ("input_tokens", "output_tokens"),
            "resolved usage reconciliation requires input and output tokens",
        ),
    ],
)
def test_v3_study_console_wire_requires_measurements_for_resolved_status(
    status_field: str,
    measurement_fields: tuple[str, ...],
    message: str,
) -> None:
    payload = json.loads(_V3_STUDY_CONSOLE_GOLDEN.read_text(encoding="utf-8"))
    attempt = payload["paired_cases"][0]["candidate"]
    attempt[status_field] = "resolved"
    for field_name in measurement_fields:
        attempt.pop(field_name)

    with pytest.raises(ValueError, match=message):
        experiment_view_from_dict(payload)


def test_v3_judge_summary_is_safe_and_invalid_integrity_suppresses_it() -> None:
    scored = {
        "status": "scored",
        "claim_status": "advisory_uncalibrated",
        "judges": [
            {
                "judge_id": "maintainer-actionability",
                "profile": "wandb/zai-org/GLM-5.2",
                "contract_digest": "1" * 64,
                "dimensions": ["bounded_next_step"],
                "calibration": {
                    "status": "pending_human_review",
                    "report_sha256": "2" * 64,
                    "cases_digest": "3" * 64,
                    "passed": False,
                },
            }
        ],
        "by_variant": {
            "baseline": {
                "maintainer-actionability.bounded_next_step": {
                    "evaluated": 2,
                    "mean": 0.4,
                }
            },
            "candidate": {
                "maintainer-actionability.bounded_next_step": {
                    "evaluated": 2,
                    "mean": 0.8,
                }
            },
        },
        "unavailable_attempts": 0,
    }

    assert (
        experiment_views_module._safe_judge_summary(
            scored,
            integrity_status="reconciled",
            attempts=4,
        )
        == scored
    )
    assert experiment_views_module._safe_judge_summary(
        scored,
        integrity_status="invalid",
        attempts=4,
    ) == {
        **scored,
        "status": "unavailable",
        "by_variant": {"baseline": {}, "candidate": {}},
        "unavailable_attempts": 4,
    }


def test_v3_judge_summary_rejects_forged_qualification_and_dimensions() -> None:
    scored = {
        "status": "scored",
        "claim_status": "advisory_uncalibrated",
        "judges": [
            {
                "judge_id": "maintainer-actionability",
                "profile": "wandb/zai-org/GLM-5.2",
                "contract_digest": "1" * 64,
                "dimensions": ["bounded_next_step"],
                "calibration": {
                    "status": "pending_human_review",
                    "report_sha256": "2" * 64,
                    "cases_digest": "3" * 64,
                    "passed": False,
                },
            }
        ],
        "by_variant": {
            "baseline": {
                "other-judge.bounded_next_step": {
                    "evaluated": 1,
                    "mean": 0.5,
                }
            },
            "candidate": {
                "other-judge.bounded_next_step": {
                    "evaluated": 1,
                    "mean": 0.5,
                }
            },
        },
        "unavailable_attempts": 0,
    }
    with pytest.raises(ValueError, match="lacks matching provenance"):
        experiment_views_module._safe_judge_summary(
            scored,
            integrity_status="reconciled",
            attempts=2,
        )

    scored["by_variant"] = {"baseline": {}, "candidate": {}}
    scored["claim_status"] = "calibrated"
    with pytest.raises(ValueError, match="calibration provenance"):
        experiment_views_module._safe_judge_summary(
            scored,
            integrity_status="reconciled",
            attempts=2,
        )


def test_v3_judge_summary_rejects_impossible_evaluated_counts() -> None:
    scored = {
        "status": "scored",
        "claim_status": "advisory_uncalibrated",
        "judges": [
            {
                "judge_id": "maintainer-actionability",
                "profile": "wandb/zai-org/GLM-5.2",
                "contract_digest": "1" * 64,
                "dimensions": ["bounded_next_step"],
                "calibration": {
                    "status": "pending_human_review",
                    "report_sha256": "2" * 64,
                    "cases_digest": "3" * 64,
                    "passed": False,
                },
            }
        ],
        "by_variant": {
            arm: {
                "maintainer-actionability.bounded_next_step": {
                    "evaluated": 999,
                    "mean": 0.5,
                }
            }
            for arm in ("baseline", "candidate")
        },
        "unavailable_attempts": 0,
    }

    with pytest.raises(ValueError, match="exceeds canonical arm attempts"):
        experiment_views_module._safe_judge_summary(
            scored,
            integrity_status="reconciled",
            attempts=4,
            arm_attempts={"baseline": 2, "candidate": 2},
        )

    unavailable = {
        **scored,
        "status": "unavailable",
        "by_variant": {"baseline": {}, "candidate": {}},
        "unavailable_attempts": 0,
    }
    with pytest.raises(ValueError, match="unavailable counts do not reconcile"):
        experiment_views_module._safe_judge_summary(
            unavailable,
            integrity_status="reconciled",
            attempts=4,
            arm_attempts={"baseline": 2, "candidate": 2},
        )


def test_judge_completeness_counts_attempts_once_across_dimensions() -> None:
    result = {
        "rows": 4,
        "baseline_passed": 1,
        "candidate_passed": 2,
        "judge_summary": {
            "status": "scored",
            "by_variant": {
                "baseline": {
                    "judge.actionability": {"evaluated": 1, "mean": 0.5},
                    "judge.grounding": {"evaluated": 1, "mean": 0.6},
                },
                "candidate": {
                    "judge.actionability": {"evaluated": 2, "mean": 0.7},
                    "judge.grounding": {"evaluated": 2, "mean": 0.8},
                },
            },
            "unavailable_attempts": 1,
        },
    }

    summaries = experiment_views_module._comparison_outcome_summaries(
        result,
        baseline_total=2,
        candidate_total=2,
        infrastructure_failures=0,
        missing_evidence=0,
    )
    judge = next(item for item in summaries if item.id == "judge-evidence")

    assert judge.status == "failed"
    assert judge.passed == 3
    assert judge.total == 4
    assert judge.unavailable == 1


def test_v3_view_rejects_published_judge_rationale() -> None:
    payload = json.loads(_V3_STUDY_CONSOLE_GOLDEN.read_text(encoding="utf-8"))
    key = "comparison.judge.maintainer-actionability.maintenance_actionability"
    payload["paired_cases"][0]["candidate"]["score_explanations"][key] = (
        "Private expected values made this look correct."
    )

    with pytest.raises(ValueError, match="must not publish rationale"):
        experiment_view_from_dict(payload)


def test_v3_view_accepts_safe_structured_score_details() -> None:
    payload = json.loads(_V3_STUDY_CONSOLE_GOLDEN.read_text(encoding="utf-8"))
    dimension = "bounded_answer"
    candidate = payload["paired_cases"][0]["candidate"]
    candidate["score_details"] = {
        dimension: {
            "what": "Checks whether the final answer is factually correct.",
            "observed": "The host scorer matched the required facts.",
            "why": "The outcome check passed.",
        }
    }

    view = experiment_view_from_dict(payload)

    assert view.paired_cases[0]["candidate"]["score_details"][dimension]["what"] == (
        "Checks whether the final answer is factually correct."
    )


def test_v3_view_rejects_sensitive_structured_score_details() -> None:
    payload = json.loads(_V3_STUDY_CONSOLE_GOLDEN.read_text(encoding="utf-8"))
    dimension = "bounded_answer"
    payload["paired_cases"][0]["candidate"]["score_details"] = {
        dimension: {
            "what": "Checks whether the final answer is bounded.",
            "observed": "The tool used a finite limit.",
            "why": "api_key=sk-example-secret-value",
        }
    }

    with pytest.raises(ValueError, match="score detail .* is sensitive"):
        experiment_view_from_dict(payload)


def test_v3_view_accepts_safe_anchored_judge_review() -> None:
    payload = json.loads(_V3_STUDY_CONSOLE_GOLDEN.read_text(encoding="utf-8"))
    candidate = payload["paired_cases"][0]["candidate"]
    candidate["judge_reviews"] = {
        "maintainer-actionability": {
            "label": "strong",
            "reason": (
                "The answer gives a concrete next action and states its evidence "
                "limit."
            ),
            "missing_evidence": False,
            "observed_cost_usd": 0.04,
            "cost_status": "observed",
        }
    }

    view = experiment_view_from_dict(payload)

    assert view.paired_cases[0]["candidate"]["judge_reviews"] == {
        "maintainer-actionability": {
            "label": "strong",
            "reason": (
                "The answer gives a concrete next action and states its evidence "
                "limit."
            ),
            "missing_evidence": False,
            "observed_cost_usd": 0.04,
            "cost_status": "observed",
        }
    }


def test_v3_view_rejects_unanchored_or_sensitive_judge_review() -> None:
    payload = json.loads(_V3_STUDY_CONSOLE_GOLDEN.read_text(encoding="utf-8"))
    candidate = payload["paired_cases"][0]["candidate"]
    candidate["judge_reviews"] = {
        "maintainer-actionability": {
            "label": "0.83",
            "reason": "api_key=sk-example-secret-value",
            "missing_evidence": False,
        }
    }

    with pytest.raises(ValueError, match="label is unsupported"):
        experiment_view_from_dict(payload)

    candidate["judge_reviews"]["maintainer-actionability"]["label"] = "strong"
    with pytest.raises(ValueError, match="reason is sensitive"):
        experiment_view_from_dict(payload)


def test_v3_view_rejects_attempt_identity_mismatch() -> None:
    payload = json.loads(_V3_STUDY_CONSOLE_GOLDEN.read_text(encoding="utf-8"))
    payload["paired_cases"][0]["candidate"]["identity"]["candidate"] = (
        "forged-candidate"
    )

    with pytest.raises(ValueError, match="identity is not canonical"):
        experiment_view_from_dict(payload)


def test_v3_pair_coordinates_match_attempt_identity() -> None:
    payload = json.loads(_V3_STUDY_CONSOLE_GOLDEN.read_text(encoding="utf-8"))
    candidate = payload["paired_cases"][0]["candidate"]
    candidate["identity"]["arm"] = "baseline"
    candidate["attempt_id"] = attempt_id(**candidate["identity"])

    with pytest.raises(ValueError, match="pair coordinates"):
        experiment_view_from_dict(payload)


def test_v3_evidence_eligible_requires_five_resolved_links() -> None:
    payload = json.loads(_V3_STUDY_CONSOLE_GOLDEN.read_text(encoding="utf-8"))
    link = payload["paired_cases"][0]["candidate"]["evidence_links"][0]
    link["status"] = "missing"
    link["reason"] = "The object could not be resolved."
    link.pop("ref")
    link.pop("url")

    with pytest.raises(ValueError, match="five resolved links"):
        experiment_view_from_dict(payload)


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


def test_design_projects_declared_factorial_treatment_arms() -> None:
    preview = _preview()
    research_view = preview["draft"]["research_view"]
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

    view = build_design_view(preview)

    assert {factor.name: factor.levels for factor in view.varied_factors} == {
        "harness": ("codex", "claude-code"),
        "variant": ("baseline", "warning-only", "action-gate"),
    }
    assert [(arm.id, arm.label, arm.factor_levels) for arm in view.treatment_arms] == [
        (
            "baseline",
            "Current behavior",
            {"repository-search": "off", "source-inspection": "standard"},
        ),
        (
            "warning-only",
            "Add a reminder",
            {"repository-search": "off", "source-inspection": "required"},
        ),
        (
            "action-gate",
            "Check risky actions",
            {"repository-search": "on", "source-inspection": "required"},
        ),
    ]


def test_design_projects_factorial_arm_levels_into_planned_cells() -> None:
    preview = _preview()
    preview["draft"]["varied_dimensions"] = [
        "harness",
        "repository search",
        "source-inspection requirement",
    ]
    preview["draft"]["research_view"]["arm_factor_levels"] = {
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

    view = build_design_view(preview)

    assert {factor.name: factor.levels for factor in view.varied_factors} == {
        "harness": ("codex", "claude-code"),
        "repository search": ("off", "on"),
        "source-inspection requirement": ("standard", "required"),
    }
    assert {
        (
            cell.factor_levels["repository search"],
            cell.factor_levels["source-inspection requirement"],
        )
        for cell in view.cells
        if cell.factor_levels["harness"] == "codex"
    } == {("off", "standard"), ("off", "required"), ("on", "required")}


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
    assert view.schema_version == 1
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


def test_evaluation_never_turns_otel_identities_into_weave_links() -> None:
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

    forbidden = {
        "agent_conversation",
        "conversation_identity",
        "invoke_agent_root",
        "trace",
        "evaluation_attempt",
    }
    assert all(
        forbidden.isdisjoint({link["kind"] for link in cell.evidence_links})
        for cell in view.cells
    )
    assert all(cell.evidence_status == "reconciled" for cell in view.cells)
    serialized = json.dumps(view.to_dict())
    assert "agent_response" not in serialized
    assert "tool_output" not in serialized


def test_campaign_evaluation_uses_exact_attempt_and_five_verified_links() -> None:
    identity = {
        "task_id": "maintenance-task",
        "arm": "memory-policy",
        "harness": "claude-code",
        "attempt": 1,
        "candidate": "candidate-locked",
        "runtime": "runtime-locked",
    }
    stable_attempt = attempt_id(**identity)
    slots = (
        ("prediction_and_score", "evaluation_attempt"),
        ("prediction", "prediction"),
        ("evaluation_root", "evaluation"),
        ("agent_root", "agent_root"),
        ("dataset", "dataset"),
    )
    links = [
        {
            "slot": slot,
            "system": "weave",
            "kind": kind,
            "ref": (
                "weave:///wandb/project/object/dataset:v1"
                if slot == "dataset"
                else f"weave:///wandb/project/call/{slot}"
            ),
            "uri": (
                "https://wandb.ai/wandb/project/weave/objects/dataset/versions/v1"
                if slot == "dataset"
                else f"https://wandb.ai/wandb/project/weave/calls/{slot}"
            ),
            "verification_status": "verified",
        }
        for slot, kind in slots
    ]
    row = {
        "prediction_id": "prediction-1",
        "attempt_id": stable_attempt,
        "attempt_identity": identity,
        "run_id": "run-1",
        "candidate_id": identity["candidate"],
        "execution_fingerprint": identity["runtime"],
        "comparison_example_id": identity["task_id"],
        "task_id": identity["task_id"],
        "task_name": "Maintenance task",
        "variant_id": identity["arm"],
        "harness": identity["harness"],
        "trial_index": 1,
        "execution_kind": "agent",
        "status": "completed",
        "pass": True,
        "trace_project": "wandb/project",
        "trace_link_status": "linked",
        "evidence_links": links,
        "skill_ids_invoked": ["evidence-policy"],
        "integration_ids_invoked": ["wandb-mcp-0-4"],
        "mcp_tool_names": ["query_wandb_tool"],
        "mcp_queried_projects": ["wandb/source"],
        "otel_root_span_ids": ["0123456789abcdef"],
        "otel_trace_ids": ["0123456789abcdef0123456789abcdef"],
    }
    record = {
        "preview": {"estimated_cells": 1, "draft": {}},
        "approval": {"approved": True},
        "outcome": {
            "row_refs": [row],
            "evidence_refs": [
                {
                    "prediction_id": "prediction-1",
                    "attempt_id": stable_attempt,
                    "links": links,
                }
            ],
            "expected_predictions": 1,
            "observed_predictions": 1,
            "observed_cost_usd": 0.1,
            "eligible": True,
            "run_status": "passed",
        },
    }

    view = build_evaluation_view(record)

    [cell] = view.cells
    assert cell.cell_id == stable_attempt
    assert cell.evidence_status == "reconciled"
    weave_links = [link for link in cell.evidence_links if link["system"] == "weave"]
    assert len(weave_links) == 5
    assert weave_links[0]["kind"] == "evaluation_attempt"
    assert cell.measures["skill_ids_invoked"] == "evidence-policy"
    assert cell.measures["integration_ids_invoked"] == "wandb-mcp-0-4"
    assert cell.measures["mcp_tool_names"] == "query_wandb_tool"
    assert cell.measures["mcp_queried_projects"] == "wandb/source"
    serialized = json.dumps(cell.to_dict())
    assert "0123456789abcdef" not in serialized


def test_evaluation_projects_one_explicit_evidence_workspace() -> None:
    raw = _record()
    for index, row in enumerate(raw["outcome"]["row_refs"]):
        row.update(
            {
                "trace_project": "ashah-weights-biases/loop-engineering-demo",
                "weave_prediction_call_id": f"prediction-call-{index}",
                "eval_predict_and_score_call_id": f"evaluation-call-{index}",
            }
        )

    view = build_evaluation_view(raw)

    assert view.evidence_scope is not None
    assert view.evidence_scope.entity == "ashah-weights-biases"
    assert view.evidence_scope.project == "loop-engineering-demo"
    assert set(view.evidence_scope.evidence_types) == {
        "agent_conversation",
        "evaluation_attempt",
        "prediction_and_score",
        "source_call",
    }


def test_evaluation_does_not_merge_multiple_evidence_workspaces() -> None:
    raw = _record()
    raw["preview"]["evidence_scope"] = {
        "entity": "ashah-weights-biases",
        "project": "loop-engineering-demo",
        "evidence_types": ["agent_conversation"],
    }
    for index, row in enumerate(raw["outcome"]["row_refs"]):
        row["trace_project"] = (
            "ashah-weights-biases/loop-engineering-demo"
            if index < 5
            else "wandb/fugue-experiments"
        )

    view = build_evaluation_view(raw)

    # A mixed row cohort is not relabelled as one workspace. The immutable
    # preview remains the declared destination for navigation.
    assert view.evidence_scope is not None
    assert view.evidence_scope.entity == "ashah-weights-biases"
    assert view.evidence_scope.project == "loop-engineering-demo"


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
            "candidate_id": "candidate-0",
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
    cell_evaluation = next(
        link
        for link in view.cells[0].evidence_links
        if link["system"] == "weave" and link["kind"] == "evaluation"
    )
    assert cell_evaluation["ref"] == "weave:///team/project/object/evaluation:v1"
    assert any(link["kind"] == "dataset" for link in view.cells[0].evidence_links)


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


def test_progress_counts_timed_out_and_not_started_as_distinct_terminal_cells() -> None:
    progress = build_progress_view(
        {
            "state": "running",
            "approval": {"approval_digest": _A},
            "preview": {**_preview(), "estimated_cells": 3},
        },
        {
            "status": "running",
            "cells": [
                {
                    "cell_id": "not-started-cell",
                    "candidate_id": "baseline-candidate",
                    "status": "not_started",
                    "harness": "claude-code",
                    "variant_id": "baseline",
                    "task_id": "task-1",
                    "benchmark_outcome": "unscored",
                },
                {
                    "cell_id": "timed-out-cell",
                    "candidate_id": "candidate-candidate",
                    "status": "timed_out",
                    "harness": "claude-code",
                    "variant_id": "candidate",
                    "task_id": "task-1",
                    "benchmark_outcome": "failed",
                },
                {
                    "cell_id": "running-cell",
                    "candidate_id": "candidate-candidate",
                    "status": "running",
                    "harness": "claude-code",
                    "variant_id": "candidate",
                    "task_id": "task-2",
                    "benchmark_outcome": "unscored",
                },
            ],
        },
    )

    assert progress.completed_cells == 2
    assert progress.state_counts["execution:not_started"] == 1
    assert progress.state_counts["execution:timed_out"] == 1
    assert progress.state_counts["execution:running"] == 1
    cells = {cell.execution_status: cell for cell in progress.cells}
    assert cells["not_started"].task_outcome == "unavailable"
    assert cells["not_started"].reason_code == "execution_not_started"
    assert cells["timed_out"].task_outcome == "failed"
    assert cells["timed_out"].reason_code == "task_not_passed"


def test_experiment_view_union_rejects_unknown_nested_fields() -> None:
    raw = build_design_view(_preview()).to_dict()
    raw["taskset"]["prompt"] = "do not publish"
    try:
        experiment_view_from_dict(raw)
    except ValueError as exc:
        assert "unknown fields" in str(exc)
    else:
        raise AssertionError("unknown experiment-view fields must be rejected")


def test_provisional_v2_evaluation_records_remain_replayable() -> None:
    view = experiment_view_from_dict(
        {
            "schema_version": 2,
            "kind": "evaluation",
            "matrix_size": 8,
            "completed_cells": 8,
            "cell_limit": 8,
            "cells": [],
            "omitted_cells": 8,
            "arm_totals": [],
            "aligned_comparisons": [],
            "mechanism_funnel": [],
            "outcome_summaries": [],
            "score_summaries": [],
            "infrastructure_health": "mechanism_only",
            "evidence_eligible": False,
            "integrity_status": "invalid",
            "evidence_grade": "invalid",
            "limitations": ["Historical projection; superseded by strict V2."],
        }
    )

    assert isinstance(view, ExperimentViewV1)
    assert view.schema_version == 2
    assert view.matrix_size == 8
    assert view.omitted_cells == 8


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
