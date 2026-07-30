from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from fugue.bench.candidates import attempt_id, attempt_identity, stable_digest
from fugue.bench.comparison import (
    COMPARISON_RUNTIME_ROOT,
    ComparisonEvaluatorV1,
    ComparisonResultV3,
    ComparisonSpecV1,
    DecisionAttestationV1,
    _apply_decision_attestation,
    _comparison_qualification_digest,
    _comparison_result_digest,
    _comparison_trial_output,
    _evaluate_decision,
    _paired_attempt_view_v3,
    _require_checkpoint_judges,
    _sanitized_answer_excerpt,
    analyze_comparison_rows,
    check_comparison,
    claim_comparison_approval,
    comparison_from_dict,
    compile_comparison,
    execute_comparison,
    load_comparison,
    materialize_comparison,
    prepare_comparison,
    preview_comparison,
    read_comparison_result,
    scaffold_comparison,
    score_comparison_rows,
    write_comparison_result,
)
from fugue.bench.comparison import _decision_policy as parse_decision_policy
from fugue.bench.operator import OperatorService
from fugue.model_plane import trace_destination_identity
from fugue.research.approvals import ApprovalLedger
from fugue.research.experiment_views import (
    ExperimentViewV2,
    ExperimentViewV3,
    build_comparison_design_view,
    build_comparison_evaluation_view,
    build_comparison_progress_view,
    experiment_view_from_dict,
)
from fugue.research.store import StudyStore

EXAMPLE = Path("examples/comparisons/source-use-replay")
LIVE_SKILL_EXAMPLE = Path("examples/comparisons/source-use-skill")
MCP_MAINTENANCE_EXAMPLE = Path(
    "examples/comparisons/wandb-mcp-maintenance"
)


def test_source_use_comparison_is_ready_and_exact() -> None:
    root = Path.cwd()
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=root)

    readiness = check_comparison(spec, repo_root=root)
    preview = preview_comparison(spec, repo_root=root)
    reparsed = comparison_from_dict(preview.comparison, repo_root=root)

    assert readiness.status == "ready"
    assert readiness.task_count == 4
    assert readiness.base_failures == 4
    assert readiness.gold_passes == 4
    assert readiness.actual_changes == ("skills",)
    assert readiness.estimated_cells == 16
    assert readiness.estimated_cost_usd == 0
    assert preview.matrix["estimated_trials"] == 16
    assert preview.matrix["applicable_cells"] == 16
    assert reparsed == spec
    assert isinstance(preview.comparison["evaluators"], list)
    assert isinstance(preview.comparison["execution"]["harnesses"], list)
    assert not (root / COMPARISON_RUNTIME_ROOT / spec.spec_digest).exists()


def test_exact_approval_may_acknowledge_reviewable_canary(
    tmp_path: Path,
) -> None:
    root = Path.cwd()
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=root)
    canary = replace(spec, execution=replace(spec.execution, attempts=1))
    preview = preview_comparison(canary, repo_root=root)
    assert preview.readiness["status"] == "needs_review"

    ledger = ApprovalLedger(StudyStore(tmp_path).path)
    approval = ledger.approve(
        subject_kind="experiment",
        preview_digest=preview.preview_digest,
        maximum_cost_usd=1,
        maximum_cells=8,
        approved_by="test-reviewer",
        operation_id="approve-reviewable-canary",
    )

    claim_comparison_approval(
        preview,
        approval_digest=approval.approval_digest,
        repo_root=tmp_path,
    )


def test_materialization_reuses_preview_operator_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path.cwd()
    operator = OperatorService(root)
    preview = preview_comparison(
        load_comparison(EXAMPLE / "comparison.yaml", repo_root=root),
        repo_root=root,
        operator=operator,
    )

    def observe_operator(*args: object, **kwargs: object) -> object:
        assert kwargs["operator"] is operator
        raise RuntimeError("operator observed")

    monkeypatch.setattr(
        "fugue.bench.comparison.preview_comparison",
        observe_operator,
    )
    with pytest.raises(RuntimeError, match="operator observed"):
        materialize_comparison(preview, repo_root=root, operator=operator)


def test_live_source_use_skill_comparison_has_locked_holdout_resources() -> None:
    root = Path.cwd()
    spec = load_comparison(
        LIVE_SKILL_EXAMPLE / "comparison.yaml", repo_root=root
    )

    readiness = check_comparison(spec, repo_root=root)
    preview = preview_comparison(spec, repo_root=root)

    assert readiness.status == "ready"
    assert readiness.task_count == 8
    assert readiness.base_failures == 8
    assert readiness.gold_passes == 8
    assert readiness.actual_changes == ("skills",)
    assert readiness.estimated_cells == 32
    assert preview.matrix["estimated_trials"] == 32
    assert {cell["harness"] for cell in preview.matrix["matrix_cells"]} == {
        "codex"
    }


@pytest.mark.parametrize(
    ("filename", "tasks", "harnesses", "attempts", "cells"),
    [
        ("discovery.yaml", 4, ("claude-code",), 1, 8),
        ("discovery-wandb.yaml", 4, ("openclaw",), 1, 8),
        ("primary.yaml", 8, ("claude-code",), 2, 32),
        ("wandb-replication.yaml", 8, ("openclaw",), 2, 32),
    ],
)
def test_mcp_maintenance_examples_have_exact_staged_designs(
    filename: str,
    tasks: int,
    harnesses: tuple[str, ...],
    attempts: int,
    cells: int,
) -> None:
    root = Path.cwd()
    spec = load_comparison(
        MCP_MAINTENANCE_EXAMPLE / filename,
        repo_root=root,
    )
    readiness = check_comparison(spec, repo_root=root)

    assert readiness.task_count == tasks
    assert spec.execution.harnesses == harnesses
    assert spec.execution.attempts == attempts
    assert readiness.estimated_cells == cells
    assert readiness.actual_changes == ("integrations",)
    assert readiness.status == "blocked"
    assert any("not adjudicated" in item for item in readiness.blockers)


def test_evidence_checkpoint_requires_serial_execution() -> None:
    root = Path.cwd()
    raw = yaml.safe_load((EXAMPLE / "comparison.yaml").read_text())
    raw["execution"]["evidence_checkpoint_cells"] = 1
    raw["execution"]["concurrency"] = 2

    with pytest.raises(
        ValueError,
        match="evidence checkpoint cells require comparison concurrency 1",
    ):
        comparison_from_dict(raw, repo_root=root, source=EXAMPLE)


def test_comparison_rejects_unknown_fields_and_undeclared_changes() -> None:
    root = Path.cwd()
    raw = yaml.safe_load((EXAMPLE / "comparison.yaml").read_text())
    raw["surprise"] = True
    with pytest.raises(ValueError, match="unknown comparison field"):
        comparison_from_dict(raw, repo_root=root, source=EXAMPLE)

    raw.pop("surprise")
    raw["changed"] = ["prompt_id"]
    spec = comparison_from_dict(raw, repo_root=root, source=EXAMPLE)
    readiness = check_comparison(spec, repo_root=root)
    assert readiness.status == "blocked"
    assert any("resolved behavior diff" in item for item in readiness.blockers)


def test_comparison_keeps_public_tasks_and_private_labels_separate(
    tmp_path: Path,
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    labels = tmp_path / "labels.jsonl"
    tasks.write_text(
        json.dumps(
            {
                "id": "leak",
                "input": {"question": "question"},
                "expected": {"answer": 1},
            }
        )
        + "\n"
    )
    labels.write_text(
        json.dumps(
            {
                "id": "leak",
                "expected": {"answer": 1},
                "base_output": {"answer": 0},
                "gold_output": {"answer": 1},
            }
        )
        + "\n"
    )
    raw = yaml.safe_load((EXAMPLE / "comparison.yaml").read_text())
    raw["taskset"] = {
        "tasks": tasks.relative_to(Path.cwd()).as_posix()
        if tasks.is_relative_to(Path.cwd())
        else tasks.as_posix(),
        "private_labels": labels.relative_to(Path.cwd()).as_posix()
        if labels.is_relative_to(Path.cwd())
        else labels.as_posix(),
    }
    spec = comparison_from_dict(raw, repo_root=tmp_path, source=tmp_path)
    with pytest.raises(ValueError, match="unknown public task .*field"):
        check_comparison(spec, repo_root=tmp_path)


def test_replay_scores_aligned_improvements_and_regressions() -> None:
    root = Path.cwd()
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=root)
    preview = preview_comparison(spec, repo_root=root)
    rows = [
        json.loads(line)
        for line in (EXAMPLE / "attempts.jsonl").read_text().splitlines()
    ]
    scored = score_comparison_rows(spec, rows, repo_root=root)
    result = analyze_comparison_rows(
        comparison_id=spec.id,
        preview_digest=preview.preview_digest,
        rows=scored,
        source="test-replay",
    )

    assert result.rows == 16
    assert result.baseline_passed == 2
    assert result.candidate_passed == 6
    assert result.improved == 5
    assert result.regressed == 1
    assert result.unchanged == 2
    assert result.incomplete == 0
    assert result.evidence_project is None
    assert result.judge_summary == {
        "status": "not_used",
        "claim_status": "not_applicable",
        "judges": [],
        "by_variant": {"baseline": {}, "candidate": {}},
        "unavailable_attempts": 0,
    }
    assert result.deterministic_summary["candidate"]["passed"] == 6
    assert result.operational_summary == {
        "execution_states": {"unknown": 16},
        "evidence_states": {"unknown": 16},
        "infrastructure_failures": 0,
        "observed_cost_usd": None,
        "cost_rows": 0,
        "latency_ms": None,
        "latency_rows": 0,
        "input_tokens": None,
        "output_tokens": None,
        "usage_rows": 0,
        "evidence_projects": [],
        "mcp_tool_usage": {},
    }


def test_comparison_evidence_project_is_locked_into_readiness_and_preview() -> None:
    root = Path.cwd()
    raw = yaml.safe_load((EXAMPLE / "comparison.yaml").read_text())
    raw["execution"]["evidence_project"] = "wandb/fugue-comparison-a"
    first = comparison_from_dict(raw, repo_root=root, source=EXAMPLE)
    first_readiness = check_comparison(first, repo_root=root)
    first_preview = preview_comparison(first, repo_root=root)

    raw["execution"]["evidence_project"] = "wandb/fugue-comparison-b"
    second = comparison_from_dict(raw, repo_root=root, source=EXAMPLE)
    second_preview = preview_comparison(second, repo_root=root)

    assert first_readiness.evidence_project == "wandb/fugue-comparison-a"
    assert first_preview.matrix["evidence_project"] == "wandb/fugue-comparison-a"
    assert first_preview.experiment["evidence_project"] == (
        "wandb/fugue-comparison-a"
    )
    design = build_comparison_design_view(first_preview.to_dict())
    progress = build_comparison_progress_view(
        first_preview.to_dict(),
        phase="running",
    )
    assert next(
        item.levels
        for item in design.fixed_conditions
        if item.name == "evidence_project"
    ) == ("wandb/fugue-comparison-a",)
    assert [cell.cell_id for cell in design.cells] == [
        cell.cell_id for cell in progress.cells
    ]
    assert all(len(cell.cell_id) == 64 for cell in design.cells)
    assert first_preview.preview_digest != second_preview.preview_digest


def test_comparison_declared_destination_overrides_legacy_test_endpoint(
    tmp_path: Path,
) -> None:
    root = Path.cwd()
    raw = yaml.safe_load((EXAMPLE / "comparison.yaml").read_text())
    raw["execution"]["evidence_project"] = (
        "wandb/fugue-mcp-release-qualification-v1"
    )
    raw["execution"]["evidence_destination"] = {
        "entity": "wandb",
        "project": "fugue-mcp-release-qualification-v1",
        "api_base_url": "https://api.wandb.ai",
        "trace_base_url": "https://trace.wandb.ai",
        "app_base_url": "https://wandb.ai",
    }
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "FUGUE_WEAVE_PROJECT=wandb/news-research-agent",
                "FUGUE_WEAVE_BASE_URL=https://api.wandb.test",
                "FUGUE_WEAVE_TRACE_SERVER_URL=https://trace.wandb.test",
                "WANDB_APP_BASE_URL=https://app.wandb.test",
            ]
        )
        + "\n"
    )
    spec = comparison_from_dict(raw, repo_root=root, source=EXAMPLE)

    preview = preview_comparison(
        spec,
        repo_root=root,
        operator=OperatorService(root, env_file=env_file),
    )

    assert preview.matrix["evidence_destination"] == (
        spec.execution.evidence_destination.to_dict()
    )
    assert preview.matrix["evidence_project"] == (
        "wandb/fugue-mcp-release-qualification-v1"
    )
    assert preview.experiment["evidence_destination"] == {
        "schema_version": 1,
        "entity": "wandb",
        "project": "fugue-mcp-release-qualification-v1",
        "api_base_url": "https://api.wandb.ai",
        "trace_base_url": "https://trace.wandb.ai",
        "app_base_url": "https://wandb.ai",
    }


def test_comparison_evaluation_projects_aligned_attempts_and_weave_links() -> None:
    result = {
        "comparison_id": "mcp-main-vs-release",
        "preview_digest": "a" * 64,
        "rows": 2,
        "baseline_passed": 0,
        "candidate_passed": 1,
        "improved": 1,
        "regressed": 0,
        "unchanged": 0,
        "incomplete": 0,
        "required_evaluations_incomplete": 0,
        "evidence_project": "wandb/project",
        "operational_summary": {
            "execution_states": {"passed": 2},
            "evidence_states": {"linked": 2},
            "infrastructure_failures": 0,
        },
        "paired_cases": [
            {
                "task_id": "evaluation-reconciliation",
                "harness": "claude-code",
                "attempt": 1,
                "baseline_prediction_id": "baseline-prediction",
                "candidate_prediction_id": "candidate-prediction",
                "baseline_passed": False,
                "candidate_passed": True,
                "status": "improved",
            }
        ],
        "evidence_links": [
            {
                "label": "Agent root — evaluation-reconciliation — candidate",
                "url": "https://wandb.ai/wandb/project/r/call/agent-candidate",
            },
            {
                "label": "Evaluation attempt — evaluation-reconciliation — candidate",
                "url": "https://wandb.ai/wandb/project/r/call/eval-candidate",
            },
        ],
    }

    view = build_comparison_evaluation_view(result)

    assert len(view.cells) == 2
    baseline, candidate = view.cells
    assert baseline.task_outcome == "failed"
    assert candidate.task_outcome == "passed"
    assert candidate.factor_levels == {"candidate": "candidate"}
    assert {item["kind"] for item in candidate.evidence_links} == {
        "agent_conversation",
        "evaluation_attempt",
    }
    assert candidate.evidence_status == "missing"
    assert view.evidence_eligible is False
    assert view.evidence_scope is not None
    assert view.evidence_scope.entity == "wandb"
    assert view.evidence_scope.project == "project"
    assert view.omitted_cells == 0


def test_comparison_result_rejects_cross_project_rows() -> None:
    rows = [
        {
            "variant_id": "baseline",
            "task_id": "task",
            "harness": "claude-code",
            "trial_index": 1,
            "trace_project": "wandb/wrong-project",
        },
        {
            "variant_id": "candidate",
            "task_id": "task",
            "harness": "claude-code",
            "trial_index": 1,
            "trace_project": "wandb/expected-project",
        },
    ]

    with pytest.raises(ValueError, match="locked evidence project"):
        analyze_comparison_rows(
            comparison_id="cross-project",
            preview_digest="a" * 64,
            rows=rows,
            source="test",
            expected_evidence_project="wandb/expected-project",
        )

    with pytest.raises(ValueError, match="disagree on the evidence project"):
        analyze_comparison_rows(
            comparison_id="cross-project",
            preview_digest="a" * 64,
            rows=rows,
            source="test",
        )


def test_comparison_result_records_locked_evidence_project() -> None:
    rows = [
        {
            "variant_id": variant,
            "task_id": "task",
            "harness": "claude-code",
            "trial_index": 1,
            "trace_project": "wandb/expected-project",
        }
        for variant in ("baseline", "candidate")
    ]

    result = analyze_comparison_rows(
        comparison_id="one-project",
        preview_digest="a" * 64,
        rows=rows,
        source="test",
        expected_evidence_project="wandb/expected-project",
    )

    assert result.evidence_project == "wandb/expected-project"


def _decision_row(
    *,
    variant: str,
    task_id: str = "release-task",
    passed: bool = True,
    queried_projects: tuple[str, ...] = ("wandb/release-project",),
) -> dict[str, object]:
    call_prefix = f"{variant}-{task_id}"
    project = "wandb/release-project"
    base_url = "https://wandb.ai/wandb/release-project/weave"

    def call_ref(call_id: str) -> str:
        return f"weave:///wandb/release-project/call/{call_id}"

    return {
        "variant_id": variant,
        "task_id": task_id,
        "harness": "claude-code",
        "trial_index": 1,
        "candidate_digest": f"sha256:{variant}",
        "runtime_lock_digest": "sha256:runtime",
        "trace_project": project,
        "trace_receipt": trace_destination_identity(
            {"FUGUE_WEAVE_PROJECT": project}
        ),
        "queried_projects": list(queried_projects),
        "pass": passed,
        "status": "passed",
        "comparison_evaluation_status": "completed",
        "comparison_required_evaluation_complete": True,
        "comparison_deterministic_scores": {
            "release.factual_correctness": passed,
            "release.locked_project_scope": passed,
        },
        "comparison_deterministic_criticality": {
            "release.factual_correctness": True,
            "release.locked_project_scope": True,
        },
        "cost_usd": 1.0,
        "latency_sec": 2.0,
        "mcp_tool_names": ["query_wandb_tool"],
        "integration_provenance": [
            {
                "kind": "mcp",
                "id": f"wandb-mcp-{variant}",
                "version_identity": (
                    "git:"
                    + ("5" if variant == "candidate" else "3") * 40
                ),
                "runtime_digest": (
                    "sha256:"
                    + ("6" if variant == "candidate" else "4") * 64
                ),
                "lock_digest": (
                    "sha256:"
                    + ("7" if variant == "candidate" else "2") * 64
                ),
            }
        ],
        "weave_tool_names": {
            "mcp__wandb__query_wandb_tool": 3,
            "Read": 2,
        },
        "evaluation_id": f"evaluation-{task_id}",
        "weave_evaluation_root_call_id": f"{call_prefix}-evaluation",
        "weave_evaluation_root_ref": call_ref(
            f"{call_prefix}-evaluation"
        ),
        "evaluation_url": f"{base_url}/evaluations/{task_id}",
        "evaluation_root_object_verified": True,
        "evaluation_root_dataset_relationship_verified": True,
        "evaluation_root_prediction_relationship_verified": True,
        "dataset_id": (
            "weave:///wandb/release-project/object/release-dataset:v1"
        ),
        "dataset_url": f"{base_url}/objects/release-dataset/versions/v1",
        "dataset_version_object_verified": True,
        "eval_predict_and_score_call_id": f"{call_prefix}-eval",
        "eval_predict_and_score_ref": call_ref(f"{call_prefix}-eval"),
        "eval_predict_and_score_url": f"{base_url}/call/{call_prefix}-eval",
        "eval_predict_and_score_object_verified": True,
        "prediction_call_id": f"{call_prefix}-prediction",
        "weave_prediction_ref": call_ref(
            f"{call_prefix}-prediction"
        ),
        "prediction_url": f"{base_url}/call/{call_prefix}-prediction",
        "weave_prediction_object_verified": True,
        "prediction_child_relationship_verified": True,
        "evaluation_prediction_graph_verified": True,
        "native_agent_root_call_id": f"{call_prefix}-agent",
        "weave_agent_root_ref": call_ref(f"{call_prefix}-agent"),
        "agent_url": f"{base_url}/call/{call_prefix}-agent",
        "trace_link_status": "linked",
        "agent_graph_verified": True,
        "otel_root_span_id": f"otel-{call_prefix}",
        "infrastructure_conformance_complete": True,
        "privacy_contract_version": 2,
        "local_artifact_privacy_scan_status": "passed",
        "local_artifact_privacy_scan_digest": "1" * 64,
        "local_artifact_privacy_match_count": 0,
        "hosted_evidence_privacy_scan_status": "passed",
        "hosted_evidence_privacy_scan_digest": "2" * 64,
        "hosted_evidence_privacy_match_count": 0,
        "sandbox_cleanup_verified": True,
        "sandbox_deleted": True,
        "harbor_config": {"environment": "docker"},
        "harbor_environment": "local_harbor_docker",
        "harbor_conformance_status": "passed",
        "harbor_conformance_receipt_digest": "f" * 64,
        "harbor_policy_attestation_verified": True,
        "private_label_boundary_verified": True,
        "orphaned_sandbox": False,
    }


def _decision_policy() -> dict[str, object]:
    return {
        "release_target": "wandb-mcp-server Python package 0.4",
        "candidate_sha": "5" * 40,
        "minimum_evidence_grade": "A",
        "human_signoff_required": True,
        "gates": [
            {
                "id": "candidate-passes",
                "label": "Candidate passes",
                "category": "task",
                "source": "task.candidate_passed",
                "operator": "gte",
                "target": 1,
                "critical": True,
            }
        ],
    }


def _release_note_gate_decision(
    status: str | None,
    *,
    policy: dict[str, object] | None = None,
    coverage: tuple[dict[str, object], ...] | None = None,
):
    rows = [
        _decision_row(variant="baseline", passed=False),
        _decision_row(variant="candidate", passed=True),
    ]
    for row in rows:
        row["infrastructure_receipt_digest"] = "9" * 64
        row["infrastructure_gate_statuses"] = (
            {"bounded-timeout": status} if status is not None else {}
        )
    return _evaluate_decision(
        policy=parse_decision_policy(policy or _decision_policy()),
        rows=rows,
        deterministic={"candidate": {"passed": 1, "evaluated": 1}},
        operational={"infrastructure_failures": 0},
        improved=1,
        regressed=0,
        incomplete=0,
        required_incomplete=0,
        integrity={
            "status": "reconciled",
            "duplicate_attempt_ids": [],
            "unresolved_evidence_attempts": 0,
            "cross_project_attempts": 0,
        },
        attestation=None,
        release_note_coverage=coverage
        or (
            {
                "release_note": "tool-and-sdk-timeout-boundaries",
                "status": "infrastructure_only",
                "task_ids": [],
                "dimensions": [],
                "infrastructure_gates": ["bounded-timeout"],
                "rationale": "qualified by a separate package gate",
            },
        ),
    )


def test_release_note_infrastructure_gate_is_implicit_and_fail_closed() -> None:
    missing = _release_note_gate_decision(None)
    failed = _release_note_gate_decision("failed")
    passed = _release_note_gate_decision("passed")

    gate_id = "release-note-infrastructure-bounded-timeout"
    assert missing.status == "blocked"
    assert next(item for item in missing.gates if item.id == gate_id).status == (
        "unavailable"
    )
    assert failed.status == "hold"
    assert next(item for item in failed.gates if item.id == gate_id).status == (
        "failed"
    )
    assert passed.status == "ready_for_signoff"
    assert next(item for item in passed.gates if item.id == gate_id).status == (
        "passed"
    )


def test_v3_release_signoff_requires_actionability_review() -> None:
    decision = _release_note_gate_decision("passed")
    assert decision.status == "ready_for_signoff"
    digest = "a" * 64
    without_review = _apply_decision_attestation(
        decision,
        DecisionAttestationV1(
            signer="release-owner",
            signed_result_digest=digest,
            signed_at="2026-07-30T00:00:00Z",
        ),
        qualification_digest=digest,
        require_actionability_review=True,
    )
    rejected = _apply_decision_attestation(
        decision,
        DecisionAttestationV1(
            signer="release-owner",
            signed_result_digest=digest,
            signed_at="2026-07-30T00:00:00Z",
            review_status="rejected",
        ),
        qualification_digest=digest,
        require_actionability_review=True,
    )
    accepted = _apply_decision_attestation(
        decision,
        DecisionAttestationV1(
            signer="release-owner",
            signed_result_digest=digest,
            signed_at="2026-07-30T00:00:00Z",
            review_status="accepted_actionable",
        ),
        qualification_digest=digest,
        require_actionability_review=True,
    )

    assert without_review.status == "invalid"
    assert rejected.status == "invalid"
    assert accepted.status == "go"


def test_release_note_gate_cannot_be_weakened_by_explicit_policy() -> None:
    policy = _decision_policy()
    policy["gates"] = [
        {
            "id": "weaken-timeout",
            "label": "Weak timeout gate",
            "category": "infrastructure",
            "source": "infrastructure.gate.bounded-timeout",
            "operator": "eq",
            "target": False,
            "critical": False,
        }
    ]

    with pytest.raises(ValueError, match="critical infrastructure eq/true"):
        _release_note_gate_decision("passed", policy=policy)


def test_core_implicit_gate_cannot_be_weakened_by_explicit_policy() -> None:
    policy = _decision_policy()
    policy["gates"] = [
        {
            "id": "weak-critical-regressions",
            "label": "Allow many critical regressions",
            "category": "task",
            "source": "task.critical_regressions",
            "operator": "lte",
            "target": 100,
            "critical": False,
        }
    ]

    with pytest.raises(ValueError, match="cannot be weakened"):
        _release_note_gate_decision("passed", policy=policy)


def test_shared_release_note_infrastructure_gate_is_emitted_once() -> None:
    coverage = (
        {
            "release_note": "timeout-a",
            "status": "infrastructure_only",
            "task_ids": [],
            "dimensions": [],
            "infrastructure_gates": ["bounded-timeout"],
            "rationale": "first behavior",
        },
        {
            "release_note": "timeout-b",
            "status": "infrastructure_only",
            "task_ids": [],
            "dimensions": [],
            "infrastructure_gates": ["bounded-timeout"],
            "rationale": "second behavior",
        },
    )
    decision = _release_note_gate_decision("passed", coverage=coverage)

    assert sum(
        item.id == "release-note-infrastructure-bounded-timeout"
        for item in decision.gates
    ) == 1


def test_v2_decision_requires_exact_human_attestation() -> None:
    rows = [
        _decision_row(variant="baseline", passed=False),
        _decision_row(variant="candidate"),
    ]
    unsigned = analyze_comparison_rows(
        comparison_id="release-decision",
        preview_digest="a" * 64,
        rows=rows,
        source="test",
        expected_evidence_project="wandb/release-project",
        decision_policy=_decision_policy(),
    )

    assert unsigned.schema_version == 2
    assert unsigned.integrity["status"] == "reconciled"
    assert unsigned.decision.status == "ready_for_signoff"
    assert unsigned.decision.evidence_grade == "A"
    assert unsigned.result_digest == unsigned.qualification_digest

    signed = analyze_comparison_rows(
        comparison_id="release-decision",
        preview_digest="a" * 64,
        rows=rows,
        source="test",
        expected_evidence_project="wandb/release-project",
        decision_policy=_decision_policy(),
        attestation={
            "signer": "release-owner",
            "signed_result_digest": unsigned.result_digest,
            "signed_at": "2026-07-29T00:00:00Z",
        },
    )
    assert signed.qualification_digest == unsigned.result_digest
    assert signed.result_digest != unsigned.result_digest
    assert signed.decision.status == "go"

    mismatched = analyze_comparison_rows(
        comparison_id="release-decision",
        preview_digest="a" * 64,
        rows=rows,
        source="test",
        expected_evidence_project="wandb/release-project",
        decision_policy=_decision_policy(),
        attestation={
            "signer": "release-owner",
            "signed_result_digest": "0" * 64,
            "signed_at": "2026-07-29T00:00:00Z",
        },
    )
    assert mismatched.decision.status == "invalid"


def test_v3_result_round_trips_source_topology_and_canonical_view(
    tmp_path: Path,
) -> None:
    comparison_path = scaffold_comparison(tmp_path)
    raw = yaml.safe_load(comparison_path.read_text())
    source_project = "wandb/source-project"
    result_project = "wandb/result-project"
    raw["schema_version"] = 3
    raw["execution"]["source_evidence_project"] = source_project
    raw["execution"]["source_evidence_destination"] = {
        "entity": "wandb",
        "project": "source-project",
        "api_base_url": "https://api.wandb.ai",
        "trace_base_url": "https://trace.wandb.ai",
        "app_base_url": "https://wandb.ai",
    }
    raw["execution"]["evidence_project"] = result_project
    raw["execution"]["evidence_destination"] = {
        "entity": "wandb",
        "project": "result-project",
        "api_base_url": "https://api.wandb.ai",
        "trace_base_url": "https://trace.wandb.ai",
        "app_base_url": "https://wandb.ai",
    }
    raw["supersedes"] = [
        {
            "result_digest": "6" * 64,
            "reason": "The historical result lacked source isolation.",
        }
    ]
    spec = comparison_from_dict(raw, repo_root=tmp_path, source=tmp_path)
    operator = OperatorService(tmp_path)
    preview = preview_comparison(spec, repo_root=tmp_path, operator=operator)
    _experiment, request = materialize_comparison(
        preview,
        repo_root=tmp_path,
        operator=operator,
        approval_digest="a" * 64,
    )
    approved = request.approved_comparison
    source_digest = approved["source_lock_digest"]
    drift = {
        "status": "matched",
        "expected_digest": source_digest,
        "observed_digest": source_digest,
    }
    result_base = "https://wandb.ai/wandb/result-project/weave"
    rows: list[dict[str, object]] = []
    for cell in approved["expected_cells"]:
        variant = str(cell["variant_id"])
        passed = variant == "candidate"
        row = _decision_row(
            variant=variant,
            task_id=str(cell["task_id"]),
            passed=passed,
            queried_projects=(source_project,),
        )
        row.update(
            {
                key: cell[key]
                for key in (
                    "attempt_id",
                    "attempt_identity",
                    "task_id",
                    "variant_id",
                    "harness",
                    "trial_index",
                    "candidate_id",
                    "execution_fingerprint",
                    "applicable",
                    "skip_reason",
                )
            }
        )
        row.pop("candidate_digest", None)
        row["run_id"] = "v3-run"
        row["trace_project"] = result_project
        row["trace_receipt"] = approved["evidence_destination"]
        row["approved_comparison"] = approved
        row["integration_provenance"] = []
        row["comparison_dimension_roles"] = {
            "release.factual_correctness": "outcome",
            "release.locked_project_scope": "safety_gate",
        }
        row["source_pre_run_drift"] = drift
        row["source_post_run_drift"] = drift
        row["prediction_id"] = f"{variant}-prediction-row"
        row["agent_response"] = {
            "project": source_project,
            "answer": "safe maintainer summary",
        }
        call_prefix = f"{variant}-{cell['trial_index']}"
        row["evaluation_url"] = (
            f"{result_base}/calls/{call_prefix}-evaluation"
        )
        row["weave_evaluation_root_call_id"] = (
            f"{call_prefix}-evaluation"
        )
        row["weave_evaluation_root_ref"] = (
            "weave:///wandb/result-project/call/"
            f"{call_prefix}-evaluation"
        )
        row["dataset_url"] = (
            f"{result_base}/objects/release-dataset/versions/v1"
        )
        row["weave_dataset_id"] = (
            "weave:///wandb/result-project/object/"
            "release-dataset:v1"
        )
        row["eval_predict_and_score_url"] = (
            f"{result_base}/calls/{call_prefix}-eval"
        )
        row["eval_predict_and_score_call_id"] = f"{call_prefix}-eval"
        row["eval_predict_and_score_ref"] = (
            "weave:///wandb/result-project/call/"
            f"{call_prefix}-eval"
        )
        row["prediction_url"] = (
            f"{result_base}/calls/{call_prefix}-prediction"
        )
        row["prediction_call_id"] = f"{call_prefix}-prediction"
        row["weave_prediction_ref"] = (
            "weave:///wandb/result-project/call/"
            f"{call_prefix}-prediction"
        )
        row["agent_url"] = f"{result_base}/calls/{call_prefix}-agent"
        row["native_agent_root_call_id"] = f"{call_prefix}-agent"
        row["weave_agent_root_ref"] = (
            "weave:///wandb/result-project/call/"
            f"{call_prefix}-agent"
        )
        rows.append(row)

    result = analyze_comparison_rows(
        comparison_id=spec.id,
        preview_digest=preview.preview_digest,
        rows=rows,
        source="v3-run",
        expected_evidence_project=result_project,
        expected_source_evidence_project=source_project,
        approved_comparison=approved,
        result_schema_version=3,
        study_intent="mcp_release_maintenance",
        supersedes=spec.supersedes,
    )

    assert isinstance(result, ComparisonResultV3)
    assert result.schema_version == 3
    assert result.evidence_topology.source_destination.project_slug == (
        source_project
    )
    assert result.evidence_topology.result_destination.project_slug == (
        result_project
    )
    assert result.behavioral_summary.status == "improved"
    assert result.supersedes[0].result_digest == "6" * 64
    assert result.task_validity[0].status == "valid"
    assert result.paired_cases[0].dimension_changes[0].role == "outcome"
    assert (
        result.paired_cases[0].candidate.actual_query_scope
        == (source_project,)
    )

    role_drift = json.loads(json.dumps(rows))
    candidate_row = next(
        row for row in role_drift if row["variant_id"] == "candidate"
    )
    candidate_row["comparison_dimension_roles"][
        "release.factual_correctness"
    ] = "mechanism"
    with pytest.raises(ValueError, match="one consistent locked role"):
        analyze_comparison_rows(
            comparison_id=spec.id,
            preview_digest=preview.preview_digest,
            rows=role_drift,
            source="v3-run",
            expected_evidence_project=result_project,
            expected_source_evidence_project=source_project,
            approved_comparison=approved,
            result_schema_version=3,
            study_intent="mcp_release_maintenance",
            supersedes=spec.supersedes,
        )

    destination = tmp_path / "v3-result"
    destination.mkdir()
    (destination / "attempts.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )
    write_comparison_result(result, destination=destination)
    assert read_comparison_result(destination / "result.json") == result

    def write_rehashed_result(
        raw_result: dict[str, object],
        filename: str,
    ) -> Path:
        raw_result["qualification_digest"] = _comparison_qualification_digest(
            raw_result
        )
        raw_result["result_digest"] = raw_result["qualification_digest"]
        path = destination / filename
        path.write_text(json.dumps(raw_result), encoding="utf-8")
        return path

    forged_identity = result.to_dict()
    forged_identity["paired_cases"][0]["candidate"]["identity"][
        "candidate"
    ] = "forged-candidate"
    with pytest.raises(ValueError, match="attempt identity disagrees"):
        read_comparison_result(
            write_rehashed_result(
                forged_identity,
                "forged-attempt-identity.json",
            )
        )

    incomplete_evidence_chain = result.to_dict()
    incomplete_evidence_chain["paired_cases"][0]["candidate"][
        "evidence_links"
    ].pop()
    with pytest.raises(ValueError, match="exactly five unique evidence links"):
        read_comparison_result(
            write_rehashed_result(
                incomplete_evidence_chain,
                "incomplete-evidence-chain.json",
            )
        )

    wrong_evidence_status = result.to_dict()
    wrong_evidence_status["paired_cases"][0]["candidate"][
        "evidence_status"
    ] = "missing"
    with pytest.raises(ValueError, match="evidence status disagrees"):
        read_comparison_result(
            write_rehashed_result(
                wrong_evidence_status,
                "wrong-evidence-status.json",
            )
        )

    wrong_call_route = result.to_dict()
    evaluation_link = next(
        item
        for item in wrong_call_route["paired_cases"][0]["candidate"][
            "evidence_links"
        ]
        if item["kind"] == "evaluation_root"
    )
    evaluation_link["ref"] = "weave:///wandb/other-project/call/forged"
    evaluation_link["url"] = (
        "https://wandb.ai/wandb/other-project/weave/calls/forged"
    )
    with pytest.raises(ValueError, match="result topology"):
        read_comparison_result(
            write_rehashed_result(
                wrong_call_route,
                "wrong-call-route.json",
            )
        )

    wrong_query_scope = result.to_dict()
    wrong_query_scope["paired_cases"][0]["candidate"][
        "actual_query_scope"
    ] = [source_project, "wandb/other-project"]
    with pytest.raises(ValueError, match="actual query scope disagrees"):
        read_comparison_result(
            write_rehashed_result(
                wrong_query_scope,
                "wrong-query-scope.json",
            )
        )

    malformed_result = result.to_dict()
    malformed_result["task_validity"] = []
    malformed_result["qualification_digest"] = _comparison_qualification_digest(
        malformed_result
    )
    malformed_result["result_digest"] = malformed_result["qualification_digest"]
    malformed_result_path = destination / "malformed-result.json"
    malformed_result_path.write_text(
        json.dumps(malformed_result),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="task validity"):
        read_comparison_result(malformed_result_path)

    contradictory_pair = result.to_dict()
    contradictory_pair["paired_cases"][0]["status"] = "regressed"
    contradictory_pair["qualification_digest"] = (
        _comparison_qualification_digest(contradictory_pair)
    )
    contradictory_pair["result_digest"] = contradictory_pair[
        "qualification_digest"
    ]
    contradictory_pair_path = destination / "contradictory-pair-result.json"
    contradictory_pair_path.write_text(
        json.dumps(contradictory_pair),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pair status disagrees"):
        read_comparison_result(contradictory_pair_path)

    contradictory_behavior = result.to_dict()
    contradictory_behavior["behavioral_summary"]["status"] = "regressed"
    contradictory_behavior["qualification_digest"] = (
        _comparison_qualification_digest(contradictory_behavior)
    )
    contradictory_behavior["result_digest"] = contradictory_behavior[
        "qualification_digest"
    ]
    contradictory_behavior_path = (
        destination / "contradictory-behavior-result.json"
    )
    contradictory_behavior_path.write_text(
        json.dumps(contradictory_behavior),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="behavioral status disagrees"):
        read_comparison_result(contradictory_behavior_path)

    unknown_decision_field = result.to_dict()
    unknown_decision_field["decision"]["invented"] = True
    unknown_decision_field["qualification_digest"] = (
        _comparison_qualification_digest(unknown_decision_field)
    )
    unknown_decision_field["result_digest"] = unknown_decision_field[
        "qualification_digest"
    ]
    unknown_decision_path = destination / "unknown-decision-result.json"
    unknown_decision_path.write_text(
        json.dumps(unknown_decision_field),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown decision summary field"):
        read_comparison_result(unknown_decision_path)

    view = build_comparison_evaluation_view(result.to_dict())
    assert isinstance(view, ExperimentViewV3)
    assert view.matrix_size == result.rows
    assert view.completed_cells == result.rows
    assert view.evidence_scope is not None
    assert (
        f"{view.evidence_scope.entity}/{view.evidence_scope.project}"
        == result_project
    )
    assert (
        view.paired_cases[0]["candidate"]["evidence_links"][0]["url"]
        .startswith("https://wandb.ai/wandb/result-project/weave/calls/")
    )
    assert experiment_view_from_dict(view.to_dict()) == view
    assert view.supersedes[0]["result_digest"] == "6" * 64

    malformed = json.loads(json.dumps(view.to_dict()))
    malformed["paired_cases"][0]["candidate"]["actual_query_scope"] = [
        "wandb/news-research-agent"
    ]
    with pytest.raises(ValueError, match="actual query scope"):
        experiment_view_from_dict(malformed)

    missing_attempts = json.loads(json.dumps(view.to_dict()))
    missing_attempts["paired_cases"] = []
    with pytest.raises(
        ValueError,
        match="paired cases must expose every aligned attempt",
    ):
        experiment_view_from_dict(missing_attempts)

    running_attempt = json.loads(json.dumps(view.to_dict()))
    running_attempt["paired_cases"][0]["candidate"][
        "execution_status"
    ] = "running"
    with pytest.raises(ValueError, match="must be terminal"):
        experiment_view_from_dict(running_attempt)

    unknown_coverage = result.to_dict()
    unknown_coverage["release_note_coverage"] = [
        {
            "release_note": "structured-query-surface",
            "status": "observed_delta",
            "task_ids": ["unknown-task"],
            "dimensions": ["unknown.dimension"],
            "infrastructure_gates": [],
            "rationale": "This mapping is deliberately invalid.",
        }
    ]
    unknown_coverage["qualification_digest"] = (
        _comparison_qualification_digest(unknown_coverage)
    )
    unknown_coverage["result_digest"] = unknown_coverage[
        "qualification_digest"
    ]
    unknown_coverage_path = destination / "unknown-coverage-result.json"
    unknown_coverage_path.write_text(
        json.dumps(unknown_coverage),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="unknown task or dimension evidence",
    ):
        read_comparison_result(unknown_coverage_path)

    release_result = analyze_comparison_rows(
        comparison_id=spec.id,
        preview_digest=preview.preview_digest,
        rows=rows,
        source="v3-run",
        expected_evidence_project=result_project,
        expected_source_evidence_project=source_project,
        approved_comparison=approved,
        result_schema_version=3,
        study_intent="mcp_release_maintenance",
        supersedes=spec.supersedes,
        decision_policy=_decision_policy(),
    )
    assert release_result.decision.status == "ready_for_signoff"

    forged_gate = release_result.to_dict()
    candidate_gate = next(
        gate
        for gate in forged_gate["decision"]["gates"]
        if gate["id"] == "candidate-passes"
    )
    candidate_gate["actual"] = 999
    with pytest.raises(ValueError, match="decision gate 'candidate-passes'"):
        read_comparison_result(
            write_rehashed_result(forged_gate, "forged-decision-gate.json")
        )

    forged_decision_identity = release_result.to_dict()
    forged_decision_identity["decision"]["candidate_sha"] = "9" * 40
    with pytest.raises(ValueError, match="decision identity disagrees"):
        read_comparison_result(
            write_rehashed_result(
                forged_decision_identity,
                "forged-decision-identity.json",
            )
        )

    signed_release_result = analyze_comparison_rows(
        comparison_id=spec.id,
        preview_digest=preview.preview_digest,
        rows=rows,
        source="v3-run",
        expected_evidence_project=result_project,
        expected_source_evidence_project=source_project,
        approved_comparison=approved,
        result_schema_version=3,
        study_intent="mcp_release_maintenance",
        supersedes=spec.supersedes,
        decision_policy=_decision_policy(),
        attestation={
            "signer": "release-owner",
            "signed_result_digest": release_result.qualification_digest,
            "signed_at": "2026-07-30T00:00:00Z",
            "review_status": "accepted_actionable",
        },
    )
    assert signed_release_result.decision.status == "go"
    rejected_actionability = signed_release_result.to_dict()
    rejected_actionability["decision"]["attestation"]["review_status"] = (
        "rejected"
    )
    rejected_actionability["result_digest"] = _comparison_result_digest(
        rejected_actionability
    )
    rejected_actionability_path = (
        destination / "rejected-actionability-attestation.json"
    )
    rejected_actionability_path.write_text(
        json.dumps(rejected_actionability),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="accepted actionability"):
        read_comparison_result(rejected_actionability_path)

    both_pass_rows = json.loads(json.dumps(rows))
    for row in both_pass_rows:
        row["pass"] = True
        row["comparison_deterministic_scores"] = {
            key: True
            for key in row["comparison_deterministic_scores"]
        }
    non_discriminating = analyze_comparison_rows(
        comparison_id=spec.id,
        preview_digest=preview.preview_digest,
        rows=both_pass_rows,
        source="v3-run",
        expected_evidence_project=result_project,
        expected_source_evidence_project=source_project,
        approved_comparison=approved,
        result_schema_version=3,
        study_intent="mcp_release_maintenance",
        supersedes=spec.supersedes,
    )
    assert non_discriminating.paired_cases[0].status == "unchanged"
    assert non_discriminating.task_validity[0].status == (
        "non_discriminating"
    )
    assert non_discriminating.task_validity[0].blockers == ()
    assert non_discriminating.behavioral_summary.status == "unchanged"
    assert non_discriminating.behavioral_summary.improved_pairs == 0


def test_v2_read_binds_go_attestation_and_all_attestation_metadata(
    tmp_path: Path,
) -> None:
    rows = [
        _decision_row(variant="baseline", passed=False),
        _decision_row(variant="candidate"),
    ]
    unsigned = analyze_comparison_rows(
        comparison_id="attestation-envelope",
        preview_digest="a" * 64,
        rows=rows,
        source="test",
        expected_evidence_project="wandb/release-project",
        decision_policy=_decision_policy(),
    )
    signed = analyze_comparison_rows(
        comparison_id="attestation-envelope",
        preview_digest="a" * 64,
        rows=rows,
        source="test",
        expected_evidence_project="wandb/release-project",
        decision_policy=_decision_policy(),
        attestation={
            "signer": "release-owner",
            "signed_result_digest": unsigned.result_digest,
            "signed_at": "2026-07-29T00:00:00Z",
        },
    )
    path = tmp_path / "result.json"
    path.write_text(json.dumps(signed.to_dict()), encoding="utf-8")
    assert read_comparison_result(path) == signed

    for field, replacement in (
        ("signer", "different-owner"),
        ("signed_at", "2026-07-30T00:00:00Z"),
    ):
        tampered = signed.to_dict()
        tampered["decision"]["attestation"][field] = replacement
        path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(ValueError, match="envelope digest"):
            read_comparison_result(path)

    unsigned_go = unsigned.to_dict()
    unsigned_go["decision"]["status"] = "go"
    path.write_text(json.dumps(unsigned_go), encoding="utf-8")
    with pytest.raises(ValueError, match="missing release attestation"):
        read_comparison_result(path)

    wrong_signed_digest = signed.to_dict()
    wrong_signed_digest["decision"]["attestation"]["signed_result_digest"] = (
        "0" * 64
    )
    wrong_signed_digest["result_digest"] = _comparison_result_digest(
        wrong_signed_digest
    )
    path.write_text(json.dumps(wrong_signed_digest), encoding="utf-8")
    with pytest.raises(ValueError, match="does not sign"):
        read_comparison_result(path)


def test_local_behavioral_verdict_is_separate_from_package_release() -> None:
    result = analyze_comparison_rows(
        comparison_id="local-behavior",
        preview_digest="b" * 64,
        rows=[
            _decision_row(variant="baseline", passed=False),
            _decision_row(variant="candidate", passed=True),
        ],
        source="test",
        expected_evidence_project="wandb/release-project",
    )

    assert result.behavioral_summary.status == "improved"
    assert result.behavioral_summary.improved_pairs == 1
    assert result.decision.status == "inconclusive"
    assert result.decision.recommendation == (
        "Package release not evaluated by this Study."
    )
    assert result.decision.candidate_sha is None
    assert [item.to_dict() for item in result.candidate_source_revisions] == [
        {
            "kind": "mcp",
            "id": "wandb-mcp-candidate",
            "version_identity": "git:" + "5" * 40,
            "runtime_digest": "sha256:" + "6" * 64,
            "lock_digest": "sha256:" + "7" * 64,
        }
    ]
    assert result.paired_cases[0].baseline.tool_calls == 5
    assert result.paired_cases[0].baseline.tools == (
        "Read",
        "mcp__wandb__query_wandb_tool",
    )
    assert len(result.paired_cases[0].baseline.evidence_links) == 5
    assert {link.status for link in result.paired_cases[0].baseline.evidence_links} == {
        "resolved"
    }
    assert (
        result.paired_cases[0]
        .baseline.infrastructure["private_label_boundary_verified"]
        is True
    )
    assert (
        "label_boundary_verified"
        not in result.paired_cases[0].baseline.infrastructure
    )
    view = build_comparison_evaluation_view(result.to_dict())
    assert isinstance(view, ExperimentViewV2)
    assert view.behavioral_summary is not None
    assert view.behavioral_summary["status"] == "improved"
    assert view.backend == "local_harbor_docker"
    assert view.candidate_sha is None
    assert view.candidate_source_revisions == (
        {
            "kind": "mcp",
            "id": "wandb-mcp-candidate",
            "version_identity": "git:" + "5" * 40,
            "runtime_digest": "sha256:" + "6" * 64,
            "lock_digest": "sha256:" + "7" * 64,
        },
    )
    assert view.paired_cases[0]["pair_id"] == result.paired_cases[0].pair_id
    assert (
        view.paired_cases[0]["baseline"]["evidence_links"][0]["url"]
        == result.paired_cases[0].baseline.evidence_links[0].url
    )

    drifted = json.loads(json.dumps(view.to_dict()))
    drifted["paired_cases"][0]["baseline"]["evidence_links"][0]["uri"] = (
        "https://wandb.ai/not-canonical"
    )
    with pytest.raises(ValueError, match="unknown fields"):
        experiment_view_from_dict(drifted)

    legacy_projection = json.loads(json.dumps(view.to_dict()))
    legacy_projection["cells"] = []
    with pytest.raises(ValueError, match="experiment view has unknown fields"):
        experiment_view_from_dict(legacy_projection)


def test_identical_critical_failures_are_unchanged_not_mixed() -> None:
    result = analyze_comparison_rows(
        comparison_id="same-failure",
        preview_digest="b" * 64,
        rows=[
            _decision_row(variant="baseline", passed=False),
            _decision_row(variant="candidate", passed=False),
        ],
        source="test",
        expected_evidence_project="wandb/release-project",
    )

    assert result.paired_cases[0].status == "unchanged"
    assert result.unchanged == 1
    assert result.mixed == 0
    # A failed candidate critical dimension still blocks the aggregate verdict;
    # it does not fabricate a difference between identical paired outcomes.
    assert result.behavioral_summary.status == "mixed"
    assert result.behavioral_summary.candidate_critical_failures == 2


def test_paired_attempt_prefers_attempt_scoped_mcp_calls() -> None:
    baseline = _decision_row(variant="baseline", passed=False)
    candidate = _decision_row(variant="candidate")
    baseline["weave_tool_names"] = {"Read": 38, "query_wandb_tool": 1}
    baseline["mcp_tool_names"] = []
    baseline["mcp_tool_calls"] = [
        {"tool": "query_wandb_tool"},
        {"tool": "get_run_history_tool"},
    ]

    result = analyze_comparison_rows(
        comparison_id="normalized-mcp-tool-count",
        preview_digest="b" * 64,
        rows=[baseline, candidate],
        source="test",
        expected_evidence_project="wandb/release-project",
    )

    attempt = result.paired_cases[0].baseline
    assert attempt is not None
    assert attempt.tool_calls == 2
    assert attempt.tools == ("get_run_history_tool", "query_wandb_tool")


def test_unverified_agent_relationship_invalidates_behavioral_evidence() -> None:
    baseline = _decision_row(variant="baseline", passed=False)
    candidate = _decision_row(variant="candidate", passed=True)
    candidate["trace_link_status"] = "unverified"

    result = analyze_comparison_rows(
        comparison_id="invalid-agent-link",
        preview_digest="c" * 64,
        rows=[baseline, candidate],
        source="test",
        expected_evidence_project="wandb/release-project",
    )

    assert result.integrity["status"] == "invalid"
    assert result.integrity["invalid_evidence_attempts"] == 1
    assert result.behavioral_summary.status == "invalid"
    candidate_attempt = result.paired_cases[0].candidate
    assert candidate_attempt is not None
    assert next(
        link for link in candidate_attempt.evidence_links if link.kind == "agent_root"
    ).status == "invalid"
    view = build_comparison_evaluation_view(result.to_dict())
    assert isinstance(view, ExperimentViewV2)
    assert view.paired_cases == ()
    assert "cells" not in view.to_dict()


@pytest.mark.parametrize(
    "field",
    (
        "evaluation_root_object_verified",
        "evaluation_root_dataset_relationship_verified",
        "evaluation_root_prediction_relationship_verified",
        "dataset_version_object_verified",
        "eval_predict_and_score_object_verified",
        "weave_prediction_object_verified",
        "prediction_child_relationship_verified",
        "evaluation_prediction_graph_verified",
        "agent_graph_verified",
    ),
)
def test_each_evidence_object_and_relationship_is_verified(
    field: str,
) -> None:
    baseline = _decision_row(variant="baseline", passed=False)
    candidate = _decision_row(variant="candidate", passed=True)
    candidate[field] = False

    result = analyze_comparison_rows(
        comparison_id="invalid-evidence-graph",
        preview_digest="c" * 64,
        rows=[baseline, candidate],
        source="test",
        expected_evidence_project="wandb/release-project",
    )

    assert result.behavioral_summary.status == "invalid"
    assert result.integrity["invalid_evidence_attempts"] == 1


def test_cross_project_tool_use_invalidates_behavior_and_study_navigation() -> None:
    rows = [
        _decision_row(variant="baseline"),
        _decision_row(
            variant="candidate",
            queried_projects=(
                "wandb/release-project",
                "wandb/news-research-agent",
            ),
        ),
    ]

    result = analyze_comparison_rows(
        comparison_id="cross-project-decision",
        preview_digest="b" * 64,
        rows=rows,
        source="test",
        expected_evidence_project="wandb/release-project",
        decision_policy=_decision_policy(),
    )

    assert result.integrity["cross_project_attempts"] == 1
    assert result.integrity["status"] == "invalid"
    assert result.behavioral_summary.status == "invalid"
    assert result.decision.status == "invalid"
    assert result.decision.evidence_grade == "invalid"
    view = build_comparison_evaluation_view(result.to_dict())
    assert isinstance(view, ExperimentViewV2)
    assert view.integrity_status == "invalid"
    assert view.behavioral_summary is not None
    assert view.behavioral_summary["status"] == "invalid"
    assert view.behavioral_summary["improved_pairs"] == 0
    assert view.behavioral_summary.get("supported_claim") is None
    assert view.backend == "local_harbor_docker"
    assert view.evidence_eligible is False
    assert view.paired_cases == ()
    assert not {
        "cells",
        "arm_totals",
        "aligned_comparisons",
        "behavioral_measures",
        "mechanism_funnel",
        "outcome_summaries",
        "score_summaries",
    } & view.to_dict().keys()


def test_unavailable_dimension_makes_the_aligned_pair_incomplete() -> None:
    baseline = _decision_row(variant="baseline", passed=True)
    candidate = _decision_row(variant="candidate", passed=True)
    candidate_scores = candidate["comparison_deterministic_scores"]
    assert isinstance(candidate_scores, dict)
    candidate_scores.pop("release.factual_correctness")

    result = analyze_comparison_rows(
        comparison_id="missing-required-dimension",
        preview_digest="d" * 64,
        rows=[baseline, candidate],
        source="test",
        expected_evidence_project="wandb/release-project",
    )

    assert result.incomplete == 1
    assert result.paired_cases[0].status == "incomplete"
    assert result.behavioral_summary.status == "incomplete"


def test_failed_harbor_conformance_invalidates_behavior_and_study() -> None:
    baseline = _decision_row(variant="baseline", passed=False)
    candidate = _decision_row(variant="candidate", passed=True)
    candidate["harbor_conformance_status"] = "failed"
    candidate["sandbox_cleanup_verified"] = False
    candidate["orphaned_sandbox"] = True

    result = analyze_comparison_rows(
        comparison_id="failed-harbor-conformance",
        preview_digest="b" * 64,
        rows=[baseline, candidate],
        source="test",
        expected_evidence_project="wandb/release-project",
    )

    assert result.integrity["status"] == "invalid"
    assert result.integrity["harbor_conformance_failed_attempts"] == 1
    assert result.behavioral_summary.status == "invalid"
    assert result.behavioral_summary.supported_claim is None
    view = build_comparison_evaluation_view(result.to_dict())
    assert isinstance(view, ExperimentViewV2)
    assert view.integrity_status == "invalid"
    assert view.behavioral_summary is not None
    assert view.behavioral_summary["status"] == "invalid"
    assert view.behavioral_summary["improved_pairs"] == 0
    assert view.backend == "local_harbor_docker"
    assert view.infrastructure_health == "failed"
    assert view.paired_cases == ()
    assert "cells" not in view.to_dict()


def test_missing_infrastructure_privacy_or_cleanup_is_blocked() -> None:
    rows = [
        _decision_row(variant="baseline"),
        _decision_row(variant="candidate"),
    ]
    for row in rows:
        row.pop("infrastructure_conformance_complete")
        row.pop("local_artifact_privacy_scan_status")
        row.pop("hosted_evidence_privacy_scan_status")
        row.pop("sandbox_cleanup_verified")

    result = analyze_comparison_rows(
        comparison_id="missing-release-evidence",
        preview_digest="c" * 64,
        rows=rows,
        source="test",
        expected_evidence_project="wandb/release-project",
        decision_policy=_decision_policy(),
    )

    assert result.decision.status == "blocked"
    assert {
        gate.id
        for gate in result.decision.gates
        if gate.status == "unavailable"
    } >= {
        "infrastructure",
        "credentials-and-private-labels",
        "sandbox-cleanup",
    }


def test_legacy_privacy_scan_complete_does_not_qualify_v2_result() -> None:
    rows = [
        _decision_row(variant="baseline"),
        _decision_row(variant="candidate"),
    ]
    for row in rows:
        for field in (
            "privacy_contract_version",
            "local_artifact_privacy_scan_status",
            "local_artifact_privacy_scan_digest",
            "local_artifact_privacy_match_count",
            "hosted_evidence_privacy_scan_status",
            "hosted_evidence_privacy_scan_digest",
            "hosted_evidence_privacy_match_count",
        ):
            row.pop(field, None)
        row["privacy_scan_complete"] = True

    result = analyze_comparison_rows(
        comparison_id="legacy-privacy-is-historical",
        preview_digest="c" * 64,
        rows=rows,
        source="test",
        expected_evidence_project="wandb/release-project",
        decision_policy=_decision_policy(),
    )

    privacy_gate = next(
        gate
        for gate in result.decision.gates
        if gate.id == "credentials-and-private-labels"
    )
    assert privacy_gate.status == "unavailable"
    assert result.behavioral_summary.status == "incomplete"
    assert all(
        attempt.infrastructure["legacy_privacy_scan_complete"] is True
        and attempt.infrastructure["privacy_complete"] is False
        for pair in result.paired_cases
        for attempt in (pair.baseline, pair.candidate)
        if attempt is not None
    )


def test_hosted_privacy_failure_invalidates_behavior_and_release() -> None:
    rows = [
        _decision_row(variant="baseline"),
        _decision_row(variant="candidate"),
    ]
    for row in rows:
        for field in (
            "harbor_config",
            "harbor_environment",
            "harbor_conformance_status",
            "harbor_conformance_receipt_digest",
            "harbor_policy_attestation_verified",
        ):
            row.pop(field, None)
        row["hosted_evidence_privacy_scan_status"] = "failed"
        row["hosted_evidence_privacy_match_count"] = 1

    result = analyze_comparison_rows(
        comparison_id="hosted-privacy-leak",
        preview_digest="c" * 64,
        rows=rows,
        source="test",
        expected_evidence_project="wandb/release-project",
        decision_policy=_decision_policy(),
    )

    assert result.integrity["status"] == "invalid"
    assert result.behavioral_summary.status == "invalid"
    assert result.behavioral_summary.supported_claim is None
    assert result.decision.status == "invalid"


def test_attempt_identity_is_stable_and_duplicates_invalidate_result() -> None:
    rows = [
        _decision_row(variant="baseline"),
        _decision_row(variant="candidate"),
    ]
    first = analyze_comparison_rows(
        comparison_id="stable-attempts",
        preview_digest="d" * 64,
        rows=rows,
        source="test",
    )
    second = analyze_comparison_rows(
        comparison_id="stable-attempts",
        preview_digest="d" * 64,
        rows=rows,
        source="test",
    )
    assert [
        getattr(case, variant).attempt_id
        for case in first.paired_cases
        for variant in ("baseline", "candidate")
    ] == [
        getattr(case, variant).attempt_id
        for case in second.paired_cases
        for variant in ("baseline", "candidate")
    ]

    duplicated = analyze_comparison_rows(
        comparison_id="duplicate-attempts",
        preview_digest="e" * 64,
        rows=[rows[0], rows[0], rows[1]],
        source="test",
    )
    assert duplicated.integrity["status"] == "invalid"
    assert duplicated.decision.status == "invalid"


def test_supplied_attempt_identity_drift_is_rejected_before_normalization() -> (
    None
):
    baseline = _decision_row(variant="baseline")
    candidate = _decision_row(variant="candidate")
    baseline["attempt_identity"] = attempt_identity(
        task_id="release-task",
        arm="candidate",
        harness="claude-code",
        attempt=1,
        candidate="sha256:baseline",
        runtime="sha256:runtime",
    )

    with pytest.raises(
        ValueError,
        match="supplied attempt identity disagrees",
    ):
        analyze_comparison_rows(
            comparison_id="attempt-identity-drift",
            preview_digest="d" * 64,
            rows=[baseline, candidate],
            source="test",
        )


def test_answer_excerpt_requires_privacy_evidence_and_redacts_json_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-ant-private-test-value"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    row = _decision_row(variant="candidate")
    row["agent_response"] = json.dumps(
        {
            "api_key": secret,
            "password": "quoted-private-password",
            "recommendation": "hold",
        }
    )

    excerpt = _sanitized_answer_excerpt(row)

    assert excerpt is not None
    assert secret not in excerpt
    assert "quoted-private-password" not in excerpt
    assert "[redacted]" in excerpt
    assert '"recommendation":"hold"' in excerpt

    row["agent_response"] = (
        "```json\n"
        + json.dumps(
            {
                "token": secret,
                "recommendation": "investigate",
            }
        )
        + "\n```"
    )
    fenced_excerpt = _sanitized_answer_excerpt(row)
    assert fenced_excerpt is not None
    assert secret not in fenced_excerpt
    assert '"recommendation":"investigate"' in fenced_excerpt

    row["agent_response"] = f'ANTHROPIC_API_KEY="{secret}" result=hold'
    text_excerpt = _sanitized_answer_excerpt(row)
    assert text_excerpt is not None
    assert secret not in text_excerpt

    row["hosted_evidence_privacy_scan_status"] = "unavailable"
    assert _sanitized_answer_excerpt(row) is None


def test_approved_comparison_manifest_reconciles_exact_run_and_coordinates(
    tmp_path: Path,
) -> None:
    root = tmp_path
    comparison_path = scaffold_comparison(root)
    spec = load_comparison(comparison_path, repo_root=root)
    operator = OperatorService(root)
    preview = preview_comparison(spec, repo_root=root, operator=operator)
    experiment, request = materialize_comparison(
        preview,
        repo_root=root,
        operator=operator,
        approval_digest="a" * 64,
    )
    approved = request.approved_comparison
    plan = operator.resolve_run_plan(
        request,
        run_id="approved-run",
        experiment=experiment,
    )
    assert len(plan.cells) == approved["expected_cell_count"]
    assert all(cell.approved_comparison == approved for cell in plan.cells)
    assert approved["preview_digest"] == preview.preview_digest
    assert approved["spec_digest"] == spec.spec_digest
    assert approved["taskset_digest"] == preview.readiness["taskset_digest"]
    assert approved["scorer_digests"] == preview.readiness["evaluator_digests"]
    assert approved["evidence_project"] == preview.matrix["evidence_project"]
    assert approved["approved_inputs_digest"] == stable_digest(
        approved["approved_inputs"]
    )
    assert approved["candidate_source_revisions_required"] is False
    rows = [
        {
            **{
                key: cell[key]
                for key in (
                    "attempt_id",
                    "attempt_identity",
                    "task_id",
                    "variant_id",
                    "harness",
                    "trial_index",
                    "candidate_id",
                    "execution_fingerprint",
                    "applicable",
                    "skip_reason",
                )
            },
            "integration_provenance": [],
            "run_id": "approved-run",
            "trace_project": approved["evidence_project"],
            "trace_receipt": approved["evidence_destination"],
            "approved_comparison": approved,
            "comparison_required_evaluation_complete": True,
        }
        for cell in approved["expected_cells"]
    ]

    result = analyze_comparison_rows(
        comparison_id=spec.id,
        preview_digest=preview.preview_digest,
        rows=rows,
        source="approved-run",
        approved_comparison=approved,
    )

    assert result.rows == approved["expected_cell_count"]
    assert result.integrity["approved_manifest_status"] == "reconciled"
    assert result.integrity["approved_manifest_digest"] == approved["lock_digest"]

    wrong_run = [dict(row) for row in rows]
    wrong_run[0]["run_id"] = "another-run"
    with pytest.raises(ValueError, match="source run"):
        analyze_comparison_rows(
            comparison_id=spec.id,
            preview_digest=preview.preview_digest,
            rows=wrong_run,
            source="approved-run",
            approved_comparison=approved,
        )

    duplicate_coordinate = [dict(row) for row in rows[:-1]]
    duplicate = dict(rows[0])
    duplicate.pop("attempt_id")
    duplicate.pop("attempt_identity")
    duplicate["candidate_id"] = "f" * 64
    duplicate_coordinate.append(duplicate)
    with pytest.raises(ValueError, match="duplicate attempt coordinates"):
        analyze_comparison_rows(
            comparison_id=spec.id,
            preview_digest=preview.preview_digest,
            rows=duplicate_coordinate,
            source="approved-run",
            approved_comparison=approved,
        )

    drifted = json.loads(json.dumps(approved))
    baseline_cells = [
        cell
        for cell in drifted["expected_cells"]
        if cell["variant_id"] == "baseline"
    ]
    drifted_cell = baseline_cells[-1]
    drifted_cell["candidate_id"] = "e" * 64
    drifted_cell["attempt_identity"] = attempt_identity(
        task_id=drifted_cell["task_id"],
        arm=drifted_cell["variant_id"],
        harness=drifted_cell["harness"],
        attempt=drifted_cell["trial_index"],
        candidate=drifted_cell["candidate_id"],
        runtime=drifted_cell["execution_fingerprint"],
    )
    drifted_cell["attempt_id"] = attempt_id(**drifted_cell["attempt_identity"])
    drifted["expected_cells_digest"] = stable_digest(drifted["expected_cells"])
    drifted["lock_digest"] = stable_digest(
        {key: value for key, value in drifted.items() if key != "lock_digest"}
    )
    drifted_rows = [
        {
            **{
                key: cell[key]
                for key in (
                    "attempt_id",
                    "attempt_identity",
                    "task_id",
                    "variant_id",
                    "harness",
                    "trial_index",
                    "candidate_id",
                    "execution_fingerprint",
                    "applicable",
                    "skip_reason",
                )
            },
            "integration_provenance": [],
            "run_id": "approved-run",
            "trace_project": drifted["evidence_project"],
            "approved_comparison": drifted,
        }
        for cell in drifted["expected_cells"]
    ]
    with pytest.raises(ValueError, match="per-arm identity drift"):
        analyze_comparison_rows(
            comparison_id=spec.id,
            preview_digest=preview.preview_digest,
            rows=drifted_rows,
            source="approved-run",
            approved_comparison=drifted,
        )

    applicability_drift = json.loads(json.dumps(approved))
    applicability_drift["expected_cells"][0]["applicable"] = False
    applicability_drift["expected_cells"][0]["skip_reason"] = "new runtime skip"
    applicability_drift["expected_cells_digest"] = stable_digest(
        applicability_drift["expected_cells"]
    )
    applicability_drift["lock_digest"] = stable_digest(
        {
            key: value
            for key, value in applicability_drift.items()
            if key != "lock_digest"
        }
    )
    with pytest.raises(ValueError, match="approved comparison"):
        operator.resolve_run_plan(
            replace(request, approved_comparison=applicability_drift),
            run_id="applicability-drift",
            experiment=experiment,
        )

    provenance_drift = [dict(row) for row in rows]
    provenance_drift[0]["integration_provenance"] = [
        {
            "id": "unapproved",
            "kind": "mcp",
            "version_identity": "git:" + "1" * 40,
            "runtime_digest": "sha256:" + "2" * 64,
        }
    ]
    with pytest.raises(ValueError, match="integration provenance drifted"):
        analyze_comparison_rows(
            comparison_id=spec.id,
            preview_digest=preview.preview_digest,
            rows=provenance_drift,
            source="approved-run",
            approved_comparison=approved,
        )

    source_required = json.loads(json.dumps(approved))
    source_required["candidate_source_revisions_required"] = True
    candidate_ids = sorted(
        {
            cell["candidate_id"]
            for cell in source_required["expected_cells"]
            if cell["variant_id"] == "candidate"
        }
    )
    source_required["candidate_source_identity_digest"] = stable_digest(
        {"candidate_ids": candidate_ids, "source_revisions": []}
    )
    source_required["lock_digest"] = stable_digest(
        {
            key: value
            for key, value in source_required.items()
            if key != "lock_digest"
        }
    )
    with pytest.raises(ValueError, match="no approved candidate source revision"):
        analyze_comparison_rows(
            comparison_id=spec.id,
            preview_digest=preview.preview_digest,
            rows=[
                {**row, "approved_comparison": source_required}
                for row in rows
            ],
            source="approved-run",
            approved_comparison=source_required,
        )

    private_digest = approved["approved_inputs"]["private_labels_sha256"]
    frozen_private = (
        root
        / ".fugue/private/comparison-inputs/labels"
        / f"{private_digest}.jsonl"
    )
    frozen_private.chmod(0o600)
    frozen_private.write_text('{"id":"drifted","expected":true}\n')
    with pytest.raises(ValueError, match="private labels immutable copy changed"):
        score_comparison_rows(
            spec,
            [],
            repo_root=root,
            approved_comparison=approved,
        )


def test_result_write_recomputes_final_exported_rows(tmp_path: Path) -> None:
    rows = [
        _decision_row(variant="baseline", passed=False),
        _decision_row(variant="candidate"),
    ]
    result = analyze_comparison_rows(
        comparison_id="recomputed-result",
        preview_digest="f" * 64,
        rows=rows,
        source="test",
        expected_evidence_project="wandb/release-project",
        decision_policy=_decision_policy(),
    )
    destination = tmp_path / "result"
    destination.mkdir()
    with pytest.raises(FileNotFoundError, match="requires final attempts.jsonl"):
        write_comparison_result(result, destination=destination)
    assert not (destination / "result.json").exists()
    (destination / "attempts.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    write_comparison_result(result, destination=destination)
    assert read_comparison_result(destination / "result.json") == result

    changed_rows = [dict(row) for row in rows]
    changed_rows[1]["pass"] = False
    (destination / "attempts.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in changed_rows)
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        RuntimeError,
        match="disagrees with the final exported attempt rows",
    ):
        write_comparison_result(result, destination=destination)


def test_v1_result_remains_readable(tmp_path: Path) -> None:
    raw = {
        "schema_version": 1,
        "comparison_id": "legacy-result",
        "preview_digest": "a" * 64,
        "source": "legacy",
        "evidence_project": None,
        "rows": 2,
        "baseline_passed": 1,
        "candidate_passed": 1,
        "improved": 0,
        "regressed": 0,
        "unchanged": 1,
        "incomplete": 0,
        "required_evaluations_incomplete": 0,
        "deterministic_summary": {},
        "judge_summary": {},
        "mechanism_summary": {},
        "operational_summary": {},
        "evidence_links": [],
        "paired_cases": [],
        "limitations": [],
    }
    raw["result_digest"] = _comparison_result_digest(raw)
    path = tmp_path / "result.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    result = read_comparison_result(path)
    assert result.schema_version == 1
    assert result.comparison_id == "legacy-result"


def test_decision_policy_change_invalidates_preview_approval() -> None:
    root = Path.cwd()
    raw = yaml.safe_load((EXAMPLE / "comparison.yaml").read_text())
    raw["decision_policy"] = _decision_policy()
    first = comparison_from_dict(raw, repo_root=root, source=EXAMPLE)
    first_preview = preview_comparison(first, repo_root=root)

    raw["decision_policy"]["gates"][0]["target"] = 2
    second = comparison_from_dict(raw, repo_root=root, source=EXAMPLE)
    second_preview = preview_comparison(second, repo_root=root)

    assert first_preview.preview_digest != second_preview.preview_digest


def test_wandb_comparison_readiness_requires_exact_runtime_lock() -> None:
    root = Path.cwd()
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=root)
    execution = replace(
        spec.execution,
        environment={
            "type": "wandb",
            "runtime_lock": ".fugue/does-not-exist.runtime-lock.json",
        },
    )

    readiness = check_comparison(
        replace(spec, execution=execution),
        repo_root=root,
    )

    assert readiness.status == "blocked"
    assert readiness.runtime_lock_digests == {}
    assert any(
        "W&B Serverless runtime" in blocker
        and "does-not-exist.runtime-lock.json" in blocker
        for blocker in readiness.blockers
    )


def test_comparison_scoring_prefers_locked_answer_artifact(
    tmp_path: Path,
) -> None:
    trial_dir = tmp_path / "trial"
    answer = trial_dir / "artifacts" / "logs" / "artifacts" / "fugue-answer.md"
    answer.parent.mkdir(parents=True)
    answer.write_text('{"answer": 42}', encoding="utf-8")

    assert (
        _comparison_trial_output(
            {
                "trial_dir": trial_dir.as_posix(),
                "agent_response": "The answer was written to the artifact.",
            }
        )
        == '{"answer": 42}'
    )
    assert _comparison_trial_output({"agent_response": "terminal answer"}) == (
        "terminal answer"
    )


def test_source_use_demo_uses_packaged_assets_outside_checkout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from argparse import Namespace

    from fugue.bench.cli import _comparison_demo

    destination = tmp_path / "result"
    assert (
        _comparison_demo(
            Namespace(
                repo_root=tmp_path,
                out=destination,
                json=True,
            )
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    written = json.loads((destination / "result.json").read_text())
    reproduction = json.loads(
        (destination / "reproduction.json").read_text()
    )
    assert output == written
    assert output["source"] == "bundled-replay"
    assert output["rows"] == 16
    assert output["incomplete"] == 0
    assert reproduction["private_labels_included"] is False


def test_public_task_resources_are_digest_locked_into_the_task(tmp_path: Path) -> None:
    (tmp_path / "configs/fugue/skills/verify-current-source").mkdir(
        parents=True
    )
    (tmp_path / "configs/fugue/skills/verify-current-source/SKILL.md").write_text(
        "---\nname: verify-current-source\ndescription: Verify sources.\n---\n"
    )
    (tmp_path / "corpus").mkdir()
    resource = tmp_path / "corpus" / "policy.json"
    resource.write_text('{"documents": []}\n')
    (tmp_path / "tasks.jsonl").write_text(
        json.dumps(
            {
                "id": "policy",
                "input": {"question": "Read the corpus and answer."},
                "resources": [
                    {
                        "path": "corpus/policy.json",
                        "target": "/workspace/resources/corpus.json",
                    }
                ],
                "partition": "holdout",
            }
        )
        + "\n"
    )
    (tmp_path / "labels.jsonl").write_text(
        json.dumps(
            {
                "id": "policy",
                "expected": {"answer": "yes"},
                "base_output": {"answer": "no"},
                "gold_output": {"answer": "yes"},
            }
        )
        + "\n"
    )
    raw = yaml.safe_load((EXAMPLE / "comparison.yaml").read_text())
    raw["taskset"] = {"tasks": "tasks.jsonl", "private_labels": "labels.jsonl"}
    spec = comparison_from_dict(raw, repo_root=tmp_path, source=tmp_path)

    experiment, manifest, public = compile_comparison(spec, repo_root=tmp_path)

    assert public[0]["attachments"] == [
        {
            "locked_relative": (
                ".fugue/runtime/comparison-inputs/resources/"
                f"{hashlib.sha256(resource.read_bytes()).hexdigest()}/policy.json"
            ),
            "sha256": hashlib.sha256(resource.read_bytes()).hexdigest(),
            "target": "/workspace/resources/corpus.json",
        }
    ]
    assert "@sha256:" in public[0]["environment"]["base_image"]
    evaluator_digest = next(
        iter(check_comparison(spec, repo_root=tmp_path).evaluator_digests.values())
    )
    assert manifest["tasks"][0]["metadata"]["task_authoring"]["profile_digests"] == {
        "comparison-evaluator:fact-and-source": evaluator_digest
    }
    assert experiment.research_view is not None
    assert experiment.research_view.scorers[0].revision == evaluator_digest


def test_mechanism_summary_keeps_assignment_registration_and_use_distinct() -> None:
    root = Path.cwd()
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=root)
    rows = [
        {
            "answer": {"amount": 125, "source": "expense-policy-v4.md"},
            "harness": "codex",
            "prediction_id": "candidate-1",
            "task_id": "expense-limit",
            "trial_index": 1,
            "variant_id": "candidate",
            "skills_assigned": ["verify-current-source"],
            "skills_registered": ["verify-current-source"],
            "skill_registration_status": "registered",
            "skill_invocation_evidence": {
                "status": "observed",
                "skills_invoked": ["verify-current-source"],
            },
            "inspected_paths": ["/workspace/resources/expense-policy-v4.md"],
        },
        {
            "answer": {"amount": 100, "source": "expense-policy-v3.md"},
            "harness": "codex",
            "prediction_id": "baseline-1",
            "task_id": "expense-limit",
            "trial_index": 1,
            "variant_id": "baseline",
        },
    ]
    scored = score_comparison_rows(spec, rows, repo_root=root)
    result = analyze_comparison_rows(
        comparison_id=spec.id,
        preview_digest="a" * 64,
        rows=scored,
        source="test",
    )

    assert result.mechanism_summary["skill_assigned"]["candidate"] == {
        "observed": 1,
        "applicable": 1,
        "unavailable": 0,
    }
    assert result.mechanism_summary["relevant_source_used"]["candidate"][
        "observed"
    ] == 1


def test_zero_row_comparison_cannot_succeed() -> None:
    with pytest.raises(ValueError, match="at least one attempt row"):
        analyze_comparison_rows(
            comparison_id="empty",
            preview_digest="0" * 64,
            rows=[],
            source="test",
        )


def test_required_judge_needs_reviewed_calibration(tmp_path: Path) -> None:
    root = Path.cwd()
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=root)
    judge = ComparisonEvaluatorV1(
        id="maintainer-review",
        type="llm_judge",
        required=True,
        profile="wandb/openai/gpt-oss-120b",
        calibration=None,
        rubric="Score evidence grounding and calibration.",
        dimensions=("evidence_grounding", "calibration"),
        evidence=("tool_names",),
        reserve_cost_usd=0.1,
    )
    blocked = replace(spec, evaluators=(*spec.evaluators, judge))
    readiness = check_comparison(blocked, repo_root=root)
    assert readiness.status == "blocked"
    assert any("no reviewed calibration" in item for item in readiness.blockers)

    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "review_status": "adjudicated",
                "reviewers_per_example": 2,
                "disagreements_adjudicated": True,
                "judge_profile": judge.profile,
                "rubric_digest": stable_digest(
                    {
                        "schema_version": 1,
                        "judge_id": judge.id,
                        "profile": judge.profile,
                        "rubric": judge.rubric,
                        "dimensions": list(judge.dimensions),
                        "evidence": list(judge.evidence),
                    }
                ),
                "examples": 48,
                "calibration_examples": 36,
                "holdout_examples": 12,
                "cases_digest": "a" * 64,
                "true_positive_rate": 0.9,
                "true_negative_rate": 0.9,
                "calibration_true_positive_rate": 0.9,
                "calibration_true_negative_rate": 0.9,
                "holdout_true_positive_rate": 0.9,
                "holdout_true_negative_rate": 0.9,
                "critical_false_passes": 0,
                "passed": True,
            }
        )
    )
    copied = root / ".fugue" / "test-comparison-calibration.json"
    copied.parent.mkdir(parents=True, exist_ok=True)
    copied.write_text(calibration.read_text())
    try:
        qualified_judge = replace(
            judge, calibration=copied.relative_to(root).as_posix()
        )
        ready = replace(
            spec,
            evaluators=(*spec.evaluators, qualified_judge),
            execution=replace(spec.execution, max_cost_usd=10),
        )
        assert check_comparison(ready, repo_root=root).status == "ready"
        split_regression = json.loads(copied.read_text())
        split_regression["holdout_true_negative_rate"] = 0.8
        copied.write_text(json.dumps(split_regression))
        split_readiness = check_comparison(ready, repo_root=root)
        assert split_readiness.status == "blocked"
        assert any(
            "calibration or holdout is below 0.85" in item
            for item in split_readiness.blockers
        )
    finally:
        copied.unlink(missing_ok=True)


def test_v3_attempt_exports_blind_judge_scores_without_rationale() -> None:
    row = {
        "attempt_id": "a" * 64,
        "attempt_identity": {
            "task_id": "maintainer-project-health",
            "arm": "candidate",
            "harness": "claude-code",
            "attempt": 1,
            "candidate": "b" * 64,
            "runtime": "c" * 64,
        },
        "prediction_id": "prediction-1",
        "pass": True,
        "status": "completed",
        "comparison_evaluation_status": "scored",
        "comparison_deterministic_scores": {
            "natural-maintainer.answer_correct": True,
        },
        "comparison_judge_scores": {
            "maintainer-actionability.maintenance_actionability": 0.8,
        },
        "mcp_tool_calls": [],
    }

    attempt = _paired_attempt_view_v3(row)

    assert attempt is not None
    assert attempt.scores == {
        "natural-maintainer.answer_correct": True,
        ("comparison.judge.maintainer-actionability.maintenance_actionability"): 0.8,
    }
    assert (
        attempt.score_explanations[
            ("comparison.judge.maintainer-actionability.maintenance_actionability")
        ]
        == "Blind judge score; no rationale or private truth is published."
    )


def test_checkpoint_requires_configured_advisory_judge_to_score() -> None:
    root = Path.cwd()
    spec = load_comparison(
        MCP_MAINTENANCE_EXAMPLE / "natural-maintainer-canary-local-v3.yaml",
        repo_root=root,
    )
    row = {
        "comparison_judges": {
            "maintainer-actionability": {
                "status": "unavailable",
                "reason": "judge evaluation failed: ReadTimeout",
            }
        }
    }

    with pytest.raises(
        RuntimeError,
        match="first-cell judge checkpoint did not score",
    ):
        _require_checkpoint_judges(
            spec,
            row,
            checkpoint_index=0,
        )

    assert row["comparison_judge_checkpoint_status"] == "failed"
    assert row["comparison_judge_checkpoint_unavailable"] == [
        "maintainer-actionability"
    ]

    scored = {
        "comparison_judges": {
            "maintainer-actionability": {"status": "scored"}
        }
    }
    _require_checkpoint_judges(
        spec,
        scored,
        checkpoint_index=0,
    )
    assert scored["comparison_judge_checkpoint_status"] == "passed"

    after_checkpoint: dict[str, object] = {}
    _require_checkpoint_judges(
        spec,
        after_checkpoint,
        checkpoint_index=1,
    )
    assert after_checkpoint == {}


def test_custom_scorer_uses_locked_sandbox_and_private_expected_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path.cwd()
    scorer_path = root / ".fugue" / "test-comparison-scorer.py"
    scorer_path.parent.mkdir(parents=True, exist_ok=True)
    scorer_path.write_text(
        "def score(task, output, evidence):\n"
        "    return {'fact_correct': output == evidence['expected']}\n"
    )
    observed: dict[str, object] = {}

    def fake_runner(*, source, evidence, reference, profile, limits):
        observed.update(
            source=source,
            evidence=evidence,
            reference=reference,
            profile=profile,
            limits=limits,
        )
        passed = reference["output"] == reference["expected"]
        return {
            "score": 1.0 if passed else 0.0,
            "reason": "custom deterministic scorer",
            "details": {"fact_correct": passed},
        }

    monkeypatch.setattr(
        "fugue.bench.task_authoring.run_inline_scorer", fake_runner
    )
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=root)
    evaluator = replace(
        spec.evaluators[0],
        checks=(),
        scorer=scorer_path.relative_to(root).as_posix(),
        runtime="python312-sandbox-v1",
        dimensions=("fact_correct",),
    )
    custom = replace(spec, evaluators=(evaluator,))
    try:
        rows = score_comparison_rows(
            custom,
            [
                {
                    "task_id": "expense-limit",
                    "variant_id": "candidate",
                    "harness": "codex",
                    "trial_index": 1,
                    "answer": {
                        "amount": 125,
                        "source": "expense-policy-v4.md",
                    },
                }
            ],
            repo_root=root,
        )
    finally:
        scorer_path.unlink(missing_ok=True)

    assert rows[0]["pass"] is True
    assert rows[0]["comparison_deterministic_scores"] == {
        "fact-and-source.fact_correct": True
    }
    assert observed["reference"] == {
        "task": {
            "id": "expense-limit",
            "input": {
                "question": (
                    "Return JSON containing the current expense amount "
                    "and its source."
                )
            },
            "resources": [],
            "tags": ["policy"],
            "partition": "holdout",
        },
        "output": {"amount": 125, "source": "expense-policy-v4.md"},
        "expected": {"amount": 125, "source": "expense-policy-v4.md"},
    }
    assert observed["evidence"] == {}
    assert "--network" not in str(observed["source"])


def test_blind_judge_receives_only_public_task_output_and_permitted_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path.cwd()
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=root)
    judge = ComparisonEvaluatorV1(
        id="maintainer-review",
        type="llm_judge",
        required=True,
        profile="wandb/openai/gpt-oss-120b",
        rubric="Score grounding, usefulness, prioritization, and calibration.",
        dimensions=(
            "evidence_grounding",
            "usefulness",
            "prioritization",
            "calibration",
        ),
        evidence=("tool_names", "inspected_paths"),
        reserve_cost_usd=0.1,
    )
    captured: dict[str, str] = {}

    def fake_post(_client, _route, _key, _env, prompt):
        captured["prompt"] = prompt
        return (
            {
                "scores": {
                    "evidence_grounding": 1,
                    "usefulness": 0.75,
                    "prioritization": 0.5,
                    "calibration": 1,
                },
                "overall_assessment": "Grounded and appropriately bounded.",
                "uncertainty": 0.1,
                "rationale": "The response cites the inspected current source.",
            },
            {"input_tokens": 100, "output_tokens": 40},
        )

    monkeypatch.setattr("fugue.bench.evaluations._post_judge", fake_post)
    judged = replace(spec, evaluators=(*spec.evaluators, judge))
    rows = score_comparison_rows(
        judged,
        [
            {
                "answer": {"amount": 125, "source": "expense-policy-v4.md"},
                "harness": "secret-harness-name",
                "prediction_id": "secret-prediction",
                "task_id": "expense-limit",
                "trial_index": 1,
                "variant_id": "secret-candidate-revision",
                "tool_calls": [
                    {
                        "name": "search",
                        "arguments": {"private": "do-not-send"},
                        "output": "private tool body",
                    }
                ],
                "mcp_tool_calls": [
                    {"tool": "query_wandb_tool"},
                ],
                "mcp_tool_names": ["summarize_evaluation_tool"],
                "inspected_paths": ["expense-policy-v4.md"],
                "comparison_deterministic_scores": {"private": True},
            }
        ],
        repo_root=root,
        env={"WANDB_API_KEY": "secret", "FUGUE_WANDB_INFERENCE_PROJECT": "wandb/test"},
    )

    prompt = captured["prompt"]
    assert "secret-candidate-revision" not in prompt
    assert "secret-harness-name" not in prompt
    assert "secret-prediction" not in prompt
    assert "do-not-send" not in prompt
    assert "private tool body" not in prompt
    assert (
        '"tool_names": ["query_wandb_tool", "search", "summarize_evaluation_tool"]'
    ) in prompt
    assert rows[0]["comparison_judge_status"] == "scored"
    assert rows[0]["comparison_required_evaluation_complete"] is True
    privacy = rows[0]["comparison_judges"]["maintainer-review"]["route_receipt"][
        "judge_input_privacy"
    ]
    assert privacy["status"] == "passed"
    assert len(privacy["payload_sha256"]) == 64


def test_blind_judge_rejects_secret_before_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path.cwd()
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=root)
    judge = ComparisonEvaluatorV1(
        id="maintainer-review",
        type="llm_judge",
        required=True,
        profile="wandb/openai/gpt-oss-120b",
        rubric="Score maintenance actionability.",
        dimensions=("maintenance_actionability",),
        evidence=("tool_names",),
        reserve_cost_usd=0.1,
    )
    provider_calls = []

    def forbidden_post(*_args, **_kwargs):
        provider_calls.append("called")
        raise AssertionError("privacy failure must stop before the provider")

    monkeypatch.setattr("fugue.bench.evaluations._post_judge", forbidden_post)
    rows = score_comparison_rows(
        replace(spec, evaluators=(*spec.evaluators, judge)),
        [
            {
                "answer": {
                    "amount": 125,
                    "source": "credential super-secret-value",
                },
                "task_id": "expense-limit",
                "trial_index": 1,
                "variant_id": "candidate",
                "harness": "claude-code",
                "mcp_tool_names": ["query_wandb_tool"],
            }
        ],
        repo_root=root,
        env={
            "WANDB_API_KEY": "super-secret-value",
            "FUGUE_WANDB_INFERENCE_PROJECT": "wandb/test",
        },
    )

    assert provider_calls == []
    assert rows[0]["comparison_judge_status"] == "unavailable"
    assert rows[0]["comparison_required_evaluation_complete"] is False


def test_scaffold_refuses_non_empty_destination(tmp_path: Path) -> None:
    destination = tmp_path / "comparison"
    scaffold_comparison(destination)
    assert (destination / "comparison.yaml").is_file()
    assert (
        destination
        / "configs/fugue/skills/verify-current-source/SKILL.md"
    ).is_file()
    spec = load_comparison(
        destination / "comparison.yaml",
        repo_root=destination,
    )
    preview = preview_comparison(spec, repo_root=destination)
    assert preview.readiness["status"] == "ready"
    assert preview.matrix["estimated_trials"] == 4
    assert not (destination / ".fugue").exists()
    with pytest.raises(FileExistsError, match="non-empty"):
        scaffold_comparison(destination)


def _local_preparation_spec(
    tmp_path: Path,
    *,
    preparation_required: bool,
) -> tuple[Path, ComparisonSpecV1]:
    comparison_path = scaffold_comparison(tmp_path / "comparison")
    raw = yaml.safe_load(comparison_path.read_text(encoding="utf-8"))
    raw["execution"]["evidence_project"] = "wandb/fugue-test"
    raw["execution"]["approval_required"] = False
    raw["execution"]["preparation_required"] = preparation_required
    comparison_path.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )
    root = comparison_path.parent
    return root, load_comparison(comparison_path, repo_root=root)


def test_local_comparison_requires_exact_preparation_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, spec = _local_preparation_spec(
        tmp_path,
        preparation_required=True,
    )
    monkeypatch.setattr(
        "fugue.bench.comparison._runtime_readiness",
        lambda *_args, **_kwargs: (
            {"agent:codex:amd64": "a" * 64, "task:policy-limit:amd64": "b" * 64},
            [],
        ),
    )

    readiness = check_comparison(spec, repo_root=root)

    assert readiness.status == "blocked"
    assert any(
        "comparison preparation is missing or drifted" in blocker
        for blocker in readiness.blockers
    )


def test_prepare_then_preview_is_stable_and_runtime_drift_invalidates_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, spec = _local_preparation_spec(
        tmp_path,
        preparation_required=True,
    )
    runtime_digest = {"value": "a" * 64}

    def runtime_readiness(*_args: object, **_kwargs: object):
        return (
            {
                "agent:codex:amd64": runtime_digest["value"],
                "task:policy-limit:amd64": "b" * 64,
            },
            [],
        )

    monkeypatch.setattr(
        "fugue.bench.comparison._runtime_readiness",
        runtime_readiness,
    )
    monkeypatch.setattr(OperatorService, "prepare", lambda *_args, **_kwargs: None)
    operator = OperatorService(root)

    receipt, preview, receipt_path = prepare_comparison(
        spec,
        repo_root=root,
        operator=operator,
    )
    second = preview_comparison(
        spec,
        repo_root=root,
        operator=operator,
    )

    assert receipt_path.is_file()
    assert second.preview_digest == preview.preview_digest
    assert second.readiness["runtime_lock_digests"][
        "comparison_preparation"
    ] == receipt["receipt_digest"]

    runtime_digest["value"] = "c" * 64
    drifted = preview_comparison(
        spec,
        repo_root=root,
        operator=operator,
    )
    assert drifted.preview_digest != preview.preview_digest
    assert drifted.readiness["status"] == "blocked"
    assert any(
        "runtime locks do not match" in blocker
        for blocker in drifted.readiness["blockers"]
    )


def test_execute_comparison_never_prepares_after_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, spec = _local_preparation_spec(
        tmp_path,
        preparation_required=False,
    )
    preview = preview_comparison(spec, repo_root=root)

    def forbidden_prepare(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("execution must not prepare after approval")

    def stop_at_execution(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("execution reached")

    monkeypatch.setattr(OperatorService, "prepare", forbidden_prepare)
    monkeypatch.setattr(OperatorService, "execute_run", stop_at_execution)

    with pytest.raises(RuntimeError, match="execution reached"):
        execute_comparison(
            preview,
            approval_digest="",
            repo_root=root,
            fetch_weave=False,
            publish_research=False,
        )
