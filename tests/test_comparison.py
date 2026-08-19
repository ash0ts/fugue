from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml

from fugue.bench.candidates import attempt_id, attempt_identity, stable_digest
from fugue.bench.comparison import (
    COMPARISON_RUNTIME_ROOT,
    MAX_COMPARISON_JUDGE_PROMPT_CHARACTERS,
    ComparisonEvaluatorV1,
    ComparisonExecutionPolicyV1,
    ComparisonResultV3,
    ComparisonSpecV1,
    DecisionAttestationV1,
    _apply_decision_attestation,
    _approved_comparison_execution_lock,
    _bind_local_execution_evidence,
    _canonical_decision_gate_policies,
    _comparison_qualification_digest,
    _comparison_result_digest,
    _comparison_trial_output,
    _evaluate_decision,
    _evaluator_digest,
    _mechanism_summary,
    _normalized_reported_project_identity,
    _paired_attempt_v3,
    _paired_attempt_view_v3,
    _prepare_comparison_scorer_runtimes,
    _request_comparison_judge,
    _require_checkpoint_evaluations,
    _require_checkpoint_judges,
    _restore_or_verify_checkpoint_receipt,
    _result_candidate_definitions,
    _result_markdown,
    _resume_approved_comparison_lock,
    _safe_comparison_score_details,
    _score_deterministic_output,
    _scorer_revisions_v3,
    analyze_comparison_rows,
    check_comparison,
    claim_comparison_approval,
    comparison_from_dict,
    compile_comparison,
    execute_comparison,
    load_comparison,
    materialize_comparison,
    prepare_comparison,
    prepared_candidate_definitions,
    preview_comparison,
    read_comparison_result,
    scaffold_comparison,
    score_comparison_rows,
    write_comparison_result,
)
from fugue.bench.comparison import _decision_policy as parse_decision_policy
from fugue.bench.evaluations import JudgeResponseError
from fugue.bench.execution_recovery import ExecutionFinalizationPending
from fugue.bench.local_evidence import (
    local_result_row_projection_digest,
    local_result_row_projection_v1,
)
from fugue.bench.operator import OperatorService
from fugue.model_plane import EvidenceMode, trace_destination_identity
from fugue.research.approvals import ApprovalLedger
from fugue.research.contracts import ResearchError
from fugue.research.database import connect_database
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
MCP_MAINTENANCE_EXAMPLE = Path("examples/comparisons/wandb-mcp-maintenance")


def test_new_programmatic_comparison_policy_defaults_to_local_evidence() -> None:
    policy = ComparisonExecutionPolicyV1(
        model="anthropic/claude-sonnet-5",
        harnesses=("claude-code",),
        attempts=1,
        concurrency=1,
        max_cost_usd=1.0,
        reserve_per_attempt_usd=1.0,
        approval_required=True,
        trace_content="full",
    )

    assert policy.evidence_mode == "local"
    assert policy.evidence_destination.kind == "local"


def test_judge_response_contract_is_explicit_and_legacy_default_is_omitted() -> None:
    legacy = ComparisonEvaluatorV1(
        id="legacy-review",
        type="llm_judge",
        required=False,
        profile="wandb/anthropic/claude-opus-4-8",
        rubric="Assess usefulness.",
        dimensions=("usefulness",),
    )
    anchored = replace(legacy, response_contract="anchored_review_v1")

    assert "response_contract" not in legacy.to_dict()
    assert anchored.to_dict()["response_contract"] == "anchored_review_v1"


def test_explicit_local_comparison_policy_cannot_silently_switch_to_weave() -> None:
    with pytest.raises(ValueError, match="W&B result destination"):
        ComparisonExecutionPolicyV1(
            model="anthropic/claude-sonnet-5",
            harnesses=("claude-code",),
            attempts=1,
            concurrency=1,
            max_cost_usd=1.0,
            reserve_per_attempt_usd=1.0,
            approval_required=True,
            trace_content="full",
            evidence_mode="local",
            evidence_project="wandb/should-not-activate",
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
    assert set(preview.matrix["candidate_definitions"]) == {
        str(item["candidate_id"]) for item in preview.matrix["matrix_cells"]
    }
    assert all(
        stable_digest(definition) == candidate_id
        for candidate_id, definition in preview.matrix["candidate_definitions"].items()
    )
    assert reparsed == spec
    assert isinstance(preview.comparison["evaluators"], list)
    assert isinstance(preview.comparison["execution"]["harnesses"], list)


def test_resume_reads_original_approved_lock_without_rewriting_it(
    tmp_path: Path,
) -> None:
    root = Path.cwd()
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=root)
    preview = preview_comparison(spec, repo_root=root)
    _, _, public_rows = compile_comparison(spec, repo_root=root)
    approved = _approved_comparison_execution_lock(
        preview,
        approval_digest="a" * 64,
        repo_root=root,
        public_rows=public_rows,
    )
    run_id = "resume-original-lock"
    snapshot = {
        "schema_version": 1,
        "run_id": run_id,
        "request": {"approved_comparison": approved},
        "snapshot_sha256": "",
        "lock_sha256": "",
    }
    digest = stable_digest(snapshot)
    snapshot["snapshot_sha256"] = digest
    snapshot["lock_sha256"] = digest
    path = tmp_path / ".fugue" / "runtime" / run_id / "input-lock.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(snapshot), encoding="utf-8")

    assert (
        _resume_approved_comparison_lock(
            repo_root=tmp_path,
            run_id=run_id,
            preview=preview,
        )
        == approved
    )

    changed = dict(snapshot)
    changed["request"] = {
        "approved_comparison": {**approved, "approval_digest": "b" * 64}
    }
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="input lock digest"):
        _resume_approved_comparison_lock(
            repo_root=tmp_path,
            run_id=run_id,
            preview=preview,
        )
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
        candidate_definitions=preview.matrix["candidate_definitions"],
    )

    assert approval.candidate_definitions == preview.matrix["candidate_definitions"]

    claim_comparison_approval(
        preview,
        approval_digest=approval.approval_digest,
        repo_root=tmp_path,
    )


def test_exact_approval_rejects_a_different_candidate_map(tmp_path: Path) -> None:
    root = Path.cwd()
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=root)
    preview = preview_comparison(
        replace(spec, execution=replace(spec.execution, attempts=1)),
        repo_root=root,
    )
    definitions = dict(preview.matrix["candidate_definitions"])
    one_candidate_id = next(iter(definitions))
    wrong_definition = {**definitions[one_candidate_id], "label": "other candidate"}
    definitions.pop(one_candidate_id)
    definitions[stable_digest(wrong_definition)] = wrong_definition
    ledger = ApprovalLedger(StudyStore(tmp_path).path)
    approval = ledger.approve(
        subject_kind="experiment",
        preview_digest=preview.preview_digest,
        maximum_cost_usd=1,
        maximum_cells=8,
        approved_by="test-reviewer",
        operation_id="approve-wrong-candidate-map",
        candidate_definitions=definitions,
    )

    with pytest.raises(ResearchError, match="accepted candidates"):
        claim_comparison_approval(
            preview,
            approval_digest=approval.approval_digest,
            repo_root=tmp_path,
        )


def test_prepared_preview_exposes_exact_candidate_map_for_approval(
    tmp_path: Path,
) -> None:
    comparison_path = scaffold_comparison(tmp_path)
    spec = load_comparison(comparison_path, repo_root=tmp_path)
    preview = preview_comparison(spec, repo_root=tmp_path)
    retained = (
        tmp_path
        / COMPARISON_RUNTIME_ROOT
        / spec.spec_digest
        / "prepared"
        / "prepared-preview.json"
    )
    retained.parent.mkdir(parents=True)
    retained.write_text(json.dumps(preview.to_dict()), encoding="utf-8")

    definitions = prepared_candidate_definitions(
        preview.preview_digest,
        repo_root=tmp_path,
    )

    assert definitions == preview.matrix["candidate_definitions"]
    assert all(stable_digest(value) == key for key, value in definitions.items())

    retained.unlink()
    with pytest.raises(ValueError, match="exact prepared preview is unavailable"):
        prepared_candidate_definitions(preview.preview_digest, repo_root=tmp_path)


def test_approval_ledger_supports_exact_preview_renewal_and_migration(
    tmp_path: Path,
) -> None:
    database = StudyStore(tmp_path).path
    ledger = ApprovalLedger(database)
    first = ledger.approve(
        subject_kind="experiment",
        preview_digest="a" * 64,
        maximum_cost_usd=10,
        maximum_cells=8,
        approved_by="operator",
        operation_id="initial-exact-approval",
    )
    assert (
        ledger.approve(
            subject_kind="experiment",
            preview_digest="a" * 64,
            maximum_cost_usd=10,
            maximum_cells=8,
            approved_by="operator",
            operation_id="initial-exact-approval",
        )
        == first
    )
    with pytest.raises(ResearchError, match="operation id"):
        ledger.approve(
            subject_kind="experiment",
            preview_digest="a" * 64,
            maximum_cost_usd=11,
            maximum_cells=8,
            approved_by="operator",
            operation_id="initial-exact-approval",
        )

    # Reproduce the pre-renewal schema and prove initialization migrates it
    # without deleting the existing immutable receipt.
    with connect_database(database) as conn:
        conn.executescript(
            """
            DROP INDEX IF EXISTS approval_subject_lookup;
            CREATE UNIQUE INDEX approval_subject
                ON execution_approvals(subject_kind, preview_digest);
            """
        )
    ledger = ApprovalLedger(database)
    renewal = ledger.approve(
        subject_kind="experiment",
        preview_digest="a" * 64,
        maximum_cost_usd=10,
        maximum_cells=8,
        approved_by="operator",
        operation_id="renew-exact-approval",
    )
    assert renewal.approval_digest != first.approval_digest
    assert renewal.approval_id != first.approval_id
    assert ledger.get(first.approval_digest) == first
    assert (
        ledger.get_for_preview(subject_kind="experiment", preview_digest="a" * 64)
        == renewal
    )

    ledger.claim(
        approval_digest=renewal.approval_digest,
        subject_kind="experiment",
        preview_digest="a" * 64,
        subject_id="comparison-run-renewed",
        estimated_cells=8,
        estimated_cost_usd=10,
    )
    with pytest.raises(ResearchError, match="already consumed"):
        ledger.claim(
            approval_digest=renewal.approval_digest,
            subject_kind="experiment",
            preview_digest="a" * 64,
            subject_id="comparison-run-other",
            estimated_cells=8,
            estimated_cost_usd=10,
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
    spec = load_comparison(LIVE_SKILL_EXAMPLE / "comparison.yaml", repo_root=root)

    readiness = check_comparison(spec, repo_root=root)
    preview = preview_comparison(spec, repo_root=root)

    assert readiness.status == "ready"
    assert readiness.task_count == 8
    assert readiness.base_failures == 8
    assert readiness.gold_passes == 8
    assert readiness.actual_changes == ("skills",)
    assert readiness.estimated_cells == 32
    assert preview.matrix["estimated_trials"] == 32
    assert {cell["harness"] for cell in preview.matrix["matrix_cells"]} == {"codex"}


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


def test_evidence_checkpoint_compiles_serial_execution_schedule() -> None:
    root = Path.cwd()
    raw = yaml.safe_load((EXAMPLE / "comparison.yaml").read_text())
    raw["execution"]["evidence_checkpoint_cells"] = 1
    raw["execution"]["concurrency"] = 2

    spec = comparison_from_dict(raw, repo_root=root, source=EXAMPLE)
    preview = preview_comparison(spec, repo_root=root)
    schedule = preview.execution_schedule["schedule"]
    checkpoint = [
        item
        for item in schedule["logical_attempts"]
        if item["stage_id"] == "checkpoint"
    ]

    assert len(checkpoint) == 2
    assert [item["block_ordinal"] for item in checkpoint] == [0, 1]
    assert all(item["attempt_ordinal"] == 0 for item in checkpoint)


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
    assert set(scored[0]["comparison_score_details"]) == {
        "answer_present",
        "expected_values",
    }
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
    assert first_preview.experiment["evidence_project"] == ("wandb/fugue-comparison-a")
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
    raw["execution"]["evidence_project"] = "wandb/fugue-mcp-release-qualification-v1"
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
    env_file.chmod(0o600)
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


def test_comparison_defaults_to_canonical_local_evidence(tmp_path: Path) -> None:
    comparison_path = scaffold_comparison(tmp_path)
    raw = yaml.safe_load(comparison_path.read_text())
    assert raw["schema_version"] == 2
    raw["execution"].pop("evidence_mode")
    spec = comparison_from_dict(raw, repo_root=tmp_path, source=tmp_path)

    assert spec.execution.evidence_mode == "local"
    assert spec.execution.evidence_project is None
    assert spec.execution.evidence_destination is not None
    assert spec.execution.evidence_destination.to_dict()["kind"] == "local"

    experiment, _manifest, _rows = compile_comparison(spec, repo_root=tmp_path)
    assert experiment.evidence_mode == "local"
    assert experiment.agent_env["FUGUE_EVIDENCE_MODE"] == "local"
    assert experiment.evidence_project is None

    raw["execution"]["evidence_mode"] = "local"
    raw["execution"]["source_evidence_project"] = "wandb/source-project"
    raw["execution"]["source_evidence_destination"] = {
        "schema_version": 1,
        "entity": "wandb",
        "project": "source-project",
        "api_base_url": "https://api.wandb.ai",
        "trace_base_url": "https://trace.wandb.ai",
        "app_base_url": "https://wandb.ai",
    }
    source_backed = comparison_from_dict(
        raw,
        repo_root=tmp_path,
        source=tmp_path,
    )
    assert source_backed.execution.evidence_mode == "local"
    assert source_backed.execution.source_evidence_project == ("wandb/source-project")
    assert source_backed.execution.evidence_project is None
    assert source_backed.execution.evidence_destination is not None
    assert source_backed.execution.evidence_destination.to_dict()["kind"] == ("local")
    source_experiment, _manifest, _rows = compile_comparison(
        source_backed,
        repo_root=tmp_path,
    )
    assert source_experiment.evidence_mode == "local"
    assert source_experiment.source_evidence_project == "wandb/source-project"
    assert source_experiment.evidence_project is None
    assert source_experiment.agent_env["FUGUE_EVIDENCE_MODE"] == "local"
    assert source_experiment.agent_env["FUGUE_SOURCE_EVIDENCE_PROJECT"] == (
        "wandb/source-project"
    )

    raw["execution"]["evidence_mode"] = "local"
    raw["execution"]["evidence_project"] = "wandb/not-local"
    with pytest.raises(ValueError, match="does not accept W&B"):
        comparison_from_dict(raw, repo_root=tmp_path, source=tmp_path)


def test_v1_comparison_without_evidence_mode_retains_weave_default(
    tmp_path: Path,
) -> None:
    comparison_path = scaffold_comparison(tmp_path)
    raw = yaml.safe_load(comparison_path.read_text())
    raw["schema_version"] = 1
    raw["execution"].pop("evidence_mode")
    raw["execution"]["evidence_project"] = "wandb/legacy-results"

    spec = comparison_from_dict(raw, repo_root=tmp_path, source=tmp_path)

    assert spec.execution.evidence_mode == "weave_required"
    assert spec.execution.evidence_project == "wandb/legacy-results"


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
        "trace_receipt": trace_destination_identity({"FUGUE_WEAVE_PROJECT": project}),
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
                    "git:" + ("5" if variant == "candidate" else "3") * 40
                ),
                "runtime_digest": (
                    "sha256:" + ("6" if variant == "candidate" else "4") * 64
                ),
                "lock_digest": (
                    "sha256:" + ("7" if variant == "candidate" else "2") * 64
                ),
            }
        ],
        "weave_tool_names": {
            "mcp__wandb__query_wandb_tool": 3,
            "Read": 2,
        },
        "evaluation_id": f"evaluation-{task_id}",
        "weave_evaluation_root_call_id": f"{call_prefix}-evaluation",
        "weave_evaluation_root_ref": call_ref(f"{call_prefix}-evaluation"),
        "evaluation_url": f"{base_url}/evaluations/{task_id}",
        "evaluation_root_object_verified": True,
        "evaluation_root_dataset_relationship_verified": True,
        "evaluation_root_prediction_relationship_verified": True,
        "dataset_id": ("weave:///wandb/release-project/object/release-dataset:v1"),
        "dataset_url": f"{base_url}/objects/release-dataset/versions/v1",
        "dataset_version_object_verified": True,
        "eval_predict_and_score_call_id": f"{call_prefix}-eval",
        "eval_predict_and_score_ref": call_ref(f"{call_prefix}-eval"),
        "eval_predict_and_score_url": f"{base_url}/call/{call_prefix}-eval",
        "eval_predict_and_score_object_verified": True,
        "prediction_call_id": f"{call_prefix}-prediction",
        "weave_prediction_ref": call_ref(f"{call_prefix}-prediction"),
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


def _local_harbor_decision_rows() -> list[dict[str, object]]:
    rows = [
        _decision_row(variant="baseline", passed=True),
        _decision_row(variant="candidate", passed=True),
    ]
    for row in rows:
        row.pop("infrastructure_conformance_complete")
        row["trace_project"] = None
        row["local_evidence_links"] = [{"system": "local_artifact"}]
        row["hosted_evidence_privacy_scan_status"] = "not_applicable"
    return rows


def _evaluate_local_harbor_decision(
    rows: list[dict[str, object]],
    *,
    evidence_mode: EvidenceMode,
):
    return _evaluate_decision(
        policy=parse_decision_policy(_decision_policy()),
        rows=rows,
        deterministic={"candidate": {"passed": 1, "evaluated": 1}},
        operational={"infrastructure_failures": 0},
        improved=0,
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
        evidence_mode=evidence_mode,
    )


def test_local_decision_uses_one_shared_harbor_receipt_without_hosted_gate() -> None:
    rows = _local_harbor_decision_rows()

    decision = _evaluate_local_harbor_decision(rows, evidence_mode="local")

    gates = {gate.id: gate for gate in decision.gates}
    assert decision.status == "ready_for_signoff"
    assert "hosted-evidence-privacy" not in gates
    assert gates["infrastructure"].status == "passed"
    assert gates["sandbox-cleanup"].status == "passed"

    hosted = _evaluate_local_harbor_decision(rows, evidence_mode="weave_required")
    hosted_gates = {gate.id: gate for gate in hosted.gates}
    assert hosted.status == "blocked"
    assert hosted_gates["hosted-evidence-privacy"].status == "unavailable"


def test_local_harbor_conformance_rejects_mixed_or_empty_run_receipts() -> None:
    mismatched = _local_harbor_decision_rows()
    mismatched[1]["harbor_conformance_receipt_digest"] = "e" * 64
    for row in mismatched:
        # A generic receipt flag cannot bypass the stricter run-wide Harbor
        # receipt contract.
        row["infrastructure_conformance_complete"] = True

    mismatch_decision = _evaluate_local_harbor_decision(
        mismatched,
        evidence_mode="local",
    )
    mismatch_gates = {gate.id: gate for gate in mismatch_decision.gates}
    assert mismatch_decision.status == "blocked"
    assert mismatch_gates["infrastructure"].status == "unavailable"
    assert mismatch_gates["sandbox-cleanup"].status == "unavailable"

    missing = _local_harbor_decision_rows()
    for row in missing:
        row["harbor_conformance_receipt_digest"] = ""
    missing_decision = _evaluate_local_harbor_decision(
        missing,
        evidence_mode="local",
    )
    missing_gates = {gate.id: gate for gate in missing_decision.gates}
    assert missing_gates["infrastructure"].status == "unavailable"
    assert missing_gates["sandbox-cleanup"].status == "unavailable"


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

    assert (
        sum(
            item.id == "release-note-infrastructure-bounded-timeout"
            for item in decision.gates
        )
        == 1
    )


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
    raw["execution"]["evidence_mode"] = "weave_required"
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
        row["comparison_judge_scores"] = {
            "maintainer-actionability.maintenance_actionability": (
                0.8 if variant == "candidate" else 0.6
            )
        }
        row["comparison_judges"] = {
            "maintainer-actionability": {
                "qualification": {
                    "judge_id": "maintainer-actionability",
                    "profile": "wandb/zai-org/GLM-5.2",
                    "contract_digest": "7" * 64,
                    "dimensions": ["maintenance_actionability"],
                    "calibration": {
                        "status": "pending_human_review",
                        "report_sha256": "8" * 64,
                        "cases_digest": "9" * 64,
                        "passed": False,
                    },
                }
            }
        }
        row["source_pre_run_drift"] = drift
        row["source_checkpoint_drift"] = drift
        row["source_post_run_drift"] = drift
        row["prediction_id"] = f"{variant}-prediction-row"
        row["agent_response"] = {
            "project": source_project,
            "answer": "safe maintainer summary",
        }
        row["usage"] = {"input_tokens": 100, "output_tokens": 25}
        row["cost_reconciliation_status"] = "resolved"
        row["latency_reconciliation_status"] = "resolved"
        row["usage_reconciliation_status"] = "resolved"
        call_prefix = f"{variant}-{cell['trial_index']}"
        row["evaluation_url"] = f"{result_base}/calls/{call_prefix}-evaluation"
        row["weave_evaluation_root_call_id"] = f"{call_prefix}-evaluation"
        row["weave_evaluation_root_ref"] = (
            f"weave:///wandb/result-project/call/{call_prefix}-evaluation"
        )
        row["dataset_url"] = f"{result_base}/objects/release-dataset/versions/v1"
        row["weave_dataset_id"] = (
            "weave:///wandb/result-project/object/release-dataset:v1"
        )
        row["eval_predict_and_score_url"] = f"{result_base}/calls/{call_prefix}-eval"
        row["eval_predict_and_score_call_id"] = f"{call_prefix}-eval"
        row["eval_predict_and_score_ref"] = (
            f"weave:///wandb/result-project/call/{call_prefix}-eval"
        )
        row["prediction_url"] = f"{result_base}/calls/{call_prefix}-prediction"
        row["prediction_call_id"] = f"{call_prefix}-prediction"
        row["weave_prediction_ref"] = (
            f"weave:///wandb/result-project/call/{call_prefix}-prediction"
        )
        row["agent_url"] = f"{result_base}/calls/{call_prefix}-agent"
        row["native_agent_root_call_id"] = f"{call_prefix}-agent"
        row["weave_agent_root_ref"] = (
            f"weave:///wandb/result-project/call/{call_prefix}-agent"
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
    assert result.evidence_topology.source_destination.project_slug == (source_project)
    assert result.evidence_topology.result_destination.project_slug == (result_project)
    assert result.behavioral_summary.status == "improved"
    assert result.supersedes[0].result_digest == "6" * 64
    assert result.task_validity[0].status == "valid"
    assert result.paired_cases[0].dimension_changes[0].role == "outcome"
    assert result.paired_cases[0].candidate.actual_query_scope == (source_project,)
    assert result.paired_cases[0].candidate.cost_reconciliation_status == "resolved"
    assert result.paired_cases[0].candidate.latency_reconciliation_status == "resolved"
    assert result.paired_cases[0].candidate.usage_reconciliation_status == "resolved"
    projected_view = build_comparison_evaluation_view(result.to_dict())
    projected_candidate = projected_view.paired_cases[0]["candidate"]
    assert projected_candidate["cost_reconciliation_status"] == "resolved"
    assert projected_candidate["latency_reconciliation_status"] == "resolved"
    assert projected_candidate["usage_reconciliation_status"] == "resolved"

    dual_chain_rows = json.loads(json.dumps(rows))
    local_binding = {
        "local_evidence_manifest_digest": "1" * 64,
        "local_evidence_manifest_file_sha256": "2" * 64,
        "local_evidence_plan_digest": "3" * 64,
        "local_evidence_attempt_record_set_digest": "4" * 64,
        "local_evidence_prediction_row_set_digest": "5" * 64,
        "local_evidence_run_receipt_digest": "6" * 64,
        "local_evidence_run_receipt_file_sha256": "7" * 64,
    }
    for index, row in enumerate(dual_chain_rows, start=1):
        attempt = str(row["attempt_id"])
        row["run_id"] = "v3-run"
        row["local_evidence_record_digest"] = f"{index:x}" * 64
        row["local_evidence_prediction_row_sha256"] = f"{index + 4:x}" * 64
        row.update(local_binding)
        row["local_evidence_links"] = [
            {
                "kind": kind,
                "status": "resolved",
                "system": "local_artifact",
                "ref": f"fugue://evidence/{attempt}/{kind}",
            }
            for kind in (
                "evaluation_root",
                "prediction_and_score",
                "prediction",
                "agent_root",
                "dataset",
            )
        ]
    dual_chain = analyze_comparison_rows(
        comparison_id=spec.id,
        preview_digest=preview.preview_digest,
        rows=dual_chain_rows,
        source="v3-run",
        expected_evidence_project=result_project,
        expected_source_evidence_project=source_project,
        approved_comparison=approved,
        result_schema_version=3,
        study_intent="mcp_release_maintenance",
        supersedes=spec.supersedes,
    )
    assert isinstance(dual_chain, ComparisonResultV3)
    assert dual_chain.evidence_backend == "weave"
    assert dual_chain.local_evidence is not None
    assert dual_chain.local_evidence["run_id"] == "v3-run"
    assert dual_chain.local_chain_integrity == "reconciled"
    assert dual_chain.hosted_chain_integrity == "reconciled"
    assert {
        item.system for item in dual_chain.paired_cases[0].candidate.evidence_links
    } == {"local_artifact"}
    assert {
        item.system
        for item in dual_chain.paired_cases[0].candidate.hosted_evidence_links
    } == {"weave"}
    local_markdown = _result_markdown(dual_chain)
    assert "Evaluation record:" in local_markdown
    assert "Prediction-and-score record:" in local_markdown
    assert "Prediction record:" in local_markdown
    assert "Provider-neutral Agent receipt:" in local_markdown
    assert "Dataset manifest:" in local_markdown
    assert "## Treatment identities" in local_markdown
    for candidate_id, definition in dual_chain.candidate_definitions.items():
        assert f"`{candidate_id}`" in local_markdown
        assert f"harness `{definition['harness']}`" in local_markdown
        assert f"model `{definition['model_route']['display_model']}`" in local_markdown
        assert f"context `{definition['context']['id']}`" in local_markdown
        for skill in definition["skills"]:
            assert f"`{skill['id']}`" in local_markdown
        for integration in definition["integrations"]:
            assert f"`{integration['id']}`" in local_markdown
        if definition.get("prompt_digest"):
            assert f"prompt `{definition['prompt_digest']}`" in local_markdown

    dual_chain_destination = tmp_path / "v3-dual-chain-result"
    dual_chain_destination.mkdir()
    (dual_chain_destination / "attempts.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in dual_chain_rows) + "\n",
        encoding="utf-8",
    )
    write_comparison_result(dual_chain, destination=dual_chain_destination)
    assert read_comparison_result(dual_chain_destination / "result.json") == dual_chain

    role_drift = json.loads(json.dumps(rows))
    candidate_row = next(row for row in role_drift if row["variant_id"] == "candidate")
    candidate_row["comparison_dimension_roles"]["release.factual_correctness"] = (
        "mechanism"
    )
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

    missing_local_binding = dual_chain.to_dict()
    missing_local_binding.pop("local_evidence")
    with pytest.raises(ValueError, match="local evidence requires its ledger binding"):
        read_comparison_result(
            write_rehashed_result(
                missing_local_binding,
                "missing-dual-chain-local-binding.json",
            )
        )

    missing_attempt_binding = dual_chain.to_dict()
    missing_attempt_binding["paired_cases"][0]["candidate"].pop(
        "local_evidence_record_digest"
    )
    with pytest.raises(ValueError, match="requires record and prediction digests"):
        read_comparison_result(
            write_rehashed_result(
                missing_attempt_binding,
                "missing-dual-chain-attempt-binding.json",
            )
        )

    expected_candidate_ids = {
        str(attempt.identity["candidate"])
        for pair in result.paired_cases
        for attempt in (pair.baseline, pair.candidate)
        if attempt is not None
    }
    assert set(result.candidate_definitions) == expected_candidate_ids
    assert all(
        stable_digest(definition) == candidate_id
        for candidate_id, definition in result.candidate_definitions.items()
    )

    missing_candidate_definitions = result.to_dict()
    missing_candidate_definitions.pop("candidate_definitions")
    with pytest.raises(
        ValueError, match="missing required field.*candidate_definitions"
    ):
        read_comparison_result(
            write_rehashed_result(
                missing_candidate_definitions,
                "missing-candidate-definitions.json",
            )
        )

    empty_candidate_definitions = result.to_dict()
    empty_candidate_definitions["candidate_definitions"] = {}
    with pytest.raises(ValueError, match="nonempty candidate definitions"):
        read_comparison_result(
            write_rehashed_result(
                empty_candidate_definitions,
                "empty-candidate-definitions.json",
            )
        )

    incomplete_candidate_definitions = result.to_dict()
    incomplete_candidate_definitions["candidate_definitions"].pop(
        next(iter(expected_candidate_ids))
    )
    with pytest.raises(ValueError, match="do not cover attempts"):
        read_comparison_result(
            write_rehashed_result(
                incomplete_candidate_definitions,
                "incomplete-candidate-definitions.json",
            )
        )

    forged_identity = result.to_dict()
    forged_identity["paired_cases"][0]["candidate"]["identity"]["candidate"] = (
        "forged-candidate"
    )
    with pytest.raises(ValueError, match="attempt identity disagrees"):
        read_comparison_result(
            write_rehashed_result(
                forged_identity,
                "forged-attempt-identity.json",
            )
        )

    incomplete_evidence_chain = result.to_dict()
    incomplete_evidence_chain["paired_cases"][0]["candidate"]["evidence_links"].pop()
    with pytest.raises(ValueError, match="exactly five unique evidence links"):
        read_comparison_result(
            write_rehashed_result(
                incomplete_evidence_chain,
                "incomplete-evidence-chain.json",
            )
        )

    wrong_evidence_status = result.to_dict()
    wrong_evidence_status["paired_cases"][0]["candidate"]["evidence_status"] = "missing"
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
        for item in wrong_call_route["paired_cases"][0]["candidate"]["evidence_links"]
        if item["kind"] == "evaluation_root"
    )
    evaluation_link["ref"] = "weave:///wandb/other-project/call/forged"
    evaluation_link["url"] = "https://wandb.ai/wandb/other-project/weave/calls/forged"
    with pytest.raises(ValueError, match="result topology"):
        read_comparison_result(
            write_rehashed_result(
                wrong_call_route,
                "wrong-call-route.json",
            )
        )

    wrong_query_scope = result.to_dict()
    wrong_query_scope["paired_cases"][0]["candidate"]["actual_query_scope"] = [
        source_project,
        "wandb/other-project",
    ]
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
    contradictory_pair["qualification_digest"] = _comparison_qualification_digest(
        contradictory_pair
    )
    contradictory_pair["result_digest"] = contradictory_pair["qualification_digest"]
    contradictory_pair_path = destination / "contradictory-pair-result.json"
    contradictory_pair_path.write_text(
        json.dumps(contradictory_pair),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pair status disagrees"):
        read_comparison_result(contradictory_pair_path)

    contradictory_behavior = result.to_dict()
    contradictory_behavior["behavioral_summary"]["status"] = "regressed"
    contradictory_behavior["qualification_digest"] = _comparison_qualification_digest(
        contradictory_behavior
    )
    contradictory_behavior["result_digest"] = contradictory_behavior[
        "qualification_digest"
    ]
    contradictory_behavior_path = destination / "contradictory-behavior-result.json"
    contradictory_behavior_path.write_text(
        json.dumps(contradictory_behavior),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="behavioral status disagrees"):
        read_comparison_result(contradictory_behavior_path)

    unknown_decision_field = result.to_dict()
    unknown_decision_field["decision"]["invented"] = True
    unknown_decision_field["qualification_digest"] = _comparison_qualification_digest(
        unknown_decision_field
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
        f"{view.evidence_scope.entity}/{view.evidence_scope.project}" == result_project
    )
    assert view.paired_cases[0]["candidate"]["evidence_links"][0]["url"].startswith(
        "https://wandb.ai/wandb/result-project/weave/calls/"
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
    running_attempt["paired_cases"][0]["candidate"]["execution_status"] = "running"
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
    unknown_coverage["qualification_digest"] = _comparison_qualification_digest(
        unknown_coverage
    )
    unknown_coverage["result_digest"] = unknown_coverage["qualification_digest"]
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
    rejected_actionability["decision"]["attestation"]["review_status"] = "rejected"
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
            key: True for key in row["comparison_deterministic_scores"]
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
    assert non_discriminating.task_validity[0].status == ("non_discriminating")
    assert non_discriminating.task_validity[0].blockers == ()
    assert non_discriminating.behavioral_summary.status == "unchanged"
    assert non_discriminating.behavioral_summary.improved_pairs == 0
    assert "release" not in non_discriminating.behavioral_summary.recommendation.lower()
    assert non_discriminating.behavioral_summary.supported_claim is not None
    assert (
        "release" not in non_discriminating.behavioral_summary.supported_claim.lower()
    )


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
    wrong_signed_digest["decision"]["attestation"]["signed_result_digest"] = "0" * 64
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
        result.paired_cases[0].baseline.infrastructure[
            "private_label_boundary_verified"
        ]
        is True
    )
    assert (
        "label_boundary_verified" not in result.paired_cases[0].baseline.infrastructure
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
    assert (
        next(
            link
            for link in candidate_attempt.evidence_links
            if link.kind == "agent_root"
        ).status
        == "invalid"
    )
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
    assert (
        not {
            "cells",
            "arm_totals",
            "aligned_comparisons",
            "behavioral_measures",
            "mechanism_funnel",
            "outcome_summaries",
            "score_summaries",
        }
        & view.to_dict().keys()
    )


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
        gate.id for gate in result.decision.gates if gate.status == "unavailable"
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


def test_supplied_attempt_identity_drift_is_rejected_before_normalization() -> None:
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
    assert approved["candidate_definitions"] == preview.matrix["candidate_definitions"]
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
        cell for cell in drifted["expected_cells"] if cell["variant_id"] == "baseline"
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
    with pytest.raises(
        ValueError,
        match=(
            "candidate definitions|per-arm identity drift|"
            "execution schedule does not cover exact cells"
        ),
    ):
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
        {key: value for key, value in source_required.items() if key != "lock_digest"}
    )
    with pytest.raises(ValueError, match="no approved candidate source revision"):
        analyze_comparison_rows(
            comparison_id=spec.id,
            preview_digest=preview.preview_digest,
            rows=[{**row, "approved_comparison": source_required} for row in rows],
            source="approved-run",
            approved_comparison=source_required,
        )

    private_digest = approved["approved_inputs"]["private_labels_sha256"]
    frozen_private = (
        root / ".fugue/private/comparison-inputs/labels" / f"{private_digest}.jsonl"
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
        "\n".join(json.dumps(row, sort_keys=True) for row in changed_rows) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        RuntimeError,
        match="disagrees with the final exported attempt rows",
    ):
        write_comparison_result(result, destination=destination)


def test_v3_candidate_definition_recompute_requires_exact_coverage() -> None:
    baseline_definition = {"arm": "baseline", "prompt_digest": "a" * 64}
    candidate_definition = {"arm": "candidate", "prompt_digest": "b" * 64}
    baseline_id = stable_digest(baseline_definition)
    candidate_id = stable_digest(candidate_definition)
    rows = [
        {"candidate_id": baseline_id},
        {"candidate_id": candidate_id},
    ]
    execution_lock = {
        "candidate_definitions": {
            baseline_id: baseline_definition,
            candidate_id: candidate_definition,
        }
    }

    assert (
        _result_candidate_definitions(
            rows,
            execution_lock=execution_lock,
        )
        == execution_lock["candidate_definitions"]
    )

    with pytest.raises(ValueError, match="nonempty candidate definitions"):
        _result_candidate_definitions(rows, execution_lock=None)

    with pytest.raises(ValueError, match="do not cover all result candidates"):
        _result_candidate_definitions(
            rows,
            execution_lock={
                "candidate_definitions": {
                    baseline_id: baseline_definition,
                }
            },
        )

    with pytest.raises(ValueError, match="candidate identity is required"):
        _result_candidate_definitions(
            [{"candidate_id": ""}],
            execution_lock=execution_lock,
        )


def test_v3_treatment_identity_markdown_uses_stored_definition_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fugue.bench import comparison as comparison_module

    candidate_id = "d" * 64

    class StoredV3:
        candidate_definitions = {
            candidate_id: {
                "harness": "claude-code",
                "model_route": {"display_model": "anthropic/claude-sonnet-5"},
                "context": {"id": "repo-context-v1"},
                "skills": [{"id": "review-plan"}],
                "integrations": [{"id": "wandb-mcp-0-4"}],
                "prompt_digest": "e" * 64,
            }
        }

    monkeypatch.setattr(comparison_module, "ComparisonResultV3", StoredV3)

    markdown = comparison_module._result_treatment_identities_markdown(StoredV3())

    assert markdown == (
        "## Treatment identities\n\n"
        f"- `{candidate_id}` — harness `claude-code`; "
        "model `anthropic/claude-sonnet-5`; context `repo-context-v1`; "
        "skills `review-plan`; integrations `wandb-mcp-0-4`; "
        f"prompt `{'e' * 64}`\n\n"
    )


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


def test_v2_result_remains_an_explicit_compatibility_path(tmp_path: Path) -> None:
    rows = [
        _decision_row(variant="baseline", passed=False),
        _decision_row(variant="candidate", passed=True),
    ]
    result = analyze_comparison_rows(
        comparison_id="legacy-v2-result",
        preview_digest="a" * 64,
        rows=rows,
        source="legacy-v2",
        expected_evidence_project="wandb/release-project",
        result_schema_version=2,
    )
    raw = result.to_dict()
    assert raw["schema_version"] == 2
    assert "candidate_definitions" not in raw
    path = tmp_path / "result-v2.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    assert read_comparison_result(path) == result


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


def test_exact_answer_artifact_populates_safe_reported_project_projection() -> None:
    exact_output = (
        '```json\n{"source_project":"wandb/fugue-mcp-release-source-v2"}\n```'
    )
    reported = _normalized_reported_project_identity(exact_output)
    assert reported == "wandb/fugue-mcp-release-source-v2"

    scored_row = {
        "attempt_id": "a" * 64,
        "reported_project_identity": reported,
        # The normalized artifact value must win after final_output is removed.
        "agent_response": "I wrote the structured answer to fugue-answer.md.",
    }
    projection = local_result_row_projection_v1(scored_row)
    assert projection["reported_project_identity"] == reported

    assert _normalized_reported_project_identity(
        '{"source_project":"wandb/other-project"}'
    ) == ("wandb/other-project")
    assert (
        _normalized_reported_project_identity(
            '{"source_project":"https://wandb.ai/wandb/project?token=secret"}'
        )
        is None
    )
    malformed_projection = local_result_row_projection_v1(
        {
            "attempt_id": "b" * 64,
            "reported_project_identity": "not a project slug",
            "agent_response": "The answer artifact contained an invalid project.",
        }
    )
    assert malformed_projection["reported_project_identity"] is None


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
    reproduction = json.loads((destination / "reproduction.json").read_text())
    assert output == written
    assert output["source"] == "bundled-replay"
    assert output["rows"] == 16
    assert output["incomplete"] == 0
    assert reproduction["private_labels_included"] is False


def test_public_task_resources_are_digest_locked_into_the_task(tmp_path: Path) -> None:
    (tmp_path / "configs/fugue/skills/verify-current-source").mkdir(parents=True)
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
    assert (
        result.mechanism_summary["relevant_source_used"]["candidate"]["observed"] == 1
    )
    assert "task_passed" not in result.mechanism_summary

    historical = _mechanism_summary(
        [
            {
                "variant_id": "candidate",
                "comparison_mechanism": {
                    "skill_invoked": "observed",
                    "task_passed": "observed",
                },
            }
        ]
    )
    assert historical["skill_invoked"]["candidate"]["observed"] == 1
    assert "task_passed" not in historical


def test_eight_row_report_uses_real_pair_counts_and_neutral_study_language() -> None:
    rows = [
        _decision_row(variant=variant, task_id=f"task-{task}", passed=True)
        for task in range(1, 5)
        for variant in ("baseline", "candidate")
    ]
    result = analyze_comparison_rows(
        comparison_id="eight-row-behavior-study",
        preview_digest="a" * 64,
        rows=rows,
        source="test",
        expected_evidence_project="wandb/release-project",
    )
    historical_wording = replace(
        result.behavioral_summary,
        recommendation=(
            "HOLD — no release-qualifying behavioral improvement was established."
        ),
    )
    historical_mechanism = {
        **result.mechanism_summary,
        "task_passed": {
            "baseline": {"observed": 4, "applicable": 4, "unavailable": 0},
            "candidate": {"observed": 4, "applicable": 4, "unavailable": 0},
        },
    }

    markdown = _result_markdown(
        replace(
            result,
            behavioral_summary=historical_wording,
            mechanism_summary=historical_mechanism,
        )
    )

    assert "- Rows: 8" in markdown
    assert (
        "Baseline tasks that passed all required gates: 4/4 aligned task attempts"
        in markdown
    )
    assert (
        "Candidate tasks that passed all required gates: 4/4 aligned task attempts"
        in markdown
    )
    assert "Baseline full result" in markdown
    assert "Candidate full result" in markdown
    assert "Paired outcome change" in markdown
    assert "0/0" not in markdown
    assert "Task Passed" not in markdown
    assert (
        "Behavioral recommendation: HOLD — no release-qualifying behavioral "
        "improvement was established."
    ) in markdown
    assert "recommendation is preserved from the canonical result" in markdown
    assert "Package release: **NOT EVALUATED**" in markdown


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
        input_cost_per_million=1.0,
        output_cost_per_million=2.0,
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


def test_llm_judge_timeout_is_strict_defaulted_and_bound_to_spec() -> None:
    root = Path.cwd()
    path = MCP_MAINTENANCE_EXAMPLE / "tool-surface-canary-local-v4.yaml"
    raw = yaml.safe_load(path.read_text())
    configured = comparison_from_dict(
        raw,
        repo_root=root,
        source=path.parent,
    )
    configured_judge = next(
        evaluator
        for evaluator in configured.evaluators
        if evaluator.type == "llm_judge"
    )

    default_raw = yaml.safe_load(path.read_text())
    next(
        evaluator
        for evaluator in default_raw["evaluators"]
        if evaluator["type"] == "llm_judge"
    ).pop("timeout_sec")
    defaulted = comparison_from_dict(
        default_raw,
        repo_root=root,
        source=path.parent,
    )
    defaulted_judge = next(
        evaluator for evaluator in defaulted.evaluators if evaluator.type == "llm_judge"
    )

    assert configured_judge.timeout_sec == 300
    assert defaulted_judge.timeout_sec is None
    assert (
        next(
            evaluator
            for evaluator in configured.to_dict()["evaluators"]
            if evaluator["type"] == "llm_judge"
        )["timeout_sec"]
        == 300
    )
    assert "timeout_sec" not in next(
        evaluator
        for evaluator in defaulted.to_dict()["evaluators"]
        if evaluator["type"] == "llm_judge"
    )
    assert _evaluator_digest(
        configured_judge,
        root,
    ) != _evaluator_digest(defaulted_judge, root)
    assert configured.spec_digest != defaulted.spec_digest


@pytest.mark.parametrize("timeout_sec", [0, 901, 1.5, True, None])
def test_llm_judge_timeout_rejects_invalid_values(timeout_sec: object) -> None:
    root = Path.cwd()
    path = MCP_MAINTENANCE_EXAMPLE / "tool-surface-canary-local-v4.yaml"
    raw = yaml.safe_load(path.read_text())
    next(
        evaluator for evaluator in raw["evaluators"] if evaluator["type"] == "llm_judge"
    )["timeout_sec"] = timeout_sec

    with pytest.raises(ValueError, match="LLM judge timeout_sec"):
        comparison_from_dict(raw, repo_root=root, source=path.parent)


def test_deterministic_evaluator_rejects_judge_timeout() -> None:
    root = Path.cwd()
    path = MCP_MAINTENANCE_EXAMPLE / "tool-surface-canary-local-v4.yaml"
    raw = yaml.safe_load(path.read_text())
    next(
        evaluator
        for evaluator in raw["evaluators"]
        if evaluator["type"] == "deterministic"
    )["timeout_sec"] = 300

    with pytest.raises(
        ValueError,
        match="supported only for LLM judge evaluators",
    ):
        comparison_from_dict(raw, repo_root=root, source=path.parent)


def test_evaluator_dimension_guidance_is_public_and_identity_bound() -> None:
    root = Path.cwd()
    path = MCP_MAINTENANCE_EXAMPLE / "tool-surface-canary-local-v4.yaml"
    raw = yaml.safe_load(path.read_text())
    deterministic = next(
        evaluator
        for evaluator in raw["evaluators"]
        if evaluator["type"] == "deterministic"
    )
    deterministic["dimension_guidance"] = {
        "answer_correct": "Checks the final answer against host-only facts."
    }

    parsed = comparison_from_dict(raw, repo_root=root, source=path.parent)
    evaluator = next(item for item in parsed.evaluators if item.type == "deterministic")
    assert evaluator.dimension_guidance == {
        "answer_correct": "Checks the final answer against host-only facts."
    }
    original_digest = parsed.spec_digest
    deterministic["dimension_guidance"]["answer_correct"] = (
        "Checks factual correctness without exposing expected values."
    )
    assert (
        comparison_from_dict(raw, repo_root=root, source=path.parent).spec_digest
        != original_digest
    )

    deterministic["dimension_guidance"] = {"unknown": "Not declared."}
    with pytest.raises(ValueError, match="guidance may reference only"):
        comparison_from_dict(raw, repo_root=root, source=path.parent)


def test_legacy_comparison_omits_empty_dimension_guidance() -> None:
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=Path.cwd())

    assert all(
        "dimension_guidance" not in evaluator
        for evaluator in spec.to_dict()["evaluators"]
    )


def test_typed_deterministic_pass_ignores_non_performance_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fugue.bench import comparison as comparison_module

    evaluator = ComparisonEvaluatorV1(
        id="facts",
        type="deterministic",
        required=True,
        scorer="scorer.py",
        runtime="python312-sandbox-v1",
        dimensions=("answer", "safety", "mechanism", "infra", "efficiency"),
        dimension_roles={
            "answer": "outcome",
            "safety": "safety_gate",
            "mechanism": "mechanism",
            "infra": "infrastructure",
            "efficiency": "efficiency",
        },
    )
    details = {
        "answer": True,
        "safety": True,
        "mechanism": False,
        "infra": False,
        "efficiency": False,
    }
    monkeypatch.setattr(
        comparison_module,
        "_run_custom_scorer",
        lambda *_args, **_kwargs: {
            "score": 0.0,
            "reason": "legacy all-dimension aggregate",
            "details": details,
        },
    )

    passed, scores = _score_deterministic_output(
        task={},
        output={},
        expected={},
        evidence={},
        evaluators=(evaluator,),
        repo_root=Path.cwd(),
    )

    assert passed is True
    assert scores["facts.mechanism"] is False
    details["safety"] = False
    passed, _scores = _score_deterministic_output(
        task={},
        output={},
        expected={},
        evidence={},
        evaluators=(evaluator,),
        repo_root=Path.cwd(),
    )
    assert passed is False


def test_custom_scorer_rejects_partial_dimension_roles() -> None:
    root = Path.cwd()
    path = MCP_MAINTENANCE_EXAMPLE / "tool-surface-canary-local-v4.yaml"
    raw = yaml.safe_load(path.read_text())
    deterministic = next(
        evaluator
        for evaluator in raw["evaluators"]
        if evaluator["type"] == "deterministic"
    )
    deterministic["dimension_roles"].pop("release_mechanism_used")

    with pytest.raises(
        ValueError,
        match=(
            "dimension roles must cover every declared dimension; missing: "
            "release_mechanism_used"
        ),
    ):
        comparison_from_dict(raw, repo_root=root, source=path.parent)


def test_score_details_explain_answer_and_component_behavior_separately() -> None:
    evaluator = ComparisonEvaluatorV1(
        id="tool-surface",
        type="deterministic",
        required=True,
        scorer="scorer.py",
        runtime="python312-sandbox-v1",
        dimensions=("answer_correct", "target_behavior_satisfied"),
        dimension_roles={
            "answer_correct": "outcome",
            "target_behavior_satisfied": "mechanism",
        },
    )

    details = _safe_comparison_score_details(
        {
            "tool-surface.answer_correct": True,
            "tool-surface.target_behavior_satisfied": False,
        },
        evaluators=(evaluator,),
        row={"mcp_tool_calls": []},
    )

    answer = details["tool-surface.answer_correct"]
    behavior = details["tool-surface.target_behavior_satisfied"]
    assert "factually correct" in answer["what"]
    assert "does not require the component" in answer["what"]
    assert "host-only required facts" in answer["why"]
    assert "without the Agent diagnosing" in behavior["what"]
    assert "does not determine task pass" in behavior["why"]


def test_score_details_preserve_partial_numeric_score_as_failure() -> None:
    evaluator = ComparisonEvaluatorV1(
        id="facts",
        type="deterministic",
        required=True,
        scorer="scorer.py",
        runtime="python312-sandbox-v1",
        dimensions=("answer_correct",),
        dimension_roles={"answer_correct": "outcome"},
    )

    details = _safe_comparison_score_details(
        {"facts.answer_correct": 0.75},
        evaluators=(evaluator,),
        row={},
    )

    answer = details["facts.answer_correct"]
    assert answer["observed"] == (
        "The deterministic scorer recorded 0.75. This gate requires 1.0 to pass."
    )
    assert "blocks task pass" in answer["why"]


def test_score_details_keep_ambiguous_legacy_checks_backward_readable() -> None:
    evaluators = tuple(
        ComparisonEvaluatorV1(
            id=evaluator_id,
            type="deterministic",
            required=True,
            checks=("answer_present",),
        )
        for evaluator_id in ("first", "second")
    )

    details = _safe_comparison_score_details(
        {"answer_present": True},
        evaluators=evaluators,
        row={},
    )

    assert details["answer_present"]["what"] == (
        "Checks whether the Agent returned a non-empty final answer."
    )
    assert details["answer_present"]["observed"] == (
        "The Agent returned a non-empty final answer."
    )
    assert "legacy evaluator has no typed role" in details["answer_present"]["why"]


def test_scorer_revision_distinguishes_digest_only_source() -> None:
    revisions = _scorer_revisions_v3(
        {
            "scorer_digests": {"facts": "1" * 64},
            "approved_inputs": {
                "evaluator_artifacts": {
                    "facts": {"scorer_sha256": "2" * 64}
                }
            },
        }
    )

    assert revisions[0].details == {
        "kind": "scorer",
        "source_sha256": "2" * 64,
        "source_reference_status": "digest_only",
        "task_pass_roles": ["outcome", "safety_gate"],
    }


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
    assert attempt.passed is True
    assert (
        attempt.score_explanations[
            ("comparison.judge.maintainer-actionability.maintenance_actionability")
        ]
        == "Blind judge score; no rationale or private truth is published."
    )
    assert "score_details" not in attempt.to_dict()
    assert attempt.judge_reviews == {}


def test_anchored_judge_projects_safe_advisory_review_per_attempt() -> None:
    root = Path.cwd()
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=root)
    judge = ComparisonEvaluatorV1(
        id="maintainer-review",
        type="llm_judge",
        required=False,
        profile="wandb/anthropic/claude-opus-4-8",
        rubric="Assess whether the response is useful and actionable.",
        dimensions=("maintenance_actionability",),
        evidence=("tool_names",),
        reserve_cost_usd=0.1,
        response_contract="anchored_review_v1",
    )

    def anchored_request(**_kwargs: object):
        return (
            {
                "scores": {"maintenance_actionability": 0.8},
                "label": "strong",
                "reason": (
                    "The answer names a concrete next action and states its "
                    "evidence limit."
                ),
                "missing_evidence": False,
            },
            {"cost_usd": 0.04},
            {"request_policy": {"automatic_retries": 0}},
        )

    rows = score_comparison_rows(
        replace(spec, evaluators=(*spec.evaluators, judge)),
        [
            {
                "answer": {"amount": 125, "source": "expense-policy-v4.md"},
                "task_id": "expense-limit",
                "trial_index": 1,
                "variant_id": "candidate",
                "harness": "claude-code",
                "mcp_tool_names": ["query_wandb_tool"],
            }
        ],
        repo_root=root,
        env={
            "ANTHROPIC_API_KEY": "secret-example-value",
            "FUGUE_WANDB_INFERENCE_PROJECT": "wandb/test",
        },
        judge_request=anchored_request,
    )

    review = rows[0]["comparison_judges"]["maintainer-review"]["review"]
    assert review == {
        "schema_version": 1,
        "label": "strong",
        "reason": (
            "The answer names a concrete next action and states its evidence limit."
        ),
        "missing_evidence": False,
    }
    assert rows[0]["comparison_judges"]["maintainer-review"]["route_receipt"][
        "response_contract"
    ] == "anchored_review_v1"
    projection = local_result_row_projection_v1(rows[0])
    assert projection["judge_reviews"] == {
        "maintainer-review": {
            "label": "strong",
            "reason": (
                "The answer names a concrete next action and states its evidence "
                "limit."
            ),
            "missing_evidence": False,
            "observed_cost_usd": 0.04,
            "cost_status": "observed",
        }
    }
    attempt = _paired_attempt_view_v3(rows[0])
    assert attempt is not None
    assert attempt.judge_reviews["maintainer-review"].label == "strong"
    assert attempt.judge_reviews["maintainer-review"].missing_evidence is False
    assert "concrete next action" not in json.dumps(attempt.score_explanations)


def test_anchored_judge_fails_closed_without_an_anchored_review() -> None:
    root = Path.cwd()
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=root)
    judge = ComparisonEvaluatorV1(
        id="maintainer-review",
        type="llm_judge",
        required=False,
        profile="wandb/anthropic/claude-opus-4-8",
        rubric="Assess whether the response is useful and actionable.",
        dimensions=("maintenance_actionability",),
        reserve_cost_usd=0.1,
        response_contract="anchored_review_v1",
    )

    def numeric_only_request(**_kwargs: object):
        return (
            {
                "scores": {"maintenance_actionability": 0.8},
                "overall_assessment": "strong",
                "uncertainty": 0.1,
                "rationale": "The response is useful.",
            },
            {"cost_usd": 0.04},
            {},
        )

    rows = score_comparison_rows(
        replace(spec, evaluators=(*spec.evaluators, judge)),
        [
            {
                "answer": {"amount": 125, "source": "expense-policy-v4.md"},
                "task_id": "expense-limit",
                "trial_index": 1,
                "variant_id": "candidate",
                "harness": "claude-code",
            }
        ],
        repo_root=root,
        env={
            "ANTHROPIC_API_KEY": "secret-example-value",
            "FUGUE_WANDB_INFERENCE_PROJECT": "wandb/test",
        },
        judge_request=numeric_only_request,
    )

    result = rows[0]["comparison_judges"]["maintainer-review"]
    assert result["status"] == "unavailable"
    assert result["failure"]["stage"] == "rubric_validation"
    assert "review" not in result
    assert "judge_reviews" not in local_result_row_projection_v1(rows[0])


def test_checkpoint_records_advisory_judge_without_gating_execution() -> None:
    root = Path.cwd()
    spec = load_comparison(
        MCP_MAINTENANCE_EXAMPLE / "tool-surface-canary-local-v4.yaml",
        repo_root=root,
    )
    judge = next(
        evaluator
        for evaluator in spec.evaluators
        if evaluator.id == "maintainer-actionability"
    )
    assert judge.timeout_sec == 300
    row = {
        "comparison_evaluation_status": "scored",
        "comparison_required_evaluation_complete": True,
        "host_evaluator_status": "passed",
        "comparison_judges": {
            "maintainer-actionability": {
                "status": "unavailable",
                "reason": "judge evaluation failed: ReadTimeout",
            }
        },
    }

    _require_checkpoint_judges(
        spec,
        row,
        checkpoint_index=0,
    )

    assert row["comparison_judge_checkpoint_status"] == "advisory_unavailable"
    assert row["comparison_judge_checkpoint_unavailable"] == [
        "maintainer-actionability"
    ]
    assert "comparison_judge_checkpoint_required_unavailable" not in row
    assert row["comparison_evaluation_status"] == "scored"
    assert row["comparison_required_evaluation_complete"] is True
    assert row["host_evaluator_status"] == "passed"

    scored = {"comparison_judges": {"maintainer-actionability": {"status": "scored"}}}
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


def test_checkpoint_stops_when_required_judge_does_not_score() -> None:
    root = Path.cwd()
    optional = load_comparison(
        MCP_MAINTENANCE_EXAMPLE / "tool-surface-canary-local-v4.yaml",
        repo_root=root,
    )
    spec = replace(
        optional,
        evaluators=tuple(
            replace(evaluator, required=True)
            if evaluator.id == "maintainer-actionability"
            else evaluator
            for evaluator in optional.evaluators
        ),
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
        match="first-cell required judge checkpoint did not score",
    ):
        _require_checkpoint_judges(spec, row, checkpoint_index=0)

    assert row["comparison_judge_checkpoint_status"] == "failed"
    assert row["comparison_judge_checkpoint_unavailable"] == [
        "maintainer-actionability"
    ]
    assert row["comparison_judge_checkpoint_required_unavailable"] == [
        "maintainer-actionability"
    ]


def test_checkpoint_stops_when_required_deterministic_scorer_is_unavailable() -> None:
    root = Path.cwd()
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=root)
    row = {
        "comparison_evaluation_status": "unavailable",
        "comparison_required_evaluation_complete": False,
        "comparison_evaluation_reason": "deterministic scorer raised",
    }

    with pytest.raises(RuntimeError, match="required evaluation did not complete"):
        _require_checkpoint_evaluations(spec, row, checkpoint_index=0)


def test_decision_policy_accepts_release_notes_without_infrastructure_gates() -> None:
    policies = _canonical_decision_gate_policies(
        (),
        implicit=(),
        release_note_coverage=(
            {
                "release_note": "bounded-history",
                "status": "observed_delta",
                "infrastructure_gates": [],
            },
        ),
    )

    assert policies == []


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

    monkeypatch.setattr("fugue.bench.task_authoring.run_inline_scorer", fake_runner)
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
                    "Return JSON containing the current expense amount and its source."
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
        profile="wandb/zai-org/GLM-5.2",
        rubric="Score grounding, usefulness, prioritization, and calibration.",
        dimensions=(
            "evidence_grounding",
            "usefulness",
            "prioritization",
            "calibration",
        ),
        evidence=("tool_names", "inspected_paths"),
        timeout_sec=300,
        reserve_cost_usd=0.1,
    )
    captured: dict[str, object] = {"requests": 0}

    def fake_post(client, _route, _key, _env, prompt):
        captured["requests"] = int(captured["requests"]) + 1
        captured["timeout_sec"] = client.timeout.read
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

    prompt = str(captured["prompt"])
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
    assert captured["requests"] == 1
    assert captured["timeout_sec"] == 300
    privacy = rows[0]["comparison_judges"]["maintainer-review"]["route_receipt"][
        "judge_input_privacy"
    ]
    assert privacy["status"] == "passed"
    assert len(privacy["payload_sha256"]) == 64
    request_policy = rows[0]["comparison_judges"]["maintainer-review"]["route_receipt"][
        "request_policy"
    ]
    assert request_policy == {
        "schema_version": 1,
        "timeout_sec": 300,
        "max_output_tokens": 1_200,
        "structured_assistant_options": {"thinking": {"type": "disabled"}},
        "automatic_retries": 0,
    }


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
    assert rows[0]["comparison_judges"]["maintainer-review"]["cost_usd"] == 0.0


def test_durable_judge_request_reuses_completed_response_without_respend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judge = ComparisonEvaluatorV1(
        id="durable-review",
        type="llm_judge",
        required=True,
        profile="wandb/openai/gpt-oss-120b",
        rubric="Score maintenance actionability.",
        dimensions=("maintenance_actionability",),
        input_cost_per_million=1.0,
        output_cost_per_million=2.0,
    )
    provider_calls: list[str] = []
    prompt_lengths: list[int] = []

    def fake_post(*args, **_kwargs):
        provider_calls.append("called")
        prompt_lengths.append(len(str(args[-1])))
        return (
            {
                "scores": {"maintenance_actionability": 0.8},
                "overall_assessment": "strong",
                "uncertainty": 0.1,
                "rationale": "The response is grounded in the inspected source.",
            },
            {"input_tokens": 100, "output_tokens": 40},
        )

    monkeypatch.setattr("fugue.bench.evaluations._post_judge", fake_post)
    row = {
        "run_id": "durable-judge",
        "attempt_id": "a" * 64,
        "answer": "A" * 60_000,
    }
    env = {
        "WANDB_API_KEY": "secret",
        "FUGUE_WANDB_INFERENCE_PROJECT": "wandb/test",
        "FUGUE_EVIDENCE_MODE": "local",
        "FUGUE_HOST_REPO_ROOT": tmp_path.as_posix(),
        "FUGUE_RUN_ID": "durable-judge",
    }

    first = _request_comparison_judge(
        evaluator=judge,
        public_task={"input": "Review the maintenance response."},
        row=row,
        env=env,
    )
    second = _request_comparison_judge(
        evaluator=judge,
        public_task={"input": "Review the maintenance response."},
        row=row,
        env=env,
    )

    assert provider_calls == ["called"]
    assert prompt_lengths == [MAX_COMPARISON_JUDGE_PROMPT_CHARACTERS]
    assert second[0] == first[0]
    assert second[1] == first[1]
    assert first[2]["durable_request_reused"] is False
    assert second[2]["durable_request_reused"] is True
    receipt = Path(str(second[2]["durable_request_receipt"]))
    assert receipt.stat().st_mode & 0o777 == 0o600


def test_concurrent_durable_judge_finalization_spends_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judge = ComparisonEvaluatorV1(
        id="concurrent-review",
        type="llm_judge",
        required=True,
        profile="wandb/openai/gpt-oss-120b",
        rubric="Score maintenance actionability.",
        dimensions=("maintenance_actionability",),
        input_cost_per_million=1.0,
        output_cost_per_million=2.0,
    )
    provider_calls = 0
    provider_started = threading.Event()
    release_provider = threading.Event()

    def fake_post(*_args, **_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        provider_started.set()
        assert release_provider.wait(timeout=5)
        return (
            {
                "scores": {"maintenance_actionability": 0.8},
                "overall_assessment": "strong",
                "uncertainty": 0.1,
                "rationale": "The response is grounded in inspected source.",
            },
            {"input_tokens": 100, "output_tokens": 40},
        )

    monkeypatch.setattr("fugue.bench.evaluations._post_judge", fake_post)
    request = {
        "evaluator": judge,
        "public_task": {"input": "Review the maintenance response."},
        "row": {
            "run_id": "concurrent-judge",
            "attempt_id": "c" * 64,
            "answer": "A bounded maintainer recommendation.",
        },
        "env": {
            "WANDB_API_KEY": "secret",
            "FUGUE_WANDB_INFERENCE_PROJECT": "wandb/test",
            "FUGUE_EVIDENCE_MODE": "local",
            "FUGUE_HOST_REPO_ROOT": tmp_path.as_posix(),
            "FUGUE_RUN_ID": "concurrent-judge",
        },
    }

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(_request_comparison_judge, **request)
        assert provider_started.wait(timeout=5)
        second = pool.submit(_request_comparison_judge, **request)
        time.sleep(0.1)
        assert provider_calls == 1
        release_provider.set()
        first_result = first.result(timeout=5)
        second_result = second.result(timeout=5)

    assert provider_calls == 1
    assert first_result[0] == second_result[0]
    assert {
        first_result[2]["durable_request_reused"],
        second_result[2]["durable_request_reused"],
    } == {False, True}


def test_durable_judge_pending_request_never_resends_ambiguous_spend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judge = ComparisonEvaluatorV1(
        id="pending-review",
        type="llm_judge",
        required=True,
        profile="wandb/openai/gpt-oss-120b",
        rubric="Score maintenance actionability.",
        dimensions=("maintenance_actionability",),
        input_cost_per_million=1.0,
        output_cost_per_million=2.0,
    )
    provider_calls: list[str] = []

    def ambiguous_failure(*_args, **_kwargs):
        provider_calls.append("called")
        raise RuntimeError("connection lost after request submission")

    monkeypatch.setattr("fugue.bench.evaluations._post_judge", ambiguous_failure)
    row = {
        "run_id": "pending-judge",
        "attempt_id": "b" * 64,
        "answer": "A bounded maintainer recommendation.",
    }
    env = {
        "WANDB_API_KEY": "secret",
        "FUGUE_WANDB_INFERENCE_PROJECT": "wandb/test",
        "FUGUE_EVIDENCE_MODE": "local",
        "FUGUE_HOST_REPO_ROOT": tmp_path.as_posix(),
        "FUGUE_RUN_ID": "pending-judge",
    }
    request = {
        "evaluator": judge,
        "public_task": {"input": "Review the maintenance response."},
        "row": row,
        "env": env,
    }

    with pytest.raises(RuntimeError, match="connection lost"):
        _request_comparison_judge(**request)
    with pytest.raises(ExecutionFinalizationPending, match="will not be resent"):
        _request_comparison_judge(**request)

    assert provider_calls == ["called"]


def test_blind_judge_read_timeout_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path.cwd()
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=root)
    judge = ComparisonEvaluatorV1(
        id="maintainer-review",
        type="llm_judge",
        required=False,
        profile="wandb/openai/gpt-oss-120b",
        rubric="Score maintenance actionability.",
        dimensions=("maintenance_actionability",),
        evidence=("tool_names",),
        timeout_sec=300,
        reserve_cost_usd=0.1,
    )
    provider_calls = 0

    def timeout_once(*_args, **_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        raise httpx.ReadTimeout("judge request exceeded its deadline")

    monkeypatch.setattr("fugue.bench.evaluations._post_judge", timeout_once)
    rows = score_comparison_rows(
        replace(spec, evaluators=(*spec.evaluators, judge)),
        [
            {
                "answer": {"amount": 125, "source": "expense-policy-v4.md"},
                "task_id": "expense-limit",
                "trial_index": 1,
                "variant_id": "candidate",
                "harness": "claude-code",
                "mcp_tool_names": ["query_wandb_tool"],
            }
        ],
        repo_root=root,
        env={
            "WANDB_API_KEY": "secret",
            "FUGUE_WANDB_INFERENCE_PROJECT": "wandb/test",
        },
    )

    assert provider_calls == 1
    assert rows[0]["comparison_judge_status"] == "unavailable"
    assert (
        rows[0]["comparison_judges"]["maintainer-review"]["reason"]
        == "judge evaluation failed: ReadTimeout"
    )
    assert rows[0]["comparison_judges"]["maintainer-review"]["failure"] == {
        "schema_version": 1,
        "stage": "provider_request",
        "code": "provider_timeout",
        "message": "judge provider request timed out",
        "exception_type": "ReadTimeout",
        "request_policy": {
            "schema_version": 1,
            "timeout_sec": 300,
            "max_output_tokens": 1_200,
            "structured_assistant_options": {},
            "automatic_retries": 0,
        },
    }
    assert rows[0]["comparison_judges"]["maintainer-review"]["cost_usd"] is None
    assert (
        rows[0]["comparison_judges"]["maintainer-review"]["accounted_cost_usd"] == 0.1
    )
    assert (
        rows[0]["comparison_judges"]["maintainer-review"]["cost_observation_complete"]
        is False
    )


def test_blind_judge_records_safe_no_json_failure_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path.cwd()
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=root)
    judge = ComparisonEvaluatorV1(
        id="maintainer-review",
        type="llm_judge",
        required=False,
        profile="wandb/zai-org/GLM-5.2",
        rubric="Score maintenance actionability.",
        dimensions=("maintenance_actionability",),
        evidence=("tool_names",),
        timeout_sec=300,
        reserve_cost_usd=0.1,
    )

    def no_json(*_args: object, **_kwargs: object) -> None:
        raise JudgeResponseError(
            stage="response_extraction",
            code="no_json_object",
            message="judge returned no JSON object",
            response_sha256="a" * 64,
            response_characters=321,
            usage={"input_tokens": 100, "output_tokens": 1_200},
        )

    monkeypatch.setattr("fugue.bench.evaluations._post_judge", no_json)
    rows = score_comparison_rows(
        replace(spec, evaluators=(*spec.evaluators, judge)),
        [
            {
                "answer": {"amount": 125, "source": "expense-policy-v4.md"},
                "task_id": "expense-limit",
                "trial_index": 1,
                "variant_id": "candidate",
                "harness": "claude-code",
                "mcp_tool_names": ["query_wandb_tool"],
            }
        ],
        repo_root=root,
        env={
            "WANDB_API_KEY": "secret",
            "FUGUE_WANDB_INFERENCE_PROJECT": "wandb/test",
        },
    )

    failure = rows[0]["comparison_judges"]["maintainer-review"]["failure"]
    assert failure == {
        "schema_version": 1,
        "stage": "response_extraction",
        "code": "no_json_object",
        "message": "judge returned no JSON object",
        "exception_type": "ValueError",
        "response_sha256": "a" * 64,
        "response_characters": 321,
        "usage": {"input_tokens": 100, "output_tokens": 1_200},
        "request_policy": {
            "schema_version": 1,
            "timeout_sec": 300,
            "max_output_tokens": 1_200,
            "structured_assistant_options": {"thinking": {"type": "disabled"}},
            "automatic_retries": 0,
        },
    }
    assert "expense-policy-v4.md" not in json.dumps(failure)


def test_blind_judge_distinguishes_strict_rubric_validation_failure() -> None:
    root = Path.cwd()
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=root)
    judge = ComparisonEvaluatorV1(
        id="maintainer-review",
        type="llm_judge",
        required=False,
        profile="wandb/zai-org/GLM-5.2",
        rubric="Score maintenance actionability.",
        dimensions=("maintenance_actionability",),
        evidence=("tool_names",),
        timeout_sec=300,
        reserve_cost_usd=0.1,
    )

    def invalid_payload(
        **_kwargs: object,
    ) -> tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
    ]:
        return (
            {"scores": {"wrong-private-dimension": 1}},
            {"input_tokens": 100, "output_tokens": 20},
            {"request_policy": {"automatic_retries": 0}},
        )

    rows = score_comparison_rows(
        replace(spec, evaluators=(*spec.evaluators, judge)),
        [
            {
                "answer": {"amount": 125, "source": "expense-policy-v4.md"},
                "task_id": "expense-limit",
                "trial_index": 1,
                "variant_id": "candidate",
                "harness": "claude-code",
                "mcp_tool_names": ["query_wandb_tool"],
            }
        ],
        repo_root=root,
        env={
            "WANDB_API_KEY": "secret",
            "FUGUE_WANDB_INFERENCE_PROJECT": "wandb/test",
        },
        judge_request=invalid_payload,
    )

    failure = rows[0]["comparison_judges"]["maintainer-review"]["failure"]
    assert failure == {
        "schema_version": 1,
        "stage": "rubric_validation",
        "code": "invalid_rubric_payload",
        "message": "judge response failed strict rubric validation",
        "exception_type": "ValueError",
        "usage": {"input_tokens": 100, "output_tokens": 20},
        "request_policy": {
            "schema_version": 1,
            "timeout_sec": 300,
            "max_output_tokens": 1_200,
            "structured_assistant_options": {"thinking": {"type": "disabled"}},
            "automatic_retries": 0,
        },
    }
    assert "wrong-private-dimension" not in json.dumps(failure)


def _patch_packaged_template_scorer(monkeypatch: pytest.MonkeyPatch) -> None:
    def score(
        _evaluator: object,
        *,
        output: object,
        expected: object,
        **_kwargs: object,
    ) -> dict[str, object]:
        passed = output == expected
        answer_present = output is not None and (
            not isinstance(output, str) or bool(output.strip())
        )
        details = {
            "answer_present": answer_present,
            "expected_values": passed,
        }
        return {
            "score": 1.0 if all(details.values()) else 0.0,
            "reason": "isolated scorer test double",
            "details": details,
        }

    monkeypatch.setattr("fugue.bench.comparison._run_custom_scorer", score)

    def prepare_scorer(*_args: object, **kwargs: object):
        from fugue.bench.task_authoring import load_task_profiles

        profile = load_task_profiles(Path(kwargs["repo_root"])).scorer_runtime(
            "python312-sandbox-v1"
        )
        return {
            "python312-sandbox-v1": {
                "image": profile.image,
                "image_id": "sha256:" + "a" * 64,
                "platform": profile.platform,
                "profile_digest": profile.profile_digest,
            }
        }

    monkeypatch.setattr(
        "fugue.bench.comparison._prepare_comparison_scorer_runtimes",
        prepare_scorer,
    )


def test_scaffold_refuses_non_empty_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_packaged_template_scorer(monkeypatch)
    destination = tmp_path / "comparison"
    scaffold_comparison(destination)
    assert (destination / "comparison.yaml").is_file()
    assert (
        destination / "configs/fugue/skills/verify-current-source/SKILL.md"
    ).is_file()
    spec = load_comparison(
        destination / "comparison.yaml",
        repo_root=destination,
    )
    preview = preview_comparison(spec, repo_root=destination)
    assert preview.readiness["status"] == "blocked"
    assert preview.readiness["task_count"] == 2
    assert preview.readiness["estimated_cells"] == 8
    assert preview.readiness["base_failures"] == 2
    assert preview.readiness["gold_passes"] == 2
    assert preview.matrix["estimated_trials"] == 8
    blockers = preview.readiness["blockers"]
    assert len(blockers) == 4
    assert all(
        blocker.startswith("local agent:")
        or blocker.startswith("local task:")
        or blocker.startswith("comparison preparation is missing or drifted;")
        for blocker in blockers
    )
    assert not (destination / ".fugue").exists()
    with pytest.raises(FileExistsError, match="non-empty"):
        scaffold_comparison(destination)


def test_unavailable_isolated_scorer_is_not_reported_as_saturation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "comparison"
    comparison_path = scaffold_comparison(destination)
    monkeypatch.setattr(
        "fugue.bench.comparison._run_custom_scorer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("scorer image is not prepared")
        ),
    )

    readiness = check_comparison(
        load_comparison(comparison_path, repo_root=destination),
        repo_root=destination,
    )

    assert any("evaluator qualification failed" in item for item in readiness.blockers)
    assert not any("saturated" in item for item in readiness.warnings)


def _local_preparation_spec(
    tmp_path: Path,
    *,
    preparation_required: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, ComparisonSpecV1]:
    _patch_packaged_template_scorer(monkeypatch)
    comparison_path = scaffold_comparison(tmp_path / "comparison")
    raw = yaml.safe_load(comparison_path.read_text(encoding="utf-8"))
    raw["execution"]["evidence_mode"] = "weave_required"
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
        monkeypatch=monkeypatch,
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
        monkeypatch=monkeypatch,
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
    assert (
        second.readiness["runtime_lock_digests"]["comparison_preparation"]
        == receipt["receipt_digest"]
    )

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


def test_scorer_preparation_pulls_and_locks_the_declared_platform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison_path = scaffold_comparison(tmp_path / "comparison")
    root = comparison_path.parent
    spec = load_comparison(comparison_path, repo_root=root)
    from fugue.bench.task_authoring import load_task_profiles

    profile = load_task_profiles(root).scorer_runtime("python312-sandbox-v1")
    wrong_platform = (
        "linux/amd64" if profile.platform == "linux/arm64" else "linux/arm64"
    )
    calls: list[list[str]] = []

    monkeypatch.setattr("fugue.bench.comparison.shutil.which", lambda _: "/docker")

    def run(command: list[str], **_kwargs: object):
        calls.append(command)
        if command[1:3] == ["image", "inspect"]:
            platform = wrong_platform if len(calls) == 1 else profile.platform
            return SimpleNamespace(
                returncode=0,
                stdout=f"{platform} sha256:{'a' * 64}\n",
            )
        assert command[1:3] == ["pull", "--platform"]
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr("fugue.bench.comparison.subprocess.run", run)

    locks = _prepare_comparison_scorer_runtimes(spec, repo_root=root)

    assert locks["python312-sandbox-v1"] == {
        "image": profile.image,
        "image_id": "sha256:" + "a" * 64,
        "platform": profile.platform,
        "profile_digest": profile.profile_digest,
    }
    inspect_calls = [
        command for command in calls if command[1:3] == ["image", "inspect"]
    ]
    assert len(inspect_calls) == 2
    assert all(
        command[3:5] == ["--platform", profile.platform] for command in inspect_calls
    )
    assert calls[1][1:4] == ["pull", "--platform", profile.platform]


def test_prepared_scorer_platform_drift_invalidates_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, spec = _local_preparation_spec(
        tmp_path,
        preparation_required=True,
        monkeypatch=monkeypatch,
    )
    monkeypatch.setattr(
        "fugue.bench.comparison._runtime_readiness",
        lambda *_args, **_kwargs: (
            {"agent:codex:amd64": "a" * 64, "task:policy-limit:amd64": "b" * 64},
            [],
        ),
    )
    monkeypatch.setattr(OperatorService, "prepare", lambda *_args, **_kwargs: None)
    prepare_comparison(spec, repo_root=root, operator=OperatorService(root))

    profiles_path = root / "configs/fugue/task-authoring/profiles.yaml"
    profiles = yaml.safe_load(profiles_path.read_text())
    runtime = profiles["scorer_runtimes"][0]
    runtime["platform"] = (
        "linux/amd64" if runtime["platform"] == "linux/arm64" else "linux/arm64"
    )
    profiles_path.write_text(yaml.safe_dump(profiles, sort_keys=False))

    drifted = preview_comparison(spec, repo_root=root, operator=OperatorService(root))

    assert drifted.readiness["status"] == "blocked"
    assert any(
        "scorer runtime 'python312-sandbox-v1' profile changed" in blocker
        for blocker in drifted.readiness["blockers"]
    )


def test_execute_comparison_never_prepares_after_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, spec = _local_preparation_spec(
        tmp_path,
        preparation_required=False,
        monkeypatch=monkeypatch,
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


def test_execute_local_comparison_never_requires_or_fetches_weave(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_packaged_template_scorer(monkeypatch)
    comparison_path = scaffold_comparison(tmp_path / "comparison")
    raw = yaml.safe_load(comparison_path.read_text(encoding="utf-8"))
    raw["execution"]["model"] = "anthropic/claude-sonnet-5"
    raw["execution"]["attempts"] = 1
    raw["execution"]["approval_required"] = False
    raw["execution"]["preparation_required"] = False
    raw["execution"]["evidence_checkpoint_cells"] = 0
    comparison_path.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )
    root = comparison_path.parent
    spec = load_comparison(comparison_path, repo_root=root)
    preview = preview_comparison(spec, repo_root=root)
    captured: dict[str, object] = {}

    for name in (
        "WANDB_API_KEY",
        "WANDB_ENTITY",
        "WANDB_PROJECT",
        "FUGUE_WEAVE_PROJECT",
        "WEAVE_PROJECT",
    ):
        monkeypatch.delenv(name, raising=False)

    def fake_execute_run(
        self: OperatorService,
        request: object,
        **_kwargs: object,
    ) -> object:
        assert "WANDB_API_KEY" not in self.env
        captured["approved"] = request.approved_comparison  # type: ignore[attr-defined]
        return SimpleNamespace(status="passed")

    def fake_export_run(
        _self: OperatorService,
        run_id: str,
        *,
        out: Path,
        fetch_weave: bool,
        to_weave: bool,
        **_kwargs: object,
    ) -> object:
        assert fetch_weave is False
        assert to_weave is False
        approved = dict(captured["approved"])  # type: ignore[arg-type]
        provenance_by_attempt = {
            str(item["attempt_id"]): list(item.get("integration_provenance") or [])
            for item in preview.matrix["matrix_cells"]
        }
        rows: list[dict[str, object]] = []
        for cell in approved["expected_cells"]:
            cell = dict(cell)
            attempt = str(cell["attempt_id"])
            passed = str(cell["variant_id"]) == "candidate"
            rows.append(
                {
                    **cell,
                    "run_id": run_id,
                    "approved_comparison": approved,
                    "trace_receipt": approved["evidence_destination"],
                    "candidate_definition": approved["candidate_definitions"][
                        str(cell["candidate_id"])
                    ],
                    "integration_provenance": provenance_by_attempt[attempt],
                    "prediction_id": f"prediction-{attempt}",
                    "pass": passed,
                    "status": "passed",
                    "comparison_evaluation_status": "completed",
                    "comparison_required_evaluation_complete": True,
                    "comparison_deterministic_scores": {
                        "fact-and-source.answer_present": passed,
                        "fact-and-source.expected_values": passed,
                    },
                    "comparison_deterministic_criticality": {
                        "fact-and-source.answer_present": True,
                        "fact-and-source.expected_values": True,
                    },
                    "comparison_dimension_roles": {
                        "fact-and-source.answer_present": "outcome",
                        "fact-and-source.expected_values": "outcome",
                    },
                    "agent_response": {
                        "amount": 125 if passed else 100,
                        "source": (
                            "expense-policy-v4.md" if passed else "expense-policy-v3.md"
                        ),
                    },
                    "queried_projects": [],
                    "cost_usd": 0.1,
                    "latency_sec": 1.0,
                    "local_evidence_links": [
                        {
                            "kind": kind,
                            "status": "resolved",
                            "system": "local_artifact",
                            "ref": (
                                f"fugue://local-evidence/{run_id}/attempt/"
                                f"{attempt}/{kind}"
                            ),
                        }
                        for kind in (
                            "evaluation_root",
                            "prediction_and_score",
                            "prediction",
                            "agent_root",
                            "dataset",
                        )
                    ],
                    "local_evidence_integrity": "resolved",
                    "privacy_contract_version": 2,
                    "local_artifact_privacy_scan_status": "passed",
                    "local_artifact_privacy_scan_digest": "1" * 64,
                    "local_artifact_privacy_match_count": 0,
                    "hosted_evidence_privacy_scan_status": "not_applicable",
                    "private_label_boundary_verified": True,
                    "harbor_environment": "local_harbor_docker",
                    "harbor_conformance_status": "passed",
                    "harbor_conformance_receipt_digest": "2" * 64,
                    "harbor_policy_attestation_verified": True,
                    "sandbox_cleanup_verified": True,
                    "sandbox_deleted": True,
                    "orphaned_sandbox": False,
                }
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        return SimpleNamespace(path=out)

    def bind_local(
        rows: list[dict[str, object]],
        *,
        repo_root: Path,
        run_id: str,
        hosted_evidence_expected: bool,
    ) -> None:
        assert repo_root == root
        assert run_id
        assert hosted_evidence_expected is False
        captured["local_rows"] = len(rows)
        for index, row in enumerate(rows, start=1):
            row.update(
                {
                    "local_evidence_record_digest": f"{index:x}" * 64,
                    "local_evidence_prediction_row_sha256": (f"{index + 2:x}" * 64),
                    "local_evidence_manifest_digest": "4" * 64,
                    "local_evidence_manifest_file_sha256": "5" * 64,
                    "local_evidence_plan_digest": "6" * 64,
                    "local_evidence_attempt_record_set_digest": "7" * 64,
                    "local_evidence_prediction_row_set_digest": "8" * 64,
                    "local_evidence_run_receipt_digest": "9" * 64,
                    "local_evidence_run_receipt_file_sha256": "a" * 64,
                }
            )

    def forbidden_hosted(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("local comparison attempted hosted evidence work")

    def bind_scored_rows(**kwargs: object) -> list[dict[str, object]]:
        rows = list(kwargs["rows"])  # type: ignore[arg-type]
        for row in rows:
            row["source_pre_run_drift"] = kwargs["source_pre_run_drift"].to_dict()
            row["source_post_run_drift"] = kwargs["source_post_run_drift"].to_dict()
        return rows

    monkeypatch.setattr(OperatorService, "execute_run", fake_execute_run)
    monkeypatch.setattr(OperatorService, "export_run", fake_export_run)
    monkeypatch.setattr(
        "fugue.bench.comparison._bind_local_execution_evidence",
        bind_local,
    )
    monkeypatch.setattr(
        "fugue.bench.comparison._score_and_bind_exported_comparison_rows",
        bind_scored_rows,
    )
    monkeypatch.setattr(
        "fugue.bench.comparison.trace_project_slug",
        forbidden_hosted,
    )
    monkeypatch.setattr(
        "fugue.bench.comparison._apply_hosted_evidence_privacy",
        forbidden_hosted,
    )

    result, result_path, markdown_path = execute_comparison(
        preview,
        approval_digest="",
        repo_root=root,
        # Even an old caller requesting hosted hydration cannot make local
        # execution contact Weave.
        fetch_weave=True,
        publish_research=False,
    )

    assert isinstance(result, ComparisonResultV3)
    assert result.evidence_project is None
    assert result.evidence_backend == "local"
    assert result.evidence_topology.result_destination.to_dict()["kind"] == ("local")
    assert result.hosted_chain_integrity == "not_applicable"
    assert result.operational_summary["evidence_states"] == {"reconciled": 4}
    assert result.evidence_links == ()
    view = build_comparison_evaluation_view(result.to_dict())
    assert isinstance(view, ExperimentViewV3)
    assert view.evidence_scope is None
    assert view.evidence_topology["result_destination"]["kind"] == "local"
    assert view.result_digest == result.result_digest
    assert view.qualification_digest == result.qualification_digest
    assert view.runtime_lock_digest == stable_digest(list(view.runtime_locks))
    assert {
        link["system"]
        for pair in view.paired_cases
        for arm in ("baseline", "candidate")
        for link in pair[arm]["evidence_links"]
    } == {"local_artifact"}
    assert experiment_view_from_dict(view.to_dict()) == view
    assert captured["local_rows"] == 4
    assert result_path.is_file()
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "]()" not in markdown
    assert "Package release: **NOT EVALUATED**" in markdown
    assert "Release-policy note: Package release was not evaluated" in markdown
    assert "behavioral study is not release-complete" not in markdown
    assert "Baseline tasks that passed all required gates:" in markdown
    assert "Candidate tasks that passed all required gates:" in markdown
    assert "| Baseline full result | Candidate full result |" in markdown
    assert "Paired outcome change |" in markdown
    assert (
        "Evidence integrity grade: **A** (evidence-link reconciliation and privacy "
        "integrity; not a behavioral-quality score)" in markdown
    )
    assert "Evidence backend: **local Fugue ledger**" in markdown
    assert result.local_evidence is not None
    assert f".fugue/runtime/{result.local_evidence['run_id']}/evidence/" in markdown
    assert "Portable evidence destination: `fugue-evidence` layout v1" in markdown
    assert "`fugue://local-evidence/" in markdown
    assert "No safe evidence links were available." not in markdown

    governed = replace(
        result,
        decision_policy=parse_decision_policy(_decision_policy()),
        decision=replace(
            result.decision,
            status="blocked",
            recommendation="Complete the missing package gates.",
        ),
    )
    governed_markdown = _result_markdown(governed)
    assert "Package release: **HOLD**" in governed_markdown
    assert (
        "This local Study does not evaluate every package-release gate."
        in governed_markdown
    )
    assert "Governed gate status: **BLOCKED**" in governed_markdown
    assert (
        "Governed gate recommendation: Complete the missing package gates."
        in governed_markdown
    )
    assert "Package release decision: **BLOCKED**" not in governed_markdown
    approved = dict(captured["approved"])  # type: ignore[arg-type]
    assert approved["approval_required"] is False
    assert approved["approval_digest"] == ""
    authorization = str(approved["execution_authorization_digest"])
    assert len(authorization) == 64
    assert set(authorization) <= set("0123456789abcdef")


def test_local_execution_binding_reconciles_manifest_and_run_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = ("1" * 64, "2" * 64)
    records = tuple(
        SimpleNamespace(
            attempt_id=attempt,
            record_digest=str(index) * 64,
            prediction_row_sha256=str(index + 2) * 64,
        )
        for index, attempt in enumerate(attempts, start=1)
    )
    manifest = SimpleNamespace(
        run_id="local-run",
        status="complete",
        destination=SimpleNamespace(
            to_dict=lambda: {
                "schema_version": 1,
                "kind": "local",
                "format": "fugue-evidence",
                "layout_version": 1,
                "destination_digest": "f" * 64,
            }
        ),
        terminal_attempt_ids=attempts,
        manifest_digest="3" * 64,
        plan_digest="7" * 64,
        attempt_record_set_digest="4" * 64,
        prediction_row_set_digest="5" * 64,
        attempt_records=records,
        run_conformance=SimpleNamespace(
            receipt_sha256="6" * 64,
            status="passed",
            enforced=True,
        ),
    )

    class FakeStore:
        def __init__(self, repo_root: Path, run_id: str) -> None:
            assert repo_root == tmp_path
            assert run_id == "local-run"
            self.manifest_path = tmp_path / "manifest.json"
            self.run_conformance_path = tmp_path / "conformance.json"
            self.manifest_path.write_text("manifest", encoding="utf-8")
            self.run_conformance_path.write_text("receipt", encoding="utf-8")

        def read_manifest(self) -> object:
            return manifest

    rows = [
        {
            "attempt_id": attempt,
            # Reproduce the projection gap from the first standalone wheel.
            "trace_receipt": None,
            "local_evidence_links": [{"system": "local_artifact"}],
            "local_evidence_record_digest": record.record_digest,
        }
        for attempt, record in zip(attempts, records, strict=True)
    ]
    for row, record in zip(rows, records, strict=True):
        record.result_row_projection_digest = local_result_row_projection_digest(row)
    manifest.result_row_projection_set_digest = stable_digest(
        [[record.attempt_id, record.result_row_projection_digest] for record in records]
    )

    def apply_conformance(
        bound_rows: list[dict[str, object]],
        *,
        repo_root: Path,
        run_id: str,
    ) -> None:
        assert repo_root == tmp_path
        assert run_id == "local-run"
        for row in bound_rows:
            row["harbor_conformance_status"] = "passed"

    monkeypatch.setattr(
        "fugue.bench.comparison.LocalEvidenceStore",
        FakeStore,
    )
    monkeypatch.setattr(
        "fugue.bench.run_conformance.read_harbor_run_conformance_receipt",
        lambda **_kwargs: {
            "backend": "local_harbor_docker",
            "status": "passed",
            "receipt_sha256": "6" * 64,
        },
    )
    monkeypatch.setattr(
        "fugue.bench.comparison._apply_harbor_conformance",
        apply_conformance,
    )

    _bind_local_execution_evidence(
        rows,
        repo_root=tmp_path,
        run_id="local-run",
    )

    assert {row["local_evidence_manifest_digest"] for row in rows} == {"3" * 64}
    assert {row["local_evidence_run_receipt_digest"] for row in rows} == {"6" * 64}
    assert {row["trace_receipt"]["kind"] for row in rows} == {"local"}
    assert {row["hosted_evidence_privacy_scan_status"] for row in rows} == {
        "not_applicable"
    }

    conflicting_rows = [
        dict(
            row,
            trace_receipt={
                **row["trace_receipt"],
                "destination_digest": "0" * 64,
            },
        )
        for row in rows
    ]
    with pytest.raises(RuntimeError, match="immutable manifest"):
        _bind_local_execution_evidence(
            conflicting_rows,
            repo_root=tmp_path,
            run_id="local-run",
        )
    conflicting_backend_rows = [dict(row, evidence_backend="weave") for row in rows]
    with pytest.raises(RuntimeError, match="evidence backend"):
        _bind_local_execution_evidence(
            conflicting_backend_rows,
            repo_root=tmp_path,
            run_id="local-run",
        )

    hosted_destination = trace_destination_identity(
        {"FUGUE_WEAVE_PROJECT": "wandb/hosted-evidence"}
    )
    hosted_rows = [
        dict(
            row,
            trace_project="wandb/hosted-evidence",
            trace_receipt=hosted_destination,
        )
        for row in rows
    ]
    for row in hosted_rows:
        row.pop("hosted_evidence_privacy_scan_status", None)
    _bind_local_execution_evidence(
        hosted_rows,
        repo_root=tmp_path,
        run_id="local-run",
        hosted_evidence_expected=True,
    )

    assert {row["trace_project"] for row in hosted_rows} == {"wandb/hosted-evidence"}
    assert {row["trace_receipt"]["destination_digest"] for row in hosted_rows} == {
        hosted_destination["destination_digest"]
    }
    assert all("hosted_evidence_privacy_scan_status" not in row for row in hosted_rows)
    assert {row["local_evidence_manifest_digest"] for row in hosted_rows} == {"3" * 64}


def test_local_checkpoint_verifies_and_persists_source_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison_path = scaffold_comparison(tmp_path / "study")
    spec = load_comparison(comparison_path, repo_root=comparison_path.parent)
    attempt_ids = ("1" * 64, "2" * 64)
    rows = {
        attempt_id: {"attempt_id": attempt_id, "pass": True}
        for attempt_id in attempt_ids
    }
    verified: list[str] = []
    monkeypatch.setattr(
        "fugue.bench.comparison._local_comparison_prediction_row",
        lambda *, attempt_id, **_kwargs: dict(rows[attempt_id]),
    )
    monkeypatch.setattr(
        "fugue.bench.comparison._verify_v3_source_drift",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "fugue.bench.comparison._verified_approved_inputs",
        lambda approved, **_kwargs: verified.append(approved["source_lock_digest"]),
    )
    approved = {"source_lock_digest": "a" * 64}
    run_dir = tmp_path / "study/.fugue/runtime/local-checkpoint-run"
    run_dir.mkdir(parents=True)

    drift = _restore_or_verify_checkpoint_receipt(
        spec=spec,
        readiness={},
        approved_comparison=approved,
        repo_root=comparison_path.parent,
        env={},
        run_id="local-checkpoint-run",
        schedule_digest="b" * 64,
        checkpoint_attempt_ids=attempt_ids,
    )

    assert drift is not None and drift.status == "matched"
    assert verified == ["a" * 64]
    receipt_path = run_dir / "comparison-checkpoint.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["source_drift"] == {
        "status": "matched",
        "expected_digest": "a" * 64,
        "observed_digest": "a" * 64,
    }

    unsigned = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    unsigned["source_drift"] = None
    receipt_path.write_text(
        json.dumps(
            {**unsigned, "receipt_digest": stable_digest(unsigned)},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="missing its source-lock verification"):
        _restore_or_verify_checkpoint_receipt(
            spec=spec,
            readiness={},
            approved_comparison=approved,
            repo_root=comparison_path.parent,
            env={},
            run_id="local-checkpoint-run",
            schedule_digest="b" * 64,
            checkpoint_attempt_ids=attempt_ids,
        )

    unsigned["source_drift"] = {
        "status": "drifted",
        "expected_digest": "a" * 64,
        "observed_digest": "b" * 64,
    }
    receipt_path.write_text(
        json.dumps(
            {**unsigned, "receipt_digest": stable_digest(unsigned)},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="did not match"):
        _restore_or_verify_checkpoint_receipt(
            spec=spec,
            readiness={},
            approved_comparison=approved,
            repo_root=comparison_path.parent,
            env={},
            run_id="local-checkpoint-run",
            schedule_digest="b" * 64,
            checkpoint_attempt_ids=attempt_ids,
        )

    unsigned["source_drift"] = {
        "status": "matched",
        "expected_digest": "c" * 64,
        "observed_digest": "c" * 64,
    }
    receipt_path.write_text(
        json.dumps(
            {**unsigned, "receipt_digest": stable_digest(unsigned)},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="approved source lock"):
        _restore_or_verify_checkpoint_receipt(
            spec=spec,
            readiness={},
            approved_comparison=approved,
            repo_root=comparison_path.parent,
            env={},
            run_id="local-checkpoint-run",
            schedule_digest="b" * 64,
            checkpoint_attempt_ids=attempt_ids,
        )

    drifted_run_dir = tmp_path / "study/.fugue/runtime/drifted-checkpoint-run"
    drifted_run_dir.mkdir(parents=True)

    def drifted_inputs(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("frozen input changed")

    monkeypatch.setattr(
        "fugue.bench.comparison._verified_approved_inputs",
        drifted_inputs,
    )
    with pytest.raises(RuntimeError, match="frozen input changed"):
        _restore_or_verify_checkpoint_receipt(
            spec=spec,
            readiness={},
            approved_comparison=approved,
            repo_root=comparison_path.parent,
            env={},
            run_id="drifted-checkpoint-run",
            schedule_digest="b" * 64,
            checkpoint_attempt_ids=attempt_ids,
        )
    assert not (drifted_run_dir / "comparison-checkpoint.json").exists()


def test_v3_attempt_construction_rejects_exported_decision_field_mutation() -> None:
    row: dict[str, object] = {
        "attempt_id": "1" * 64,
        "attempt_identity": {"task_id": "task-1"},
        "prediction_id": "prediction-1",
        "pass": True,
        "status": "passed",
        "comparison_evaluation_status": "scored",
        "comparison_deterministic_scores": {"facts.answer_correct": True},
        "comparison_score_details": {
            "facts.answer_correct": {
                "what": "Checks whether the answer is correct.",
                "observed": "The host scorer matched the required facts.",
                "why": "The outcome check passed.",
            }
        },
        "agent_response": "bounded answer",
        "cost_usd": 0.25,
        "latency_sec": 1.5,
        "usage": {"input_tokens": 100, "output_tokens": 25},
        "cost_reconciliation_status": "resolved",
        "latency_reconciliation_status": "resolved",
        "usage_reconciliation_status": "resolved",
        "local_evidence_links": [
            {
                "kind": kind,
                "status": "resolved",
                "system": "local_artifact",
                "ref": f"fugue://local/{kind}",
            }
            for kind in (
                "evaluation_root",
                "prediction_and_score",
                "prediction",
                "agent_root",
                "dataset",
            )
        ],
        "local_evidence_record_digest": "2" * 64,
        "local_evidence_prediction_row_sha256": "3" * 64,
    }
    row["local_evidence_result_row_projection_digest"] = (
        local_result_row_projection_digest(row)
    )
    attempt = _paired_attempt_view_v3(row)
    assert attempt is not None
    assert attempt.passed is True
    assert attempt.cost_reconciliation_status == "resolved"
    assert attempt.latency_reconciliation_status == "resolved"
    assert attempt.usage_reconciliation_status == "resolved"
    assert attempt.score_details["facts.answer_correct"].what == (
        "Checks whether the answer is correct."
    )

    mutations = (
        {"pass": False},
        {"comparison_deterministic_scores": {"facts.answer_correct": False}},
        {
            "comparison_score_details": {
                "facts.answer_correct": {
                    "what": "Altered criterion.",
                    "observed": "The host scorer matched the required facts.",
                    "why": "The outcome check passed.",
                }
            }
        },
        {"agent_response": "altered excerpt"},
        {"cost_usd": 99.0},
        {"cost_reconciliation_status": "unresolved"},
        {"latency_reconciliation_status": "unresolved"},
        {"usage_reconciliation_status": "unresolved"},
    )
    for mutation in mutations:
        altered = {**row, **mutation}
        with pytest.raises(ValueError, match="decision projection digest"):
            _paired_attempt_view_v3(altered)

    with pytest.raises(ValueError, match="decision projection digest"):
        replace(
            attempt,
            score_explanations={"facts.answer_correct": "altered explanation"},
        )

    serialized = attempt.to_dict()
    assert serialized["cost_reconciliation_status"] == "resolved"
    assert serialized["latency_reconciliation_status"] == "resolved"
    assert serialized["usage_reconciliation_status"] == "resolved"
    serialized["cost_usd"] = 88.0
    with pytest.raises(ValueError, match="decision projection digest"):
        _paired_attempt_v3(serialized)


def test_v3_attempt_reconciliation_statuses_are_source_authoritative() -> None:
    recorded_metrics: dict[str, object] = {
        "attempt_id": "4" * 64,
        "status": "passed",
        "cost_usd": 0.25,
        "latency_sec": 1.5,
        "usage": {"input_tokens": 100, "output_tokens": 25},
        "local_usage_status": "available",
        "weave_usage_status": "available",
    }

    projection = local_result_row_projection_v1(recorded_metrics)
    for field_name in (
        "cost_reconciliation_status",
        "latency_reconciliation_status",
        "usage_reconciliation_status",
    ):
        assert field_name not in projection

    explicit_nulls = {
        **recorded_metrics,
        "cost_reconciliation_status": None,
        "latency_reconciliation_status": None,
        "usage_reconciliation_status": None,
    }
    null_projection = local_result_row_projection_v1(explicit_nulls)
    assert null_projection == projection

    published = {
        **recorded_metrics,
        "cost_reconciliation_status": "resolved",
        "latency_reconciliation_status": "unresolved",
        "usage_reconciliation_status": "unavailable",
    }
    published_projection = local_result_row_projection_v1(published)
    assert published_projection["cost_reconciliation_status"] == "resolved"
    assert published_projection["latency_reconciliation_status"] == "unresolved"
    assert published_projection["usage_reconciliation_status"] == "unavailable"

    with pytest.raises(ValueError, match="cost_reconciliation_status must be one of"):
        local_result_row_projection_v1(
            {**recorded_metrics, "cost_reconciliation_status": "available"}
        )
