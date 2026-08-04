from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml

from fugue.bench.candidates import attempt_id, attempt_identity, stable_digest
from fugue.bench.comparison import (
    APPROVED_COMPARISON_LOCK_NAME,
    COMPARISON_PRIMARY_ARTIFACT_CONTRACT_VERSION,
    COMPARISON_PRIMARY_ARTIFACT_MAX_BYTES,
    COMPARISON_RESULT_ROOT,
    COMPARISON_RUNTIME_ROOT,
    ComparisonEvaluatorV1,
    ComparisonPostTrialVerifierV1,
    ComparisonResultV3,
    ComparisonSpecV1,
    DecisionAttestationV1,
    _apply_decision_attestation,
    _candidate_source_revisions,
    _canonical_decision_gate_policies,
    _comparison_judge_response_schema,
    _comparison_primary_output_projection,
    _comparison_qualification_digest,
    _comparison_reserved_cost_per_attempt,
    _comparison_result_digest,
    _comparison_run_exportability_issue,
    _comparison_trial_output_with_receipt,
    _ComparisonRuntimeBudget,
    _evaluate_decision,
    _evaluator_digest,
    _evaluator_runtime_readiness,
    _judge_review_label,
    _paired_attempt_view,
    _paired_attempt_view_v3,
    _prepare_comparison_judge_input,
    _prepare_evaluator_runtimes,
    _reported_project_identity,
    _request_comparison_judge,
    _require_checkpoint_judges,
    _sanitized_answer_excerpt,
    _validate_comparison_judge_payload,
    analyze_comparison_rows,
    attest_comparison_decision,
    check_comparison,
    claim_comparison_approval,
    comparison_from_dict,
    comparison_judge_public_rubric_contract,
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
from fugue.bench.evaluations import (
    JUDGE_JSON_MAX_RESPONSE_CHARACTERS,
    JudgeResponseError,
)
from fugue.bench.operator import OperatorService, PreviewCellSummary
from fugue.model_plane import trace_destination_identity
from fugue.research.approvals import ApprovalLedger
from fugue.research.contracts import ResearchError
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


def _primary_output_receipt(
    value: object,
    *,
    source: str = "row_output",
) -> dict[str, object]:
    encoded = (
        value.encode()
        if isinstance(value, str)
        else json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    )
    return {
        "schema_version": 1,
        "status": "complete",
        "source": source,
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "truncated": False,
    }


def _fake_scorer_receipts(
    *,
    source: str,
    evidence: Mapping[str, object],
    reference: Mapping[str, object],
    profile: object,
) -> dict[str, object]:
    serialized_input = json.dumps(
        {"evidence": evidence, "reference": reference},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    input_unsigned = {
        "schema_version": 1,
        "status": "bound",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "input_bytes": len(serialized_input),
        "input_sha256": hashlib.sha256(serialized_input).hexdigest(),
        "evidence_digest": stable_digest(evidence),
        "reference_digest": stable_digest(reference),
        "reference_output_digest": stable_digest(reference.get("output")),
        "runtime_profile_id": profile.id,
        "runtime_profile_digest": profile.profile_digest,
        "runtime_image": profile.image,
        "runtime_platform": profile.platform,
    }
    runtime_unsigned = {
        "schema_version": 1,
        "status": "verified_absent",
        "container_name_sha256": "7" * 64,
    }
    return {
        "fugue_input_receipt": {
            **input_unsigned,
            "receipt_digest": stable_digest(input_unsigned),
        },
        "fugue_runtime_receipt": {
            **runtime_unsigned,
            "receipt_digest": stable_digest(runtime_unsigned),
        },
    }


def test_evaluator_runtime_preparation_locks_exact_profile_image_and_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = SimpleNamespace(
        id="node22-verifier-v1",
        profile_digest="a" * 64,
        image="example/verifier@sha256:" + "b" * 64,
        platform="linux/arm64",
        command=("node", "/input/scorer.py", "/input/input.json"),
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(
        "fugue.bench.comparison._comparison_evaluator_runtime_profiles",
        lambda spec, repo_root: (profile,),
    )
    monkeypatch.setattr("fugue.bench.comparison.shutil.which", lambda _: "/docker")

    def fake_run(command, **kwargs):
        commands.append(command)
        return __import__("subprocess").CompletedProcess(command, 0, "pulled", "")

    image = {
        "Id": "sha256:" + "c" * 64,
        "Os": "linux",
        "Architecture": "arm64",
    }
    monkeypatch.setattr("fugue.bench.comparison.subprocess.run", fake_run)
    monkeypatch.setattr("fugue.bench.comparison.inspect_docker_image", lambda _: image)

    locks = _prepare_evaluator_runtimes(SimpleNamespace(), repo_root=tmp_path)
    assert len(locks) == 1
    assert locks[0]["profile_digest"] == profile.profile_digest
    assert locks[0]["image_id"] == image["Id"]
    assert locks[0]["platform"] == "linux/arm64"
    assert commands == [
        ["/docker", "pull", "--platform", "linux/arm64", profile.image]
    ]
    digests, blockers = _evaluator_runtime_readiness(
        SimpleNamespace(), repo_root=tmp_path
    )
    assert not blockers
    assert digests == {
        "evaluator:node22-verifier-v1:linux/arm64": locks[0]["lock_digest"]
    }


def test_evaluator_runtime_profile_drift_invalidates_prepared_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = SimpleNamespace(
        id="node22-verifier-v1",
        profile_digest="a" * 64,
        image="example/verifier@sha256:" + "b" * 64,
        platform="linux/arm64",
        command=("node", "/input/scorer.py", "/input/input.json"),
    )
    monkeypatch.setattr(
        "fugue.bench.comparison._comparison_evaluator_runtime_profiles",
        lambda spec, repo_root: (original,),
    )
    monkeypatch.setattr("fugue.bench.comparison.shutil.which", lambda _: "/docker")
    monkeypatch.setattr(
        "fugue.bench.comparison.subprocess.run",
        lambda command, **kwargs: __import__("subprocess").CompletedProcess(
            command, 0, "pulled", ""
        ),
    )
    monkeypatch.setattr(
        "fugue.bench.comparison.inspect_docker_image",
        lambda _: {
            "Id": "sha256:" + "c" * 64,
            "Os": "linux",
            "Architecture": "arm64",
        },
    )
    _prepare_evaluator_runtimes(SimpleNamespace(), repo_root=tmp_path)
    drifted = SimpleNamespace(**{**vars(original), "command": ("node", "changed")})
    monkeypatch.setattr(
        "fugue.bench.comparison._comparison_evaluator_runtime_profiles",
        lambda spec, repo_root: (drifted,),
    )
    digests, blockers = _evaluator_runtime_readiness(
        SimpleNamespace(), repo_root=tmp_path
    )
    assert not digests
    assert blockers == [
        "local evaluator:node22-verifier-v1:linux/arm64 is not prepared and "
        "locked; run `fugue compare SPEC --prepare`"
    ]


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
        input_bindings={
            "trace_audit_selection_digest": "d" * 64,
        },
    )

    claimed = claim_comparison_approval(
        preview,
        approval_digest=approval.approval_digest,
        repo_root=tmp_path,
        execution_instance_id="comparison-run-one",
    )
    assert claimed.input_bindings == {
        "trace_audit_selection_digest": "d" * 64,
    }
    claim_comparison_approval(
        preview,
        approval_digest=approval.approval_digest,
        repo_root=tmp_path,
        execution_instance_id="comparison-run-one",
    )
    with pytest.raises(ResearchError) as reused:
        claim_comparison_approval(
            preview,
            approval_digest=approval.approval_digest,
            repo_root=tmp_path,
            execution_instance_id="comparison-run-two",
        )
    assert reused.value.code == "approval_consumed"


def test_approval_input_bindings_flow_into_the_attempt_execution_lock(
    tmp_path: Path,
) -> None:
    comparison_path = scaffold_comparison(tmp_path)
    spec = load_comparison(comparison_path, repo_root=tmp_path)
    preview = preview_comparison(spec, repo_root=tmp_path)
    _experiment, request = materialize_comparison(
        preview,
        repo_root=tmp_path,
        approval_digest="a" * 64,
        approval_input_bindings={
            "trace_audit_selection_digest": "b" * 64,
            "trace_audit_selection_file_sha256": "c" * 64,
        },
    )

    lock = request.approved_comparison
    assert lock["approval_input_bindings"] == {
        "trace_audit_selection_digest": "b" * 64,
        "trace_audit_selection_file_sha256": "c" * 64,
    }
    assert lock["qualification_input_digests"] == lock[
        "approval_input_bindings"
    ]


def test_declared_qualification_input_is_preview_and_approval_bound(
    tmp_path: Path,
) -> None:
    comparison_path = scaffold_comparison(tmp_path)
    input_path = tmp_path / "analysis-profile.json"
    input_path.write_text('{"status":"frozen"}\n', encoding="utf-8")
    spec = load_comparison(comparison_path, repo_root=tmp_path)
    spec = comparison_from_dict(
        replace(
            spec,
            spec_digest="",
            execution=replace(
                spec.execution,
                qualification_inputs={
                    "confirmatory_analysis_profile_sha256": input_path.name,
                },
            ),
        ).to_dict(),
        repo_root=tmp_path,
        source=tmp_path,
    )
    preview = preview_comparison(spec, repo_root=tmp_path)
    expected = hashlib.sha256(input_path.read_bytes()).hexdigest()
    assert preview.readiness["qualification_input_digests"] == {
        "confirmatory_analysis_profile_sha256": expected,
    }
    _experiment, request = materialize_comparison(
        preview,
        repo_root=tmp_path,
        approval_digest="a" * 64,
    )
    assert request.approved_comparison["qualification_input_digests"] == {
        "confirmatory_analysis_profile_sha256": expected,
    }

    input_path.write_text('{"status":"changed"}\n', encoding="utf-8")
    changed = preview_comparison(spec, repo_root=tmp_path)
    assert changed.preview_digest != preview.preview_digest


def test_execute_comparison_allocates_run_identity_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview = SimpleNamespace(preview_digest="a" * 64)
    captured: dict[str, object] = {}
    expected = (object(), tmp_path / "result.json", tmp_path / "result.md")

    monkeypatch.setattr(
        "fugue.bench.execution.new_run_id",
        lambda: "comparison-run-direct",
    )

    def execute(*args: object, **kwargs: object) -> object:
        captured["run_id"] = kwargs["run_id"]
        captured["context_run_id"] = kwargs["projection_context"].run_id
        return expected

    monkeypatch.setattr("fugue.bench.comparison._execute_comparison", execute)

    assert execute_comparison(
        preview,  # type: ignore[arg-type]
        approval_digest="b" * 64,
        repo_root=tmp_path,
    ) == expected
    assert captured == {
        "run_id": "comparison-run-direct",
        "context_run_id": "comparison-run-direct",
    }


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


def test_comparison_compiles_declared_counterbalanced_schedule() -> None:
    root = Path.cwd()
    raw = yaml.safe_load((EXAMPLE / "comparison.yaml").read_text())
    raw["execution"]["scheduling_seed"] = "comparison-confirmatory-v1"
    spec = comparison_from_dict(raw, repo_root=root, source=EXAMPLE)

    experiment, _, _ = compile_comparison(spec, repo_root=root)

    assert spec.execution.scheduling_seed == "comparison-confirmatory-v1"
    assert experiment.default_preset == "counterbalanced"
    assert len(experiment.presets) == 1
    assert experiment.presets[0].scheduling_seed == "comparison-confirmatory-v1"
    assert experiment.artifacts == [
        {"source": "/logs/artifacts/fugue-answer.md"}
    ]


def test_runtime_budget_blocks_parallel_paid_launches() -> None:
    root = Path.cwd()
    raw = yaml.safe_load((EXAMPLE / "comparison.yaml").read_text())
    raw["execution"]["evidence_checkpoint_cells"] = 0
    raw["execution"]["concurrency"] = 2
    raw["execution"]["reserve_per_attempt_usd"] = 1
    raw["execution"]["max_cost_usd"] = 16
    spec = comparison_from_dict(raw, repo_root=root, source=EXAMPLE)

    readiness = check_comparison(spec, repo_root=root)

    assert readiness.status == "blocked"
    assert "runtime budget enforcement requires comparison concurrency=1" in (
        readiness.blockers
    )


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
        "accounted_cost_usd": None,
        "accounted_cost_rows": 0,
        "latency_ms": None,
        "latency_rows": 0,
        "input_tokens": None,
        "output_tokens": None,
        "usage_rows": 0,
        "evidence_projects": [],
        "mcp_tool_usage": {},
        "agent_trajectory_tool_activity": {},
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
        "conversation_correlation_verified": True,
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


def test_v2_historical_null_attempt_outcomes_keep_their_digest_bytes(
    tmp_path: Path,
) -> None:
    result = analyze_comparison_rows(
        comparison_id="historical-v2-result",
        preview_digest="a" * 64,
        rows=[
            _decision_row(variant="baseline", passed=False),
            _decision_row(variant="candidate"),
        ],
        source="test",
        expected_evidence_project="wandb/release-project",
        decision_policy=_decision_policy(),
    )
    raw = result.to_dict()
    for pair in raw["paired_cases"]:
        for arm in ("baseline", "candidate"):
            attempt = pair[arm]
            assert attempt["benchmark_outcome"] is None
            assert attempt["runtime_outcome"] is None
            attempt.pop("weave_agent_evidence_call_id", None)
            for link in attempt["evidence_links"]:
                link.pop("evidence_kind", None)
                link.pop("native_trajectory_status", None)
                link.pop("conversation_correlation_status", None)
    raw["qualification_digest"] = _comparison_qualification_digest(raw)
    raw["result_digest"] = raw["qualification_digest"]
    path = tmp_path / "historical-v2-result.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    reloaded = read_comparison_result(path)

    assert reloaded.to_dict() == raw


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
    preview_by_attempt = {
        str(cell["attempt_id"]): cell for cell in preview.matrix["matrix_cells"]
    }
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
        row["skill_provenance"] = list(
            preview_by_attempt[str(cell["attempt_id"])]["skill_provenance"]
        )
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
        row["source_post_run_drift"] = drift
        row["prediction_id"] = f"{variant}-prediction-row"
        row["agent_response"] = {
            "project": source_project,
            "answer": "safe maintainer summary",
        }
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

    baseline_timeout = next(
        row for row in rows if row["variant_id"] == "baseline"
    )
    baseline_timeout["benchmark_outcome"] = "failed"
    baseline_timeout["runtime_outcome"] = "timed_out"

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
    timeout_pair = next(
        pair
        for pair in result.paired_cases
        if pair.baseline.attempt_id == baseline_timeout["attempt_id"]
    )
    assert timeout_pair.baseline.benchmark_outcome == "failed"
    assert timeout_pair.baseline.runtime_outcome == "timed_out"
    assert result.operational_summary["agent_timeouts"] == 1
    assert result.operational_summary["infrastructure_failures"] == 0

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
    with pytest.raises(ValueError, match="exactly one typed Agent evidence link"):
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
    assert view.evidence_eligible is True
    assert view.infrastructure_health != "failed"
    timeout_view_pair = next(
        pair
        for pair in view.paired_cases
        if pair["baseline"]["attempt_id"] == baseline_timeout["attempt_id"]
    )
    assert timeout_view_pair["baseline"]["benchmark_outcome"] == "failed"
    assert timeout_view_pair["baseline"]["runtime_outcome"] == "timed_out"
    assert any(
        "task/runtime failures, not infrastructure failures" in item
        for item in view.limitations
    )

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

    signing_rows = []
    for row in rows:
        signing_row = dict(row)
        signing_row.pop("approved_comparison")
        signing_row["approved_comparison_lock_digest"] = approved["lock_digest"]
        signing_rows.append(signing_row)
    referenced_release_result = analyze_comparison_rows(
        comparison_id=spec.id,
        preview_digest=preview.preview_digest,
        rows=signing_rows,
        source="v3-run",
        expected_evidence_project=result_project,
        expected_source_evidence_project=source_project,
        approved_comparison=approved,
        result_schema_version=3,
        study_intent="mcp_release_maintenance",
        supersedes=spec.supersedes,
        decision_policy=_decision_policy(),
    )
    assert referenced_release_result.decision.status == "ready_for_signoff"
    signing_destination = tmp_path / "signing-result"
    signing_destination.mkdir()
    (signing_destination / "attempts.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in signing_rows) + "\n",
        encoding="utf-8",
    )
    write_comparison_result(
        referenced_release_result,
        destination=signing_destination,
        approved_comparison=approved,
    )
    signed_from_references = attest_comparison_decision(
        result_path=signing_destination / "result.json",
        signer="release-owner",
        signed_at="2026-07-29T00:00:00Z",
    )
    assert signed_from_references.decision.status == "go"

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

    blocked_improvement_rows = json.loads(json.dumps(rows))
    blocked_candidate = next(
        row
        for row in blocked_improvement_rows
        if row["variant_id"] == "candidate"
    )
    blocked_candidate["pass"] = False
    blocked_candidate["comparison_deterministic_scores"][
        "release.locked_project_scope"
    ] = False
    blocked_task_id = str(blocked_candidate["task_id"])
    blocked_harness = str(blocked_candidate["harness"])
    blocked_attempt = int(blocked_candidate["trial_index"])
    blocked_improvement = analyze_comparison_rows(
        comparison_id=spec.id,
        preview_digest=preview.preview_digest,
        rows=blocked_improvement_rows,
        source="v3-run",
        expected_evidence_project=result_project,
        expected_source_evidence_project=source_project,
        approved_comparison=approved,
        result_schema_version=3,
        study_intent="mcp_release_maintenance",
        supersedes=spec.supersedes,
    )
    blocked_pair = next(
        pair
        for pair in blocked_improvement.paired_cases
        if pair.task_id == blocked_task_id
        and pair.harness == blocked_harness
        and pair.attempt == blocked_attempt
    )
    assert {
        change.id: change.status for change in blocked_pair.dimension_changes
    } == {
        "release.factual_correctness": "improved",
        "release.locked_project_scope": "unchanged",
    }
    assert blocked_pair.status == "unchanged"
    assert blocked_improvement.behavioral_summary.status == "unchanged"
    assert blocked_improvement.behavioral_summary.critical_blockers == (
        f"{blocked_pair.task_id}: release.locked_project_scope failed for the "
        "candidate",
    )
    assert blocked_improvement.behavioral_summary.supported_claim == (
        "No release-qualifying improvement was established across 2 aligned "
        "pair(s)."
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


@pytest.mark.parametrize(
    "timeout_evidence",
    ("explicit_outcomes", "legacy_exception", "legacy_structured_event"),
)
def test_agent_timeout_is_replayed_as_task_runtime_outcome_not_infrastructure(
    tmp_path: Path,
    timeout_evidence: str,
) -> None:
    baseline = _decision_row(variant="baseline", passed=False)
    candidate = _decision_row(variant="candidate", passed=True)
    if timeout_evidence == "explicit_outcomes":
        baseline["benchmark_outcome"] = "failed"
        baseline["runtime_outcome"] = "timed_out"
    elif timeout_evidence == "legacy_exception":
        baseline["exception_class"] = "AgentTimeoutError"
    else:
        baseline["error_events"] = [
            {
                "origin": "agent",
                "kind": "agent_timeout",
                "terminal": True,
            }
        ]

    rows = [baseline, candidate]
    result = analyze_comparison_rows(
        comparison_id=f"agent-timeout-{timeout_evidence}",
        preview_digest="c" * 64,
        rows=rows,
        source="agent-timeout-replay",
        expected_evidence_project="wandb/release-project",
    )

    assert result.operational_summary["agent_timeouts"] == 1
    assert result.operational_summary["infrastructure_failures"] == 0
    timeout_attempt = result.paired_cases[0].baseline
    assert timeout_attempt is not None
    assert timeout_attempt.execution_status == "completed"
    assert timeout_attempt.passed is False
    assert timeout_attempt.benchmark_outcome == "failed"
    assert timeout_attempt.runtime_outcome == "timed_out"

    destination = tmp_path / timeout_evidence
    destination.mkdir()
    (destination / "attempts.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    write_comparison_result(result, destination=destination)
    assert read_comparison_result(destination / "result.json") == result

    view = build_comparison_evaluation_view(result.to_dict())
    assert isinstance(view, ExperimentViewV2)
    assert view.evidence_eligible is True
    assert view.infrastructure_health != "failed"
    assert view.paired_cases[0]["baseline"]["benchmark_outcome"] == "failed"
    assert view.paired_cases[0]["baseline"]["runtime_outcome"] == "timed_out"
    assert any(
        "task/runtime failures, not infrastructure failures" in item
        for item in view.limitations
    )
    assert experiment_view_from_dict(view.to_dict()) == view


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
    assert result.behavioral_summary.status == "unchanged"
    assert result.behavioral_summary.candidate_critical_failures == 2
    assert result.behavioral_summary.critical_blockers == (
        "release-task: release.factual_correctness failed for the candidate",
        "release-task: release.locked_project_scope failed for the candidate",
    )
    assert result.behavioral_summary.supported_claim is None


def test_improvement_with_shared_critical_failure_is_unchanged_and_blocked() -> None:
    baseline = _decision_row(variant="baseline", passed=False)
    candidate = _decision_row(variant="candidate", passed=False)
    baseline["comparison_deterministic_scores"] = {
        "release.missing_evidence_status": False,
        "release.terminal_success_or_stop_semantics": False,
    }
    candidate["comparison_deterministic_scores"] = {
        "release.missing_evidence_status": True,
        "release.terminal_success_or_stop_semantics": False,
    }
    criticality = {
        "release.missing_evidence_status": True,
        "release.terminal_success_or_stop_semantics": True,
    }
    baseline["comparison_deterministic_criticality"] = criticality
    candidate["comparison_deterministic_criticality"] = criticality

    result = analyze_comparison_rows(
        comparison_id="improved-but-blocked",
        preview_digest="b" * 64,
        rows=[baseline, candidate],
        source="test",
        expected_evidence_project="wandb/release-project",
    )

    pair = result.paired_cases[0]
    assert {change.id: change.status for change in pair.dimension_changes} == {
        "release.missing_evidence_status": "improved",
        "release.terminal_success_or_stop_semantics": "unchanged",
    }
    # A shared failure is not a regression, so it cannot make the pair mixed.
    # It still prevents a release-qualifying "improved" verdict.
    assert pair.status == "unchanged"
    assert result.mixed == 0
    assert result.regressed == 0
    assert result.behavioral_summary.status == "unchanged"
    assert result.behavioral_summary.candidate_critical_failures == 1
    assert result.behavioral_summary.critical_blockers == (
        "release-task: release.terminal_success_or_stop_semantics failed for "
        "the candidate",
    )
    assert result.behavioral_summary.supported_claim is None


def test_pure_regression_is_not_relabelled_mixed_by_candidate_failure() -> None:
    result = analyze_comparison_rows(
        comparison_id="pure-regression",
        preview_digest="b" * 64,
        rows=[
            _decision_row(variant="baseline", passed=True),
            _decision_row(variant="candidate", passed=False),
        ],
        source="test",
        expected_evidence_project="wandb/release-project",
    )

    assert result.paired_cases[0].status == "regressed"
    assert result.behavioral_summary.status == "regressed"
    assert result.regressed == 1
    assert result.mixed == 0


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


def test_comparison_recovers_usage_and_agent_activity_from_export_fields() -> None:
    baseline = _decision_row(variant="baseline", passed=False)
    candidate = _decision_row(variant="candidate")
    for row, latency, input_tokens, output_tokens, call_count, names in (
        (baseline, 12.5, 100, 20, 12, {"Bash": 8, "Read": 4}),
        (candidate, 14.25, 150, 30, 14, {"Bash": 9, "Read": 5}),
    ):
        row.pop("latency_sec")
        row["wall_time_sec"] = latency
        row["local_usage_status"] = "available"
        row["n_input_tokens"] = input_tokens
        row["n_output_tokens"] = output_tokens
        row["mcp_tool_names"] = []
        row["mcp_tool_calls"] = []
        row["agent_trajectory_tool_activity_status"] = "available"
        row["agent_trajectory_tool_call_count"] = call_count
        row["agent_trajectory_tool_names"] = names
    # Native summaries may be cumulative or missing. The local trajectory is
    # the attempt-scoped fallback for general Agent activity only.
    baseline["weave_tool_names"] = {"Bash": 24, "Read": 12}
    candidate["weave_tool_names"] = {}

    result = analyze_comparison_rows(
        comparison_id="agent-trajectory-telemetry",
        preview_digest="b" * 64,
        rows=[baseline, candidate],
        source="test",
        expected_evidence_project="wandb/release-project",
    )

    baseline_attempt = result.paired_cases[0].baseline
    candidate_attempt = result.paired_cases[0].candidate
    assert baseline_attempt is not None
    assert candidate_attempt is not None
    assert baseline_attempt.latency_sec == 12.5
    assert candidate_attempt.latency_sec == 14.25
    assert (baseline_attempt.input_tokens, baseline_attempt.output_tokens) == (
        100.0,
        20.0,
    )
    assert (candidate_attempt.input_tokens, candidate_attempt.output_tokens) == (
        150.0,
        30.0,
    )
    assert baseline_attempt.tool_calls == 12
    assert baseline_attempt.tools == ("Bash", "Read")
    assert candidate_attempt.tool_calls == 14
    assert candidate_attempt.tools == ("Bash", "Read")
    assert result.operational_summary["latency_ms"] == 26750.0
    assert result.operational_summary["latency_rows"] == 2
    assert result.operational_summary["input_tokens"] == 250
    assert result.operational_summary["output_tokens"] == 50
    assert result.operational_summary["usage_rows"] == 2
    # Agent trajectory fallback never fabricates MCP mechanism evidence.
    assert result.operational_summary["mcp_tool_usage"] == {}
    assert result.operational_summary["agent_trajectory_tool_activity"] == {
        "baseline": {
            "calls": 12,
            "rows": 1,
            "tools": {"Bash": 8, "Read": 4},
        },
        "candidate": {
            "calls": 14,
            "rows": 1,
            "tools": {"Bash": 9, "Read": 5},
        },
    }


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


@pytest.mark.parametrize("correlation", [False, None])
def test_unverified_conversation_correlation_invalidates_behavioral_evidence(
    correlation: bool | None,
) -> None:
    baseline = _decision_row(variant="baseline", passed=False)
    candidate = _decision_row(variant="candidate", passed=True)
    if correlation is None:
        candidate.pop("conversation_correlation_verified")
    else:
        candidate["conversation_correlation_verified"] = correlation

    result = analyze_comparison_rows(
        comparison_id="invalid-conversation-correlation",
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
    agent_link = next(
        link
        for link in candidate_attempt.evidence_links
        if link.kind == "agent_root"
    )
    assert agent_link.status == "invalid"
    assert agent_link.conversation_correlation_status == (
        "unverified" if correlation is False else "not_recorded"
    )


def test_cross_transport_receipt_is_not_presented_as_agent_root() -> None:
    baseline = _decision_row(variant="baseline", passed=False)
    candidate = _decision_row(variant="candidate", passed=True)
    receipt_call_id = str(candidate.pop("native_agent_root_call_id"))
    candidate.update(
        {
            "weave_agent_root_call_id": receipt_call_id,
            "weave_agent_root_evidence_kind": (
                "native_otel_cross_transport_receipt_v1"
            ),
            "weave_agent_root_is_native_call": False,
            "otel_trace_id": "a" * 32,
            "otel_root_span_id": "b" * 16,
            "conversation_correlation_verified": True,
            "agent_cross_transport_edge": {
                "schema_version": 1,
                "status": "verified",
                "source_system": "otel",
                "source_trace_id": "a" * 32,
                "source_span_id": "b" * 16,
                "receipt_system": "weave",
                "receipt_call_id": receipt_call_id,
            },
        }
    )

    result = analyze_comparison_rows(
        comparison_id="typed-agent-evidence",
        preview_digest="c" * 64,
        rows=[baseline, candidate],
        source="test",
        expected_evidence_project="wandb/release-project",
    )

    candidate_attempt = result.paired_cases[0].candidate
    assert candidate_attempt is not None
    receipt = next(
        link
        for link in candidate_attempt.evidence_links
        if link.kind == "agent_evidence_receipt"
    )
    assert receipt.status == "resolved"
    assert receipt.evidence_kind == "native_otel_cross_transport_receipt_v1"
    assert receipt.native_trajectory_status == "otel_correlated"
    assert receipt.conversation_correlation_status == "verified"
    assert candidate_attempt.weave_agent_root_call_id is None
    assert candidate_attempt.weave_agent_evidence_call_id == receipt_call_id

    view = build_comparison_evaluation_view(result.to_dict())
    candidate_view = view.paired_cases[0]["candidate"]
    receipt_view = next(
        link
        for link in candidate_view["evidence_links"]
        if link["kind"] == "agent_evidence_receipt"
    )
    assert receipt_view["native_trajectory_status"] == "otel_correlated"
    assert receipt_view["conversation_correlation_status"] == "verified"
    assert "weave_agent_root_call_id" not in candidate_view
    assert candidate_view["weave_agent_evidence_call_id"] == receipt_call_id


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


def test_answer_excerpt_and_reported_project_use_scored_primary_artifact() -> None:
    row = _decision_row(variant="candidate")
    primary = {
        "project": "wandb/scored-primary-project",
        "recommendation": "advance",
    }
    receipt = _primary_output_receipt(primary, source="fugue_answer_artifact")
    row["comparison_primary_output_projection"] = (
        _comparison_primary_output_projection(
            value=primary,
            receipt=receipt,
            env={},
            row=row,
        )
    )
    row["agent_response"] = json.dumps(
        {
            "project": "wandb/different-conversation-project",
            "recommendation": "hold",
        }
    )

    excerpt = _sanitized_answer_excerpt(row)

    assert excerpt is not None
    assert "scored-primary-project" in excerpt
    assert "different-conversation-project" not in excerpt
    assert _reported_project_identity(row) == "wandb/scored-primary-project"


def test_primary_artifact_projection_never_persists_excerpt_after_privacy_failure() -> None:
    row = _decision_row(variant="candidate")
    row["private_label_leak"] = True
    private_value = {"expected": "host-only-answer", "recommendation": "advance"}
    projection = _comparison_primary_output_projection(
        value=private_value,
        receipt=_primary_output_receipt(private_value),
        env={},
        row=row,
    )
    assert projection["status"] == "unavailable"
    assert "sanitized_excerpt" not in projection
    assert "reported_project_identity" not in projection
    assert "host-only-answer" not in json.dumps(projection)


def test_answer_excerpt_respects_serialized_utf8_byte_limit() -> None:
    row = _decision_row(variant="candidate")
    row["agent_response"] = "é" * 1_000

    excerpt = _sanitized_answer_excerpt(row)

    assert excerpt is not None
    assert len(excerpt.encode()) == 1_000

    row["agent_response"] = ("a" * 999) + " " + "remainder"
    excerpt = _sanitized_answer_excerpt(row)
    assert excerpt == "a" * 999


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
    assert approved["candidate_source_revisions_required"] is True
    assert len(approved["candidate_source_revisions"]) == 1
    assert approved["candidate_source_revisions"][0]["kind"] == "skill"
    assert approved["candidate_source_revisions"][0]["version_identity"].startswith(
        "digest:sha256:"
    )
    preview_by_attempt = {
        str(cell["attempt_id"]): cell for cell in preview.matrix["matrix_cells"]
    }
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
            "skill_provenance": list(
                preview_by_attempt[str(cell["attempt_id"])]["skill_provenance"]
            ),
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

    referenced_rows = []
    for row in rows:
        referenced = dict(row)
        referenced.pop("approved_comparison")
        referenced["approved_comparison_lock_digest"] = approved["lock_digest"]
        referenced_rows.append(referenced)
    referenced_result = analyze_comparison_rows(
        comparison_id=spec.id,
        preview_digest=preview.preview_digest,
        rows=referenced_rows,
        source="approved-run",
        approved_comparison=approved,
    )
    assert referenced_result.integrity["approved_manifest_status"] == "reconciled"
    referenced_destination = tmp_path / "referenced-result"
    referenced_destination.mkdir()
    (referenced_destination / "attempts.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in referenced_rows)
        + "\n",
        encoding="utf-8",
    )
    write_comparison_result(
        referenced_result,
        destination=referenced_destination,
        approved_comparison=approved,
    )
    assert json.loads(
        (referenced_destination / "approved-comparison.lock.json").read_text()
    ) == approved
    write_comparison_result(
        referenced_result,
        destination=referenced_destination,
    )
    sidecar_path = referenced_destination / "approved-comparison.lock.json"
    sidecar_payload = sidecar_path.read_text(encoding="utf-8")
    sidecar_path.unlink()
    with pytest.raises(ValueError, match="no approved execution lock"):
        write_comparison_result(
            referenced_result,
            destination=referenced_destination,
        )
    tampered_sidecar = json.loads(sidecar_payload)
    tampered_sidecar["lock_digest"] = "0" * 64
    sidecar_path.write_text(json.dumps(tampered_sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="digest does not match"):
        write_comparison_result(
            referenced_result,
            destination=referenced_destination,
        )
    sidecar_path.write_text(sidecar_payload, encoding="utf-8")

    wrong_reference = [dict(row) for row in referenced_rows]
    wrong_reference[0]["approved_comparison_lock_digest"] = "f" * 64
    with pytest.raises(ValueError, match="exact lock digest"):
        analyze_comparison_rows(
            comparison_id=spec.id,
            preview_digest=preview.preview_digest,
            rows=wrong_reference,
            source="approved-run",
            approved_comparison=approved,
        )
    with pytest.raises(ValueError, match="no approved execution lock"):
        analyze_comparison_rows(
            comparison_id=spec.id,
            preview_digest=preview.preview_digest,
            rows=referenced_rows,
            source="approved-run",
        )

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
    source_required["candidate_source_revisions"] = []
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
        _comparison_trial_output_with_receipt(
            {
                "trial_dir": trial_dir.as_posix(),
                "agent_response": "The answer was written to the artifact.",
            }
        )[0]
        == '{"answer": 42}'
    )
    assert _comparison_trial_output_with_receipt(
        {"agent_response": "terminal answer"}
    )[0] == "terminal answer"


def test_approved_scoring_rebinds_primary_output_from_the_host_artifact(
    tmp_path: Path,
) -> None:
    comparison_path = scaffold_comparison(tmp_path)
    spec = load_comparison(comparison_path, repo_root=tmp_path)
    preview = preview_comparison(spec, repo_root=tmp_path)
    _experiment, request = materialize_comparison(
        preview,
        repo_root=tmp_path,
        approval_digest="a" * 64,
    )
    approved = request.approved_comparison
    trial_dir = (
        tmp_path
        / ".fugue"
        / "runtime"
        / "jobs"
        / spec.id
        / "approved-run"
        / "job"
        / "trial"
    )
    answer = trial_dir / "artifacts" / "logs" / "artifacts" / "fugue-answer.md"
    answer.parent.mkdir(parents=True)
    host_output = '{"answer":"host-authenticated"}'
    answer.write_text(host_output, encoding="utf-8")
    expected_paths = ["/logs/artifacts/fugue-answer.md"]
    base_row = {
        "task_id": "unlabeled-task",
        "variant_id": "baseline",
        "harness": "codex",
        "run_id": "approved-run",
        "trial_dir": trial_dir.as_posix(),
        "expected_artifact_paths": expected_paths,
        "final_output": '{"answer":"caller-controlled"}',
    }

    [scored] = score_comparison_rows(
        spec,
        [base_row],
        repo_root=tmp_path,
        approved_comparison=approved,
    )

    assert scored["final_output"] == host_output
    assert scored["comparison_primary_output"]["sha256"] == hashlib.sha256(
        host_output.encode()
    ).hexdigest()

    forged_output = '{"answer":"forged-receipt"}'
    forged_encoded = forged_output.encode()
    forged_receipt = {
        "schema_version": 1,
        "status": "complete",
        "source": "fugue_answer_artifact",
        "source_path": "/logs/artifacts/fugue-answer.md",
        "path": "artifacts/logs/artifacts/fugue-answer.md",
        "contract_version": COMPARISON_PRIMARY_ARTIFACT_CONTRACT_VERSION,
        "locked_artifact_paths_digest": stable_digest(expected_paths),
        "bytes": len(forged_encoded),
        "sha256": hashlib.sha256(forged_encoded).hexdigest(),
        "truncated": False,
    }
    with pytest.raises(ValueError, match="extracted primary artifact receipt"):
        score_comparison_rows(
            spec,
            [
                {
                    **base_row,
                    "final_output": forged_output,
                    "comparison_primary_output": forged_receipt,
                }
            ],
            repo_root=tmp_path,
            approved_comparison=approved,
        )


def test_comparison_primary_artifact_is_complete_past_legacy_truncation(
    tmp_path: Path,
) -> None:
    trial_dir = tmp_path / "trial"
    answer = trial_dir / "artifacts" / "logs" / "artifacts" / "fugue-answer.md"
    answer.parent.mkdir(parents=True)
    expected = "complete-plan\n" + ("verified section\n" * 2_000)
    assert len(expected) > 16_000
    answer.write_text(expected, encoding="utf-8")

    value, receipt = _comparison_trial_output_with_receipt(
        {
            "trial_dir": trial_dir.as_posix(),
            "expected_artifact_paths": ["/logs/artifacts/fugue-answer.md"],
        }
    )

    assert value == expected
    assert receipt["bytes"] == len(expected.encode())
    assert receipt["sha256"] == hashlib.sha256(expected.encode()).hexdigest()


def test_canonical_primary_receipt_rejects_wrong_source_identity(
    tmp_path: Path,
) -> None:
    trial_dir = tmp_path / "trial"
    answer = trial_dir / "artifacts" / "logs" / "artifacts" / "fugue-answer.md"
    answer.parent.mkdir(parents=True)
    answer.write_text("complete", encoding="utf-8")
    value, receipt = _comparison_trial_output_with_receipt(
        {
            "trial_dir": trial_dir.as_posix(),
            "expected_artifact_paths": ["/logs/artifacts/fugue-answer.md"],
        }
    )
    receipt["source"] = "row_output"
    row = _decision_row(variant="candidate")
    row["expected_artifact_paths"] = ["/logs/artifacts/fugue-answer.md"]

    with pytest.raises(ValueError, match="canonical primary artifact identity"):
        _comparison_primary_output_projection(
            value=value,
            receipt=receipt,
            env={},
            row=row,
        )


@pytest.mark.parametrize("expected_paths", [[], "fugue-answer.md", ["other.md"]])
def test_comparison_primary_artifact_identity_fails_closed_when_bound_malformed(
    tmp_path: Path,
    expected_paths: object,
) -> None:
    trial_dir = tmp_path / "trial"
    answer = trial_dir / "artifacts" / "logs" / "artifacts" / "fugue-answer.md"
    answer.parent.mkdir(parents=True)
    answer.write_text("complete", encoding="utf-8")

    with pytest.raises(ValueError, match="expected artifact identity"):
        _comparison_trial_output_with_receipt(
            {
                "trial_dir": trial_dir.as_posix(),
                "expected_artifact_paths": expected_paths,
            }
        )


def test_comparison_bound_primary_artifact_cannot_fall_back_to_terminal_output(
    tmp_path: Path,
) -> None:
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()

    with pytest.raises(ValueError, match="primary answer artifact"):
        _comparison_trial_output_with_receipt(
            {
                "trial_dir": trial_dir.as_posix(),
                "expected_artifact_paths": ["/logs/artifacts/fugue-answer.md"],
                "agent_response": "must not be used",
            }
        )


def test_comparison_primary_artifact_allows_other_locked_outputs(
    tmp_path: Path,
) -> None:
    trial_dir = tmp_path / "trial"
    answer = trial_dir / "artifacts" / "logs" / "artifacts" / "fugue-answer.md"
    answer.parent.mkdir(parents=True)
    answer.write_text("complete", encoding="utf-8")

    value, receipt = _comparison_trial_output_with_receipt(
        {
            "trial_dir": trial_dir.as_posix(),
            "expected_artifact_paths": [
                "/logs/artifacts/supporting.json",
                "/logs/artifacts/fugue-answer.md",
            ],
        }
    )

    assert value == "complete"
    assert receipt["source_path"] == "/logs/artifacts/fugue-answer.md"


def test_comparison_primary_artifact_rejects_symlink(
    tmp_path: Path,
) -> None:
    trial_dir = tmp_path / "trial"
    answer = trial_dir / "artifacts" / "logs" / "artifacts" / "fugue-answer.md"
    answer.parent.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("must not be read", encoding="utf-8")
    answer.symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        _comparison_trial_output_with_receipt(
            {
                "trial_dir": trial_dir.as_posix(),
                "expected_artifact_paths": ["/logs/artifacts/fugue-answer.md"],
            }
        )


def test_comparison_primary_artifact_rejects_symlinked_trial_root(
    tmp_path: Path,
) -> None:
    real_trial = tmp_path / "real-trial"
    answer = real_trial / "artifacts" / "logs" / "artifacts" / "fugue-answer.md"
    answer.parent.mkdir(parents=True)
    answer.write_text("must not be read through a linked root", encoding="utf-8")
    linked_trial = tmp_path / "linked-trial"
    linked_trial.symlink_to(real_trial, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        _comparison_trial_output_with_receipt(
            {
                "trial_dir": linked_trial.as_posix(),
                "expected_artifact_paths": ["/logs/artifacts/fugue-answer.md"],
            }
        )


def test_comparison_primary_artifact_rejects_symlinked_trial_ancestor(
    tmp_path: Path,
) -> None:
    trusted_root = tmp_path / "trusted-jobs"
    trusted_root.mkdir()
    outside_parent = tmp_path / "outside-parent"
    real_trial = outside_parent / "trial"
    answer = real_trial / "artifacts" / "logs" / "artifacts" / "fugue-answer.md"
    answer.parent.mkdir(parents=True)
    answer.write_text("must not be read through a linked ancestor", encoding="utf-8")
    linked_parent = trusted_root / "linked-parent"
    linked_parent.symlink_to(outside_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        _comparison_trial_output_with_receipt(
            {
                "trial_dir": (linked_parent / "trial").as_posix(),
                "expected_artifact_paths": ["/logs/artifacts/fugue-answer.md"],
            },
            trusted_root=trusted_root,
        )


def test_comparison_primary_artifact_fails_closed_when_oversized(
    tmp_path: Path,
) -> None:
    trial_dir = tmp_path / "trial"
    answer = trial_dir / "artifacts" / "logs" / "artifacts" / "fugue-answer.md"
    answer.parent.mkdir(parents=True)
    answer.write_bytes(b"x" * (COMPARISON_PRIMARY_ARTIFACT_MAX_BYTES + 1))

    with pytest.raises(ValueError, match="exceeds the locked byte limit"):
        _comparison_trial_output_with_receipt({"trial_dir": trial_dir.as_posix()})


@pytest.mark.parametrize("trial_dir", [None, "does-not-exist"])
def test_comparison_primary_artifact_binding_requires_host_trial_directory(
    trial_dir: str | None,
) -> None:
    row: dict[str, object] = {
        "expected_artifact_paths": ["/logs/artifacts/fugue-answer.md"],
        "agent_response": "must not become the scored fallback",
    }
    if trial_dir is not None:
        row["trial_dir"] = trial_dir

    with pytest.raises(ValueError, match="trial directory is missing"):
        _comparison_trial_output_with_receipt(row)


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
    spec = replace(
        spec,
        baseline=replace(
            spec.baseline,
            skills=("verify-current-source-before",),
        ),
    )
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
            "skills_assigned": ["verify-current-source-before"],
            "skills_registered": ["verify-current-source-before"],
            "skill_registration_status": "registered",
            "skill_invocation_evidence": {
                "status": "observed",
                "skills_invoked": ["verify-current-source-before"],
            },
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
    assert result.mechanism_summary["skill_assigned"]["baseline"] == {
        "observed": 1,
        "applicable": 1,
        "unavailable": 0,
    }
    assert result.mechanism_summary["skill_invoked"]["baseline"] == {
        "observed": 1,
        "applicable": 1,
        "unavailable": 0,
    }
    assert (
        result.mechanism_summary["relevant_source_used"]["candidate"]["observed"] == 1
    )


def test_candidate_source_revisions_include_exact_git_skill() -> None:
    revisions = _candidate_source_revisions(
        [
            {
                "variant_id": "candidate",
                "integration_provenance": [],
                "skill_provenance": [
                    {
                        "id": "writing-plans-candidate",
                        "digest": "sha256:" + "a" * 64,
                        "resolved_commit": "b" * 40,
                    }
                ],
            },
            {
                "variant_id": "candidate",
                "integration_provenance": [],
                "skill_provenance": [
                    {
                        "id": "writing-plans-candidate",
                        "digest": "sha256:" + "a" * 64,
                        "resolved_commit": "b" * 40,
                    }
                ],
            },
        ]
    )

    assert [item.to_dict() for item in revisions] == [
        {
            "kind": "skill",
            "id": "writing-plans-candidate",
            "version_identity": "git:" + "b" * 40,
            "runtime_digest": "sha256:" + "a" * 64,
        }
    ]


def test_candidate_source_revisions_include_project_owned_skill_digest() -> None:
    digest = "sha256:" + "c" * 64
    revisions = _candidate_source_revisions(
        [
            {
                "variant_id": "candidate",
                "integration_provenance": [],
                "skill_provenance": [
                    {
                        "id": "project-writing-plans",
                        "digest": digest,
                        "license_status": "project-owned",
                    }
                ],
            }
        ]
    )

    assert [item.to_dict() for item in revisions] == [
        {
            "kind": "skill",
            "id": "project-writing-plans",
            "version_identity": f"digest:{digest}",
            "runtime_digest": digest,
        }
    ]


def test_preview_cells_carry_skill_provenance_into_approval_lineage() -> None:
    assert "skill_provenance" in PreviewCellSummary.__dataclass_fields__


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


def test_llm_judge_digest_binds_input_sanitizer_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path.cwd()
    judge = ComparisonEvaluatorV1(
        id="maintainer-review",
        type="llm_judge",
        required=False,
        profile="anthropic/claude-sonnet-5",
        rubric="Score maintenance actionability.",
        dimensions=("maintenance_actionability",),
        evidence=("inspected_paths",),
    )
    original = _evaluator_digest(judge, root)

    monkeypatch.setattr(
        "fugue.bench.comparison.COMPARISON_JUDGE_INPUT_SANITIZER_VERSION",
        999,
    )

    assert _evaluator_digest(judge, root) != original


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


def test_v3_attempt_exports_safe_anchored_blind_judge_review() -> None:
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
        "comparison_judges": {
            "maintainer-actionability": {
                "status": "scored",
                "scores": {"maintenance_actionability": 0.8},
                "overall_assessment": "Concrete and bounded maintainer advice.",
                "rationale": "Private reasoning is not projected.",
                "missing_evidence": False,
            }
        },
        "mcp_tool_calls": [],
    }

    legacy_attempt = _paired_attempt_view(row)
    attempt = _paired_attempt_view_v3(row)

    assert legacy_attempt is not None
    assert legacy_attempt.judge_reviews["maintainer-actionability"].label == "strong"
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
    assert attempt.judge_reviews["maintainer-actionability"].to_dict() == {
        "label": "strong",
        "reason": "Concrete and bounded maintainer advice.",
        "missing_evidence": False,
        "cost_status": "unavailable",
    }


def test_missing_evidence_judge_is_unusable_and_not_numerically_aggregated() -> None:
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
            "maintainer.answer_correct": True,
        },
        "comparison_judges": {
            "maintainer-actionability": {
                "status": "missing_evidence",
                "scores": {"maintenance_actionability": 1.0},
                "overall_assessment": "Required repository evidence is missing.",
                "rationale": "No grounded judgment is possible.",
                "missing_evidence": True,
            }
        },
        "mcp_tool_calls": [],
    }

    attempt = _paired_attempt_view_v3(row)

    assert attempt is not None
    assert attempt.scores == {"maintainer.answer_correct": True}
    assert attempt.judge_reviews["maintainer-actionability"].to_dict() == {
        "label": "unusable",
        "reason": "Required repository evidence is missing.",
        "missing_evidence": True,
        "cost_status": "unavailable",
    }


@pytest.mark.parametrize(
    ("score", "label"),
    [
        (0.0, "unusable"),
        (0.25, "weak"),
        (0.5, "adequate"),
        (0.75, "strong"),
        (0.9, "exceptional"),
    ],
)
def test_blind_judge_labels_have_stable_public_anchors(
    score: float, label: str
) -> None:
    assert _judge_review_label(score) == label


def test_blind_judge_response_schema_is_exact_and_describes_soft_text_bound() -> None:
    schema = _comparison_judge_response_schema(("usefulness", "grounding"))

    assert schema["additionalProperties"] is False
    assert schema["required"] == [
        "scores",
        "overall_assessment",
        "uncertainty",
        "missing_evidence",
        "rationale",
    ]
    assert schema["properties"]["scores"]["additionalProperties"] is False
    assert schema["properties"]["scores"]["required"] == [
        "usefulness",
        "grounding",
    ]
    assert (
        "requested maximum 500 characters"
        in schema["properties"]["rationale"]["description"]
    )


def test_blind_judge_accepts_long_but_bounded_reason_and_rejects_hard_overflow() -> (
    None
):
    judge = ComparisonEvaluatorV1(
        id="maintainer-review",
        type="llm_judge",
        required=False,
        profile="anthropic/claude-sonnet-5",
        rubric="Score maintenance actionability.",
        dimensions=("maintenance_actionability",),
        evidence=("tool_names",),
    )
    payload = {
        "scores": {"maintenance_actionability": 0.75},
        "overall_assessment": "Useful and grounded.",
        "uncertainty": 0.2,
        "missing_evidence": False,
        "rationale": "r" * 800,
    }

    assert _validate_comparison_judge_payload(judge, payload)["rationale"] == (
        "r" * 800
    )
    payload["scores_explanation"] = "unsupported"
    with pytest.raises(ValueError, match="scores_explanation"):
        _validate_comparison_judge_payload(judge, payload)
    payload.pop("scores_explanation")
    payload["rationale"] = "r" * (JUDGE_JSON_MAX_RESPONSE_CHARACTERS + 1)
    with pytest.raises(ValueError, match="rationale"):
        _validate_comparison_judge_payload(judge, payload)


def test_anthropic_blind_judge_sends_the_bound_response_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judge = ComparisonEvaluatorV1(
        id="maintainer-review",
        type="llm_judge",
        required=False,
        profile="anthropic/claude-sonnet-5",
        rubric="Score maintenance actionability.",
        dimensions=("maintenance_actionability",),
        evidence=("tool_names",),
    )
    captured: dict[str, object] = {}

    def post_judge(*args: object, **kwargs: object):
        captured.update(kwargs)
        captured["prompt"] = args[4]
        return (
            {
                "scores": {"maintenance_actionability": 0.75},
                "overall_assessment": "Useful and grounded.",
                "uncertainty": 0.2,
                "missing_evidence": False,
                "rationale": "The response cites the inspected path.",
            },
            {"input_tokens": 100, "output_tokens": 20},
        )

    monkeypatch.setattr("fugue.bench.evaluations._post_judge", post_judge)
    _, _, receipt = _request_comparison_judge(
        evaluator=judge,
        public_task={"input": "Review this change."},
        row={
            "answer": "A grounded response.",
            "comparison_primary_output": _primary_output_receipt(
                "A grounded response.",
                source="fugue_answer_artifact",
            ),
        },
        env={"ANTHROPIC_API_KEY": "test-secret"},
    )

    assert captured["response_schema"] == _comparison_judge_response_schema(
        judge.dimensions
    )
    assert receipt["response_schema_digest"] == stable_digest(
        captured["response_schema"]
    )
    assert receipt["response_request_mode"] == ("anthropic_json_schema_no_thinking_v2")
    assert receipt["response_validator_version"] == 2
    assert receipt["public_rubric_contract_digest"] == stable_digest(
        comparison_judge_public_rubric_contract(
            rubric=str(judge.rubric or ""),
            dimensions=judge.dimensions,
        )
    )
    assert receipt["requested_text_max_characters"] == 500
    assert receipt["response_max_characters"] == JUDGE_JSON_MAX_RESPONSE_CHARACTERS
    assert receipt["maximum_prompt_characters"] == 48_000
    assert receipt["primary_response"] == {
        "schema_version": 1,
        "status": "complete",
        "source": "fugue_answer_artifact",
        "primary_artifact_receipt_digest": stable_digest(
            _primary_output_receipt(
                "A grounded response.",
                source="fugue_answer_artifact",
            )
        ),
        "source_characters": 20,
        "source_bytes": 20,
        "source_sha256": hashlib.sha256(b"A grounded response.").hexdigest(),
        "provider_characters": 20,
        "provider_bytes": 20,
        "provider_sha256": hashlib.sha256(b"A grounded response.").hexdigest(),
        "privacy_transformed": False,
        "truncated": False,
    }
    assert "complete primary artifact" in str(captured["prompt"])
    assert "fixture placeholders remain intact" in str(captured["prompt"])
    assert "declared privacy redaction" in str(captured["prompt"])
    assert receipt["request_policy"]["structured_assistant_options"] == {
        "thinking": {"type": "disabled"}
    }


def test_live_blind_judge_receives_complete_primary_artifact_past_legacy_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judge = ComparisonEvaluatorV1(
        id="maintainer-review",
        type="llm_judge",
        required=False,
        profile="anthropic/claude-sonnet-5",
        rubric="Score maintenance actionability.",
        dimensions=("maintenance_actionability",),
        evidence=("inspected_paths",),
    )
    artifact = "# Complete plan\n" + ("Verified section.\n" * 1_500)
    artifact += 'secret = Path(values["wandb_api_key_file"]).read_text()\n'
    artifact += "WANDB_API_KEY=example-placeholder\n"
    assert len(artifact.encode()) > 16_000
    captured: dict[str, str] = {}

    def post_judge(*args: object, **_kwargs: object):
        captured["prompt"] = str(args[4])
        return (
            {
                "scores": {"maintenance_actionability": 0.75},
                "overall_assessment": "Useful and grounded.",
                "uncertainty": 0.2,
                "missing_evidence": False,
                "rationale": "The complete plan is reviewable.",
            },
            {"input_tokens": 100, "output_tokens": 20},
        )

    monkeypatch.setattr("fugue.bench.evaluations._post_judge", post_judge)
    _, _, receipt = _request_comparison_judge(
        evaluator=judge,
        public_task={"input": "Review this plan."},
        row={
            "answer": artifact,
            "comparison_primary_output": _primary_output_receipt(
                artifact,
                source="fugue_answer_artifact",
            ),
        },
        env={"ANTHROPIC_API_KEY": "configured-secret-value"},
    )

    provider_payload = json.loads(captured["prompt"].split("\n\n", 1)[1])
    assert provider_payload["response"] == artifact
    assert len(captured["prompt"]) < 48_000
    assert receipt["primary_response"]["source_bytes"] == len(artifact.encode())
    assert receipt["primary_response"]["provider_bytes"] == len(artifact.encode())
    assert receipt["primary_response"]["privacy_transformed"] is False
    assert receipt["primary_response"]["truncated"] is False


def test_live_blind_judge_fails_closed_instead_of_truncating_payload() -> None:
    judge = ComparisonEvaluatorV1(
        id="maintainer-review",
        type="llm_judge",
        required=False,
        profile="anthropic/claude-sonnet-5",
        rubric="Score maintenance actionability.",
        dimensions=("maintenance_actionability",),
        evidence=("tool_names",),
    )

    with pytest.raises(ValueError, match="prompt exceeds the locked bound"):
        _request_comparison_judge(
            evaluator=judge,
            public_task={"input": "Review this change."},
            row={
                "answer": "x" * 60_000,
                "comparison_primary_output": _primary_output_receipt(
                    "x" * 60_000
                ),
            },
            env={"ANTHROPIC_API_KEY": "test-secret"},
        )


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


def test_cancelled_comparison_row_never_invokes_scorer_or_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path.cwd()
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=root)
    calls: list[str] = []

    def forbidden_scorer(*_args: object, **_kwargs: object) -> object:
        calls.append("scorer")
        raise AssertionError("cancelled row must not be scored")

    def forbidden_judge(*_args: object, **_kwargs: object) -> object:
        calls.append("judge")
        raise AssertionError("cancelled row must not be judged")

    monkeypatch.setattr(
        "fugue.bench.comparison._run_custom_scorer",
        forbidden_scorer,
    )
    [row] = score_comparison_rows(
        spec,
        [
            {
                "task_id": "expense-limit",
                "variant_id": "baseline",
                "harness": "claude-code",
                "trial_index": 1,
                "status": "cancelled",
                "runtime_outcome": "cancelled",
            }
        ],
        repo_root=root,
        env={"ANTHROPIC_API_KEY": "configured-secret"},
        judge_request=forbidden_judge,
    )

    assert calls == []
    assert row["comparison_evaluation_status"] == "unavailable"
    assert row["comparison_required_evaluation_complete"] is False


def test_cancelled_comparison_row_clears_stale_outcome_claims() -> None:
    root = Path.cwd()
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=root)
    stale = {
        "task_id": "expense-limit",
        "variant_id": "baseline",
        "harness": "claude-code",
        "trial_index": 1,
        "status": "cancelled",
        "runtime_outcome": "cancelled",
        "pass": True,
        "benchmark_pass": True,
        "benchmark_outcome": "passed",
        "reward": 1.0,
        "host_evaluator_status": "scored",
        "evaluation_overall": 1.0,
        "evaluation_judge_status": "scored",
        "evaluation_correctness": 1.0,
        "judge_correctness": 1.0,
        "evaluation_rubrics": [{"id": "quality"}],
        "judge_overall": 0.9,
        "adapter_outcome": {
            "deterministic_verification": {"state": "passed"},
        },
        "comparison_deterministic_scores": {"facts.answer": 1.0},
        "comparison_dimension_roles": {"facts.answer": "outcome"},
        "comparison_deterministic_criticality": {"facts.answer": True},
        "comparison_mechanism": {"assigned": True, "used": True},
        "comparison_judge_scores": {"judge.useful": 1.0},
        "comparison_judge_accounted_cost_usd": 0.5,
        "comparison_judges": {"judge": {"status": "scored"}},
        "comparison_host_verifier_receipts": [{"status": "passed"}],
    }

    [row] = score_comparison_rows(spec, [stale], repo_root=root)

    assert row["pass"] is None
    assert row["benchmark_pass"] is None
    assert row["benchmark_outcome"] == "unscored"
    assert row["reward"] is None
    assert row["host_evaluator_status"] == "not_run"
    assert row["comparison_evaluation_status"] == "unavailable"
    assert row["comparison_judge_status"] == "unavailable"
    assert all(
        field not in row
        for field in (
            "comparison_deterministic_scores",
            "comparison_dimension_roles",
            "comparison_deterministic_criticality",
            "comparison_mechanism",
            "comparison_judge_scores",
            "comparison_judge_accounted_cost_usd",
            "comparison_host_verifier_receipts",
            "evaluation_overall",
            "evaluation_judge_status",
            "evaluation_correctness",
            "judge_correctness",
            "evaluation_rubrics",
            "judge_overall",
            "adapter_outcome",
        )
    )


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
            **_fake_scorer_receipts(
                source=source,
                evidence=evidence,
                reference=reference,
                profile=profile,
            ),
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
    scorer_receipt = rows[0]["comparison_deterministic_scorer_receipts"][
        "fact-and-source"
    ]
    assert scorer_receipt["status"] == "bound"
    assert scorer_receipt["primary_artifact_sha256"] == rows[0][
        "comparison_primary_output"
    ]["sha256"]
    assert scorer_receipt["receipt_digest"] == stable_digest(
        {
            key: value
            for key, value in scorer_receipt.items()
            if key != "receipt_digest"
        }
    )
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
    assert observed["evidence"] == {
        "attempt_id": rows[0]["attempt_id"],
        "task_id": "expense-limit",
        "variant_id": "candidate",
        "harness": "codex",
        "trial_index": 1,
    }
    assert "--network" not in str(observed["source"])


def test_optional_deterministic_mechanism_does_not_change_required_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path.cwd()
    scorer_path = root / ".fugue" / "test-optional-mechanism-scorer.py"
    scorer_path.parent.mkdir(parents=True, exist_ok=True)
    scorer_path.write_text(
        "def score(task, output, evidence):\n"
        "    return {'skill_rule_opened': False}\n"
    )

    def fake_runner(*, source, evidence, reference, profile, limits):
        return {
            "score": 0.0,
            "reason": "custom deterministic scorer",
            "details": {"skill_rule_opened": False},
            **_fake_scorer_receipts(
                source=source,
                evidence=evidence,
                reference=reference,
                profile=profile,
            ),
        }

    monkeypatch.setattr("fugue.bench.task_authoring.run_inline_scorer", fake_runner)
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=root)
    optional = ComparisonEvaluatorV1(
        id="skill-mechanism",
        type="deterministic",
        required=False,
        scorer=scorer_path.relative_to(root).as_posix(),
        runtime="python312-sandbox-v1",
        dimensions=("skill_rule_opened",),
        dimension_roles={"skill_rule_opened": "mechanism"},
    )
    try:
        [row] = score_comparison_rows(
            replace(spec, evaluators=(*spec.evaluators, optional)),
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

    assert row["pass"] is True
    assert row["comparison_deterministic_scores"][
        "skill-mechanism.skill_rule_opened"
    ] is False
    assert row["comparison_dimension_roles"][
        "skill-mechanism.skill_rule_opened"
    ] == "mechanism"


def test_required_evaluator_excludes_mechanism_from_task_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path.cwd()
    scorer_path = root / ".fugue" / "test-role-aware-scorer.py"
    scorer_path.parent.mkdir(parents=True, exist_ok=True)
    scorer_path.write_text(
        "def score(task, output, evidence):\n"
        "    return {'fact_correct': True, 'skill_rule_opened': False}\n"
    )

    def fake_runner(*, source, evidence, reference, profile, limits):
        return {
            "score": 0.0,
            "reason": "custom deterministic scorer",
            "details": {
                "fact_correct": True,
                "skill_rule_opened": False,
            },
            **_fake_scorer_receipts(
                source=source,
                evidence=evidence,
                reference=reference,
                profile=profile,
            ),
        }

    monkeypatch.setattr("fugue.bench.task_authoring.run_inline_scorer", fake_runner)
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=root)
    evaluator = replace(
        spec.evaluators[0],
        checks=(),
        scorer=scorer_path.relative_to(root).as_posix(),
        runtime="python312-sandbox-v1",
        dimensions=("fact_correct", "skill_rule_opened"),
        dimension_roles={
            "fact_correct": "outcome",
            "skill_rule_opened": "mechanism",
        },
    )
    try:
        [row] = score_comparison_rows(
            replace(spec, schema_version=3, evaluators=(evaluator,)),
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

    assert row["pass"] is True
    assert row["comparison_deterministic_scores"] == {
        "fact-and-source.fact_correct": True,
        "fact-and-source.skill_rule_opened": False,
    }


def test_deterministic_scorer_rejects_primary_artifact_receipt_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path.cwd()
    scorer_path = root / ".fugue" / "test-primary-binding-scorer.py"
    scorer_path.parent.mkdir(parents=True, exist_ok=True)
    scorer_path.write_text(
        "def score(task, output, evidence):\n"
        "    return {'fact_correct': True}\n"
    )

    def fake_runner(*, source, evidence, reference, profile, limits):
        return {
            "score": 1.0,
            "reason": "custom deterministic scorer",
            "details": {"fact_correct": True},
            **_fake_scorer_receipts(
                source=source,
                evidence=evidence,
                reference=reference,
                profile=profile,
            ),
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
    answer = {"amount": 125, "source": "expense-policy-v4.md"}
    wrong_receipt = _primary_output_receipt({"different": "artifact"})
    try:
        with pytest.raises(
            ValueError,
            match="canonical result projection disagrees",
        ):
            score_comparison_rows(
                custom,
                [
                    {
                        "task_id": "expense-limit",
                        "variant_id": "candidate",
                        "harness": "codex",
                        "trial_index": 1,
                        "answer": answer,
                        "comparison_primary_output": wrong_receipt,
                    }
                ],
                repo_root=root,
            )
    finally:
        scorer_path.unlink(missing_ok=True)


def test_custom_host_verifier_is_frozen_and_overrides_its_bound_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path.cwd()
    scorer_path = root / ".fugue" / "test-host-verifier-scorer.py"
    verifier_path = root / ".fugue" / "test-host-verifier.mjs"
    scorer_path.parent.mkdir(parents=True, exist_ok=True)
    scorer_path.write_text(
        "def score(task, output, evidence):\n"
        "    return {'fact_correct': output == evidence['expected']}\n"
    )
    verifier_path.write_text("process.stdout.write('{}')\n")
    calls: list[dict[str, object]] = []

    def fake_runner(*, source, evidence, reference, profile, limits):
        calls.append(
            {
                "source": source,
                "evidence": evidence,
                "reference": reference,
                "profile": profile,
                "limits": limits,
            }
        )
        passed = reference["output"] == reference["expected"]
        assert evidence["host_verifier_receipt"]["status"] == "passed"
        return {
            "score": 1.0 if passed else 0.0,
            "reason": "custom deterministic scorer",
            "details": {"fact_correct": passed},
            **_fake_scorer_receipts(
                source=source,
                evidence=evidence,
                reference=reference,
                profile=profile,
            ),
        }

    monkeypatch.setattr("fugue.bench.task_authoring.run_inline_scorer", fake_runner)

    def fake_verifier(evaluator, *, task, output, evidence, **kwargs):
        cleanup_unsigned = {
            "schema_version": 1,
            "status": "verified_absent",
            "container_name_sha256": "4" * 64,
        }
        cleanup = {
            **cleanup_unsigned,
            "receipt_digest": stable_digest(cleanup_unsigned),
        }
        unsigned = {
            "schema_version": 2,
            "kind": "post_trial_verifier_receipt",
            "evaluator_id": evaluator.id,
            "task_id": task["id"],
            "attempt_id": evidence["attempt_id"],
            "status": "passed",
            "failure_kind": None,
            "runtime": "node-v22.23.1",
            "command": ["node", "--test", "tests/task.test.mjs"],
            "exit_code": 0,
            "test_count": 1,
            "pass_count": 1,
            "fail_count": 0,
            "output_sha256": "a" * 64,
            "base_archive_sha256": "b" * 64,
            "public_test_sha256": "c" * 64,
            "submitted_artifact_sha256": stable_digest(output),
            "final_tree_sha256": "d" * 64,
            "verifier_source_sha256": "e" * 64,
            "runtime_profile_id": "node22-verifier-v1",
            "runtime_profile_digest": "f" * 64,
            "runtime_image": "example/verifier@sha256:" + "1" * 64,
            "runtime_platform": "linux/arm64",
            "runtime_image_id": "sha256:" + "2" * 64,
            "runtime_lock_digest": "3" * 64,
            "runtime_cleanup": cleanup,
        }
        return {
            "score": 1.0,
            "reason": "frozen host verification passed",
            "details": {**unsigned, "receipt_digest": stable_digest(unsigned)},
        }

    monkeypatch.setattr("fugue.bench.comparison._run_custom_verifier", fake_verifier)
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=root)
    evaluator = replace(
        spec.evaluators[0],
        checks=(),
        scorer=scorer_path.relative_to(root).as_posix(),
        runtime="python312-sandbox-v1",
        verifier=ComparisonPostTrialVerifierV1(
            type="node_test",
            source=verifier_path.relative_to(root).as_posix(),
            runtime="node22-verifier-v1",
            dimension="fact_correct",
        ),
        dimensions=("fact_correct",),
    )
    try:
        rows = score_comparison_rows(
            replace(spec, evaluators=(evaluator,)),
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
        verifier_path.unlink(missing_ok=True)

    assert [call["profile"].id for call in calls] == ["python312-sandbox-v1"]
    assert rows[0]["pass"] is True
    assert rows[0]["comparison_host_verifier_receipts"] == {
        "fact-and-source": {
            "schema_version": 2,
            "kind": "post_trial_verifier_receipt",
            "evaluator_id": "fact-and-source",
            "task_id": "expense-limit",
            "attempt_id": rows[0]["attempt_id"],
            "status": "passed",
            "failure_kind": None,
            "runtime": "node-v22.23.1",
            "command": ["node", "--test", "tests/task.test.mjs"],
            "exit_code": 0,
            "test_count": 1,
            "pass_count": 1,
            "fail_count": 0,
            "output_sha256": "a" * 64,
            "base_archive_sha256": "b" * 64,
            "public_test_sha256": "c" * 64,
            "submitted_artifact_sha256": stable_digest(
                {"amount": 125, "source": "expense-policy-v4.md"}
            ),
            "final_tree_sha256": "d" * 64,
            "verifier_source_sha256": "e" * 64,
            "runtime_profile_id": "node22-verifier-v1",
            "runtime_profile_digest": "f" * 64,
            "runtime_image": "example/verifier@sha256:" + "1" * 64,
            "runtime_platform": "linux/arm64",
            "runtime_image_id": "sha256:" + "2" * 64,
            "runtime_lock_digest": "3" * 64,
            "runtime_cleanup": rows[0]["comparison_host_verifier_receipts"]
            ["fact-and-source"]["runtime_cleanup"],
            "receipt_digest": rows[0]["comparison_host_verifier_receipts"]
            ["fact-and-source"]["receipt_digest"],
        }
    }


@pytest.fixture
def allow_unit_judge_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "fugue.bench.comparison._judge_execution_calibration_issue",
        lambda *_args, **_kwargs: None,
    )


def test_blind_judge_receives_only_public_task_output_and_permitted_evidence(
    monkeypatch: pytest.MonkeyPatch,
    allow_unit_judge_execution: None,
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
                "missing_evidence": False,
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
                "inspected_paths_status": "available",
                "comparison_deterministic_scores": {"private": True},
                "weave_usage_status": "available",
                "weave_cost_status": "available",
                "weave_total_cost_usd": 1.0,
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
    assert "Path lists are audited activity, not proof of correctness" in prompt
    assert "available empty changed_paths list is expected" in prompt
    assert "implementation proof that the public task did not request" in prompt
    assert '"inspected_paths_status": "available"' in prompt
    assert rows[0]["comparison_judge_status"] == "scored"
    assert rows[0]["comparison_required_evaluation_complete"] is True
    assert captured["requests"] == 1
    assert captured["timeout_sec"] == 300
    assert rows[0]["comparison_judge_accounted_cost_usd"] == 0.1
    assert rows[0]["accounted_cost_usd"] == 1.1
    judge_cost = rows[0]["comparison_judges"]["maintainer-review"]
    assert judge_cost["observed_cost_usd"] is None
    assert judge_cost["accounted_reserve_usd"] == 0.1
    assert judge_cost["cost_status"] == "unavailable"
    privacy = rows[0]["comparison_judges"]["maintainer-review"]["route_receipt"][
        "judge_input_privacy"
    ]
    assert privacy["status"] == "passed"
    assert len(privacy["payload_sha256"]) == 64
    request_policy = rows[0]["comparison_judges"]["maintainer-review"]["route_receipt"][
        "request_policy"
    ]
    assert request_policy == {
        "schema_version": 2,
        "timeout_sec": 300,
        "max_output_tokens": 1_200,
        "max_response_characters": 16_000,
        "structured_assistant_options": {"thinking": {"type": "disabled"}},
        "automatic_retries": 0,
    }


def test_blind_judge_preserves_safe_credential_placeholders_before_provider(
    monkeypatch: pytest.MonkeyPatch,
    allow_unit_judge_execution: None,
) -> None:
    root = Path.cwd()
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=root)
    judge = ComparisonEvaluatorV1(
        id="maintainer-review",
        type="llm_judge",
        required=True,
        profile="anthropic/claude-sonnet-5",
        rubric="Score maintenance actionability.",
        dimensions=("maintenance_actionability",),
        evidence=("inspected_paths",),
        reserve_cost_usd=0.1,
    )
    captured: dict[str, object] = {"calls": 0}

    def capture_post(
        _client: object,
        _route: object,
        _key: str,
        _env: object,
        prompt: str,
        **_kwargs: object,
    ) -> tuple[dict[str, object], dict[str, int]]:
        captured["calls"] = int(captured["calls"]) + 1
        captured["prompt"] = prompt
        return (
            {
                "scores": {"maintenance_actionability": 0.75},
                "overall_assessment": "Useful and grounded.",
                "uncertainty": 0.2,
                "missing_evidence": False,
                "rationale": "The plan is reviewable and appropriately bounded.",
            },
            {"input_tokens": 100, "output_tokens": 20},
        )

    monkeypatch.setattr("fugue.bench.evaluations._post_judge", capture_post)
    rows = score_comparison_rows(
        replace(spec, evaluators=(*spec.evaluators, judge)),
        [
            {
                "answer": {
                    "plan": "\n".join(
                        (
                            "Use API_KEY=example-placeholder in the local fixture.",
                            'env_file.write_text("WANDB_API_KEY=from-env-file\\n")',
                            'first.write_text("WANDB_API_KEY=first-key\\n")',
                            'second.write_text("WANDB_API_KEY=second-key\\n")',
                            'secret = Path(values["wandb_api_key_file"]).read_text()',
                            "Then verify the credential is not committed.",
                        )
                    )
                },
                "task_id": "expense-limit",
                "trial_index": 1,
                "variant_id": "candidate",
                "harness": "claude-code",
                "inspected_paths": ["README.md"],
                "inspected_paths_status": "available",
            }
        ],
        repo_root=root,
        env={"ANTHROPIC_API_KEY": "configured-secret-value"},
    )

    assert captured["calls"] == 1
    prompt = str(captured["prompt"])
    provider_payload = json.loads(prompt.split("\n\n", 1)[1])
    serialized_payload = json.dumps(provider_payload, sort_keys=True)
    assert "API_KEY=example-placeholder" in serialized_payload
    assert "WANDB_API_KEY=from-env-file" in serialized_payload
    assert "WANDB_API_KEY=first-key" in serialized_payload
    assert "WANDB_API_KEY=second-key" in serialized_payload
    assert "secret = Path" in serialized_payload
    assert "[redacted]" not in serialized_payload
    assert "configured-secret-value" not in serialized_payload
    assert "private_expected_values" not in serialized_payload
    assert "variant_id" not in serialized_payload
    result = rows[0]["comparison_judges"]["maintainer-review"]
    assert result["status"] == "scored"
    route = result["route_receipt"]
    privacy = route["judge_input_privacy"]
    assert privacy["contract"] == "fugue-judge-input-sanitization-v3"
    assert privacy["transform"] == "high-confidence-credential-redaction-v3"
    assert privacy["transformed"] is False
    assert privacy["transformed_value_count"] == 0
    assert privacy["source_payload_sha256"] == privacy["provider_payload_sha256"]
    assert privacy["primary_response"]["privacy_transformed"] is False
    assert (
        privacy["primary_response"]["source_sha256"]
        == privacy["primary_response"]["provider_sha256"]
    )
    assert privacy["provider_payload_sha256"] == stable_digest(provider_payload)
    assert route["provider_payload_sha256"] == privacy["provider_payload_sha256"]
    assert route["input_transform_receipt_digest"] == privacy["receipt_digest"]
    assert privacy["receipt_digest"] == stable_digest(
        {key: value for key, value in privacy.items() if key != "receipt_digest"}
    )


def test_blind_judge_redacts_unknown_credentials_without_corrupting_code() -> None:
    judge = ComparisonEvaluatorV1(
        id="maintainer-review",
        type="llm_judge",
        required=True,
        profile="anthropic/claude-sonnet-5",
        rubric="Score maintenance actionability.",
        dimensions=("maintenance_actionability",),
        evidence=("inspected_paths",),
    )
    answer = {
        "plan": "\n".join(
            (
                'token = "unreviewed-live-credential-1234"',
                "Authorization: Bearer unreviewed-bearer-1234",
                "API_KEY=unreviewed-api-key-1234",
                "API_KEY=prod-example-ACTUALSECRET123",
                "API_KEY=abcd$rest",
                "API_KEY=<prod-real-secret-1234>",
                "Authorization: Bearer <prod-bearer-secret-1234>",
                "password: correct horse battery staple",
                "secret: alpha bravo charlie delta",
                "token: echo foxtrot golf hotel",
                "api_key: india juliet kilo lima",
                "token = sample-live-credential-9999",
                'token = os.getenv("TOKEN", "fallback-secret-9876")',
                'secret = config["secret"] # comment-secret-9876',
                'secret = Path(values["wandb_api_key_file"]).read_text()',
                'api_key = Path("/run/secrets/wandb")',
                'token = os.getenv("TOKEN")',
                'secret = config["secret"]',
            )
        ),
        "api_key": "unreviewed-structured-key-1234",
    }
    prepared, receipt = _prepare_comparison_judge_input(
        evaluator=judge,
        public_task={"input": "Review the repair plan.", "tags": ["maintenance"]},
        row={
            "answer": answer,
            "comparison_primary_output": _primary_output_receipt(answer),
            "inspected_paths": ["README.md"],
            "inspected_paths_status": "available",
        },
        env={"ANTHROPIC_API_KEY": "configured-secret-value"},
    )

    serialized = json.dumps(prepared, sort_keys=True)
    assert "unreviewed-live-credential-1234" not in serialized
    assert "unreviewed-bearer-1234" not in serialized
    assert "unreviewed-api-key-1234" not in serialized
    assert "unreviewed-structured-key-1234" not in serialized
    assert "prod-example-ACTUALSECRET123" not in serialized
    assert "abcd$rest" not in serialized
    assert "prod-real-secret-1234" not in serialized
    assert "prod-bearer-secret-1234" not in serialized
    assert "correct horse battery staple" not in serialized
    assert "alpha bravo charlie delta" not in serialized
    assert "echo foxtrot golf hotel" not in serialized
    assert "india juliet kilo lima" not in serialized
    assert "sample-live-credential-9999" not in serialized
    assert "fallback-secret-9876" not in serialized
    assert "comment-secret-9876" not in serialized
    assert 'token = \\"[redacted]\\"' in serialized
    assert "Authorization: Bearer [redacted]" in serialized
    assert "API_KEY=[redacted]" in serialized
    assert "secret = Path" in serialized
    assert 'api_key = Path(\\"/run/secrets/wandb\\")' in serialized
    assert 'token = os.getenv(\\"TOKEN\\")' in serialized
    assert 'secret = config[\\"secret\\"]' in serialized
    assert receipt["contract"] == "fugue-judge-input-sanitization-v3"
    assert receipt["transformed"] is True
    assert receipt["transformed_value_count"] == 2
    assert receipt["primary_response"]["privacy_transformed"] is True
    assert (
        receipt["primary_response"]["source_sha256"]
        != receipt["primary_response"]["provider_sha256"]
    )


def test_blind_judge_rejects_untracked_prepared_payload_transformation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judge = ComparisonEvaluatorV1(
        id="maintainer-review",
        type="llm_judge",
        required=True,
        profile="anthropic/claude-sonnet-5",
        rubric="Score maintenance actionability.",
        dimensions=("maintenance_actionability",),
        evidence=("inspected_paths",),
    )
    public_task = {"input": "Review the repair plan.", "tags": ["maintenance"]}
    answer = {"plan": "Set API_KEY=example-placeholder in the test fixture."}
    row = {
        "answer": answer,
        "comparison_primary_output": _primary_output_receipt(answer),
        "inspected_paths": ["README.md"],
        "inspected_paths_status": "available",
    }
    env = {"ANTHROPIC_API_KEY": "configured-secret-value"}
    prepared, receipt = _prepare_comparison_judge_input(
        evaluator=judge,
        public_task=public_task,
        row=row,
        env=env,
    )
    tampered = json.loads(json.dumps(prepared))
    tampered["response"]["plan"] += " untracked"
    provider_calls: list[str] = []

    def forbidden_post(*_args: object, **_kwargs: object) -> None:
        provider_calls.append("called")
        raise AssertionError("untracked input must stop before the provider")

    monkeypatch.setattr("fugue.bench.evaluations._post_judge", forbidden_post)
    with pytest.raises(
        ValueError,
        match="prepared judge payload does not match deterministic sanitization",
    ):
        _request_comparison_judge(
            evaluator=judge,
            public_task=public_task,
            row=row,
            env=env,
            prepared_payload=tampered,
            input_transform_receipt=receipt,
        )
    assert provider_calls == []


def test_blind_judge_rejects_secret_in_provider_response_without_persisting_it(
    allow_unit_judge_execution: None,
) -> None:
    root = Path.cwd()
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=root)
    judge = ComparisonEvaluatorV1(
        id="maintainer-review",
        type="llm_judge",
        required=False,
        profile="anthropic/claude-sonnet-5",
        rubric="Score maintenance actionability.",
        dimensions=("maintenance_actionability",),
        evidence=("tool_names",),
        reserve_cost_usd=0.1,
    )
    secret = "response-secret-value"

    def secret_response(
        *,
        evaluator: ComparisonEvaluatorV1,
        public_task: dict[str, object],
        row: dict[str, object],
        env: dict[str, str],
    ):
        del evaluator, public_task, row, env
        return (
            {
                "scores": {"maintenance_actionability": 0.75},
                "overall_assessment": "Useful and grounded.",
                "uncertainty": 0.2,
                "missing_evidence": False,
                "rationale": f"The response unexpectedly repeated {secret}.",
            },
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
            }
        ],
        repo_root=root,
        env={"ANTHROPIC_API_KEY": secret},
        judge_request=secret_response,
    )

    failure = rows[0]["comparison_judges"]["maintainer-review"]["failure"]
    assert failure["stage"] == "output_privacy"
    assert failure["code"] == "output_privacy_rejected"
    assert failure["usage"] == {"input_tokens": 100, "output_tokens": 20}
    assert failure["response_characters"] > 0
    assert len(failure["response_sha256"]) == 64
    assert secret not in json.dumps(rows[0], sort_keys=True)


def test_blind_judge_rejects_secret_before_provider_call(
    monkeypatch: pytest.MonkeyPatch,
    allow_unit_judge_execution: None,
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
    failure = rows[0]["comparison_judges"]["maintainer-review"]["failure"]
    assert failure["stage"] == "input_privacy"
    assert failure["code"] == "input_privacy_rejected"
    assert "super-secret-value" not in json.dumps(failure, sort_keys=True)


def test_blind_judge_read_timeout_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
    allow_unit_judge_execution: None,
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
            "schema_version": 2,
            "timeout_sec": 300,
            "max_output_tokens": 1_200,
            "max_response_characters": 16_000,
            "structured_assistant_options": {},
            "automatic_retries": 0,
        },
    }
    failed_judge = rows[0]["comparison_judges"]["maintainer-review"]
    assert failed_judge["observed_cost_usd"] is None
    assert failed_judge["accounted_reserve_usd"] == 0.1
    assert failed_judge["cost_status"] == "unavailable"
    assert rows[0]["comparison_judge_accounted_cost_usd"] == 0.1


def test_blind_judge_records_safe_no_json_failure_metadata(
    monkeypatch: pytest.MonkeyPatch,
    allow_unit_judge_execution: None,
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
            "schema_version": 2,
            "timeout_sec": 300,
            "max_output_tokens": 1_200,
            "max_response_characters": 16_000,
            "structured_assistant_options": {"thinking": {"type": "disabled"}},
            "automatic_retries": 0,
        },
    }
    assert "expense-policy-v4.md" not in json.dumps(failure)


def test_blind_judge_distinguishes_strict_rubric_validation_failure(
    allow_unit_judge_execution: None,
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
        "request_policy": {
            "schema_version": 2,
            "timeout_sec": 300,
            "max_output_tokens": 1_200,
            "max_response_characters": 16_000,
            "structured_assistant_options": {"thinking": {"type": "disabled"}},
            "automatic_retries": 0,
        },
    }
    assert "wrong-private-dimension" not in json.dumps(failure)


def test_scaffold_refuses_non_empty_destination(tmp_path: Path) -> None:
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
    local_source_digests = {
        "source_lock": "c" * 64,
        "source_lock_file": "d" * 64,
    }
    monkeypatch.setattr(
        "fugue.bench.comparison._local_source_lock_readiness",
        lambda *_args, **_kwargs: (local_source_digests, []),
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
    assert receipt["qualification_input_digests"] == local_source_digests
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

    execute_calls = 0

    def stop_at_execution(*args: object, **_kwargs: object) -> None:
        nonlocal execute_calls
        execute_calls += 1
        persisted_path = (
            root
            / COMPARISON_RESULT_ROOT
            / preview.preview_digest
            / APPROVED_COMPARISON_LOCK_NAME
        )
        persisted = json.loads(persisted_path.read_text(encoding="utf-8"))
        supplied = args[1].approved_comparison
        assert persisted == supplied
        unsigned = {
            key: value for key, value in persisted.items() if key != "lock_digest"
        }
        assert persisted["lock_digest"] == stable_digest(unsigned)
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
    assert execute_calls == 1

    sidecar = (
        root
        / COMPARISON_RESULT_ROOT
        / preview.preview_digest
        / APPROVED_COMPARISON_LOCK_NAME
    )
    tampered = json.loads(sidecar.read_text(encoding="utf-8"))
    tampered["lock_digest"] = "0" * 64
    sidecar.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="different immutable content"):
        execute_comparison(
            preview,
            approval_digest="",
            repo_root=root,
            fetch_weave=False,
            publish_research=False,
        )
    assert execute_calls == 1


def test_runtime_budget_cancels_before_remaining_cells_when_projection_drifts() -> None:
    budget = _ComparisonRuntimeBudget(
        max_cost_usd=5,
        reserve_per_attempt_usd=1,
        total_cells=3,
    )
    first = {"attempt_id": "a" * 64, "cost_usd": 4.5}

    with pytest.raises(RuntimeError, match="remaining cells exceeds"):
        budget.observe(first)

    assert budget.cancellation_event.is_set()
    assert budget.accounted_cost_usd == 4.5
    assert first["comparison_runtime_budget"] == {
        "schema_version": 1,
        "status": "failed",
        "attempt_id": "a" * 64,
        "cell_accounted_cost_usd": 4.5,
        "cell_accounted_cost_source": "cost_usd",
        "accounted_cost_usd": 4.5,
        "approved_max_cost_usd": 5,
        "reserve_per_remaining_attempt_usd": 1,
        "terminal_cells": 1,
        "remaining_cells": 2,
        "reserved_remaining_cost_usd": 2,
        "projected_minimum_cost_usd": 6.5,
        "reason": budget.failure_reason,
    }


def test_reserved_cost_per_attempt_includes_advisory_judge() -> None:
    spec = load_comparison(
        Path("examples/comparisons/anthropic-skill-creator-upgrade/comparison.yaml"),
        repo_root=Path.cwd(),
    )

    assert _comparison_reserved_cost_per_attempt(spec) == 8.5


@pytest.mark.parametrize("max_cost_usd", [0, 10])
def test_zero_reserve_comparison_must_run_serially(max_cost_usd: float) -> None:
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=Path.cwd())
    paid = replace(
        spec,
        execution=replace(
            spec.execution,
            max_cost_usd=max_cost_usd,
            reserve_per_attempt_usd=0,
            concurrency=2,
        ),
    )

    readiness = check_comparison(paid, repo_root=Path.cwd())

    assert readiness.status == "blocked"
    assert "runtime budget enforcement requires comparison concurrency=1" in (
        readiness.blockers
    )


def test_runtime_budget_fails_closed_when_started_attempt_has_no_cost() -> None:
    budget = _ComparisonRuntimeBudget(
        max_cost_usd=10,
        reserve_per_attempt_usd=1,
        total_cells=2,
    )
    row = {"attempt_id": "b" * 64, "exception_class": "AgentTimeoutError"}

    with pytest.raises(RuntimeError, match="cost evidence is unavailable"):
        budget.observe(row)

    assert budget.cancellation_event.is_set()
    assert row["comparison_runtime_budget"]["cell_accounted_cost_usd"] is None
    assert row["comparison_runtime_budget"]["cell_accounted_cost_source"] is None
    assert row["comparison_runtime_budget"]["remaining_cells"] == 1


def test_runtime_budget_uses_verified_weave_cost_for_timed_out_attempt() -> None:
    budget = _ComparisonRuntimeBudget(
        max_cost_usd=10,
        reserve_per_attempt_usd=1,
        total_cells=1,
    )
    row = {
        "attempt_id": "d" * 64,
        "exception_class": "AgentTimeoutError",
        "weave_usage_status": "available",
        "weave_cost_status": "available",
        "weave_total_cost_usd": 4.337110345377283,
    }

    budget.observe(row)

    assert not budget.cancellation_event.is_set()
    assert budget.accounted_cost_usd == pytest.approx(4.337110345377283)
    assert row["comparison_runtime_budget"]["cell_accounted_cost_usd"] == pytest.approx(
        4.337110345377283
    )
    assert (
        row["comparison_runtime_budget"]["cell_accounted_cost_source"]
        == "weave_total_cost_usd"
    )


def test_runtime_budget_accepts_explicitly_no_cost_comparison_rows() -> None:
    budget = _ComparisonRuntimeBudget(
        max_cost_usd=0,
        reserve_per_attempt_usd=0,
        total_cells=1,
    )
    row = {"attempt_id": "c" * 64}

    budget.observe(row)

    assert not budget.cancellation_event.is_set()
    assert row["comparison_runtime_budget"]["status"] == "within_budget"
    assert row["comparison_runtime_budget"]["cell_accounted_cost_usd"] == 0
    assert row["comparison_runtime_budget"]["cell_accounted_cost_source"] is None


def test_terminal_task_failure_is_exportable_but_cancellation_is_not() -> None:
    terminal_failure = SimpleNamespace(
        status="failed",
        failed=1,
        cancelled=0,
        interrupted=0,
        pending=0,
        observability_status="passed",
        evaluation_failures=(),
    )
    cancelled = SimpleNamespace(
        **{
            **terminal_failure.__dict__,
            "cancelled": 1,
        }
    )
    interrupted = SimpleNamespace(
        **{
            **terminal_failure.__dict__,
            "interrupted": 1,
        }
    )

    assert _comparison_run_exportability_issue(terminal_failure) is None
    assert "cancelled_cells=1" in str(
        _comparison_run_exportability_issue(cancelled)
    )
    assert "interrupted_cells=1" in str(
        _comparison_run_exportability_issue(interrupted)
    )
