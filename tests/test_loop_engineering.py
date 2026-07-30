from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from fugue.bench.ai import _intervention_component_locks, get_analysis
from fugue.bench.analysis_contracts import (
    AlignedAnalysisV1,
    AlignedArmV1,
    AlignedAttemptSetV1,
    AlignedContrastV1,
    AlignedDimensionV1,
    EvidenceDriftCheckV1,
    EvidenceTopologyV1,
    LockDescriptorV1,
    TaskStratifiedSummaryV1,
    TaskValidityV1,
)
from fugue.bench.campaign_lifecycle import get_campaign
from fugue.bench.candidates import stable_digest
from fugue.bench.comparison import (
    AttemptEvidenceLinkV1,
    BehavioralSummaryV1,
    CandidateSourceRevisionV1,
    ComparisonResultV3,
    DecisionSummaryV1,
    DimensionChangeV2,
    PairedAttemptV3,
    PairedCaseV3,
)
from fugue.bench.intervention_provenance import (
    build_intervention_component_lock,
    write_intervention_component_lock,
)
from fugue.bench.library import get_experiment
from fugue.bench.loop_failure import (
    build_comparison_failure_lock,
    validate_comparison_failure_lock,
)
from fugue.model_plane import EvidenceDestinationV1

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_ID = "mcp-main-vs-0-4-natural-maintainer-canary-v3"
SOURCE_PROJECT = "wandb/fugue-mcp-release-source-v2"
RESULT_PROJECT = "wandb/fugue-mcp-release-qualification-v1"
LOOP_PROJECT = "wandb/fugue-claude-loop-engineering-v1"


def _destination(project: str) -> EvidenceDestinationV1:
    entity, name = project.split("/", 1)
    return EvidenceDestinationV1(
        entity=entity,
        project=name,
        api_base_url="https://api.wandb.ai",
        trace_base_url="https://trace.wandb.ai",
        app_base_url="https://wandb.ai",
    )


def _links(prefix: str) -> tuple[AttemptEvidenceLinkV1, ...]:
    base = f"https://wandb.ai/{RESULT_PROJECT}/weave"
    return (
        AttemptEvidenceLinkV1(
            kind="evaluation_root",
            status="resolved",
            ref=f"{prefix}-evaluation",
            url=f"{base}/calls/{prefix}-evaluation",
        ),
        AttemptEvidenceLinkV1(
            kind="prediction_and_score",
            status="resolved",
            ref=f"{prefix}-score",
            url=f"{base}/calls/{prefix}-score",
        ),
        AttemptEvidenceLinkV1(
            kind="prediction",
            status="resolved",
            ref=f"{prefix}-prediction",
            url=f"{base}/calls/{prefix}-prediction",
        ),
        AttemptEvidenceLinkV1(
            kind="agent_root",
            status="resolved",
            ref=f"{prefix}-agent",
            url=f"{base}/calls/{prefix}-agent",
        ),
        AttemptEvidenceLinkV1(
            kind="dataset",
            status="resolved",
            ref=f"weave:///{RESULT_PROJECT}/object/dataset:v1",
            url=f"{base}/objects/dataset/versions/v1",
        ),
    )


def _attempt(
    *,
    task_id: str,
    arm: str,
    attempt: int,
    passed: bool,
) -> PairedAttemptV3:
    candidate = stable_digest({"arm": arm})
    runtime = stable_digest({"runtime": "local-harbor"})
    identity = {
        "task_id": task_id,
        "arm": arm,
        "harness": "claude-code",
        "attempt": attempt,
        "candidate": candidate,
        "runtime": runtime,
    }
    attempt_id = stable_digest({"schema_version": 1, **identity})
    return PairedAttemptV3(
        attempt_id=attempt_id,
        identity=identity,
        prediction_id=f"prediction-{task_id}-{arm}-{attempt}",
        passed=passed,
        execution_status="passed",
        evaluation_status="scored",
        evidence_status="reconciled",
        cost_usd=0.25,
        latency_sec=12.0,
        input_tokens=100.0,
        output_tokens=20.0,
        tool_calls=2,
        tools=("query_wandb_tool",),
        queried_projects=(SOURCE_PROJECT,),
        scores={"maintainer.factual_correctness": passed},
        score_explanations={
            "maintainer.factual_correctness": "safe deterministic explanation"
        },
        sanitized_answer_excerpt=None,
        actual_query_scope=(SOURCE_PROJECT,),
        reported_project_identity=SOURCE_PROJECT,
        evidence_links=_links(f"{task_id}-{arm}-{attempt}"),
        weave_agent_root_call_id=f"{task_id}-{arm}-{attempt}-agent",
        execution_fingerprint=runtime,
        runtime_lock_digest=runtime,
        infrastructure={"harbor": "passed"},
    )


def _pair(
    *,
    task_id: str,
    attempt: int,
    baseline_passed: bool,
    candidate_passed: bool,
) -> PairedCaseV3:
    if baseline_passed == candidate_passed:
        status = "unchanged"
        dimension_status = "unchanged"
    elif candidate_passed:
        status = "improved"
        dimension_status = "improved"
    else:
        status = "regressed"
        dimension_status = "regressed"
    return PairedCaseV3(
        pair_id=stable_digest(
            {
                "schema_version": 1,
                "task_id": task_id,
                "harness": "claude-code",
                "attempt": attempt,
            }
        ),
        task_id=task_id,
        harness="claude-code",
        attempt=attempt,
        status=status,  # type: ignore[arg-type]
        dimension_changes=(
            DimensionChangeV2(
                id="maintainer.factual_correctness",
                label="Factual correctness",
                status=dimension_status,  # type: ignore[arg-type]
                baseline=baseline_passed,
                candidate=candidate_passed,
                critical=True,
                role="outcome",
            ),
        ),
        baseline=_attempt(
            task_id=task_id,
            arm="baseline",
            attempt=attempt,
            passed=baseline_passed,
        ),
        candidate=_attempt(
            task_id=task_id,
            arm="candidate",
            attempt=attempt,
            passed=candidate_passed,
        ),
        task_label=task_id,
    )


def _result() -> ComparisonResultV3:
    pairs = (
        _pair(
            task_id="reconcile-evaluations",
            attempt=1,
            baseline_passed=False,
            candidate_passed=True,
        ),
        _pair(
            task_id="reconcile-evaluations",
            attempt=2,
            baseline_passed=False,
            candidate_passed=True,
        ),
        _pair(
            task_id="project-health",
            attempt=1,
            baseline_passed=True,
            candidate_passed=True,
        ),
        _pair(
            task_id="project-health",
            attempt=2,
            baseline_passed=True,
            candidate_passed=True,
        ),
    )
    source_digest = stable_digest({"source": "locked"})
    drift = EvidenceDriftCheckV1(
        status="matched",
        expected_digest=source_digest,
        observed_digest=source_digest,
    )
    topology = EvidenceTopologyV1(
        source_destination=_destination(SOURCE_PROJECT),
        result_destination=_destination(RESULT_PROJECT),
        source_lock_digest=source_digest,
        pre_run_drift=drift,
        post_run_drift=drift,
        execution_identity=stable_digest({"runtime": "local-harbor"}),
    )
    aligned_attempts = tuple(
        AlignedAttemptSetV1(
            alignment_id=pair.pair_id,
            task_id=pair.task_id,
            harness=pair.harness,
            attempt=pair.attempt,
            attempt_ids_by_arm={
                "baseline": pair.baseline.attempt_id,  # type: ignore[union-attr]
                "candidate": pair.candidate.attempt_id,  # type: ignore[union-attr]
            },
        )
        for pair in pairs
    )
    analysis = AlignedAnalysisV1(
        study_intent="mcp_release_maintenance",
        reference_arm="baseline",
        arms=(
            AlignedArmV1(id="baseline", label="main"),
            AlignedArmV1(id="candidate", label="staging"),
        ),
        contrasts=(
            AlignedContrastV1(
                id="staging-vs-main",
                reference_arm="baseline",
                treatment_arms=("candidate",),
                dimensions=(
                    AlignedDimensionV1(
                        id="maintainer.factual_correctness",
                        label="Factual correctness",
                        role="outcome",
                        critical=True,
                    ),
                ),
            ),
        ),
        aligned_attempts=aligned_attempts,
        task_summaries=(
            TaskStratifiedSummaryV1(
                task_id="reconcile-evaluations",
                validity="valid",
                pair_counts={"improved": 2},
            ),
            TaskStratifiedSummaryV1(
                task_id="project-health",
                validity="non_discriminating",
                pair_counts={"unchanged": 2},
                blockers=("task did not discriminate revisions",),
            ),
        ),
    )
    behavioral = BehavioralSummaryV1(
        status="improved",
        recommendation="Investigate the repeated baseline failure.",
        improved_pairs=2,
        regressed_pairs=0,
        mixed_pairs=0,
        unchanged_pairs=2,
        incomplete_pairs=0,
        candidate_critical_failures=0,
        critical_blockers=(),
        supported_claim="Candidate fixed the locked reconciliation task.",
        limitations=("This is a bounded canary.",),
        next_action="Lock one repeated failure for loop engineering.",
    )
    decision = DecisionSummaryV1(
        status="hold",
        recommendation="Package release remains on hold.",
        release_target="wandb-mcp-server Python package 0.4",
        candidate_sha="4" * 40,
        evidence_grade="A",
        gates=(),
        critical_blockers=("release gates are incomplete",),
        limitations=("Local behavior is not package qualification.",),
        next_action="Complete package gates.",
        human_signoff_required=True,
    )
    cohort_lineage = {
        "schema_version": 1,
        "source_lock_digest": source_digest,
        "taskset_digest": "1" * 64,
        "private_labels_digest": "2" * 64,
        "arms": {
            "baseline": {"behavior_digest": "3" * 64},
            "candidate": {"behavior_digest": "4" * 64},
        },
        "execution": {"runtime_digest": "5" * 64},
        "scorer_digests": {"maintainer-scorer": "b" * 64},
    }
    cohort_lineage["lineage_digest"] = stable_digest(cohort_lineage)
    return ComparisonResultV3(
        schema_version=3,
        comparison_id=COMPARISON_ID,
        preview_digest="a" * 64,
        source="local-harbor-mcp-canary",
        evidence_project=RESULT_PROJECT,
        rows=8,
        baseline_passed=2,
        candidate_passed=4,
        improved=2,
        regressed=0,
        mixed=0,
        unchanged=2,
        incomplete=0,
        required_evaluations_incomplete=0,
        deterministic_summary={},
        judge_summary={},
        mechanism_summary={},
        operational_summary={},
        evidence_links=(),
        paired_cases=pairs,
        limitations=("bounded canary",),
        integrity={
            "status": "reconciled",
            "row_count": 8,
            "unique_attempts": 8,
            "duplicate_attempt_ids": [],
            "unresolved_evidence_attempts": 0,
            "invalid_evidence_attempts": 0,
            "cross_project_attempts": 0,
        },
        behavioral_summary=behavioral,
        decision_policy=None,
        decision=decision,
        evidence_topology=topology,
        aligned_analysis=analysis,
        task_validity=(
            TaskValidityV1(
                task_id="reconcile-evaluations",
                status="valid",
                discriminating_dimensions=("maintainer.factual_correctness",),
            ),
            TaskValidityV1(
                task_id="project-health",
                status="non_discriminating",
                reasons=("both revisions passed",),
            ),
        ),
        release_note_coverage=(),
        scorer_revisions=(
            LockDescriptorV1(
                id="maintainer-scorer",
                label="Maintainer scorer",
                digest="b" * 64,
            ),
        ),
        runtime_locks=(
            LockDescriptorV1(
                id="local-harbor",
                label="Local Harbor runtime",
                digest="c" * 64,
            ),
        ),
        cohort_lineage=cohort_lineage,
        candidate_source_revisions=(
            CandidateSourceRevisionV1(
                kind="mcp",
                id="wandb-mcp-main",
                version_identity="git:" + "5" * 40,
                runtime_digest="sha256:" + "6" * 64,
                lock_digest="sha256:" + "7" * 64,
            ),
            CandidateSourceRevisionV1(
                kind="mcp",
                id="wandb-mcp-0-4-staging",
                version_identity="git:" + "4" * 40,
                runtime_digest="sha256:" + "8" * 64,
                lock_digest="sha256:" + "9" * 64,
            ),
        ),
        qualification_digest="d" * 64,
        result_digest="e" * 64,
    )


def _build_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result: ComparisonResultV3 | None = None,
    *,
    preview_spec_digest: str = "f" * 64,
    expected_spec_digest: str = "f" * 64,
) -> dict[str, object]:
    result_path = tmp_path / "result.json"
    result_path.write_text('{"schema_version":3}\n', encoding="utf-8")
    preview_path = tmp_path / "prepared-preview.json"
    preview = {
        "schema_version": 3,
        "comparison": {
            "id": COMPARISON_ID,
            "spec_digest": preview_spec_digest,
        },
        "readiness": {},
        "matrix": {},
        "experiment": {},
        "manifest": {},
        "preview_digest": "",
    }
    preview["preview_digest"] = stable_digest(preview)
    preview_path.write_text(json.dumps(preview), encoding="utf-8")
    selected_result = replace(
        result or _result(),
        preview_digest=str(preview["preview_digest"]),
    )
    monkeypatch.setattr(
        "fugue.bench.loop_failure.read_comparison_result",
        lambda _: selected_result,
    )
    baseline = selected_result.paired_cases[0].baseline
    assert baseline is not None
    return build_comparison_failure_lock(
        result_path=result_path,
        preview_path=preview_path,
        task_id="reconcile-evaluations",
        arm="baseline",
        primary_attempt_id=baseline.attempt_id,
        expected_comparison_id=COMPARISON_ID,
        expected_source_project=SOURCE_PROJECT,
        expected_result_project=RESULT_PROJECT,
        expected_harness="claude-code",
        expected_tasks=2,
        expected_attempts=2,
        spec_digest=expected_spec_digest,
        locked_at="2026-07-30T10:00:00Z",
        required_source_ids=("wandb-mcp-main", "wandb-mcp-0-4-staging"),
    )


def test_v3_failure_lock_binds_repeated_real_failure_without_answers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = _build_lock(tmp_path, monkeypatch)
    validated = validate_comparison_failure_lock(lock)

    assert validated["failure"]["repeated_attempt_ids"]
    assert len(validated["failure"]["repeated_attempt_ids"]) == 2
    assert validated["failure"]["failed_critical_dimensions"] == [
        "maintainer.factual_correctness"
    ]
    assert validated["source"]["source_project"] == SOURCE_PROJECT
    assert validated["source"]["result_project"] == RESULT_PROJECT
    assert validated["source"]["spec_digest"] == "f" * 64
    assert validated["source"]["preview_artifact_sha256"]
    assert set(validated["locks"]["arm_candidates"]) == {
        "baseline",
        "candidate",
    }
    assert validated["locks"]["arm_runtimes"]["baseline"]
    serialized = json.dumps(validated, sort_keys=True).lower()
    assert "expected_answer" not in serialized
    assert "sanitized_answer_excerpt" not in serialized
    assert "news" + "-research-agent" not in serialized

    tampered = json.loads(json.dumps(validated))
    tampered["locks"]["arm_candidates"]["baseline"] = "0" * 64
    tampered["lock_sha256"] = stable_digest(
        {**tampered, "lock_sha256": ""}
    )
    with pytest.raises(ValueError, match="primary attempt candidate"):
        validate_comparison_failure_lock(tampered)


def test_failure_lock_rejects_preview_spec_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="preview spec digest changed"):
        _build_lock(
            tmp_path,
            monkeypatch,
            preview_spec_digest="0" * 64,
        )


def test_failure_lock_rejects_non_valid_or_drifted_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _result()
    invalid_task = replace(
        result,
        task_validity=(
            TaskValidityV1(
                task_id="reconcile-evaluations",
                status="inconclusive",
                reasons=("human review required",),
            ),
            result.task_validity[1],
        ),
    )
    with pytest.raises(ValueError, match="not valid discriminating"):
        _build_lock(tmp_path, monkeypatch, invalid_task)

    drifted = EvidenceDriftCheckV1(
        status="drifted",
        expected_digest="1" * 64,
        observed_digest="2" * 64,
        reason="source changed",
    )
    drifted_result = replace(
        result,
        evidence_topology=replace(
            result.evidence_topology,
            post_run_drift=drifted,
            topology_digest="",
        ),
    )
    with pytest.raises(ValueError, match="source evidence drifted"):
        _build_lock(tmp_path, monkeypatch, drifted_result)


def test_loop_catalog_is_local_dynamic_and_holdout_locked() -> None:
    experiment = get_experiment("claude-loop-skill-mcp", REPO_ROOT)
    campaign = get_campaign("claude-loop-skill-mcp-v1", REPO_ROOT)
    analysis = get_analysis("claude-loop-discovery-selection", REPO_ROOT)
    presets = {item.id: item for item in experiment.presets}

    assert experiment.source_evidence_project == SOURCE_PROJECT
    assert experiment.evidence_project == LOOP_PROJECT
    assert [item.id for item in experiment.variants] == [
        "production",
        "skill-only",
        "mcp-only",
        "combined",
    ]
    assert {
        skill
        for variant in experiment.variants
        for skill in variant.skills
    } == {"loop-production-skill", "loop-intervention-skill"}
    assert {
        integration.id
        for variant in experiment.variants
        for integration in variant.integrations
    } == {"loop-production-mcp", "loop-intervention-mcp"}
    assert presets["discovery"].environment == {"type": "docker"}
    assert presets["discovery"].n_tasks == 2
    assert presets["holdout"].environment == {"type": "docker"}
    assert presets["holdout"].n_tasks == 4
    assert presets["holdout"].selection_lock_required is True
    assert presets["holdout"].selection_lock_kind == "intervention"
    assert campaign.limits.total_cost_usd == 20
    assert [stage.max_cells for stage in campaign.stages] == [8, 8]
    assert campaign.task_authoring is not None
    assert campaign.task_authoring.adaptive_discovery is True
    assert analysis.selection is not None
    assert analysis.selection.require_skill_invocation is True
    assert analysis.selection.minimum_paired_pass_rate_delta == 0.5


def test_loop_assets_do_not_claim_old_smoke_or_fake_trace() -> None:
    paths = (
        REPO_ROOT / "configs/fugue/experiments/claude-loop-skill-mcp.yaml",
        REPO_ROOT / "configs/fugue/campaigns/claude-loop-skill-mcp-v1.yaml",
        REPO_ROOT
        / "examples/loop-engineering/wandb-evidence-loop/README.md",
        REPO_ROOT
        / "examples/loop-engineering/wandb-evidence-loop/agent-prompt.md",
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert "a2bae727" not in text
    assert "571ee85" not in text
    assert "50-call" not in text
    assert "339-second" not in text
    assert "news" + "-research-agent" not in text
    assert COMPARISON_ID in text


def test_loop_selection_resolves_only_locked_components_in_selected_arm(
    tmp_path: Path,
) -> None:
    skill = build_intervention_component_lock(
        kind="skill",
        component_id="loop-intervention-skill",
        lock_digest="7" * 64,
        repository="https://github.com/wandb/fugue",
        source_commit="8" * 40,
        source_tree="9" * 40,
    )
    mcp = build_intervention_component_lock(
        kind="mcp",
        component_id="loop-intervention-mcp",
        lock_digest="a" * 64,
        repository="https://github.com/wandb/wandb-mcp-server",
        source_commit="b" * 40,
        source_tree="c" * 40,
        release_target="wandb-mcp-server Python package 0.4",
        superseded_release_candidate_sha="d" * 40,
        release_requalification_required=True,
    )
    paths = (
        write_intervention_component_lock(
            tmp_path / ".fugue/interventions/skill.json",
            skill,
        ).relative_to(tmp_path),
        write_intervention_component_lock(
            tmp_path / ".fugue/interventions/mcp.json",
            mcp,
        ).relative_to(tmp_path),
    )
    tags = [
        f"intervention-component-lock:{path.as_posix()}"
        for path in paths
    ]
    rows = [
        {
            "variant_id": "combined",
            "skill_ids": ["loop-intervention-skill"],
            "integration_ids": ["loop-intervention-mcp"],
            "tags": tags,
        },
        {
            "variant_id": "combined",
            "skill_ids": ["loop-intervention-skill"],
            "integration_ids": ["loop-intervention-mcp"],
            "tags": list(reversed(tags)),
        },
    ]

    resolved = _intervention_component_locks(
        rows,
        selected_variant="combined",
        repo_root=tmp_path,
    )

    assert resolved == (mcp, skill)


def _verification_module():
    path = (
        REPO_ROOT
        / "examples/loop-engineering/wandb-evidence-loop/verify_evidence.py"
    )
    spec = importlib.util.spec_from_file_location("loop_verify_evidence", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_loop_mcp_use_requires_host_observed_calls_from_exact_integration() -> None:
    module = _verification_module()
    row = {
        "integration_ids": ["loop-intervention-mcp"],
        "integration_ids_invoked": ["some-other-mcp"],
        "mcp_tool_calls": [
            {
                "integration_id": "some-other-mcp",
                "terminal_status": "succeeded",
                "successful": True,
                "response_metadata_verified": True,
            }
        ],
    }

    assert (
        module._integration_use_observed(row, "loop-intervention-mcp")
        is False
    )
    row["integration_ids_invoked"] = ["loop-intervention-mcp"]
    row["mcp_tool_calls"][0]["integration_id"] = "loop-intervention-mcp"
    assert (
        module._integration_use_observed(row, "loop-intervention-mcp")
        is True
    )


def _verification_row(
    *,
    phase: str,
    task: str,
    variant: str,
    candidate_id: str,
    passed: bool,
    failure_lock: dict[str, object],
) -> dict[str, object]:
    identity = stable_digest(
        {"phase": phase, "task": task, "variant": variant}
    )
    weave_base = f"https://wandb.ai/{LOOP_PROJECT}/weave"
    call_base = f"weave:///{LOOP_PROJECT}/call"
    evaluation_id = f"evaluation-{identity}"
    predict_and_score_id = f"predict-and-score-{identity}"
    prediction_call_id = f"prediction-call-{identity}"
    agent_id = f"agent-{identity}"
    row: dict[str, object] = {
        "attempt_id": identity,
        "prediction_id": f"prediction-{identity}",
        "comparison_example_id": task,
        "task_id": task,
        "variant_id": variant,
        "candidate_id": candidate_id,
        "harness": "claude-code",
        "trial_index": 1,
        "pass": passed,
        "trace_project": LOOP_PROJECT,
        "trace_receipt": {"app_base_url": "https://wandb.ai"},
        "trace_link_status": "linked",
        "weave_evaluation_root_call_id": evaluation_id,
        "weave_evaluation_root_ref": f"{call_base}/{evaluation_id}",
        "weave_evaluation_root_url": f"{weave_base}/calls/{evaluation_id}",
        "evaluation_root_object_verified": True,
        "eval_predict_and_score_call_id": predict_and_score_id,
        "eval_predict_and_score_ref": f"{call_base}/{predict_and_score_id}",
        "eval_predict_and_score_url": (
            f"{weave_base}/calls/{predict_and_score_id}"
        ),
        "eval_predict_and_score_object_verified": True,
        "weave_prediction_call_id": prediction_call_id,
        "weave_prediction_ref": f"{call_base}/{prediction_call_id}",
        "weave_prediction_url": f"{weave_base}/calls/{prediction_call_id}",
        "weave_prediction_object_verified": True,
        "weave_agent_root_call_id": agent_id,
        "weave_agent_root_ref": f"{call_base}/{agent_id}",
        "weave_agent_root_url": f"{weave_base}/calls/{agent_id}",
        "agent_graph_verified": True,
        "weave_dataset_id": f"weave:///{LOOP_PROJECT}/object/loop-dataset:v1",
        "weave_dataset_ref": f"weave:///{LOOP_PROJECT}/object/loop-dataset:v1",
        "weave_dataset_url": f"{weave_base}/objects/loop-dataset/versions/v1",
        "dataset_version_object_verified": True,
        "evaluation_publication_mode": "live",
        "harbor_environment": "local_harbor_docker",
        "harbor_conformance_status": "passed",
        "harbor_policy_attestation_verified": True,
        "privacy_contract_version": 2,
        "local_artifact_privacy_scan_status": "passed",
        "hosted_evidence_privacy_scan_status": "passed",
        "private_label_boundary_verified": True,
        "sandbox_cleanup_verified": True,
        "orphaned_sandbox": False,
        "execution_fingerprint": stable_digest({"phase": phase}),
        "run_snapshot_sha256": "2" * 64,
        "task_suite_digest": (
            "6" * 64 if phase == "discovery" else "7" * 64
        ),
        "source_commit": "3" * 40,
        "source_tree": "4" * 40,
        "source_dirty_digest": "",
        "weave_tool_names": {"query_wandb_tool": 1},
        "skill_invocation_evidence": {
            "status": "observed",
            "skills_invoked": [
                (
                    "loop-intervention-skill"
                    if variant in {"skill-only", "combined"}
                    else "loop-production-skill"
                )
            ],
        },
        "skill_provenance": [
            {
                "id": (
                    "loop-intervention-skill"
                    if variant in {"skill-only", "combined"}
                    else "loop-production-skill"
                ),
                "digest": (
                    "7" * 64
                    if variant in {"skill-only", "combined"}
                    else "8" * 64
                ),
            }
        ],
        "integration_provenance": [
            {
                "id": (
                    "loop-intervention-mcp"
                    if variant in {"mcp-only", "combined"}
                    else "loop-production-mcp"
                ),
                "lock_digest": (
                    "sha256:" + "a" * 64
                    if variant in {"mcp-only", "combined"}
                    else "sha256:" + "b" * 64
                ),
            }
        ],
        "integration_ids_invoked": [
            (
                "loop-intervention-mcp"
                if variant in {"mcp-only", "combined"}
                else "loop-production-mcp"
            )
        ],
        "mcp_tool_calls": [
            {
                "tool": "query_wandb_tool",
                "integration_id": (
                    "loop-intervention-mcp"
                    if variant in {"mcp-only", "combined"}
                    else "loop-production-mcp"
                ),
                "terminal_status": "succeeded",
                "successful": True,
                "response_metadata_verified": True,
            }
        ],
    }
    if phase == "discovery" and variant == "production" and not passed:
        row["source_failure"] = {
            "lock_sha256": failure_lock["lock_sha256"],
            "task_id": failure_lock["failure"]["task_id"],  # type: ignore[index]
            "primary_attempt_id": failure_lock["failure"][  # type: ignore[index]
                "primary_attempt_id"
            ],
        }
    return row


def test_loop_qualification_receipt_accepts_exact_8_plus_8(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _verification_module()
    candidates = {
        variant: stable_digest({"candidate": variant})
        for variant in ("production", "skill-only", "mcp-only", "combined")
    }
    discovery_tasks = ("a" * 64, "b" * 64)
    holdout_tasks = ("c" * 64, "d" * 64, "e" * 64, "f" * 64)
    failure = {
        "lock_sha256": "1" * 64,
        "locked_at": "2026-07-30T10:00:00+00:00",
        "failure": {
            "task_id": "reconcile-evaluations",
            "primary_attempt_id": "9" * 64,
        },
    }
    discovery = [
        _verification_row(
            phase="discovery",
            task=task,
            variant=variant,
            candidate_id=candidates[variant],
            passed=not (task == discovery_tasks[0] and variant == "production"),
            failure_lock=failure,
        )
        for task in discovery_tasks
        for variant in ("production", "skill-only", "mcp-only", "combined")
    ]
    holdout = [
        _verification_row(
            phase="holdout",
            task=task,
            variant=variant,
            candidate_id=candidates[variant],
            passed=True,
            failure_lock=failure,
        )
        for task in holdout_tasks
        for variant in ("production", "combined")
    ]
    discovery_path = tmp_path / "discovery.jsonl"
    holdout_path = tmp_path / "holdout.jsonl"
    discovery_path.write_text(
        "".join(json.dumps(row) + "\n" for row in discovery),
        encoding="utf-8",
    )
    holdout_path.write_text(
        "".join(json.dumps(row) + "\n" for row in holdout),
        encoding="utf-8",
    )
    selection = SimpleNamespace(
        selected_variant_id="combined",
        rankings=tuple(
            {
                "variant_id": variant,
                "candidate_digest": candidate_id,
            }
            for variant, candidate_id in candidates.items()
        ),
        discovery_run_snapshot_sha256s=("2" * 64,),
        comparison_example_ids=discovery_tasks,
        discovery_variant_ids=(
            "production",
            "skill-only",
            "mcp-only",
            "combined",
        ),
        selected_components=(
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
        ),
        failure_lock_sha256="1" * 64,
        discovery_suite_sha256="6" * 64,
        holdout_suite_sha256="7" * 64,
        failure_locked_at="2026-07-30T10:00:00+00:00",
        source_commit="3" * 40,
        source_tree="4" * 40,
        source_dirty_digest="",
        lock_sha256="5" * 64,
    )
    monkeypatch.setattr(
        module,
        "read_intervention_selection_lock",
        lambda _: selection,
    )
    monkeypatch.setattr(
        module,
        "read_comparison_failure_lock",
        lambda _: failure,
    )
    monkeypatch.setattr(
        module,
        "resolve_fugue_source_provenance",
        lambda _: {
            "commit": "3" * 40,
            "tree": "4" * 40,
            "dirty": False,
            "dirty_digest": "",
        },
    )
    monkeypatch.setattr(
        module,
        "verify_intervention_component_checkout",
        lambda component, _path: {
            "schema_version": 1,
            "kind": "intervention-component-checkout-verification",
            "component_id": component.component_id,
            "verified": True,
            "pr_tree_matches_qualified_tree": True,
            "release_requalification_required": (
                component.release_requalification_required
            ),
            "blockers": [],
            "receipt_digest": "e" * 64,
        },
    )

    receipt = module.verify(
        discovery_path=discovery_path,
        holdout_path=holdout_path,
        selection_lock_path=tmp_path / "selection.json",
        failure_lock_path=tmp_path / "failure.json",
        repo_root=tmp_path,
        component_worktrees={
            "loop-intervention-skill": tmp_path / "skill",
            "loop-intervention-mcp": tmp_path / "mcp",
        },
    )

    assert receipt["qualified"] is True
    assert receipt["discovery_candidate_improvements"] == 1
    assert receipt["holdout_regressions"] == 0
    assert receipt["release_requalification_required"] is True
    assert receipt["superseded_release_candidates"] == ["d" * 40]
    assert len(receipt["component_checkout_receipts"]) == 2
    assert receipt["blockers"] == []
