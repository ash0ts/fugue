from __future__ import annotations

import hashlib
import json
import re
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

import fugue.bench.local_publication as publication
from fugue.bench.candidates import attempt_id, attempt_identity, stable_digest
from fugue.bench.comparison import (
    ComparisonResultV3,
    analyze_comparison_rows,
    comparison_from_dict,
    materialize_comparison,
    preview_comparison,
    scaffold_comparison,
    write_comparison_result,
)
from fugue.bench.export import PredictionRowV1
from fugue.bench.files import atomic_write_json
from fugue.bench.local_evidence import (
    AgentEvidenceReceiptV1,
    LocalArtifactRefV1,
    LocalAttemptPlanV1,
    LocalEvidenceCoordinator,
    LocalEvidenceStore,
    build_local_evidence_run_plan,
    local_attempt_refs,
)
from fugue.bench.local_publication import (
    LocalResultPublicationError,
    MissingWeaveExtraError,
    StudyPublicationScopeV1,
    WeaveHostedObjectRefV1,
    WeavePublicationOutcomeV1,
    WeavePublicationReceiptV1,
    WeavePublicationTargetV1,
    publish_local_result_to_weave,
    read_weave_publication_receipt,
    weave_publication_receipt_from_dict,
    weave_publisher_from_environment,
)
from fugue.bench.operator import OperatorService
from fugue.bench.task_presentation import (
    PublicPromptPartV1,
    TaskPresentationV1,
)


class _FakeComparisonResultV3:
    def __init__(self, manifest: Any, manifest_file_sha256: str) -> None:
        self.schema_version = 3
        self.source = manifest.run_id
        self.result_digest = stable_digest({"result": manifest.run_id})
        self.qualification_digest = stable_digest({"qualification": manifest.run_id})
        self.evidence_backend = "local"
        self.publication_status = "not_requested"
        self.local_chain_integrity = "reconciled"
        self.hosted_chain_integrity = "not_applicable"
        self.comparison_id = "local-publication-comparison"
        self.evidence_topology = SimpleNamespace(
            result_destination=manifest.destination
        )
        conformance = manifest.run_conformance
        self.local_evidence = {
            "schema_version": 1,
            "run_id": manifest.run_id,
            "manifest_digest": manifest.manifest_digest,
            "manifest_file_sha256": manifest_file_sha256,
            "plan_digest": manifest.plan_digest,
            "attempt_record_set_digest": manifest.attempt_record_set_digest,
            "prediction_row_set_digest": manifest.prediction_row_set_digest,
            "run_conformance_receipt_digest": (
                conformance.receipt_sha256 if conformance is not None else "0" * 64
            ),
            "run_conformance_file_sha256": (
                conformance.sha256 if conformance is not None else "0" * 64
            ),
            **(
                {
                    "result_row_projection_set_digest": (
                        manifest.result_row_projection_set_digest
                    )
                }
                if manifest.result_row_projection_set_digest is not None
                else {}
            ),
        }
        self.rows = len(manifest.planned_attempt_ids)
        records = {item.attempt_id: item for item in manifest.attempt_records}
        self.paired_cases = tuple(
            SimpleNamespace(
                task_id="task-1",
                harness="claude-code",
                attempt=1,
                baseline=SimpleNamespace(
                    attempt_id=attempt,
                    identity={
                        "candidate": stable_digest({"candidate": "baseline"}),
                        "runtime": stable_digest({"runtime": "harbor-local-v1"}),
                    },
                    prediction_id=stable_digest(
                        {"record_type": "prediction", "attempt_id": attempt}
                    ),
                    execution_status="unknown",
                    evaluation_status="unknown",
                    evidence_status="reconciled",
                    passed=None,
                    cost_usd=None,
                    latency_sec=None,
                    input_tokens=None,
                    output_tokens=None,
                    cost_reconciliation_status=None,
                    latency_reconciliation_status=None,
                    usage_reconciliation_status=None,
                    tool_calls=0,
                    tools=(),
                    queried_projects=(),
                    scores={},
                    score_explanations={},
                    sanitized_answer_excerpt=None,
                    actual_query_scope=(),
                    reported_project_identity=None,
                    execution_fingerprint=None,
                    runtime_lock_digest=None,
                    local_evidence_record_digest=(
                        records[attempt].record_digest
                        if attempt in records
                        else "0" * 64
                    ),
                    local_prediction_row_sha256=(
                        records[attempt].prediction_row_sha256
                        if attempt in records
                        else "0" * 64
                    ),
                    local_result_row_projection_digest=(
                        records[attempt].result_row_projection_digest
                        if attempt in records
                        else "0" * 64
                    ),
                ),
                candidate=None,
            )
            for attempt in manifest.planned_attempt_ids
        )


def _canonical_manifest(tmp_path: Path, *, complete: bool = True):
    run_id = "local-publication-run-v1"
    candidate = stable_digest({"candidate": "baseline"})
    runtime = stable_digest({"runtime": "harbor-local-v1"})
    identity = attempt_identity(
        task_id="task-1",
        arm="baseline",
        harness="claude-code",
        attempt=1,
        candidate=candidate,
        runtime=runtime,
    )
    canonical_attempt_id = attempt_id(**identity)
    attempt = LocalAttemptPlanV1(
        run_id=run_id,
        cell_id="task-1-baseline-1",
        attempt_id=canonical_attempt_id,
        attempt_identity=identity,
        prediction_id=stable_digest(
            {"record_type": "prediction", "attempt_id": canonical_attempt_id}
        ),
        evaluation_scope_id=stable_digest({"evaluation": run_id}),
        dataset_id=stable_digest({"dataset": run_id}),
    )
    plan = build_local_evidence_run_plan(
        run_id=run_id,
        run_snapshot_sha256=stable_digest({"snapshot": run_id}),
        evaluation_asset_lock_sha256=stable_digest({"assets": run_id}),
        attempts=(attempt,),
    )
    store = LocalEvidenceStore(tmp_path, run_id)
    coordinator = LocalEvidenceCoordinator(store, plan)
    if complete:
        harbor_receipt = {
            "schema_version": 2,
            "run_id": run_id,
            "backend": "local_harbor_docker",
            "status": "passed",
            "enforced": True,
            "generated_at": "2026-08-17T12:00:00+00:00",
            "receipt_sha256": "",
        }
        harbor_receipt["receipt_sha256"] = stable_digest(harbor_receipt)
        atomic_write_json(store.run_conformance_path, harbor_receipt)
        transcript_path = (
            tmp_path
            / ".fugue"
            / "runtime"
            / run_id
            / "agent-native"
            / canonical_attempt_id
            / "transcript.jsonl"
        )
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text(
            json.dumps(
                {
                    "attempt_id": canonical_attempt_id,
                    "session_id": "session-publication-test",
                    "answer": "bounded public answer",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raw_transcript = transcript_path.read_bytes()
        transcript_artifact = LocalArtifactRefV1(
            path=transcript_path.relative_to(tmp_path).as_posix(),
            sha256=hashlib.sha256(raw_transcript).hexdigest(),
            size_bytes=len(raw_transcript),
            media_type="application/x-ndjson",
        )
        coordinator.begin_attempt(canonical_attempt_id)
        coordinator.finish_attempt(
            canonical_attempt_id,
            terminal_status="passed",
            prediction_row=PredictionRowV1(
                prediction_id=attempt.prediction_id,
                run_id=run_id,
                candidate_id=candidate,
                comparison_example_id=stable_digest({"example": "task-1"}),
                trial_index=1,
                execution_kind="agent",
                source_record_type="trial",
                payload={
                    "attempt_id": canonical_attempt_id,
                    "attempt_identity": identity,
                    "answer": "bounded public answer",
                },
            ).to_dict(),
            scores={"answer_correct": True},
            attempt_payload={
                "attempt_id": canonical_attempt_id,
                "terminal_status": "passed",
                "receipts": {
                    "privacy": {"status": "passed"},
                    "policy": {"status": "passed"},
                    "usage": {"status": "passed"},
                    "cleanup": {"status": "passed"},
                },
            },
            agent_receipt=AgentEvidenceReceiptV1(
                attempt_id=canonical_attempt_id,
                planned_conversation_id="planned-publication-test",
                primary_session_id="session-publication-test",
                child_session_ids=(),
                artifacts=(transcript_artifact,),
                transcript_artifact=transcript_artifact,
                transcript_session_id="session-publication-test",
                correlation_verified=True,
                tool_event_count=0,
                tool_events_sha256=stable_digest([]),
                response_sha256=stable_digest({"answer": "bounded public answer"}),
            ),
            recorded_at="2026-08-17T12:00:00+00:00",
        )
        manifest = coordinator.finalize()
    else:
        manifest = store.build_manifest()
        atomic_write_json(store.manifest_path, manifest.to_dict())
    return store.manifest_path, manifest


def _real_interrupted_v3_result(
    tmp_path: Path,
) -> tuple[Path, ComparisonResultV3, Path, Any]:
    """Build one digest-valid local V3 result with no behavioral verdicts."""

    comparison_path = scaffold_comparison(tmp_path)
    raw = yaml.safe_load(comparison_path.read_text(encoding="utf-8"))
    raw["execution"]["attempts"] = 1
    task_path = tmp_path / raw["taskset"]["tasks"]
    label_path = tmp_path / raw["taskset"]["private_labels"]
    task_line = task_path.read_text(encoding="utf-8").splitlines()[0]
    label_line = label_path.read_text(encoding="utf-8").splitlines()[0]
    task_path.write_text(task_line + "\n", encoding="utf-8")
    label_path.write_text(label_line + "\n", encoding="utf-8")
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
    run_id = "local-v3-interrupted-publication"
    task_id = str(approved["expected_cells"][0]["task_id"])
    presentation = TaskPresentationV1(
        task_id=task_id,
        title="Find the current reimbursement cap",
        public_prompt=(
            PublicPromptPartV1(
                order=1,
                text="Inspect the supplied policy files and return the current cap.",
            ),
        ),
        required_output="Return JSON with amount_usd and source.",
        public_acceptance_criteria=(
            "Use the currently effective policy file.",
            "Return the requested JSON fields.",
        ),
        scenario="Policy maintenance",
        tags=("skill", "current-source"),
        partition="development",
    )
    plans: list[LocalAttemptPlanV1] = []
    rows: list[dict[str, Any]] = []
    drift = {
        "status": "matched",
        "expected_digest": approved["source_lock_digest"],
        "observed_digest": approved["source_lock_digest"],
    }
    for cell in approved["expected_cells"]:
        arm = str(cell["variant_id"])
        arm_label = "Baseline" if arm == "baseline" else "Candidate"
        treatment_summary = (
            "Claude Code without the assigned Skill."
            if arm == "baseline"
            else "Claude Code with the locked verify-current-source Skill."
        )
        prediction_id = stable_digest(
            {"record_type": "prediction", "attempt_id": cell["attempt_id"]}
        )
        plan = LocalAttemptPlanV1(
            run_id=run_id,
            cell_id=str(cell.get("cell_id") or cell["attempt_id"]),
            attempt_id=str(cell["attempt_id"]),
            attempt_identity=dict(cell["attempt_identity"]),
            prediction_id=prediction_id,
            evaluation_scope_id=stable_digest(
                {"run_id": run_id, "attempt_id": cell["attempt_id"], "kind": "eval"}
            ),
            dataset_id=stable_digest(
                {
                    "run_id": run_id,
                    "attempt_id": cell["attempt_id"],
                    "kind": "dataset",
                }
            ),
            arm_label=arm_label,
            treatment_summary=treatment_summary,
            task_presentation=presentation,
        )
        plans.append(plan)
        refs = local_attempt_refs(plan)
        rows.append(
            {
                **dict(cell),
                "run_id": run_id,
                "prediction_id": prediction_id,
                "approved_comparison": approved,
                "trace_receipt": approved["evidence_destination"],
                "source_pre_run_drift": drift,
                "source_checkpoint_drift": drift,
                "source_post_run_drift": drift,
                "status": "failed",
                "runtime_outcome": "interrupted",
                "terminal_kind": "transport_interrupted",
                "benchmark_outcome": "unscored",
                "pass": None,
                "comparison_evaluation_status": "unavailable",
                "comparison_evaluation_reason": (
                    "Agent execution was interrupted; no behavioral output is "
                    "available to score"
                ),
                "comparison_required_evaluation_complete": False,
                "evidence_backend": "local",
                "task_presentation": presentation.to_dict(),
                "arm_label": arm_label,
                "treatment_summary": treatment_summary,
                "local_evidence_links": [
                    {
                        "kind": kind,
                        "status": "resolved",
                        "system": "local_artifact",
                        "ref": refs[kind],
                    }
                    for kind in (
                        "evaluation_root",
                        "prediction_and_score",
                        "prediction",
                        "agent_root",
                        "dataset",
                    )
                ],
                "privacy_contract_version": 2,
                "local_artifact_privacy_scan_status": "passed",
                "hosted_evidence_privacy_scan_status": "not_applicable",
                "private_label_boundary_verified": True,
                "cost_reconciliation_status": "unavailable",
                "latency_reconciliation_status": "unavailable",
                "usage_reconciliation_status": "unavailable",
            }
        )

    plan = build_local_evidence_run_plan(
        run_id=run_id,
        run_snapshot_sha256=stable_digest({"snapshot": run_id}),
        evaluation_asset_lock_sha256=stable_digest({"assets": run_id}),
        attempts=tuple(plans),
    )
    store = LocalEvidenceStore(tmp_path, run_id)
    coordinator = LocalEvidenceCoordinator(store, plan)
    harbor_receipt = {
        "schema_version": 2,
        "run_id": run_id,
        "backend": "local_harbor_docker",
        "status": "passed",
        "enforced": True,
        "generated_at": "2026-08-19T20:00:00+00:00",
        "receipt_sha256": "",
    }
    harbor_receipt["receipt_sha256"] = stable_digest(harbor_receipt)
    atomic_write_json(store.run_conformance_path, harbor_receipt)
    for plan_attempt, row in zip(plans, rows, strict=True):
        session_id = f"session-{plan_attempt.attempt_id[:16]}"
        transcript_path = (
            tmp_path
            / ".fugue"
            / "runtime"
            / run_id
            / "agent-native"
            / plan_attempt.attempt_id
            / "transcript.jsonl"
        )
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text(
            json.dumps(
                {
                    "attempt_id": plan_attempt.attempt_id,
                    "session_id": session_id,
                    "event": "Agent session started before controller interruption.",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        transcript_raw = transcript_path.read_bytes()
        transcript = LocalArtifactRefV1(
            path=transcript_path.relative_to(tmp_path).as_posix(),
            sha256=hashlib.sha256(transcript_raw).hexdigest(),
            size_bytes=len(transcript_raw),
            media_type="application/x-ndjson",
        )
        coordinator.begin_attempt(plan_attempt.attempt_id)
        coordinator.finish_attempt(
            plan_attempt.attempt_id,
            terminal_status="failed",
            prediction_row=PredictionRowV1(
                prediction_id=plan_attempt.prediction_id,
                run_id=run_id,
                candidate_id=plan_attempt.candidate_id,
                comparison_example_id=stable_digest(
                    {"task_id": task_id, "attempt_id": plan_attempt.attempt_id}
                ),
                trial_index=1,
                execution_kind="agent",
                source_record_type="trial",
                payload=row,
            ).to_dict(),
            scores={},
            attempt_payload={
                "attempt_id": plan_attempt.attempt_id,
                "terminal_status": "failed",
                "receipts": {
                    "privacy": {"status": "passed"},
                    "policy": {"status": "passed"},
                    "usage": {"status": "not_applicable"},
                    "cleanup": {"status": "passed"},
                },
            },
            agent_receipt=AgentEvidenceReceiptV1(
                attempt_id=plan_attempt.attempt_id,
                planned_conversation_id=f"planned-{plan_attempt.attempt_id[:16]}",
                primary_session_id=session_id,
                child_session_ids=(),
                artifacts=(transcript,),
                transcript_artifact=transcript,
                transcript_session_id=session_id,
                correlation_verified=True,
                tool_event_count=0,
                tool_events_sha256=stable_digest([]),
                response_sha256=None,
            ),
            recorded_at="2026-08-19T20:00:00+00:00",
        )
    manifest = coordinator.finalize()
    manifest_file_sha256 = hashlib.sha256(store.manifest_path.read_bytes()).hexdigest()
    records = {item.attempt_id: item for item in manifest.attempt_records}
    conformance = manifest.run_conformance
    assert conformance is not None
    binding = {
        "local_evidence_manifest_digest": manifest.manifest_digest,
        "local_evidence_manifest_file_sha256": manifest_file_sha256,
        "local_evidence_plan_digest": manifest.plan_digest,
        "local_evidence_attempt_record_set_digest": manifest.attempt_record_set_digest,
        "local_evidence_prediction_row_set_digest": manifest.prediction_row_set_digest,
        "local_evidence_result_row_projection_set_digest": (
            manifest.result_row_projection_set_digest
        ),
        "local_evidence_run_receipt_digest": conformance.receipt_sha256,
        "local_evidence_run_receipt_file_sha256": conformance.sha256,
    }
    for row in rows:
        record = records[str(row["attempt_id"])]
        row.update(binding)
        row["local_evidence_record_digest"] = record.record_digest
        row["local_evidence_prediction_row_sha256"] = record.prediction_row_sha256
        row["local_evidence_result_row_projection_digest"] = (
            record.result_row_projection_digest
        )
    result = analyze_comparison_rows(
        comparison_id=spec.id,
        preview_digest=preview.preview_digest,
        rows=rows,
        source=run_id,
        approved_comparison=approved,
        result_schema_version=3,
        study_intent="local_publication_nonbehavioral_regression",
    )
    assert isinstance(result, ComparisonResultV3)
    destination = tmp_path / "real-v3-result"
    destination.mkdir()
    (destination / "attempts.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    result_path, _markdown_path = write_comparison_result(
        result,
        destination=destination,
    )
    return result_path, result, store.manifest_path, manifest


def _result_fixture(tmp_path: Path, manifest: Any) -> tuple[Path, Any]:
    result_path = tmp_path / "results" / "result.json"
    atomic_write_json(
        result_path,
        {"schema_version": 3, "fixture": "validated through the V3 reader seam"},
    )
    manifest_path = LocalEvidenceStore(tmp_path, manifest.run_id).manifest_path
    manifest_file_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return result_path, _FakeComparisonResultV3(
        manifest,
        manifest_file_sha256,
    )


def _outcome(
    manifest: Any,
    target: WeavePublicationTargetV1,
    *,
    publisher_revision: str = "publisher-revision-v1",
) -> WeavePublicationOutcomeV1:
    objects: list[WeaveHostedObjectRefV1] = []
    for current_attempt in manifest.planned_attempt_ids:
        for kind in (
            "evaluation_root",
            "prediction_and_score",
            "prediction",
            "agent_evidence_receipt",
            "dataset",
        ):
            object_id = f"{kind}-{current_attempt[:16]}"
            object_type = "object" if kind == "dataset" else "call"
            objects.append(
                WeaveHostedObjectRefV1(
                    attempt_id=current_attempt,
                    kind=kind,
                    target=target,
                    object_id=object_id,
                    ref=(f"weave:///{target.project_slug}/{object_type}/{object_id}"),
                )
            )
    return WeavePublicationOutcomeV1(
        target=target,
        objects=tuple(sorted(objects, key=lambda item: (item.attempt_id, item.kind))),
        publisher_id="fugue-weave-publisher",
        publisher_revision=publisher_revision,
    )


def _target(
    *,
    project: str = "local-result",
    study_scope: StudyPublicationScopeV1 | None = None,
    api_base_url: str = "https://api.wandb.ai",
    trace_base_url: str = "https://trace.wandb.ai",
    app_base_url: str = "https://wandb.ai",
) -> WeavePublicationTargetV1:
    return publication.weave_publication_target_from_environment(
        f"wandb/{project}",
        {
            "FUGUE_WEAVE_BASE_URL": api_base_url,
            "FUGUE_WEAVE_TRACE_SERVER_URL": trace_base_url,
            "WANDB_APP_BASE_URL": app_base_url,
        },
        study_scope=study_scope,
    )


def _patch_v3_reader(
    monkeypatch: pytest.MonkeyPatch,
    fake_result: _FakeComparisonResultV3,
) -> None:
    monkeypatch.setattr(publication, "ComparisonResultV3", _FakeComparisonResultV3)
    monkeypatch.setattr(
        publication,
        "read_comparison_result",
        lambda _path: fake_result,
    )


def test_publication_is_digest_bound_idempotent_and_does_not_rewrite_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = _canonical_manifest(tmp_path)
    result_path, result = _result_fixture(tmp_path, manifest)
    _patch_v3_reader(monkeypatch, result)
    target = _target()
    calls = 0

    def publisher(*_args: Any) -> WeavePublicationOutcomeV1:
        nonlocal calls
        calls += 1
        return _outcome(manifest, target)

    before = result_path.read_bytes()
    receipt = publish_local_result_to_weave(
        result_path,
        manifest_path,
        target=target,
        publisher=publisher,
        clock=lambda: datetime(2026, 8, 17, 12, tzinfo=UTC),
    )
    replay = publish_local_result_to_weave(
        result_path,
        manifest_path,
        target="wandb/local-result",
        publisher=publisher,
    )

    assert result_path.read_bytes() == before
    assert calls == 1
    assert replay == receipt
    assert receipt.result_digest == result.result_digest
    assert receipt.local_manifest_digest == manifest.manifest_digest
    assert receipt.target.project_slug == "wandb/local-result"
    assert len(receipt.hosted_objects) == 5
    agent = next(
        item for item in receipt.hosted_objects if item.kind == "agent_evidence_receipt"
    )
    assert agent.native_agent_call is False
    assert (
        read_weave_publication_receipt(
            result_path.with_name("weave-publication-receipt.json")
        )
        == receipt
    )


def test_real_digest_valid_nonbehavioral_v3_publishes_no_task_claims(
    tmp_path: Path,
) -> None:
    result_path, result, manifest_path, manifest = _real_interrupted_v3_result(
        tmp_path
    )
    target = _target(project="nonbehavioral-publication")
    observed: list[ComparisonResultV3] = []

    def publisher(
        published_result: ComparisonResultV3,
        _manifest: Any,
        _target: WeavePublicationTargetV1,
    ) -> WeavePublicationOutcomeV1:
        observed.append(published_result)
        return _outcome(manifest, target)

    receipt = publish_local_result_to_weave(
        result_path,
        manifest_path,
        target=target,
        publisher=publisher,
        clock=lambda: datetime(2026, 8, 19, 20, tzinfo=UTC),
    )

    assert receipt.result_digest == result.result_digest
    assert observed == [result]
    assert result.behavioral_summary.status == "incomplete"
    assert result.deterministic_summary == {
        "baseline": {"passed": 0, "evaluated": 0, "dimensions": {}},
        "candidate": {"passed": 0, "evaluated": 0, "dimensions": {}},
    }
    assert result.judge_summary["status"] == "not_used"
    assert result.mechanism_summary == {}
    assert all(
        attempt is not None
        and attempt.execution_status == "interrupted"
        and attempt.passed is None
        and attempt.scores == {}
        and attempt.task_result is None
        for pair in result.paired_cases
        for attempt in (pair.baseline, pair.candidate)
    )
    assert receipt.target.project_slug == "wandb/nonbehavioral-publication"
    assert len(receipt.hosted_objects) == 10


def test_publication_rejects_incomplete_local_chain_before_publisher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = _canonical_manifest(tmp_path, complete=False)
    result_path, result = _result_fixture(tmp_path, manifest)
    _patch_v3_reader(monkeypatch, result)
    called = False

    def publisher(*_args: Any) -> WeavePublicationOutcomeV1:
        nonlocal called
        called = True
        raise AssertionError("publisher must not run")

    with pytest.raises(LocalResultPublicationError, match="not complete"):
        publish_local_result_to_weave(
            result_path,
            manifest_path,
            target="wandb/local-result",
            publisher=publisher,
        )
    assert called is False


def test_publication_requires_result_local_evidence_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = _canonical_manifest(tmp_path)
    result_path, result = _result_fixture(tmp_path, manifest)
    result.local_evidence = None
    _patch_v3_reader(monkeypatch, result)
    called = False

    def publisher(*_args: Any) -> WeavePublicationOutcomeV1:
        nonlocal called
        called = True
        raise AssertionError("publisher must not run")

    with pytest.raises(LocalResultPublicationError, match="missing.*binding"):
        publish_local_result_to_weave(
            result_path,
            manifest_path,
            target="wandb/local-result",
            publisher=publisher,
        )
    assert called is False


@pytest.mark.parametrize(
    "field",
    (
        "manifest_digest",
        "manifest_file_sha256",
        "plan_digest",
        "attempt_record_set_digest",
        "prediction_row_set_digest",
        "run_conformance_receipt_digest",
        "run_conformance_file_sha256",
    ),
)
def test_publication_recomputes_every_result_local_evidence_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    manifest_path, manifest = _canonical_manifest(tmp_path)
    result_path, result = _result_fixture(tmp_path, manifest)
    result.local_evidence[field] = "f" * 64
    _patch_v3_reader(monkeypatch, result)
    called = False

    def publisher(*_args: Any) -> WeavePublicationOutcomeV1:
        nonlocal called
        called = True
        raise AssertionError("publisher must not run")

    with pytest.raises(LocalResultPublicationError, match=field):
        publish_local_result_to_weave(
            result_path,
            manifest_path,
            target="wandb/local-result",
            publisher=publisher,
        )
    assert called is False


@pytest.mark.parametrize(
    ("attribute", "message"),
    (
        ("local_evidence_record_digest", "record digest"),
        ("local_prediction_row_sha256", "prediction digest"),
    ),
)
def test_publication_recomputes_per_attempt_local_digests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    message: str,
) -> None:
    manifest_path, manifest = _canonical_manifest(tmp_path)
    result_path, result = _result_fixture(tmp_path, manifest)
    setattr(result.paired_cases[0].baseline, attribute, "f" * 64)
    _patch_v3_reader(monkeypatch, result)
    called = False

    def publisher(*_args: Any) -> WeavePublicationOutcomeV1:
        nonlocal called
        called = True
        raise AssertionError("publisher must not run")

    with pytest.raises(LocalResultPublicationError, match=message):
        publish_local_result_to_weave(
            result_path,
            manifest_path,
            target="wandb/local-result",
            publisher=publisher,
        )
    assert called is False


@pytest.mark.parametrize(
    ("attribute", "value"),
    (
        ("passed", False),
        ("scores", {"answer_correct": False}),
        ("score_explanations", {"answer_correct": "altered explanation"}),
        ("sanitized_answer_excerpt", "altered excerpt"),
        ("cost_usd", 42.0),
    ),
)
def test_publication_rejects_decision_field_mutation_against_prediction_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    value: Any,
) -> None:
    manifest_path, manifest = _canonical_manifest(tmp_path)
    result_path, result = _result_fixture(tmp_path, manifest)
    setattr(result.paired_cases[0].baseline, attribute, value)
    _patch_v3_reader(monkeypatch, result)
    called = False

    def publisher(*_args: Any) -> WeavePublicationOutcomeV1:
        nonlocal called
        called = True
        raise AssertionError("publisher must not run")

    with pytest.raises(LocalResultPublicationError, match="decision fields"):
        publish_local_result_to_weave(
            result_path,
            manifest_path,
            target="wandb/local-result",
            publisher=publisher,
        )
    assert called is False


def test_publication_detects_manifest_raw_byte_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = _canonical_manifest(tmp_path)
    result_path, result = _result_fixture(tmp_path, manifest)
    _patch_v3_reader(monkeypatch, result)
    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")

    with pytest.raises(
        LocalResultPublicationError,
        match="manifest_file_sha256",
    ):
        publish_local_result_to_weave(
            result_path,
            manifest_path,
            target="wandb/local-result",
            publisher=lambda *_args: (_ for _ in ()).throw(
                AssertionError("publisher must not run")
            ),
        )


def test_publication_detects_harbor_receipt_raw_byte_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = _canonical_manifest(tmp_path)
    result_path, result = _result_fixture(tmp_path, manifest)
    _patch_v3_reader(monkeypatch, result)
    receipt_path = LocalEvidenceStore(
        tmp_path,
        manifest.run_id,
    ).run_conformance_path
    receipt_path.write_bytes(receipt_path.read_bytes() + b"\n")

    with pytest.raises(
        LocalResultPublicationError,
        match="manifest failed canonical recomputation",
    ):
        publish_local_result_to_weave(
            result_path,
            manifest_path,
            target="wandb/local-result",
            publisher=lambda *_args: (_ for _ in ()).throw(
                AssertionError("publisher must not run")
            ),
        )


def test_publication_rejects_result_mutation_and_writes_no_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = _canonical_manifest(tmp_path)
    result_path, result = _result_fixture(tmp_path, manifest)
    _patch_v3_reader(monkeypatch, result)
    target = _target()

    def publisher(*_args: Any) -> WeavePublicationOutcomeV1:
        result_path.write_text('{"mutated":true}\n', encoding="utf-8")
        return _outcome(manifest, target)

    with pytest.raises(LocalResultPublicationError, match="mutated the canonical"):
        publish_local_result_to_weave(
            result_path,
            manifest_path,
            target=target,
            publisher=publisher,
        )
    assert not result_path.with_name("weave-publication-receipt.json").exists()


@pytest.mark.parametrize("failure", ["wrong_project", "secret"])
def test_publication_rejects_wrong_project_refs_and_secret_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    manifest_path, manifest = _canonical_manifest(tmp_path)
    result_path, result = _result_fixture(tmp_path, manifest)
    _patch_v3_reader(monkeypatch, result)
    target = _target()
    secret = "super-secret-value-123"

    def publisher(*_args: Any) -> dict[str, Any]:
        raw = _outcome(
            manifest,
            target,
            publisher_revision=(secret if failure == "secret" else "revision-v1"),
        ).to_dict()
        if failure == "wrong_project":
            raw["objects"][0]["target"]["project"] = "other-project"
            raw["objects"][0]["ref"] = raw["objects"][0]["ref"].replace(
                "/local-result/", "/other-project/"
            )
        return raw

    expected = "destination disagree" if failure == "wrong_project" else "secret"
    with pytest.raises(ValueError, match=expected):
        publish_local_result_to_weave(
            result_path,
            manifest_path,
            target=target,
            publisher=publisher,
            secret_values=(secret,),
        )


def test_publication_rejects_conflicting_existing_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = _canonical_manifest(tmp_path)
    result_path, result = _result_fixture(tmp_path, manifest)
    _patch_v3_reader(monkeypatch, result)
    target = _target()

    receipt = publish_local_result_to_weave(
        result_path,
        manifest_path,
        target=target,
        publisher=lambda *_args: _outcome(manifest, target),
        clock=lambda: datetime(2026, 8, 17, 12, tzinfo=UTC),
    )
    changed_digest = "f" * 64
    conflicting = replace(
        receipt,
        publication_id=stable_digest(
            {
                "schema_version": 1,
                "target": target.to_dict(),
                "result_digest": changed_digest,
                "local_manifest_digest": manifest.manifest_digest,
            }
        ),
        result_digest=changed_digest,
        receipt_digest="",
    )
    receipt_path = result_path.with_name("weave-publication-receipt.json")
    atomic_write_json(receipt_path, conflicting.to_dict())

    with pytest.raises(LocalResultPublicationError, match="conflicting immutable"):
        publish_local_result_to_weave(
            result_path,
            manifest_path,
            target=target,
            publisher=lambda *_args: _outcome(manifest, target),
        )


def test_actual_v3_reader_rejects_tampered_result_before_publication(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _canonical_manifest(tmp_path)
    result_path, _result = _result_fixture(tmp_path, manifest)
    called = False

    def publisher(*_args: Any) -> WeavePublicationOutcomeV1:
        nonlocal called
        called = True
        raise AssertionError("publisher must not run")

    with pytest.raises(ValueError, match="comparison result"):
        publish_local_result_to_weave(
            result_path,
            manifest_path,
            target="wandb/local-result",
            publisher=publisher,
        )
    assert called is False


def test_publication_receipt_rejects_digest_tampering(tmp_path: Path) -> None:
    target = WeavePublicationTargetV1(entity="wandb", project="local-result")
    attempt = "a" * 64
    outcome = _outcome(
        SimpleNamespace(planned_attempt_ids=(attempt,)),
        target,
    )
    result_digest = "b" * 64
    manifest_digest = "c" * 64
    receipt = WeavePublicationReceiptV1(
        publication_id=stable_digest(
            {
                "schema_version": 1,
                "target": target.to_dict(),
                "result_digest": result_digest,
                "local_manifest_digest": manifest_digest,
            }
        ),
        target=target,
        result_digest=result_digest,
        qualification_digest="d" * 64,
        result_file_sha256="e" * 64,
        local_manifest_digest=manifest_digest,
        local_manifest_file_sha256="f" * 64,
        hosted_objects=outcome.objects,
        publisher_id=outcome.publisher_id,
        publisher_revision=outcome.publisher_revision,
        status="published",
        published_at="2026-08-17T12:00:00+00:00",
    )
    raw = receipt.to_dict()
    raw["publisher_revision"] = "forged"
    path = tmp_path / "receipt.json"
    atomic_write_json(path, raw)

    with pytest.raises(ValueError, match="receipt digest does not match"):
        read_weave_publication_receipt(path)


def test_legacy_project_only_receipt_is_readable_but_cannot_be_republished(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_target = WeavePublicationTargetV1(
        entity="wandb",
        project="local-result",
    )
    attempt = "a" * 64
    outcome = _outcome(
        SimpleNamespace(planned_attempt_ids=(attempt,)),
        legacy_target,
    )
    result_digest = "b" * 64
    manifest_digest = "c" * 64
    receipt = WeavePublicationReceiptV1(
        publication_id=stable_digest(
            {
                "schema_version": 1,
                "target": legacy_target.to_dict(),
                "result_digest": result_digest,
                "local_manifest_digest": manifest_digest,
            }
        ),
        target=legacy_target,
        result_digest=result_digest,
        qualification_digest="d" * 64,
        result_file_sha256="e" * 64,
        local_manifest_digest=manifest_digest,
        local_manifest_file_sha256="f" * 64,
        hosted_objects=outcome.objects,
        publisher_id=outcome.publisher_id,
        publisher_revision=outcome.publisher_revision,
        status="published",
        published_at="2026-08-17T12:00:00+00:00",
    )

    assert weave_publication_receipt_from_dict(receipt.to_dict()) == receipt
    assert receipt.target.destination is None

    manifest_path, manifest = _canonical_manifest(tmp_path)
    result_path, result = _result_fixture(tmp_path, manifest)
    _patch_v3_reader(monkeypatch, result)
    called = False

    def publisher(*_args: Any) -> WeavePublicationOutcomeV1:
        nonlocal called
        called = True
        raise AssertionError("legacy target must be rejected before publication")

    with pytest.raises(
        LocalResultPublicationError,
        match="endpoint-bound evidence destination",
    ):
        publish_local_result_to_weave(
            result_path,
            manifest_path,
            target=legacy_target,
            publisher=publisher,
        )
    assert called is False


def test_endpoint_change_changes_publication_identity_and_conflicts_with_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = _canonical_manifest(tmp_path)
    result_path, result = _result_fixture(tmp_path, manifest)
    _patch_v3_reader(monkeypatch, result)
    target_a = _target(
        api_base_url="https://api-a.example.test",
        trace_base_url="https://trace-a.example.test",
        app_base_url="https://app-a.example.test",
    )
    target_b = _target(
        api_base_url="https://api-b.example.test",
        trace_base_url="https://trace-b.example.test",
        app_base_url="https://app-b.example.test",
    )
    first_path = tmp_path / "receipt-a.json"
    second_path = tmp_path / "receipt-b.json"
    receipt_a = publish_local_result_to_weave(
        result_path,
        manifest_path,
        target=target_a,
        publisher=lambda *_args: _outcome(manifest, target_a),
        receipt_path=first_path,
    )
    receipt_b = publish_local_result_to_weave(
        result_path,
        manifest_path,
        target=target_b,
        publisher=lambda *_args: _outcome(manifest, target_b),
        receipt_path=second_path,
    )

    assert target_a.destination is not None
    assert target_b.destination is not None
    assert (
        target_a.destination.destination_digest
        != target_b.destination.destination_digest
    )
    assert receipt_a.publication_id != receipt_b.publication_id

    with pytest.raises(LocalResultPublicationError, match="conflicting immutable"):
        publish_local_result_to_weave(
            result_path,
            manifest_path,
            target=target_b,
            publisher=lambda *_args: _outcome(manifest, target_b),
            receipt_path=first_path,
        )


def test_study_scope_is_bound_to_publication_identity_and_weave_attributes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _manifest_path, manifest = _canonical_manifest(tmp_path)
    _result_path, result = _result_fixture(tmp_path, manifest)
    scope = StudyPublicationScopeV1(
        research_id="fugue-standalone-lab-v1",
        study_id="prompt-change-live-v1",
    )
    target = WeavePublicationTargetV1(
        entity="wandb",
        project="fugue-experiments",
        study_scope=scope,
        destination=_target(project="fugue-experiments").destination,
    )
    client = _FakeWeaveClient(target)
    _install_fake_weave(monkeypatch, target, client)

    publisher = weave_publisher_from_environment({"WANDB_API_KEY": "test-only-key"})
    outcome = publisher(result, manifest, target)

    assert outcome.target.study_scope == scope
    for call in client.calls.values():
        assert call.attributes["fugue.research_id"] == scope.research_id
        assert call.attributes["fugue.study_key"] == scope.study_id
        assert "wandb.research_id" not in call.attributes
        assert "wandb.study_id" not in call.attributes
        assert call.attributes["fugue.comparison_id"] == result.comparison_id
    dataset = next(iter(client.objects.values()))
    assert dataset["fugue_research_id"] == scope.research_id
    assert dataset["fugue_study_key"] == scope.study_id
    assert "wandb_research_id" not in dataset
    assert "wandb_study_id" not in dataset


def test_target_preserves_legacy_positional_schema_version() -> None:
    target = WeavePublicationTargetV1("wandb", "fugue-experiments", 1)

    assert target.schema_version == 1
    assert target.study_scope is None
    assert target.destination is None
    assert target.to_dict() == {
        "entity": "wandb",
        "project": "fugue-experiments",
        "schema_version": 1,
    }

    with pytest.raises(ValueError, match="Study scope"):
        WeavePublicationTargetV1(
            "wandb",
            "fugue-experiments",
            1,
            1,  # type: ignore[arg-type]
        )


def test_study_scope_change_conflicts_with_existing_publication_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = _canonical_manifest(tmp_path)
    result_path, result = _result_fixture(tmp_path, manifest)
    _patch_v3_reader(monkeypatch, result)
    first_scope = StudyPublicationScopeV1(
        research_id="fugue-standalone-lab-v1",
        study_id="prompt-change-live-v1",
    )
    first_target = WeavePublicationTargetV1(
        entity="wandb",
        project="fugue-experiments",
        study_scope=first_scope,
        destination=_target(project="fugue-experiments").destination,
    )
    publish_local_result_to_weave(
        result_path,
        manifest_path,
        target=first_target,
        publisher=lambda *_args: _outcome(manifest, first_target),
    )
    changed_scope = StudyPublicationScopeV1(
        research_id="another-research",
        study_id="prompt-change-live-v1",
    )
    changed_target = WeavePublicationTargetV1(
        entity="wandb",
        project="fugue-experiments",
        study_scope=changed_scope,
        destination=_target(project="fugue-experiments").destination,
    )

    with pytest.raises(LocalResultPublicationError, match="conflicting immutable"):
        publish_local_result_to_weave(
            result_path,
            manifest_path,
            target=changed_target,
            publisher=lambda *_args: _outcome(manifest, changed_target),
        )


def test_weave_publisher_is_lazy_and_has_actionable_missing_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_name: str):
        raise ModuleNotFoundError("No module named 'weave'")

    monkeypatch.setattr(publication.importlib, "import_module", missing)

    with pytest.raises(MissingWeaveExtraError, match=r"fugue\[weave\]"):
        weave_publisher_from_environment({})


def test_weave_publisher_rejects_environment_endpoint_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _manifest_path, manifest = _canonical_manifest(tmp_path)
    _result_path, result = _result_fixture(tmp_path, manifest)
    target = _target(
        api_base_url="https://api-a.example.test",
        trace_base_url="https://trace-a.example.test",
        app_base_url="https://app-a.example.test",
    )
    client = _FakeWeaveClient(target)
    _install_fake_weave(monkeypatch, target, client)
    publisher = weave_publisher_from_environment(
        {
            "WANDB_API_KEY": "test-only-key",
            "FUGUE_WEAVE_BASE_URL": "https://api-b.example.test",
            "FUGUE_WEAVE_TRACE_SERVER_URL": "https://trace-b.example.test",
            "WANDB_APP_BASE_URL": "https://app-b.example.test",
        }
    )

    with pytest.raises(
        LocalResultPublicationError,
        match="environment disagrees",
    ):
        publisher(result, manifest, target)
    assert client.calls == {}


class _FakeWeaveClient:
    def __init__(self, target: WeavePublicationTargetV1) -> None:
        self.project_id = target.project_slug
        self.target = target
        self.calls: dict[str, SimpleNamespace] = {}
        self.objects: dict[str, dict[str, Any]] = {}
        self.created_call_ids: list[str] = []
        self.finished_call_ids: list[str] = []
        self.published_object_refs: list[str] = []
        self.fail_after_finish: int | None = None
        self.fail_after_create: int | None = None
        self.duplicate_after_create_kind: str | None = None
        self.tamper_call_kind: str | None = None
        self.tamper_call_field = "output"
        self.tamper_dataset = False
        self.call_queries: list[dict[str, Any]] = []
        self.flush_count = 0

    def create_call(
        self,
        _op,
        inputs,
        parent=None,
        attributes=None,
        display_name=None,
        *,
        use_stack=True,
    ):
        del use_stack
        call_id = f"call-{len(self.created_call_ids) + 1:04d}"
        trace_id = getattr(parent, "trace_id", None) or (
            f"trace-{len(self.created_call_ids) + 1:04d}"
        )
        call = SimpleNamespace(
            id=call_id,
            parent_id=getattr(parent, "id", None),
            project_id=self.target.project_slug,
            trace_id=trace_id,
            inputs=dict(inputs),
            attributes=dict(attributes or {}),
            display_name=display_name,
            output=None,
            ended_at=None,
            exception=None,
        )
        self.calls[call.id] = call
        self.created_call_ids.append(call.id)
        if self.duplicate_after_create_kind == call.attributes.get(
            "fugue.evidence.object_kind"
        ):
            duplicate_id = f"duplicate-{call.id}"
            self.calls[duplicate_id] = SimpleNamespace(
                **{
                    **vars(call),
                    "id": duplicate_id,
                    "ended_at": "2026-08-17T12:00:00+00:00",
                }
            )
            self.duplicate_after_create_kind = None
        if self.fail_after_create == len(self.created_call_ids):
            self.fail_after_create = None
            raise RuntimeError("simulated controller interruption during Call creation")
        return call

    def finish_call(self, call, output=None):
        call.output = dict(output or {})
        call.ended_at = "2026-08-17T12:00:00+00:00"
        self.finished_call_ids.append(call.id)
        if self.fail_after_finish == len(self.finished_call_ids):
            self.fail_after_finish = None
            raise RuntimeError("simulated controller interruption")

    def flush(self) -> None:
        self.flush_count += 1
        if any(call.ended_at is None for call in self.calls.values()):
            raise AssertionError("flush called while a Weave Call is open")

    def get_call(self, call_id):
        if call_id not in self.calls:
            raise ValueError(f"Call not found: {call_id}")
        call = self.calls[call_id]
        if (
            self.tamper_call_kind is not None
            and call.ended_at is not None
            and call.attributes.get("fugue.evidence.object_kind")
            == self.tamper_call_kind
        ):
            overrides: dict[str, Any]
            if self.tamper_call_field == "parent":
                overrides = {"parent_id": "forged-parent"}
            elif self.tamper_call_field == "project":
                overrides = {"project_id": "wandb/another-project"}
            elif self.tamper_call_field == "identity":
                overrides = {
                    "attributes": {
                        **call.attributes,
                        "fugue.result_digest": "f" * 64,
                    }
                }
            else:
                overrides = {"output": {"forged": True}}
            return SimpleNamespace(
                **{
                    **vars(call),
                    **overrides,
                }
            )
        return call

    def get_calls(self, *, query=None, limit=None, **_kwargs):
        assert isinstance(query, dict)
        self.call_queries.append(query)
        conditions = query["$expr"]["$and"]

        def field_value(call: Any, path: str) -> Any:
            # Match Weave 0.53.6 dynamic-field semantics: unescaped dots
            # traverse JSON objects, while ``\.`` addresses a literal dot in
            # one key.  Fugue deliberately publishes flat namespaced keys.
            parts = [
                part.replace(r"\.", ".")
                for part in re.split(r"(?<!\\)\.", path)
            ]
            value: Any = {"attributes": call.attributes}
            for part in parts:
                if not isinstance(value, dict) or part not in value:
                    return None
                value = value[part]
            return value

        matches = [
            call
            for call in self.calls.values()
            if all(
                field_value(call, field["$getField"]) == literal["$literal"]
                for field, literal in (
                    condition["$eq"] for condition in conditions
                )
            )
        ]
        return matches[:limit]

    def get(self, ref, *, objectify=True):
        del objectify
        value = dict(self.objects[ref.uri])
        if self.tamper_dataset:
            value["attempt_id"] = "f" * 64
        return value

    def publish(self, value, *, name):
        digest = stable_digest(value)
        uri = f"weave:///{self.target.project_slug}/object/{name}:{digest}"
        self.objects[uri] = dict(value)
        self.published_object_refs.append(uri)
        return SimpleNamespace(uri=uri)


class _ObjectifiedWeaveRef:
    def __init__(self, uri: str, payload: dict[str, Any]) -> None:
        self.uri = uri
        self.payload = payload

    def model_dump(self) -> dict[str, Any]:
        return dict(self.payload)


class _ObjectifyingWeaveDict(dict[str, Any]):
    """Model Weave readback without losing the stored reference identity."""

    def items(self):
        for key, value in dict.items(self):
            if isinstance(value, _ObjectifiedWeaveRef):
                yield key, value.payload
            else:
                yield key, value


def test_weave_call_input_comparison_uses_raw_immutable_ref_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = publication.importlib.import_module

    def import_module(name: str):
        if name == "weave.trace.refs":
            return SimpleNamespace(Ref=_ObjectifiedWeaveRef)
        if name == "weave.trace.vals":
            return SimpleNamespace(WeaveDict=_ObjectifyingWeaveDict)
        return original_import(name)

    monkeypatch.setattr(publication.importlib, "import_module", import_module)
    uri = "weave:///wandb/local-result/object/frozen-dataset:sha256"
    observed = _ObjectifyingWeaveDict(
        {"dataset_ref": _ObjectifiedWeaveRef(uri, {"attempt_id": "a" * 64})}
    )

    assert publication._canonical_values_equal(
        observed,
        {"dataset_ref": uri},
    )
    assert not publication._canonical_values_equal(
        observed,
        {"dataset_ref": ("weave:///wandb/local-result/object/other-dataset:sha256")},
    )


def _install_fake_weave(
    monkeypatch: pytest.MonkeyPatch,
    target: WeavePublicationTargetV1,
    client: _FakeWeaveClient,
):
    from fugue import weave_support

    def parse_ref(uri: str):
        prefix = "weave:///"
        assert uri.startswith(prefix)
        entity, project, _kind, _object_id = uri.removeprefix(prefix).split("/", 3)
        return SimpleNamespace(entity=entity, project=project, uri=uri)

    fake_weave = SimpleNamespace(
        __version__="test-sdk",
        get_client=lambda: client,
        publish=client.publish,
        ref=parse_ref,
    )
    monkeypatch.setattr(
        publication.importlib,
        "import_module",
        lambda name: fake_weave if name == "weave" else None,
    )

    @contextmanager
    def destination_session(_project, _env):
        yield fake_weave

    monkeypatch.setattr(
        weave_support,
        "weave_destination_session",
        destination_session,
    )
    return fake_weave


def test_real_weave_adapter_emits_nested_five_object_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _manifest_path, manifest = _canonical_manifest(tmp_path)
    _result_path, result = _result_fixture(tmp_path, manifest)
    target = _target()
    client = _FakeWeaveClient(target)
    _install_fake_weave(monkeypatch, target, client)

    publisher = weave_publisher_from_environment({"WANDB_API_KEY": "test-only-key"})
    outcome = publisher(result, manifest, target)

    assert isinstance(outcome, WeavePublicationOutcomeV1)
    assert len(outcome.objects) == 5
    assert len(client.finished_call_ids) == 4
    assert len(set(client.created_call_ids)) == 4
    assert client.flush_count == 1
    assert all(call.ended_at is not None for call in client.calls.values())
    agent_call = next(
        call
        for call in client.calls.values()
        if call.attributes["fugue.evidence.object_kind"] == "agent_evidence_receipt"
    )
    assert agent_call.output["native_agent_call"] is False
    by_kind = {item.kind: item for item in outcome.objects}
    assert by_kind["agent_evidence_receipt"].native_agent_call is False
    assert by_kind["dataset"].ref.startswith(f"weave:///{target.project_slug}/object/")
    assert outcome.publisher_revision == (
        "v3-public-query-readback+weave-test-sdk"
    )
    root = next(
        call
        for call in client.calls.values()
        if call.attributes["fugue.evidence.object_kind"] == "evaluation_root"
    )
    assert root.trace_id != root.id
    assert all(call.trace_id == root.trace_id for call in client.calls.values())
    expected_query_fields = {
        r"attributes.fugue\.publication_id",
        r"attributes.fugue\.attempt_id",
        r"attributes.fugue\.evidence\.object_kind",
    }
    assert client.call_queries
    assert all(
        {
            condition["$eq"][0]["$getField"]
            for condition in query["$expr"]["$and"]
        }
        == expected_query_fields
        for query in client.call_queries
    )


def test_real_weave_adapter_publishes_human_task_and_result_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _manifest_path, manifest = _canonical_manifest(tmp_path)
    _result_path, result = _result_fixture(tmp_path, manifest)
    pair = result.paired_cases[0]
    attempt = pair.baseline
    attempt.execution_status = "completed"
    attempt.evaluation_status = "scored"
    presentation = TaskPresentationV1(
        task_id="task-1",
        title="Create a platform-bound Skill",
        public_prompt=(
            PublicPromptPartV1(
                order=1,
                text="Create a Skill package for the declared platform.",
            ),
        ),
        required_output="Return a valid packaged Skill.",
        public_acceptance_criteria=(
            "The package preserves the required compatibility metadata.",
        ),
        tags=("skill-package",),
        partition="development",
    )
    attempt.task_presentation = presentation
    attempt.arm_label = "Baseline a5bcdd7"
    attempt.treatment_summary = "Exact upstream Skill Creator at a5bcdd7."
    attempt.passed = False
    attempt.scores = {
        "package_contract_valid": False,
        "assigned_skill_opened": True,
    }
    attempt.score_details = {
        "package_contract_valid": {
            "what": "The package must satisfy the public package contract.",
            "observed": "The required compatibility metadata was absent.",
            "why": "A consumer cannot select the Skill safely.",
            "evidence": "scorer:package-contract-v1",
        }
    }
    attempt.judge_reviews = {
        "usefulness": {
            "label": "adequate",
            "reason": "The instructions are readable, but compatibility is missing.",
            "missing_evidence": False,
        }
    }
    attempt.task_result = {
        "schema_version": 1,
        "task_passed": False,
        "outcome_summary": "The package omitted required compatibility metadata.",
        "failed_required_checks": [
            {
                "id": "package_contract_valid",
                "label": "Package contract valid",
                "explanation": "The required compatibility metadata was absent.",
                "critical": True,
            }
        ],
        "answer_digest": None,
        "agent_execution_status": "completed",
        "evidence_integrity_status": "verified",
    }
    pair.dimension_changes = (
        SimpleNamespace(id="package_contract_valid", role="safety_gate"),
        SimpleNamespace(id="assigned_skill_opened", role="mechanism"),
    )
    target = _target()
    client = _FakeWeaveClient(target)
    _install_fake_weave(monkeypatch, target, client)

    publisher = weave_publisher_from_environment({"WANDB_API_KEY": "test-only-key"})
    publisher(result, manifest, target)

    assert len(client.published_object_refs) == 1
    dataset = next(iter(client.objects.values()))
    assert dataset["rows"][0]["task_presentation"] == presentation.to_dict()
    assert (
        dataset["rows"][0]["row_payload_digest"] != presentation.task_definition_digest
    )
    replay_row = dict(dataset["rows"][0])
    payload_digest = replay_row.pop("row_payload_digest")
    assert payload_digest == publication._weave_row_digest(replay_row)
    by_kind = {
        call.attributes["fugue.evidence.object_kind"]: call
        for call in client.calls.values()
    }
    scored = by_kind["prediction_and_score"]
    assert scored.display_name.startswith(
        "Published scored attempt: DID NOT PASS · Create a platform-bound Skill"
    )
    assert scored.output["task_result"] == attempt.task_result
    assert scored.output["scores"] == {
        "task_passed": False,
        "safety__package_contract_valid": False,
        "mechanism__assigned_skill_opened": True,
    }
    assert scored.output["score_details"] == attempt.score_details
    assert scored.output["judge_evidence"] == {
        "status": "available",
        "advisory": True,
        "reviews": attempt.judge_reviews,
    }
    agent = by_kind["agent_evidence_receipt"]
    assert agent.output["agent_execution_status"] == "completed"
    assert agent.output["native_agent_call"] is False


def test_real_weave_adapter_does_not_recreate_pre_agent_task_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _manifest_path, manifest = _canonical_manifest(tmp_path)
    _result_path, result = _result_fixture(tmp_path, manifest)
    attempt = result.paired_cases[0].baseline
    attempt.execution_status = "not_started"
    attempt.evaluation_status = "unavailable"
    # Reproduce a malformed downstream replay. Publication must trust the
    # typed runtime outcome and withhold every stale behavioral claim.
    attempt.passed = True
    attempt.scores = {"answer_correct": True}
    attempt.score_details = {
        "answer_correct": {
            "what": "The answer must be correct.",
            "observed": "A stale scorer said it passed.",
            "why": "This must not survive a pre-Agent terminal.",
        }
    }
    attempt.score_explanations = {"answer_correct": "stale pass"}
    attempt.judge_reviews = {
        "usefulness": {
            "label": "strong",
            "reason": "stale review",
            "missing_evidence": False,
        }
    }
    attempt.task_result = {
        "schema_version": 1,
        "task_passed": True,
        "outcome_summary": "A stale task verdict.",
        "failed_required_checks": [],
        "answer_digest": None,
        "agent_execution_status": "completed",
        "evidence_integrity_status": "verified",
    }
    target = _target()
    client = _FakeWeaveClient(target)
    _install_fake_weave(monkeypatch, target, client)

    publisher = weave_publisher_from_environment({"WANDB_API_KEY": "test-only-key"})
    publisher(result, manifest, target)

    by_kind = {
        call.attributes["fugue.evidence.object_kind"]: call
        for call in client.calls.values()
    }
    scored = by_kind["prediction_and_score"]
    prediction = by_kind["prediction"]
    agent = by_kind["agent_evidence_receipt"]
    assert scored.display_name.startswith("Published scored attempt: UNSCORED")
    assert "fugue.task_verdict" not in scored.attributes
    assert "task_result" not in scored.output
    assert "task_result" not in prediction.output
    assert scored.output["scores"] == {}
    assert scored.output["score_details"] == {}
    assert scored.output["judge_evidence"]["status"] == "not_applicable"
    assert prediction.output["score_explanations"] == {}
    assert prediction.output["sanitized_answer_excerpt"] is None
    assert "Agent execution (no scored answer)" in prediction.display_name
    assert agent.output["agent_execution_status"] == "not_started"


@pytest.mark.parametrize(
    "tamper",
    ["dataset", "prediction_output", "parent", "project", "identity"],
)
def test_real_weave_adapter_requires_authoritative_matching_readback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    _manifest_path, manifest = _canonical_manifest(tmp_path)
    _result_path, result = _result_fixture(tmp_path, manifest)
    target = _target()
    client = _FakeWeaveClient(target)
    client.tamper_dataset = tamper == "dataset"
    client.tamper_call_kind = "prediction" if tamper != "dataset" else None
    client.tamper_call_field = tamper.removeprefix("prediction_")
    _install_fake_weave(monkeypatch, target, client)

    publisher = weave_publisher_from_environment({"WANDB_API_KEY": "test-only-key"})
    with pytest.raises(
        RuntimeError,
        match=(
            "readback.*disagrees|conflicting Call|another project|"
            "lost its evidence parent|attributes disagree"
        ),
    ):
        publisher(result, manifest, target)


def test_real_weave_adapter_resumes_fully_terminal_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _manifest_path, manifest = _canonical_manifest(tmp_path)
    _result_path, result = _result_fixture(tmp_path, manifest)
    target = _target()
    client = _FakeWeaveClient(target)
    client.fail_after_finish = 4
    _install_fake_weave(monkeypatch, target, client)
    publisher = weave_publisher_from_environment({"WANDB_API_KEY": "test-only-key"})

    with pytest.raises(RuntimeError, match="simulated controller interruption"):
        publisher(result, manifest, target)
    created_before_retry = tuple(client.created_call_ids)
    finished_before_retry = tuple(client.finished_call_ids)
    assert len(created_before_retry) == 4
    assert len(finished_before_retry) == 4
    assert client.flush_count == 0

    outcome = publisher(result, manifest, target)

    assert tuple(client.created_call_ids) == created_before_retry
    assert tuple(client.finished_call_ids) == finished_before_retry
    assert client.flush_count == 1
    assert len(set(client.published_object_refs)) == 1
    assert len(outcome.objects) == 5
    assert len({(item.attempt_id, item.kind) for item in outcome.objects}) == 5


def test_real_weave_adapter_retries_dataset_only_after_unflushed_call_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _manifest_path, manifest = _canonical_manifest(tmp_path)
    _result_path, result = _result_fixture(tmp_path, manifest)
    target = _target()
    client = _FakeWeaveClient(target)
    client.fail_after_create = 2
    _install_fake_weave(monkeypatch, target, client)
    publisher = weave_publisher_from_environment({"WANDB_API_KEY": "test-only-key"})

    with pytest.raises(
        RuntimeError,
        match="simulated controller interruption during Call creation",
    ):
        publisher(result, manifest, target)
    assert tuple(client.created_call_ids) == ("call-0001", "call-0002")
    # Model process loss under calls-complete: unmatched local Starts never
    # became authoritative Calls, while the content-addressed Dataset remains.
    client.calls.clear()

    outcome = publisher(result, manifest, target)

    assert tuple(client.created_call_ids) == (
        "call-0001",
        "call-0002",
        "call-0003",
        "call-0004",
        "call-0005",
        "call-0006",
    )
    assert len(set(client.published_object_refs)) == 1
    assert len(outcome.objects) == 5
    assert len({(item.attempt_id, item.kind) for item in outcome.objects}) == 5


def test_real_weave_adapter_rejects_unfinished_authoritative_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _manifest_path, manifest = _canonical_manifest(tmp_path)
    _result_path, result = _result_fixture(tmp_path, manifest)
    target = _target()
    client = _FakeWeaveClient(target)
    _install_fake_weave(monkeypatch, target, client)
    publisher = weave_publisher_from_environment({"WANDB_API_KEY": "test-only-key"})
    publisher(result, manifest, target)
    prediction = next(
        call
        for call in client.calls.values()
        if call.attributes["fugue.evidence.object_kind"] == "prediction"
    )
    prediction.output = None
    prediction.ended_at = None
    created_before_retry = tuple(client.created_call_ids)
    finished_before_retry = tuple(client.finished_call_ids)
    flushes_before_retry = client.flush_count

    with pytest.raises(RuntimeError, match="unfinished remote Call"):
        publisher(result, manifest, target)

    assert tuple(client.created_call_ids) == created_before_retry
    assert tuple(client.finished_call_ids) == finished_before_retry
    assert client.flush_count == flushes_before_retry


def test_real_weave_adapter_fails_closed_on_duplicate_call_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _manifest_path, manifest = _canonical_manifest(tmp_path)
    _result_path, result = _result_fixture(tmp_path, manifest)
    target = _target()
    client = _FakeWeaveClient(target)
    client.duplicate_after_create_kind = "prediction"
    _install_fake_weave(monkeypatch, target, client)
    publisher = weave_publisher_from_environment({"WANDB_API_KEY": "test-only-key"})

    with pytest.raises(RuntimeError, match="duplicate Call race"):
        publisher(result, manifest, target)

    predictions = [
        call
        for call in client.calls.values()
        if call.attributes["fugue.evidence.object_kind"] == "prediction"
    ]
    assert len(predictions) == 2
    assert all(call.ended_at is not None for call in predictions)


def test_real_weave_adapter_rejects_conflicting_existing_call_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _manifest_path, manifest = _canonical_manifest(tmp_path)
    _result_path, result = _result_fixture(tmp_path, manifest)
    target = _target()
    client = _FakeWeaveClient(target)
    _install_fake_weave(monkeypatch, target, client)
    publisher = weave_publisher_from_environment({"WANDB_API_KEY": "test-only-key"})
    publisher(result, manifest, target)
    prediction = next(
        call
        for call in client.calls.values()
        if call.attributes["fugue.evidence.object_kind"] == "prediction"
    )
    prediction.attributes["fugue.evidence.destination_digest"] = "f" * 64
    created_before_retry = tuple(client.created_call_ids)

    with pytest.raises(RuntimeError, match="attributes disagree"):
        publisher(result, manifest, target)

    assert tuple(client.created_call_ids) == created_before_retry
