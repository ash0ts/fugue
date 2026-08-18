from __future__ import annotations

import ast
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from fugue.bench.candidates import attempt_id, attempt_identity, stable_digest
from fugue.bench.export import PredictionRowV1
from fugue.bench.local_evidence import (
    AgentEvidenceReceiptV1,
    LocalArtifactRefV1,
    LocalAttemptPlanV1,
    LocalEvidenceCoordinator,
    LocalEvidenceIntegrityError,
    LocalEvidenceStore,
    build_local_evidence_run_plan,
    local_evidence_run_plan_from_dict,
)


def _attempt(
    *,
    run_id: str,
    task_id: str = "task-1",
    arm: str = "baseline",
    index: int = 1,
) -> LocalAttemptPlanV1:
    candidate_id = stable_digest({"candidate": arm})
    runtime_id = stable_digest({"runtime": "harbor-local-v1"})
    identity = attempt_identity(
        task_id=task_id,
        arm=arm,
        harness="claude-code",
        attempt=index,
        candidate=candidate_id,
        runtime=runtime_id,
    )
    canonical_id = attempt_id(**identity)
    return LocalAttemptPlanV1(
        run_id=run_id,
        cell_id=f"{task_id}-{arm}-{index}",
        attempt_id=canonical_id,
        attempt_identity=identity,
        prediction_id=stable_digest(
            {"record_type": "prediction", "attempt_id": canonical_id}
        ),
        evaluation_scope_id=stable_digest(
            {"evaluation": run_id, "candidate": candidate_id}
        ),
        dataset_id=stable_digest({"dataset": run_id}),
    )


def _coordinator(
    repo_root: Path,
    *,
    run_id: str = "local-evidence-run-v1",
    attempts: tuple[LocalAttemptPlanV1, ...] | None = None,
    secret_values: tuple[str, ...] = (),
    require_run_conformance: bool = False,
) -> tuple[LocalEvidenceCoordinator, tuple[LocalAttemptPlanV1, ...]]:
    planned = attempts or (_attempt(run_id=run_id),)
    plan = build_local_evidence_run_plan(
        run_id=run_id,
        run_snapshot_sha256=stable_digest({"snapshot": run_id}),
        evaluation_asset_lock_sha256=stable_digest({"assets": run_id}),
        attempts=planned,
        require_run_conformance=require_run_conformance,
    )
    store = LocalEvidenceStore(repo_root, run_id)
    return (
        LocalEvidenceCoordinator(store, plan, secret_values=secret_values),
        plan.attempts,
    )


def _agent_receipt(
    repo_root: Path,
    attempt: LocalAttemptPlanV1,
) -> AgentEvidenceReceiptV1:
    relative = Path(
        ".fugue",
        "runtime",
        attempt.run_id,
        "agent-native",
        attempt.attempt_id,
        "transcript.jsonl",
    )
    target = repo_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(
        {
            "attempt_id": attempt.attempt_id,
            "session_id": f"session-{attempt.attempt_id[:12]}",
            "message": "bounded public answer",
        },
        sort_keys=True,
    )
    target.write_text(body + "\n", encoding="utf-8")
    raw = target.read_bytes()
    artifact = LocalArtifactRefV1(
        path=relative.as_posix(),
        sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
        media_type="application/x-ndjson",
    )
    return AgentEvidenceReceiptV1(
        attempt_id=attempt.attempt_id,
        planned_conversation_id=f"planned-{attempt.attempt_id[:12]}",
        primary_session_id=f"session-{attempt.attempt_id[:12]}",
        child_session_ids=(),
        artifacts=(artifact,),
        transcript_artifact=artifact,
        transcript_session_id=f"session-{attempt.attempt_id[:12]}",
        correlation_verified=True,
        tool_event_count=0,
        tool_events_sha256=hashlib.sha256(b"[]").hexdigest(),
        response_sha256=stable_digest({"answer": "bounded public answer"}),
    )


def _prediction(attempt: LocalAttemptPlanV1) -> dict[str, Any]:
    return PredictionRowV1(
        prediction_id=attempt.prediction_id,
        run_id=attempt.run_id,
        candidate_id=attempt.candidate_id,
        comparison_example_id=stable_digest(
            {"example": attempt.attempt_identity["task_id"]}
        ),
        trial_index=int(attempt.attempt_identity["attempt"]),
        execution_kind="agent",
        source_record_type="trial",
        payload={
            "attempt_id": attempt.attempt_id,
            "attempt_identity": attempt.attempt_identity,
            "answer": "bounded public answer",
        },
    ).to_dict()


def _finish(
    coordinator: LocalEvidenceCoordinator,
    repo_root: Path,
    attempt: LocalAttemptPlanV1,
):
    coordinator.begin_attempt(attempt.attempt_id)
    return coordinator.finish_attempt(
        attempt.attempt_id,
        terminal_status="passed",
        prediction_row=_prediction(attempt),
        scores={"answer_correct": 1, "evidence_honest": 1},
        agent_receipt=_agent_receipt(repo_root, attempt),
        attempt_payload={
            "receipts": {
                name: {"status": "passed"}
                for name in ("privacy", "policy", "usage", "cleanup")
            }
        },
        recorded_at="2026-08-17T12:00:00+00:00",
    )


def _write_run_conformance(
    coordinator: LocalEvidenceCoordinator,
    *,
    status: str = "passed",
    enforced: bool = True,
) -> Path:
    payload = {
        "schema_version": 2,
        "run_id": coordinator.plan.run_id,
        "backend": "local_harbor_docker",
        "status": status,
        "enforced": enforced,
        "receipt_sha256": "",
    }
    payload["receipt_sha256"] = stable_digest(payload)
    path = coordinator.store.run_conformance_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_local_evidence_round_trip_preserves_canonical_identities(
    tmp_path: Path,
) -> None:
    coordinator, (attempt,) = _coordinator(tmp_path)

    overlay = coordinator.begin_attempt(attempt.attempt_id)
    assert overlay["FUGUE_ATTEMPT_ID"] == attempt.attempt_id
    assert overlay["FUGUE_LOCAL_PREDICTION_ID"] == attempt.prediction_id

    record = coordinator.finish_attempt(
        attempt.attempt_id,
        terminal_status="passed",
        prediction_row=_prediction(attempt),
        scores={"answer_correct": 1, "evidence_honest": 1},
        agent_receipt=_agent_receipt(tmp_path, attempt),
        attempt_payload={
            "receipts": {
                name: {"status": "passed"}
                for name in ("privacy", "policy", "usage", "cleanup")
            }
        },
        recorded_at="2026-08-17T12:00:00+00:00",
    )
    manifest = coordinator.finalize()

    assert record.attempt.attempt_id == attempt.attempt_id
    assert record.attempt.prediction_id == attempt.prediction_id
    assert record.integrity_status == "resolved"
    assert {node.kind for node in record.nodes} == {
        "evaluation_root",
        "prediction_and_score",
        "prediction",
        "agent_root",
        "dataset",
    }
    assert {edge.relationship for edge in record.edges} == {
        "evaluation_has_dataset",
        "evaluation_has_prediction_and_score",
        "prediction_and_score_has_prediction",
        "prediction_has_agent_root",
    }
    assert record.agent_receipt.native_weave_call is False
    assert manifest.status == "complete"
    assert manifest.terminal_attempt_ids == (attempt.attempt_id,)
    assert coordinator.store.read_manifest() == manifest
    receipt = coordinator.publication_receipt()
    assert receipt.manifest_digest == manifest.manifest_digest
    assert receipt.plan_digest == coordinator.plan.plan_digest
    assert receipt.manifest_path == "manifest.json"
    assert [event.event for event in coordinator.store.read_events()] == [
        "run_initialized",
        "attempt_opened",
        "attempt_finalized",
        "manifest_finalized",
        "publication_receipt_written",
    ]


def test_strict_run_plan_reader_rejects_unknown_fields_and_drift(
    tmp_path: Path,
) -> None:
    coordinator, _ = _coordinator(tmp_path)
    raw = coordinator.plan.to_dict()
    raw["unexpected"] = True
    with pytest.raises(ValueError, match="unknown local evidence run plan"):
        local_evidence_run_plan_from_dict(raw)

    drifted = coordinator.plan.to_dict()
    drifted["run_snapshot_sha256"] = stable_digest({"snapshot": "changed"})
    with pytest.raises(ValueError, match="run-plan digest does not match"):
        local_evidence_run_plan_from_dict(drifted)


def test_terminal_attempt_is_exactly_once_and_idempotent(tmp_path: Path) -> None:
    coordinator, (attempt,) = _coordinator(tmp_path)
    record = _finish(coordinator, tmp_path, attempt)

    replay = coordinator.finish_attempt(
        attempt.attempt_id,
        terminal_status="passed",
        prediction_row=_prediction(attempt),
        scores={"answer_correct": 1, "evidence_honest": 1},
        agent_receipt=record.agent_receipt,
    )
    assert replay == record
    with pytest.raises(LocalEvidenceIntegrityError, match="already terminal"):
        coordinator.begin_attempt(attempt.attempt_id)

    changed = _prediction(attempt)
    changed["answer"] = "conflicting answer"
    with pytest.raises(LocalEvidenceIntegrityError, match="conflicting terminal"):
        coordinator.finish_attempt(
            attempt.attempt_id,
            terminal_status="passed",
            prediction_row=changed,
            scores={"answer_correct": 1, "evidence_honest": 1},
            agent_receipt=record.agent_receipt,
        )


def test_manifest_recomputation_detects_artifact_tampering(tmp_path: Path) -> None:
    coordinator, (attempt,) = _coordinator(tmp_path)
    record = _finish(coordinator, tmp_path, attempt)
    coordinator.finalize()
    prediction = next(node for node in record.nodes if node.kind == "prediction")
    assert prediction.artifact is not None
    target = coordinator.store.root / prediction.artifact.path
    target.write_text('{"tampered":true}\n', encoding="utf-8")

    with pytest.raises(LocalEvidenceIntegrityError, match="artifact digest changed"):
        coordinator.store.read_manifest()


def test_manifest_cannot_finalize_an_incomplete_run(tmp_path: Path) -> None:
    run_id = "incomplete-local-run"
    attempts = (
        _attempt(run_id=run_id, task_id="task-a", arm="baseline"),
        _attempt(run_id=run_id, task_id="task-a", arm="candidate"),
    )
    coordinator, planned = _coordinator(
        tmp_path,
        run_id=run_id,
        attempts=attempts,
    )
    _finish(coordinator, tmp_path, planned[0])

    manifest = coordinator.store.build_manifest()
    assert manifest.status == "incomplete"
    with pytest.raises(LocalEvidenceIntegrityError, match="status is incomplete"):
        coordinator.finalize()


def test_required_missing_run_conformance_keeps_manifest_incomplete(
    tmp_path: Path,
) -> None:
    coordinator, (attempt,) = _coordinator(
        tmp_path,
        run_id="missing-run-conformance",
        require_run_conformance=True,
    )
    _finish(coordinator, tmp_path, attempt)

    manifest = coordinator.store.build_manifest()

    assert manifest.run_conformance_required is True
    assert manifest.run_conformance is None
    assert manifest.status == "incomplete"
    with pytest.raises(LocalEvidenceIntegrityError, match="status is incomplete"):
        coordinator.finalize()


def test_failed_run_conformance_invalidates_complete_attempts(tmp_path: Path) -> None:
    coordinator, (attempt,) = _coordinator(
        tmp_path,
        run_id="failed-run-conformance",
        require_run_conformance=True,
    )
    _finish(coordinator, tmp_path, attempt)
    _write_run_conformance(coordinator, status="failed")

    manifest = coordinator.store.build_manifest()

    assert manifest.run_conformance is not None
    assert manifest.run_conformance.status == "failed"
    assert manifest.status == "invalid"
    with pytest.raises(LocalEvidenceIntegrityError, match="status is invalid"):
        coordinator.finalize()


def test_tampered_run_conformance_fails_manifest_recomputation(
    tmp_path: Path,
) -> None:
    coordinator, (attempt,) = _coordinator(
        tmp_path,
        run_id="tampered-run-conformance",
        require_run_conformance=True,
    )
    _finish(coordinator, tmp_path, attempt)
    path = _write_run_conformance(coordinator)
    manifest = coordinator.finalize()
    assert manifest.status == "complete"

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["enforced"] = False
    tampered["receipt_sha256"] = ""
    tampered["receipt_sha256"] = stable_digest(tampered)
    path.write_text(json.dumps(tampered, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(
        LocalEvidenceIntegrityError,
        match="manifest disagrees with run artifacts",
    ):
        coordinator.store.read_manifest()
    with pytest.raises(
        LocalEvidenceIntegrityError,
        match="manifest disagrees with run artifacts",
    ):
        coordinator.publication_receipt()


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"private_labels": {"answer": "host-only"}}, "private or credential"),
        ({"answer": "contains local-secret-value"}, "configured secret value"),
    ],
)
def test_private_truth_and_secret_values_fail_closed(
    tmp_path: Path,
    payload: dict[str, Any],
    error: str,
) -> None:
    coordinator, (attempt,) = _coordinator(
        tmp_path,
        secret_values=("local-secret-value",),
    )
    coordinator.begin_attempt(attempt.attempt_id)
    row = _prediction(attempt)
    row.update(payload)

    with pytest.raises(ValueError, match=error):
        coordinator.finish_attempt(
            attempt.attempt_id,
            terminal_status="passed",
            prediction_row=row,
            scores={"answer_correct": 1},
            agent_receipt=_agent_receipt(tmp_path, attempt),
        )


def test_missing_agent_evidence_is_invalid_not_a_fake_call(tmp_path: Path) -> None:
    coordinator, (attempt,) = _coordinator(tmp_path)
    coordinator.begin_attempt(attempt.attempt_id)
    receipt = AgentEvidenceReceiptV1(
        attempt_id=attempt.attempt_id,
        planned_conversation_id=f"planned-{attempt.attempt_id[:12]}",
        primary_session_id=None,
        child_session_ids=(),
        artifacts=(),
        transcript_artifact=None,
        transcript_session_id=None,
        correlation_verified=False,
        tool_event_count=None,
        tool_events_sha256=None,
        response_sha256=None,
        status="missing",
        reason="native transcript was not finalized",
    )
    record = coordinator.finish_attempt(
        attempt.attempt_id,
        terminal_status="failed",
        prediction_row=_prediction(attempt),
        scores={"answer_correct": 0},
        agent_receipt=receipt,
        attempt_payload={
            "receipts": {
                name: {"status": "passed"}
                for name in ("privacy", "policy", "usage", "cleanup")
            }
        },
    )

    assert record.integrity_status == "invalid"
    assert record.agent_receipt.native_weave_call is False
    assert coordinator.store.build_manifest().status == "invalid"
    with pytest.raises(LocalEvidenceIntegrityError, match="status is invalid"):
        coordinator.finalize()


def test_concurrent_attempts_do_not_cross_link(tmp_path: Path) -> None:
    run_id = "concurrent-local-run"
    attempts = (
        _attempt(run_id=run_id, task_id="task-a", arm="baseline"),
        _attempt(run_id=run_id, task_id="task-a", arm="candidate"),
    )
    coordinator, planned = _coordinator(
        tmp_path,
        run_id=run_id,
        attempts=attempts,
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        records = tuple(
            pool.map(lambda item: _finish(coordinator, tmp_path, item), planned)
        )

    manifest = coordinator.finalize()
    assert manifest.status == "complete"
    assert len(set(manifest.terminal_attempt_ids)) == 2
    for record in records:
        refs = {node.kind: node.ref for node in record.nodes}
        assert record.attempt.attempt_id in refs["agent_root"]
        assert record.attempt.prediction_id in refs["prediction"]
        assert all(
            other.attempt.attempt_id not in refs["agent_root"]
            for other in records
            if other != record
        )


def test_same_candidate_attempts_share_scope_without_cross_linking_artifacts(
    tmp_path: Path,
) -> None:
    run_id = "same-candidate-local-run"
    attempts = (
        _attempt(
            run_id=run_id,
            task_id="task-a",
            arm="same-candidate",
            index=1,
        ),
        _attempt(
            run_id=run_id,
            task_id="task-a",
            arm="same-candidate",
            index=2,
        ),
    )
    assert attempts[0].candidate_id == attempts[1].candidate_id
    assert attempts[0].attempt_id != attempts[1].attempt_id
    coordinator, planned = _coordinator(
        tmp_path,
        run_id=run_id,
        attempts=attempts,
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        records = tuple(
            pool.map(lambda item: _finish(coordinator, tmp_path, item), planned)
        )

    manifest = coordinator.finalize()
    assert manifest.status == "complete"
    assert len(manifest.attempt_records) == 2
    nodes = [
        {node.kind: node for node in record.nodes}
        for record in records
    ]
    # Dataset and Evaluation scope are intentionally candidate-level.
    assert nodes[0]["dataset"].ref == nodes[1]["dataset"].ref
    assert nodes[0]["evaluation_root"].ref == nodes[1]["evaluation_root"].ref
    for kind in ("prediction_and_score", "prediction", "agent_root"):
        assert nodes[0][kind].ref != nodes[1][kind].ref
        assert nodes[0][kind].artifact != nodes[1][kind].artifact
    for index, record in enumerate(records):
        other = records[1 - index]
        by_kind = nodes[index]
        assert record.attempt.attempt_id in by_kind["prediction_and_score"].ref
        assert record.attempt.attempt_id in by_kind["agent_root"].ref
        assert record.attempt.prediction_id in by_kind["prediction"].ref
        assert other.attempt.attempt_id not in by_kind["prediction_and_score"].ref
        assert other.attempt.attempt_id not in by_kind["agent_root"].ref
        assert other.attempt.prediction_id not in by_kind["prediction"].ref
        transcript_path = record.agent_receipt.transcript_artifact
        assert transcript_path is not None
        assert record.attempt.attempt_id in transcript_path.path
        assert other.attempt.attempt_id not in transcript_path.path


def test_local_evidence_module_does_not_import_weave() -> None:
    module_path = Path(__file__).parents[1] / "fugue" / "bench" / "local_evidence.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        str(node.module or "").split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert "weave" not in imported
