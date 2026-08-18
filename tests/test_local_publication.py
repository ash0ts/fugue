from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import fugue.bench.local_publication as publication
from fugue.bench.candidates import attempt_id, attempt_identity, stable_digest
from fugue.bench.export import PredictionRowV1
from fugue.bench.files import atomic_write_json
from fugue.bench.local_evidence import (
    AgentEvidenceReceiptV1,
    LocalArtifactRefV1,
    LocalAttemptPlanV1,
    LocalEvidenceCoordinator,
    LocalEvidenceStore,
    build_local_evidence_run_plan,
)
from fugue.bench.local_publication import (
    LocalResultPublicationError,
    MissingWeaveExtraError,
    WeaveHostedObjectRefV1,
    WeavePublicationOutcomeV1,
    WeavePublicationReceiptV1,
    WeavePublicationTargetV1,
    publish_local_result_to_weave,
    read_weave_publication_receipt,
    weave_publisher_from_environment,
)


class _FakeComparisonResultV3:
    def __init__(self, manifest: Any, manifest_file_sha256: str) -> None:
        self.schema_version = 3
        self.source = manifest.run_id
        self.result_digest = stable_digest({"result": manifest.run_id})
        self.qualification_digest = stable_digest(
            {"qualification": manifest.run_id}
        )
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
                    execution_status="passed",
                    evidence_status="reconciled",
                    passed=True,
                    scores={"answer_correct": True},
                    score_explanations={"answer_correct": "The answer matched."},
                    sanitized_answer_excerpt="bounded public answer",
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
                    ref=(
                        f"weave:///{target.project_slug}/{object_type}/{object_id}"
                    ),
                )
            )
    return WeavePublicationOutcomeV1(
        target=target,
        objects=tuple(sorted(objects, key=lambda item: (item.attempt_id, item.kind))),
        publisher_id="fugue-weave-publisher",
        publisher_revision=publisher_revision,
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
    target = WeavePublicationTargetV1(entity="wandb", project="local-result")
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
        item
        for item in receipt.hosted_objects
        if item.kind == "agent_evidence_receipt"
    )
    assert agent.native_agent_call is False
    assert read_weave_publication_receipt(
        result_path.with_name("weave-publication-receipt.json")
    ) == receipt


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
    target = WeavePublicationTargetV1(entity="wandb", project="local-result")

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
    target = WeavePublicationTargetV1(entity="wandb", project="local-result")
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

    expected = "another project" if failure == "wrong_project" else "secret"
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
    target = WeavePublicationTargetV1(entity="wandb", project="local-result")

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


def test_weave_publisher_is_lazy_and_has_actionable_missing_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_name: str):
        raise ModuleNotFoundError("No module named 'weave'")

    monkeypatch.setattr(publication.importlib, "import_module", missing)

    with pytest.raises(MissingWeaveExtraError, match=r'fugue\[weave\]'):
        weave_publisher_from_environment({})


def test_real_weave_adapter_emits_nested_five_object_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fugue import weave_support

    _manifest_path, manifest = _canonical_manifest(tmp_path)
    _result_path, result = _result_fixture(tmp_path, manifest)
    target = WeavePublicationTargetV1(entity="wandb", project="local-result")
    finished: list[tuple[str, dict[str, Any]]] = []

    class FakeClient:
        project_id = target.project_slug

        @staticmethod
        def create_call(
            op,
            inputs,
            parent=None,
            attributes=None,
            display_name=None,
            *,
            use_stack=True,
            _call_id_override=None,
        ):
            del op, inputs, attributes, display_name, use_stack
            return SimpleNamespace(
                id=_call_id_override,
                parent_id=getattr(parent, "id", None),
                project_id=target.project_slug,
                ref=SimpleNamespace(
                    uri=(
                        f"weave:///{target.project_slug}/call/"
                        f"{_call_id_override}"
                    )
                ),
            )

        @staticmethod
        def finish_call(call, output=None):
            finished.append((call.id, dict(output or {})))

        @staticmethod
        def flush() -> None:
            return None

    client = FakeClient()

    def publish_object(_value, *, name):
        return SimpleNamespace(
            uri=f"weave:///{target.project_slug}/object/{name}:digest-v1"
        )

    fake_weave = SimpleNamespace(
        __version__="test-sdk",
        get_client=lambda: client,
        publish=publish_object,
    )
    monkeypatch.setattr(
        publication.importlib,
        "import_module",
        lambda name: fake_weave if name == "weave" else None,
    )
    monkeypatch.setattr(
        weave_support,
        "initialize_weave",
        lambda _project, _env: fake_weave,
    )

    publisher = weave_publisher_from_environment({"WANDB_API_KEY": "test-only-key"})
    outcome = publisher(result, manifest, target)

    assert isinstance(outcome, WeavePublicationOutcomeV1)
    assert len(outcome.objects) == 5
    assert len(finished) == 4
    assert finished[0][1]["native_agent_call"] is False
    by_kind = {item.kind: item for item in outcome.objects}
    assert by_kind["agent_evidence_receipt"].native_agent_call is False
    assert by_kind["dataset"].ref.startswith(
        f"weave:///{target.project_slug}/object/"
    )
