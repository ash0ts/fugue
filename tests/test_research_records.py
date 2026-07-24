from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from fugue.research.contracts import (
    RESEARCH_SCHEMA_VERSION,
    ExperimentPreviewV1,
    ExperimentRecordV1,
    ResearchError,
    build_experiment_draft,
    now,
    sign_preview,
    sign_record,
    study_update_from_dict,
)
from fugue.research.records import (
    HttpResearchRecordSink,
    JsonlResearchRecordSink,
    ResearchRecordPublisher,
    public_evidence_selector,
    research_log_event_from_dict,
    sign_research_log_event,
)
from fugue.research.store import StudyStore

_A = "a" * 64
_B = "b" * 64


def _store(tmp_path: Path) -> StudyStore:
    store = StudyStore(tmp_path)
    store.create_study(
        study_id="research-1",
        title="Private title",
        campaign_id="campaign-1",
        question="Private research question",
        operation_id="create-research",
    )
    return store


def _preview(*, proposal_id: str = "proposal-1") -> ExperimentPreviewV1:
    draft = build_experiment_draft(
        study_id="research-1",
        campaign_id="campaign-1",
        proposal_id=proposal_id,
        stage_id="discovery",
        question="Private controlled question",
        hypothesis="Private hypothesis",
        fixed_dimensions=["model"],
        varied_dimensions=["loop"],
        measured_dimensions=["pass"],
        display_labels={
            "loop": "Loop design",
            "baseline": "Current behavior",
        },
        experiment_id="comparison-1",
        model="model-1",
        n_attempts=1,
        n_concurrent=1,
    )
    return sign_preview(
        ExperimentPreviewV1(
            schema_version=RESEARCH_SCHEMA_VERSION,
            study_id="research-1",
            experiment_id=f"research-1.{proposal_id}",
            campaign_id="campaign-1",
            catalog_digest=_A,
            policy_digest=_B,
            draft=draft.to_dict(),
            task_suite_preview=None,
            plan_receipt={"plan_digest": _A},
            estimated_cells=6,
            estimated_calls={"agent": 6},
            estimated_cost_usd=45.0,
            eligible=True,
            blockers=(),
        )
    )


def test_research_log_contract_is_strict_and_content_addressed() -> None:
    raw = {
        "schema_version": 1,
        "producer_event_id": "producer-1",
        "sequence": 1,
        "timestamp": "2026-07-22T12:00:00Z",
        "source": "fixture",
        "actor": {"actor_type": "service", "name": "fixture"},
        "research_id": "research-1",
        "study_id": "study-1",
        "classification": "evidence",
        "state": "evaluating",
        "message": "Evidence reconciled.",
        "evidence": [
            {
                "system": "weave",
                "kind": "evaluation",
                "ref": "evaluation-1",
                "digest": _A,
            }
        ],
    }
    event = research_log_event_from_dict(raw, require_digest=False)
    assert research_log_event_from_dict(event.to_dict()) == event
    with pytest.raises(ValueError, match="unknown fields"):
        research_log_event_from_dict({**event.to_dict(), "prompt": "private"})
    with pytest.raises(ValueError, match="event_digest"):
        research_log_event_from_dict({**event.to_dict(), "message": "changed"})
    with pytest.raises(ValueError, match="size limit"):
        research_log_event_from_dict(
            {**raw, "summary": {"too_large": "x" * 70_000}},
            require_digest=False,
        )
    with pytest.raises(ValueError, match="private field"):
        research_log_event_from_dict(
            {**raw, "summary": {"hidden_reasoning": "private"}},
            require_digest=False,
        )
    with pytest.raises(ValueError, match="http or https"):
        research_log_event_from_dict(
            {
                **raw,
                "evidence": [
                    {
                        "system": "artifact",
                        "kind": "artifact",
                        "ref": "artifact-1",
                        "uri": "file:///private/result.json",
                    }
                ],
            },
            require_digest=False,
        )
    with pytest.raises(ValueError, match="timezone"):
        research_log_event_from_dict(
            {**raw, "timestamp": "2026-07-22T12:00:00"},
            require_digest=False,
        )


def test_public_evidence_selectors_keep_identities_not_private_material() -> None:
    assert public_evidence_selector(
        {
            "entity": "example",
            "project": "evaluation",
            "call_id": "call-1",
            "expected_paths": ["private/gold.py"],
            "criteria": {"answer": "private"},
        }
    ) == {
        "entity": "example",
        "project": "evaluation",
        "call_id": "call-1",
    }


def test_preview_is_unpublished_until_approval_request(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert len(store.research_log_events()) == 1
    preview = _preview()
    assert len(store.research_log_events()) == 1

    request = store.record_approval_request(
        preview,
        operation_id="request-approval",
    )
    assert request.state == "awaiting_approval"
    assert request.study_id == preview.experiment_id
    assert request.reserved_cost_usd == 45.0
    assert request.summary["planned_cells"] == 6
    assert (
        store.record_approval_request(
            preview,
            operation_id="request-approval",
        )
        == request
    )
    assert len(store.research_log_events()) == 2

    serialized = json.dumps([item.to_dict() for item in store.research_log_events()])
    assert "Private research question" not in serialized
    assert request.summary["experiment_view"]["kind"] == "design"
    assert (
        request.summary["experiment_view"]["question"] == "Private controlled question"
    )
    assert request.summary["experiment_view"]["hypothesis"] == "Private hypothesis"
    [factor] = request.summary["experiment_view"]["varied_factors"]
    assert factor["label"] == "Loop design"
    assert "prompt" not in request.summary["experiment_view"]


def test_approval_request_operation_conflict_is_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_approval_request(_preview(), operation_id="request-shared")
    with pytest.raises(ResearchError, match="different input"):
        store.record_approval_request(
            _preview(proposal_id="proposal-2"),
            operation_id="request-shared",
        )


def test_experiment_state_and_sourced_update_append_safe_records(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    preview = _preview()
    timestamp = now()
    record = sign_record(
        ExperimentRecordV1(
            schema_version=1,
            id=preview.experiment_id,
            study_id=preview.study_id,
            campaign_id=preview.campaign_id,
            state="queued",
            draft=preview.draft,
            preview=preview.to_dict(),
            approval=None,
            parent_experiment_ids=(),
            proposal=None,
            plan=preview.plan_receipt,
            task_suite_lock=None,
            prepared_plan=None,
            admission=None,
            run_id=None,
            outcome=None,
            evaluation=None,
            analysis=None,
            error=None,
            created_at=timestamp,
            updated_at=timestamp,
        )
    )
    store.insert_experiment(record, operation_id="start-1", input_digest=_A)
    queued = store.research_log_events()[-1]
    assert queued.study_id == record.id
    assert queued.state == "preparing"
    assert queued.summary["planned_cells"] == 6

    completed = sign_record(
        replace(
            record,
            state="completed",
            run_id="run-1",
            outcome={
                "outcome_id": "outcome-1",
                "outcome_digest": _A,
                "run_snapshot_sha256": _B,
                "expected_predictions": 6,
                "observed_predictions": 6,
                "passed": 4,
                "failed": 2,
                "not_applicable": 0,
                "eligible": True,
                "limitations": ["private limitation text"],
                "observed_cost_usd": 12.5,
            },
            updated_at=now(),
        )
    )
    store.update_experiment(
        completed,
        event_type="experiment_completed",
        message="Study completed with immutable evidence.",
        release=True,
    )
    terminal = store.research_log_events()[-1]
    assert terminal.state == "completed"
    assert terminal.classification == "result"
    assert terminal.observed_cost_usd == 12.5
    assert terminal.summary["passed"] == 4
    assert terminal.summary["limitation_count"] == 1
    assert "private limitation text" not in json.dumps(terminal.to_dict())
    assert terminal.summary["experiment_view"]["kind"] == "evaluation"
    assert terminal.summary["experiment_view"]["limitations"] == [
        "Additional limitations are recorded in the immutable Fugue outcome."
    ]

    study = store.get_study("research-1")
    store.update_study(
        study.id,
        study_update_from_dict(
            {
                "message": "private note body",
                "attribution": {"actor_type": "human", "name": "operator"},
            }
        ),
        operation_id="private-note",
        expected_revision=study.revision,
    )
    assert "private note body" not in json.dumps(
        store.research_log_events()[-1].to_dict()
    )


def test_historical_experiment_views_are_backfilled_without_execution(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    preview = _preview()
    timestamp = now()
    record = sign_record(
        ExperimentRecordV1(
            schema_version=1,
            id=preview.experiment_id,
            study_id=preview.study_id,
            campaign_id=preview.campaign_id,
            state="queued",
            draft=preview.draft,
            preview=preview.to_dict(),
            approval={"approval_digest": _A},
            parent_experiment_ids=(),
            proposal=None,
            plan=preview.plan_receipt,
            task_suite_lock=None,
            prepared_plan=None,
            admission={"reserved_cost_usd": 45.0},
            run_id=None,
            outcome=None,
            evaluation=None,
            analysis=None,
            error=None,
            created_at=timestamp,
            updated_at=timestamp,
        )
    )
    store.insert_experiment(record, operation_id="start-old", input_digest=_A)
    export_rows = [
        {
            "schema_version": 1,
            "prediction_schema_version": 1,
            "prediction_id": f"prediction-{index}",
            "run_id": "run-old",
            "candidate_id": f"candidate-{index}",
            "comparison_example_id": "paired-support-review",
            "trial_index": 1,
            "execution_kind": "agent",
            "status": "passed",
            "pass": index < 2,
            "workload_id": "support-data-authority-suite",
            "task_name": "Paired support review",
            "harness": "codex" if index % 2 else "claude-code",
            "variant_id": ("action-gate" if index < 2 else "baseline"),
            "context_system_id": "none",
            "trace_link_status": "linked",
            "trace_project": "team/evaluations",
            "weave_call_id": f"call-{index}",
            "weave_conversation_ids": [f"conversation-{index}"],
            "weave_root_span_ids": [f"root-{index}"],
            "weave_trace_ids": [f"trace-{index}"],
            "runtime_equivalence_status": "equivalent",
            "runtime_drift": False,
        }
        for index in range(6)
    ]
    export_payload = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in export_rows
    ).encode()
    export_path = tmp_path / ".fugue" / "runtime" / "historical-export.jsonl"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_bytes(export_payload)
    completed = sign_record(
        replace(
            record,
            state="completed",
            run_id="run-old",
            outcome={
                "outcome_id": "outcome-old",
                "outcome_digest": _A,
                "run_snapshot_sha256": _B,
                "expected_predictions": 6,
                "observed_predictions": 6,
                "passed": 2,
                "failed": 4,
                "not_applicable": 0,
                "eligible": True,
                "limitations": ["private limitation text"],
                "observed_cost_usd": 1.53,
                "export_path": export_path.relative_to(tmp_path).as_posix(),
                "export_sha256": hashlib.sha256(export_payload).hexdigest(),
                "row_refs": [
                    {
                        key: value
                        for key, value in row.items()
                        if not key.startswith("weave_") and key != "trace_project"
                    }
                    for row in export_rows
                ],
            },
            updated_at=now(),
        )
    )
    store.update_experiment(
        completed,
        event_type="experiment_completed",
        message="Historical experiment completed.",
        release=True,
    )

    # Simulate a database produced before experiment-view publication existed.
    with store._connect() as conn:
        old_sequences = [
            event.sequence
            for event in store.research_log_events()
            if event.study_id == record.id
        ]
        conn.executemany(
            "DELETE FROM research_log_events WHERE sequence=?",
            [(sequence,) for sequence in old_sequences],
        )
    assert store.ensure_experiment_view_projection_events() == 2
    projected = [
        event for event in store.research_log_events() if event.study_id == record.id
    ]
    assert [event.summary["experiment_view"]["kind"] for event in projected] == [
        "design",
        "evaluation",
    ]
    assert projected[-1].state == "completed"
    assert projected[-1].summary["passed"] == 2
    assert (
        projected[-1].summary["experiment_view"]["infrastructure_health"]
        == "unavailable"
    )
    assert projected[-1].observed_cost_usd == 1.53
    assert all(
        any(link["system"] == "weave" for link in cell["evidence_links"])
        for cell in projected[-1].summary["experiment_view"]["cells"]
    )
    assert "private limitation text" not in json.dumps(
        [event.to_dict() for event in projected]
    )
    assert store.ensure_experiment_view_projection_events() == 0

    restarted = StudyStore(tmp_path)
    assert restarted.ensure_experiment_view_projection_events() == 0
    assert [event.producer_event_id for event in restarted.research_log_events()] == [
        event.producer_event_id for event in store.research_log_events()
    ]


def test_jsonl_publication_is_ordered_idempotent_and_restart_safe(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.record_approval_request(_preview(), operation_id="request-1")
    sink = JsonlResearchRecordSink(tmp_path / "projection" / "events.jsonl")
    publisher = ResearchRecordPublisher(store, [sink])

    assert publisher.flush() == {"delivered": 2, "failed": 0}
    assert publisher.flush() == {"delivered": 0, "failed": 0}
    rows = [
        json.loads(line) for line in sink.path.read_text(encoding="utf-8").splitlines()
    ]
    assert [item["sequence"] for item in rows] == [1, 2]

    restarted = StudyStore(tmp_path)
    assert ResearchRecordPublisher(restarted, [sink]).flush() == {
        "delivered": 0,
        "failed": 0,
    }
    status = restarted.research_publication_status()
    assert status["event_count"] == 2
    assert status["deliveries"][0]["state"] == "delivered"


def test_jsonl_publication_recovers_a_missing_index_and_serializes_writers(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    event = store.research_log_events()[0]
    sink = JsonlResearchRecordSink(tmp_path / "projection" / "events.jsonl")
    sink.publish(event)
    sink.path.with_suffix(f"{sink.path.suffix}.index.json").unlink()

    with ThreadPoolExecutor(max_workers=4) as executor:
        tuple(executor.map(sink.publish, [event] * 8))

    assert len(sink.path.read_text(encoding="utf-8").splitlines()) == 1
    conflicting = sign_research_log_event(
        replace(event, message="Conflicting replay.", event_digest="")
    )
    with pytest.raises(ResearchError, match="different content"):
        sink.publish(conflicting)

    second = store.record_approval_request(_preview(), operation_id="request-1")
    with sink.path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(second.to_dict(), sort_keys=True) + "\n")
    sink.publish(second)
    assert len(sink.path.read_text(encoding="utf-8").splitlines()) == 2


def test_failed_sink_remains_pending_without_changing_research(
    tmp_path: Path,
) -> None:
    class BrokenSink:
        sink_id = "broken"

        def publish(self, _: object) -> None:
            raise RuntimeError("console unavailable")

    store = _store(tmp_path)
    before = store.get_study("research-1")
    publisher = ResearchRecordPublisher(store, [BrokenSink()])
    assert publisher.flush() == {"delivered": 0, "failed": 1}
    assert store.get_study("research-1") == before
    assert len(store.pending_research_log_events("broken")) == 1
    assert store.research_publication_status()["deliveries"][0]["state"] == "failed"


def test_http_sink_uses_ingest_auth_and_producer_idempotency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[object] = []

    class Response:
        status = 201

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_: object) -> None:
            return None

    def urlopen(request: object, *, timeout: float) -> Response:
        assert timeout == 3
        requests.append(request)
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    event = _store(tmp_path).research_log_events()[0]
    sink = HttpResearchRecordSink(
        "http://127.0.0.1:3000/api/research-log-events",
        "ingest-secret",
        timeout=3,
    )
    sink.publish(event)

    request = requests[0]
    assert request.get_header("Authorization") == "Bearer ingest-secret"
    assert request.get_header("Idempotency-key") == event.producer_event_id
    assert json.loads(request.data)["event_digest"] == event.event_digest
