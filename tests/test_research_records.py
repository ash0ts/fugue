from __future__ import annotations

import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from fugue.bench.analysis_contracts import aligned_analysis_from_dict
from fugue.bench.candidates import attempt_id as canonical_attempt_id
from fugue.bench.candidates import stable_digest
from fugue.bench.local_publication import (
    StudyPublicationScopeV1,
    WeaveHostedObjectRefV1,
    WeavePublicationReceiptV1,
    WeavePublicationTargetV1,
)
from fugue.model_plane import default_evidence_destination
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
from fugue.research.experiment_views import (
    ExperimentViewV3,
    experiment_view_from_dict,
)
from fugue.research.records import (
    RESEARCH_LOG_MAX_BYTES,
    HttpResearchRecordSink,
    JsonlResearchRecordSink,
    ResearchLogEventV1,
    ResearchRecordPublisher,
    experiment_view_manifest_from_dict,
    experiment_view_page_from_dict,
    public_evidence_selector,
    research_log_event_from_dict,
    sign_research_log_event,
    weave_publication_evidence_from_dict,
    weave_publication_evidence_from_receipt,
)
from fugue.research.store import StudyStore, _paged_terminal_experiment_view

_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64


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


def _event_with_encoded_size(
    store: StudyStore,
    *,
    encoded_size: int,
) -> ResearchLogEventV1:
    event = replace(
        store.research_log_events()[0],
        summary={"public_note": ""},
        event_digest="",
    )
    unsigned = event.to_dict()
    unsigned.pop("event_digest", None)
    encoded = json.dumps(unsigned, separators=(",", ":")).encode()
    padding = encoded_size - len(encoded)
    assert padding >= 0
    event = replace(event, summary={"public_note": "x" * padding})
    unsigned = event.to_dict()
    unsigned.pop("event_digest", None)
    assert (
        len(json.dumps(unsigned, separators=(",", ":")).encode())
        == encoded_size
    )
    return event


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


def _large_terminal_v3_view(*, pair_count: int = 96) -> ExperimentViewV3:
    result_project = "wandb/fugue-mcp-release-qualification-v1"
    result_weave_url = f"https://wandb.ai/{result_project}/weave"
    fixture = (
        Path(__file__).parent
        / "fixtures"
        / "experiment-view-v3-study-console-golden.json"
    )
    raw = json.loads(fixture.read_text(encoding="utf-8"))
    template_pair = raw["paired_cases"][0]
    pairs: list[dict[str, object]] = []
    aligned_attempts: list[dict[str, object]] = []
    task_pair_counts: dict[str, int] = {}
    for index in range(1, pair_count + 1):
        task_number = ((index - 1) // 2) + 1
        attempt_number = ((index - 1) % 2) + 1
        task_id = f"task-{task_number:03d}"
        task_label = f"Maintenance task {task_number:03d}"
        pair = copy.deepcopy(template_pair)
        pair["task_id"] = task_id
        pair["task_label"] = task_label
        pair["attempt"] = attempt_number
        pair["pair_id"] = stable_digest(
            {
                "task_id": task_id,
                "harness": "claude-code",
                "attempt": attempt_number,
            }
        )
        attempt_ids: dict[str, str] = {}
        for arm in ("baseline", "candidate"):
            attempt = pair[arm]
            identity = attempt["identity"]
            identity["task_id"] = task_id
            identity["attempt"] = attempt_number
            attempt_id = canonical_attempt_id(**identity)
            attempt["attempt_id"] = attempt_id
            attempt["prediction_id"] = f"{arm}-prediction-{index:03d}"
            attempt["weave_agent_root_call_id"] = f"{arm}-agent-root-{index:03d}"
            attempt["otel_root_span_id"] = f"{arm}-otel-root-{index:03d}"
            for link in attempt["evidence_links"]:
                kind = link["kind"]
                if kind == "dataset":
                    link["ref"] = (
                        f"weave:///{result_project}/object/release-dataset:v1"
                    )
                    link["url"] = (
                        f"{result_weave_url}/objects/release-dataset/versions/v1"
                    )
                    continue
                call_id = (
                    f"{arm}-agent-root-{index:03d}"
                    if kind == "agent_root"
                    else f"{arm}-{kind}-{index:03d}"
                )
                link["ref"] = f"weave:///{result_project}/call/{call_id}"
                link["url"] = f"{result_weave_url}/calls/{call_id}"
            attempt_ids[arm] = attempt_id
        pairs.append(pair)
        aligned_attempts.append(
            {
                "alignment_id": stable_digest(
                    {
                        "task_id": task_id,
                        "harness": "claude-code",
                        "attempt": attempt_number,
                    }
                ),
                "task_id": task_id,
                "task_label": task_label,
                "harness": "claude-code",
                "attempt": attempt_number,
                "attempt_ids_by_arm": attempt_ids,
            }
        )
        task_pair_counts[task_id] = task_pair_counts.get(task_id, 0) + 1

    task_summaries: list[dict[str, object]] = []
    task_validity: list[dict[str, object]] = []
    for task_id, task_pair_count in task_pair_counts.items():
        task_summaries.append(
            {
                "task_id": task_id,
                "validity": "valid",
                "pair_counts": {"improved": task_pair_count},
            }
        )
        task_validity.append(
            {
                "task_id": task_id,
                "status": "valid",
                "discriminating_dimensions": ["bounded_answer"],
            }
        )

    analysis = raw["aligned_analysis"]
    analysis["aligned_attempts"] = aligned_attempts
    analysis["task_summaries"] = task_summaries
    analysis.pop("analysis_digest", None)
    raw["aligned_analysis"] = aligned_analysis_from_dict(analysis).to_dict()
    raw["paired_cases"] = pairs
    raw["task_validity"] = task_validity
    raw["matrix_size"] = pair_count * 2
    raw["completed_cells"] = pair_count * 2
    raw["state_counts"] = {"completed": pair_count * 2}
    raw["behavioral_summary"].update(
        {
            "improved_pairs": pair_count,
            "supported_claim": f"Candidate improved {pair_count} aligned pairs.",
        }
    )
    raw["judge_summary"] = {
        "status": "not_used",
        "claim_status": "not_applicable",
        "judges": [],
        "by_variant": {"baseline": {}, "candidate": {}},
        "unavailable_attempts": 0,
    }
    view = experiment_view_from_dict(raw)
    assert isinstance(view, ExperimentViewV3)
    return view


def _published_v3_view(*, pair_count: int = 4) -> ExperimentViewV3:
    view = _large_terminal_v3_view(pair_count=pair_count)
    return replace(
        view,
        result_digest=_A,
        qualification_digest=_B,
        evidence_links=(
            {
                "kind": "comparison_rows",
                "system": "fugue",
                "ref": "20260820T023521-83a3add82a",
            },
            {
                "kind": "comparison_result",
                "system": "fugue",
                "ref": f".fugue/results/comparisons/{_C}/result.json",
                "digest": _A,
            },
        ),
    )


def _weave_publication_receipt(
    view: ExperimentViewV3,
    *,
    research_id: str = "research-1",
    study_id: str = "study-1",
    result_digest: str | None = None,
    qualification_digest: str | None = None,
    published_at: str = "2026-08-20T12:00:00+00:00",
) -> WeavePublicationReceiptV1:
    target = WeavePublicationTargetV1(
        entity="wandb",
        project="fugue-result-project",
        study_scope=StudyPublicationScopeV1(
            research_id=research_id,
            study_id=study_id,
        ),
        destination=default_evidence_destination("wandb/fugue-result-project"),
    )
    attempt_ids = tuple(
        sorted(
            str(attempt["attempt_id"])
            for pair in view.paired_cases
            for arm in ("baseline", "candidate")
            if isinstance((attempt := pair.get(arm)), dict)
        )
    )
    objects: list[WeaveHostedObjectRefV1] = []
    for attempt_id in attempt_ids:
        for kind in (
            "evaluation_root",
            "prediction_and_score",
            "prediction",
            "agent_evidence_receipt",
            "dataset",
        ):
            object_id = f"{kind}-{attempt_id[:16]}"
            object_type = "object" if kind == "dataset" else "call"
            objects.append(
                WeaveHostedObjectRefV1(
                    attempt_id=attempt_id,
                    kind=kind,
                    target=target,
                    object_id=object_id,
                    ref=(f"weave:///{target.project_slug}/{object_type}/{object_id}"),
                )
            )
    selected_result_digest = result_digest or _A
    publication_id = stable_digest(
        {
            "schema_version": 1,
            "target": target.to_dict(),
            "result_digest": selected_result_digest,
            "local_manifest_digest": _D,
        }
    )
    return WeavePublicationReceiptV1(
        publication_id=publication_id,
        target=target,
        result_digest=selected_result_digest,
        qualification_digest=qualification_digest or _B,
        result_file_sha256="e" * 64,
        local_manifest_digest=_D,
        local_manifest_file_sha256="f" * 64,
        hosted_objects=tuple(
            sorted(objects, key=lambda item: (item.attempt_id, item.kind))
        ),
        publisher_id="fugue-weave-publisher",
        publisher_revision="publisher-revision-v1",
        status="published",
        published_at=published_at,
    )


def test_weave_publication_evidence_recomputes_the_original_receipt() -> None:
    view = _published_v3_view()
    receipt = _weave_publication_receipt(view)
    evidence = weave_publication_evidence_from_receipt(receipt.to_dict())

    assert weave_publication_evidence_from_dict(evidence.to_dict()) == evidence
    assert len(evidence.attempt_ids) == 8
    assert len(evidence.hosted_objects) == 40
    assert all(item.native_agent_call is False for item in evidence.hosted_objects)

    mutated = evidence.to_dict()
    mutated["publisher_revision"] = "unbound-revision"
    with pytest.raises(ValueError, match="receipt digest does not recompute"):
        weave_publication_evidence_from_dict(mutated)


def test_weave_publication_evidence_requires_exact_public_object_shape() -> None:
    receipt = _weave_publication_receipt(_published_v3_view())
    incomplete = replace(
        receipt,
        hosted_objects=receipt.hosted_objects[:-1],
        receipt_digest="",
    )

    with pytest.raises(ValueError, match="exact five-object chain"):
        weave_publication_evidence_from_receipt(incomplete.to_dict())

    cross_transport = receipt.to_dict()
    cross_transport["hosted_objects"][0]["native_agent_call"] = True
    cross_transport.pop("receipt_digest")
    cross_transport["receipt_digest"] = stable_digest(cross_transport)
    with pytest.raises(ValueError, match="not a native Weave Agent call"):
        weave_publication_evidence_from_receipt(cross_transport)


def test_store_records_one_idempotent_weave_publication_event(tmp_path: Path) -> None:
    store = _store(tmp_path)
    view = _published_v3_view()
    terminal = store.record_experiment_view_event(
        research_id="research-1",
        experiment_id="study-1",
        producer_event_id="study-1-result",
        classification="result",
        state="completed",
        message="Published the canonical terminal result.",
        view=view,
    )
    publication = weave_publication_evidence_from_receipt(
        _weave_publication_receipt(view).to_dict()
    )

    delivered = store.record_weave_publication_evidence(publication)

    assert delivered.classification == "evidence"
    assert delivered.state == "completed"
    assert delivered.research_id == "research-1"
    assert delivered.study_id == "study-1"
    assert delivered.summary == {"weave_publication": publication.to_dict()}
    assert delivered.evidence[0].ref == (
        f"weave-publication:{publication.publication_id}"
    )
    assert delivered.evidence[0].digest == publication.receipt_digest
    assert delivered.evidence[0].selector == {
        "entity": "wandb",
        "project": "fugue-result-project",
    }
    assert not delivered.evidence[0].ref.startswith("/")
    assert research_log_event_from_dict(delivered.to_dict()) == delivered
    assert terminal in store.research_log_events()

    before = tuple(store.research_log_events())
    assert store.record_weave_publication_evidence(publication) == delivered
    assert tuple(store.research_log_events()) == before


def test_store_rejects_conflicting_weave_publication_receipt(tmp_path: Path) -> None:
    store = _store(tmp_path)
    view = _published_v3_view()
    store.record_experiment_view_event(
        research_id="research-1",
        experiment_id="study-1",
        producer_event_id="study-1-result",
        classification="result",
        state="completed",
        message="Published the canonical terminal result.",
        view=view,
    )
    receipt = _weave_publication_receipt(view)
    store.record_weave_publication_evidence(
        weave_publication_evidence_from_receipt(receipt.to_dict())
    )
    conflicting = replace(
        receipt,
        published_at="2026-08-20T12:01:00+00:00",
        receipt_digest="",
    )

    with pytest.raises(ResearchError, match="another receipt"):
        store.record_weave_publication_evidence(
            weave_publication_evidence_from_receipt(conflicting.to_dict())
        )


@pytest.mark.parametrize(
    "publication",
    (
        lambda view: _weave_publication_receipt(view, study_id="another-study"),
        lambda view: _weave_publication_receipt(view, result_digest=_C),
        lambda view: _weave_publication_receipt(view, qualification_digest=_C),
        lambda _view: _weave_publication_receipt(_published_v3_view(pair_count=1)),
    ),
    ids=("scope", "result", "qualification", "attempt-set"),
)
def test_store_rejects_publication_that_disagrees_with_terminal_projection(
    tmp_path: Path,
    publication,
) -> None:
    store = _store(tmp_path)
    view = _published_v3_view()
    store.record_experiment_view_event(
        research_id="research-1",
        experiment_id="study-1",
        producer_event_id="study-1-result",
        classification="result",
        state="completed",
        message="Published the canonical terminal result.",
        view=view,
    )

    with pytest.raises(ResearchError, match="projection"):
        store.record_weave_publication_evidence(
            weave_publication_evidence_from_receipt(publication(view).to_dict())
        )


def test_weave_publication_event_rejects_private_summary_fields(tmp_path: Path) -> None:
    store = _store(tmp_path)
    view = _published_v3_view()
    store.record_experiment_view_event(
        research_id="research-1",
        experiment_id="study-1",
        producer_event_id="study-1-result",
        classification="result",
        state="completed",
        message="Published the canonical terminal result.",
        view=view,
    )
    delivered = store.record_weave_publication_evidence(
        weave_publication_evidence_from_receipt(
            _weave_publication_receipt(view).to_dict()
        )
    )
    raw = delivered.to_dict()
    raw.pop("event_digest")
    raw["summary"]["weave_publication"]["credentials"] = "must-not-publish"

    with pytest.raises(ValueError, match="private field"):
        research_log_event_from_dict(raw, require_digest=False)


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
            {
                **raw,
                "summary": {
                    "too_large": "x" * RESEARCH_LOG_MAX_BYTES,
                },
            },
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


def test_research_log_contract_accepts_512_kib_and_rejects_the_next_byte(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    accepted = _event_with_encoded_size(
        store,
        encoded_size=RESEARCH_LOG_MAX_BYTES,
    )
    rejected = _event_with_encoded_size(
        store,
        encoded_size=RESEARCH_LOG_MAX_BYTES + 1,
    )

    assert sign_research_log_event(accepted).event_digest
    with pytest.raises(ValueError, match="publication size limit"):
        sign_research_log_event(rejected)


def test_paged_experiment_contract_validates_digests_and_terminal_state() -> None:
    [page], manifest = _paged_terminal_experiment_view(
        _large_terminal_v3_view(pair_count=1)
    )
    assert experiment_view_page_from_dict(page.to_dict()) == page
    assert experiment_view_manifest_from_dict(manifest.to_dict()) == manifest

    with pytest.raises(ValueError, match="page digest"):
        experiment_view_page_from_dict({**page.to_dict(), "page_digest": _A})
    with pytest.raises(ValueError, match="manifest digest"):
        experiment_view_manifest_from_dict(
            {**manifest.to_dict(), "manifest_digest": _A}
        )

    common = {
        "schema_version": 1,
        "producer_event_id": "projection-1",
        "sequence": 1,
        "timestamp": "2026-08-19T12:00:00Z",
        "source": "fixture",
        "actor": {"actor_type": "service", "name": "fixture"},
        "research_id": "research-1",
        "study_id": "study-1",
        "classification": "evidence",
        "message": "Published bounded experiment evidence.",
    }
    page_event = research_log_event_from_dict(
        {
            **common,
            "state": "evaluating",
            "summary": {"experiment_view_page": page.to_dict()},
        },
        require_digest=False,
    )
    assert page_event.state == "evaluating"
    with pytest.raises(ValueError, match="pages cannot close"):
        research_log_event_from_dict(
            {
                **common,
                "state": "completed",
                "summary": {"experiment_view_page": page.to_dict()},
            },
            require_digest=False,
        )
    with pytest.raises(ValueError, match="manifests must declare a terminal"):
        research_log_event_from_dict(
            {
                **common,
                "state": "evaluating",
                "summary": {"experiment_view_manifest": manifest.to_dict()},
            },
            require_digest=False,
        )


def test_small_v3_view_remains_one_inline_event(tmp_path: Path) -> None:
    store = _store(tmp_path)
    view = _large_terminal_v3_view(pair_count=1)

    event = store.record_experiment_view_event(
        research_id="research-1",
        experiment_id="small-v3",
        producer_event_id="small-v3-result",
        classification="result",
        state="completed",
        message="Published one terminal experiment result.",
        view=view,
    )

    assert stable_digest(event.summary["experiment_view"]) == stable_digest(
        view.to_dict()
    )
    projected = [
        item for item in store.research_log_events() if item.study_id == "small-v3"
    ]
    assert projected == [event]


def test_192_attempt_v3_view_pages_recovers_and_replays_idempotently(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    view = _large_terminal_v3_view()
    pages, expected_manifest = _paged_terminal_experiment_view(view)
    assert len(pages) > 1
    producer_event_id = "large-v3-result"

    first_page = pages[0]
    first_page_event_id = "fugue:experiment-view-page:" + stable_digest(
        {"producer_event_id": producer_event_id, "page_index": 0}
    )
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        store._append_research_log_event(
            conn,
            producer_event_id=first_page_event_id,
            research_id="research-1",
            study_id="large-v3",
            classification="evidence",
            state="evaluating",
            message=f"Published terminal experiment evidence page 1 of {len(pages)}.",
            progress={"page_index": 0, "page_count": len(pages)},
            summary={"experiment_view_page": first_page.to_dict()},
        )
        conn.commit()

    manifest_event = store.record_experiment_view_event(
        research_id="research-1",
        experiment_id="large-v3",
        producer_event_id=producer_event_id,
        classification="result",
        state="completed",
        message="Published one terminal experiment result.",
        progress={"completed": 192, "total": 192},
        view=view,
    )
    assert manifest_event.summary == {
        "experiment_view_manifest": expected_manifest.to_dict()
    }
    projected = [
        item for item in store.research_log_events() if item.study_id == "large-v3"
    ]
    assert len(projected) == len(pages) + 1
    assert all(
        len(
            json.dumps(
                {
                    key: value
                    for key, value in event.to_dict().items()
                    if key != "event_digest"
                },
                separators=(",", ":"),
            ).encode()
        )
        <= RESEARCH_LOG_MAX_BYTES
        for event in projected
    )
    page_values = sorted(
        (
            experiment_view_page_from_dict(event.summary["experiment_view_page"])
            for event in projected
            if "experiment_view_page" in event.summary
        ),
        key=lambda item: item.page_index,
    )
    manifest = experiment_view_manifest_from_dict(
        manifest_event.summary["experiment_view_manifest"]
    )
    attempts = [attempt for page in page_values for attempt in page.attempts]
    pairs = [pair for page in page_values for pair in page.paired_cases]
    assert len(attempts) == manifest.attempt_count == 192
    assert len(pairs) == manifest.paired_case_count == 96
    assert len({str(item["attempt_id"]) for item in attempts}) == 192
    assert stable_digest(
        {**manifest.projection, "attempts": attempts, "paired_cases": pairs}
    ) == manifest.projection_digest
    reassembled = experiment_view_from_dict(
        {**manifest.projection, "paired_cases": pairs}
    )
    assert reassembled == view

    before = tuple(store.research_log_events())
    repeated = store.record_experiment_view_event(
        research_id="research-1",
        experiment_id="large-v3",
        producer_event_id=producer_event_id,
        classification="result",
        state="completed",
        message="Published one terminal experiment result.",
        progress={"completed": 192, "total": 192},
        view=view,
    )
    assert repeated == manifest_event
    assert tuple(store.research_log_events()) == before


def test_nonterminal_oversized_v3_view_fails_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="only terminal V3"):
        store.record_experiment_view_event(
            research_id="research-1",
            experiment_id="large-v3-running",
            producer_event_id="large-v3-running",
            classification="evidence",
            state="evaluating",
            message="Experiment is still evaluating.",
            view=_large_terminal_v3_view(),
        )
    assert all(
        item.study_id != "large-v3-running" for item in store.research_log_events()
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
    assert store.get_latest_approval_preview("research-1") == preview


def test_approval_design_is_reprojected_without_starting_an_experiment(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    preview = _preview()
    store.record_approval_request(preview, operation_id="request-approval")
    with store._connect() as conn:
        conn.execute(
            "DELETE FROM research_log_events "
            "WHERE json_extract(event_json, '$.study_id')=?",
            (preview.experiment_id,),
        )
        conn.execute(
            "DELETE FROM experiment_view_projection_state WHERE experiment_id=?",
            (preview.experiment_id,),
        )

    assert store.ensure_experiment_view_projection_events() == 1
    [projected] = [
        event
        for event in store.research_log_events()
        if event.study_id == preview.experiment_id
    ]
    assert ":approval-view-design-v13-" in projected.producer_event_id
    assert projected.state == "awaiting_approval"
    assert projected.summary["experiment_view"]["kind"] == "design"
    assert projected.summary["experiment_view"]["approval_state"] == "awaiting_approval"
    assert store.list_experiments("research-1") == ()
    assert store.ensure_experiment_view_projection_events() == 0


def test_approval_preview_recovery_backfills_an_existing_request(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    preview = _preview()
    request = store.record_approval_request(preview, operation_id="request-approval")
    with store._connect() as conn:
        conn.execute(
            "UPDATE approval_requests SET preview_json=NULL WHERE preview_digest=?",
            (preview.preview_digest,),
        )

    assert (
        store.record_approval_request(preview, operation_id="request-approval")
        == request
    )
    assert store.get_latest_approval_preview("research-1") == preview


def test_approval_preview_recovery_requires_a_durable_request(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)

    with pytest.raises(ResearchError, match="no durable approval preview"):
        store.get_latest_approval_preview("research-1")


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

    study = store.get_study("research-1")
    store.update_study(
        study.id,
        study_update_from_dict(
            {
                "message": "Record a bounded conclusion.",
                "results": [
                    {
                        "id": "result-1",
                        "statement": "The locked comparison completed.",
                        "kind": "controlled_experiment_result",
                        "conditions": {"experiment_id": completed.id},
                        "sources": [
                            {
                                "kind": "evaluation",
                                "ref": "evaluation-1",
                                "digest": _A,
                            }
                        ],
                    }
                ],
                "attribution": {"actor_type": "agent", "name": "researcher"},
            }
        ),
        operation_id="record-result",
        expected_revision=study.revision,
    )
    assert store.ensure_result_projection_events() == 1
    result_event = store.research_log_events()[-1]
    assert result_event.classification == "result"
    assert result_event.study_id == completed.id
    assert result_event.summary["result"]["statement"] == (
        "The locked comparison completed."
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
    assert all(
        ":experiment-view-" in event.producer_event_id
        and "-v13-" in event.producer_event_id
        for event in projected
    )
    assert [event.summary["experiment_view"]["kind"] for event in projected] == [
        "design",
        "evaluation",
    ]
    assert projected[-1].state == "completed"
    assert projected[-1].classification == "evidence"
    assert projected[-1].summary["passed"] == 2
    assert (
        projected[-1].summary["experiment_view"]["infrastructure_health"]
        == "unavailable"
    )
    assert projected[-1].observed_cost_usd == 1.53
    assert all(
        not any(link["system"] == "weave" for link in cell["evidence_links"])
        for cell in projected[-1].summary["experiment_view"]["cells"]
    )
    assert "private limitation text" not in json.dumps(
        [event.to_dict() for event in projected]
    )
    assert store.ensure_experiment_view_projection_events() == 0

    # A fresh checkout may no longer have the private export mount. A
    # content-addressed public bundle reconstructs only normalized safe fields.
    export_path.unlink()
    bundle_path = (
        tmp_path
        / "configs"
        / "fugue"
        / "public-exports"
        / f"{hashlib.sha256(export_payload).hexdigest()}.json"
    )
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_export_sha256": hashlib.sha256(export_payload).hexdigest(),
                "source_evidence": {
                    "project": "team/reviewed-source",
                    "selected_call_ids": ["source-call-1"],
                },
                "rows": [
                    {
                        **row,
                        "private_prompt": "must never enter the projection",
                    }
                    for row in export_rows
                ],
            }
        ),
        encoding="utf-8",
    )
    with store._connect() as conn:
        conn.execute(
            "DELETE FROM research_log_events "
            "WHERE producer_event_id LIKE '%:experiment-view-%'"
        )
        conn.execute(
            "DELETE FROM experiment_view_projection_state WHERE experiment_id=?",
            (record.id,),
        )
    assert store.ensure_experiment_view_projection_events() == 2
    replayed = [
        event for event in store.research_log_events() if event.study_id == record.id
    ]
    assert "private_prompt" not in json.dumps(
        [event.to_dict() for event in replayed]
    )
    assert all(
        not any(link["system"] == "weave" for link in cell["evidence_links"])
        for cell in replayed[-1].summary["experiment_view"]["cells"]
    )
    assert any(
        link["kind"] == "source_call"
        and link["ref"] == "team/reviewed-source/call/source-call-1"
        for link in replayed[-1].summary["experiment_view"]["evidence_links"]
    )
    mismatched = json.loads(bundle_path.read_text(encoding="utf-8"))
    mismatched["rows"][0]["candidate_id"] = "different-candidate"
    bundle_path.write_text(json.dumps(mismatched), encoding="utf-8")
    assert (
        store._portable_projection_rows(
            completed,
            source_export_sha256=hashlib.sha256(export_payload).hexdigest(),
        )
        == ()
    )

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


def test_operator_replay_republishes_delivered_events_without_new_operations(
    tmp_path: Path,
) -> None:
    class RecordingSink:
        sink_id = "projection-console"

        def __init__(self) -> None:
            self.events = []

        def publish(self, event: object) -> None:
            self.events.append(event)

    store = _store(tmp_path)
    store.record_approval_request(_preview(), operation_id="request-1")
    before = store.get_study("research-1")
    sink = RecordingSink()
    publisher = ResearchRecordPublisher(store, [sink])

    assert publisher.flush() == {"delivered": 2, "failed": 0}
    sink.events.clear()
    assert publisher.replay(research_id="research-1") == {
        "delivered": 2,
        "failed": 0,
    }

    assert [event.sequence for event in sink.events] == [1, 2]
    assert store.get_study("research-1") == before
    assert store.research_publication_status()["event_count"] == 2


def test_publication_scope_filters_live_flush_and_operator_replay(
    tmp_path: Path,
) -> None:
    class RecordingSink:
        sink_id = "projection-console"

        def __init__(self) -> None:
            self.events = []

        def publish(self, event: object) -> None:
            self.events.append(event)

    store = _store(tmp_path)
    store.create_study(
        study_id="research-2",
        title="Another Study",
        campaign_id="campaign-1",
        question="Another question",
        operation_id="create-research-2",
    )
    sink = RecordingSink()
    publisher = ResearchRecordPublisher(
        store,
        [sink],
        research_ids=("research-1",),
    )

    assert publisher.flush() == {"delivered": 1, "failed": 0}
    assert [event.research_id for event in sink.events] == ["research-1"]
    sink.events.clear()
    assert publisher.replay() == {"delivered": 1, "failed": 0}
    assert [event.research_id for event in sink.events] == ["research-1"]
    with pytest.raises(ValueError, match="outside the configured scope"):
        publisher.replay(research_id="research-2")


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
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201)

    transport = httpx.MockTransport(handler)
    original_client = httpx.Client
    monkeypatch.setattr(
        "fugue.research.records.httpx.Client",
        lambda **kwargs: original_client(transport=transport, **kwargs),
    )
    event = _store(tmp_path).research_log_events()[0]
    sink = HttpResearchRecordSink(
        "http://127.0.0.1:3000/api/research-log-events",
        "ingest-secret",
        timeout=3,
    )
    sink.publish(event)

    request = requests[0]
    assert request.headers["Authorization"] == "Bearer ingest-secret"
    assert request.headers["Idempotency-Key"] == event.producer_event_id
    assert json.loads(request.content)["event_digest"] == event.event_digest


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/events",
        "https://user:secret@example.com/ingest",
        "https://example.com/ingest#fragment",
        "https:///missing-host",
    ],
)
def test_http_sink_rejects_ambiguous_or_credentialed_urls(url: str) -> None:
    with pytest.raises(ValueError, match="must use http or https"):
        HttpResearchRecordSink(url, "ingest-secret")
