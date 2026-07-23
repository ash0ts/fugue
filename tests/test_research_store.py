from __future__ import annotations

from pathlib import Path

import pytest

from fugue.research.contracts import (
    ResearchError,
    study_from_dict,
    study_update_from_dict,
)
from fugue.research.store import StudyStore

_DIGEST = "a" * 64


def _store(tmp_path: Path) -> StudyStore:
    return StudyStore(tmp_path)


def _create(store: StudyStore) -> None:
    store.create_study(
        study_id="study-1",
        title="Harness behavior",
        campaign_id="campaign-1",
        question="Which loop components change outcomes?",
        operation_id="create-study-1",
    )


def test_create_is_idempotent_and_rejects_changed_input(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _create(store)
    repeated = store.create_study(
        study_id="study-1",
        title="Harness behavior",
        campaign_id="campaign-1",
        question="Which loop components change outcomes?",
        operation_id="create-study-1",
    )
    assert repeated.revision == 1
    with pytest.raises(ResearchError, match="different input"):
        store.create_study(
            study_id="study-1",
            title="Changed",
            campaign_id="campaign-1",
            question="Which loop components change outcomes?",
            operation_id="create-study-1",
        )


def test_study_schema_rejects_unknown_fields_and_digest_tampering(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _create(store)
    raw = store.get_study("study-1").to_dict()
    with pytest.raises(ValueError, match="unknown field"):
        study_from_dict({**raw, "winner": "harness-a"})
    with pytest.raises(ValueError, match="study_digest"):
        study_from_dict({**raw, "title": "tampered"})


def test_notes_do_not_rewrite_brief_and_results_require_provenance(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _create(store)
    note_update = study_update_from_dict(
        {
            "message": "The first branch was inconclusive.",
            "attribution": {"actor_type": "agent", "name": "outer-loop"},
        }
    )
    study = store.update_study(
        "study-1",
        note_update,
        operation_id="note-1",
        expected_revision=1,
    )
    assert study.brief.findings == ""
    assert study.notes[0].text == "The first branch was inconclusive."

    result_update = study_update_from_dict(
        {
            "results": [
                {
                    "id": "result-1",
                    "statement": "Harness A and B differed on the locked tasks.",
                    "kind": "contrast",
                    "outcome": "observed difference",
                    "estimate": {"value": 0.25, "kind": "pass-rate difference"},
                    "population": "four locked tasks",
                    "conditions": {"attempts": 2},
                    "sample_size": 8,
                    "aggregation": "task aligned",
                    "exclusions": ["not a universal ranking"],
                    "sources": [
                        {"kind": "analysis", "ref": "analysis-1", "digest": _DIGEST}
                    ],
                    "analysis_source": {
                        "kind": "analysis",
                        "ref": "analysis-1",
                        "digest": _DIGEST,
                    },
                }
            ],
            "brief_patch": {"findings": "The locked comparison differed."},
            "brief_sources": {
                "findings": [
                    {"kind": "analysis", "ref": "analysis-1", "digest": _DIGEST}
                ]
            },
        }
    )
    study = store.update_study(
        "study-1", result_update, operation_id="result-1", expected_revision=2
    )
    assert study.results[0].sources[0].digest == _DIGEST
    assert study.brief.findings == "The locked comparison differed."
    assert store.context("study-1").results[0]["id"] == "result-1"
    assert store.reconstruct_study("study-1", revision=2).brief.findings == ""
    assert store.reconstruct_study("study-1") == study


def test_revision_conflicts_and_immutable_result_ids(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _create(store)
    update = study_update_from_dict({"message": "one note"})
    store.update_study("study-1", update, operation_id="note-1", expected_revision=1)
    with pytest.raises(ResearchError, match="revision is 2"):
        store.update_study(
            "study-1", update, operation_id="note-2", expected_revision=1
        )


def test_context_is_bounded_and_marks_omissions(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _create(store)
    for index in range(8):
        store.update_study(
            "study-1",
            study_update_from_dict({"message": f"note {index} " + "x" * 200}),
            operation_id=f"note-{index}",
        )
    context = store.context("study-1", max_notes=8, max_chars=1800)
    assert context.omissions["notes"] > 0
    assert len(str(context.to_dict())) < 2400
