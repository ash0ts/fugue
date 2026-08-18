from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import fugue.bench.research_index as research_index_module
from fugue.bench.candidates import stable_digest
from fugue.bench.files import atomic_write_json
from fugue.bench.local_publication import (
    StudyPublicationScopeV1,
    WeaveHostedObjectRefV1,
    WeavePublicationReceiptV1,
    WeavePublicationTargetV1,
)
from fugue.bench.research_index import (
    ResearchIndexError,
    ResearchIndexPublicationOutcomeV1,
    ResearchIndexPublicationTargetV1,
    ResearchIndexSourceV1,
    build_research_index,
    publish_research_index,
    read_research_index,
    read_research_index_publication_receipt,
    write_research_index,
)

_KINDS = (
    "agent_evidence_receipt",
    "dataset",
    "evaluation_root",
    "prediction",
    "prediction_and_score",
)


def _publication_target(
    project: str = "wandb/community-studies",
    *,
    api_base_url: str = "https://api.wandb.ai",
    app_base_url: str = "https://wandb.ai",
) -> ResearchIndexPublicationTargetV1:
    return ResearchIndexPublicationTargetV1(
        project=project,
        api_base_url=api_base_url,
        app_base_url=app_base_url,
    )


@dataclass(frozen=True)
class _FakeAttempt:
    attempt_id: str
    identity: dict[str, str]


@dataclass(frozen=True)
class _FakePair:
    baseline: _FakeAttempt
    candidate: _FakeAttempt
    harness: str = "claude-code"


@dataclass(frozen=True)
class _FakeComparisonResultV3:
    comparison_id: str
    result_digest: str
    qualification_digest: str
    rows: int
    paired_cases: tuple[_FakePair, ...]
    candidate_definitions: dict[str, dict[str, str]]
    behavioral_summary: SimpleNamespace
    decision: SimpleNamespace
    evidence_backend: str = "local"
    local_chain_integrity: str = "reconciled"
    hosted_chain_integrity: str = "not_applicable"


def _result(comparison_id: str) -> _FakeComparisonResultV3:
    baseline_definition = {"arm": "baseline", "study": comparison_id}
    candidate_definition = {"arm": "candidate", "study": comparison_id}
    baseline_id = stable_digest(baseline_definition)
    candidate_id = stable_digest(candidate_definition)
    return _FakeComparisonResultV3(
        comparison_id=comparison_id,
        result_digest=stable_digest({"result": comparison_id}),
        qualification_digest=stable_digest({"qualification": comparison_id}),
        rows=2,
        paired_cases=(
            _FakePair(
                baseline=_FakeAttempt(
                    stable_digest({"attempt": comparison_id + "-b"}),
                    {"candidate": baseline_id, "harness": "claude-code"},
                ),
                candidate=_FakeAttempt(
                    stable_digest({"attempt": comparison_id + "-c"}),
                    {"candidate": candidate_id, "harness": "claude-code"},
                ),
            ),
        ),
        candidate_definitions={
            baseline_id: baseline_definition,
            candidate_id: candidate_definition,
        },
        behavioral_summary=SimpleNamespace(
            status="unchanged",
            recommendation="Keep the current revision pending a stronger result.",
        ),
        decision=SimpleNamespace(evidence_grade="A"),
    )


def _write_result(path: Path, result: _FakeComparisonResultV3) -> bytes:
    payload = {
        "schema_version": 3,
        "comparison_id": result.comparison_id,
        "result_digest": result.result_digest,
        "qualification_digest": result.qualification_digest,
    }
    atomic_write_json(path, payload)
    return path.read_bytes()


def _hosted_objects(
    result: _FakeComparisonResultV3,
    target: WeavePublicationTargetV1,
    *,
    kinds: tuple[str, ...] = _KINDS,
) -> tuple[WeaveHostedObjectRefV1, ...]:
    values = []
    for pair in result.paired_cases:
        for attempt in (pair.baseline, pair.candidate):
            for kind in kinds:
                object_id = f"{attempt.attempt_id[:12]}-{kind}"
                object_type = "object" if kind == "dataset" else "call"
                values.append(
                    WeaveHostedObjectRefV1(
                        attempt_id=attempt.attempt_id,
                        kind=kind,
                        target=target,
                        object_id=object_id,
                        ref=(
                            f"weave:///{target.project_slug}/{object_type}/"
                            f"{object_id}"
                        ),
                    )
                )
    return tuple(sorted(values, key=lambda item: (item.attempt_id, item.kind)))


def _write_receipt(
    path: Path,
    *,
    result: _FakeComparisonResultV3,
    result_bytes: bytes,
    research_id: str = "skill-upgrade-research-v1",
    study_id: str | None = None,
    project: str = "skill-upgrade-results-v1",
    qualification_digest: str | None = None,
    result_digest: str | None = None,
    result_file_sha256: str | None = None,
    kinds: tuple[str, ...] = _KINDS,
) -> WeavePublicationReceiptV1:
    target = WeavePublicationTargetV1(
        entity="wandb",
        project=project,
        study_scope=StudyPublicationScopeV1(
            research_id=research_id,
            study_id=study_id or result.comparison_id,
        ),
    )
    selected_result_digest = result_digest or result.result_digest
    manifest_digest = stable_digest({"manifest": result.comparison_id})
    receipt = WeavePublicationReceiptV1(
        publication_id=stable_digest(
            {
                "schema_version": 1,
                "target": target.to_dict(),
                "result_digest": selected_result_digest,
                "local_manifest_digest": manifest_digest,
            }
        ),
        target=target,
        result_digest=selected_result_digest,
        qualification_digest=(qualification_digest or result.qualification_digest),
        result_file_sha256=(
            result_file_sha256 or hashlib.sha256(result_bytes).hexdigest()
        ),
        local_manifest_digest=manifest_digest,
        local_manifest_file_sha256=stable_digest(
            {"manifest-file": result.comparison_id}
        ),
        hosted_objects=_hosted_objects(result, target, kinds=kinds),
        publisher_id="fake-weave-publisher",
        publisher_revision="test-v1",
        status="published",
        published_at="2026-08-18T12:00:00+00:00",
    )
    atomic_write_json(path, receipt.to_dict())
    return receipt


def _source_pair(
    tmp_path: Path,
    *,
    comparison_id: str,
    research_id: str = "skill-upgrade-research-v1",
    study_id: str | None = None,
    **receipt_changes,
) -> tuple[ResearchIndexSourceV1, _FakeComparisonResultV3]:
    result = _result(comparison_id)
    result_path = tmp_path / f"{comparison_id}-result.json"
    receipt_path = tmp_path / f"{comparison_id}-receipt.json"
    result_bytes = _write_result(result_path, result)
    _write_receipt(
        receipt_path,
        result=result,
        result_bytes=result_bytes,
        research_id=research_id,
        study_id=study_id,
        **receipt_changes,
    )
    return (
        ResearchIndexSourceV1(
            result_path=result_path,
            publication_receipt_path=receipt_path,
        ),
        result,
    )


def _patch_results(
    monkeypatch: pytest.MonkeyPatch,
    sources: list[tuple[ResearchIndexSourceV1, _FakeComparisonResultV3]],
) -> None:
    monkeypatch.setattr(
        research_index_module,
        "ComparisonResultV3",
        _FakeComparisonResultV3,
    )
    by_id = {result.comparison_id: result for _source, result in sources}
    monkeypatch.setattr(
        research_index_module,
        "comparison_result_from_json",
        lambda payload: by_id[json.loads(payload)["comparison_id"]],
    )


def _index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source_b = _source_pair(tmp_path, comparison_id="study-b")
    source_a = _source_pair(tmp_path, comparison_id="study-a")
    sources = [source_b, source_a]
    _patch_results(monkeypatch, sources)
    result_bytes = [item[0].result_path.read_bytes() for item in sources]
    receipt_bytes = [
        item[0].publication_receipt_path.read_bytes() for item in sources
    ]
    index = build_research_index(
        research_id="skill-upgrade-research-v1",
        title="Public Skill upgrade evidence",
        objective="Compare exact revisions without pooling unlike tasks.",
        sources=[item[0] for item in sources],
    )
    assert [item[0].result_path.read_bytes() for item in sources] == result_bytes
    assert [
        item[0].publication_receipt_path.read_bytes() for item in sources
    ] == receipt_bytes
    return index


def test_build_write_and_read_research_index_is_canonical_and_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _index(tmp_path, monkeypatch)

    assert [item.study_id for item in index.studies] == ["study-a", "study-b"]
    assert index.study_count == 2
    assert index.total_rows == 4
    assert {item.project for item in index.studies} == {
        "wandb/skill-upgrade-results-v1"
    }
    assert all(item.published_chain_integrity == "reconciled" for item in index.studies)
    assert all(
        item.recommendation
        == "Keep the current revision pending a stronger result."
        for item in index.studies
    )
    assert all(len(item.evidence_refs) == 10 for item in index.studies)
    assert all(len(item.candidate_ids) == 2 for item in index.studies)
    assert all(
        [(item.role, item.harness) for item in study.candidate_assignments]
        == [("baseline", "claude-code"), ("candidate", "claude-code")]
        for study in index.studies
    )
    for study in index.studies:
        assert hashlib.sha256(study.result_json.encode()).hexdigest() == (
            study.result_file_sha256
        )
        assert hashlib.sha256(
            study.publication_receipt_json.encode()
        ).hexdigest() == study.publication_receipt_file_sha256

    path = tmp_path / "research-index.json"
    write_research_index(path, index)
    before = path.read_bytes()
    assert read_research_index(path) == index
    assert write_research_index(path, index) == path
    assert path.read_bytes() == before

    changed = json.loads(path.read_text())
    changed["title"] = "A conflicting title"
    path.write_text(json.dumps(changed))
    with pytest.raises(ResearchIndexError, match="conflicting immutable"):
        write_research_index(path, index)


def test_build_rejects_duplicate_study_ids_and_scope_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _source_pair(tmp_path, comparison_id="first", study_id="shared")
    second = _source_pair(tmp_path, comparison_id="second", study_id="shared")
    _patch_results(monkeypatch, [first, second])
    with pytest.raises(ResearchIndexError, match="duplicate Study"):
        build_research_index(
            research_id="skill-upgrade-research-v1",
            title="Skill studies",
            objective="Compare revisions.",
            sources=[first[0], second[0]],
        )

    wrong_scope = _source_pair(
        tmp_path,
        comparison_id="wrong-scope",
        research_id="another-research",
    )
    _patch_results(monkeypatch, [wrong_scope])
    with pytest.raises(ResearchIndexError, match="different Research scope"):
        build_research_index(
            research_id="skill-upgrade-research-v1",
            title="Skill studies",
            objective="Compare revisions.",
            sources=[wrong_scope[0]],
        )


def test_build_requires_typed_sources(
    tmp_path: Path,
) -> None:
    with pytest.raises(ResearchIndexError, match="ResearchIndexSourceV1"):
        build_research_index(
            research_id="skill-upgrade-research-v1",
            title="Skill studies",
            objective="Compare revisions.",
            sources=[(tmp_path / "result.json", tmp_path / "receipt.json")],  # type: ignore[list-item]
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"result_digest": "f" * 64}, "result digest disagrees"),
        ({"qualification_digest": "e" * 64}, "qualification digest disagrees"),
        ({"result_file_sha256": "d" * 64}, "exact result file bytes"),
    ],
)
def test_build_rejects_result_receipt_binding_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: dict[str, str],
    message: str,
) -> None:
    source = _source_pair(
        tmp_path,
        comparison_id="mismatch",
        **change,
    )
    _patch_results(monkeypatch, [source])
    with pytest.raises(ResearchIndexError, match=message):
        build_research_index(
            research_id="skill-upgrade-research-v1",
            title="Skill studies",
            objective="Compare revisions.",
            sources=[source[0]],
        )


def test_build_rejects_incomplete_evidence_chain_and_non_v3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incomplete = _source_pair(
        tmp_path,
        comparison_id="incomplete-chain",
        kinds=tuple(kind for kind in _KINDS if kind != "dataset"),
    )
    _patch_results(monkeypatch, [incomplete])
    with pytest.raises(ResearchIndexError, match="five-object chain"):
        build_research_index(
            research_id="skill-upgrade-research-v1",
            title="Skill studies",
            objective="Compare revisions.",
            sources=[incomplete[0]],
        )

    monkeypatch.setattr(
        research_index_module,
        "comparison_result_from_json",
        lambda _payload: object(),
    )
    monkeypatch.setattr(
        research_index_module,
        "ComparisonResultV3",
        research_index_module.ComparisonResultV3,
    )
    with pytest.raises(ResearchIndexError, match="ComparisonResultV3"):
        build_research_index(
            research_id="skill-upgrade-research-v1",
            title="Skill studies",
            objective="Compare revisions.",
            sources=[incomplete[0]],
        )


def test_read_rejects_digest_drift_and_unknown_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _index(tmp_path, monkeypatch)
    path = tmp_path / "research-index.json"
    write_research_index(path, index)
    value = json.loads(path.read_text())
    value["index_digest"] = "0" * 64
    atomic_write_json(path, value)
    with pytest.raises(ValueError, match="digest does not match"):
        read_research_index(path)

    value = index.to_dict()
    value["unreviewed_claim"] = True
    atomic_write_json(path, value)
    with pytest.raises(ValueError, match="unknown unreviewed_claim"):
        read_research_index(path)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("result_json", "comparison result source bytes disagree"),
        (
            "publication_receipt_json",
            "publication receipt source bytes disagree",
        ),
    ],
)
def test_read_rejects_tampered_embedded_source_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    message: str,
) -> None:
    index = _index(tmp_path, monkeypatch)
    raw = index.to_dict()
    raw["studies"][0][field] = '{"tampered":true}\n'
    raw["index_digest"] = stable_digest(
        {key: value for key, value in raw.items() if key != "index_digest"}
    )
    path = tmp_path / "tampered-source.json"
    atomic_write_json(path, raw)

    with pytest.raises(ValueError, match=message):
        read_research_index(path)


def test_read_rejects_candidate_assignment_to_undefined_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _index(tmp_path, monkeypatch)
    raw = index.to_dict()
    raw["studies"][0]["candidate_assignments"][0]["candidate_id"] = "f" * 64
    raw["index_digest"] = stable_digest(
        {key: value for key, value in raw.items() if key != "index_digest"}
    )
    path = tmp_path / "undefined-candidate.json"
    atomic_write_json(path, raw)

    with pytest.raises(ValueError, match="reference exactly the defined candidates"):
        read_research_index(path)


def test_read_rejects_rehashed_embedded_result_identity_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _index(tmp_path, monkeypatch)
    raw = index.to_dict()
    embedded = json.loads(raw["studies"][0]["result_json"])
    embedded["comparison_id"] = "different-study"
    replacement = json.dumps(embedded, indent=2, sort_keys=True) + "\n"
    raw["studies"][0]["result_json"] = replacement
    raw["studies"][0]["result_file_sha256"] = hashlib.sha256(
        replacement.encode()
    ).hexdigest()
    raw["index_digest"] = stable_digest(
        {key: value for key, value in raw.items() if key != "index_digest"}
    )
    path = tmp_path / "rehashed-result-conflict.json"
    atomic_write_json(path, raw)

    with pytest.raises(ValueError, match="canonical validation"):
        read_research_index(path)


@pytest.mark.parametrize(
    "projection",
    [
        "behavioral_status",
        "recommendation",
        "candidate_assignments",
        "evidence_refs",
        "study_id",
        "project",
        "research_id",
    ],
)
def test_read_rederives_every_index_projection_from_embedded_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    projection: str,
) -> None:
    index = _index(tmp_path, monkeypatch)
    raw = index.to_dict()
    study = raw["studies"][0]
    if projection == "behavioral_status":
        study[projection] = "improved"
    elif projection == "recommendation":
        study[projection] = "A claim not present in the canonical result."
    elif projection == "candidate_assignments":
        first, second = study[projection]
        first["candidate_id"], second["candidate_id"] = (
            second["candidate_id"],
            first["candidate_id"],
        )
    elif projection == "evidence_refs":
        study[projection][0]["ref"] += "-different"
    elif projection == "project":
        study[projection] = "wandb/different-project"
        for ref in study["evidence_refs"]:
            ref["ref"] = ref["ref"].replace(
                "wandb/skill-upgrade-results-v1",
                "wandb/different-project",
            )
    elif projection == "research_id":
        raw["research_id"] = "different-research-v1"
        for item in raw["studies"]:
            item["research_id"] = "different-research-v1"
    else:
        study[projection] = "different-study-v1"
    raw["index_digest"] = stable_digest(
        {key: value for key, value in raw.items() if key != "index_digest"}
    )
    path = tmp_path / f"rehashed-{projection}.json"
    atomic_write_json(path, raw)

    with pytest.raises(
        ValueError,
        match="projection disagrees|canonical sources disagree",
    ):
        read_research_index(path)


def test_read_rejects_rehashed_nested_sensitive_key_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _index(tmp_path, monkeypatch)
    raw = index.to_dict()
    study = raw["studies"][0]
    embedded = json.loads(study["result_json"])
    embedded["candidate_payload"] = {
        "safe": {"nested": {"api_key": "not-allowed-in-an-index"}}
    }
    replacement = json.dumps(embedded, indent=2, sort_keys=True) + "\n"
    study["result_json"] = replacement
    study["result_file_sha256"] = hashlib.sha256(replacement.encode()).hexdigest()
    raw["index_digest"] = stable_digest(
        {key: value for key, value in raw.items() if key != "index_digest"}
    )
    path = tmp_path / "rehashed-sensitive-key.json"
    atomic_write_json(path, raw)

    with pytest.raises(ValueError, match=r"sensitive key .*api_key"):
        read_research_index(path)


def test_build_rejects_conflicting_candidate_treatment_coordinate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, result = _source_pair(tmp_path, comparison_id="conflicting-treatment")
    replacement = {"arm": "replacement", "study": result.comparison_id}
    replacement_id = stable_digest(replacement)
    second_pair = _FakePair(
        baseline=_FakeAttempt(
            stable_digest({"attempt": result.comparison_id + "-replacement-b"}),
            {"candidate": replacement_id, "harness": "claude-code"},
        ),
        candidate=_FakeAttempt(
            stable_digest({"attempt": result.comparison_id + "-replacement-c"}),
            result.paired_cases[0].candidate.identity,
        ),
    )
    conflicting = _FakeComparisonResultV3(
        **{
            **result.__dict__,
            "rows": 4,
            "paired_cases": (*result.paired_cases, second_pair),
            "candidate_definitions": {
                **result.candidate_definitions,
                replacement_id: replacement,
            },
        }
    )
    result_bytes = source.result_path.read_bytes()
    _write_receipt(
        source.publication_receipt_path,
        result=conflicting,
        result_bytes=result_bytes,
    )
    _patch_results(monkeypatch, [(source, conflicting)])

    with pytest.raises(
        ResearchIndexError,
        match="coordinate maps to multiple candidates",
    ):
        build_research_index(
            research_id="skill-upgrade-research-v1",
            title="Skill studies",
            objective="Compare revisions.",
            sources=[source],
        )


def test_build_rejects_configured_and_secret_shaped_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_pair(tmp_path, comparison_id="secret-check")
    _patch_results(monkeypatch, [source])
    configured_secret = "shared-secret-value-123"
    with pytest.raises(ResearchIndexError, match="configured secret"):
        build_research_index(
            research_id="skill-upgrade-research-v1",
            title=f"Study {configured_secret}",
            objective="Compare revisions.",
            sources=[source[0]],
            secret_values=(configured_secret,),
        )
    with pytest.raises(ResearchIndexError, match="secret-shaped"):
        build_research_index(
            research_id="skill-upgrade-research-v1",
            title="Skill studies",
            objective="Compare revisions with api_key=definitely-not-public.",
            sources=[source[0]],
        )


def test_optional_publication_is_digest_bound_idempotent_and_preserves_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _index(tmp_path, monkeypatch)
    index_path = tmp_path / "research-index.json"
    write_research_index(index_path, index)
    before = index_path.read_bytes()
    calls = []

    target = _publication_target()

    def publisher(observed_index, observed_bytes, observed_target):
        calls.append((observed_index, observed_bytes, observed_target))
        return ResearchIndexPublicationOutcomeV1(
            target=observed_target,
            run_url="https://wandb.ai/wandb/community-studies/runs/index-v1",
            artifact_url=(
                "https://wandb.ai/wandb/community-studies/artifacts/"
                "research-index/v1"
            ),
            report_url="https://wandb.ai/wandb/community-studies/reports/v1",
            report_status="published",
            publisher_id="fake-wandb-index-publisher",
            publisher_revision="test-v1",
        )

    receipt = publish_research_index(
        index_path,
        target=target,
        publisher=publisher,
        clock=lambda: datetime(2026, 8, 18, 12, 0, tzinfo=UTC),
    )
    assert len(calls) == 1
    assert calls[0][0] == index
    assert calls[0][1] == before
    assert index_path.read_bytes() == before
    assert receipt.index_digest == index.index_digest
    assert receipt.index_file_sha256 == hashlib.sha256(before).hexdigest()

    second = publish_research_index(
        index_path,
        target=target,
        publisher=lambda *_args: pytest.fail("idempotent publication called publisher"),
    )
    assert second == receipt
    receipt_path = tmp_path / "research-index-publication-receipt.json"
    assert read_research_index_publication_receipt(receipt_path) == receipt


def test_optional_publication_rejects_changed_project_and_index_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _index(tmp_path, monkeypatch)
    index_path = tmp_path / "research-index.json"
    write_research_index(index_path, index)

    target = _publication_target()
    with pytest.raises(ResearchIndexError, match="different destination"):
        publish_research_index(
            index_path,
            target=target,
            publisher=lambda *_args: ResearchIndexPublicationOutcomeV1(
                target=_publication_target("wandb/another-project"),
                run_url="https://wandb.ai/wandb/another-project/runs/index-v1",
                artifact_url="https://wandb.ai/wandb/another-project/artifacts/v1",
                report_url=None,
                report_status="unavailable",
                publisher_id="fake",
                publisher_revision="test-v1",
            ),
        )

    def mutating_publisher(_index, _bytes, _target):
        index_path.write_text("{}")
        return ResearchIndexPublicationOutcomeV1(
            target=target,
            run_url="https://wandb.ai/wandb/community-studies/runs/index-v1",
            artifact_url="https://wandb.ai/wandb/community-studies/artifacts/v1",
            report_url=None,
            report_status="unavailable",
            publisher_id="fake",
            publisher_revision="test-v1",
        )

    with pytest.raises(ResearchIndexError, match="changed while building"):
        publish_research_index(
            index_path,
            target=target,
            publisher=mutating_publisher,
        )
