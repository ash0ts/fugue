from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import fugue.bench.research_report as report_module
from fugue.bench.candidates import stable_digest
from fugue.bench.files import atomic_write_json
from fugue.bench.research_index import (
    ResearchCandidateAssignmentV1,
    ResearchIndexPublicationTargetV1,
)
from fugue.bench.research_report import (
    RESEARCH_INDEX_REPORT_WARNING,
    ResearchIndexReportError,
    ResearchIndexReportPublicationOutcomeV1,
    build_research_index_report_projection,
    publish_research_index_report,
    read_research_index_report_publication_receipt,
)


def _sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    index_path = tmp_path / "research-index.json"
    index_receipt_path = tmp_path / "research-index-publication-receipt.json"
    atomic_write_json(index_path, {"kind": "canonical-index"})
    atomic_write_json(index_receipt_path, {"kind": "canonical-index-receipt"})
    baseline = stable_digest({"candidate": "baseline"})
    candidate = stable_digest({"candidate": "candidate"})
    attempt = stable_digest({"attempt": "one"})
    study = SimpleNamespace(
        study_id="study-a",
        comparison_id="exact-revision-comparison",
        project="wandb/study-results",
        result_digest=stable_digest({"result": "study-a"}),
        qualification_digest=stable_digest({"qualification": "study-a"}),
        behavioral_status="unchanged",
        behavioral_recommendation=(
            "Keep the current revision until evidence improves."
        ),
        decision_status="inconclusive",
        decision_recommendation="Package release was not evaluated.",
        task_validity_status="valid",
        rows=2,
        evidence_integrity_grade="A",
        evidence_backend="weave",
        local_chain_integrity="reconciled",
        result_hosted_chain_integrity="reconciled",
        published_chain_integrity="reconciled",
        candidate_assignments=(
            ResearchCandidateAssignmentV1(
                role="baseline", harness="claude-code", candidate_id=baseline
            ),
            ResearchCandidateAssignmentV1(
                role="candidate", harness="claude-code", candidate_id=candidate
            ),
        ),
        evidence_refs=(
            SimpleNamespace(
                attempt_id=attempt,
                kind="prediction_and_score",
                ref="weave:///wandb/study-results/call/predict-and-score-1",
            ),
        ),
    )
    index = SimpleNamespace(
        research_id="community-research-v1",
        title="Community Skill evidence",
        objective="Compare exact revisions without pooling unlike tasks.",
        index_digest=stable_digest({"index": "community-research-v1"}),
        studies=(study,),
        study_count=1,
        total_rows=2,
    )
    target = ResearchIndexPublicationTargetV1(
        project="wandb/research-index",
        api_base_url="https://api.wandb.ai",
        app_base_url="https://wandb.ai",
    )
    receipt = SimpleNamespace(
        research_id=index.research_id,
        index_digest=index.index_digest,
        index_file_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
        status="published",
        publication_id=stable_digest({"publication": "index"}),
        receipt_digest=stable_digest({"receipt": "index"}),
        target=target,
        run_url="https://wandb.ai/wandb/research-index/runs/index-run-1",
        artifact_url=(
            "https://wandb.ai/wandb/research-index/artifacts/research-index/v1"
        ),
    )
    monkeypatch.setattr(report_module, "read_research_index", lambda _path: index)
    monkeypatch.setattr(
        report_module,
        "read_research_index_publication_receipt",
        lambda _path: receipt,
    )
    return index_path, index_receipt_path


def _outcome(
    projection,
    *,
    report_id: str = "report-123",
    report_url: str | None = None,
) -> ResearchIndexReportPublicationOutcomeV1:
    return ResearchIndexReportPublicationOutcomeV1(
        target=projection.target,
        report_id=report_id,
        report_url=report_url
        or (
            "https://wandb.ai/wandb/research-index/reports/"
            f"community-research--{report_id}"
        ),
        report_api=projection.report_api,
        report_api_version=projection.report_api_version,
        api_stability=projection.api_stability,
        readback_projection_digest=projection.projection_digest,
        rendered_content_digest=stable_digest(
            {"rendered": projection.projection_digest}
        ),
        readback_status="reconciled",
        publisher_id="fake-wandb-report-publisher",
        publisher_revision="test-v1",
    )


def test_projection_binds_exact_sources_and_keeps_studies_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_path, index_receipt_path = _sources(tmp_path, monkeypatch)
    index_before = index_path.read_bytes()
    receipt_before = index_receipt_path.read_bytes()

    projection = build_research_index_report_projection(
        index_path,
        index_receipt_path,
    )

    assert projection.research_id == "community-research-v1"
    assert projection.study_count == 1
    assert projection.total_rows == 2
    assert [item.study_id for item in projection.studies] == ["study-a"]
    assert projection.report_title.endswith(projection.projection_digest[:12])
    assert RESEARCH_INDEX_REPORT_WARNING in projection.report_description
    assert projection.api_stability == "public_preview"
    assert projection.warning == RESEARCH_INDEX_REPORT_WARNING
    assert projection.studies[0].primary_evidence_url.endswith(
        "/weave/calls/predict-and-score-1"
    )
    assert index_path.read_bytes() == index_before
    assert index_receipt_path.read_bytes() == receipt_before


def test_publish_writes_immutable_receipt_and_reruns_live_readback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_path, index_receipt_path = _sources(tmp_path, monkeypatch)
    calls: list[str] = []

    def publisher(
        projection,
        *,
        expected_report_id: str | None = None,
        expected_report_url: str | None = None,
    ):
        calls.append(projection.projection_digest)
        if expected_report_id is None:
            assert expected_report_url is None
            return _outcome(projection)
        assert expected_report_url is not None
        return _outcome(
            projection,
            report_id=expected_report_id,
            report_url=expected_report_url,
        )

    receipt_path = tmp_path / "report-receipt.json"
    receipt = publish_research_index_report(
        index_path,
        index_receipt_path,
        publisher,
        receipt_path=receipt_path,
        clock=lambda: datetime(2026, 8, 18, 12, 0, tzinfo=UTC),
    )
    before = receipt_path.read_bytes()

    assert receipt.status == "published_and_reconciled"
    assert receipt.access_mode == "project_settings"
    assert receipt.share_link_action == "not_requested"
    assert receipt.current_pointer_status == "not_managed"
    assert receipt.readback_projection_digest == receipt.projection_digest
    assert read_research_index_report_publication_receipt(receipt_path) == receipt

    repeated = publish_research_index_report(
        index_path,
        index_receipt_path,
        publisher,
        receipt_path=receipt_path,
    )
    assert repeated == receipt
    assert len(calls) == 2
    assert receipt_path.read_bytes() == before


def test_existing_receipt_rejects_changed_live_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_path, index_receipt_path = _sources(tmp_path, monkeypatch)
    receipt_path = tmp_path / "report-receipt.json"
    publish_research_index_report(
        index_path,
        index_receipt_path,
        _outcome,
        receipt_path=receipt_path,
    )

    def changed(
        projection,
        *,
        expected_report_id: str | None = None,
        expected_report_url: str | None = None,
    ):
        assert expected_report_id == "report-123"
        assert expected_report_url is not None
        value = _outcome(projection)
        return replace(
            value,
            report_id="another-report",
            report_url=(
                "https://wandb.ai/wandb/research-index/reports/changed--another-report"
            ),
        )

    with pytest.raises(ResearchIndexReportError, match="live.*disagrees"):
        publish_research_index_report(
            index_path,
            index_receipt_path,
            changed,
            receipt_path=receipt_path,
        )


def test_existing_receipt_rejects_changed_rendered_content_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_path, index_receipt_path = _sources(tmp_path, monkeypatch)
    receipt_path = tmp_path / "report-receipt.json"
    publish_research_index_report(
        index_path,
        index_receipt_path,
        _outcome,
        receipt_path=receipt_path,
    )

    def changed_rendering(projection, **_expected):
        return replace(_outcome(projection), rendered_content_digest="f" * 64)

    with pytest.raises(ResearchIndexReportError, match="live.*disagrees"):
        publish_research_index_report(
            index_path,
            index_receipt_path,
            changed_rendering,
            receipt_path=receipt_path,
        )


@pytest.mark.parametrize("padding_free_url", [False, True])
def test_padded_report_id_round_trips_without_identity_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    padding_free_url: bool,
) -> None:
    index_path, index_receipt_path = _sources(tmp_path, monkeypatch)
    report_id = "VmlldzoxNzY2NjU4Nw=="
    url_id = report_id.rstrip("=") if padding_free_url else report_id
    report_url = (
        f"https://wandb.ai/wandb/research-index/reports/community-research--{url_id}"
    )

    def publisher(
        projection,
        *,
        expected_report_id: str | None = None,
        expected_report_url: str | None = None,
    ):
        if expected_report_id is not None:
            assert expected_report_id == report_id
            assert expected_report_url == report_url
        return _outcome(
            projection,
            report_id=report_id,
            report_url=report_url,
        )

    receipt_path = tmp_path / "padded-report-receipt.json"
    receipt = publish_research_index_report(
        index_path,
        index_receipt_path,
        publisher,
        receipt_path=receipt_path,
    )

    assert receipt.report_id == report_id
    assert receipt.report_url == report_url
    assert read_research_index_report_publication_receipt(receipt_path) == receipt
    assert (
        publish_research_index_report(
            index_path,
            index_receipt_path,
            publisher,
            receipt_path=receipt_path,
        )
        == receipt
    )


def test_publication_rejects_wrong_readback_and_source_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_path, index_receipt_path = _sources(tmp_path, monkeypatch)

    def wrong_digest(projection):
        return replace(
            _outcome(projection),
            readback_projection_digest="0" * 64,
        )

    with pytest.raises(ResearchIndexReportError, match="readback disagrees"):
        publish_research_index_report(
            index_path,
            index_receipt_path,
            wrong_digest,
            receipt_path=tmp_path / "wrong.json",
        )

    def mutating(projection):
        index_path.write_text("changed")
        return _outcome(projection)

    with pytest.raises(ResearchIndexReportError, match="changed during Report"):
        publish_research_index_report(
            index_path,
            index_receipt_path,
            mutating,
            receipt_path=tmp_path / "mutated.json",
        )


def test_projection_rejects_configured_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_path, index_receipt_path = _sources(tmp_path, monkeypatch)
    with pytest.raises(ResearchIndexReportError, match="contains a secret value"):
        build_research_index_report_projection(
            index_path,
            index_receipt_path,
            report_description=(
                f"{RESEARCH_INDEX_REPORT_WARNING}\n\nsecret-value-12345"
            ),
            secret_values=("secret-value-12345",),
        )


def test_projection_rejects_cross_origin_study_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_path, index_receipt_path = _sources(tmp_path, monkeypatch)
    projection = build_research_index_report_projection(
        index_path,
        index_receipt_path,
    )
    cross_origin_study = replace(
        projection.studies[0],
        primary_evidence_url=(
            "https://outside.example/wandb/study-results/weave/calls/call-1"
        ),
        study_digest="",
    )

    with pytest.raises(ValueError, match="origin"):
        replace(
            projection,
            studies=(cross_origin_study,),
            projection_digest=projection.projection_digest,
        )


@pytest.mark.parametrize(
    "hostile_url",
    [
        "https://wandb.ai/wandb/research-index/runs/run-1\n[evil](https://evil.example)",
        "https://wandb.ai/wandb/research-index/runs/run 1",
        'https://wandb.ai/wandb/research-index/runs/"run-1"',
        "https://wandb.ai/wandb/research-index/runs/run\x00one",
        "https://wandb.ai/wandb/research-index/runs/run\x1bone",
        "https://wandb.ai/wandb/research-index/runs/run\x7fone",
    ],
)
def test_projection_rejects_markdown_unsafe_source_urls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hostile_url: str,
) -> None:
    index_path, index_receipt_path = _sources(tmp_path, monkeypatch)
    receipt = report_module.read_research_index_publication_receipt(index_receipt_path)
    hostile_receipt = SimpleNamespace(**{**vars(receipt), "run_url": hostile_url})
    monkeypatch.setattr(
        report_module,
        "read_research_index_publication_receipt",
        lambda _path: hostile_receipt,
    )

    with pytest.raises(ResearchIndexReportError, match="canonical validation"):
        build_research_index_report_projection(index_path, index_receipt_path)


def test_report_title_is_canonicalized_before_it_is_digest_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_path, index_receipt_path = _sources(tmp_path, monkeypatch)
    projection = build_research_index_report_projection(
        index_path,
        index_receipt_path,
        report_title="Release / why? #1 % results",
    )

    title_base = projection.report_title.rsplit(" · ", 1)[0]
    assert title_base == "Release why 1 results"
    assert not any(character in title_base for character in "/?#%")


def test_report_url_must_bind_exact_project_and_report_id() -> None:
    target = ResearchIndexPublicationTargetV1(
        project="wandb/research-index",
        api_base_url="https://api.wandb.ai",
        app_base_url="https://wandb.ai",
    )
    with pytest.raises(ValueError, match="exact Report"):
        ResearchIndexReportPublicationOutcomeV1(
            target=target,
            report_id="report-123",
            report_url=(
                "https://wandb.ai/wandb/other-project/reports/community--report-123"
            ),
            report_api="wandb-workspaces.reports.v2",
            report_api_version="0.4.5",
            api_stability="public_preview",
            readback_projection_digest="a" * 64,
            rendered_content_digest="b" * 64,
            readback_status="reconciled",
            publisher_id="publisher",
            publisher_revision="v1",
        )
