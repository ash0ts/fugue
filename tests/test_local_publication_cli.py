from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_local_publication import (
    _canonical_manifest,
    _FakeComparisonResultV3,
    _outcome,
    _result_fixture,
)

import fugue.bench.comparison as comparison
import fugue.bench.local_publication as publication
from fugue.bench.cli import main
from fugue.bench.local_publication import MissingWeaveExtraError


def _patch_result_readers(
    monkeypatch: pytest.MonkeyPatch,
    result: _FakeComparisonResultV3,
) -> None:
    monkeypatch.setattr(comparison, "ComparisonResultV3", _FakeComparisonResultV3)
    monkeypatch.setattr(comparison, "read_comparison_result", lambda _path: result)
    monkeypatch.setattr(publication, "ComparisonResultV3", _FakeComparisonResultV3)
    monkeypatch.setattr(publication, "read_comparison_result", lambda _path: result)


def test_publish_weave_cli_preserves_result_digest_and_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _manifest_path, manifest = _canonical_manifest(tmp_path)
    result_path, result = _result_fixture(tmp_path, manifest)
    _patch_result_readers(monkeypatch, result)
    called_projects: list[str] = []

    def factory(_env):
        def publisher(_result, _manifest, target):
            called_projects.append(target.project_slug)
            return _outcome(manifest, target)

        return publisher

    monkeypatch.setattr(publication, "weave_publisher_from_environment", factory)
    before = result_path.read_bytes()

    assert (
        main(
            [
                "publish",
                "weave",
                result_path.as_posix(),
                "--project",
                "wandb/local-result",
                "--json",
                "--repo-root",
                tmp_path.as_posix(),
                "--env-file",
                (tmp_path / "missing.env").as_posix(),
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert called_projects == ["wandb/local-result"]
    assert result_path.read_bytes() == before
    assert payload["result_digest"] == result.result_digest
    assert payload["local_manifest_digest"] == manifest.manifest_digest
    assert payload["target"]["entity"] == "wandb"
    assert payload["target"]["project"] == "local-result"
    assert payload["target"]["destination"]["project_slug"] == ("wandb/local-result")
    assert payload["target"]["destination"]["api_base_url"] == ("https://api.wandb.ai")
    assert payload["target"]["destination"]["trace_base_url"] == (
        "https://trace.wandb.ai"
    )
    assert payload["target"]["destination"]["app_base_url"] == ("https://wandb.ai")
    assert payload["status"] == "published"


def test_publish_weave_cli_reports_missing_optional_extra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _manifest_path, manifest = _canonical_manifest(tmp_path)
    result_path, result = _result_fixture(tmp_path, manifest)
    _patch_result_readers(monkeypatch, result)

    def missing(_env):
        raise MissingWeaveExtraError(
            'Weave publication requires `pip install "fugue[weave]"`.'
        )

    monkeypatch.setattr(
        publication,
        "weave_publisher_from_environment",
        missing,
    )

    assert (
        main(
            [
                "publish",
                "weave",
                result_path.as_posix(),
                "--project",
                "wandb/local-result",
                "--json",
                "--repo-root",
                tmp_path.as_posix(),
                "--env-file",
                (tmp_path / "missing.env").as_posix(),
            ]
        )
        == 2
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked"
    assert payload["error_type"] == "missing_weave_extra"
    assert 'pip install "fugue[weave]"' in payload["message"]


def test_publish_weave_cli_warns_about_the_privacy_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _manifest_path, manifest = _canonical_manifest(tmp_path)
    result_path, result = _result_fixture(tmp_path, manifest)
    _patch_result_readers(monkeypatch, result)

    def factory(_env):
        return lambda _result, _manifest, target: _outcome(manifest, target)

    monkeypatch.setattr(publication, "weave_publisher_from_environment", factory)

    assert (
        main(
            [
                "publish",
                "weave",
                result_path.as_posix(),
                "--project",
                "wandb/local-result",
                "--repo-root",
                tmp_path.as_posix(),
                "--env-file",
                (tmp_path / "missing.env").as_posix(),
            ]
        )
        == 0
    )

    output = " ".join(capsys.readouterr().out.split())
    assert "Privacy boundary" in output
    assert "Raw local transcript and tool-event artifact files remain local" in output
    assert "Local evidence manifest digest" in output


def test_publish_weave_cli_binds_explicit_research_and_study_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _manifest_path, manifest = _canonical_manifest(tmp_path)
    result_path, result = _result_fixture(tmp_path, manifest)
    _patch_result_readers(monkeypatch, result)
    observed = []
    validations = []
    deliveries = []

    research_database = tmp_path / ".fugue" / "research.db"
    research_database.parent.mkdir(parents=True, exist_ok=True)
    research_database.write_bytes(b"existing Research database")

    class FakeStudyStore:
        def __init__(self, root: Path) -> None:
            assert root == tmp_path

        def validate_weave_publication_projection(self, **kwargs) -> None:
            validations.append(kwargs)

        def record_weave_publication_evidence(self, publication_evidence):
            deliveries.append(publication_evidence)
            return publication_evidence

    monkeypatch.setattr("fugue.research.store.StudyStore", FakeStudyStore)

    def factory(_env):
        def publisher(_result, _manifest, target):
            observed.append(target)
            return _outcome(manifest, target)

        return publisher

    monkeypatch.setattr(publication, "weave_publisher_from_environment", factory)
    assert (
        main(
            [
                "publish",
                "weave",
                result_path.as_posix(),
                "--project",
                "wandb/fugue-experiments",
                "--research-id",
                "fugue-standalone-lab-v1",
                "--study-id",
                "prompt-change-live-v1",
                "--json",
                "--repo-root",
                tmp_path.as_posix(),
                "--env-file",
                (tmp_path / "missing.env").as_posix(),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert observed[0].study_scope.research_id == "fugue-standalone-lab-v1"
    assert observed[0].study_scope.study_id == "prompt-change-live-v1"
    assert payload["target"]["study_scope"] == {
        "research_id": "fugue-standalone-lab-v1",
        "schema_version": 1,
        "study_id": "prompt-change-live-v1",
    }
    assert validations == [
        {
            "research_id": "fugue-standalone-lab-v1",
            "experiment_id": "prompt-change-live-v1",
            "result_digest": result.result_digest,
            "qualification_digest": result.qualification_digest,
            "attempt_ids": manifest.planned_attempt_ids,
        }
    ]
    assert len(deliveries) == 1
    assert deliveries[0].receipt_digest == payload["receipt_digest"]
    assert deliveries[0].attempt_ids == manifest.planned_attempt_ids


def test_scoped_publish_retry_delivers_existing_receipt_without_republishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path, manifest = _canonical_manifest(tmp_path)
    result_path, result = _result_fixture(tmp_path, manifest)
    _patch_result_readers(monkeypatch, result)
    research_database = tmp_path / ".fugue" / "research.db"
    research_database.parent.mkdir(parents=True, exist_ok=True)
    research_database.write_bytes(b"existing Research database")
    publisher_factory_calls = 0
    publisher_calls = 0
    delivery_calls = 0
    delivered = []

    def factory(_env):
        nonlocal publisher_factory_calls
        publisher_factory_calls += 1

        def publisher(_result, _manifest, target):
            nonlocal publisher_calls
            publisher_calls += 1
            return _outcome(manifest, target)

        return publisher

    class FakeStudyStore:
        def __init__(self, root: Path) -> None:
            assert root == tmp_path

        def validate_weave_publication_projection(self, **_kwargs) -> None:
            return None

        def record_weave_publication_evidence(self, publication_evidence):
            nonlocal delivery_calls
            delivery_calls += 1
            if delivery_calls == 1:
                raise RuntimeError("simulated Research delivery interruption")
            delivered.append(publication_evidence)
            return publication_evidence

    monkeypatch.setattr(publication, "weave_publisher_from_environment", factory)
    monkeypatch.setattr("fugue.research.store.StudyStore", FakeStudyStore)
    arguments = [
        "publish",
        "weave",
        result_path.as_posix(),
        "--project",
        "wandb/fugue-experiments",
        "--research-id",
        "fugue-standalone-lab-v1",
        "--study-id",
        "prompt-change-live-v1",
        "--json",
        "--repo-root",
        tmp_path.as_posix(),
        "--env-file",
        (tmp_path / "missing.env").as_posix(),
    ]
    result_before = result_path.read_bytes()
    manifest_before = manifest_path.read_bytes()

    with pytest.raises(
        RuntimeError,
        match="simulated Research delivery interruption",
    ):
        main(arguments)

    receipt_path = result_path.with_name("weave-publication-receipt.json")
    receipt_before = receipt_path.read_bytes()
    capsys.readouterr()

    assert main(arguments) == 0
    payload = json.loads(capsys.readouterr().out)

    assert publisher_factory_calls == 1
    assert publisher_calls == 1
    assert delivery_calls == 2
    assert len(delivered) == 1
    assert delivered[0].receipt_digest == payload["receipt_digest"]
    assert result_path.read_bytes() == result_before
    assert manifest_path.read_bytes() == manifest_before
    assert receipt_path.read_bytes() == receipt_before


def test_publish_weave_cli_rejects_study_without_research(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _manifest_path, manifest = _canonical_manifest(tmp_path)
    result_path, result = _result_fixture(tmp_path, manifest)
    _patch_result_readers(monkeypatch, result)

    with pytest.raises(
        publication.LocalResultPublicationError,
        match="--study-id requires --research-id",
    ):
        main(
            [
                "publish",
                "weave",
                result_path.as_posix(),
                "--project",
                "wandb/fugue-experiments",
                "--study-id",
                "orphan-study",
                "--repo-root",
                tmp_path.as_posix(),
                "--env-file",
                (tmp_path / "missing.env").as_posix(),
            ]
        )
