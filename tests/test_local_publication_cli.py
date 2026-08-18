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
