from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_research_index import _index

import fugue.bench.research_index as research_index
from fugue.bench.cli import main
from fugue.bench.research_index import ResearchIndexPublicationOutcomeV1


def test_study_index_cli_builds_the_same_canonical_local_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = _index(tmp_path, monkeypatch)
    sources = []
    for study_id in ("study-a", "study-b"):
        sources.extend(
            [
                "--source",
                (tmp_path / f"{study_id}-result.json").as_posix(),
                (tmp_path / f"{study_id}-receipt.json").as_posix(),
            ]
        )
    output = tmp_path / "research-index.json"

    assert (
        main(
            [
                "study",
                "index",
                "--research-id",
                expected.research_id,
                "--title",
                expected.title,
                "--objective",
                expected.objective,
                *sources,
                "--output",
                output.as_posix(),
                "--env-file",
                (tmp_path / "missing.env").as_posix(),
                "--repo-root",
                tmp_path.as_posix(),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload == expected.to_dict()
    assert research_index.read_research_index(output) == expected


def test_publish_wandb_index_cli_reports_missing_optional_extra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = _index(tmp_path, monkeypatch)
    index_path = tmp_path / "research-index.json"
    research_index.write_research_index(index_path, expected)

    from fugue.bench import wandb_research_index

    def missing(_env):
        raise wandb_research_index.MissingWandbIndexExtraError(
            'W&B index publication requires `pip install "fugue[wandb-index]"`.'
        )

    monkeypatch.setattr(
        wandb_research_index,
        "wandb_research_index_publisher_from_environment",
        missing,
    )
    assert (
        main(
            [
                "publish",
                "wandb-index",
                index_path.as_posix(),
                "--project",
                "wandb/fugue-standalone-lab-v1",
                "--env-file",
                (tmp_path / "missing.env").as_posix(),
                "--repo-root",
                tmp_path.as_posix(),
                "--json",
            ]
        )
        == 2
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked"
    assert payload["error_type"] == "missing_wandb_index_extra"
    assert "fugue[wandb-index]" in payload["message"]


def test_publish_wandb_index_cli_preserves_index_and_returns_verified_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = _index(tmp_path, monkeypatch)
    index_path = tmp_path / "research-index.json"
    receipt_path = tmp_path / "publication-receipt.json"
    research_index.write_research_index(index_path, expected)
    before = index_path.read_bytes()
    observed = []

    from fugue.bench import wandb_research_index

    def factory(_env):
        def publisher(index, index_bytes, target):
            observed.append((index, index_bytes, target))
            return ResearchIndexPublicationOutcomeV1(
                target=target,
                run_url="https://wandb.ai/wandb/fugue-lab/runs/index-run",
                artifact_url=(
                    "https://wandb.ai/wandb/fugue-lab/artifacts/"
                    "study-index/research-index/v0"
                ),
                report_url=None,
                report_status="unavailable",
                publisher_id="test-publisher",
                publisher_revision="v1",
            )

        return publisher

    monkeypatch.setattr(
        wandb_research_index,
        "wandb_research_index_publisher_from_environment",
        factory,
    )
    assert (
        main(
            [
                "publish",
                "wandb-index",
                index_path.as_posix(),
                "--project",
                "wandb/fugue-lab",
                "--receipt",
                receipt_path.as_posix(),
                "--env-file",
                (tmp_path / "missing.env").as_posix(),
                "--repo-root",
                tmp_path.as_posix(),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert len(observed) == 1
    assert observed[0][0:2] == (expected, before)
    assert observed[0][2].project == "wandb/fugue-lab"
    assert index_path.read_bytes() == before
    assert payload["index_digest"] == expected.index_digest
    assert payload["run_url"].endswith("/runs/index-run")
    assert payload["report_status"] == "unavailable"
    receipt = research_index.read_research_index_publication_receipt(receipt_path)
    assert receipt.receipt_digest == payload["receipt_digest"]
