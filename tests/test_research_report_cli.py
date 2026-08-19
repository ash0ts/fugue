from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fugue.bench.cli import main


def _arguments(tmp_path: Path) -> list[str]:
    index = tmp_path / "research-index.json"
    index_receipt = tmp_path / "research-index-publication-receipt.json"
    index.write_text("{}\n")
    index_receipt.write_text("{}\n")
    return [
        "publish",
        "wandb-report",
        index.as_posix(),
        "--index-receipt",
        index_receipt.as_posix(),
        "--receipt",
        (tmp_path / "report-receipt.json").as_posix(),
        "--env-file",
        (tmp_path / "missing.env").as_posix(),
        "--repo-root",
        tmp_path.as_posix(),
        "--json",
    ]


def test_publish_wandb_report_cli_reports_missing_optional_extra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from fugue.bench import wandb_research_report

    def missing(_env):
        raise wandb_research_report.MissingWandbReportExtraError(
            "Install `fugue[wandb-report]`."
        )

    monkeypatch.setattr(
        wandb_research_report,
        "wandb_research_report_publisher_from_environment",
        missing,
    )

    assert main(_arguments(tmp_path)) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked"
    assert payload["error_type"] == "missing_wandb_report_extra"
    assert "fugue[wandb-report]" in payload["message"]


def test_publish_wandb_report_cli_returns_verified_projection_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from fugue.bench import research_report, wandb_research_report

    publisher = object()
    monkeypatch.setattr(
        wandb_research_report,
        "wandb_research_report_publisher_from_environment",
        lambda _env: publisher,
    )
    observed = []
    receipt = SimpleNamespace(
        target=SimpleNamespace(project="wandb/fugue-research"),
        report_url=(
            "https://wandb.ai/wandb/fugue-research/reports/Fugue-Research--Vmlldzox"
        ),
        projection_digest="a" * 64,
        readback_status="reconciled",
        receipt_digest="b" * 64,
        to_dict=lambda: {
            "schema_version": 1,
            "status": "published_and_reconciled",
            "report_url": (
                "https://wandb.ai/wandb/fugue-research/reports/Fugue-Research--Vmlldzox"
            ),
            "projection_digest": "a" * 64,
            "receipt_digest": "b" * 64,
        },
    )

    def publish(index, index_receipt, selected_publisher, **kwargs):
        observed.append((index, index_receipt, selected_publisher, kwargs))
        return receipt

    monkeypatch.setattr(research_report, "publish_research_index_report", publish)

    assert main(_arguments(tmp_path)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "published_and_reconciled"
    assert payload["projection_digest"] == "a" * 64
    assert len(observed) == 1
    assert observed[0][2] is publisher
    assert observed[0][3]["receipt_path"].name == "report-receipt.json"


@pytest.mark.parametrize(
    ("error", "exit_code", "status", "error_type"),
    [
        (
            "retryable",
            3,
            "retryable",
            "wandb_report_publication_retryable",
        ),
        (
            "contract",
            2,
            "blocked",
            "wandb_report_publication_blocked",
        ),
    ],
)
def test_publish_wandb_report_cli_distinguishes_retryable_and_contract_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: str,
    exit_code: int,
    status: str,
    error_type: str,
) -> None:
    from fugue.bench import research_report, wandb_research_report

    monkeypatch.setattr(
        wandb_research_report,
        "wandb_research_report_publisher_from_environment",
        lambda _env: object(),
    )

    def fail(*_args, **_kwargs):
        if error == "retryable":
            raise wandb_research_report.RetryableWandbResearchReportPublicationError(
                "Report readback is not available yet."
            )
        raise research_report.ResearchIndexReportError(
            "The Report conflicts with its bound projection."
        )

    monkeypatch.setattr(research_report, "publish_research_index_report", fail)

    assert main(_arguments(tmp_path)) == exit_code
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == status
    assert payload["error_type"] == error_type
