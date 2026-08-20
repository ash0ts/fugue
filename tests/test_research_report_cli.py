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


def _patch_research_delivery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    record=None,
) -> dict[str, list[object]]:
    from fugue.bench import research_report

    database = tmp_path / ".fugue" / "research.db"
    database.parent.mkdir(parents=True, exist_ok=True)
    database.write_bytes(b"existing Research database")
    membership = SimpleNamespace(study_id="study-v13")
    evidence = SimpleNamespace(to_dict=lambda: {"schema_version": 1})
    observed: dict[str, list[object]] = {
        "validations": [],
        "deliveries": [],
    }

    class FakeStudyStore:
        def __init__(self, root: Path) -> None:
            assert root == tmp_path

        def validate_research_index_report_memberships(self, **kwargs) -> None:
            observed["validations"].append(kwargs)

        def record_research_index_report_publication_evidence(self, publication):
            observed["deliveries"].append(publication)
            if record is not None:
                return record(publication)
            return publication

    monkeypatch.setattr("fugue.research.store.StudyStore", FakeStudyStore)
    monkeypatch.setattr(
        research_report,
        "build_research_index_report_study_memberships",
        lambda *_args, **_kwargs: ("research-v1", (membership,)),
    )
    monkeypatch.setattr(
        research_report,
        "build_research_index_report_publication_evidence",
        lambda *_args, **_kwargs: evidence,
    )
    return observed


def test_publish_wandb_report_cli_reports_missing_optional_extra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from fugue.bench import wandb_research_report

    _patch_research_delivery(tmp_path, monkeypatch)

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

    observed_delivery = _patch_research_delivery(tmp_path, monkeypatch)

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
        kwargs["receipt_path"].write_text('{"receipt":"published"}\n')
        return receipt

    monkeypatch.setattr(research_report, "publish_research_index_report", publish)

    assert main(_arguments(tmp_path)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "published_and_reconciled"
    assert payload["projection_digest"] == "a" * 64
    assert len(observed) == 1
    assert observed[0][2] is publisher
    assert observed[0][3]["receipt_path"].name == "report-receipt.json"
    assert len(observed_delivery["validations"]) == 1
    assert len(observed_delivery["deliveries"]) == 1


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

    _patch_research_delivery(tmp_path, monkeypatch)

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


def test_report_research_delivery_retry_does_not_rewrite_wandb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from fugue.bench import research_report, wandb_research_report

    delivery_calls = 0

    def record(_publication):
        nonlocal delivery_calls
        delivery_calls += 1
        if delivery_calls == 1:
            raise RuntimeError("simulated Research delivery interruption")
        return _publication

    observed_delivery = _patch_research_delivery(
        tmp_path,
        monkeypatch,
        record=record,
    )
    publisher_factory_calls = 0
    publisher_calls = 0
    verify_calls = 0
    receipt_path = tmp_path / "report-receipt.json"
    receipt = SimpleNamespace(
        target=SimpleNamespace(project="wandb/fugue-research"),
        report_url=(
            "https://wandb.ai/wandb/fugue-research/reports/"
            "Fugue-Research--report-v13"
        ),
        projection_digest="a" * 64,
        readback_status="reconciled",
        receipt_digest="b" * 64,
        to_dict=lambda: {
            "schema_version": 1,
            "status": "published_and_reconciled",
            "report_url": (
                "https://wandb.ai/wandb/fugue-research/reports/"
                "Fugue-Research--report-v13"
            ),
            "projection_digest": "a" * 64,
            "receipt_digest": "b" * 64,
        },
    )

    def factory(_env):
        nonlocal publisher_factory_calls
        publisher_factory_calls += 1
        return object()

    def publish(_index, _index_receipt, _publisher, **kwargs):
        nonlocal publisher_calls
        publisher_calls += 1
        kwargs["receipt_path"].write_text('{"receipt":"published"}\n')
        return receipt

    def verify(*_args, **_kwargs):
        nonlocal verify_calls
        verify_calls += 1
        return receipt

    monkeypatch.setattr(
        wandb_research_report,
        "wandb_research_report_publisher_from_environment",
        factory,
    )
    monkeypatch.setattr(research_report, "publish_research_index_report", publish)
    monkeypatch.setattr(
        research_report,
        "verify_research_index_report_publication",
        verify,
    )
    arguments = _arguments(tmp_path)
    index_path = tmp_path / "research-index.json"
    index_receipt_path = tmp_path / "research-index-publication-receipt.json"
    index_before = index_path.read_bytes()
    index_receipt_before = index_receipt_path.read_bytes()

    with pytest.raises(RuntimeError, match="delivery interruption"):
        main(arguments)
    report_receipt_before = receipt_path.read_bytes()
    capsys.readouterr()

    assert main(arguments) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "published_and_reconciled"
    assert publisher_factory_calls == 1
    assert publisher_calls == 1
    assert verify_calls == 1
    assert delivery_calls == 2
    assert len(observed_delivery["deliveries"]) == 2
    assert index_path.read_bytes() == index_before
    assert index_receipt_path.read_bytes() == index_receipt_before
    assert receipt_path.read_bytes() == report_receipt_before
