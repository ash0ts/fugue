from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_local_publication import (
    _canonical_manifest,
    _FakeComparisonResultV3,
)

import fugue.bench.comparison as comparison
import fugue.bench.local_publication as publication
from fugue.bench.cli import main
from fugue.bench.comparison import (
    COMPARISON_RESULT_ROOT,
    ComparisonResultV1,
    _legacy_comparison_result_digest,
)


def _legacy_result() -> ComparisonResultV1:
    unsigned = ComparisonResultV1(
        schema_version=1,
        comparison_id="legacy-local-result",
        preview_digest="a" * 64,
        source="legacy-run",
        evidence_project=None,
        rows=2,
        baseline_passed=0,
        candidate_passed=1,
        improved=1,
        regressed=0,
        unchanged=0,
        incomplete=0,
        required_evaluations_incomplete=0,
        deterministic_summary={},
        judge_summary={"status": "not_used"},
        mechanism_summary={},
        operational_summary={
            "infrastructure_failures": 0,
            "execution_states": {"completed": 2},
            "evidence_states": {"reconciled": 2},
            "observed_cost_usd": 0.25,
            "mcp_tool_usage": {},
        },
        evidence_links=(),
        paired_cases=(
            {
                "task_id": "task-1",
                "harness": "claude-code",
                "attempt": 1,
                "baseline_passed": False,
                "candidate_passed": True,
                "status": "improved",
            },
        ),
        limitations=("Historical result without a local evidence manifest.",),
    )
    return replace(
        unsigned,
        result_digest=_legacy_comparison_result_digest(unsigned.to_dict()),
    )


def _write_latest(root: Path, result: ComparisonResultV1) -> tuple[Path, Path]:
    destination = root / COMPARISON_RESULT_ROOT / result.preview_digest
    destination.mkdir(parents=True)
    result_path = destination / "result.json"
    markdown_path = destination / "result.md"
    result_path.write_text(json.dumps(result.to_dict()), encoding="utf-8")
    markdown_path.write_text("STALE MARKDOWN MUST NOT BE RENDERED", encoding="utf-8")
    (root / COMPARISON_RESULT_ROOT / "latest.json").write_text(
        json.dumps(
            {
                "comparison_id": result.comparison_id,
                "result": result_path.relative_to(root).as_posix(),
                "markdown": markdown_path.relative_to(root).as_posix(),
            }
        ),
        encoding="utf-8",
    )
    return result_path, markdown_path


def test_result_cli_validates_and_reserializes_latest_and_named_result(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = _legacy_result()
    _write_latest(tmp_path, result)

    for selected in ("latest", result.comparison_id):
        assert (
            main(
                [
                    "result",
                    selected,
                    "--repo-root",
                    tmp_path.as_posix(),
                    "--json",
                ]
            )
            == 0
        )
        assert json.loads(capsys.readouterr().out) == json.loads(
            json.dumps(result.to_dict())
        )


def test_result_cli_rejects_tampered_result_json(
    tmp_path: Path,
) -> None:
    result = _legacy_result()
    result_path, _markdown_path = _write_latest(tmp_path, result)
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    raw["candidate_passed"] = 2
    result_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="result digest does not match"):
        main(["result", "latest", "--repo-root", tmp_path.as_posix(), "--json"])


def test_result_cli_rejects_latest_pointer_that_escapes_result_root(
    tmp_path: Path,
) -> None:
    result_root = tmp_path / COMPARISON_RESULT_ROOT
    result_root.mkdir(parents=True)
    outside = tmp_path / "outside" / "result.json"
    outside.parent.mkdir()
    outside.write_text("{}", encoding="utf-8")
    (result_root / "latest.json").write_text(
        json.dumps(
            {
                "result": outside.relative_to(tmp_path).as_posix(),
                "markdown": (outside.parent / "result.md")
                .relative_to(tmp_path)
                .as_posix(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes the result root"):
        main(["result", "latest", "--repo-root", tmp_path.as_posix(), "--json"])


def test_result_cli_renders_fresh_markdown_and_labels_legacy_boundary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = _legacy_result()
    _write_latest(tmp_path, result)

    assert main(["result", "latest", "--repo-root", tmp_path.as_posix()]) == 0
    rendered = capsys.readouterr().out
    assert "STALE MARKDOWN MUST NOT BE RENDERED" not in rendered
    assert result.comparison_id in rendered
    assert "historical V1/V2 result passed its canonical result-digest" in rendered
    assert "not qualified local evidence" in rendered


def test_result_cli_recomputes_the_bound_v3_local_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path, manifest = _canonical_manifest(tmp_path)
    manifest_file_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    result = _FakeComparisonResultV3(manifest, manifest_file_sha256)
    destination = tmp_path / COMPARISON_RESULT_ROOT / ("b" * 64)
    destination.mkdir(parents=True)
    result_path = destination / "result.json"
    markdown_path = destination / "result.md"
    result_path.write_text('{"schema_version": 3}', encoding="utf-8")
    markdown_path.write_text("stale", encoding="utf-8")
    (tmp_path / COMPARISON_RESULT_ROOT / "latest.json").write_text(
        json.dumps(
            {
                "result": result_path.relative_to(tmp_path).as_posix(),
                "markdown": markdown_path.relative_to(tmp_path).as_posix(),
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(comparison, "ComparisonResultV3", _FakeComparisonResultV3)
    monkeypatch.setattr(comparison, "read_comparison_result", lambda _path: result)
    monkeypatch.setattr(comparison, "_result_markdown", lambda _result: "verified\n")
    monkeypatch.setattr(publication, "ComparisonResultV3", _FakeComparisonResultV3)

    assert main(["result", "latest", "--repo-root", tmp_path.as_posix()]) == 0
    assert "verified" in capsys.readouterr().out

    attempt_path = next((manifest_path.parent / "attempt-records").glob("*.json"))
    attempt_path.write_text("{}", encoding="utf-8")
    with pytest.raises(
        publication.LocalResultPublicationError,
        match="canonical recomputation",
    ):
        main(["result", "latest", "--repo-root", tmp_path.as_posix()])
