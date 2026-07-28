from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from fugue.bench.comparison import (
    COMPARISON_RUNTIME_ROOT,
    ComparisonEvaluatorV1,
    analyze_comparison_rows,
    check_comparison,
    comparison_from_dict,
    load_comparison,
    preview_comparison,
    scaffold_comparison,
    score_comparison_rows,
)

EXAMPLE = Path("examples/comparisons/source-use-replay")


def test_source_use_comparison_is_ready_and_exact() -> None:
    root = Path.cwd()
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=root)

    readiness = check_comparison(spec, repo_root=root)
    preview = preview_comparison(spec, repo_root=root)

    assert readiness.status == "ready"
    assert readiness.task_count == 4
    assert readiness.base_failures == 4
    assert readiness.gold_passes == 4
    assert readiness.actual_changes == ("skills",)
    assert readiness.estimated_cells == 16
    assert readiness.estimated_cost_usd == 0
    assert preview.matrix["estimated_trials"] == 16
    assert preview.matrix["applicable_cells"] == 16
    assert not (root / COMPARISON_RUNTIME_ROOT / spec.spec_digest).exists()


def test_comparison_rejects_unknown_fields_and_undeclared_changes() -> None:
    root = Path.cwd()
    raw = yaml.safe_load((EXAMPLE / "comparison.yaml").read_text())
    raw["surprise"] = True
    with pytest.raises(ValueError, match="unknown comparison field"):
        comparison_from_dict(raw, repo_root=root, source=EXAMPLE)

    raw.pop("surprise")
    raw["changed"] = ["prompt_id"]
    spec = comparison_from_dict(raw, repo_root=root, source=EXAMPLE)
    readiness = check_comparison(spec, repo_root=root)
    assert readiness.status == "blocked"
    assert any("resolved behavior diff" in item for item in readiness.blockers)


def test_comparison_keeps_public_tasks_and_private_labels_separate(
    tmp_path: Path,
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    labels = tmp_path / "labels.jsonl"
    tasks.write_text(
        json.dumps(
            {
                "id": "leak",
                "input": {"question": "question"},
                "expected": {"answer": 1},
            }
        )
        + "\n"
    )
    labels.write_text(
        json.dumps(
            {
                "id": "leak",
                "expected": {"answer": 1},
                "base_output": {"answer": 0},
                "gold_output": {"answer": 1},
            }
        )
        + "\n"
    )
    raw = yaml.safe_load((EXAMPLE / "comparison.yaml").read_text())
    raw["taskset"] = {
        "tasks": tasks.relative_to(Path.cwd()).as_posix()
        if tasks.is_relative_to(Path.cwd())
        else tasks.as_posix(),
        "private_labels": labels.relative_to(Path.cwd()).as_posix()
        if labels.is_relative_to(Path.cwd())
        else labels.as_posix(),
    }
    spec = comparison_from_dict(raw, repo_root=tmp_path, source=tmp_path)
    with pytest.raises(ValueError, match="unknown public task .*field"):
        check_comparison(spec, repo_root=tmp_path)


def test_replay_scores_aligned_improvements_and_regressions() -> None:
    root = Path.cwd()
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=root)
    preview = preview_comparison(spec, repo_root=root)
    rows = [
        json.loads(line)
        for line in (EXAMPLE / "attempts.jsonl").read_text().splitlines()
    ]
    scored = score_comparison_rows(spec, rows, repo_root=root)
    result = analyze_comparison_rows(
        comparison_id=spec.id,
        preview_digest=preview.preview_digest,
        rows=scored,
        source="test-replay",
    )

    assert result.rows == 16
    assert result.baseline_passed == 2
    assert result.candidate_passed == 6
    assert result.improved == 5
    assert result.regressed == 1
    assert result.unchanged == 2
    assert result.incomplete == 0


def test_zero_row_comparison_cannot_succeed() -> None:
    with pytest.raises(ValueError, match="at least one attempt row"):
        analyze_comparison_rows(
            comparison_id="empty",
            preview_digest="0" * 64,
            rows=[],
            source="test",
        )


def test_required_judge_needs_reviewed_calibration(tmp_path: Path) -> None:
    root = Path.cwd()
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=root)
    judge = ComparisonEvaluatorV1(
        id="maintainer-review",
        type="llm_judge",
        required=True,
        profile="wandb/openai/gpt-oss-120b",
        calibration=None,
        reserve_cost_usd=0.1,
    )
    blocked = replace(spec, evaluators=(*spec.evaluators, judge))
    readiness = check_comparison(blocked, repo_root=root)
    assert readiness.status == "blocked"
    assert any("no reviewed calibration" in item for item in readiness.blockers)

    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "examples": 48,
                "true_positive_rate": 0.9,
                "true_negative_rate": 0.9,
                "critical_false_passes": 0,
            }
        )
    )
    copied = root / ".fugue" / "test-comparison-calibration.json"
    copied.parent.mkdir(parents=True, exist_ok=True)
    copied.write_text(calibration.read_text())
    try:
        qualified_judge = replace(
            judge, calibration=copied.relative_to(root).as_posix()
        )
        ready = replace(
            spec,
            evaluators=(*spec.evaluators, qualified_judge),
            execution=replace(spec.execution, max_cost_usd=10),
        )
        assert check_comparison(ready, repo_root=root).status == "ready"
    finally:
        copied.unlink(missing_ok=True)


def test_scaffold_refuses_non_empty_destination(tmp_path: Path) -> None:
    destination = tmp_path / "comparison"
    scaffold_comparison(destination)
    assert (destination / "comparison.yaml").is_file()
    assert (destination / "skills/verify-current-source/SKILL.md").is_file()
    with pytest.raises(FileExistsError, match="non-empty"):
        scaffold_comparison(destination)
