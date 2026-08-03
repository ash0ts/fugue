from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from fugue.bench.candidates import stable_digest
from fugue.bench.comparison import (
    _judge_calibration_value_issue,
    _judge_execution_calibration_issue,
    load_comparison,
)
from fugue.bench.judge_input import (
    comparison_judge_input_sanitizer_conformance_cases,
    sanitize_comparison_judge_value,
)

REPO_ROOT = Path(__file__).parents[1]
CAMPAIGN = Path("examples/comparisons/community-skill-upgrades")
ACTIVE_SPECS = (
    Path("examples/comparisons/superpowers-writing-plans-upgrade/comparison-v5.yaml"),
    Path("examples/comparisons/anthropic-skill-creator-upgrade/comparison-v2.yaml"),
    Path("examples/comparisons/vercel-react-best-practices-upgrade/comparison-v2.yaml"),
)


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_compatibility_root(tmp_path: Path) -> None:
    files = (
        CAMPAIGN / "judge-calibration-v2.json",
        CAMPAIGN / "judge-calibration.json",
        CAMPAIGN / "judge-calibration-result-v1.json",
        CAMPAIGN / "judge-sanitizer-compatibility-v2.json",
        CAMPAIGN / "judge-calibration-cases.jsonl",
        CAMPAIGN / "calibration-prior-runs.json",
        CAMPAIGN / "run_synthetic_calibration.py",
        CAMPAIGN / "validate_campaign.py",
        CAMPAIGN / "build_judge_sanitizer_compatibility_v2.py",
        Path("fugue/bench/judge_input.py"),
    )
    for relative in files:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, target)


def _judge():
    spec = load_comparison(
        REPO_ROOT
        / "examples/comparisons/anthropic-skill-creator-upgrade/confirmatory-v1.yaml",
        repo_root=REPO_ROOT,
    )
    return next(item for item in spec.evaluators if item.type == "llm_judge")


def test_v2_compatibility_artifacts_rebuild_exactly() -> None:
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / CAMPAIGN / "build_judge_sanitizer_compatibility_v2.py"),
            "--repo-root",
            str(REPO_ROOT),
            "--check",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def test_v2_proves_48_payloads_equivalent_and_keeps_judge_advisory() -> None:
    compatibility = _json(
        REPO_ROOT / CAMPAIGN / "judge-sanitizer-compatibility-v2.json"
    )
    report = _json(REPO_ROOT / CAMPAIGN / "judge-calibration-v2.json")

    assert compatibility["status"] == "passed"
    assert compatibility["summary"] == {
        "cases": 48,
        "byte_equivalent_payloads": 48,
        "byte_equivalent_prompts": 48,
        "transformed_payloads": 0,
    }
    assert compatibility["offline_conformance"]["summary"] == {
        "cases": 8,
        "credential_redaction_cases": 4,
        "safe_preservation_cases": 4,
        "passed": 8,
    }
    assert report["review_status"] == "pending_human_review"
    assert report["passed"] is False
    assert _judge_execution_calibration_issue(
        _judge(),
        repo_root=REPO_ROOT,
        approved_inputs=None,
    ) is None


def test_active_specs_use_v2_but_historical_specs_retain_v1() -> None:
    for path in ACTIVE_SPECS:
        spec = load_comparison(REPO_ROOT / path, repo_root=REPO_ROOT)
        judge = next(item for item in spec.evaluators if item.type == "llm_judge")
        assert judge.calibration == (
            "examples/comparisons/community-skill-upgrades/"
            "judge-calibration-v2.json"
        )
        assert judge.required is False

    historical = load_comparison(
        REPO_ROOT
        / "examples/comparisons/superpowers-writing-plans-upgrade/confirmatory-v4.yaml",
        repo_root=REPO_ROOT,
    )
    historical_judge = next(
        item for item in historical.evaluators if item.type == "llm_judge"
    )
    assert historical_judge.calibration.endswith("judge-calibration.json")


def test_v2_offline_conformance_is_recomputed_not_trusted(tmp_path: Path) -> None:
    _copy_compatibility_root(tmp_path)
    calibration_path = tmp_path / CAMPAIGN / "judge-calibration-v2.json"
    compatibility_path = (
        tmp_path / CAMPAIGN / "judge-sanitizer-compatibility-v2.json"
    )
    value = _json(compatibility_path)
    value["offline_conformance"]["cases"][0]["actual_output_sha256"] = "0" * 64
    unsigned = dict(value)
    unsigned.pop("receipt_digest")
    value["receipt_digest"] = stable_digest(unsigned)
    compatibility_path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    calibration = _json(calibration_path)
    calibration["execution_gate"]["compatibility_artifact_sha256"] = (
        _file_sha256(compatibility_path)
    )
    calibration["execution_gate"]["compatibility_receipt_digest"] = value[
        "receipt_digest"
    ]
    calibration_path.write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    judge = replace(
        _judge(),
        calibration=(CAMPAIGN / "judge-calibration-v2.json").as_posix(),
    )

    assert "offline case drifted" in str(
        _judge_execution_calibration_issue(
            judge,
            repo_root=tmp_path,
            approved_inputs=None,
        )
    )


def test_v2_human_calibration_state_is_supported_but_still_pending() -> None:
    report = _json(REPO_ROOT / CAMPAIGN / "judge-calibration-v2.json")

    assert _judge_calibration_value_issue(_judge(), report) == (
        "judge community-usefulness calibration is not adjudicated"
    )


def test_sanitizer_conformance_fixtures_are_idempotent() -> None:
    cases = comparison_judge_input_sanitizer_conformance_cases()
    assert {case["family"] for case in cases} == {
        "credential_redaction",
        "safe_placeholder_preservation",
    }
    for case in cases:
        actual = sanitize_comparison_judge_value(case["input"])
        assert actual == case["expected_output"]
        assert (actual != case["input"]) is case["expected_transformed"]
        assert sanitize_comparison_judge_value(actual) == actual
