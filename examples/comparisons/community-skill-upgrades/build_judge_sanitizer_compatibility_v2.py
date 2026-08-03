#!/usr/bin/env python3
"""Build the model-free V2 compatibility proof for the V1 judge run.

The historical 48 provider responses are not rerun or relabeled. This builder
archives the immutable V1 result, reconstructs every provider-visible request
from the locked runner/cases/rubric, and proves that the current judge-input
sanitizer leaves those exact payloads and prompts byte-equivalent. Separate
offline cases prove representative redaction and safe-placeholder behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import run_synthetic_calibration as historical_runner

from fugue.bench.candidates import stable_digest
from fugue.bench.comparison import comparison_judge_request_payload
from fugue.bench.judge_input import (
    COMPARISON_JUDGE_INPUT_SANITIZER_CONTRACT,
    COMPARISON_JUDGE_INPUT_SANITIZER_IMPLEMENTATION_SHA256,
    COMPARISON_JUDGE_INPUT_SANITIZER_TRANSFORM,
    COMPARISON_JUDGE_INPUT_SANITIZER_VERSION,
    comparison_judge_input_sanitizer_conformance_cases,
    sanitize_comparison_judge_value,
)

CAMPAIGN_DIR = Path("examples/comparisons/community-skill-upgrades")
V1_REPORT_PATH = CAMPAIGN_DIR / "judge-calibration.json"
V1_RUNTIME_RESULT_PATH = Path(
    ".fugue/runtime/community-skill-upgrades/judge-calibration.result.json"
)
V1_ARCHIVE_RESULT_PATH = CAMPAIGN_DIR / "judge-calibration-result-v1.json"
V2_COMPATIBILITY_PATH = CAMPAIGN_DIR / "judge-sanitizer-compatibility-v2.json"
V2_REPORT_PATH = CAMPAIGN_DIR / "judge-calibration-v2.json"
BUILDER_PATH = CAMPAIGN_DIR / "build_judge_sanitizer_compatibility_v2.py"
SANITIZER_PATH = Path("fugue/bench/judge_input.py")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(value)
    return rows


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _transformed_value_count(source: Any, target: Any) -> int:
    if isinstance(source, Mapping) and isinstance(target, Mapping):
        if set(source) != set(target):
            raise ValueError("sanitizer changed payload fields")
        return sum(
            _transformed_value_count(source[key], target[key]) for key in source
        )
    if (
        isinstance(source, Sequence)
        and not isinstance(source, str | bytes)
        and isinstance(target, Sequence)
        and not isinstance(target, str | bytes)
    ):
        if len(source) != len(target):
            raise ValueError("sanitizer changed payload length")
        return sum(
            _transformed_value_count(before, after)
            for before, after in zip(source, target, strict=True)
        )
    if type(source) is not type(target):
        raise ValueError("sanitizer changed payload type")
    return int(source != target)


def _case_receipts(
    *, rows: Sequence[Mapping[str, Any]], rubric: Mapping[str, Any]
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for row in rows:
        prompt = historical_runner._prompt(rubric, row)
        prefix, serialized_payload = prompt.split("\n\n", 1)
        source_payload = json.loads(serialized_payload)
        permitted_evidence = dict(row["permitted_evidence"])
        for field_name in ("inspected_paths", "changed_paths"):
            if field_name in permitted_evidence:
                permitted_evidence[f"{field_name}_status"] = "available"
        reconstructed_payload = comparison_judge_request_payload(
            public_task=row["public_task"],
            response=row["response"],
            permitted_evidence=permitted_evidence,
            rubric=str(rubric["rubric"]),
            dimensions=tuple(str(item) for item in rubric["dimensions"]),
        )
        if source_payload != reconstructed_payload:
            raise ValueError(
                f"historical payload reconstruction drifted for {row['id']}"
            )
        provider_payload = sanitize_comparison_judge_value(source_payload)
        transformed_value_count = _transformed_value_count(
            source_payload, provider_payload
        )
        provider_prompt = prefix + "\n\n" + json.dumps(
            provider_payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        transform_receipt = {
            "schema_version": 1,
            "case_id": str(row["id"]),
            "sanitizer_version": COMPARISON_JUDGE_INPUT_SANITIZER_VERSION,
            "sanitizer_contract": COMPARISON_JUDGE_INPUT_SANITIZER_CONTRACT,
            "sanitizer_transform": COMPARISON_JUDGE_INPUT_SANITIZER_TRANSFORM,
            "sanitizer_implementation_sha256": (
                COMPARISON_JUDGE_INPUT_SANITIZER_IMPLEMENTATION_SHA256
            ),
            "source_payload_sha256": stable_digest(source_payload),
            "provider_payload_sha256": stable_digest(provider_payload),
            "transformed": transformed_value_count > 0,
            "transformed_value_count": transformed_value_count,
        }
        receipts.append(
            {
                "case_id": str(row["id"]),
                "source_payload_sha256": transform_receipt[
                    "source_payload_sha256"
                ],
                "provider_payload_sha256": transform_receipt[
                    "provider_payload_sha256"
                ],
                "historical_prompt_sha256": hashlib.sha256(
                    prompt.encode()
                ).hexdigest(),
                "current_prompt_sha256": hashlib.sha256(
                    provider_prompt.encode()
                ).hexdigest(),
                "transformed": transform_receipt["transformed"],
                "transformed_value_count": transformed_value_count,
                "transform_receipt_digest": stable_digest(transform_receipt),
            }
        )
    return receipts


def _offline_conformance() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in comparison_judge_input_sanitizer_conformance_cases():
        actual = sanitize_comparison_judge_value(case["input"])
        expected = case["expected_output"]
        transformed = actual != case["input"]
        passed = bool(
            actual == expected
            and transformed is bool(case["expected_transformed"])
            and sanitize_comparison_judge_value(actual) == actual
        )
        rows.append(
            {
                "case_id": str(case["id"]),
                "family": str(case["family"]),
                "input_sha256": stable_digest(case["input"]),
                "expected_output_sha256": stable_digest(expected),
                "actual_output_sha256": stable_digest(actual),
                "transformed": transformed,
                "passed": passed,
            }
        )
    summary = {
        "cases": len(rows),
        "credential_redaction_cases": sum(
            row["family"] == "credential_redaction" for row in rows
        ),
        "safe_preservation_cases": sum(
            row["family"] == "safe_placeholder_preservation" for row in rows
        ),
        "passed": sum(bool(row["passed"]) for row in rows),
    }
    return {
        "status": "passed" if summary["passed"] == summary["cases"] else "failed",
        "cases": rows,
        "summary": summary,
    }


def build_compatibility(repo_root: Path) -> dict[str, Any]:
    report_path = repo_root / V1_REPORT_PATH
    result_path = repo_root / V1_ARCHIVE_RESULT_PATH
    cases_path = repo_root / CAMPAIGN_DIR / "judge-calibration-cases.jsonl"
    rubric_path = repo_root / CAMPAIGN_DIR / "judge-rubric.json"
    report = _json(report_path)
    result = _json(result_path)
    result_unsigned = dict(result)
    result_digest = str(result_unsigned.pop("result_digest", ""))
    if result_digest != stable_digest(result_unsigned):
        raise ValueError("archived V1 calibration result digest disagrees")
    gate = report.get("execution_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("V1 calibration report has no execution gate")
    if result_digest != result.get("result_digest"):
        raise ValueError("V1 calibration result identity is invalid")
    rows = _jsonl(cases_path)
    rubric = _json(rubric_path)
    case_receipts = _case_receipts(rows=rows, rubric=rubric)
    offline = _offline_conformance()
    summary = {
        "cases": len(case_receipts),
        "byte_equivalent_payloads": sum(
            row["source_payload_sha256"] == row["provider_payload_sha256"]
            for row in case_receipts
        ),
        "byte_equivalent_prompts": sum(
            row["historical_prompt_sha256"] == row["current_prompt_sha256"]
            for row in case_receipts
        ),
        "transformed_payloads": sum(
            bool(row["transformed"]) for row in case_receipts
        ),
    }
    passed = bool(
        summary
        == {
            "cases": 48,
            "byte_equivalent_payloads": 48,
            "byte_equivalent_prompts": 48,
            "transformed_payloads": 0,
        }
        and offline["status"] == "passed"
    )
    unsigned = {
        "schema_version": 2,
        "kind": "comparison_judge_sanitizer_compatibility",
        "status": "passed" if passed else "failed",
        "claim_scope": "no_new_model_payload_compatibility",
        "source_calibration": {
            "report_path": V1_REPORT_PATH.as_posix(),
            "report_sha256": _file_sha256(report_path),
            "historical_runtime_result_path": V1_RUNTIME_RESULT_PATH.as_posix(),
            "archived_result_path": V1_ARCHIVE_RESULT_PATH.as_posix(),
            "archived_result_sha256": _file_sha256(result_path),
            "result_digest": result_digest,
            "preview_digest": str(result["preview_digest"]),
            "approval_digest": str(result["approval_digest"]),
            "model": str(result["model"]),
            "cases_artifact_path": str(gate["cases_artifact_path"]),
            "cases_artifact_sha256": str(gate["cases_artifact_sha256"]),
            "cases_digest": str(gate["cases_digest"]),
            "rubric_digest": str(gate["rubric_digest"]),
            "runner_artifact_path": str(gate["runner_artifact_path"]),
            "runner_artifact_sha256": str(gate["runner_artifact_sha256"]),
        },
        "sanitizer": {
            "version": COMPARISON_JUDGE_INPUT_SANITIZER_VERSION,
            "contract": COMPARISON_JUDGE_INPUT_SANITIZER_CONTRACT,
            "transform": COMPARISON_JUDGE_INPUT_SANITIZER_TRANSFORM,
            "implementation_path": SANITIZER_PATH.as_posix(),
            "implementation_sha256": (
                COMPARISON_JUDGE_INPUT_SANITIZER_IMPLEMENTATION_SHA256
            ),
            "builder_artifact_path": BUILDER_PATH.as_posix(),
            "builder_artifact_sha256": _file_sha256(repo_root / BUILDER_PATH),
        },
        "cases": case_receipts,
        "summary": summary,
        "offline_conformance": offline,
        "limitations": [
            "No model request was rerun; V2 reuses the immutable V1 responses.",
            "The historical runner did not persist request bodies, so exact prompts are deterministically reconstructed from its locked code, cases, and rubric.",
            "Offline redaction cases validate the sanitizer implementation, not the model's response to transformed inputs.",
            "The judge remains advisory until a two-reviewer human calibration is adjudicated.",
        ],
    }
    return {**unsigned, "receipt_digest": stable_digest(unsigned)}


def build_v2_report(
    *, v1_report: Mapping[str, Any], compatibility: Mapping[str, Any], repo_root: Path
) -> dict[str, Any]:
    report = json.loads(json.dumps(v1_report))
    report["schema_version"] = 2
    gate = report["execution_gate"]
    gate["kind"] = "synthetic_blinded_advisory_v2"
    gate["result_path"] = V1_ARCHIVE_RESULT_PATH.as_posix()
    gate.update(
        {
            "compatibility_artifact_path": V2_COMPATIBILITY_PATH.as_posix(),
            "compatibility_artifact_sha256": _file_sha256(
                repo_root / V2_COMPATIBILITY_PATH
            ),
            "compatibility_receipt_digest": compatibility["receipt_digest"],
            "sanitizer_version": COMPARISON_JUDGE_INPUT_SANITIZER_VERSION,
            "sanitizer_contract": COMPARISON_JUDGE_INPUT_SANITIZER_CONTRACT,
            "sanitizer_transform": COMPARISON_JUDGE_INPUT_SANITIZER_TRANSFORM,
            "sanitizer_implementation_sha256": (
                COMPARISON_JUDGE_INPUT_SANITIZER_IMPLEMENTATION_SHA256
            ),
        }
    )
    report["note"] = (
        "The immutable V1 48-case model result remains advisory. V2 performs no "
        "new model calls: it binds the current sanitizer, proves all 48 locked "
        "provider payloads are byte-equivalent, and separately validates "
        "redaction and safe-placeholder conformance. Human calibration remains "
        "pending."
    )
    return report


def _write_or_check(path: Path, value: Mapping[str, Any], *, check: bool) -> None:
    expected = _canonical_bytes(value)
    if check:
        if not path.is_file() or path.read_bytes() != expected:
            raise ValueError(f"generated artifact drifted: {path}")
        return
    path.write_bytes(expected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    runtime_result = repo_root / V1_RUNTIME_RESULT_PATH
    archived_result = repo_root / V1_ARCHIVE_RESULT_PATH
    if args.check:
        if not archived_result.is_file():
            raise ValueError("archived V1 calibration result is unavailable")
        if runtime_result.is_file() and runtime_result.read_bytes() != archived_result.read_bytes():
            raise ValueError("runtime and archived V1 calibration results disagree")
    else:
        if not runtime_result.is_file():
            raise ValueError("runtime V1 calibration result is unavailable")
        if archived_result.is_file() and archived_result.read_bytes() != runtime_result.read_bytes():
            raise ValueError("refusing to replace a different archived V1 result")
        archived_result.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(runtime_result, archived_result)

    compatibility = build_compatibility(repo_root)
    _write_or_check(
        repo_root / V2_COMPATIBILITY_PATH,
        compatibility,
        check=args.check,
    )
    v2_report = build_v2_report(
        v1_report=_json(repo_root / V1_REPORT_PATH),
        compatibility=compatibility,
        repo_root=repo_root,
    )
    _write_or_check(repo_root / V2_REPORT_PATH, v2_report, check=args.check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
