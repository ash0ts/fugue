#!/usr/bin/env python3
"""Run a blinded Sonnet diagnostic against authored synthetic references.

This output is deliberately separate from ``judge-calibration.json``. Synthetic
gold can expose an obviously bad rubric or model response, but it cannot replace
the required two-reviewer calibration and holdout adjudication.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import validate_campaign as contract

MODEL = "anthropic/claude-sonnet-5"
SYNTHETIC_BUDGET_CEILING_USD = 8
MAX_REQUESTS = 48
MAX_PROMPT_CHARACTERS = 12_000
MAX_OUTPUT_TOKENS_PER_REQUEST = 1_200


def _prompt(rubric: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    payload = {
        "public_task": row["public_task"],
        "response": row["response"],
        "permitted_evidence": row["permitted_evidence"],
        "rubric": {
            "text": rubric["rubric"],
            "dimensions": rubric["dimensions"],
            "label_bands": [
                {
                    "id": item["id"],
                    "score_band": item["score_band"],
                    "description": item["description"],
                }
                for item in rubric["labels"]
            ],
        },
    }
    prompt = (
        "Blindly judge one synthetic Skill-maintenance response. You do not know "
        "the treatment, revision, authored reference, or human labels. Use only "
        "the supplied task, response, permitted path evidence, and rubric. Return "
        "one JSON object with: scores (one 0..1 number per dimension), "
        "overall_assessment (1..500 characters), uncertainty (0..1), "
        "missing_evidence (boolean), and rationale (1..500 characters). Do not "
        "return hidden reasoning or chain of thought.\n\n"
        + json.dumps(payload, sort_keys=True, separators=(",", ":"))
    )
    if len(prompt) > MAX_PROMPT_CHARACTERS:
        raise ValueError("synthetic calibration prompt exceeds the locked bound")
    return prompt


def _validate_provider_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("judge response must be an object")
    expected = {
        "scores",
        "overall_assessment",
        "uncertainty",
        "missing_evidence",
        "rationale",
    }
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown or missing:
        raise ValueError(
            "judge response fields do not match: "
            f"unknown={unknown}, missing={missing}"
        )
    scores = value["scores"]
    if not isinstance(scores, Mapping) or set(scores) != set(
        contract.EXPECTED_DIMENSIONS
    ):
        raise ValueError("judge response dimensions do not match the rubric")
    normalized_scores: dict[str, float] = {}
    for dimension in contract.EXPECTED_DIMENSIONS:
        score = scores[dimension]
        if (
            not isinstance(score, int | float)
            or isinstance(score, bool)
            or not 0 <= float(score) <= 1
        ):
            raise ValueError(f"judge score {dimension} must be between zero and one")
        normalized_scores[dimension] = float(score)
    assessment = str(value["overall_assessment"]).strip()
    rationale = str(value["rationale"]).strip()
    uncertainty = value["uncertainty"]
    missing_evidence = value["missing_evidence"]
    if not assessment or len(assessment) > 500:
        raise ValueError("judge overall_assessment must contain 1-500 characters")
    if not rationale or len(rationale) > 500:
        raise ValueError("judge rationale must contain 1-500 characters")
    if (
        not isinstance(uncertainty, int | float)
        or isinstance(uncertainty, bool)
        or not 0 <= float(uncertainty) <= 1
    ):
        raise ValueError("judge uncertainty must be between zero and one")
    if not isinstance(missing_evidence, bool):
        raise ValueError("judge missing_evidence must be boolean")
    mean_score = sum(normalized_scores.values()) / len(normalized_scores)
    return {
        "dimension_scores": normalized_scores,
        "overall_label": contract.label_from_mean(mean_score),
        "reason": assessment,
        "rationale": rationale,
        "uncertainty": float(uncertainty),
        "missing_evidence": missing_evidence,
    }


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _synthetic_metrics(
    rows: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return contract.recompute_synthetic_metrics(rows, results)


def run_synthetic_calibration(
    *,
    root: Path,
    output: Path,
    requester: Callable[[str], tuple[dict[str, Any], dict[str, Any]]],
    secret_values: Sequence[str] = (),
    preview_digest: str | None = None,
    approval_digest: str | None = None,
) -> dict[str, Any]:
    summary = contract.validate_all(root)
    manifest = contract._load_json(root / "campaign-manifest.json", "campaign manifest")
    rubric_value = contract._load_json(
        root / manifest["calibration"]["rubric"], "judge rubric"
    )
    rubric_summary = contract.validate_rubric(rubric_value)
    rows = contract.read_cases(root / manifest["calibration"]["cases"])
    if len(rows) != MAX_REQUESTS:
        raise ValueError("synthetic calibration must make exactly 48 requests")
    cases_summary = contract.validate_cases(rows)
    preview = calibration_preview(root)
    resolved_preview_digest = preview_digest or str(preview["preview_digest"])
    if resolved_preview_digest != preview["preview_digest"]:
        raise ValueError("synthetic calibration preview digest drifted")
    if approval_digest is not None and not contract.SHA256.fullmatch(approval_digest):
        raise ValueError("synthetic calibration approval digest is invalid")
    results: list[dict[str, Any]] = []
    for row in rows:
        prompt = _prompt(rubric_value, row)
        _reject_secret_values(prompt, secret_values, "synthetic judge prompt")
        payload, usage = requester(prompt)
        validated = _validate_provider_payload(payload)
        _reject_secret_values(
            json.dumps(validated, sort_keys=True),
            secret_values,
            "synthetic judge result",
        )
        results.append(
            {
                "case_id": row["id"],
                "repository_id": row["repository_id"],
                "split": row["split"],
                **validated,
                "usage": {
                    "input_tokens": usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                },
            }
        )
        partial = _result_document(
            cases_artifact_path=summary["calibration_cases_artifact_path"],
            cases_artifact_sha256=summary["calibration_cases_artifact_sha256"],
            cases_digest=cases_summary["cases_digest"],
            rubric_digest=rubric_summary["contract_digest"],
            rows=rows,
            results=results,
            preview_digest=resolved_preview_digest,
            approval_digest=approval_digest,
        )
        contract.validate_synthetic_result(
            root=root,
            value=partial,
            require_complete=False,
            require_approval=approval_digest is not None,
        )
        _atomic_write(output, partial)
    final = _result_document(
        cases_artifact_path=summary["calibration_cases_artifact_path"],
        cases_artifact_sha256=summary["calibration_cases_artifact_sha256"],
        cases_digest=cases_summary["cases_digest"],
        rubric_digest=rubric_summary["contract_digest"],
        rows=rows,
        results=results,
        preview_digest=resolved_preview_digest,
        approval_digest=approval_digest,
    )
    contract.validate_synthetic_result(
        root=root,
        value=final,
        require_complete=True,
        require_approval=approval_digest is not None,
    )
    _atomic_write(output, final)
    return final


def _result_document(
    *,
    cases_artifact_path: str,
    cases_artifact_sha256: str,
    cases_digest: str,
    rubric_digest: str,
    rows: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    preview_digest: str | None,
    approval_digest: str | None,
) -> dict[str, Any]:
    complete = len(results) == len(rows) == 48
    receipt_unsigned = {
        "schema_version": 1,
        "role": "advisory_synthetic_calibration",
        "model": MODEL,
        "cases_artifact_path": cases_artifact_path,
        "cases_artifact_sha256": cases_artifact_sha256,
        "cases_digest": cases_digest,
        "rubric_digest": rubric_digest,
        "request_count": len(results),
        "maximum_requests": MAX_REQUESTS,
        "maximum_prompt_characters": MAX_PROMPT_CHARACTERS,
        "maximum_output_tokens_per_request": MAX_OUTPUT_TOKENS_PER_REQUEST,
        "automatic_retries": 0,
        "preview_digest": preview_digest,
        "approval_digest": approval_digest,
        "serialized_input_fields": [
            "public_task",
            "response",
            "permitted_evidence",
            "rubric",
        ],
        "secret_value_scan": "passed",
    }
    receipt = {
        **receipt_unsigned,
        "receipt_digest": contract.stable_digest(receipt_unsigned),
    }
    unsigned = {
        "schema_version": 1,
        "kind": "synthetic_gold_diagnostic",
        "status": "completed" if complete else "incomplete",
        "model": MODEL,
        "cases_artifact_path": cases_artifact_path,
        "cases_artifact_sha256": cases_artifact_sha256,
        "cases_digest": cases_digest,
        "rubric_digest": rubric_digest,
        "preview_digest": preview_digest,
        "approval_digest": approval_digest,
        "requested_cases": len(rows),
        "completed_cases": len(results),
        "budget_ceiling_usd": SYNTHETIC_BUDGET_CEILING_USD,
        "observed_cost_usd": None,
        "accounted_cost_usd": round(
            len(results) * SYNTHETIC_BUDGET_CEILING_USD / MAX_REQUESTS,
            6,
        ),
        "results": list(results),
        "synthetic_metrics": _synthetic_metrics(rows, results),
        "human_review_status": "pending_human_review",
        "human_calibration_satisfied": False,
        "receipt": receipt,
        "note": (
            "Passing synthetic authored-reference metrics unlock advisory "
            "same-family judging for the three canaries. They do not update or "
            "satisfy the two-reviewer judge-qualification report."
        ),
    }
    return {**unsigned, "result_digest": contract.stable_digest(unsigned)}


def _reject_secret_values(text: str, secrets: Sequence[str], label: str) -> None:
    for secret in secrets:
        if secret and len(secret) >= 8 and secret in text:
            raise ValueError(f"{label} contains a configured secret value")


def dry_run_summary(root: Path) -> dict[str, Any]:
    summary = contract.validate_all(root)
    preview = calibration_preview(root)
    return {
        "schema_version": 1,
        "status": "dry_run",
        "model": MODEL,
        "requests": MAX_REQUESTS,
        "maximum_prompt_characters": MAX_PROMPT_CHARACTERS,
        "maximum_output_tokens_per_request": MAX_OUTPUT_TOKENS_PER_REQUEST,
        "automatic_retries": 0,
        "budget_ceiling_usd": SYNTHETIC_BUDGET_CEILING_USD,
        "calibration_status": summary["calibration_status"],
        "human_calibration_satisfied": False,
        "preview_digest": preview["preview_digest"],
    }


def calibration_preview(root: Path) -> dict[str, Any]:
    summary = contract.validate_all(root)
    return contract.calibration_preview_document(
        cases_artifact_path=summary["calibration_cases_artifact_path"],
        cases_artifact_sha256=summary["calibration_cases_artifact_sha256"],
        cases_digest=summary["calibration_digest"],
        rubric_digest=summary["rubric_digest"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=contract.ROOT)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--approval")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--confirm-paid-synthetic-calibration", action="store_true")
    args = parser.parse_args()
    if args.preview:
        print(json.dumps(calibration_preview(args.root.resolve()), sort_keys=True))
        return 0
    if args.dry_run:
        print(json.dumps(dry_run_summary(args.root.resolve()), sort_keys=True))
        return 0
    if args.env_file is None or args.output is None or args.approval is None:
        parser.error(
            "paid execution requires --env-file, --output, and --approval"
        )

    from fugue.bench.evaluations import request_json_judge
    from fugue.bench.operator import load_env
    from fugue.redaction import secrets_from_env
    from fugue.research.approvals import ApprovalLedger
    from fugue.research.store import StudyStore

    env = load_env(args.env_file.resolve())
    preview = calibration_preview(args.root.resolve())
    repo_root = args.root.resolve().parents[2]
    ApprovalLedger(StudyStore(repo_root).path).claim(
        approval_digest=args.approval,
        subject_kind="experiment",
        preview_digest=preview["preview_digest"],
        subject_id=f"calibration-{preview['preview_digest'][:20]}",
        estimated_cells=MAX_REQUESTS,
        estimated_cost_usd=SYNTHETIC_BUDGET_CEILING_USD,
    )

    def requester(prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
        return request_json_judge(model=MODEL, env=env, prompt=prompt)

    result = run_synthetic_calibration(
        root=args.root.resolve(),
        output=args.output.resolve(),
        requester=requester,
        secret_values=tuple(secrets_from_env(env)),
        preview_digest=preview["preview_digest"],
        approval_digest=args.approval,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "completed_cases": result["completed_cases"],
                "synthetic_thresholds_passed": result["synthetic_metrics"][
                    "synthetic_thresholds_passed"
                ],
                "human_calibration_satisfied": False,
                "output": args.output.resolve().as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
