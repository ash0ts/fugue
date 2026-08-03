#!/usr/bin/env python3
"""Run a blinded Sonnet diagnostic against authored synthetic references.

This output is deliberately separate from ``judge-calibration.json``. Synthetic
gold can expose an obviously bad rubric or model response, but it cannot replace
the required two-reviewer calibration and holdout adjudication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import validate_campaign as contract

MODEL = "anthropic/claude-sonnet-5"
SYNTHETIC_BUDGET_CEILING_USD = contract.CALIBRATION_RUN_MAXIMUM_COST_USD
CAMPAIGN_ALLOCATION_USD = contract.CALIBRATION_CAMPAIGN_ALLOCATION_USD
PRIOR_FAILED_REQUESTS = contract.CALIBRATION_PRIOR_FAILED_REQUESTS
PRIOR_ACCOUNTED_RESERVE_USD = contract.CALIBRATION_PRIOR_ACCOUNTED_RESERVE_USD
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
    exact_shape = json.dumps(
        contract.provider_response_example(),
        sort_keys=True,
        separators=(",", ":"),
    )
    dimensions = ", ".join(contract.EXPECTED_DIMENSIONS)
    top_level_fields = ", ".join(contract.PROVIDER_RESPONSE_TOP_LEVEL_FIELDS)
    prompt = (
        "Blindly judge one synthetic Skill-maintenance response. You do not know "
        "the treatment, revision, authored reference, or human labels. Use only "
        "the supplied task, response, permitted path evidence, and rubric. Return "
        "exactly one raw JSON object, with no Markdown or surrounding text. The "
        f"object must contain exactly these top-level fields and no others: "
        f"{top_level_fields}. The scores object must contain exactly these "
        f"dimensions and no others: {dimensions}. Every score and uncertainty "
        "must be a JSON number from 0 through 1, missing_evidence must be a JSON "
        "boolean, and overall_assessment and rationale must each contain 1-500 "
        "characters. Replace only the example values in this exact JSON shape: "
        f"{exact_shape}. Never return scores_explanation, hidden reasoning, or "
        "chain of thought.\n\n"
        + json.dumps(payload, sort_keys=True, separators=(",", ":"))
    )
    if len(prompt) > MAX_PROMPT_CHARACTERS:
        raise ValueError("synthetic calibration prompt exceeds the locked bound")
    return prompt


def _validate_provider_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("judge response must be an object")
    expected = set(contract.PROVIDER_RESPONSE_TOP_LEVEL_FIELDS)
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown or missing:
        raise ValueError(
            f"judge response fields do not match: unknown={unknown}, missing={missing}"
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
    for ordinal, row in enumerate(rows, start=1):
        prompt = _prompt(rubric_value, row)
        _reject_secret_values(prompt, secret_values, "synthetic judge prompt")
        payload: dict[str, Any] | None = None
        usage: dict[str, Any] = {}
        try:
            payload, usage = requester(prompt)
        except Exception as exc:
            failure = _failure_document(
                preview=preview,
                approval_digest=approval_digest,
                case_id=str(row["id"]),
                case_ordinal=ordinal,
                completed_cases=len(results),
                error=exc,
                payload=None,
                response_returned=False,
                usage=getattr(exc, "usage", {}),
            )
            _validate_failure_document(failure, preview=preview)
            failure_output = _failure_output_path(output)
            _atomic_write(failure_output, failure)
            exc.add_note(f"failure receipt: {failure_output.as_posix()}")
            raise
        try:
            validated = _validate_provider_payload(payload)
        except (TypeError, ValueError) as exc:
            failure = _failure_document(
                preview=preview,
                approval_digest=approval_digest,
                case_id=str(row["id"]),
                case_ordinal=ordinal,
                completed_cases=len(results),
                error=exc,
                payload=payload,
                response_returned=True,
                usage=usage,
            )
            _validate_failure_document(failure, preview=preview)
            failure_output = _failure_output_path(output)
            _atomic_write(failure_output, failure)
            exc.add_note(f"failure receipt: {failure_output.as_posix()}")
            raise
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
            runner_artifact_path=str(preview["runner_artifact_path"]),
            runner_artifact_sha256=str(preview["runner_artifact_sha256"]),
            response_schema_digest=str(preview["response_schema_digest"]),
            response_request_mode=str(preview["response_request_mode"]),
            response_validator_version=int(preview["response_validator_version"]),
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
        runner_artifact_path=str(preview["runner_artifact_path"]),
        runner_artifact_sha256=str(preview["runner_artifact_sha256"]),
        response_schema_digest=str(preview["response_schema_digest"]),
        response_request_mode=str(preview["response_request_mode"]),
        response_validator_version=int(preview["response_validator_version"]),
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
    runner_artifact_path: str,
    runner_artifact_sha256: str,
    response_schema_digest: str,
    response_request_mode: str,
    response_validator_version: int,
    rows: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    preview_digest: str | None,
    approval_digest: str | None,
) -> dict[str, Any]:
    complete = len(results) == len(rows) == 48
    run_accounted_cost = round(
        len(results) * SYNTHETIC_BUDGET_CEILING_USD / MAX_REQUESTS,
        6,
    )
    cumulative_accounted_cost = round(
        PRIOR_ACCOUNTED_RESERVE_USD + run_accounted_cost,
        6,
    )
    receipt_unsigned = {
        "schema_version": 1,
        "role": "advisory_synthetic_calibration",
        "model": MODEL,
        "cases_artifact_path": cases_artifact_path,
        "cases_artifact_sha256": cases_artifact_sha256,
        "cases_digest": cases_digest,
        "rubric_digest": rubric_digest,
        "runner_artifact_path": runner_artifact_path,
        "runner_artifact_sha256": runner_artifact_sha256,
        "response_schema_digest": response_schema_digest,
        "response_request_mode": response_request_mode,
        "response_validator_version": response_validator_version,
        "request_count": len(results),
        "maximum_requests": MAX_REQUESTS,
        "maximum_prompt_characters": MAX_PROMPT_CHARACTERS,
        "maximum_output_tokens_per_request": MAX_OUTPUT_TOKENS_PER_REQUEST,
        "automatic_retries": 0,
        "campaign_allocation_usd": CAMPAIGN_ALLOCATION_USD,
        "prior_failed_requests": PRIOR_FAILED_REQUESTS,
        "prior_accounted_reserve_usd": PRIOR_ACCOUNTED_RESERVE_USD,
        "run_accounted_cost_usd": run_accounted_cost,
        "cumulative_accounted_cost_usd": cumulative_accounted_cost,
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
        "runner_artifact_path": runner_artifact_path,
        "runner_artifact_sha256": runner_artifact_sha256,
        "response_schema_digest": response_schema_digest,
        "response_request_mode": response_request_mode,
        "response_validator_version": response_validator_version,
        "preview_digest": preview_digest,
        "approval_digest": approval_digest,
        "requested_cases": len(rows),
        "completed_cases": len(results),
        "budget_ceiling_usd": SYNTHETIC_BUDGET_CEILING_USD,
        "observed_cost_usd": None,
        "accounted_cost_usd": run_accounted_cost,
        "campaign_allocation_usd": CAMPAIGN_ALLOCATION_USD,
        "prior_failed_requests": PRIOR_FAILED_REQUESTS,
        "prior_accounted_reserve_usd": PRIOR_ACCOUNTED_RESERVE_USD,
        "cumulative_accounted_cost_usd": cumulative_accounted_cost,
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


def _safe_usage(value: Any) -> dict[str, int | None]:
    supplied = value if isinstance(value, Mapping) else {}
    normalized: dict[str, int | None] = {}
    for field in ("input_tokens", "output_tokens"):
        amount = supplied.get(field)
        normalized[field] = (
            amount
            if isinstance(amount, int) and not isinstance(amount, bool) and amount >= 0
            else None
        )
    return normalized


def _failure_output_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".failure.json")


def _failure_document(
    *,
    preview: Mapping[str, Any],
    approval_digest: str | None,
    case_id: str,
    case_ordinal: int,
    completed_cases: int,
    error: Exception,
    payload: Any,
    response_returned: bool,
    usage: Any,
) -> dict[str, Any]:
    if response_returned:
        canonical_response = json.dumps(
            payload,
            default=lambda _: "<unsupported-json-value>",
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        response_sha256: str | None = hashlib.sha256(
            canonical_response.encode()
        ).hexdigest()
        response_characters: int | None = len(canonical_response)
        error_code = "strict_response_contract_rejected"
    else:
        supplied_sha256 = str(getattr(error, "response_sha256", "") or "")
        supplied_characters = getattr(error, "response_characters", None)
        response_sha256 = (
            supplied_sha256 if contract.SHA256.fullmatch(supplied_sha256) else None
        )
        response_characters = (
            supplied_characters
            if isinstance(supplied_characters, int)
            and not isinstance(supplied_characters, bool)
            and supplied_characters >= 0
            else None
        )
        supplied_code = str(getattr(error, "code", "") or "")
        error_code = (
            supplied_code
            if supplied_code
            and len(supplied_code) <= 64
            and all(
                character.isalnum() or character == "_" for character in supplied_code
            )
            else "provider_request_failed"
        )
    attempted_requests = completed_cases + 1
    run_accounted_reserve = round(
        attempted_requests * SYNTHETIC_BUDGET_CEILING_USD / MAX_REQUESTS,
        6,
    )
    cumulative_accounted_reserve = round(
        PRIOR_ACCOUNTED_RESERVE_USD + run_accounted_reserve,
        6,
    )
    unsigned = {
        "schema_version": 1,
        "kind": "synthetic_judge_request_failure",
        "status": "failed",
        "model": MODEL,
        "case_id": case_id,
        "case_ordinal": case_ordinal,
        "completed_cases": completed_cases,
        "attempted_requests": attempted_requests,
        "maximum_requests": MAX_REQUESTS,
        "automatic_retries": 0,
        "error_code": error_code,
        "response_sha256": response_sha256,
        "response_characters": response_characters,
        "usage": _safe_usage(usage),
        "preview_digest": preview["preview_digest"],
        "approval_digest": approval_digest,
        "runner_artifact_path": preview["runner_artifact_path"],
        "runner_artifact_sha256": preview["runner_artifact_sha256"],
        "response_schema_digest": preview["response_schema_digest"],
        "response_request_mode": preview["response_request_mode"],
        "response_validator_version": preview["response_validator_version"],
        "campaign_allocation_usd": CAMPAIGN_ALLOCATION_USD,
        "prior_failed_requests": PRIOR_FAILED_REQUESTS,
        "prior_accounted_reserve_usd": PRIOR_ACCOUNTED_RESERVE_USD,
        "run_accounted_reserve_usd": run_accounted_reserve,
        "cumulative_accounted_reserve_usd": cumulative_accounted_reserve,
        "raw_response_persisted": False,
    }
    return {**unsigned, "failure_digest": contract.stable_digest(unsigned)}


def _validate_failure_document(
    value: Mapping[str, Any], *, preview: Mapping[str, Any]
) -> None:
    expected_fields = {
        "schema_version",
        "kind",
        "status",
        "model",
        "case_id",
        "case_ordinal",
        "completed_cases",
        "attempted_requests",
        "maximum_requests",
        "automatic_retries",
        "error_code",
        "response_sha256",
        "response_characters",
        "usage",
        "preview_digest",
        "approval_digest",
        "runner_artifact_path",
        "runner_artifact_sha256",
        "response_schema_digest",
        "response_request_mode",
        "response_validator_version",
        "campaign_allocation_usd",
        "prior_failed_requests",
        "prior_accounted_reserve_usd",
        "run_accounted_reserve_usd",
        "cumulative_accounted_reserve_usd",
        "raw_response_persisted",
        "failure_digest",
    }
    if set(value) != expected_fields:
        raise ValueError("synthetic judge failure receipt fields do not match")
    unsigned = dict(value)
    supplied_digest = str(unsigned.pop("failure_digest", "") or "")
    if not contract.SHA256.fullmatch(supplied_digest) or supplied_digest != (
        contract.stable_digest(unsigned)
    ):
        raise ValueError("synthetic judge failure receipt digest disagrees")
    for field in (
        "preview_digest",
        "runner_artifact_path",
        "runner_artifact_sha256",
        "response_schema_digest",
        "response_request_mode",
        "response_validator_version",
    ):
        if value[field] != preview[field]:
            raise ValueError(f"synthetic judge failure receipt {field} drifted")
    if (
        value["schema_version"] != 1
        or value["kind"] != "synthetic_judge_request_failure"
        or value["status"] != "failed"
        or value["model"] != MODEL
        or value["automatic_retries"] != 0
        or value["maximum_requests"] != MAX_REQUESTS
        or value["raw_response_persisted"] is not False
    ):
        raise ValueError("synthetic judge failure receipt contract disagrees")
    attempted = value["attempted_requests"]
    if (
        not isinstance(attempted, int)
        or isinstance(attempted, bool)
        or attempted != value["completed_cases"] + 1
        or value["case_ordinal"] != attempted
        or not 1 <= attempted <= MAX_REQUESTS
    ):
        raise ValueError("synthetic judge failure receipt request count disagrees")
    expected_run_reserve = round(
        attempted * SYNTHETIC_BUDGET_CEILING_USD / MAX_REQUESTS,
        6,
    )
    expected_cumulative = round(
        PRIOR_ACCOUNTED_RESERVE_USD + expected_run_reserve,
        6,
    )
    if (
        value["campaign_allocation_usd"] != CAMPAIGN_ALLOCATION_USD
        or value["prior_failed_requests"] != PRIOR_FAILED_REQUESTS
        or value["prior_accounted_reserve_usd"] != PRIOR_ACCOUNTED_RESERVE_USD
        or value["run_accounted_reserve_usd"] != expected_run_reserve
        or value["cumulative_accounted_reserve_usd"] != expected_cumulative
        or expected_cumulative > CAMPAIGN_ALLOCATION_USD
    ):
        raise ValueError("synthetic judge failure receipt cost contract disagrees")


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
    runner = contract.calibration_runner_artifact(root)
    return contract.calibration_preview_document(
        cases_artifact_path=summary["calibration_cases_artifact_path"],
        cases_artifact_sha256=summary["calibration_cases_artifact_sha256"],
        cases_digest=summary["calibration_digest"],
        rubric_digest=summary["rubric_digest"],
        runner_artifact_path=runner["path"],
        runner_artifact_sha256=runner["sha256"],
        response_schema_digest=contract.provider_response_schema_digest(),
        response_request_mode=contract.CALIBRATION_RESPONSE_REQUEST_MODE,
        response_validator_version=contract.PROVIDER_RESPONSE_VALIDATOR_VERSION,
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
        parser.error("paid execution requires --env-file, --output, and --approval")

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
        return request_json_judge(
            model=MODEL,
            env=env,
            prompt=prompt,
            response_schema=contract.provider_response_schema(),
        )

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
