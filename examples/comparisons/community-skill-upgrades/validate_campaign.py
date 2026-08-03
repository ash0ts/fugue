#!/usr/bin/env python3
"""Strict offline validation for the community Skill upgrade campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
HEX_SHA = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")
MIN_RATE = 0.85
EXPECTED_DIMENSIONS = (
    "useful_actionability",
    "repository_grounding",
    "reviewability",
    "risk_calibration",
)
EXPECTED_EVIDENCE = ("inspected_paths", "changed_paths")
EXPECTED_REPOSITORIES = (
    "superpowers-writing-plans",
    "anthropic-skill-creator",
    "vercel-react-best-practices",
)
LABELS = ("unusable", "weak", "adequate", "strong", "exceptional")
PASSING_LABELS = {"strong", "exceptional"}
CALIBRATION_CAMPAIGN_ALLOCATION_USD = 8.0
CALIBRATION_PRIOR_FAILED_REQUESTS = 1
CALIBRATION_PRIOR_ACCOUNTED_RESERVE_USD = 0.166667
CALIBRATION_RUN_MAXIMUM_COST_USD = 7.833333
CALIBRATION_RESPONSE_REQUEST_MODE = "anthropic_json_schema_v1"
PROVIDER_RESPONSE_VALIDATOR_VERSION = 1
RUNNER_ARTIFACT_PATH = (
    "examples/comparisons/community-skill-upgrades/run_synthetic_calibration.py"
)
PROVIDER_RESPONSE_TOP_LEVEL_FIELDS = (
    "scores",
    "overall_assessment",
    "uncertainty",
    "missing_evidence",
    "rationale",
)

MANIFEST_FIELDS = {
    "schema_version",
    "id",
    "title",
    "execution_order",
    "calibration",
    "studies",
    "execution",
    "budget_ledger",
    "scientific_report_template",
}
CALIBRATION_FIELDS = {
    "id",
    "rubric",
    "cases",
    "cases_sha256",
    "report",
    "expected_examples",
    "calibration_examples",
    "holdout_examples",
    "budget_usd",
    "status",
}
SYNTHETIC_RESULT_FIELDS = {
    "schema_version",
    "kind",
    "status",
    "model",
    "cases_artifact_path",
    "cases_artifact_sha256",
    "cases_digest",
    "rubric_digest",
    "runner_artifact_path",
    "runner_artifact_sha256",
    "response_schema_digest",
    "response_request_mode",
    "response_validator_version",
    "preview_digest",
    "approval_digest",
    "requested_cases",
    "completed_cases",
    "budget_ceiling_usd",
    "observed_cost_usd",
    "accounted_cost_usd",
    "campaign_allocation_usd",
    "prior_failed_requests",
    "prior_accounted_reserve_usd",
    "cumulative_accounted_cost_usd",
    "results",
    "synthetic_metrics",
    "human_review_status",
    "human_calibration_satisfied",
    "receipt",
    "note",
    "result_digest",
}
SYNTHETIC_RESULT_ROW_FIELDS = {
    "case_id",
    "repository_id",
    "split",
    "dimension_scores",
    "overall_label",
    "reason",
    "rationale",
    "uncertainty",
    "missing_evidence",
    "usage",
}
SYNTHETIC_RECEIPT_FIELDS = {
    "schema_version",
    "role",
    "model",
    "cases_artifact_path",
    "cases_artifact_sha256",
    "cases_digest",
    "rubric_digest",
    "runner_artifact_path",
    "runner_artifact_sha256",
    "response_schema_digest",
    "response_request_mode",
    "response_validator_version",
    "request_count",
    "maximum_requests",
    "maximum_prompt_characters",
    "maximum_output_tokens_per_request",
    "automatic_retries",
    "campaign_allocation_usd",
    "prior_failed_requests",
    "prior_accounted_reserve_usd",
    "run_accounted_cost_usd",
    "cumulative_accounted_cost_usd",
    "preview_digest",
    "approval_digest",
    "serialized_input_fields",
    "secret_value_scan",
    "receipt_digest",
}
STUDY_FIELDS = {
    "id",
    "spec",
    "repository",
    "skill_path",
    "baseline_commit",
    "candidate_commit",
    "evidence_project",
    "task_count",
    "variant_count",
    "attempts",
    "expected_cells",
    "budget_usd",
    "status",
}
EXECUTION_FIELDS = {
    "backend",
    "harness",
    "model",
    "concurrency",
    "checkpoint_cells",
    "preparation_required",
    "approval_required",
    "cumulative_budget_usd",
    "stop_on_checkpoint_failure",
}
RUBRIC_FIELDS = {
    "schema_version",
    "id",
    "profile",
    "role",
    "rubric",
    "dimensions",
    "evidence",
    "labels",
    "passing_labels",
    "blinded_fields",
    "critical_policy",
}
LABEL_FIELDS = {"id", "transport_value", "score_band", "description"}
CASE_FIELDS = {
    "id",
    "repository_id",
    "scenario_id",
    "split",
    "public_task",
    "response",
    "permitted_evidence",
    "authored_reference",
    "judge_result",
    "reviews",
    "adjudicated_label",
}
LEDGER_FIELDS = {
    "schema_version",
    "campaign_id",
    "currency",
    "campaign_ceiling_usd",
    "status",
    "allocations",
    "total_allocated_usd",
    "total_approved_usd",
    "total_observed_usd",
    "note",
}
ALLOCATION_FIELDS = {
    "stage_id",
    "cap_usd",
    "approval_status",
    "approval_digest",
    "observed_usd",
}
CALIBRATION_REPORT_FIELDS = {
    "schema_version",
    "review_status",
    "reviewers_per_example",
    "disagreements_adjudicated",
    "judge_profile",
    "rubric_digest",
    "cases_digest",
    "score_threshold",
    "examples",
    "calibration_examples",
    "holdout_examples",
    "true_positive_rate",
    "true_negative_rate",
    "calibration_true_positive_rate",
    "calibration_true_negative_rate",
    "holdout_true_positive_rate",
    "holdout_true_negative_rate",
    "critical_false_passes",
    "passed",
    "distinct_reviewers",
    "disagreements",
    "execution_gate",
    "note",
}
REPORT_TEMPLATE_FIELDS = {
    "schema_version",
    "status",
    "study_id",
    "evidence_project",
    "exact_revisions",
    "task_validity",
    "behavioral_finding",
    "deterministic_results",
    "judge_results",
    "skill_use_evidence",
    "efficiency",
    "evidence_links",
    "limitations",
    "conclusion",
}


def stable_digest(value: Any) -> str:
    """Match Fugue's canonical JSON SHA-256 digest."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    """Return the digest of the exact artifact bytes, not parsed JSON content."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def provider_response_schema() -> dict[str, Any]:
    """Return the exact fail-closed JSON contract used by the paid judge."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(PROVIDER_RESPONSE_TOP_LEVEL_FIELDS),
        "properties": {
            "scores": {
                "type": "object",
                "additionalProperties": False,
                "required": list(EXPECTED_DIMENSIONS),
                "properties": {
                    dimension: {"type": "number"} for dimension in EXPECTED_DIMENSIONS
                },
            },
            "overall_assessment": {"type": "string"},
            "uncertainty": {"type": "number"},
            "missing_evidence": {"type": "boolean"},
            "rationale": {"type": "string"},
        },
    }


def provider_response_schema_digest() -> str:
    return stable_digest(provider_response_schema())


def provider_response_example() -> dict[str, Any]:
    """Return one compact valid shape for the provider prompt."""

    return {
        "scores": {dimension: 0.0 for dimension in EXPECTED_DIMENSIONS},
        "overall_assessment": "brief evidence-bounded assessment",
        "uncertainty": 0.0,
        "missing_evidence": False,
        "rationale": "brief evidence-bounded rationale",
    }


def calibration_runner_artifact(root: Path) -> dict[str, str]:
    """Bind the exact paid runner bytes that implement prompt and parsing."""

    path = (root / "run_synthetic_calibration.py").resolve()
    if not path.is_file():
        raise ValueError("synthetic calibration runner does not exist")
    repository_root = root.resolve().parents[2]
    try:
        repository_relative = path.relative_to(repository_root).as_posix()
    except ValueError as exc:
        raise ValueError(
            "synthetic calibration runner is outside the repository"
        ) from exc
    if repository_relative != RUNNER_ARTIFACT_PATH:
        raise ValueError("synthetic calibration runner path is not canonical")
    return {"path": repository_relative, "sha256": file_sha256(path)}


def calibration_preview_document(
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
) -> dict[str, Any]:
    unsigned = {
        "schema_version": 1,
        "kind": "synthetic_judge_calibration_preview",
        "id": "community-skill-judge-calibration-v1",
        "model": "anthropic/claude-sonnet-5",
        "cases_artifact_path": cases_artifact_path,
        "cases_artifact_sha256": cases_artifact_sha256,
        "cases_digest": cases_digest,
        "rubric_digest": rubric_digest,
        "runner_artifact_path": runner_artifact_path,
        "runner_artifact_sha256": runner_artifact_sha256,
        "response_schema_digest": response_schema_digest,
        "response_request_mode": response_request_mode,
        "response_validator_version": response_validator_version,
        "requests": 48,
        "maximum_prompt_characters": 12_000,
        "maximum_output_tokens_per_request": 1_200,
        "automatic_retries": 0,
        "campaign_allocation_usd": CALIBRATION_CAMPAIGN_ALLOCATION_USD,
        "prior_failed_requests": CALIBRATION_PRIOR_FAILED_REQUESTS,
        "prior_accounted_reserve_usd": (CALIBRATION_PRIOR_ACCOUNTED_RESERVE_USD),
        "maximum_cost_usd": CALIBRATION_RUN_MAXIMUM_COST_USD,
        "serialized_input_fields": [
            "public_task",
            "response",
            "permitted_evidence",
            "rubric",
        ],
        "human_calibration_satisfied": False,
    }
    return {**unsigned, "preview_digest": stable_digest(unsigned)}


def synthetic_execution_gate(
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
) -> dict[str, Any]:
    preview = calibration_preview_document(
        cases_artifact_path=cases_artifact_path,
        cases_artifact_sha256=cases_artifact_sha256,
        cases_digest=cases_digest,
        rubric_digest=rubric_digest,
        runner_artifact_path=runner_artifact_path,
        runner_artifact_sha256=runner_artifact_sha256,
        response_schema_digest=response_schema_digest,
        response_request_mode=response_request_mode,
        response_validator_version=response_validator_version,
    )
    return {
        "kind": "synthetic_blinded_advisory_v1",
        "required_before_agent_trials": True,
        "result_path": (
            ".fugue/runtime/community-skill-upgrades/judge-calibration.result.json"
        ),
        "preview_digest": preview["preview_digest"],
        "model": preview["model"],
        "cases_artifact_path": cases_artifact_path,
        "cases_artifact_sha256": cases_artifact_sha256,
        "cases_digest": cases_digest,
        "rubric_digest": rubric_digest,
        "runner_artifact_path": runner_artifact_path,
        "runner_artifact_sha256": runner_artifact_sha256,
        "response_schema_digest": response_schema_digest,
        "response_request_mode": response_request_mode,
        "response_validator_version": response_validator_version,
        "examples": 48,
        "calibration_examples": 36,
        "holdout_examples": 12,
        "minimum_true_positive_rate": MIN_RATE,
        "minimum_true_negative_rate": MIN_RATE,
        "maximum_critical_false_passes": 0,
        "campaign_allocation_usd": preview["campaign_allocation_usd"],
        "prior_failed_requests": preview["prior_failed_requests"],
        "prior_accounted_reserve_usd": preview["prior_accounted_reserve_usd"],
        "maximum_cost_usd": preview["maximum_cost_usd"],
    }


def _load_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(allowed - set(value))
    if unknown:
        raise ValueError(f"{label} has unknown field(s): {', '.join(unknown)}")
    if missing:
        raise ValueError(f"{label} is missing field(s): {', '.join(missing)}")


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _money(value: Any, label: str, *, allow_zero: bool = False) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if result < 0 or (result == 0 and not allow_zero):
        raise ValueError(
            f"{label} must be {'non-negative' if allow_zero else 'positive'}"
        )
    return result


def _safe_relative_file(root: Path, value: Any, label: str) -> Path:
    text = str(value or "")
    path = Path(text)
    if not text or path.is_absolute():
        raise ValueError(f"{label} must be a repository-relative path")
    resolved = (root / path).resolve()
    comparisons_root = root.parent.resolve()
    if not resolved.is_relative_to(comparisons_root):
        raise ValueError(f"{label} leaves the comparisons directory")
    return resolved


def frozen_cases_artifact(
    *,
    root: Path,
    calibration: Mapping[str, Any],
) -> dict[str, str]:
    """Resolve and verify the checked-in calibration case artifact."""

    cases_path = _safe_relative_file(
        root,
        calibration.get("cases"),
        "calibration cases",
    )
    if not cases_path.is_file():
        raise ValueError("calibration cases does not exist")
    expected_sha256 = str(calibration.get("cases_sha256") or "")
    if not SHA256.fullmatch(expected_sha256):
        raise ValueError("calibration cases_sha256 must be a lowercase SHA-256")
    actual_sha256 = file_sha256(cases_path)
    if actual_sha256 != expected_sha256:
        raise ValueError("calibration cases artifact SHA-256 drifted")
    repository_root = root.resolve().parents[2]
    try:
        repository_relative = cases_path.relative_to(repository_root).as_posix()
    except ValueError as exc:
        raise ValueError(
            "calibration cases artifact is outside the repository"
        ) from exc
    return {
        "path": repository_relative,
        "sha256": actual_sha256,
    }


def validate_manifest(  # noqa: C901 - strict schema validation is explicit.
    value: Mapping[str, Any],
    *,
    root: Path,
    require_specs: bool = False,
) -> dict[str, Any]:
    _reject_unknown(value, MANIFEST_FIELDS, "campaign manifest")
    if value["schema_version"] != 1:
        raise ValueError("campaign manifest must use schema_version 1")
    if value["id"] != "community-skill-upgrade-campaign-v1":
        raise ValueError("campaign manifest id is not canonical")

    calibration = value["calibration"]
    if not isinstance(calibration, Mapping):
        raise ValueError("campaign calibration must be an object")
    _reject_unknown(calibration, CALIBRATION_FIELDS, "campaign calibration")
    expected_calibration = {
        "id": "community-skill-judge-calibration-v1",
        "expected_examples": 48,
        "calibration_examples": 36,
        "holdout_examples": 12,
        "budget_usd": 8,
        "status": "pending_human_review",
    }
    for field, expected in expected_calibration.items():
        if calibration[field] != expected:
            raise ValueError(f"campaign calibration {field} must equal {expected!r}")
    for field in ("rubric", "cases", "report"):
        path = _safe_relative_file(root, calibration[field], f"calibration {field}")
        if not path.is_file():
            raise ValueError(f"calibration {field} does not exist")
    cases_artifact = frozen_cases_artifact(root=root, calibration=calibration)
    runner_artifact = calibration_runner_artifact(root)

    studies = value["studies"]
    if not isinstance(studies, list) or len(studies) != 3:
        raise ValueError("campaign must declare exactly three Studies")
    study_ids: list[str] = []
    evidence_projects: list[str] = []
    total_cells = 0
    study_budget = 0.0
    for index, study in enumerate(studies):
        if not isinstance(study, Mapping):
            raise ValueError(f"campaign study {index} must be an object")
        _reject_unknown(study, STUDY_FIELDS, f"campaign study {index}")
        study_id = str(study["id"])
        study_ids.append(study_id)
        evidence_project = str(study["evidence_project"])
        evidence_projects.append(evidence_project)
        if not evidence_project.startswith("wandb/fugue-"):
            raise ValueError(f"{study_id} has a non-campaign evidence project")
        if "news-research-agent" in evidence_project:
            raise ValueError(f"{study_id} routes to a forbidden legacy project")
        for field in ("baseline_commit", "candidate_commit"):
            if not HEX_SHA.fullmatch(str(study[field])):
                raise ValueError(f"{study_id} {field} must be an exact Git SHA")
        if study["baseline_commit"] == study["candidate_commit"]:
            raise ValueError(f"{study_id} compares identical revisions")
        expected_cells = (
            _positive_int(study["task_count"], f"{study_id} task_count")
            * _positive_int(study["variant_count"], f"{study_id} variant_count")
            * _positive_int(study["attempts"], f"{study_id} attempts")
        )
        if study["expected_cells"] != expected_cells or expected_cells != 4:
            raise ValueError(f"{study_id} must declare exactly four cells")
        if study["status"] != "blocked_on_calibration":
            raise ValueError(f"{study_id} must remain blocked on calibration")
        spec = _safe_relative_file(root, study["spec"], f"{study_id} spec")
        if require_specs and not spec.is_file():
            raise ValueError(f"{study_id} spec does not exist: {spec}")
        total_cells += expected_cells
        study_budget += _money(study["budget_usd"], f"{study_id} budget")

    if study_ids != list(value["execution_order"]):
        raise ValueError("campaign execution_order must match the declared Study order")
    if len(set(study_ids)) != 3 or len(set(evidence_projects)) != 3:
        raise ValueError("campaign Study ids and evidence projects must be unique")

    execution = value["execution"]
    if not isinstance(execution, Mapping):
        raise ValueError("campaign execution must be an object")
    _reject_unknown(execution, EXECUTION_FIELDS, "campaign execution")
    expected_execution = {
        "backend": "docker",
        "harness": "claude-code",
        "model": "anthropic/claude-sonnet-5",
        "concurrency": 1,
        "checkpoint_cells": 1,
        "preparation_required": True,
        "approval_required": True,
        "cumulative_budget_usd": 110,
        "stop_on_checkpoint_failure": True,
    }
    if dict(execution) != expected_execution:
        raise ValueError("campaign execution policy does not match the locked canary")
    total_budget = study_budget + _money(
        calibration["budget_usd"], "calibration budget"
    )
    if total_budget != float(execution["cumulative_budget_usd"]):
        raise ValueError(
            "campaign Study and calibration budgets do not sum to the ceiling"
        )
    if total_cells != 12:
        raise ValueError("campaign must contain exactly twelve Agent cells")
    return {
        "study_ids": study_ids,
        "total_cells": total_cells,
        "campaign_ceiling_usd": total_budget,
        "cases_artifact_path": cases_artifact["path"],
        "cases_artifact_sha256": cases_artifact["sha256"],
        "runner_artifact_path": runner_artifact["path"],
        "runner_artifact_sha256": runner_artifact["sha256"],
        "response_schema_digest": provider_response_schema_digest(),
        "response_request_mode": CALIBRATION_RESPONSE_REQUEST_MODE,
        "response_validator_version": PROVIDER_RESPONSE_VALIDATOR_VERSION,
    }


def validate_rubric(value: Mapping[str, Any]) -> dict[str, Any]:
    _reject_unknown(value, RUBRIC_FIELDS, "judge rubric")
    if value["schema_version"] != 1:
        raise ValueError("judge rubric must use schema_version 1")
    if value["id"] != "community-usefulness":
        raise ValueError("judge rubric id is not canonical")
    if value["profile"] != "anthropic/claude-sonnet-5":
        raise ValueError("judge profile must use the locked Sonnet 5 route")
    if value["role"] != "advisory":
        raise ValueError("community judge must remain advisory")
    if tuple(value["dimensions"]) != EXPECTED_DIMENSIONS:
        raise ValueError("judge dimensions do not match the campaign contract")
    if tuple(value["evidence"]) != EXPECTED_EVIDENCE:
        raise ValueError("judge evidence does not match the campaign contract")
    if set(value["passing_labels"]) != PASSING_LABELS:
        raise ValueError("judge passing labels must be strong and exceptional")
    labels = value["labels"]
    if not isinstance(labels, list) or len(labels) != 5:
        raise ValueError("judge rubric must define five label anchors")
    expected_values = (0.0, 0.25, 0.5, 0.75, 1.0)
    expected_bands = (
        "0.00 <= mean < 0.25",
        "0.25 <= mean < 0.50",
        "0.50 <= mean < 0.75",
        "0.75 <= mean < 0.90",
        "0.90 <= mean <= 1.00",
    )
    for index, item in enumerate(labels):
        if not isinstance(item, Mapping):
            raise ValueError("judge label anchor must be an object")
        _reject_unknown(item, LABEL_FIELDS, f"judge label {index}")
        if (
            item["id"] != LABELS[index]
            or item["transport_value"] != expected_values[index]
            or item["score_band"] != expected_bands[index]
            or not str(item["description"]).strip()
        ):
            raise ValueError(f"judge label {index} does not match its locked anchor")
    blinded = value["blinded_fields"]
    if not isinstance(blinded, list) or not {
        "variant_id",
        "candidate_id",
        "source_commit",
    }.issubset(blinded):
        raise ValueError("judge rubric does not blind treatment identity")
    if not str(value["rubric"]).strip() or not str(value["critical_policy"]).strip():
        raise ValueError("judge rubric text and critical policy are required")
    contract = {
        "schema_version": 1,
        "judge_id": value["id"],
        "profile": value["profile"],
        "rubric": value["rubric"],
        "dimensions": list(value["dimensions"]),
        "evidence": list(value["evidence"]),
    }
    return {"contract_digest": stable_digest(contract)}


def validate_study_specs(
    *,
    root: Path,
    manifest: Mapping[str, Any],
    rubric: Mapping[str, Any],
) -> None:
    expected_contract = {
        "id": rubric["id"],
        "profile": rubric["profile"],
        "rubric": rubric["rubric"],
        "dimensions": list(rubric["dimensions"]),
        "evidence": list(rubric["evidence"]),
    }
    expected_calibration = (root / "judge-calibration.json").resolve()
    for study in manifest["studies"]:
        study_id = str(study["id"])
        spec_path = _safe_relative_file(root, study["spec"], f"{study_id} spec")
        if not spec_path.is_file():
            raise ValueError(f"{study_id} spec does not exist: {spec_path}")
        raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError(f"{study_id} spec must be a YAML object")
        if raw.get("id") != study_id:
            raise ValueError(f"{study_id} spec id does not match the manifest")
        evaluators = raw.get("evaluators")
        if not isinstance(evaluators, list):
            raise ValueError(f"{study_id} evaluators must be a list")
        judges = [
            item
            for item in evaluators
            if isinstance(item, Mapping) and item.get("id") == rubric["id"]
        ]
        if len(judges) != 1:
            raise ValueError(f"{study_id} must declare exactly one shared judge")
        judge = judges[0]
        actual_contract = {
            "id": judge.get("id"),
            "profile": judge.get("profile"),
            "rubric": judge.get("rubric"),
            "dimensions": judge.get("dimensions"),
            "evidence": judge.get("evidence"),
        }
        if actual_contract != expected_contract:
            raise ValueError(f"{study_id} shared judge contract drifted")
        if judge.get("type") != "llm_judge" or judge.get("required") is not False:
            raise ValueError(f"{study_id} shared judge must remain advisory")
        calibration_path = (
            spec_path.parent / str(judge.get("calibration") or "")
        ).resolve()
        if calibration_path != expected_calibration:
            raise ValueError(f"{study_id} calibration path drifted")
        execution = raw.get("execution")
        if not isinstance(execution, Mapping):
            raise ValueError(f"{study_id} execution must be an object")
        if execution.get("evidence_project") != study["evidence_project"]:
            raise ValueError(f"{study_id} evidence project drifted")
        if execution.get("model") != manifest["execution"]["model"]:
            raise ValueError(f"{study_id} model route drifted")
        if execution.get("harnesses") != [manifest["execution"]["harness"]]:
            raise ValueError(f"{study_id} harness drifted")
        if (
            execution.get("attempts") != study["attempts"]
            or execution.get("concurrency") != manifest["execution"]["concurrency"]
            or execution.get("evidence_checkpoint_cells")
            != manifest["execution"]["checkpoint_cells"]
            or execution.get("max_cost_usd") != study["budget_usd"]
        ):
            raise ValueError(f"{study_id} execution bounds drifted")


def label_from_mean(score: float) -> str:
    if not 0 <= score <= 1:
        raise ValueError("judge mean score must be between zero and one")
    if score < 0.25:
        return "unusable"
    if score < 0.5:
        return "weak"
    if score < 0.75:
        return "adequate"
    if score < 0.9:
        return "strong"
    return "exceptional"


def read_cases(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"calibration line {number} must be a JSON object")
        rows.append(value)
    return rows


def _validate_judge_result(case_id: str, value: Any) -> tuple[str, bool] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{case_id} judge_result must be an object")
    _reject_unknown(
        value,
        {"dimension_scores", "overall_label", "reason", "missing_evidence"},
        f"{case_id} judge_result",
    )
    scores = value["dimension_scores"]
    if not isinstance(scores, Mapping) or set(scores) != set(EXPECTED_DIMENSIONS):
        raise ValueError(f"{case_id} judge dimensions do not match the rubric")
    numeric: list[float] = []
    for dimension in EXPECTED_DIMENSIONS:
        score = scores[dimension]
        if not isinstance(score, int | float) or isinstance(score, bool):
            raise ValueError(f"{case_id} {dimension} score must be numeric")
        numeric.append(float(score))
    mean = sum(numeric) / len(numeric)
    label = label_from_mean(mean)
    if value["overall_label"] != label:
        raise ValueError(f"{case_id} overall label does not match mean score")
    if not isinstance(value["missing_evidence"], bool):
        raise ValueError(f"{case_id} missing_evidence must be boolean")
    reason = str(value["reason"]).strip()
    if not reason or len(reason) > 500:
        raise ValueError(f"{case_id} judge reason must contain 1-500 characters")
    return label, bool(value["missing_evidence"])


def validate_cases(  # noqa: C901 - one pass validates cross-case invariants.
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(rows) != 48:
        raise ValueError("judge calibration must contain exactly 48 cases")
    ids: list[str] = []
    responses: list[str] = []
    scenario_splits: dict[str, set[str]] = defaultdict(set)
    scenario_labels: dict[str, list[str]] = defaultdict(list)
    repo_split_labels: dict[tuple[str, str], list[str]] = defaultdict(list)
    completed = True
    final_labels: list[str] = []
    predicted_labels: list[str] = []
    evaluated_splits: list[str] = []
    reviewer_ids: set[str] = set()
    disagreements = 0
    critical_false_passes = 0

    for index, item in enumerate(rows):
        _reject_unknown(item, CASE_FIELDS, f"calibration case {index}")
        case_id = str(item["id"])
        if not case_id:
            raise ValueError("calibration case id must be non-empty")
        ids.append(case_id)
        repository_id = str(item["repository_id"])
        if repository_id not in EXPECTED_REPOSITORIES:
            raise ValueError(f"{case_id} has an unknown repository id")
        split = str(item["split"])
        if split not in {"calibration", "holdout"}:
            raise ValueError(f"{case_id} has an invalid split")
        scenario_id = str(item["scenario_id"])
        if not scenario_id:
            raise ValueError(f"{case_id} scenario id is empty")
        scenario_splits[scenario_id].add(split)

        public_task = item["public_task"]
        if not isinstance(public_task, Mapping):
            raise ValueError(f"{case_id} public_task must be an object")
        _reject_unknown(public_task, {"question"}, f"{case_id} public_task")
        if not str(public_task["question"]).strip():
            raise ValueError(f"{case_id} question is empty")
        response = str(item["response"]).strip()
        if not response:
            raise ValueError(f"{case_id} response is empty")
        responses.append(response)

        evidence = item["permitted_evidence"]
        if not isinstance(evidence, Mapping):
            raise ValueError(f"{case_id} permitted_evidence must be an object")
        _reject_unknown(evidence, set(EXPECTED_EVIDENCE), f"{case_id} evidence")
        for field in EXPECTED_EVIDENCE:
            paths = evidence[field]
            if not isinstance(paths, list) or any(
                not isinstance(path, str) or not path or Path(path).is_absolute()
                for path in paths
            ):
                raise ValueError(f"{case_id} {field} must contain relative paths")

        reference = item["authored_reference"]
        if not isinstance(reference, Mapping):
            raise ValueError(f"{case_id} authored_reference must be an object")
        _reject_unknown(
            reference,
            {"label", "critical_false_pass"},
            f"{case_id} authored_reference",
        )
        label = str(reference["label"])
        if label not in {"pass", "fail"}:
            raise ValueError(f"{case_id} authored label is invalid")
        if not isinstance(reference["critical_false_pass"], bool):
            raise ValueError(f"{case_id} critical_false_pass must be boolean")
        scenario_labels[scenario_id].append(label)
        repo_split_labels[(repository_id, split)].append(label)

        judge = _validate_judge_result(case_id, item["judge_result"])
        reviews = item["reviews"]
        if not isinstance(reviews, list):
            raise ValueError(f"{case_id} reviews must be a list")
        if judge is None and not reviews and item["adjudicated_label"] is None:
            completed = False
            continue
        if judge is None or len(reviews) != 2:
            raise ValueError(f"{case_id} has partially completed calibration data")
        review_ids: list[str] = []
        review_labels: list[str] = []
        for review in reviews:
            if not isinstance(review, Mapping):
                raise ValueError(f"{case_id} review must be an object")
            _reject_unknown(review, {"reviewer", "label"}, f"{case_id} review")
            reviewer = str(review["reviewer"]).strip()
            review_label = str(review["label"])
            if not reviewer or review_label not in {"pass", "fail"}:
                raise ValueError(f"{case_id} review is incomplete")
            review_ids.append(reviewer)
            review_labels.append(review_label)
            reviewer_ids.add(reviewer)
        if len(set(review_ids)) != 2:
            raise ValueError(f"{case_id} requires two distinct reviewers")
        adjudicated = item["adjudicated_label"]
        if review_labels[0] != review_labels[1]:
            disagreements += 1
            if adjudicated not in {"pass", "fail"}:
                raise ValueError(f"{case_id} disagreement requires adjudication")
            final_label = str(adjudicated)
        else:
            if adjudicated not in {None, review_labels[0]}:
                raise ValueError(f"{case_id} adjudication contradicts reviewers")
            final_label = review_labels[0]
        judge_label, missing_evidence = judge
        predicted = (
            "pass" if judge_label in PASSING_LABELS and not missing_evidence else "fail"
        )
        final_labels.append(final_label)
        predicted_labels.append(predicted)
        evaluated_splits.append(split)
        if (
            bool(reference["critical_false_pass"])
            and final_label == "fail"
            and predicted == "pass"
        ):
            critical_false_passes += 1

    if len(set(ids)) != 48:
        raise ValueError("calibration case ids must be unique")
    if len(set(responses)) != 48:
        raise ValueError("calibration responses must be distinct")
    for repository_id in EXPECTED_REPOSITORIES:
        calibration_labels = repo_split_labels[(repository_id, "calibration")]
        holdout_labels = repo_split_labels[(repository_id, "holdout")]
        if Counter(calibration_labels) != {"pass": 6, "fail": 6}:
            raise ValueError(
                f"{repository_id} must have a balanced 12-case calibration"
            )
        if Counter(holdout_labels) != {"pass": 2, "fail": 2}:
            raise ValueError(f"{repository_id} must have a balanced four-case holdout")
    if any(len(splits) != 1 for splits in scenario_splits.values()):
        raise ValueError("scenario families may not cross calibration splits")
    if any(
        Counter(labels) != {"pass": 2, "fail": 2} for labels in scenario_labels.values()
    ):
        raise ValueError(
            "each scenario family must contain two passes and two failures"
        )

    rates = _classification_rates(final_labels, predicted_labels)
    split_rates = {
        split: _classification_rates(
            [
                actual
                for actual, row_split in zip(
                    final_labels, evaluated_splits, strict=True
                )
                if row_split == split
            ],
            [
                predicted
                for predicted, row_split in zip(
                    predicted_labels, evaluated_splits, strict=True
                )
                if row_split == split
            ],
        )
        for split in ("calibration", "holdout")
    }
    return {
        "completed": completed,
        "cases_digest": stable_digest([dict(row) for row in rows]),
        "examples": len(rows),
        "calibration_examples": sum(row["split"] == "calibration" for row in rows),
        "holdout_examples": sum(row["split"] == "holdout" for row in rows),
        "true_positive_rate": rates["true_positive_rate"],
        "true_negative_rate": rates["true_negative_rate"],
        "calibration_true_positive_rate": split_rates["calibration"][
            "true_positive_rate"
        ],
        "calibration_true_negative_rate": split_rates["calibration"][
            "true_negative_rate"
        ],
        "holdout_true_positive_rate": split_rates["holdout"]["true_positive_rate"],
        "holdout_true_negative_rate": split_rates["holdout"]["true_negative_rate"],
        "critical_false_passes": critical_false_passes,
        "reviewer_ids": reviewer_ids,
        "disagreements": disagreements,
    }


def _classification_rates(
    actual: Sequence[str], predicted: Sequence[str]
) -> dict[str, float]:
    positives = actual.count("pass")
    negatives = actual.count("fail")
    true_positives = sum(
        left == right == "pass" for left, right in zip(actual, predicted, strict=True)
    )
    true_negatives = sum(
        left == right == "fail" for left, right in zip(actual, predicted, strict=True)
    )
    return {
        "true_positive_rate": true_positives / positives if positives else 0.0,
        "true_negative_rate": true_negatives / negatives if negatives else 0.0,
    }


def recompute_synthetic_metrics(
    rows: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute synthetic calibration metrics from frozen private references."""

    by_id = {str(row["id"]): row for row in rows}
    actual: list[str] = []
    predicted: list[str] = []
    splits: list[str] = []
    critical_false_passes = 0
    for result in results:
        case_id = str(result["case_id"])
        if case_id not in by_id:
            raise ValueError(f"synthetic result has unknown case id {case_id!r}")
        row = by_id[case_id]
        actual_label = str(row["authored_reference"]["label"])
        predicted_label = (
            "pass"
            if result["overall_label"] in PASSING_LABELS
            and not result["missing_evidence"]
            else "fail"
        )
        actual.append(actual_label)
        predicted.append(predicted_label)
        splits.append(str(row["split"]))
        if (
            row["authored_reference"]["critical_false_pass"]
            and actual_label == "fail"
            and predicted_label == "pass"
        ):
            critical_false_passes += 1
    rates = _classification_rates(actual, predicted)
    split_rates: dict[str, dict[str, float]] = {}
    for split in ("calibration", "holdout"):
        split_actual = [
            label
            for label, row_split in zip(actual, splits, strict=True)
            if row_split == split
        ]
        split_predicted = [
            label
            for label, row_split in zip(predicted, splits, strict=True)
            if row_split == split
        ]
        split_rates[split] = _classification_rates(split_actual, split_predicted)
    complete = len(results) == len(rows) == 48
    passed = bool(
        complete
        and rates["true_positive_rate"] >= MIN_RATE
        and rates["true_negative_rate"] >= MIN_RATE
        and all(
            values["true_positive_rate"] >= MIN_RATE
            and values["true_negative_rate"] >= MIN_RATE
            for values in split_rates.values()
        )
        and critical_false_passes == 0
    )
    return {
        "examples": len(results),
        "true_positive_rate": round(rates["true_positive_rate"], 6),
        "true_negative_rate": round(rates["true_negative_rate"], 6),
        "calibration_true_positive_rate": round(
            split_rates["calibration"]["true_positive_rate"], 6
        ),
        "calibration_true_negative_rate": round(
            split_rates["calibration"]["true_negative_rate"], 6
        ),
        "holdout_true_positive_rate": round(
            split_rates["holdout"]["true_positive_rate"], 6
        ),
        "holdout_true_negative_rate": round(
            split_rates["holdout"]["true_negative_rate"], 6
        ),
        "critical_false_passes": critical_false_passes,
        "balanced_accuracy": round(
            (rates["true_positive_rate"] + rates["true_negative_rate"]) / 2,
            6,
        ),
        "synthetic_thresholds_passed": passed,
    }


def _validate_synthetic_result_row(
    value: Any,
    *,
    frozen: Mapping[str, Any],
    index: int,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"synthetic result row {index} must be an object")
    _reject_unknown(value, SYNTHETIC_RESULT_ROW_FIELDS, f"synthetic result row {index}")
    case_id = str(value["case_id"])
    if case_id != frozen["id"]:
        raise ValueError("synthetic result case sequence or identity drifted")
    if value["repository_id"] != frozen["repository_id"]:
        raise ValueError(f"synthetic result {case_id} repository drifted")
    if value["split"] != frozen["split"]:
        raise ValueError(f"synthetic result {case_id} split drifted")
    scores = value["dimension_scores"]
    if not isinstance(scores, Mapping) or set(scores) != set(EXPECTED_DIMENSIONS):
        raise ValueError(f"synthetic result {case_id} dimensions drifted")
    numeric: list[float] = []
    for dimension in EXPECTED_DIMENSIONS:
        score = scores[dimension]
        if (
            not isinstance(score, int | float)
            or isinstance(score, bool)
            or not 0 <= float(score) <= 1
        ):
            raise ValueError(f"synthetic result {case_id} score {dimension} is invalid")
        numeric.append(float(score))
    expected_label = label_from_mean(sum(numeric) / len(numeric))
    if value["overall_label"] != expected_label:
        raise ValueError(f"synthetic result {case_id} label disagrees with scores")
    for field in ("reason", "rationale"):
        text = str(value[field]).strip()
        if not text or len(text) > 500:
            raise ValueError(
                f"synthetic result {case_id} {field} must contain 1-500 characters"
            )
    uncertainty = value["uncertainty"]
    if (
        not isinstance(uncertainty, int | float)
        or isinstance(uncertainty, bool)
        or not 0 <= float(uncertainty) <= 1
    ):
        raise ValueError(f"synthetic result {case_id} uncertainty is invalid")
    if not isinstance(value["missing_evidence"], bool):
        raise ValueError(f"synthetic result {case_id} missing_evidence is invalid")
    usage = value["usage"]
    if not isinstance(usage, Mapping):
        raise ValueError(f"synthetic result {case_id} usage must be an object")
    _reject_unknown(usage, {"input_tokens", "output_tokens"}, f"{case_id} usage")
    for field in ("input_tokens", "output_tokens"):
        amount = usage[field]
        if amount is not None and (
            not isinstance(amount, int) or isinstance(amount, bool) or amount < 0
        ):
            raise ValueError(f"synthetic result {case_id} {field} is invalid")


def validate_synthetic_result(  # noqa: C901 - explicit artifact audit.
    *,
    root: Path,
    value: Any,
    require_complete: bool = True,
    require_approval: bool = True,
) -> dict[str, Any]:
    """Validate a result against the exact frozen cases and recompute its metrics."""

    if not isinstance(value, Mapping):
        raise ValueError("synthetic calibration result must be an object")
    _reject_unknown(value, SYNTHETIC_RESULT_FIELDS, "synthetic calibration result")
    supplied_digest = str(value["result_digest"] or "")
    unsigned = dict(value)
    unsigned.pop("result_digest")
    if not SHA256.fullmatch(supplied_digest) or supplied_digest != stable_digest(
        unsigned
    ):
        raise ValueError("synthetic calibration result digest disagrees")

    manifest = _load_json(root / "campaign-manifest.json", "campaign manifest")
    manifest_summary = validate_manifest(manifest, root=root)
    rubric_value = _load_json(root / manifest["calibration"]["rubric"], "judge rubric")
    rubric = validate_rubric(rubric_value)
    rows = read_cases(root / manifest["calibration"]["cases"])
    cases = validate_cases(rows)
    expected_report = build_calibration_report(
        cases,
        rubric,
        cases_artifact_path=manifest_summary["cases_artifact_path"],
        cases_artifact_sha256=manifest_summary["cases_artifact_sha256"],
        runner_artifact_path=manifest_summary["runner_artifact_path"],
        runner_artifact_sha256=manifest_summary["runner_artifact_sha256"],
        response_schema_digest=manifest_summary["response_schema_digest"],
        response_request_mode=manifest_summary["response_request_mode"],
        response_validator_version=manifest_summary["response_validator_version"],
    )
    report = _load_json(root / manifest["calibration"]["report"], "calibration report")
    validate_calibration_report(report, expected_report)
    gate = expected_report["execution_gate"]

    if value["schema_version"] != 1 or value["kind"] != "synthetic_gold_diagnostic":
        raise ValueError("synthetic calibration result identity is unsupported")
    if value["model"] != gate["model"]:
        raise ValueError("synthetic calibration model drifted")
    bindings = {
        "cases_artifact_path": gate["cases_artifact_path"],
        "cases_artifact_sha256": gate["cases_artifact_sha256"],
        "cases_digest": gate["cases_digest"],
        "rubric_digest": gate["rubric_digest"],
        "runner_artifact_path": gate["runner_artifact_path"],
        "runner_artifact_sha256": gate["runner_artifact_sha256"],
        "response_schema_digest": gate["response_schema_digest"],
        "response_request_mode": gate["response_request_mode"],
        "response_validator_version": gate["response_validator_version"],
        "campaign_allocation_usd": gate["campaign_allocation_usd"],
        "prior_failed_requests": gate["prior_failed_requests"],
        "prior_accounted_reserve_usd": gate["prior_accounted_reserve_usd"],
        "preview_digest": gate["preview_digest"],
    }
    for field, expected in bindings.items():
        if value[field] != expected:
            raise ValueError(f"synthetic calibration {field} drifted")

    approval_digest = value["approval_digest"]
    if approval_digest is not None and not SHA256.fullmatch(str(approval_digest)):
        raise ValueError("synthetic calibration approval digest is invalid")
    if require_approval and approval_digest is None:
        raise ValueError("synthetic calibration approval is unavailable")

    results = value["results"]
    if not isinstance(results, list) or len(results) > len(rows):
        raise ValueError("synthetic calibration result rows are invalid")
    for index, result in enumerate(results):
        _validate_synthetic_result_row(result, frozen=rows[index], index=index)
    completed = len(results)
    complete = completed == len(rows) == 48
    if value["requested_cases"] != len(rows) or value["completed_cases"] != completed:
        raise ValueError("synthetic calibration result counts disagree")
    if value["status"] != ("completed" if complete else "incomplete"):
        raise ValueError("synthetic calibration result status disagrees")
    if require_complete and not complete:
        raise ValueError("synthetic calibration result is incomplete")
    if (
        value["budget_ceiling_usd"] != gate["maximum_cost_usd"]
        or value["observed_cost_usd"] is not None
    ):
        raise ValueError("synthetic calibration cost contract drifted")
    expected_accounted = round(
        completed * float(gate["maximum_cost_usd"]) / int(gate["examples"]),
        6,
    )
    if value["accounted_cost_usd"] != expected_accounted:
        raise ValueError("synthetic calibration accounted cost disagrees")
    expected_cumulative = round(
        float(gate["prior_accounted_reserve_usd"]) + expected_accounted,
        6,
    )
    if value[
        "cumulative_accounted_cost_usd"
    ] != expected_cumulative or expected_cumulative > float(
        gate["campaign_allocation_usd"]
    ):
        raise ValueError("synthetic calibration cumulative cost exceeds allocation")

    recomputed = recompute_synthetic_metrics(rows, results)
    if value["synthetic_metrics"] != recomputed:
        raise ValueError("synthetic calibration metrics disagree with frozen cases")
    if require_complete and recomputed["synthetic_thresholds_passed"] is not True:
        raise ValueError("synthetic calibration did not pass locked thresholds")
    if (
        value["human_review_status"] != "pending_human_review"
        or value["human_calibration_satisfied"] is not False
    ):
        raise ValueError("synthetic result cannot claim human calibration")
    if not str(value["note"]).strip():
        raise ValueError("synthetic calibration note is required")

    receipt = value["receipt"]
    if not isinstance(receipt, Mapping):
        raise ValueError("synthetic calibration receipt must be an object")
    _reject_unknown(receipt, SYNTHETIC_RECEIPT_FIELDS, "synthetic calibration receipt")
    receipt_unsigned = dict(receipt)
    receipt_digest = str(receipt_unsigned.pop("receipt_digest") or "")
    if not SHA256.fullmatch(receipt_digest) or receipt_digest != stable_digest(
        receipt_unsigned
    ):
        raise ValueError("synthetic calibration receipt digest disagrees")
    expected_receipt = {
        "schema_version": 1,
        "role": "advisory_synthetic_calibration",
        "model": gate["model"],
        **bindings,
        "request_count": completed,
        "maximum_requests": 48,
        "maximum_prompt_characters": 12_000,
        "maximum_output_tokens_per_request": 1_200,
        "automatic_retries": 0,
        "run_accounted_cost_usd": expected_accounted,
        "cumulative_accounted_cost_usd": expected_cumulative,
        "approval_digest": approval_digest,
        "serialized_input_fields": [
            "public_task",
            "response",
            "permitted_evidence",
            "rubric",
        ],
        "secret_value_scan": "passed",
    }
    if receipt_unsigned != expected_receipt:
        raise ValueError("synthetic calibration receipt contract disagrees")
    return {
        "status": value["status"],
        "completed_cases": completed,
        "synthetic_metrics": recomputed,
        "result_digest": supplied_digest,
        "approval_digest": approval_digest,
    }


def build_calibration_report(
    cases: Mapping[str, Any],
    rubric: Mapping[str, Any],
    *,
    cases_artifact_path: str,
    cases_artifact_sha256: str,
    runner_artifact_path: str,
    runner_artifact_sha256: str,
    response_schema_digest: str,
    response_request_mode: str,
    response_validator_version: int,
) -> dict[str, Any]:
    passed = bool(
        cases["completed"]
        and cases["true_positive_rate"] >= MIN_RATE
        and cases["true_negative_rate"] >= MIN_RATE
        and cases["calibration_true_positive_rate"] >= MIN_RATE
        and cases["calibration_true_negative_rate"] >= MIN_RATE
        and cases["holdout_true_positive_rate"] >= MIN_RATE
        and cases["holdout_true_negative_rate"] >= MIN_RATE
        and cases["critical_false_passes"] == 0
    )
    return {
        "schema_version": 1,
        "review_status": "adjudicated"
        if cases["completed"]
        else "pending_human_review",
        "reviewers_per_example": 2 if cases["completed"] else 0,
        "disagreements_adjudicated": bool(cases["completed"]),
        "judge_profile": "anthropic/claude-sonnet-5",
        "rubric_digest": rubric["contract_digest"],
        "cases_digest": cases["cases_digest"],
        "score_threshold": 0.75,
        "examples": cases["examples"],
        "calibration_examples": cases["calibration_examples"],
        "holdout_examples": cases["holdout_examples"],
        "true_positive_rate": round(cases["true_positive_rate"], 6),
        "true_negative_rate": round(cases["true_negative_rate"], 6),
        "calibration_true_positive_rate": round(
            cases["calibration_true_positive_rate"], 6
        ),
        "calibration_true_negative_rate": round(
            cases["calibration_true_negative_rate"], 6
        ),
        "holdout_true_positive_rate": round(cases["holdout_true_positive_rate"], 6),
        "holdout_true_negative_rate": round(cases["holdout_true_negative_rate"], 6),
        "critical_false_passes": cases["critical_false_passes"],
        "passed": passed,
        "distinct_reviewers": len(cases["reviewer_ids"]),
        "disagreements": cases["disagreements"],
        "execution_gate": synthetic_execution_gate(
            cases_artifact_path=cases_artifact_path,
            cases_artifact_sha256=cases_artifact_sha256,
            cases_digest=cases["cases_digest"],
            rubric_digest=rubric["contract_digest"],
            runner_artifact_path=runner_artifact_path,
            runner_artifact_sha256=runner_artifact_sha256,
            response_schema_digest=response_schema_digest,
            response_request_mode=response_request_mode,
            response_validator_version=response_validator_version,
        ),
        "note": (
            "Calibration passed the locked two-reviewer and held-out thresholds."
            if passed
            else "The blinded 48-case synthetic result gates advisory same-family "
            "judging before any Agent trial. It does not replace the two-reviewer "
            "calibration required for a judge-qualified outcome claim."
        ),
    }


def validate_calibration_report(
    value: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    _reject_unknown(value, CALIBRATION_REPORT_FIELDS, "calibration report")
    if dict(value) != dict(expected):
        raise ValueError("calibration report does not match the cases and rubric")


def validate_ledger(value: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    _reject_unknown(value, LEDGER_FIELDS, "campaign budget ledger")
    if value["schema_version"] != 1 or value["campaign_id"] != manifest["id"]:
        raise ValueError("campaign budget ledger identity does not match")
    if value["currency"] != "USD" or value["status"] != "unapproved":
        raise ValueError("campaign budget ledger must remain unapproved in source")
    allocations = value["allocations"]
    if not isinstance(allocations, list) or len(allocations) != 4:
        raise ValueError("campaign budget ledger must contain four allocations")
    expected_ids = [manifest["calibration"]["id"]] + [
        study["id"] for study in manifest["studies"]
    ]
    expected_caps = [manifest["calibration"]["budget_usd"]] + [
        study["budget_usd"] for study in manifest["studies"]
    ]
    for index, allocation in enumerate(allocations):
        if not isinstance(allocation, Mapping):
            raise ValueError("campaign budget allocation must be an object")
        _reject_unknown(allocation, ALLOCATION_FIELDS, f"budget allocation {index}")
        if allocation["stage_id"] != expected_ids[index]:
            raise ValueError("campaign budget allocation order or identity drifted")
        if allocation["cap_usd"] != expected_caps[index]:
            raise ValueError("campaign budget allocation cap drifted")
        if (
            allocation["approval_status"] != "not_approved"
            or allocation["approval_digest"] is not None
            or allocation["observed_usd"] != 0
        ):
            raise ValueError(
                "checked-in budget allocations cannot claim approval or spend"
            )
    total = sum(_money(item["cap_usd"], "allocation cap") for item in allocations)
    if not (
        total
        == float(value["total_allocated_usd"])
        == float(value["campaign_ceiling_usd"])
        == float(manifest["execution"]["cumulative_budget_usd"])
        == 110.0
    ):
        raise ValueError("campaign budget ledger totals do not reconcile")
    if value["total_approved_usd"] != 0 or value["total_observed_usd"] != 0:
        raise ValueError("checked-in budget ledger cannot claim approval or spend")


def validate_report_template(value: Mapping[str, Any]) -> None:
    _reject_unknown(value, REPORT_TEMPLATE_FIELDS, "scientific report template")
    if value["schema_version"] != 1 or value["status"] != "pending_execution":
        raise ValueError("scientific report template must remain pending execution")
    for field in (
        "study_id",
        "evidence_project",
        "task_validity",
        "behavioral_finding",
        "conclusion",
    ):
        if value[field] is not None:
            raise ValueError(f"scientific report template cannot pre-claim {field}")
    revisions = value["exact_revisions"]
    if revisions != {"baseline": None, "candidate": None}:
        raise ValueError("scientific report template cannot pre-claim revisions")
    efficiency = value["efficiency"]
    if efficiency != {"latency": None, "tokens": None, "cost": None}:
        raise ValueError("scientific report template cannot pre-claim efficiency")
    for field in (
        "deterministic_results",
        "judge_results",
        "skill_use_evidence",
        "evidence_links",
    ):
        if value[field] != []:
            raise ValueError(f"scientific report template {field} must start empty")
    if not isinstance(value["limitations"], list) or len(value["limitations"]) < 3:
        raise ValueError("scientific report template must predeclare limitations")


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_judge_input(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(
        json.dumps(
            {
                "id": row["id"],
                "repository_id": row["repository_id"],
                "public_task": row["public_task"],
                "response": row["response"],
                "permitted_evidence": row["permitted_evidence"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    )
    path.write_text(body, encoding="utf-8")


def validate_all(root: Path, *, require_specs: bool = False) -> dict[str, Any]:
    manifest = _load_json(root / "campaign-manifest.json", "campaign manifest")
    manifest_summary = validate_manifest(
        manifest, root=root, require_specs=require_specs
    )
    rubric_value = _load_json(root / manifest["calibration"]["rubric"], "judge rubric")
    rubric = validate_rubric(rubric_value)
    validate_study_specs(root=root, manifest=manifest, rubric=rubric_value)
    rows = read_cases(root / manifest["calibration"]["cases"])
    cases = validate_cases(rows)
    expected_report = build_calibration_report(
        cases,
        rubric,
        cases_artifact_path=manifest_summary["cases_artifact_path"],
        cases_artifact_sha256=manifest_summary["cases_artifact_sha256"],
        runner_artifact_path=manifest_summary["runner_artifact_path"],
        runner_artifact_sha256=manifest_summary["runner_artifact_sha256"],
        response_schema_digest=manifest_summary["response_schema_digest"],
        response_request_mode=manifest_summary["response_request_mode"],
        response_validator_version=manifest_summary["response_validator_version"],
    )
    report = _load_json(root / manifest["calibration"]["report"], "calibration report")
    validate_calibration_report(report, expected_report)
    ledger = _load_json(root / manifest["budget_ledger"], "campaign budget ledger")
    validate_ledger(ledger, manifest)
    report_template = _load_json(
        root / manifest["scientific_report_template"], "scientific report template"
    )
    validate_report_template(report_template)
    return {
        "schema_version": 1,
        "campaign_id": manifest["id"],
        "fixtures_valid": True,
        "execution_eligible": bool(report["passed"]),
        "calibration_status": report["review_status"],
        "calibration_cases": report["examples"],
        "calibration_digest": report["cases_digest"],
        "calibration_cases_artifact_path": manifest_summary["cases_artifact_path"],
        "calibration_cases_artifact_sha256": manifest_summary["cases_artifact_sha256"],
        "rubric_digest": report["rubric_digest"],
        "agent_cells": manifest_summary["total_cells"],
        "campaign_ceiling_usd": manifest_summary["campaign_ceiling_usd"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--require-specs", action="store_true")
    parser.add_argument("--require-reviewed-calibration", action="store_true")
    parser.add_argument("--emit-judge-input", type=Path)
    parser.add_argument("--write-report", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    manifest = _load_json(root / "campaign-manifest.json", "campaign manifest")
    manifest_summary = validate_manifest(
        manifest,
        root=root,
        require_specs=args.require_specs,
    )
    rubric_value = _load_json(root / manifest["calibration"]["rubric"], "judge rubric")
    rubric = validate_rubric(rubric_value)
    rows = read_cases(root / manifest["calibration"]["cases"])
    cases = validate_cases(rows)
    if args.emit_judge_input:
        _write_judge_input(args.emit_judge_input, rows)
    if args.write_report:
        _atomic_write_json(
            args.write_report,
            build_calibration_report(
                cases,
                rubric,
                cases_artifact_path=manifest_summary["cases_artifact_path"],
                cases_artifact_sha256=manifest_summary["cases_artifact_sha256"],
                runner_artifact_path=manifest_summary["runner_artifact_path"],
                runner_artifact_sha256=manifest_summary["runner_artifact_sha256"],
                response_schema_digest=manifest_summary["response_schema_digest"],
                response_request_mode=manifest_summary["response_request_mode"],
                response_validator_version=manifest_summary[
                    "response_validator_version"
                ],
            ),
        )
    summary = validate_all(root, require_specs=args.require_specs)
    print(json.dumps(summary, sort_keys=True))
    if args.require_reviewed_calibration and not summary["execution_eligible"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
