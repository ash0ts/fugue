#!/usr/bin/env python3
"""Recompute preregistered task-clustered Skill comparison statistics.

The canonical Fugue attempt rows are the only numerical input. Attempts are
replicates within a task; they are never treated as independent sampled tasks.
The script intentionally refuses incomplete, duplicated, weakly linked, or
cross-project rows before computing an effect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

TERMINAL_STATUSES = {"passed", "failed"}
OUTCOME_ROLES = {"outcome"}
SAFETY_ROLES = {"safety_gate"}
REQUIRED_LINK_FIELDS = (
    "weave_evaluation_root_call_id",
    "eval_predict_and_score_call_id",
    "weave_prediction_call_id",
    "weave_agent_root_call_id",
    "weave_dataset_id",
)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"attempt row {line_number} must be an object")
        rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires a nonempty sample")
    return sum(values) / len(values)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires a nonempty sample")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _cluster_interval(
    task_differences: Mapping[str, float], *, samples: int, seed: int
) -> tuple[float, float]:
    task_ids = sorted(task_differences)
    if not task_ids:
        raise ValueError("cluster interval requires nonzero tasks")
    generator = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        selected = [generator.choice(task_ids) for _ in task_ids]
        estimates.append(_mean([task_differences[item] for item in selected]))
    return _quantile(estimates, 0.025), _quantile(estimates, 0.975)


def _two_sided_sign_p(task_differences: Mapping[str, float]) -> float:
    positive = sum(value > 0 for value in task_differences.values())
    negative = sum(value < 0 for value in task_differences.values())
    total = positive + negative
    if total == 0:
        return 1.0
    extreme = min(positive, negative)
    one_tail = sum(math.comb(total, index) for index in range(extreme + 1)) / (2**total)
    return min(1.0, 2 * one_tail)


def _holm(values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (dimension, value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * value)
        running = max(running, candidate)
        adjusted[dimension] = running
    return adjusted


def _checked_text(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"attempt {row.get('attempt_id')} has no {field}")
    return value


def _validate_rows(  # noqa: C901 - one fail-closed cross-row integrity gate.
    *,
    rows: Sequence[Mapping[str, Any]],
    study: Mapping[str, Any],
    expected_attempts: int,
) -> tuple[list[str], dict[str, str]]:
    expected_project = str(study["evidence_project"])
    expected_cells = int(study["cells"])
    if len(rows) != expected_cells:
        raise ValueError(f"expected {expected_cells} attempt rows, found {len(rows)}")
    identities: set[str] = set()
    tasks: dict[tuple[str, str], set[int]] = defaultdict(set)
    dimension_roles: dict[str, str] = {}
    for row in rows:
        attempt_id = _checked_text(row, "attempt_id")
        if attempt_id in identities:
            raise ValueError(f"duplicate attempt identity: {attempt_id}")
        identities.add(attempt_id)
        task_id = _checked_text(row, "task_id")
        variant = _checked_text(row, "variant_id")
        if variant not in {"baseline", "candidate"}:
            raise ValueError(f"unexpected treatment arm: {variant}")
        trial = row.get("trial_index")
        if not isinstance(trial, int) or isinstance(trial, bool) or trial < 1:
            raise ValueError(f"attempt {attempt_id} has an invalid trial index")
        tasks[(task_id, variant)].add(trial)
        if row.get("status") not in TERMINAL_STATUSES:
            raise ValueError(f"attempt {attempt_id} is not terminal")
        if row.get("result_evidence_project") != expected_project:
            raise ValueError(f"attempt {attempt_id} is in the wrong evidence project")
        if row.get("trace_link_status") != "linked":
            raise ValueError(f"attempt {attempt_id} has unresolved evidence links")
        if row.get("sandbox_cleanup_verified") is not True:
            raise ValueError(f"attempt {attempt_id} has no verified Harbor cleanup")
        if row.get("private_label_boundary_verified") is not True:
            raise ValueError(
                f"attempt {attempt_id} has no private-label boundary receipt"
            )
        for field in REQUIRED_LINK_FIELDS:
            _checked_text(row, field)
        raw_scores = row.get("comparison_deterministic_scores")
        raw_roles = row.get("comparison_dimension_roles")
        if not isinstance(raw_scores, Mapping) or not raw_scores:
            raise ValueError(f"attempt {attempt_id} has no deterministic scores")
        if not isinstance(raw_roles, Mapping) or set(raw_scores) != set(raw_roles):
            raise ValueError(f"attempt {attempt_id} score roles are incomplete")
        for dimension, role in raw_roles.items():
            if role not in {
                "outcome",
                "mechanism",
                "safety_gate",
                "infrastructure",
                "efficiency",
            }:
                raise ValueError(f"attempt {attempt_id} has an unknown dimension role")
            prior = dimension_roles.setdefault(str(dimension), str(role))
            if prior != role:
                raise ValueError(f"dimension role drift: {dimension}")
    task_ids = sorted({task_id for task_id, _ in tasks})
    if len(task_ids) != int(study["tasks"]):
        raise ValueError("unique task count disagrees with the preregistration")
    expected_trials = set(range(1, expected_attempts + 1))
    for task_id in task_ids:
        for variant in ("baseline", "candidate"):
            if tasks[(task_id, variant)] != expected_trials:
                raise ValueError(f"task {task_id}/{variant} has incomplete attempts")
    return task_ids, dimension_roles


def _dimension_analysis(
    *,
    rows: Sequence[Mapping[str, Any]],
    task_ids: Sequence[str],
    dimension_roles: Mapping[str, str],
    bootstrap_samples: int,
    seed_material: str,
) -> list[dict[str, Any]]:
    task_arm: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        scores = row["comparison_deterministic_scores"]
        for dimension, score in scores.items():
            if not isinstance(score, bool):
                raise ValueError(f"deterministic score {dimension} must be boolean")
            task_arm[
                (str(row["task_id"]), str(row["variant_id"]), str(dimension))
            ].append(float(score))
    values: list[dict[str, Any]] = []
    raw_p: dict[str, float] = {}
    task_differences_by_dimension: dict[str, dict[str, float]] = {}
    for dimension in sorted(dimension_roles):
        task_differences = {
            task_id: _mean(task_arm[(task_id, "candidate", dimension)])
            - _mean(task_arm[(task_id, "baseline", dimension)])
            for task_id in task_ids
        }
        task_differences_by_dimension[dimension] = task_differences
        raw_p[dimension] = _two_sided_sign_p(task_differences)
    confirmatory_p = {
        dimension: value
        for dimension, value in raw_p.items()
        if dimension_roles[dimension] in OUTCOME_ROLES | SAFETY_ROLES
    }
    adjusted = _holm(confirmatory_p)
    for dimension in sorted(dimension_roles):
        task_differences = task_differences_by_dimension[dimension]
        seed = int(
            hashlib.sha256(f"{seed_material}:{dimension}".encode()).hexdigest()[:16],
            16,
        )
        low, high = _cluster_interval(
            task_differences, samples=bootstrap_samples, seed=seed
        )
        values.append(
            {
                "dimension": dimension,
                "role": dimension_roles[dimension],
                "baseline_attempt_pass_rate": _mean(
                    [
                        float(score)
                        for row in rows
                        if row["variant_id"] == "baseline"
                        for name, score in row[
                            "comparison_deterministic_scores"
                        ].items()
                        if name == dimension
                    ]
                ),
                "candidate_attempt_pass_rate": _mean(
                    [
                        float(score)
                        for row in rows
                        if row["variant_id"] == "candidate"
                        for name, score in row[
                            "comparison_deterministic_scores"
                        ].items()
                        if name == dimension
                    ]
                ),
                "task_level_mean_difference": _mean(list(task_differences.values())),
                "task_cluster_bootstrap_95_ci": [low, high],
                "positive_tasks": sum(value > 0 for value in task_differences.values()),
                "negative_tasks": sum(value < 0 for value in task_differences.values()),
                "zero_tasks": sum(value == 0 for value in task_differences.values()),
                "exact_two_sided_sign_p": raw_p[dimension],
                "holm_adjusted_p": adjusted.get(dimension),
            }
        )
    return values


def _classify(
    dimensions: Sequence[Mapping[str, Any]], *, meaningful: float, equivalence: float
) -> dict[str, Any]:
    safety_regressions = [
        str(item["dimension"])
        for item in dimensions
        if item["role"] == "safety_gate"
        and float(item["task_level_mean_difference"]) < 0
    ]
    outcomes = [item for item in dimensions if item["role"] == "outcome"]
    improved = [
        str(item["dimension"])
        for item in outcomes
        if float(item["task_level_mean_difference"]) >= meaningful
        and float(item["task_cluster_bootstrap_95_ci"][0]) > 0
        and float(item["holm_adjusted_p"] or 1) <= 0.05
    ]
    regressed = [
        str(item["dimension"])
        for item in outcomes
        if float(item["task_cluster_bootstrap_95_ci"][1]) < -equivalence
    ]
    equivalent = bool(outcomes) and all(
        float(item["task_cluster_bootstrap_95_ci"][0]) >= -equivalence
        and float(item["task_cluster_bootstrap_95_ci"][1]) <= equivalence
        for item in outcomes
    )
    if safety_regressions or regressed:
        status = "regressed"
    elif improved:
        status = "improved"
    elif equivalent:
        status = "unchanged"
    else:
        status = "inconclusive"
    return {
        "status": status,
        "improved_dimensions": improved,
        "regressed_dimensions": regressed,
        "safety_regressions": safety_regressions,
        "equivalence_established": equivalent,
    }


def analyze(
    *,
    attempts_path: Path,
    result_path: Path,
    campaign_manifest_path: Path,
    preregistration_path: Path,
    study_id: str,
) -> dict[str, Any]:
    result = _load_json(result_path, "comparison result")
    manifest = _load_json(campaign_manifest_path, "conference manifest")
    preregistration = _load_json(preregistration_path, "preregistration")
    studies = [
        item
        for item in manifest["final_four"]
        if isinstance(item, Mapping) and item.get("id") == study_id
    ]
    if len(studies) != 1 or studies[0].get("kind") != "governed_comparison":
        raise ValueError(
            "study id does not identify one governed final-four comparison"
        )
    study = studies[0]
    if result.get("comparison_id") != study_id:
        raise ValueError("result comparison identity drifted")
    if result.get("evidence_project") != study["evidence_project"]:
        raise ValueError("result destination drifted")
    rows = _read_rows(attempts_path)
    task_ids, roles = _validate_rows(
        rows=rows,
        study=study,
        expected_attempts=int(study["attempts"]),
    )
    seed_material = hashlib.sha256(
        (
            _sha256(attempts_path)
            + _sha256(result_path)
            + _sha256(preregistration_path)
        ).encode()
    ).hexdigest()
    dimensions = _dimension_analysis(
        rows=rows,
        task_ids=task_ids,
        dimension_roles=roles,
        bootstrap_samples=10_000,
        seed_material=seed_material,
    )
    finding = _classify(
        dimensions,
        meaningful=float(
            preregistration["endpoints"]["minimum_meaningful_improvement"]
        ),
        equivalence=float(preregistration["endpoints"]["equivalence_margin"]),
    )
    observed_costs = [
        float(row["cost_usd"])
        for row in rows
        if isinstance(row.get("cost_usd"), int | float)
        and not isinstance(row.get("cost_usd"), bool)
    ]
    wall_times = [float(row.get("wall_time_sec") or 0) for row in rows]
    report = {
        "schema_version": 1,
        "kind": "community_skill_confirmatory_analysis",
        "campaign_id": manifest["id"],
        "study_id": study_id,
        "status": "complete",
        "finding": finding,
        "scope": preregistration["claims_scope"],
        "exact_revisions": {
            "baseline": study["baseline_commit"],
            "candidate": study["candidate_commit"],
        },
        "evidence_project": study["evidence_project"],
        "counts": {
            "tasks": len(task_ids),
            "attempts_per_task_per_arm": study["attempts"],
            "arms": 2,
            "rows": len(rows),
        },
        "dimensions": dimensions,
        "efficiency": {
            "observed_cost_usd": sum(observed_costs),
            "cost_rows": len(observed_costs),
            "wall_time_seconds": sum(wall_times),
        },
        "integrity": {
            "attempts_sha256": _sha256(attempts_path),
            "result_sha256": _sha256(result_path),
            "preregistration_sha256": _sha256(preregistration_path),
            "all_rows_terminal_unique_linked_private_and_clean": True,
        },
        "limitations": [
            preregistration["diagnostic_no_skill_contrast"]["claim_limitation"],
            "The Agent and advisory judge share a model family.",
            "Human-review qualification and the frozen trace audit are separate gates.",
            "Repositories are analyzed separately and are not a universal Skill ranking.",
        ],
    }
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["analysis_digest"] = hashlib.sha256(encoded).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("conference-campaign-manifest.json"),
    )
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path(__file__).with_name("conference-preregistration.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        attempts_path=args.attempts.resolve(),
        result_path=args.result.resolve(),
        campaign_manifest_path=args.manifest.resolve(),
        preregistration_path=args.preregistration.resolve(),
        study_id=args.study_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({"status": "complete", "analysis_digest": report["analysis_digest"]})
    )


if __name__ == "__main__":
    main()
