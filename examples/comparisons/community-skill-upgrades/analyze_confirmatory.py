#!/usr/bin/env python3
"""Recompute preregistered, task-clustered Skill comparison statistics.

This is deliberately stricter than the generic per-attempt behavioral summary.
It accepts only a canonical ComparisonResultV3 which recomputes exactly from the
exported attempt rows, then applies the frozen repository-specific holdout
analysis. Development tasks are descriptive and never enter primary inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fugue.bench.candidates import stable_digest
from fugue.bench.comparison import (
    ComparisonResultV3,
    _comparison_cohort_lineage,
    analyze_comparison_rows,
    load_comparison,
    read_comparison_result,
)

TERMINAL_STATUSES = {
    "agent_completed",
    "completed",
    "failed",
    "passed",
}
REQUIRED_LINK_KINDS = {
    "evaluation_root",
    "prediction_and_score",
    "prediction",
    "agent_root",
    "dataset",
}
REQUIRED_AUDIT_CHECKS = {
    "attempt_identity",
    "result_project",
    "candidate_and_runtime_locks",
    "skill_registered_opened_and_invoked",
    "host_verifier_receipt",
    "privacy",
    "cleanup",
}
PROFILE_PATH = Path(__file__).with_name("confirmatory-analysis-profiles.json")


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
    if not rows:
        raise ValueError("confirmatory analysis requires nonzero attempt rows")
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_digest(value: Any) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", str(value or "")))


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
    if samples < 1 or seed < 0:
        raise ValueError("bootstrap samples and seed must be frozen positive values")
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
    one_tail = sum(math.comb(total, index) for index in range(extreme + 1)) / (
        2**total
    )
    return min(1.0, 2 * one_tail)


def _holm(values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (family, value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * value)
        running = max(running, candidate)
        adjusted[family] = running
    return adjusted


def _profile_for_study(
    path: Path,
    study_id: str,
    *,
    campaign_id: str | None = None,
) -> tuple[dict[str, Any], str]:
    document = _load_json(path, "confirmatory analysis profiles")
    if (
        document.get("schema_version") != 1
        or document.get("status") != "frozen_before_execution"
        or not str(document.get("campaign_id") or "")
    ):
        raise ValueError("unsupported confirmatory analysis profile schema")
    if campaign_id is not None and document.get("campaign_id") != campaign_id:
        raise ValueError("analysis profile belongs to another campaign")
    profiles = document.get("profiles")
    if not isinstance(profiles, list):
        raise ValueError("confirmatory analysis profiles must be a list")
    matches = [
        item
        for item in profiles
        if isinstance(item, dict) and study_id in item.get("study_ids", [])
    ]
    pending = [
        item
        for item in profiles
        if isinstance(item, dict) and item.get("pending_study_id") == study_id
    ]
    if pending:
        raise ValueError(
            "confirmatory analysis profile is pending an exact amendment, "
            "manifest, and spec binding"
        )
    if len(matches) != 1:
        raise ValueError("study must identify exactly one confirmatory analysis profile")
    profile = dict(matches[0])
    if profile.get("sensitivity_analysis", {}).get("status") != "not_implemented":
        raise ValueError("unimplemented sensitivity analysis must be explicit")
    return profile, _sha256(path)


def _nested(value: Mapping[str, Any], path: Sequence[str], label: str) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise ValueError(f"{label} path is unavailable: {'.'.join(path)}")
        current = current[key]
    return current


def _load_profile_preregistration(
    profile: Mapping[str, Any], *, profile_path: Path, study_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_path = str(profile.get("preregistration") or "")
    path = (profile_path.parent / raw_path).resolve()
    expected_sha256 = str(profile.get("preregistration_sha256") or "")
    if (
        not _is_digest(expected_sha256)
        or not path.is_file()
        or _sha256(path) != expected_sha256
    ):
        raise ValueError("repository preregistration is unavailable or drifted")
    preregistration = _load_json(path, "repository preregistration")
    treatments = preregistration.get("treatments")
    arms = preregistration.get("arms")
    for arm in ("baseline", "candidate"):
        expected_revision = str(profile.get(f"{arm}_commit") or "")
        observed: Any = preregistration.get(f"{arm}_commit")
        if isinstance(treatments, Mapping) and isinstance(
            treatments.get(arm), Mapping
        ):
            observed = treatments[arm].get("commit")
        elif isinstance(arms, Mapping):
            observed = arms.get(arm)
        if not expected_revision or observed != expected_revision:
            raise ValueError(
                f"analysis profile {arm} revision disagrees with preregistration"
            )
    binding: dict[str, Any] = {
        "path": path.as_posix(),
        "sha256": expected_sha256,
    }
    amendments = profile.get("amendments") or {}
    if not isinstance(amendments, Mapping):
        raise ValueError("analysis-profile amendments must be a mapping")
    if study_id in amendments:
        amendment = amendments[study_id]
        if not isinstance(amendment, Mapping):
            raise ValueError("analysis-profile amendment is malformed")
        amendment_path = (
            profile_path.parent / str(amendment.get("path") or "")
        ).resolve()
        amendment_sha = str(amendment.get("sha256") or "")
        if (
            not _is_digest(amendment_sha)
            or not amendment_path.is_file()
            or _sha256(amendment_path) != amendment_sha
        ):
            raise ValueError("repository preregistration amendment drifted")
        amendment_value = _load_json(amendment_path, "preregistration amendment")
        replacement = amendment_value.get("replacement_execution")
        if (
            not isinstance(replacement, Mapping)
            or replacement.get("comparison_id") != study_id
            or amendment_value.get("changes", {}).get("hypotheses") is not False
            or amendment_value.get("changes", {}).get("taskset") is not False
            or amendment_value.get("changes", {}).get("treatments") is not False
            or amendment_value.get("changes", {}).get("evaluators") is not False
        ):
            raise ValueError("preregistration amendment changes behavioral inputs")
        binding["amendment"] = {
            "path": amendment_path.as_posix(),
            "sha256": amendment_sha,
            "amendment_digest": amendment_value.get("amendment_digest"),
        }
    return preregistration, binding


def _campaign_study(
    manifest: Mapping[str, Any], study_id: str
) -> Mapping[str, Any]:
    final_four = manifest.get("final_four")
    if not isinstance(final_four, list):
        raise ValueError("conference manifest final_four must be a list")
    matches = [
        item
        for item in final_four
        if isinstance(item, Mapping) and item.get("id") == study_id
    ]
    if len(matches) != 1 or matches[0].get("kind") != "governed_comparison":
        raise ValueError(
            "study id does not identify one governed final-four comparison"
        )
    return matches[0]


def _preregistered_input_bindings(
    *,
    profile: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    profile_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    locked = preregistration.get("locked_inputs")
    configured = profile.get("locked_input_paths")
    if locked is None and configured is None:
        return {"status": "not_declared", "inputs": []}
    if not isinstance(locked, Mapping) or not isinstance(configured, Mapping):
        raise ValueError("preregistered locked-input path bindings are incomplete")
    digest_fields = {
        str(key) for key in locked if str(key).endswith("_sha256")
    }
    if set(configured) != digest_fields:
        raise ValueError("analysis profile does not bind every locked input digest")
    root = repo_root.resolve()
    bindings: list[dict[str, Any]] = []
    for field in sorted(digest_fields):
        expected = str(locked.get(field) or "")
        path = (profile_path.parent / str(configured[field])).resolve()
        if (
            not _is_digest(expected)
            or not path.is_relative_to(root)
            or not path.is_file()
            or _sha256(path) != expected
        ):
            raise ValueError(f"preregistered locked input drifted: {field}")
        bindings.append(
            {
                "field": field,
                "path": path.relative_to(root).as_posix(),
                "sha256": expected,
            }
        )
    return {
        "status": "matched",
        "inputs": bindings,
        "bindings_digest": stable_digest(bindings),
    }


def _public_tasks(path: Path) -> list[dict[str, Any]]:
    rows = _read_rows(path)
    ids = [str(row.get("id") or "") for row in rows]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("public taskset contains missing or duplicate task ids")
    return rows


def _partition_tasks(
    profile: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, tuple[str, ...]]]:
    partition = profile.get("partition")
    if not isinstance(partition, Mapping):
        raise ValueError("analysis profile has no partition contract")
    kind = partition.get("kind")
    task_by_id = {str(item["id"]): item for item in tasks}
    if kind == "preregistration_lists":
        development = tuple(
            str(item)
            for item in _nested(
                preregistration,
                partition.get("development_path") or [],
                "development task",
            )
        )
        holdout = tuple(
            str(item)
            for item in _nested(
                preregistration,
                partition.get("holdout_path") or [],
                "holdout task",
            )
        )
    elif kind == "task_field":
        field = str(partition.get("field") or "")
        development_values = set(partition.get("development_values") or [])
        holdout_values = set(partition.get("holdout_values") or [])
        development = tuple(
            str(item["id"])
            for item in tasks
            if item.get(field) in development_values
        )
        holdout = tuple(
            str(item["id"])
            for item in tasks
            if item.get(field) in holdout_values
        )
    else:
        raise ValueError("unsupported task partition contract")
    if (
        len(development) != 8
        or len(holdout) != 16
        or len(set(development)) != 8
        or len(set(holdout)) != 16
        or set(development) & set(holdout)
        or set(development) | set(holdout) != set(task_by_id)
    ):
        raise ValueError("confirmatory task partition must be exact 8/16 disjoint")
    tags = {
        task_id: tuple(str(tag) for tag in task_by_id[task_id].get("tags") or [])
        for task_id in task_by_id
    }
    return tuple(development), tuple(holdout), tags


def _approved_execution_lock(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    locks = [row.get("approved_comparison") for row in rows]
    if any(not isinstance(item, Mapping) for item in locks):
        raise ValueError("every attempt must carry its approved execution lock")
    assert all(isinstance(item, Mapping) for item in locks)
    digests = {stable_digest(item) for item in locks}
    if len(digests) != 1:
        raise ValueError("attempt rows disagree on the approved execution lock")
    approved = dict(locks[0])
    supplied = str(approved.get("lock_digest") or "")
    unsigned = {key: value for key, value in approved.items() if key != "lock_digest"}
    if not _is_digest(supplied) or supplied != stable_digest(unsigned):
        raise ValueError("approved execution lock digest does not match")
    if not _is_digest(approved.get("approval_digest")):
        raise ValueError("approved execution lock has no exact approval digest")
    return approved


def _recompute_canonical_result(
    result: ComparisonResultV3,
    rows: Sequence[Mapping[str, Any]],
    approved: Mapping[str, Any],
) -> None:
    recomputed = analyze_comparison_rows(
        comparison_id=result.comparison_id,
        preview_digest=result.preview_digest,
        rows=rows,
        source=result.source,
        expected_evidence_project=result.evidence_project,
        expected_source_evidence_project=(
            result.evidence_topology.source_destination.project_slug
        ),
        approved_comparison=approved,
        decision_policy=result.decision_policy,
        attestation=result.decision.attestation,
        result_schema_version=3,
        study_intent=result.aligned_analysis.study_intent,
        evidence_topology=result.evidence_topology,
        release_note_coverage=result.release_note_coverage,
        supersedes=result.supersedes,
    )
    if recomputed.to_dict() != result.to_dict():
        raise ValueError(
            "ComparisonResultV3 disagrees with the exact exported attempt rows"
        )


def _source_revisions(lineage: Mapping[str, Any], arm: str) -> tuple[str, ...]:
    arms = lineage.get("arms")
    if not isinstance(arms, Mapping) or not isinstance(arms.get(arm), Mapping):
        raise ValueError(f"cohort lineage has no {arm} arm")
    revisions = arms[arm].get("source_revisions")
    if not isinstance(revisions, list):
        raise ValueError(f"cohort lineage {arm} revisions are unavailable")
    return tuple(
        str(item.get("version_identity") or "")
        for item in revisions
        if isinstance(item, Mapping) and item.get("kind") == "skill"
    )


def _canonical_attempt_ids(result: ComparisonResultV3) -> set[str]:
    return {
        attempt.attempt_id
        for pair in result.paired_cases
        for attempt in (pair.baseline, pair.candidate)
        if attempt is not None
    }


def _validate_bindings(  # noqa: C901 - one bounded cross-artifact gate.
    *,
    result: ComparisonResultV3,
    rows: Sequence[Mapping[str, Any]],
    approved: Mapping[str, Any],
    spec: Any,
    study: Mapping[str, Any],
    profile: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    if result.comparison_id != study.get("id") or spec.id != result.comparison_id:
        raise ValueError("comparison, Study, and spec identities disagree")
    if result.preview_digest != approved.get("preview_digest"):
        raise ValueError("result preview digest disagrees with approval")
    if approved.get("comparison_id") != result.comparison_id:
        raise ValueError("approved comparison identity drifted")
    if approved.get("spec_digest") != spec.spec_digest:
        raise ValueError("approved spec digest disagrees with the checked-in spec")
    if result.evidence_project != study.get("evidence_project"):
        raise ValueError("result evidence project disagrees with the campaign")
    if result.evidence_project != approved.get("evidence_project"):
        raise ValueError("result evidence project disagrees with the approval")
    if approved.get("expected_cell_count") != len(rows) or result.rows != len(rows):
        raise ValueError("canonical result, approval, and attempt row counts disagree")
    raw_attempt_ids = [str(row.get("attempt_id") or "") for row in rows]
    if (
        any(not _is_digest(item) for item in raw_attempt_ids)
        or len(raw_attempt_ids) != len(set(raw_attempt_ids))
        or set(raw_attempt_ids) != _canonical_attempt_ids(result)
    ):
        raise ValueError("exported and canonical attempt identities disagree")
    if result.integrity.get("approved_manifest_digest") != approved.get(
        "lock_digest"
    ):
        raise ValueError("result does not bind the exact approved execution lock")
    if result.integrity.get("expected_cell_count") != len(rows):
        raise ValueError("result integrity has the wrong approved cell count")
    integrity_failures = {
        "status": result.integrity.get("status") != "reconciled",
        "unresolved_evidence": bool(
            result.integrity.get("unresolved_evidence_attempts")
        ),
        "duplicates": bool(result.integrity.get("duplicate_attempt_ids")),
        "cross_project": bool(result.integrity.get("cross_project_attempts")),
        "harbor_failed": bool(
            result.integrity.get("harbor_conformance_failed_attempts")
        ),
        "harbor_unavailable": bool(
            result.integrity.get("harbor_conformance_unavailable_attempts")
        ),
        "privacy_failed": bool(
            result.integrity.get("local_artifact_privacy_failed_attempts")
            or result.integrity.get("hosted_evidence_privacy_failed_attempts")
        ),
        "privacy_unavailable": bool(
            result.integrity.get("local_artifact_privacy_unavailable_attempts")
            or result.integrity.get("hosted_evidence_privacy_unavailable_attempts")
        ),
    }
    failed = sorted(key for key, value in integrity_failures.items() if value)
    if failed:
        raise ValueError("canonical result integrity is incomplete: " + ", ".join(failed))
    topology = result.evidence_topology
    if (
        topology.pre_run_drift.status != "matched"
        or topology.post_run_drift.status != "matched"
        or topology.execution_identity != approved.get("evidence_topology_identity")
        or topology.source_lock_digest != approved.get("source_lock_digest")
        or topology.result_destination.project_slug != result.evidence_project
        or topology.source_destination.project_slug
        != approved.get("source_evidence_project")
    ):
        raise ValueError("source/result topology or drift evidence is not exact")
    expected_lineage = _comparison_cohort_lineage(
        spec,
        repo_root=repo_root,
        source_lock_digest=topology.source_lock_digest,
    )
    if result.cohort_lineage != expected_lineage:
        raise ValueError("canonical cohort lineage disagrees with checked-in inputs")
    expected_revisions = {
        "baseline": f"git:{profile['baseline_commit']}",
        "candidate": f"git:{profile['candidate_commit']}",
    }
    for arm, revision in expected_revisions.items():
        if _source_revisions(result.cohort_lineage, arm) != (revision,):
            raise ValueError(f"{arm} source revision disagrees with preregistration")
    if not result.scorer_revisions or not result.runtime_locks:
        raise ValueError("canonical result is missing scorer or runtime locks")
    scorer_digests = result.cohort_lineage.get("scorer_digests")
    if not isinstance(scorer_digests, Mapping) or scorer_digests != {
        item.id: item.digest for item in result.scorer_revisions
    }:
        raise ValueError("canonical scorer revision bindings disagree")
    validity = {item.task_id: item for item in result.task_validity}
    if len(validity) != len(result.task_validity):
        raise ValueError("canonical task validity contains duplicate task ids")
    blocked_validity = {
        task_id: item.status
        for task_id, item in validity.items()
        if item.status in {"invalid", "drifted", "inconclusive"}
    }
    if blocked_validity:
        raise ValueError(
            "task validity blocks confirmatory inference: "
            + ", ".join(f"{key}={value}" for key, value in sorted(blocked_validity.items()))
        )
    return {
        "preview_digest": result.preview_digest,
        "approval_digest": approved["approval_digest"],
        "approved_execution_lock_digest": approved["lock_digest"],
        "spec_digest": spec.spec_digest,
        "result_digest": result.result_digest,
        "qualification_digest": result.qualification_digest,
        "topology_digest": topology.topology_digest,
        "source_lock_digest": topology.source_lock_digest,
        "source_destination": topology.source_destination.to_dict(),
        "result_destination": topology.result_destination.to_dict(),
        "source_pre_run_drift": topology.pre_run_drift.to_dict(),
        "source_post_run_drift": topology.post_run_drift.to_dict(),
        "candidate_source_revisions": [
            item.to_dict() for item in result.candidate_source_revisions
        ],
        "scorer_revisions": [item.to_dict() for item in result.scorer_revisions],
        "runtime_locks": [item.to_dict() for item in result.runtime_locks],
        "task_validity_digest": stable_digest(
            [item.to_dict() for item in result.task_validity]
        ),
    }


def _validate_matrix(
    *,
    rows: Sequence[Mapping[str, Any]],
    tasks: Sequence[str],
    attempts: int,
    harnesses: Sequence[str],
) -> None:
    if attempts != 4 or tuple(harnesses) != ("claude-code",):
        raise ValueError("confirmatory design requires four Claude Code attempts")
    expected = {
        (task_id, arm, "claude-code", attempt)
        for task_id in tasks
        for arm in ("baseline", "candidate")
        for attempt in range(1, attempts + 1)
    }
    observed: set[tuple[str, str, str, int]] = set()
    for row in rows:
        status = str(row.get("status") or row.get("execution_status") or "")
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"attempt {row.get('attempt_id')} is not terminal")
        coordinate = (
            str(row.get("task_id") or ""),
            str(row.get("variant_id") or ""),
            str(row.get("harness") or ""),
            int(row.get("trial_index") or 0),
        )
        if coordinate in observed:
            raise ValueError(f"duplicate task/arm/attempt coordinate: {coordinate}")
        observed.add(coordinate)
    if observed != expected:
        raise ValueError("attempt rows do not match the exact preregistered matrix")


def _score(row: Mapping[str, Any], dimension: str) -> bool:
    scores = row.get("comparison_deterministic_scores")
    if not isinstance(scores, Mapping) or type(scores.get(dimension)) is not bool:
        raise ValueError(
            f"attempt {row.get('attempt_id')} lacks Boolean score {dimension}"
        )
    return bool(scores[dimension])


def _metric_attempt_value(
    row: Mapping[str, Any], contract: Mapping[str, Any]
) -> float:
    aggregation = contract.get("aggregation")
    if aggregation == "dimension":
        return float(_score(row, str(contract.get("dimension") or "")))
    dimensions = tuple(str(item) for item in contract.get("dimensions") or [])
    if not dimensions:
        raise ValueError("multi-dimension metric requires dimensions")
    values = [float(_score(row, dimension)) for dimension in dimensions]
    if aggregation == "all_dimensions":
        return float(all(value == 1 for value in values))
    if aggregation == "mean_dimensions":
        return _mean(values)
    raise ValueError(f"unsupported metric aggregation: {aggregation}")


def _task_ids_for_family(
    family: Mapping[str, Any],
    holdout: Sequence[str],
    tags: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    tag = family.get("task_tag")
    selected = tuple(
        task_id
        for task_id in holdout
        if tag is None or str(tag) in tags.get(task_id, ())
    )
    if not selected:
        raise ValueError(f"primary family {family.get('id')} has no holdout tasks")
    return selected


def _metric_analysis(
    *,
    rows: Sequence[Mapping[str, Any]],
    task_ids: Sequence[str],
    contract: Mapping[str, Any],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    task_set = set(task_ids)
    for row in rows:
        if str(row.get("task_id")) not in task_set:
            continue
        grouped[(str(row["task_id"]), str(row["variant_id"]))].append(
            _metric_attempt_value(row, contract)
        )
    differences: dict[str, float] = {}
    for task_id in task_ids:
        baseline = grouped[(task_id, "baseline")]
        candidate = grouped[(task_id, "candidate")]
        if len(baseline) != 4 or len(candidate) != 4:
            raise ValueError(f"task {task_id} does not have four attempts per arm")
        differences[task_id] = _mean(candidate) - _mean(baseline)
    low, high = _cluster_interval(differences, samples=samples, seed=seed)
    baseline_values = [
        value
        for task_id in task_ids
        for value in grouped[(task_id, "baseline")]
    ]
    candidate_values = [
        value
        for task_id in task_ids
        for value in grouped[(task_id, "candidate")]
    ]
    return {
        "task_count": len(task_ids),
        "baseline_attempt_completion": _mean(baseline_values),
        "candidate_attempt_completion": _mean(candidate_values),
        "task_level_mean_difference": _mean(list(differences.values())),
        "task_cluster_bootstrap_95_ci": [low, high],
        "positive_tasks": sum(value > 0 for value in differences.values()),
        "negative_tasks": sum(value < 0 for value in differences.values()),
        "zero_tasks": sum(value == 0 for value in differences.values()),
        "exact_two_sided_sign_p": _two_sided_sign_p(differences),
        "task_differences": differences,
        "bootstrap": {"samples": samples, "seed": seed},
    }


def _descriptive_dimensions(
    rows: Sequence[Mapping[str, Any]], task_ids: Sequence[str]
) -> list[dict[str, Any]]:
    task_set = set(task_ids)
    dimensions = sorted(
        {
            str(dimension)
            for row in rows
            if str(row.get("task_id")) in task_set
            for dimension, role in (row.get("comparison_dimension_roles") or {}).items()
            if role in {"outcome", "safety_gate"}
        }
    )
    result: list[dict[str, Any]] = []
    for dimension in dimensions:
        values: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            if str(row.get("task_id")) in task_set:
                values[str(row.get("variant_id"))].append(
                    float(_score(row, dimension))
                )
        result.append(
            {
                "dimension": dimension,
                "baseline_attempt_completion": _mean(values["baseline"]),
                "candidate_attempt_completion": _mean(values["candidate"]),
                "descriptive_difference": _mean(values["candidate"])
                - _mean(values["baseline"]),
            }
        )
    return result


def _safety_analysis(
    *,
    rows: Sequence[Mapping[str, Any]],
    holdout: Sequence[str],
    dimensions: Sequence[str],
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    return [
        {
            "dimension": dimension,
            **{
                key: value
                for key, value in _metric_analysis(
                    rows=rows,
                    task_ids=holdout,
                    contract={"aggregation": "dimension", "dimension": dimension},
                    samples=samples,
                    seed=seed,
                ).items()
                if key
                in {
                    "baseline_attempt_completion",
                    "candidate_attempt_completion",
                    "task_level_mean_difference",
                    "positive_tasks",
                    "negative_tasks",
                    "zero_tasks",
                    "task_differences",
                }
            },
        }
        for dimension in dimensions
    ]


def _candidate_gate_failures(
    *,
    rows: Sequence[Mapping[str, Any]],
    holdout: Sequence[str],
    dimensions: Sequence[str],
) -> list[dict[str, Any]]:
    task_set = set(holdout)
    failures: list[dict[str, Any]] = []
    for row in rows:
        if row.get("variant_id") != "candidate" or row.get("task_id") not in task_set:
            continue
        criticality = row.get("comparison_deterministic_criticality")
        if not isinstance(criticality, Mapping):
            raise ValueError("candidate attempt has no deterministic criticality map")
        for dimension in dimensions:
            if criticality.get(dimension) is not True:
                raise ValueError(f"preregistered critical dimension is not critical: {dimension}")
            if not _score(row, dimension):
                failures.append(
                    {
                        "task_id": row["task_id"],
                        "attempt": row["trial_index"],
                        "dimension": dimension,
                    }
                )
    return failures


def _decision(
    *,
    profile: Mapping[str, Any],
    families: Sequence[Mapping[str, Any]],
    composite: Mapping[str, Any],
    safety: Sequence[Mapping[str, Any]],
    candidate_gate_failures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    policy = profile["decision"]
    meaningful = float(policy["minimum_meaningful_improvement"])
    equivalence = float(policy["equivalence_margin"])
    alpha = float(policy["alpha"])
    significant = [
        str(item["id"])
        for item in families
        if float(item["task_level_mean_difference"]) >= meaningful
        and float(item["task_cluster_bootstrap_95_ci"][0]) > 0
        and float(item["holm_adjusted_p"]) <= alpha
    ]
    primary_regressions = [
        {
            "family": str(item["id"]),
            "task_ids": sorted(
                task_id
                for task_id, difference in item["task_differences"].items()
                if float(difference) < 0
            ),
        }
        for item in families
        if int(item["negative_tasks"]) > 0
    ]
    safety_regressions = [
        {
            "dimension": str(item["dimension"]),
            "task_ids": sorted(
                task_id
                for task_id, difference in item["task_differences"].items()
                if float(difference) < 0
            ),
        }
        for item in safety
        if int(item["negative_tasks"]) > 0
    ]
    observed_primary_improvement = any(
        int(item["positive_tasks"]) > 0 for item in families
    )
    requirement = policy["significant_family_requirement"]
    significant_requirement_met = (
        len(significant) == len(families)
        if requirement == "all"
        else bool(significant)
        if requirement == "any"
        else False
    )
    if requirement not in {"all", "any"}:
        raise ValueError("unsupported significant-family decision requirement")
    composite_threshold_met = bool(
        float(composite["task_level_mean_difference"]) >= meaningful
        and float(composite["task_cluster_bootstrap_95_ci"][0]) > 0
    )
    if not policy["composite_threshold_required"]:
        composite_threshold_met = True
    equivalent = all(
        float(item["task_cluster_bootstrap_95_ci"][0]) >= -equivalence
        and float(item["task_cluster_bootstrap_95_ci"][1]) <= equivalence
        for item in families
    )
    if safety_regressions:
        status = "regressed"
    elif observed_primary_improvement and primary_regressions:
        status = "mixed"
    elif primary_regressions:
        status = "regressed"
    elif (
        significant_requirement_met
        and composite_threshold_met
        and not candidate_gate_failures
    ):
        status = "improved"
    elif equivalent and not candidate_gate_failures:
        status = "unchanged"
    else:
        status = "inconclusive"
    blockers = [
        *(
            "safety regression: "
            f"{item['dimension']} on {', '.join(item['task_ids'])}"
            for item in safety_regressions
        ),
        *(
            "primary regression: "
            f"{item['family']} on {', '.join(item['task_ids'])}"
            for item in primary_regressions
        ),
    ]
    if candidate_gate_failures:
        blockers.append(
            f"{len(candidate_gate_failures)} candidate critical task-attempt dimension(s) failed"
        )
    if status == "inconclusive" and not blockers:
        blockers.append("preregistered improvement or equivalence boundary was not met")
    return {
        "status": status,
        "significant_primary_families": significant,
        "primary_regressions": primary_regressions,
        "safety_regressions": safety_regressions,
        "candidate_critical_failures": list(candidate_gate_failures),
        "equivalence_established": equivalent,
        "critical_blockers": blockers,
        "rule": {
            "minimum_meaningful_improvement": meaningful,
            "equivalence_margin": equivalence,
            "alpha": alpha,
            "significant_family_requirement": requirement,
            "composite_threshold_required": policy[
                "composite_threshold_required"
            ],
        },
    }


def _mechanism_summary(
    result: ComparisonResultV3,
    rows: Sequence[Mapping[str, Any]],
    holdout: Sequence[str],
) -> dict[str, Any]:
    expected_per_arm = len(rows) // 2
    required_skill_stages = ("skill_assigned", "skill_registered", "skill_invoked")
    for stage in required_skill_stages:
        stage_value = result.mechanism_summary.get(stage)
        if not isinstance(stage_value, Mapping):
            raise ValueError(f"canonical result is missing {stage} mechanism evidence")
        for arm in ("baseline", "candidate"):
            value = stage_value.get(arm)
            if (
                not isinstance(value, Mapping)
                or set(value) != {"observed", "applicable", "unavailable"}
                or value.get("observed") != expected_per_arm
                or value.get("applicable") != expected_per_arm
                or value.get("unavailable") != 0
            ):
                raise ValueError(
                    f"canonical result has incomplete {stage} evidence for {arm}"
                )
    task_set = set(holdout)
    dimensions = sorted(
        {
            str(dimension)
            for row in rows
            if str(row.get("task_id")) in task_set
            for dimension, role in (row.get("comparison_dimension_roles") or {}).items()
            if role == "mechanism"
        }
    )
    summaries = []
    for dimension in dimensions:
        by_arm: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            if row.get("task_id") in task_set:
                by_arm[str(row.get("variant_id"))].append(
                    float(_score(row, dimension))
                )
        summaries.append(
            {
                "dimension": dimension,
                "baseline_rate": _mean(by_arm["baseline"]),
                "candidate_rate": _mean(by_arm["candidate"]),
            }
        )
    return {
        "role": "mechanism_only_not_task_outcome",
        "canonical_skill_evidence": result.mechanism_summary,
        "deterministic_dimensions": summaries,
    }


def _judge_summary(result: ComparisonResultV3) -> dict[str, Any]:
    labels: dict[str, Counter[str]] = defaultdict(Counter)
    missing: dict[str, int] = defaultdict(int)
    reviews: list[dict[str, Any]] = []
    for pair in result.paired_cases:
        for arm, attempt in (("baseline", pair.baseline), ("candidate", pair.candidate)):
            assert attempt is not None
            for judge_id, review in attempt.judge_reviews.items():
                labels[arm][review.label] += 1
                missing[arm] += int(review.missing_evidence)
                reviews.append(
                    {
                        "attempt_id": attempt.attempt_id,
                        "task_id": pair.task_id,
                        "arm": arm,
                        "judge_id": judge_id,
                        "label": review.label,
                        "reason": review.reason,
                        "missing_evidence": review.missing_evidence,
                    }
                )
    return {
        "claim_role": "secondary_advisory_same_model_family",
        "human_qualified": False,
        "label_counts_by_arm": {
            arm: dict(sorted(counts.items())) for arm, counts in sorted(labels.items())
        },
        "missing_evidence_by_arm": dict(sorted(missing.items())),
        "reviews": reviews,
        "canonical_summary": result.judge_summary,
    }


def _efficiency_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = {
        "cost_usd": "cost_usd",
        "latency_sec": "latency_sec",
        "input_tokens": "input_tokens",
        "output_tokens": "output_tokens",
        "tool_calls": "tool_call_count",
    }
    by_arm: dict[str, dict[str, Any]] = {}
    for arm in ("baseline", "candidate"):
        arm_rows = [row for row in rows if row.get("variant_id") == arm]
        summaries: dict[str, Any] = {}
        for label, field in fields.items():
            values = [
                float(row[field])
                for row in arm_rows
                if isinstance(row.get(field), int | float)
                and not isinstance(row.get(field), bool)
                and math.isfinite(float(row[field]))
            ]
            summaries[label] = {
                "observed": len(values),
                "planned": len(arm_rows),
                "mean": _mean(values) if values else None,
                "total": sum(values) if values else None,
            }
        by_arm[arm] = summaries
    return {
        "role": "secondary_efficiency_not_task_correctness",
        "by_arm": by_arm,
    }


def _direct_links(result: ComparisonResultV3) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    for pair in result.paired_cases:
        for arm, attempt in (("baseline", pair.baseline), ("candidate", pair.candidate)):
            if attempt is None:
                raise ValueError("confirmatory result contains an incomplete pair")
            by_kind = {item.kind: item for item in attempt.evidence_links}
            if set(by_kind) != REQUIRED_LINK_KINDS or any(
                item.status != "resolved" or not item.ref or not item.url
                for item in by_kind.values()
            ):
                raise ValueError(
                    f"attempt {attempt.attempt_id} lacks five resolved evidence links"
                )
            links.append(
                {
                    "attempt_id": attempt.attempt_id,
                    "task_id": pair.task_id,
                    "arm": arm,
                    "links": {
                        kind: by_kind[kind].to_dict()
                        for kind in sorted(REQUIRED_LINK_KINDS)
                    },
                }
            )
    return links


def _aligned_case_evidence(result: ComparisonResultV3) -> list[dict[str, Any]]:
    def safe_attempt(attempt: Any) -> dict[str, Any]:
        deterministic_scores = {
            key: value
            for key, value in attempt.scores.items()
            if not key.startswith("comparison.judge.")
        }
        return {
            "attempt_id": attempt.attempt_id,
            "identity": dict(attempt.identity),
            "passed": attempt.passed,
            "execution_status": attempt.execution_status,
            "evaluation_status": attempt.evaluation_status,
            "evidence_status": attempt.evidence_status,
            "deterministic_scores": deterministic_scores,
            "score_explanations": {
                key: attempt.score_explanations[key]
                for key in deterministic_scores
                if key in attempt.score_explanations
            },
            "sanitized_answer_excerpt": attempt.sanitized_answer_excerpt,
            "tools": list(attempt.tools),
            "actual_query_scope": list(attempt.actual_query_scope),
            "reported_project_identity": attempt.reported_project_identity,
            "cost_usd": attempt.cost_usd,
            "latency_sec": attempt.latency_sec,
            "input_tokens": attempt.input_tokens,
            "output_tokens": attempt.output_tokens,
            "tool_calls": attempt.tool_calls,
            "execution_fingerprint": attempt.execution_fingerprint,
            "runtime_lock_digest": attempt.runtime_lock_digest,
        }

    rows: list[dict[str, Any]] = []
    for pair in result.paired_cases:
        if pair.baseline is None or pair.candidate is None:
            raise ValueError("confirmatory result contains an incomplete pair")
        rows.append(
            {
                "pair_id": pair.pair_id,
                "task_id": pair.task_id,
                "task_label": pair.task_label,
                "harness": pair.harness,
                "attempt": pair.attempt,
                "canonical_pair_status": pair.status,
                "dimension_changes": [
                    item.to_dict() for item in pair.dimension_changes
                ],
                "baseline": safe_attempt(pair.baseline),
                "candidate": safe_attempt(pair.candidate),
            }
        )
    return rows


def _selection_unsigned(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    unsigned.pop("selection_digest", None)
    return unsigned


def _audit_attested_digest(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("reviewer_attestations", None)
    return stable_digest(unsigned)


def _post_result_audit_ids(result: ComparisonResultV3) -> set[str]:
    required: set[str] = set()
    for pair in result.paired_cases:
        critical_failure = any(
            change.critical and change.candidate is False
            for change in pair.dimension_changes
        )
        if pair.status in {"improved", "regressed", "mixed"} or critical_failure:
            for attempt in (pair.baseline, pair.candidate):
                if attempt is not None:
                    required.add(attempt.attempt_id)
    return required


def _validate_trace_audit(  # noqa: C901 - one fail-closed review gate.
    *,
    result: ComparisonResultV3,
    selection_path: Path,
    review_path: Path,
    profile: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    campaign_id: str,
) -> dict[str, Any]:
    selection = _load_json(selection_path, "trace-audit selection")
    review = _load_json(review_path, "completed trace audit")
    supplied_selection_digest = str(selection.get("selection_digest") or "")
    if (
        selection.get("schema_version") != 1
        or selection.get("kind") != "blinded_trace_audit_selection"
        or selection.get("selection_frozen_before_execution") is not True
        or selection.get("preview_digest") != result.preview_digest
        or not _is_digest(supplied_selection_digest)
        or supplied_selection_digest != stable_digest(_selection_unsigned(selection))
    ):
        raise ValueError("trace-audit selection is stale or malformed")
    if selection.get("population_pairs") != len(result.paired_cases):
        raise ValueError("trace-audit selection population disagrees with result")
    fraction = float(selection.get("sampling_fraction") or 0)
    minimum = float(profile["trace_audit"]["minimum_fraction"])
    if not 0 < minimum <= fraction <= 1:
        raise ValueError("trace-audit selection is below the frozen minimum")
    selected_pairs = selection.get("selected_pairs")
    if not isinstance(selected_pairs, list) or not selected_pairs:
        raise ValueError("trace-audit selection contains no paired artifacts")
    selected_ids: list[str] = []
    selected_tasks: set[str] = set()
    for item in selected_pairs:
        if not isinstance(item, Mapping) or any(
            key in item for key in ("baseline_attempt_id", "candidate_attempt_id")
        ):
            raise ValueError("trace-audit selection exposes treatment identity")
        if set(item) != {
            "pair_token",
            "task_id",
            "harness",
            "attempt",
            "partition",
            "artifact_a_attempt_id",
            "artifact_b_attempt_id",
        }:
            raise ValueError("trace-audit selected pair is malformed")
        selected_tasks.add(str(item["task_id"]))
        selected_ids.extend(
            [str(item["artifact_a_attempt_id"]), str(item["artifact_b_attempt_id"])]
        )
    if (
        selection.get("selected_attempt_ids") != sorted(selected_ids)
        or len(selected_ids) != len(set(selected_ids))
        or not set(selected_ids) <= _canonical_attempt_ids(result)
    ):
        raise ValueError("trace-audit selected attempt identities disagree")
    family_path = profile["trace_audit"].get("behavior_families_path")
    if family_path:
        families = _nested(preregistration, family_path, "behavior families")
        if not isinstance(families, Mapping):
            raise ValueError("trace-audit behavior families are malformed")
        for family, task_ids in families.items():
            if not selected_tasks.intersection(str(item) for item in task_ids):
                raise ValueError(f"trace-audit selection misses behavior family {family}")
    required_ids = set(selected_ids) | _post_result_audit_ids(result)
    if (
        review.get("schema_version") != 1
        or review.get("kind") != "completed_blinded_trace_audit"
        or review.get("status") != "completed"
        or review.get("campaign_id") != campaign_id
        or review.get("study_id") != result.comparison_id
        or review.get("preview_digest") != result.preview_digest
        or review.get("result_digest") != result.result_digest
        or review.get("selection_digest") != supplied_selection_digest
        or review.get("selected_attempt_ids") != sorted(selected_ids)
        or review.get("required_attempt_ids") != sorted(required_ids)
    ):
        raise ValueError("completed trace audit is incomplete or targets another result")
    reviewed = review.get("reviewed_attempts")
    if not isinstance(reviewed, list) or {
        str(item.get("attempt_id") or "")
        for item in reviewed
        if isinstance(item, Mapping)
    } != required_ids:
        raise ValueError("completed trace audit omits a required attempt")
    attestations = review.get("reviewer_attestations")
    if not isinstance(attestations, list) or len(attestations) != 2:
        raise ValueError("completed trace audit requires two signed reviewers")
    reviewers = {
        str(item.get("reviewer") or "")
        for item in attestations
        if isinstance(item, Mapping)
    }
    if len(reviewers) != 2 or "" in reviewers:
        raise ValueError("trace audit requires two distinct reviewers")
    attested_digest = _audit_attested_digest(review)
    for attestation in attestations:
        if (
            not isinstance(attestation, Mapping)
            or set(attestation) != {"reviewer", "reviewed_at", "artifact_digest"}
            or not str(attestation.get("reviewed_at") or "")
            or attestation.get("artifact_digest") != attested_digest
        ):
            raise ValueError("trace-audit reviewer attestation is stale")
    for artifact in reviewed:
        if not isinstance(artifact, Mapping) or set(artifact) != {
            "attempt_id",
            "reviews",
            "adjudication",
        }:
            raise ValueError("trace-audit artifact review is malformed")
        reviews = artifact.get("reviews")
        if not isinstance(reviews, list) or len(reviews) != 2:
            raise ValueError("each trace-audit artifact requires two reviews")
        dispositions: set[str] = set()
        artifact_reviewers: set[str] = set()
        for item in reviews:
            if not isinstance(item, Mapping) or set(item) != {
                "reviewer",
                "disposition",
                "checks",
                "reason",
            }:
                raise ValueError("trace-audit review is malformed")
            artifact_reviewers.add(str(item["reviewer"]))
            dispositions.add(str(item["disposition"]))
            checks = item["checks"]
            if (
                not isinstance(checks, Mapping)
                or set(checks) != REQUIRED_AUDIT_CHECKS
                or any(value is not True for value in checks.values())
            ):
                raise ValueError("trace-audit review did not verify every check")
        if artifact_reviewers != reviewers:
            raise ValueError("trace-audit artifact reviewers disagree")
        adjudication = artifact.get("adjudication")
        if len(dispositions) > 1:
            if not isinstance(adjudication, Mapping):
                raise ValueError("discordant trace reviews require adjudication")
            effective = str(adjudication.get("disposition") or "")
        elif adjudication is not None:
            raise ValueError("concordant trace reviews cannot carry adjudication")
        else:
            effective = next(iter(dispositions))
        if effective != "verified":
            raise ValueError("trace-audit finding requires canonical result correction")
    return {
        "status": "completed",
        "selection_digest": supplied_selection_digest,
        "selection_file_sha256": _sha256(selection_path),
        "completed_audit_digest": stable_digest(review),
        "completed_audit_file_sha256": _sha256(review_path),
        "selected_attempts": len(selected_ids),
        "required_attempts": len(required_ids),
        "reviewers": sorted(reviewers),
    }


def _preregistered_analysis(
    *,
    rows: Sequence[Mapping[str, Any]],
    profile: Mapping[str, Any],
    development: Sequence[str],
    holdout: Sequence[str],
    tags: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    bootstrap = profile["bootstrap"]
    samples = int(bootstrap["samples"])
    seed = int(bootstrap["seed"])
    raw_families: list[dict[str, Any]] = []
    raw_p: dict[str, float] = {}
    for raw_family in profile["primary_families"]:
        family = dict(raw_family)
        task_ids = _task_ids_for_family(family, holdout, tags)
        analysis = _metric_analysis(
            rows=rows,
            task_ids=task_ids,
            contract=family,
            samples=samples,
            seed=seed,
        )
        value = {"id": family["id"], "task_ids": list(task_ids), **analysis}
        raw_families.append(value)
        raw_p[str(family["id"])] = float(analysis["exact_two_sided_sign_p"])
    adjusted = _holm(raw_p)
    family_count = len(raw_families)
    families = [
        {
            **item,
            "holm_adjusted_p": adjusted[str(item["id"])],
            "multiplicity_family": "preregistered_primary_only",
        }
        for item in raw_families
    ]
    attainability = [
        {
            "family": item["id"],
            "task_count": item["task_count"],
            "minimum_raw_two_sided_sign_p": min(
                1.0,
                2 / (2 ** int(item["task_count"])),
            ),
            "minimum_holm_adjusted_p_at_rank_one": min(
                1.0,
                family_count * 2 / (2 ** int(item["task_count"])),
            ),
            "can_clear_frozen_alpha": (
                family_count * 2 / (2 ** int(item["task_count"]))
                <= float(profile["decision"]["alpha"])
            ),
        }
        for item in families
    ]
    composite = _metric_analysis(
        rows=rows,
        task_ids=holdout,
        contract=profile["primary_composite"],
        samples=samples,
        seed=seed,
    )
    safety = _safety_analysis(
        rows=rows,
        holdout=holdout,
        dimensions=profile.get("safety_dimensions") or [],
        samples=samples,
        seed=seed,
    )
    gate_failures = _candidate_gate_failures(
        rows=rows,
        holdout=holdout,
        dimensions=profile.get("candidate_all_attempt_dimensions") or [],
    )
    finding = _decision(
        profile=profile,
        families=families,
        composite=composite,
        safety=safety,
        candidate_gate_failures=gate_failures,
    )
    return {
        "primary_partition": "holdout",
        "inference_unit": "task",
        "attempt_role": "within_task_replication",
        "development_is_primary_evidence": False,
        "holdout": {
            "task_count": len(holdout),
            "task_ids": list(holdout),
            "primary_families": families,
            "primary_composite": composite,
            "safety_conjunction_gates": safety,
            "candidate_all_attempt_gate_failures": gate_failures,
            "primary_test_attainability": attainability,
        },
        "development_descriptive": {
            "task_count": len(development),
            "task_ids": list(development),
            "dimensions": _descriptive_dimensions(rows, development),
            "excluded_from_primary_inference": True,
        },
        "finding": finding,
        "sensitivity_analysis": profile["sensitivity_analysis"],
    }


def analyze(
    *,
    attempts_path: Path,
    result_path: Path,
    campaign_manifest_path: Path,
    preregistration_path: Path,
    study_id: str,
    audit_selection_path: Path,
    audit_review_path: Path,
    profile_path: Path = PROFILE_PATH,
) -> dict[str, Any]:
    manifest = _load_json(campaign_manifest_path, "conference manifest")
    campaign_preregistration = _load_json(
        preregistration_path, "campaign preregistration"
    )
    if (
        campaign_preregistration.get("status") != "frozen_before_execution"
        or manifest.get("id") != campaign_preregistration.get("id")
    ):
        raise ValueError("campaign preregistration is not frozen for this manifest")
    study = _campaign_study(manifest, study_id)
    profile, profile_sha256 = _profile_for_study(
        profile_path,
        study_id,
        campaign_id=str(manifest.get("id") or ""),
    )
    repository_preregistration, preregistration_binding = (
        _load_profile_preregistration(
            profile, profile_path=profile_path, study_id=study_id
        )
    )
    repo_root = campaign_manifest_path.resolve().parents[3]
    locked_input_bindings = _preregistered_input_bindings(
        profile=profile,
        preregistration=repository_preregistration,
        profile_path=profile_path,
        repo_root=repo_root,
    )
    spec_path = (campaign_manifest_path.parent / str(study["spec"])).resolve()
    spec = load_comparison(spec_path, repo_root=repo_root)
    tasks = _public_tasks(repo_root / spec.taskset.tasks)
    development, holdout, tags = _partition_tasks(
        profile, repository_preregistration, tasks
    )
    rows = _read_rows(attempts_path)
    _validate_matrix(
        rows=rows,
        tasks=[str(item["id"]) for item in tasks],
        attempts=spec.execution.attempts,
        harnesses=spec.execution.harnesses,
    )
    result = read_comparison_result(result_path)
    if not isinstance(result, ComparisonResultV3):
        raise ValueError("confirmatory analysis requires ComparisonResultV3")
    approved = _approved_execution_lock(rows)
    _recompute_canonical_result(result, rows, approved)
    bindings = _validate_bindings(
        result=result,
        rows=rows,
        approved=approved,
        spec=spec,
        study=study,
        profile=profile,
        repo_root=repo_root,
    )
    validity_by_id = {item.task_id: item for item in result.task_validity}
    if set(validity_by_id) != {str(item["id"]) for item in tasks}:
        raise ValueError("canonical task validity does not cover the frozen taskset")
    trace_audit = _validate_trace_audit(
        result=result,
        selection_path=audit_selection_path,
        review_path=audit_review_path,
        profile=profile,
        preregistration=repository_preregistration,
        campaign_id=str(manifest["id"]),
    )
    deterministic = _preregistered_analysis(
        rows=rows,
        profile=profile,
        development=development,
        holdout=holdout,
        tags=tags,
    )
    report: dict[str, Any] = {
        "schema_version": 2,
        "kind": "community_skill_confirmatory_analysis",
        "campaign_id": manifest["id"],
        "study_id": study_id,
        "status": "complete",
        "scope": campaign_preregistration["claims_scope"],
        "exact_revisions": {
            "baseline": profile["baseline_commit"],
            "candidate": profile["candidate_commit"],
        },
        "evidence_project": result.evidence_project,
        "counts": {
            "development_tasks": len(development),
            "holdout_tasks": len(holdout),
            "attempts_per_task_per_arm": spec.execution.attempts,
            "arms": 2,
            "rows": len(rows),
        },
        "deterministic": deterministic,
        "judge": _judge_summary(result),
        "mechanism": _mechanism_summary(result, rows, holdout),
        "efficiency": _efficiency_summary(rows),
        "task_validity": {
            "development": [validity_by_id[item].to_dict() for item in development],
            "holdout": [validity_by_id[item].to_dict() for item in holdout],
        },
        "canonical_aligned_analysis": result.aligned_analysis.to_dict(),
        "aligned_cases": _aligned_case_evidence(result),
        "direct_links": _direct_links(result),
        "trace_audit": trace_audit,
        "integrity": {
            **bindings,
            "attempts_sha256": _sha256(attempts_path),
            "attempts_rows_digest": result.integrity["rows_digest"],
            "result_file_sha256": _sha256(result_path),
            "campaign_manifest_sha256": _sha256(campaign_manifest_path),
            "campaign_preregistration_sha256": _sha256(preregistration_path),
            "repository_preregistration": preregistration_binding,
            "preregistered_locked_inputs": locked_input_bindings,
            "analysis_profile_sha256": profile_sha256,
            "canonical_result_recomputed": True,
        },
        "limitations": [
            campaign_preregistration["diagnostic_no_skill_contrast"][
                "claim_limitation"
            ],
            "The Agent and advisory judge share a model family; judge evidence is not a qualified outcome.",
            "The mixed-effects sensitivity analysis is not implemented and no sensitivity claim is made.",
            "Four attempts estimate within-task stochasticity but do not increase the independent task count.",
            "Repositories are analyzed separately and are not a universal Skill ranking.",
        ],
    }
    report["analysis_digest"] = stable_digest(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--audit-selection", type=Path, required=True)
    parser.add_argument("--audit-review", type=Path, required=True)
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
    parser.add_argument("--profiles", type=Path, default=PROFILE_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(
        attempts_path=args.attempts,
        result_path=args.result,
        campaign_manifest_path=args.manifest,
        preregistration_path=args.preregistration,
        study_id=args.study_id,
        audit_selection_path=args.audit_selection,
        audit_review_path=args.audit_review,
        profile_path=args.profiles,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
