#!/usr/bin/env python3
"""Generate one strict public report from one canonical V2/V3 Skill result.

V2 results predate TaskValidityV1, EvidenceTopologyV1, aligned-analysis,
score-explanation, and sanitized-excerpt contracts.  Reports generated from
V2 therefore mark those claims unavailable instead of reconstructing them
from display data.  V3 keeps the stronger contract and its existing checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fugue.bench.candidates import stable_digest
from fugue.bench.comparison import (
    ComparisonResultV2,
    ComparisonResultV3,
    ComparisonSpecV1,
    load_comparison,
    read_comparison_result,
)

SupportedResult = ComparisonResultV2 | ComparisonResultV3

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
TEMPLATE = ROOT / "scientific-report-template.json"
MANIFEST = ROOT / "campaign-manifest.json"
TERMINAL_STATES = {
    "completed",
    "failed",
    "cancelled",
    "interrupted",
    "not_applicable",
}
JUDGE_LABELS = {"unusable", "weak", "adequate", "strong", "exceptional"}
BEHAVIORAL_STATUSES = {
    "invalid",
    "incomplete",
    "improved",
    "regressed",
    "mixed",
    "unchanged",
}
TASK_VALIDITY_STATUSES = {
    "valid",
    "non_discriminating",
    "drifted",
    "invalid",
    "inconclusive",
    "not_assessed",
}
REPORT_FIELDS = {
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
LOCKED_LIMITATIONS = (
    "The initial canary has one attempt per task and treatment, so it cannot establish repeatability.",
    "The Agent and blind judge use the same model family; deterministic gates remain authoritative.",
    "Results are task- and revision-specific and do not rank repositories or Skills universally.",
)
V2_LIMITATIONS = (
    "ComparisonResultV2 does not contain TaskValidityV1; task validity is not assessed or reconstructed in this report.",
    "ComparisonResultV2 does not contain EvidenceTopologyV1 or aligned-analysis contracts; this report verifies its declared result destination and resolved attempt links without making source-topology or drift claims.",
    "ComparisonResultV2 does not contain cohort lineage, the baseline source revision, score explanations, or sanitized answer excerpts; revisions come from the checked-in campaign contract, the candidate revision is matched to result metadata, and unavailable presentation fields remain empty.",
)
_SHA40 = re.compile(r"[0-9a-f]{40}")
_FORBIDDEN_PUBLIC_KEYS = {
    "private_labels",
    "private_label",
    "authored_reference",
    "adjudicated_label",
    "expected",
}


def _load_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _within(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"{label} must remain inside the repository")
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{label} must be a regular file")
    return resolved


def _study_contract(
    *,
    campaign_root: Path,
    spec_path: Path,
    study_id: str,
) -> dict[str, Any]:
    manifest = _load_object(campaign_root / "campaign-manifest.json", "campaign manifest")
    studies = manifest.get("studies")
    if not isinstance(studies, list):
        raise ValueError("campaign studies must be a list")
    matches = [
        item
        for item in studies
        if isinstance(item, Mapping) and item.get("id") == study_id
    ]
    if len(matches) != 1:
        raise ValueError("result does not identify exactly one campaign Study")
    study = dict(matches[0])
    declared_spec = (campaign_root / str(study.get("spec") or "")).resolve()
    if declared_spec != spec_path.resolve():
        raise ValueError("Study spec path disagrees with the campaign manifest")
    for field in ("baseline_commit", "candidate_commit"):
        if not _SHA40.fullmatch(str(study.get(field) or "")):
            raise ValueError(f"campaign Study {field} is not an exact revision")
    return study


def _public_task_ids(path: Path) -> tuple[str, ...]:
    ids: list[str] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"public task line {number} must be an object")
        task_id = str(value.get("id") or "")
        if not task_id:
            raise ValueError(f"public task line {number} has no id")
        ids.append(task_id)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("public task ids must be nonzero and unique")
    return tuple(ids)


def _evaluator_digest(evaluator: Any, repo_root: Path) -> str:
    value = evaluator.to_dict()
    for field, digest_field in (
        ("scorer", "scorer_sha256"),
        ("calibration", "calibration_sha256"),
    ):
        relative = getattr(evaluator, field)
        if relative:
            value[digest_field] = _sha256(
                _within(repo_root / relative, repo_root, f"evaluator {field}")
            )
    return stable_digest(value)


def _verify_exact_spec_v3(  # noqa: C901 - one bounded cross-artifact audit.
    *,
    result: ComparisonResultV3,
    spec: ComparisonSpecV1,
    study: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, str]:
    if result.comparison_id != spec.id:
        raise ValueError("result comparison id disagrees with the Study spec")
    if spec.execution.evidence_project != result.evidence_project:
        raise ValueError("result project disagrees with the Study spec")
    if study.get("evidence_project") != result.evidence_project:
        raise ValueError("result project disagrees with the campaign manifest")
    if (
        spec.execution.evidence_destination is None
        or spec.execution.evidence_destination.to_dict()
        != result.evidence_topology.result_destination.to_dict()
    ):
        raise ValueError("result evidence destination disagrees with the Study spec")
    if spec.execution.attempts != 1:
        raise ValueError("community canary reports require exactly one attempt")
    judges = [item for item in spec.evaluators if item.type == "llm_judge"]
    if len(judges) != 1 or judges[0].required is not False:
        raise ValueError("community report requires one advisory judge")
    if judges[0].profile != spec.execution.model:
        raise ValueError("community report requires the locked same-family judge")

    lineage = result.cohort_lineage
    execution = lineage.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("result cohort execution lineage is unavailable")
    expected_execution = {
        "model": spec.execution.model,
        "harnesses": list(spec.execution.harnesses),
        "trace_content": spec.execution.trace_content,
        "environment_digest": stable_digest(spec.execution.environment),
        "source_evidence_project": spec.execution.source_evidence_project,
        "source_evidence_destination": (
            spec.execution.source_evidence_destination.to_dict()
            if spec.execution.source_evidence_destination is not None
            else None
        ),
        "result_evidence_project": spec.execution.evidence_project,
        "result_evidence_destination": spec.execution.evidence_destination.to_dict(),
    }
    if dict(execution) != expected_execution:
        raise ValueError("result execution lineage disagrees with the Study spec")

    tasks_path = _within(repo_root / spec.taskset.tasks, repo_root, "public taskset")
    labels_path = _within(
        repo_root / spec.taskset.private_labels,
        repo_root,
        "private-label artifact",
    )
    if lineage.get("taskset_digest") != _sha256(tasks_path):
        raise ValueError("result taskset revision disagrees with the Study spec")
    if lineage.get("private_labels_digest") != _sha256(labels_path):
        raise ValueError("result private-label revision disagrees with the Study spec")
    expected_scorers = {
        item.id: _evaluator_digest(item, repo_root) for item in spec.evaluators
    }
    if lineage.get("scorer_digests") != expected_scorers:
        raise ValueError("result scorer revisions disagree with the Study spec")

    arms = lineage.get("arms")
    if not isinstance(arms, Mapping):
        raise ValueError("result cohort arm lineage is unavailable")
    revisions = {
        "baseline": str(study["baseline_commit"]),
        "candidate": str(study["candidate_commit"]),
    }
    candidates = {"baseline": spec.baseline, "candidate": spec.candidate}
    if any(len(candidate.skills) != 1 for candidate in candidates.values()):
        raise ValueError("community Skill reports require one Skill per arm")
    candidate_revision: Mapping[str, Any] | None = None
    for arm, candidate in candidates.items():
        raw_arm = arms.get(arm)
        if not isinstance(raw_arm, Mapping):
            raise ValueError(f"result {arm} lineage is unavailable")
        if raw_arm.get("behavior_digest") != stable_digest(candidate.behavior()):
            raise ValueError(f"result {arm} behavior disagrees with the Study spec")
        source_revisions = raw_arm.get("source_revisions")
        if not isinstance(source_revisions, list) or len(source_revisions) != 1:
            raise ValueError(f"result {arm} must bind one exact Skill revision")
        revision = source_revisions[0]
        if not isinstance(revision, Mapping):
            raise ValueError(f"result {arm} Skill revision is invalid")
        if (
            revision.get("kind") != "skill"
            or revision.get("id") != candidate.skills[0]
            or revision.get("version_identity") != f"git:{revisions[arm]}"
            or not str(revision.get("runtime_digest") or "").startswith("sha256:")
        ):
            raise ValueError(f"result {arm} Skill revision disagrees with the campaign")
        if arm == "candidate":
            candidate_revision = revision

    if candidate_revision is None or len(result.candidate_source_revisions) != 1:
        raise ValueError("result candidate source revision is unavailable")
    published_candidate = result.candidate_source_revisions[0]
    if (
        published_candidate.kind != candidate_revision["kind"]
        or published_candidate.id != candidate_revision["id"]
        or published_candidate.version_identity
        != candidate_revision["version_identity"]
        or published_candidate.runtime_digest
        != candidate_revision["runtime_digest"]
    ):
        raise ValueError("result candidate source revision disagrees with its lineage")

    expected_coordinates = {
        (task_id, harness, attempt)
        for task_id in _public_task_ids(tasks_path)
        for harness in spec.execution.harnesses
        for attempt in range(1, spec.execution.attempts + 1)
    }
    observed_coordinates = {
        (pair.task_id, pair.harness, pair.attempt) for pair in result.paired_cases
    }
    if observed_coordinates != expected_coordinates:
        raise ValueError("result task matrix disagrees with the Study spec")
    return revisions


def _verify_exact_spec_v2(
    *,
    result: ComparisonResultV2,
    spec: ComparisonSpecV1,
    study: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, str]:
    """Verify only identities that a canonical V2 result actually carries.

    V2 has no cohort-lineage or evidence-topology object.  In particular, it
    cannot independently prove the baseline source commit, so the public
    report keeps that fact in its locked V2 limitations rather than inferring
    it from labels, candidate hashes, or presentation fields.
    """

    if result.comparison_id != spec.id:
        raise ValueError("result comparison id disagrees with the Study spec")
    if spec.execution.evidence_project != result.evidence_project:
        raise ValueError("result project disagrees with the Study spec")
    if study.get("evidence_project") != result.evidence_project:
        raise ValueError("result project disagrees with the campaign manifest")
    destination = spec.execution.evidence_destination
    if destination is None or result.evidence_destination != destination.to_dict():
        raise ValueError("result evidence destination disagrees with the Study spec")
    if spec.execution.attempts != 1:
        raise ValueError("community canary reports require exactly one attempt")
    judges = [item for item in spec.evaluators if item.type == "llm_judge"]
    if len(judges) != 1 or judges[0].required is not False:
        raise ValueError("community report requires one advisory judge")
    if judges[0].profile != spec.execution.model:
        raise ValueError("community report requires the locked same-family judge")

    revisions = {
        "baseline": str(study["baseline_commit"]),
        "candidate": str(study["candidate_commit"]),
    }
    if len(spec.baseline.skills) != 1 or len(spec.candidate.skills) != 1:
        raise ValueError("community Skill reports require one Skill per arm")
    if len(result.candidate_source_revisions) != 1:
        raise ValueError("result candidate source revision is unavailable")
    published_candidate = result.candidate_source_revisions[0]
    if (
        published_candidate.kind != "skill"
        or published_candidate.id != spec.candidate.skills[0]
        or published_candidate.version_identity != f"git:{revisions['candidate']}"
        or not published_candidate.runtime_digest.startswith("sha256:")
    ):
        raise ValueError("result candidate Skill revision disagrees with the campaign")

    tasks_path = _within(repo_root / spec.taskset.tasks, repo_root, "public taskset")
    expected_coordinates = {
        (task_id, harness, attempt)
        for task_id in _public_task_ids(tasks_path)
        for harness in spec.execution.harnesses
        for attempt in range(1, spec.execution.attempts + 1)
    }
    observed_coordinates = {
        (pair.task_id, pair.harness, pair.attempt) for pair in result.paired_cases
    }
    if observed_coordinates != expected_coordinates:
        raise ValueError("result task matrix disagrees with the Study spec")
    return revisions


def _verify_exact_spec(
    *,
    result: SupportedResult,
    spec: ComparisonSpecV1,
    study: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, str]:
    if isinstance(result, ComparisonResultV3):
        return _verify_exact_spec_v3(
            result=result,
            spec=spec,
            study=study,
            repo_root=repo_root,
        )
    if isinstance(result, ComparisonResultV2):
        return _verify_exact_spec_v2(
            result=result,
            spec=spec,
            study=study,
            repo_root=repo_root,
        )
    raise ValueError("scientific reports require ComparisonResultV2 or V3")


def _attempts(result: SupportedResult) -> list[tuple[Any, str, Any]]:
    selected: list[tuple[Any, str, Any]] = []
    for pair in result.paired_cases:
        for arm, attempt in (("baseline", pair.baseline), ("candidate", pair.candidate)):
            if attempt is None:
                raise ValueError("scientific reports require complete paired attempts")
            if attempt.execution_status not in TERMINAL_STATES:
                raise ValueError("scientific reports require terminal attempts")
            if attempt.evidence_status != "reconciled":
                raise ValueError("scientific reports require reconciled attempt evidence")
            if len(attempt.evidence_links) != 5 or any(
                link.status != "resolved" or not link.ref or not link.url
                for link in attempt.evidence_links
            ):
                raise ValueError("scientific reports require five resolved Weave links")
            selected.append((pair, arm, attempt))
    if len(selected) != result.rows:
        raise ValueError("scientific report attempt count disagrees with result rows")
    return selected


def _deterministic_rows(result: SupportedResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in result.paired_cases:
        assert pair.baseline is not None and pair.candidate is not None
        baseline_scores = {
            key: value
            for key, value in pair.baseline.scores.items()
            if not key.startswith("comparison.judge.")
        }
        candidate_scores = {
            key: value
            for key, value in pair.candidate.scores.items()
            if not key.startswith("comparison.judge.")
        }
        baseline_explanations = (
            {
                key: pair.baseline.score_explanations[key]
                for key in baseline_scores
            }
            if isinstance(result, ComparisonResultV3)
            else {}
        )
        candidate_explanations = (
            {
                key: pair.candidate.score_explanations[key]
                for key in candidate_scores
            }
            if isinstance(result, ComparisonResultV3)
            else {}
        )
        rows.append(
            {
                "pair_id": pair.pair_id,
                "task_id": pair.task_id,
                "task_label": pair.task_label,
                "harness": pair.harness,
                "attempt": pair.attempt,
                "classification": pair.status,
                "dimensions": [item.to_dict() for item in pair.dimension_changes],
                "baseline": {
                    "attempt_id": pair.baseline.attempt_id,
                    "passed": pair.baseline.passed,
                    "execution_status": pair.baseline.execution_status,
                    "scores": baseline_scores,
                    "score_explanations": baseline_explanations,
                    "sanitized_answer_excerpt": (
                        pair.baseline.sanitized_answer_excerpt
                        if isinstance(result, ComparisonResultV3)
                        else None
                    ),
                    "presentation_evidence_status": (
                        "available"
                        if isinstance(result, ComparisonResultV3)
                        else "unavailable_in_v2"
                    ),
                    "tools": list(pair.baseline.tools),
                },
                "candidate": {
                    "attempt_id": pair.candidate.attempt_id,
                    "passed": pair.candidate.passed,
                    "execution_status": pair.candidate.execution_status,
                    "scores": candidate_scores,
                    "score_explanations": candidate_explanations,
                    "sanitized_answer_excerpt": (
                        pair.candidate.sanitized_answer_excerpt
                        if isinstance(result, ComparisonResultV3)
                        else None
                    ),
                    "presentation_evidence_status": (
                        "available"
                        if isinstance(result, ComparisonResultV3)
                        else "unavailable_in_v2"
                    ),
                    "tools": list(pair.candidate.tools),
                },
            }
        )
    return rows


def _judge_rows(result: SupportedResult, judge_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair, arm, attempt in _attempts(result):
        if isinstance(result, ComparisonResultV3) and set(attempt.judge_reviews) != {
            judge_id
        }:
            raise ValueError("every attempt must carry the one locked advisory judge")
        review = attempt.judge_reviews.get(judge_id)
        if review is None:
            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "task_id": pair.task_id,
                    "arm": arm,
                    "attempt_id": attempt.attempt_id,
                    "judge_id": judge_id,
                    "role": "advisory_same_family",
                    "status": "unavailable",
                    "label": None,
                    "reason": (
                        "ComparisonResultV2 contains no completed advisory judge "
                        "review for this attempt."
                    ),
                    "missing_evidence": True,
                    "cost_status": "unavailable",
                    "observed_cost_usd": None,
                    "accounted_reserve_usd": None,
                }
            )
            continue
        if review.label not in JUDGE_LABELS:
            raise ValueError("judge result uses an unanchored label")
        rows.append(
            {
                "pair_id": pair.pair_id,
                "task_id": pair.task_id,
                "arm": arm,
                "attempt_id": attempt.attempt_id,
                "judge_id": judge_id,
                "role": "advisory_same_family",
                "status": "observed",
                "label": review.label,
                "reason": review.reason,
                "missing_evidence": review.missing_evidence,
                "cost_status": review.cost_status,
                "observed_cost_usd": review.observed_cost_usd,
                "accounted_reserve_usd": review.accounted_reserve_usd,
            }
        )
    return rows


def _skill_evidence(
    result: SupportedResult,
    spec: ComparisonSpecV1,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skills = {"baseline": list(spec.baseline.skills), "candidate": list(spec.candidate.skills)}
    for stage in (
        "skill_assigned",
        "skill_registered",
        "skill_invoked",
        "relevant_source_returned",
        "relevant_source_opened",
        "relevant_source_used",
    ):
        stage_value = result.mechanism_summary.get(stage)
        if not isinstance(stage_value, Mapping):
            raise ValueError(f"result is missing {stage} mechanism evidence")
        for arm in ("baseline", "candidate"):
            value = stage_value.get(arm)
            if not isinstance(value, Mapping) or set(value) != {
                "observed",
                "applicable",
                "unavailable",
            }:
                raise ValueError(f"result {stage} {arm} evidence is malformed")
            rows.append(
                {
                    "stage": stage,
                    "arm": arm,
                    "skill_ids": skills[arm],
                    "observed": int(value["observed"]),
                    "applicable": int(value["applicable"]),
                    "unavailable": int(value["unavailable"]),
                }
            )
    return rows


def _coverage(values: Sequence[float | None]) -> dict[str, Any]:
    observed = [float(value) for value in values if value is not None]
    status = (
        "complete"
        if len(observed) == len(values)
        else "unavailable"
        if not observed
        else "partial"
    )
    return {
        "status": status,
        "observed_rows": len(observed),
        "expected_rows": len(values),
        "total": round(sum(observed), 6) if observed else None,
    }


def _efficiency(result: SupportedResult) -> dict[str, Any]:
    by_arm = {"baseline": [], "candidate": []}
    judge_reviews = []
    selected_attempts = _attempts(result)
    for _pair, arm, attempt in selected_attempts:
        by_arm[arm].append(attempt)
        judge_reviews.extend(attempt.judge_reviews.values())
    latency = {
        arm: _coverage([item.latency_sec for item in attempts])
        for arm, attempts in by_arm.items()
    }
    tokens: dict[str, Any] = {}
    for arm, attempts in by_arm.items():
        input_summary = _coverage([item.input_tokens for item in attempts])
        output_summary = _coverage([item.output_tokens for item in attempts])
        totals = [
            (
                float(item.input_tokens) + float(item.output_tokens)
                if item.input_tokens is not None and item.output_tokens is not None
                else None
            )
            for item in attempts
        ]
        tokens[arm] = {
            "input": input_summary,
            "output": output_summary,
            "combined": _coverage(totals),
        }
    operational = result.operational_summary
    expected_rows = result.rows
    observed_rows = int(operational.get("cost_rows") or 0)
    accounted_rows = int(operational.get("accounted_cost_rows") or 0)
    judge_observed = [
        item.observed_cost_usd
        for item in judge_reviews
        if item.cost_status == "observed" and item.observed_cost_usd is not None
    ]
    judge_reserves = [
        item.accounted_reserve_usd
        for item in judge_reviews
        if item.accounted_reserve_usd is not None
    ]
    cost = {
        "agent_observed": {
            "status": (
                "complete"
                if observed_rows == expected_rows
                else "unavailable"
                if observed_rows == 0
                else "partial"
            ),
            "observed_rows": observed_rows,
            "expected_rows": expected_rows,
            "total_usd": operational.get("observed_cost_usd"),
        },
        "accounted": {
            "status": (
                "complete"
                if accounted_rows == expected_rows
                else "unavailable"
                if accounted_rows == 0
                else "partial"
            ),
            "observed_rows": accounted_rows,
            "expected_rows": expected_rows,
            "total_usd": operational.get("accounted_cost_usd"),
        },
        "advisory_judge": {
            "status": (
                "observed"
                if len(judge_observed) == len(selected_attempts)
                else "unavailable"
                if not judge_observed
                else "partial"
            ),
            "observed_reviews": len(judge_observed),
            "expected_reviews": len(selected_attempts),
            "observed_total_usd": (
                round(sum(judge_observed), 6) if judge_observed else None
            ),
            "accounted_reserve_usd": round(sum(judge_reserves), 6),
        },
    }
    return {"latency": latency, "tokens": tokens, "cost": cost}


def _evidence_links(result: SupportedResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair, arm, attempt in _attempts(result):
        for link in attempt.evidence_links:
            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "task_id": pair.task_id,
                    "arm": arm,
                    "attempt_id": attempt.attempt_id,
                    "kind": link.kind,
                    "status": link.status,
                    "ref": link.ref,
                    "url": link.url,
                }
            )
    return rows


def _limitations(template: Mapping[str, Any], result: SupportedResult) -> list[str]:
    locked = template.get("limitations")
    if not isinstance(locked, list) or tuple(locked[:3]) != LOCKED_LIMITATIONS:
        raise ValueError("scientific report template limitations drifted")
    extras = (
        *(V2_LIMITATIONS if isinstance(result, ComparisonResultV2) else ()),
        *result.limitations,
        *result.behavioral_summary.limitations,
        *result.decision.limitations,
    )
    return list(dict.fromkeys((*LOCKED_LIMITATIONS, *extras)))


def _task_validity(result: SupportedResult) -> list[dict[str, Any]]:
    if isinstance(result, ComparisonResultV3):
        return [item.to_dict() for item in result.task_validity]
    return [
        {
            "task_id": pair.task_id,
            "status": "not_assessed",
            "reasons": [
                "ComparisonResultV2 does not carry TaskValidityV1; no task-validity claim is made."
            ],
        }
        for pair in result.paired_cases
    ]


def _validate_template(value: Mapping[str, Any]) -> None:
    if set(value) != REPORT_FIELDS:
        raise ValueError("scientific report template fields drifted")
    if value["schema_version"] != 1 or value["status"] != "pending_execution":
        raise ValueError("scientific report template must remain pending")
    for field in (
        "study_id",
        "evidence_project",
        "task_validity",
        "behavioral_finding",
        "conclusion",
    ):
        if value[field] is not None:
            raise ValueError(f"scientific report template preclaims {field}")
    if value["exact_revisions"] != {"baseline": None, "candidate": None}:
        raise ValueError("scientific report template preclaims exact revisions")
    for field in (
        "deterministic_results",
        "judge_results",
        "skill_use_evidence",
        "evidence_links",
    ):
        if value[field] != []:
            raise ValueError(f"scientific report template preclaims {field}")
    if value["efficiency"] != {"latency": None, "tokens": None, "cost": None}:
        raise ValueError("scientific report template preclaims efficiency")
    if tuple(value["limitations"]) != LOCKED_LIMITATIONS:
        raise ValueError("scientific report template limitations drifted")


def _reject_private_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _FORBIDDEN_PUBLIC_KEYS:
                raise ValueError("scientific report contains a private-label field")
            _reject_private_keys(item)
    elif isinstance(value, list | tuple):
        for item in value:
            _reject_private_keys(item)


def validate_report(  # noqa: C901 - one strict report-schema boundary.
    value: Mapping[str, Any],
) -> None:
    unknown = sorted(set(value) - REPORT_FIELDS)
    missing = sorted(REPORT_FIELDS - set(value))
    if unknown or missing:
        raise ValueError(
            f"scientific report fields disagree: unknown={unknown}, missing={missing}"
        )
    if value["schema_version"] != 1 or value["status"] != "completed":
        raise ValueError("scientific report must be a completed schema-v1 artifact")
    if not str(value["study_id"] or "") or not str(value["evidence_project"] or ""):
        raise ValueError("scientific report identity is incomplete")
    revisions = value["exact_revisions"]
    if not isinstance(revisions, Mapping) or set(revisions) != {"baseline", "candidate"}:
        raise ValueError("scientific report revisions are malformed")
    if any(not _SHA40.fullmatch(str(item)) for item in revisions.values()):
        raise ValueError("scientific report revisions must be exact Git SHAs")
    validity = value["task_validity"]
    if not isinstance(validity, list) or not validity:
        raise ValueError("scientific report requires task validity")
    if any(item.get("status") not in TASK_VALIDITY_STATUSES for item in validity):
        raise ValueError("scientific report contains invalid task validity")
    if any(
        item.get("status") == "not_assessed" and not item.get("reasons")
        for item in validity
    ):
        raise ValueError("unassessed task validity requires an explicit reason")
    finding = value["behavioral_finding"]
    if not isinstance(finding, Mapping) or finding.get("status") not in BEHAVIORAL_STATUSES:
        raise ValueError("scientific report behavioral finding is invalid")
    for field in (
        "deterministic_results",
        "judge_results",
        "skill_use_evidence",
        "evidence_links",
    ):
        if not isinstance(value[field], list) or not value[field]:
            raise ValueError(f"scientific report {field} must be nonzero")
    for item in value["judge_results"]:
        if item.get("status") not in {"observed", "unavailable"}:
            raise ValueError("scientific report contains an invalid judge status")
        if item.get("status") == "observed":
            if item.get("label") not in JUDGE_LABELS:
                raise ValueError("scientific report contains an unanchored judge label")
        elif item.get("label") is not None:
            raise ValueError("unavailable judge evidence cannot carry a label")
    efficiency = value["efficiency"]
    if not isinstance(efficiency, Mapping) or set(efficiency) != {
        "latency",
        "tokens",
        "cost",
    }:
        raise ValueError("scientific report efficiency is malformed")
    links = value["evidence_links"]
    if any(
        item.get("status") != "resolved"
        or not item.get("ref")
        or not item.get("url")
        for item in links
    ):
        raise ValueError("scientific report evidence links are unresolved")
    limitations = value["limitations"]
    if not isinstance(limitations, list) or tuple(limitations[:3]) != LOCKED_LIMITATIONS:
        raise ValueError("scientific report required limitations are missing")
    if not str(value["conclusion"] or "").strip():
        raise ValueError("scientific report conclusion is empty")
    _reject_private_keys(value)


def generate_report(
    *,
    result: SupportedResult,
    spec: ComparisonSpecV1,
    study: Mapping[str, Any],
    template: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    if not isinstance(result, ComparisonResultV2 | ComparisonResultV3):
        raise ValueError("scientific reports require ComparisonResultV2 or V3")
    _validate_template(template)
    if result.integrity.get("status") != "reconciled":
        raise ValueError("scientific reports require reconciled result integrity")
    revisions = _verify_exact_spec(
        result=result,
        spec=spec,
        study=study,
        repo_root=repo_root,
    )
    _attempts(result)
    judges = [item for item in spec.evaluators if item.type == "llm_judge"]
    behavior = result.behavioral_summary
    supported = behavior.supported_claim or "No positive behavioral claim is supported."
    report = {
        "schema_version": 1,
        "status": "completed",
        "study_id": result.comparison_id,
        "evidence_project": result.evidence_project,
        "exact_revisions": revisions,
        "task_validity": _task_validity(result),
        "behavioral_finding": {
            "source_result_schema_version": result.schema_version,
            "task_validity_basis": (
                "canonical_task_validity_v1"
                if isinstance(result, ComparisonResultV3)
                else "not_assessed_in_v2"
            ),
            "evidence_topology_basis": (
                "canonical_evidence_topology_v1"
                if isinstance(result, ComparisonResultV3)
                else "unavailable_in_v2"
            ),
            "status": behavior.status,
            "recommendation": behavior.recommendation,
            "supported_claim": behavior.supported_claim,
            "critical_blockers": list(behavior.critical_blockers),
            "next_action": behavior.next_action,
            "release_decision": result.decision.to_dict(),
        },
        "deterministic_results": _deterministic_rows(result),
        "judge_results": _judge_rows(result, judges[0].id),
        "skill_use_evidence": _skill_evidence(result, spec),
        "efficiency": _efficiency(result),
        "evidence_links": _evidence_links(result),
        "limitations": _limitations(template, result),
        "conclusion": (
            f"{behavior.status.upper()}: {behavior.recommendation} "
            f"{supported} Next action: {behavior.next_action}"
        ),
    }
    validate_report(report)
    return report


def build_report(
    *,
    result_path: Path,
    spec_path: Path,
    repo_root: Path = REPO_ROOT,
    campaign_root: Path = ROOT,
    template_path: Path = TEMPLATE,
) -> dict[str, Any]:
    resolved_result = _within(result_path, repo_root, "comparison result")
    resolved_spec = _within(spec_path, repo_root, "Study spec")
    result = read_comparison_result(resolved_result)
    if not isinstance(result, ComparisonResultV2 | ComparisonResultV3):
        raise ValueError("scientific reports require ComparisonResultV2 or V3")
    spec = load_comparison(resolved_spec, repo_root=repo_root)
    study = _study_contract(
        campaign_root=campaign_root,
        spec_path=resolved_spec,
        study_id=result.comparison_id,
    )
    template = _load_object(template_path, "scientific report template")
    return generate_report(
        result=result,
        spec=spec,
        study=study,
        template=template,
        repo_root=repo_root,
    )


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    report = build_report(
        result_path=args.result,
        spec_path=args.spec,
        repo_root=repo_root,
        campaign_root=repo_root / "examples/comparisons/community-skill-upgrades",
        template_path=(
            repo_root
            / "examples/comparisons/community-skill-upgrades/"
            "scientific-report-template.json"
        ),
    )
    _atomic_write(args.output.resolve(), report)
    print(
        json.dumps(
            {
                "status": "completed",
                "study_id": report["study_id"],
                "behavioral_status": report["behavioral_finding"]["status"],
                "output": args.output.resolve().as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
