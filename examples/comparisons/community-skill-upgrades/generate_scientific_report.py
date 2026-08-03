#!/usr/bin/env python3
"""Generate one strict public report from one canonical V2/V3 Skill result.

V2 results predate TaskValidityV1, EvidenceTopologyV1, aligned-analysis,
score-explanation, and sanitized-excerpt contracts.  Reports generated from
V2 therefore mark those claims unavailable instead of reconstructing them
from display data.  V3 keeps the stronger contract and additionally supports
preregistered, repeated confirmatory studies with a frozen blinded trace audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
TEMPLATE = ROOT / "scientific-report-template-v2.json"
MANIFEST = ROOT / "campaign-manifest.json"
CONFIRMATORY_MANIFEST = ROOT / "conference-campaign-manifest.json"
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
    "source_result",
    "study_contract",
    "preregistration",
    "trace_audit",
    "task_validity",
    "behavioral_finding",
    "deterministic_results",
    "judge_results",
    "mechanism_results",
    "efficiency",
    "evidence_links",
    "limitations",
    "conclusion",
}
COMMON_LIMITATIONS = (
    "The Agent and blind judge use the same model family; deterministic gates remain authoritative.",
    "Results are task- and revision-specific and do not rank repositories or Skills universally.",
)
CANARY_LIMITATION = (
    "The initial canary has one attempt per task and treatment, so it cannot establish repeatability.",
)
LOCKED_LIMITATIONS = (*CANARY_LIMITATION, *COMMON_LIMITATIONS)
V2_LIMITATIONS = (
    "ComparisonResultV2 does not contain TaskValidityV1; task validity is not assessed or reconstructed in this report.",
    "ComparisonResultV2 does not contain EvidenceTopologyV1 or aligned-analysis contracts; this report verifies its declared result destination and resolved attempt links without making source-topology or drift claims.",
    "ComparisonResultV2 does not contain cohort lineage, the baseline source revision, score explanations, or sanitized answer excerpts; revisions come from the checked-in campaign contract, the candidate revision is matched to result metadata, and unavailable presentation fields remain empty.",
)
_SHA40 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_AUDIT_REQUIRED_CHECKS = {
    "attempt_identity",
    "result_project",
    "candidate_and_runtime_locks",
    "skill_registered_opened_and_invoked",
    "host_verifier_receipt",
    "privacy",
    "cleanup",
}
_AUDIT_BLINDING_TERMS = {"arm", "baseline", "candidate", "treatment", "variant"}
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
    matches: list[tuple[Path, Mapping[str, Any], Mapping[str, Any]]] = []
    for manifest_path, collection in (
        (campaign_root / "campaign-manifest.json", "studies"),
        (campaign_root / "conference-campaign-manifest.json", "final_four"),
        (campaign_root / "canary-execution-policy-v2.json", "studies"),
    ):
        if not manifest_path.is_file():
            continue
        manifest = _load_object(manifest_path, "campaign manifest")
        studies = manifest.get(collection)
        if not isinstance(studies, list):
            raise ValueError(f"campaign {collection} must be a list")
        matches.extend(
            (manifest_path, manifest, item)
            for item in studies
            if isinstance(item, Mapping)
            and item.get("kind") != "blinded_judge_diagnostic"
            and item.get("id") == study_id
        )
    if len(matches) != 1:
        raise ValueError("result does not identify exactly one campaign Study")
    manifest_path, manifest, raw_study = matches[0]
    study = dict(raw_study)
    declared_spec = (campaign_root / str(study.get("spec") or "")).resolve()
    if declared_spec != spec_path.resolve():
        raise ValueError("Study spec path disagrees with the campaign manifest")
    for field in ("baseline_commit", "candidate_commit"):
        if not _SHA40.fullmatch(str(study.get(field) or "")):
            raise ValueError(f"campaign Study {field} is not an exact revision")
    study["_manifest"] = {
        "id": str(manifest.get("id") or "community-skill-upgrade-canary-v1"),
        "path": manifest_path.relative_to(campaign_root).as_posix(),
        "sha256": _sha256(manifest_path),
    }
    study["_study_contract_digest"] = stable_digest(raw_study)
    study["_spec"] = {
        "path": declared_spec.relative_to(campaign_root.parents[2]).as_posix(),
        "sha256": _sha256(declared_spec),
    }
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


def _verify_design_contract(
    *, spec: ComparisonSpecV1, study: Mapping[str, Any], task_count: int
) -> None:
    attempts = study.get("attempts")
    declared_tasks = study.get("tasks", study.get("task_count"))
    declared_cells = study.get("cells", study.get("expected_cells"))
    expected_cells = (
        task_count * 2 * spec.execution.attempts * len(spec.execution.harnesses)
    )
    if attempts != spec.execution.attempts:
        raise ValueError("Study attempts disagree with the campaign manifest")
    if declared_tasks != task_count:
        raise ValueError("Study task count disagrees with the campaign manifest")
    if declared_cells != expected_cells:
        raise ValueError("Study cell count disagrees with the campaign manifest")
    if spec.execution.attempts < 1:
        raise ValueError("community reports require at least one attempt")


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
    tasks_path = _within(repo_root / spec.taskset.tasks, repo_root, "public taskset")
    task_ids = _public_task_ids(tasks_path)
    _verify_design_contract(spec=spec, study=study, task_count=len(task_ids))
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
        or published_candidate.runtime_digest != candidate_revision["runtime_digest"]
    ):
        raise ValueError("result candidate source revision disagrees with its lineage")

    expected_coordinates = {
        (task_id, harness, attempt)
        for task_id in task_ids
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
    task_ids = _public_task_ids(tasks_path)
    _verify_design_contract(spec=spec, study=study, task_count=len(task_ids))
    expected_coordinates = {
        (task_id, harness, attempt)
        for task_id in task_ids
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


def _required_digest(value: Any, label: str) -> str:
    digest = str(value or "")
    if not _SHA256.fullmatch(digest):
        raise ValueError(f"{label} is not a canonical SHA-256 digest")
    return digest


def _study_binding(
    study: Mapping[str, Any], *, spec: ComparisonSpecV1
) -> dict[str, Any]:
    manifest = study.get("_manifest")
    if not isinstance(manifest, Mapping) or set(manifest) != {"id", "path", "sha256"}:
        raise ValueError("Study is missing its exact campaign-manifest binding")
    manifest_sha256 = _required_digest(manifest.get("sha256"), "manifest digest")
    declared = {
        key: value for key, value in study.items() if not str(key).startswith("_")
    }
    contract_digest = _required_digest(
        study.get("_study_contract_digest"), "Study contract digest"
    )
    if contract_digest != stable_digest(declared):
        raise ValueError("Study contract digest does not match the manifest entry")
    spec_binding = study.get("_spec")
    if not isinstance(spec_binding, Mapping) or set(spec_binding) != {"path", "sha256"}:
        raise ValueError("Study is missing its exact spec binding")
    spec_sha256 = _required_digest(spec_binding.get("sha256"), "Study spec digest")
    return {
        "manifest_id": str(manifest.get("id") or ""),
        "manifest_path": str(manifest.get("path") or ""),
        "manifest_sha256": manifest_sha256,
        "study_contract_digest": contract_digest,
        "spec_path": str(spec_binding.get("path") or ""),
        "spec_sha256": spec_sha256,
        "attempts_per_task_arm": spec.execution.attempts,
        "planned_cells": int(study.get("cells", study.get("expected_cells")) or 0),
    }


def _source_result_binding(
    result: SupportedResult,
    *,
    source_result_sha256: str,
    canonical_result_verified: bool,
) -> dict[str, Any]:
    if canonical_result_verified is not True:
        raise ValueError("scientific reports require canonical result verification")
    result_digest = _required_digest(
        getattr(result, "result_digest", ""), "canonical result digest"
    )
    qualification_digest = _required_digest(
        getattr(result, "qualification_digest", ""),
        "canonical qualification digest",
    )
    return {
        "schema_version": result.schema_version,
        "preview_digest": _required_digest(
            getattr(result, "preview_digest", ""), "canonical preview digest"
        ),
        "result_digest": result_digest,
        "qualification_digest": qualification_digest,
        "file_sha256": _required_digest(source_result_sha256, "result file digest"),
        "canonical_reader_verified": True,
    }


def _preregistration_binding(
    *,
    study: Mapping[str, Any],
    campaign_root: Path,
) -> dict[str, Any]:
    declarations = study.get("preregistration")
    if declarations is None:
        return {
            "status": "not_required_for_one_attempt_canary",
            "artifacts": [],
            "binding_digest": None,
        }
    if not isinstance(declarations, list) or not declarations:
        raise ValueError("confirmatory Study preregistration must be nonzero")
    artifacts: list[dict[str, str]] = []
    for declaration in declarations:
        if not isinstance(declaration, Mapping) or set(declaration) != {
            "role",
            "path",
            "sha256",
            "identity_field",
            "identity",
            "applies_to",
        }:
            raise ValueError("preregistration declaration is malformed")
        path = _within(
            campaign_root / str(declaration["path"]),
            campaign_root.parents[2],
            "preregistration",
        )
        expected_sha = _required_digest(declaration["sha256"], "preregistration digest")
        if _sha256(path) != expected_sha:
            raise ValueError("preregistration digest disagrees with the manifest")
        value = _load_object(path, "preregistration")
        identity_field = str(declaration["identity_field"])
        identity = str(declaration["identity"])
        if declaration.get("applies_to") != study.get("id"):
            raise ValueError("preregistration does not apply to this exact Study")
        if value.get(identity_field) != identity:
            raise ValueError("preregistration identity disagrees with the manifest")
        if not str(value.get("status") or "").startswith("frozen"):
            raise ValueError("preregistration was not frozen before execution")
        artifacts.append(
            {
                "role": str(declaration["role"]),
                "identity": identity,
                "applies_to": str(declaration["applies_to"]),
                "path": path.relative_to(campaign_root.parents[2]).as_posix(),
                "sha256": expected_sha,
            }
        )
    return {
        "status": "bound",
        "artifacts": artifacts,
        "binding_digest": stable_digest(artifacts),
    }


def _reject_unblinded_selection(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            tokens = set(re.split(r"[^a-z]+", str(key).lower()))
            if tokens & _AUDIT_BLINDING_TERMS:
                raise ValueError(
                    "reviewer-facing trace-audit selection exposes an arm label"
                )
            _reject_unblinded_selection(item)
    elif isinstance(value, list | tuple):
        for item in value:
            _reject_unblinded_selection(item)
    elif isinstance(value, str):
        tokens = set(re.split(r"[^a-z]+", value.lower()))
        if tokens & {"baseline", "candidate", "treatment", "variant"}:
            raise ValueError(
                "reviewer-facing trace-audit selection exposes an arm label"
            )


def _attempt_ids(result: ComparisonResultV3) -> set[str]:
    return {attempt.attempt_id for _pair, _arm, attempt in _attempts(result)}


def _required_post_result_audit_ids(result: ComparisonResultV3) -> set[str]:
    required: set[str] = set()
    for pair in result.paired_cases:
        assert pair.baseline is not None and pair.candidate is not None
        dimensions = [item.to_dict() for item in pair.dimension_changes]
        critical_failure = any(
            item.get("critical") is True
            and (
                item.get("status") in {"regressed", "mixed"}
                or item.get("candidate") is False
            )
            for item in dimensions
        )
        if pair.status in {"improved", "regressed", "mixed"} or critical_failure:
            required.update((pair.baseline.attempt_id, pair.candidate.attempt_id))
    return required


def _selection_unsigned(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    unsigned.pop("selection_digest", None)
    return unsigned


def _audit_attested_digest(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("reviewer_attestations", None)
    return stable_digest(unsigned)


def _validate_trace_audit(  # noqa: C901 - one strict cross-artifact audit.
    *,
    result: ComparisonResultV3,
    study: Mapping[str, Any],
    selection: Mapping[str, Any],
    review: Mapping[str, Any],
    selection_sha256: str,
    review_sha256: str,
) -> dict[str, Any]:
    policy = study.get("trace_audit")
    if not isinstance(policy, Mapping):
        raise ValueError("confirmatory Study is missing its trace-audit policy")
    expected_selection_fields = {
        "schema_version",
        "kind",
        "preview_digest",
        "selection_frozen_before_execution",
        "sampling_fraction",
        "population_pairs",
        "selected_pairs",
        "selected_attempt_ids",
        "post_result_additions",
        "selection_digest",
    }
    if set(selection) != expected_selection_fields:
        raise ValueError("trace-audit selection schema is malformed")
    _reject_unblinded_selection(selection)
    selection_digest = _required_digest(
        selection.get("selection_digest"), "trace-audit selection digest"
    )
    if selection_digest != stable_digest(_selection_unsigned(selection)):
        raise ValueError("trace-audit selection digest does not match")
    if selection.get("preview_digest") != result.preview_digest:
        raise ValueError("trace-audit selection targets a different preview")
    if selection.get("selection_frozen_before_execution") is not True:
        raise ValueError("trace-audit selection was not frozen before execution")
    if selection.get("population_pairs") != len(result.paired_cases):
        raise ValueError("trace-audit population disagrees with canonical pairs")
    minimum_fraction = float(policy.get("minimum_fraction") or 0)
    fraction = float(selection.get("sampling_fraction") or 0)
    if not 0 < minimum_fraction <= fraction <= 1:
        raise ValueError("trace-audit selection is below the frozen minimum")
    raw_pairs = selection.get("selected_pairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ValueError("trace-audit selection has no blinded pairs")
    selected_ids: list[str] = []
    selected_tasks: set[str] = set()
    selected_coordinates: set[tuple[str, str, int]] = set()
    canonical_pairs = {
        (pair.task_id, pair.harness, pair.attempt): pair for pair in result.paired_cases
    }
    for pair in raw_pairs:
        if not isinstance(pair, Mapping) or set(pair) != {
            "pair_token",
            "task_id",
            "harness",
            "attempt",
            "partition",
            "artifact_a_attempt_id",
            "artifact_b_attempt_id",
        }:
            raise ValueError("trace-audit selected pair is malformed")
        coordinate = (
            str(pair["task_id"]),
            str(pair["harness"]),
            int(pair["attempt"]),
        )
        if coordinate in selected_coordinates or coordinate not in canonical_pairs:
            raise ValueError("trace-audit selected pair identity is not canonical")
        selected_coordinates.add(coordinate)
        selected_tasks.add(coordinate[0])
        artifact_ids = {
            str(pair["artifact_a_attempt_id"]),
            str(pair["artifact_b_attempt_id"]),
        }
        canonical_pair = canonical_pairs[coordinate]
        assert (
            canonical_pair.baseline is not None and canonical_pair.candidate is not None
        )
        if artifact_ids != {
            canonical_pair.baseline.attempt_id,
            canonical_pair.candidate.attempt_id,
        }:
            raise ValueError("trace-audit blinded artifacts do not match their pair")
        expected_pair_token = hashlib.sha256(
            (
                f"{result.preview_digest}:pair:{coordinate[0]}:"
                f"{coordinate[1]}:{coordinate[2]}"
            ).encode()
        ).hexdigest()
        if pair["pair_token"] != expected_pair_token:
            raise ValueError("trace-audit pair token does not match")
        selected_ids.extend(sorted(artifact_ids))
    if len(raw_pairs) < math.ceil(len(result.paired_cases) * fraction):
        raise ValueError(
            "trace-audit selected pair count is below its declared fraction"
        )
    declared_ids = selection.get("selected_attempt_ids")
    if declared_ids != sorted(selected_ids) or len(selected_ids) != len(
        set(selected_ids)
    ):
        raise ValueError("trace-audit selected attempt identities disagree")
    all_attempt_ids = _attempt_ids(result)
    if not set(selected_ids) <= all_attempt_ids:
        raise ValueError("trace-audit selection contains an unknown attempt")
    families = policy.get("required_behavior_families") or {}
    if not isinstance(families, Mapping):
        raise ValueError("trace-audit behavior-family policy is malformed")
    for family, task_ids in families.items():
        if not isinstance(task_ids, list) or not selected_tasks.intersection(task_ids):
            raise ValueError(f"trace-audit selection misses behavior family {family}")

    expected_review_fields = {
        "schema_version",
        "kind",
        "campaign_id",
        "study_id",
        "status",
        "preview_digest",
        "result_digest",
        "selection_digest",
        "selected_attempt_ids",
        "required_attempt_ids",
        "reviewed_attempts",
        "reviewer_attestations",
        "limitations",
    }
    if set(review) != expected_review_fields:
        raise ValueError("completed trace-audit schema is malformed")
    if (
        review.get("schema_version") != 1
        or review.get("kind") != "completed_blinded_trace_audit"
        or review.get("status") != "completed"
        or review.get("study_id") != result.comparison_id
        or review.get("preview_digest") != result.preview_digest
        or review.get("result_digest") != result.result_digest
        or review.get("selection_digest") != selection_digest
        or review.get("selected_attempt_ids") != sorted(selected_ids)
    ):
        raise ValueError("completed trace audit targets different immutable inputs")
    manifest = study.get("_manifest")
    if not isinstance(manifest, Mapping) or review.get("campaign_id") != manifest.get(
        "id"
    ):
        raise ValueError("completed trace audit targets a different campaign")
    required_ids = set(selected_ids) | _required_post_result_audit_ids(result)
    if review.get("required_attempt_ids") != sorted(required_ids):
        raise ValueError("completed trace audit omits required post-result artifacts")
    reviewed = review.get("reviewed_attempts")
    reviewed_ids = [
        str(item.get("attempt_id") or "")
        for item in reviewed or ()
        if isinstance(item, Mapping)
    ]
    if (
        not isinstance(reviewed, list)
        or len(reviewed_ids) != len(required_ids)
        or set(reviewed_ids) != required_ids
    ):
        raise ValueError("completed trace audit does not cover every required artifact")
    attestations = review.get("reviewer_attestations")
    if not isinstance(attestations, list) or len(attestations) != 2:
        raise ValueError("completed trace audit requires two reviewer attestations")
    reviewers = [
        str(item.get("reviewer") or "")
        for item in attestations
        if isinstance(item, Mapping)
    ]
    if (
        len(reviewers) != 2
        or len(set(reviewers)) != 2
        or any(not item for item in reviewers)
    ):
        raise ValueError("completed trace audit requires two distinct reviewers")
    attested_digest = _audit_attested_digest(review)
    for attestation in attestations:
        if not isinstance(attestation, Mapping) or set(attestation) != {
            "reviewer",
            "reviewed_at",
            "artifact_digest",
        }:
            raise ValueError("trace-audit reviewer attestation is malformed")
        if (
            not str(attestation.get("reviewed_at") or "")
            or attestation.get("artifact_digest") != attested_digest
        ):
            raise ValueError("trace-audit reviewer attestation is unsigned or stale")
    for artifact in reviewed:
        if not isinstance(artifact, Mapping) or set(artifact) != {
            "attempt_id",
            "reviews",
            "adjudication",
        }:
            raise ValueError("trace-audit artifact review is malformed")
        reviews = artifact.get("reviews")
        if not isinstance(reviews, list) or len(reviews) != 2:
            raise ValueError("every trace-audit artifact requires two reviews")
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
            if not str(item.get("reason") or "").strip():
                raise ValueError("trace-audit review reason is empty")
            artifact_reviewers.add(str(item["reviewer"]))
            disposition = str(item["disposition"])
            if disposition not in {"verified", "unclear", "finding"}:
                raise ValueError("trace-audit review disposition is invalid")
            dispositions.add(disposition)
            checks = item.get("checks")
            if (
                not isinstance(checks, Mapping)
                or set(checks) != _AUDIT_REQUIRED_CHECKS
                or any(value is not True for value in checks.values())
            ):
                raise ValueError(
                    "trace-audit review did not verify every required check"
                )
        if artifact_reviewers != set(reviewers):
            raise ValueError(
                "trace-audit artifact reviewers disagree with attestations"
            )
        adjudication = artifact.get("adjudication")
        if len(dispositions) > 1:
            if (
                not isinstance(adjudication, Mapping)
                or set(adjudication) != {"reviewer", "disposition", "reason"}
                or not str(adjudication.get("reviewer") or "")
                or not str(adjudication.get("disposition") or "")
                or not str(adjudication.get("reason") or "").strip()
            ):
                raise ValueError("discordant trace-audit reviews require adjudication")
            effective_disposition = str(adjudication["disposition"])
        elif adjudication is not None:
            raise ValueError("concordant trace-audit reviews cannot carry adjudication")
        else:
            effective_disposition = next(iter(dispositions))
        if effective_disposition != "verified":
            raise ValueError("trace-audit findings require canonical result correction")
    return {
        "status": "completed",
        "selection_digest": selection_digest,
        "selection_file_sha256": _required_digest(
            selection_sha256, "trace-audit selection file digest"
        ),
        "completed_audit_digest": stable_digest(review),
        "completed_audit_file_sha256": _required_digest(
            review_sha256, "completed trace-audit file digest"
        ),
        "selected_attempts": len(selected_ids),
        "required_attempts": len(required_ids),
        "reviewers": reviewers,
        "policy_digest": stable_digest(policy),
    }


def _attempts(result: SupportedResult) -> list[tuple[Any, str, Any]]:
    selected: list[tuple[Any, str, Any]] = []
    for pair in result.paired_cases:
        for arm, attempt in (
            ("baseline", pair.baseline),
            ("candidate", pair.candidate),
        ):
            if attempt is None:
                raise ValueError("scientific reports require complete paired attempts")
            if attempt.execution_status not in TERMINAL_STATES:
                raise ValueError("scientific reports require terminal attempts")
            if attempt.evidence_status != "reconciled":
                raise ValueError(
                    "scientific reports require reconciled attempt evidence"
                )
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
            {key: pair.baseline.score_explanations[key] for key in baseline_scores}
            if isinstance(result, ComparisonResultV3)
            else {}
        )
        candidate_explanations = (
            {key: pair.candidate.score_explanations[key] for key in candidate_scores}
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
                    "host_verifier_receipts": dict(
                        getattr(pair.baseline, "infrastructure", {}).get(
                            "host_verifier_receipts", {}
                        )
                    ),
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
                    "host_verifier_receipts": dict(
                        getattr(pair.candidate, "infrastructure", {}).get(
                            "host_verifier_receipts", {}
                        )
                    ),
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
    skills = {
        "baseline": list(spec.baseline.skills),
        "candidate": list(spec.candidate.skills),
    }
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


def _limitations(
    template: Mapping[str, Any], result: SupportedResult, spec: ComparisonSpecV1
) -> list[str]:
    locked = template.get("limitations")
    if not isinstance(locked, list) or tuple(locked) != COMMON_LIMITATIONS:
        raise ValueError("scientific report template limitations drifted")
    extras = (
        *(CANARY_LIMITATION if spec.execution.attempts == 1 else ()),
        *(V2_LIMITATIONS if isinstance(result, ComparisonResultV2) else ()),
        *result.limitations,
        *result.behavioral_summary.limitations,
        *result.decision.limitations,
    )
    return list(dict.fromkeys((*COMMON_LIMITATIONS, *extras)))


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
    if value["schema_version"] != 2 or value["status"] != "pending_execution":
        raise ValueError("scientific report template must remain pending")
    for field in (
        "study_id",
        "evidence_project",
        "source_result",
        "study_contract",
        "preregistration",
        "trace_audit",
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
        "mechanism_results",
        "evidence_links",
    ):
        if value[field] != []:
            raise ValueError(f"scientific report template preclaims {field}")
    if value["efficiency"] != {"latency": None, "tokens": None, "cost": None}:
        raise ValueError("scientific report template preclaims efficiency")
    if tuple(value["limitations"]) != COMMON_LIMITATIONS:
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
    if value["schema_version"] != 2 or value["status"] != "completed":
        raise ValueError("scientific report must be a completed schema-v2 artifact")
    if not str(value["study_id"] or "") or not str(value["evidence_project"] or ""):
        raise ValueError("scientific report identity is incomplete")
    revisions = value["exact_revisions"]
    if not isinstance(revisions, Mapping) or set(revisions) != {
        "baseline",
        "candidate",
    }:
        raise ValueError("scientific report revisions are malformed")
    if any(not _SHA40.fullmatch(str(item)) for item in revisions.values()):
        raise ValueError("scientific report revisions must be exact Git SHAs")
    source_result = value["source_result"]
    if (
        not isinstance(source_result, Mapping)
        or source_result.get("canonical_reader_verified") is not True
        or any(
            not _SHA256.fullmatch(str(source_result.get(field) or ""))
            for field in (
                "preview_digest",
                "result_digest",
                "qualification_digest",
                "file_sha256",
            )
        )
    ):
        raise ValueError("scientific report source result is not canonically bound")
    study_contract = value["study_contract"]
    if not isinstance(study_contract, Mapping) or any(
        not _SHA256.fullmatch(str(study_contract.get(field) or ""))
        for field in (
            "manifest_sha256",
            "study_contract_digest",
            "spec_sha256",
        )
    ):
        raise ValueError("scientific report Study contract is not exact")
    preregistration = value["preregistration"]
    if not isinstance(preregistration, Mapping) or preregistration.get(
        "status"
    ) not in {
        "bound",
        "not_required_for_one_attempt_canary",
    }:
        raise ValueError("scientific report preregistration binding is malformed")
    trace_audit = value["trace_audit"]
    if not isinstance(trace_audit, Mapping) or trace_audit.get("status") not in {
        "completed",
        "not_required_for_one_attempt_canary",
    }:
        raise ValueError("scientific report trace-audit binding is malformed")
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
    if (
        not isinstance(finding, Mapping)
        or finding.get("status") not in BEHAVIORAL_STATUSES
    ):
        raise ValueError("scientific report behavioral finding is invalid")
    for field in (
        "deterministic_results",
        "judge_results",
        "mechanism_results",
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
        item.get("status") != "resolved" or not item.get("ref") or not item.get("url")
        for item in links
    ):
        raise ValueError("scientific report evidence links are unresolved")
    limitations = value["limitations"]
    if not isinstance(limitations, list) or not all(
        item in limitations for item in COMMON_LIMITATIONS
    ):
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
    campaign_root: Path = ROOT,
    source_result_sha256: str,
    canonical_result_verified: bool,
    audit_selection: Mapping[str, Any] | None = None,
    audit_review: Mapping[str, Any] | None = None,
    audit_selection_sha256: str | None = None,
    audit_review_sha256: str | None = None,
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
    source_result = _source_result_binding(
        result,
        source_result_sha256=source_result_sha256,
        canonical_result_verified=canonical_result_verified,
    )
    study_contract = _study_binding(study, spec=spec)
    preregistration = _preregistration_binding(
        study=study,
        campaign_root=campaign_root,
    )
    if spec.execution.attempts == 1:
        if any(
            item is not None
            for item in (
                audit_selection,
                audit_review,
                audit_selection_sha256,
                audit_review_sha256,
            )
        ):
            raise ValueError("one-attempt canaries cannot claim a confirmatory audit")
        trace_audit = {
            "status": "not_required_for_one_attempt_canary",
            "selection_digest": None,
            "completed_audit_digest": None,
        }
    else:
        if not isinstance(result, ComparisonResultV3):
            raise ValueError("repeated confirmatory reports require ComparisonResultV3")
        if (
            audit_selection is None
            or audit_review is None
            or audit_selection_sha256 is None
            or audit_review_sha256 is None
        ):
            raise ValueError(
                "repeated confirmatory reports require a completed frozen trace audit"
            )
        trace_audit = _validate_trace_audit(
            result=result,
            study=study,
            selection=audit_selection,
            review=audit_review,
            selection_sha256=audit_selection_sha256,
            review_sha256=audit_review_sha256,
        )
    judges = [item for item in spec.evaluators if item.type == "llm_judge"]
    behavior = result.behavioral_summary
    supported = behavior.supported_claim or "No positive behavioral claim is supported."
    report = {
        "schema_version": 2,
        "status": "completed",
        "study_id": result.comparison_id,
        "evidence_project": result.evidence_project,
        "exact_revisions": revisions,
        "source_result": source_result,
        "study_contract": study_contract,
        "preregistration": preregistration,
        "trace_audit": trace_audit,
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
        "mechanism_results": _skill_evidence(result, spec),
        "efficiency": _efficiency(result),
        "evidence_links": _evidence_links(result),
        "limitations": _limitations(template, result, spec),
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
    audit_selection_path: Path | None = None,
    audit_review_path: Path | None = None,
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
    audit_selection = (
        _load_object(
            _within(audit_selection_path, repo_root, "trace-audit selection"),
            "trace-audit selection",
        )
        if audit_selection_path is not None
        else None
    )
    audit_review = (
        _load_object(
            _within(audit_review_path, repo_root, "completed trace audit"),
            "completed trace audit",
        )
        if audit_review_path is not None
        else None
    )
    return generate_report(
        result=result,
        spec=spec,
        study=study,
        template=template,
        repo_root=repo_root,
        campaign_root=campaign_root,
        source_result_sha256=_sha256(resolved_result),
        canonical_result_verified=True,
        audit_selection=audit_selection,
        audit_review=audit_review,
        audit_selection_sha256=(
            _sha256(audit_selection_path.resolve())
            if audit_selection_path is not None
            else None
        ),
        audit_review_sha256=(
            _sha256(audit_review_path.resolve())
            if audit_review_path is not None
            else None
        ),
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
    parser.add_argument("--audit-selection", type=Path)
    parser.add_argument("--audit-review", type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    report = build_report(
        result_path=args.result,
        spec_path=args.spec,
        repo_root=repo_root,
        campaign_root=repo_root / "examples/comparisons/community-skill-upgrades",
        template_path=(
            repo_root / "examples/comparisons/community-skill-upgrades/"
            "scientific-report-template-v2.json"
        ),
        audit_selection_path=args.audit_selection,
        audit_review_path=args.audit_review,
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
