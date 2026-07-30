from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from fugue.bench.candidates import stable_digest
from fugue.bench.comparison import (
    ComparisonResultV3,
    PairedAttemptV3,
    PairedCaseV3,
    read_comparison_result,
)
from fugue.bench.files import atomic_write_json

FAILURE_LOCK_SCHEMA_VERSION = 1
FAILURE_LOCK_KIND = "comparison-repeated-failure-lock"
FailureArm = Literal["baseline", "candidate"]

_DIGEST = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_CALL_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_EVIDENCE_KINDS = {
    "evaluation_root",
    "prediction_and_score",
    "prediction",
    "agent_root",
    "dataset",
}
_CALL_EVIDENCE_KINDS = {
    "evaluation_root",
    "prediction_and_score",
    "prediction",
    "agent_root",
}
_PRIVATE_KEYS = {
    "expected",
    "expected_answer",
    "expected_values",
    "gold",
    "gold_output",
    "private_evaluation",
    "private_label",
    "private_labels",
}
_SYNTHETIC_MARKERS = {"fixture", "manual", "placeholder", "synthetic"}


def build_comparison_failure_lock(
    *,
    result_path: Path,
    task_id: str,
    arm: FailureArm,
    primary_attempt_id: str,
    expected_comparison_id: str,
    expected_source_project: str,
    expected_result_project: str,
    expected_harness: str,
    expected_tasks: int,
    expected_attempts: int,
    spec_digest: str,
    required_source_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Freeze a repeated real failure from one canonical V3 comparison.

    Only public identities, failed dimension IDs, immutable locks, and resolved
    evidence references are copied. Agent output and private truth are not.
    """

    result = read_comparison_result(result_path)
    if not isinstance(result, ComparisonResultV3):
        raise ValueError("failure locking requires ComparisonResultV3")
    _validate_result_boundary(
        result,
        expected_comparison_id=expected_comparison_id,
        expected_source_project=expected_source_project,
        expected_result_project=expected_result_project,
        expected_harness=expected_harness,
        expected_tasks=expected_tasks,
        expected_attempts=expected_attempts,
        required_source_ids=required_source_ids,
    )
    selected_pairs = sorted(
        (
            pair
            for pair in result.paired_cases
            if pair.task_id == task_id and pair.harness == expected_harness
        ),
        key=lambda pair: pair.attempt,
    )
    if len(selected_pairs) != expected_attempts or {
        pair.attempt for pair in selected_pairs
    } != set(range(1, expected_attempts + 1)):
        raise ValueError("selected task is not repeated across every locked attempt")
    validity = next(
        (item for item in result.task_validity if item.task_id == task_id),
        None,
    )
    if validity is None or validity.status != "valid":
        raise ValueError("selected failure task is not valid discriminating evidence")

    locked_attempts: list[dict[str, Any]] = []
    repeated_failures: list[set[str]] = []
    for pair in selected_pairs:
        attempt = pair.baseline if arm == "baseline" else pair.candidate
        locked = _locked_attempt(
            pair,
            attempt,
            arm=arm,
            source_project=expected_source_project,
            result_project=expected_result_project,
        )
        locked_attempts.append(locked)
        repeated_failures.append(set(locked["failed_critical_dimensions"]))
    common_failures = sorted(set.intersection(*repeated_failures))
    if not common_failures:
        raise ValueError(
            "selected arm does not repeat one critical outcome or safety failure"
        )
    primary = next(
        (
            attempt
            for attempt in locked_attempts
            if attempt["attempt_id"] == primary_attempt_id
        ),
        None,
    )
    if primary is None:
        raise ValueError("primary attempt is not one of the repeated failures")

    resolved_path = result_path.resolve()
    topology = result.evidence_topology
    unsigned = {
        "schema_version": FAILURE_LOCK_SCHEMA_VERSION,
        "kind": FAILURE_LOCK_KIND,
        "source": {
            "comparison_id": result.comparison_id,
            "result_digest": result.result_digest,
            "qualification_digest": result.qualification_digest,
            "result_artifact_sha256": hashlib.sha256(
                resolved_path.read_bytes()
            ).hexdigest(),
            "preview_digest": result.preview_digest,
            "result_source": result.source,
            "spec_digest": _digest(spec_digest, "comparison spec digest"),
            "source_project": expected_source_project,
            "result_project": expected_result_project,
            "source_lock_digest": topology.source_lock_digest,
            "evidence_topology_digest": topology.topology_digest,
            "aligned_analysis_digest": result.aligned_analysis.analysis_digest,
        },
        "failure": {
            "task_id": task_id,
            "harness": expected_harness,
            "arm": arm,
            "primary_attempt_id": primary_attempt_id,
            "repeated_attempt_ids": [
                value["attempt_id"] for value in locked_attempts
            ],
            "failed_critical_dimensions": common_failures,
        },
        "primary_attempt": primary,
        "repeated_attempts": locked_attempts,
        "locks": {
            "candidate_sources": [
                item.to_dict() for item in result.candidate_source_revisions
            ],
            "runtime_locks": [item.to_dict() for item in result.runtime_locks],
            "scorer_revisions": [
                item.to_dict() for item in result.scorer_revisions
            ],
        },
        "review_status": "reviewed",
        "lock_sha256": "",
    }
    _reject_private_fields(unsigned)
    return {**unsigned, "lock_sha256": _lock_digest(unsigned)}


def write_comparison_failure_lock(
    path: Path,
    value: Mapping[str, Any],
) -> Path:
    validated = validate_comparison_failure_lock(value)
    destination = path.resolve()
    atomic_write_json(destination, validated, mode=0o600)
    return destination


def read_comparison_failure_lock(
    path: Path,
    *,
    expected_comparison_id: str | None = None,
    expected_source_project: str | None = None,
    expected_result_project: str | None = None,
) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("failure lock must be an object")
    value = validate_comparison_failure_lock(raw)
    source = value["source"]
    for expected, field in (
        (expected_comparison_id, "comparison_id"),
        (expected_source_project, "source_project"),
        (expected_result_project, "result_project"),
    ):
        if expected is not None and source[field] != expected:
            raise ValueError(f"failure lock {field} does not match")
    return value


def validate_comparison_failure_lock(value: Mapping[str, Any]) -> dict[str, Any]:
    _strict_keys(
        value,
        {
            "schema_version",
            "kind",
            "source",
            "failure",
            "primary_attempt",
            "repeated_attempts",
            "locks",
            "review_status",
            "lock_sha256",
        },
        "failure lock",
    )
    if (
        value.get("schema_version") != FAILURE_LOCK_SCHEMA_VERSION
        or value.get("kind") != FAILURE_LOCK_KIND
        or value.get("review_status") != "reviewed"
    ):
        raise ValueError("failure lock is not a reviewed repeated-failure lock")
    supplied_digest = _digest(value.get("lock_sha256"), "failure lock digest")
    if supplied_digest != _lock_digest({**value, "lock_sha256": ""}):
        raise ValueError("failure lock digest does not match")

    source = _mapping(value.get("source"), "failure source")
    _strict_keys(
        source,
        {
            "comparison_id",
            "result_digest",
            "qualification_digest",
            "result_artifact_sha256",
            "preview_digest",
            "result_source",
            "spec_digest",
            "source_project",
            "result_project",
            "source_lock_digest",
            "evidence_topology_digest",
            "aligned_analysis_digest",
        },
        "failure source",
    )
    for field in (
        "comparison_id",
        "result_source",
        "source_project",
        "result_project",
    ):
        _text(source.get(field), field)
    for field in (
        "result_digest",
        "qualification_digest",
        "result_artifact_sha256",
        "preview_digest",
        "spec_digest",
        "source_lock_digest",
        "evidence_topology_digest",
        "aligned_analysis_digest",
    ):
        _digest(source.get(field), field)
    _project(str(source["source_project"]), "source project")
    _project(str(source["result_project"]), "result project")
    if source["source_project"] == source["result_project"]:
        raise ValueError("failure lock source and result projects must be isolated")

    failure = _mapping(value.get("failure"), "failure identity")
    _strict_keys(
        failure,
        {
            "task_id",
            "harness",
            "arm",
            "primary_attempt_id",
            "repeated_attempt_ids",
            "failed_critical_dimensions",
        },
        "failure identity",
    )
    _text(failure.get("task_id"), "failure task")
    _text(failure.get("harness"), "failure harness")
    arm = failure.get("arm")
    if arm not in {"baseline", "candidate"}:
        raise ValueError("failure arm must be baseline or candidate")
    primary_attempt_id = _digest(
        failure.get("primary_attempt_id"), "primary attempt id"
    )
    repeated_ids = _digest_sequence(
        failure.get("repeated_attempt_ids"),
        "repeated attempt ids",
        minimum=2,
    )
    if len(set(repeated_ids)) != len(repeated_ids):
        raise ValueError("repeated attempt IDs must be unique")
    if primary_attempt_id not in repeated_ids:
        raise ValueError("primary attempt must be one of the repeated attempts")
    common_failures = _text_sequence(
        failure.get("failed_critical_dimensions"),
        "failed critical dimensions",
        minimum=1,
    )

    attempts = _attempt_sequence(
        value.get("repeated_attempts"),
        arm=str(arm),
        source_project=str(source["source_project"]),
        result_project=str(source["result_project"]),
    )
    if [item["attempt_id"] for item in attempts] != repeated_ids:
        raise ValueError("repeated attempt order or identity changed")
    if any(
        not set(common_failures) <= set(item["failed_critical_dimensions"])
        for item in attempts
    ):
        raise ValueError("repeated attempts do not share the locked failure")
    for attempt in attempts:
        identity = {
            "task_id": str(failure["task_id"]),
            "arm": str(arm),
            "harness": str(failure["harness"]),
            "attempt": int(attempt["attempt_number"]),
            "candidate": str(attempt["candidate_identity"]),
            "runtime": str(attempt["runtime_identity"]),
        }
        if attempt["attempt_id"] != stable_digest(
            {"schema_version": 1, **identity}
        ):
            raise ValueError("locked repeated attempt identity is not canonical")
    primary = _validate_locked_attempt(
        value.get("primary_attempt"),
        arm=str(arm),
        source_project=str(source["source_project"]),
        result_project=str(source["result_project"]),
    )
    matching = next(
        (item for item in attempts if item["attempt_id"] == primary_attempt_id),
        None,
    )
    if primary != matching:
        raise ValueError("primary attempt differs from its repeated-attempt record")
    _validate_lock_descriptors(value.get("locks"))
    _reject_private_fields(value)
    return json.loads(json.dumps(value, sort_keys=True))


def _validate_result_boundary(
    result: ComparisonResultV3,
    *,
    expected_comparison_id: str,
    expected_source_project: str,
    expected_result_project: str,
    expected_harness: str,
    expected_tasks: int,
    expected_attempts: int,
    required_source_ids: Sequence[str],
) -> None:
    if result.comparison_id != expected_comparison_id:
        raise ValueError("comparison result is not the required V3 canary")
    topology = result.evidence_topology
    if (
        topology.source_destination.project_slug != expected_source_project
        or topology.result_destination.project_slug != expected_result_project
        or result.evidence_project != expected_result_project
    ):
        raise ValueError("comparison result uses the wrong source/result topology")
    if (
        topology.pre_run_drift.status != "matched"
        or topology.post_run_drift.status != "matched"
    ):
        raise ValueError("comparison source evidence drifted")
    if any(marker in result.source.lower() for marker in _SYNTHETIC_MARKERS):
        raise ValueError("comparison result source is synthetic or manually seeded")
    integrity = result.integrity
    expected_rows = expected_tasks * expected_attempts * 2
    if (
        integrity.get("status") != "reconciled"
        or result.rows != expected_rows
        or int(integrity.get("row_count") or 0) != expected_rows
        or int(integrity.get("unique_attempts") or 0) != expected_rows
        or integrity.get("duplicate_attempt_ids")
        or int(integrity.get("unresolved_evidence_attempts") or 0) != 0
        or int(integrity.get("invalid_evidence_attempts") or 0) != 0
        or int(integrity.get("cross_project_attempts") or 0) != 0
    ):
        raise ValueError("comparison result integrity is not fully reconciled")
    expected_pairs = expected_tasks * expected_attempts
    if len(result.paired_cases) != expected_pairs:
        raise ValueError("comparison result has the wrong aligned pair count")
    if {pair.harness for pair in result.paired_cases} != {expected_harness}:
        raise ValueError("comparison result uses a different harness")
    task_ids = {pair.task_id for pair in result.paired_cases}
    if len(task_ids) != expected_tasks:
        raise ValueError("comparison result has the wrong task count")
    for task in task_ids:
        attempts = {
            pair.attempt for pair in result.paired_cases if pair.task_id == task
        }
        if attempts != set(range(1, expected_attempts + 1)):
            raise ValueError("comparison attempts are not aligned by task")
    available_source_ids = {
        item.id for item in result.candidate_source_revisions
    }
    if not set(required_source_ids) <= available_source_ids:
        raise ValueError("comparison result lacks a required candidate source lock")
    if not result.runtime_locks or not result.scorer_revisions:
        raise ValueError("comparison result lacks runtime or scorer locks")


def _locked_attempt(
    pair: PairedCaseV3,
    attempt: PairedAttemptV3 | None,
    *,
    arm: FailureArm,
    source_project: str,
    result_project: str,
) -> dict[str, Any]:
    if attempt is None:
        raise ValueError(f"{arm} attempt is missing")
    if attempt.passed is not False:
        raise ValueError(f"{arm} attempt did not reproduce a deterministic failure")
    critical_failures = sorted(
        change.id
        for change in pair.dimension_changes
        if change.critical
        and change.role in {"outcome", "safety_gate"}
        and (
            (change.baseline is False and arm == "baseline")
            or (change.candidate is False and arm == "candidate")
        )
    )
    if not critical_failures:
        raise ValueError(f"{arm} attempt has no critical outcome or safety failure")
    identity = dict(attempt.identity)
    _strict_keys(
        identity,
        {"task_id", "arm", "harness", "attempt", "candidate", "runtime"},
        f"{arm} attempt identity",
    )
    if (
        identity.get("task_id") != pair.task_id
        or identity.get("arm") != arm
        or identity.get("harness") != pair.harness
        or identity.get("attempt") != pair.attempt
    ):
        raise ValueError(f"{arm} attempt identity does not match its aligned pair")
    expected_attempt_id = stable_digest({"schema_version": 1, **identity})
    if attempt.attempt_id != expected_attempt_id:
        raise ValueError(f"{arm} attempt identity is not canonical")
    candidate_identity = _identity(
        identity.get("candidate"), f"{arm} candidate identity"
    )
    runtime_identity = _identity(
        identity.get("runtime"), f"{arm} runtime identity"
    )
    execution_identity = _identity(
        attempt.execution_fingerprint, f"{arm} execution fingerprint"
    )
    runtime_lock_digest = (
        _identity(attempt.runtime_lock_digest, f"{arm} runtime lock")
        if attempt.runtime_lock_digest
        else None
    )
    if runtime_identity not in {execution_identity, runtime_lock_digest}:
        raise ValueError(f"{arm} runtime identity disagrees with execution evidence")
    if not attempt.prediction_id:
        raise ValueError(f"{arm} prediction identity is missing")
    if attempt.evidence_status != "reconciled":
        raise ValueError(f"{arm} evidence is not reconciled")
    if attempt.actual_query_scope != (source_project,):
        raise ValueError(f"{arm} queried outside the locked source project")
    links = _resolved_links(
        attempt,
        result_project=result_project,
    )
    return {
        "attempt_id": attempt.attempt_id,
        "attempt_number": pair.attempt,
        "candidate_identity": candidate_identity,
        "runtime_identity": runtime_identity,
        "execution_fingerprint": execution_identity,
        "runtime_lock_digest": runtime_lock_digest,
        "prediction_id": str(attempt.prediction_id),
        "actual_query_scope": list(attempt.actual_query_scope),
        "reported_project_identity": attempt.reported_project_identity,
        "failed_critical_dimensions": critical_failures,
        "evidence_links": links,
    }


def _attempt_sequence(
    raw: Any,
    *,
    arm: str,
    source_project: str,
    result_project: str,
) -> list[dict[str, Any]]:
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise ValueError("repeated attempts must be a list")
    values = [
        _validate_locked_attempt(
            item,
            arm=arm,
            source_project=source_project,
            result_project=result_project,
        )
        for item in raw
    ]
    if len(values) < 2:
        raise ValueError("a repeated failure requires at least two attempts")
    numbers = [int(item["attempt_number"]) for item in values]
    if numbers != list(range(1, len(values) + 1)):
        raise ValueError("repeated failure attempts must be ordered and contiguous")
    return values


def _validate_locked_attempt(
    raw: Any,
    *,
    arm: str,
    source_project: str,
    result_project: str,
) -> dict[str, Any]:
    value = _mapping(raw, f"{arm} locked attempt")
    _strict_keys(
        value,
        {
            "attempt_id",
            "attempt_number",
            "candidate_identity",
            "runtime_identity",
            "execution_fingerprint",
            "runtime_lock_digest",
            "prediction_id",
            "actual_query_scope",
            "reported_project_identity",
            "failed_critical_dimensions",
            "evidence_links",
        },
        f"{arm} locked attempt",
    )
    normalized = {
        "attempt_id": _digest(value.get("attempt_id"), f"{arm} attempt id"),
        "attempt_number": _positive_int(
            value.get("attempt_number"), f"{arm} attempt number"
        ),
        "candidate_identity": _identity(
            value.get("candidate_identity"), f"{arm} candidate identity"
        ),
        "runtime_identity": _identity(
            value.get("runtime_identity"), f"{arm} runtime identity"
        ),
        "execution_fingerprint": _identity(
            value.get("execution_fingerprint"), f"{arm} execution fingerprint"
        ),
        "runtime_lock_digest": (
            _identity(value.get("runtime_lock_digest"), f"{arm} runtime lock")
            if value.get("runtime_lock_digest")
            else None
        ),
        "prediction_id": _text(
            value.get("prediction_id"), f"{arm} prediction identity"
        ),
        "actual_query_scope": _text_sequence(
            value.get("actual_query_scope"), f"{arm} actual query scope", minimum=1
        ),
        "reported_project_identity": (
            _text(value.get("reported_project_identity"), "reported project")
            if value.get("reported_project_identity")
            else None
        ),
        "failed_critical_dimensions": _text_sequence(
            value.get("failed_critical_dimensions"),
            f"{arm} failed dimensions",
            minimum=1,
        ),
        "evidence_links": _locked_links(
            value.get("evidence_links"),
            result_project=result_project,
        ),
    }
    if normalized["actual_query_scope"] != [source_project]:
        raise ValueError(f"{arm} actual query scope changed")
    if normalized["runtime_identity"] not in {
        normalized["execution_fingerprint"],
        normalized["runtime_lock_digest"],
    }:
        raise ValueError(f"{arm} runtime identity changed")
    return normalized


def _resolved_links(
    attempt: PairedAttemptV3,
    *,
    result_project: str,
) -> list[dict[str, str]]:
    links = [item.to_dict() for item in attempt.evidence_links]
    return _locked_links(links, result_project=result_project)


def _locked_links(raw: Any, *, result_project: str) -> list[dict[str, str]]:
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise ValueError("attempt evidence links must be a list")
    result: list[dict[str, str]] = []
    for item in raw:
        value = _mapping(item, "attempt evidence link")
        _reject_unknown_keys(
            value,
            {"kind", "status", "system", "ref", "url", "reason"},
            "attempt evidence link",
        )
        for required in ("kind", "status", "system", "ref", "url"):
            if required not in value:
                raise ValueError(
                    f"attempt evidence link is missing field: {required}"
                )
        kind = _text(value.get("kind"), "evidence kind")
        if kind not in _EVIDENCE_KINDS:
            raise ValueError(f"unknown evidence link kind: {kind}")
        if value.get("status") != "resolved" or value.get("system") != "weave":
            raise ValueError(f"{kind} is not resolved Weave evidence")
        ref = _text(value.get("ref"), f"{kind} evidence ref")
        url = _wandb_url(value.get("url"), f"{kind} evidence URL")
        if f"/{result_project}/" not in url:
            raise ValueError(f"{kind} evidence URL uses a different project")
        if kind in _CALL_EVIDENCE_KINDS:
            if not _CALL_ID.fullmatch(ref):
                raise ValueError(f"{kind} does not use a canonical Weave Call ID")
            if f"/weave/calls/{ref}" not in url:
                raise ValueError(f"{kind} does not use the canonical Call route")
        result.append(
            {
                "kind": kind,
                "status": "resolved",
                "system": "weave",
                "ref": ref,
                "url": url,
            }
        )
    if len(result) != 5 or {item["kind"] for item in result} != _EVIDENCE_KINDS:
        raise ValueError("attempt requires exactly five resolved Weave links")
    return sorted(result, key=lambda item: item["kind"])


def _validate_lock_descriptors(raw: Any) -> None:
    value = _mapping(raw, "failure source locks")
    _strict_keys(
        value,
        {"candidate_sources", "runtime_locks", "scorer_revisions"},
        "failure source locks",
    )
    for key in ("runtime_locks", "scorer_revisions"):
        items = value.get(key)
        if not isinstance(items, Sequence) or isinstance(items, str | bytes):
            raise ValueError(f"{key} must be a list")
        if not items:
            raise ValueError(f"{key} must be nonempty")
        for item in items:
            descriptor = _mapping(item, key)
            _reject_unknown_keys(
                descriptor,
                {"id", "label", "digest", "details"},
                key,
            )
            if not {"id", "label", "digest"} <= set(descriptor):
                raise ValueError(f"{key} descriptor is incomplete")
            _text(descriptor.get("id"), f"{key} id")
            _text(descriptor.get("label"), f"{key} label")
            _digest(descriptor.get("digest"), f"{key} digest")
    sources = value.get("candidate_sources")
    if not isinstance(sources, Sequence) or isinstance(sources, str | bytes):
        raise ValueError("candidate_sources must be a list")
    for item in sources:
        source = _mapping(item, "candidate source")
        _reject_unknown_keys(
            source,
            {"kind", "id", "version_identity", "runtime_digest", "lock_digest"},
            "candidate source",
        )
        if not {
            "kind",
            "id",
            "version_identity",
            "runtime_digest",
        } <= set(source):
            raise ValueError("candidate source descriptor is incomplete")
        for key in ("kind", "id", "version_identity"):
            _text(source.get(key), f"candidate source {key}")
        _identity(source.get("runtime_digest"), "candidate runtime digest")
        if source.get("lock_digest"):
            _identity(source.get("lock_digest"), "candidate source lock")


def _lock_digest(value: Mapping[str, Any]) -> str:
    return stable_digest({**value, "lock_sha256": ""})


def _strict_keys(
    value: Mapping[str, Any],
    allowed: set[str],
    name: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(allowed - set(value))
    if unknown:
        raise ValueError(f"{name} has unknown field(s): {', '.join(unknown)}")
    if missing:
        raise ValueError(f"{name} is missing field(s): {', '.join(missing)}")


def _reject_unknown_keys(
    value: Mapping[str, Any],
    allowed: set[str],
    name: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{name} has unknown field(s): {', '.join(unknown)}")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _digest(value: Any, name: str) -> str:
    text = _text(value, name)
    if not _DIGEST.fullmatch(text):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return text


def _identity(value: Any, name: str) -> str:
    return _digest(value, name)


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _text_sequence(value: Any, name: str, *, minimum: int) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{name} must be a list")
    result = [_text(item, name) for item in value]
    if len(result) < minimum or len(set(result)) != len(result):
        raise ValueError(f"{name} must contain {minimum}+ unique values")
    return result


def _digest_sequence(value: Any, name: str, *, minimum: int) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{name} must be a list")
    result = [_digest(item, name) for item in value]
    if len(result) < minimum:
        raise ValueError(f"{name} must contain at least {minimum} values")
    return result


def _project(value: str, name: str) -> str:
    parts = value.split("/")
    if len(parts) != 2 or any(not part for part in parts):
        raise ValueError(f"{name} must be an entity/project slug")
    return value


def _wandb_url(value: Any, name: str) -> str:
    text = _text(value, name)
    parsed = urlparse(text)
    host = str(parsed.hostname or "").lower()
    if parsed.scheme != "https" or not (
        host == "wandb.ai" or host.endswith(".wandb.ai")
    ):
        raise ValueError(f"{name} must be an allowlisted HTTPS W&B URL")
    return text


def _reject_private_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() in _PRIVATE_KEYS:
                raise ValueError("failure lock may not contain private truth")
            _reject_private_fields(nested)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for nested in value:
            _reject_private_fields(nested)
