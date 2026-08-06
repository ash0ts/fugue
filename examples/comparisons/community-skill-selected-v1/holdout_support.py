"""Campaign-local preparation and operator policy for the public Skill studies.

This module intentionally lives beside the campaign manifest.  Fugue core only
validates the generic, digest-bound authorization artifacts emitted here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

import httpx
import yaml
from filelock import FileLock

from fugue.bench.candidates import stable_digest
from fugue.bench.comparison import load_comparison
from fugue.bench.sealed_holdouts import (
    validate_followup_allocation_receipt,
    validate_holdout_authorization,
)
from fugue.bench.study_advancement import (
    HoldoutExposureAuditV1,
    build_holdout_exposure_audit,
    read_holdout_exposure_audit,
    read_study_advancement_decision,
    verify_fresh_holdout_exposure_audit,
    write_holdout_exposure_audit,
)
from fugue.model_plane import resolve_evidence_destination, trace_api_key
from fugue.weave_support import resolved_weave_trace_server_url

_DIGEST = "0123456789abcdef"
_CHECKED_IN_MANIFEST = json.loads(
    Path(__file__).with_name("sealed-holdouts.json").read_text(encoding="utf-8")
)
_LANES = tuple(item["id"] for item in _CHECKED_IN_MANIFEST["lanes"])
_QUERY_FIELDS = tuple(_CHECKED_IN_MANIFEST["audit"]["queried_fields"])
_SAFE_PROJECTION_FIELDS = {
    "id",
    "op_name",
    "task_ids",
    "prompt_fingerprints",
    "input_fingerprints",
    "resource_fingerprints",
}
_FORBIDDEN_EXPOSURE_KEYS = {
    "answer",
    "cost",
    "expected",
    "feedback",
    "gold",
    "label",
    "output",
    "score",
    "summary",
}
_CAMPAIGN = Path("examples/comparisons/community-skill-selected-v1")
_PRIVATE_ROOT = Path(".fugue/private/community-skill-selected-v1/sealed-holdouts")
_HISTORICAL_RECEIPT = _PRIVATE_ROOT / "historical-exposure-receipt.json"
_ALLOCATION_LEDGER = Path(
    ".fugue/private/community-skill-selected-v1/campaign-allocation-ledger.json"
)
_ALLOCATION_LOCK = Path(
    ".fugue/private/community-skill-selected-v1/campaign-allocation-ledger.lock"
)


_SELECTION_FIELDS = {
    f"{prefix}{field}"
    for prefix in ("", "reserve_")
    for field in (
        "task_id",
        "behavior_family",
        "task_sha256",
        "private_label_sha256",
        "resource_sha256",
        "resource_target",
    )
}


def load_sealed_holdout_manifest(path: Path) -> dict[str, Any]:
    """Validate the campaign's identity-only, manifest-driven holdout policy."""

    value = _json_mapping(path)
    if (
        set(value)
        != {
            "schema_version",
            "kind",
            "id",
            "historical_exposure_receipt",
            "audit",
            "lanes",
        }
        or value.get("schema_version") != 1
        or value.get("kind") != "sealed_holdout_manifest"
    ):
        raise ValueError("unsupported sealed holdout manifest")
    historical = _mapping(
        value["historical_exposure_receipt"], "historical receipt policy"
    )
    if historical != {
        "schema_version": 1,
        "kind": "historical_holdout_exposure_receipt",
        "required": True,
        "operator_filename": historical.get("operator_filename"),
    } or not historical["operator_filename"]:
        raise ValueError("historical exposure receipt policy changed")
    audit = _mapping(value["audit"], "holdout audit policy")
    if (
        set(audit)
        != {"maximum_age_seconds", "queried_fields", "historical_projects"}
        or type(audit["maximum_age_seconds"]) is not int
        or not 60 <= audit["maximum_age_seconds"] <= 14_400
        or tuple(audit["queried_fields"]) != _QUERY_FIELDS
    ):
        raise ValueError("sealed holdout audit policy changed")
    projects = _sorted_unique_text(audit["historical_projects"], "historical project")
    if not projects or any(not item.startswith("wandb/") for item in projects):
        raise ValueError("historical projects must be exact wandb project refs")

    selected: set[str] = set()
    reserves: set[str] = set()
    lane_ids: list[str] = []
    for raw_lane in _sequence(value["lanes"], "sealed holdout lanes"):
        lane = _mapping(raw_lane, "sealed holdout lane")
        if set(lane) != {"id", "study_id", "selections"}:
            raise ValueError("sealed holdout lane fields do not match")
        lane_ids.append(_text(lane["id"], "sealed holdout lane id"))
        _text(lane["study_id"], "sealed holdout study id")
        for raw_item in _sequence(lane["selections"], "sealed holdout selections"):
            item = _mapping(raw_item, "sealed holdout selection")
            if set(item) != _SELECTION_FIELDS:
                raise ValueError("sealed holdout selection fields do not match")
            task_id = _text(item["task_id"], "sealed holdout task id")
            reserve_id = _text(item["reserve_task_id"], "reserve holdout task id")
            if (
                task_id == reserve_id
                or item["behavior_family"] != item["reserve_behavior_family"]
                or task_id in selected
                or reserve_id in reserves
            ):
                raise ValueError("a reserve must use the exact selected behavior family")
            selected.add(task_id)
            reserves.add(reserve_id)
            for key, candidate in item.items():
                if key.endswith("_sha256"):
                    _digest(candidate, key)
                elif key.endswith("_target") and not str(candidate).startswith(
                    "/workspace/resources/"
                ):
                    raise ValueError("holdout resources must use the reviewed input mount")
    if tuple(lane_ids) != _LANES or selected & reserves:
        raise ValueError("sealed holdout manifest lane or reserve identities changed")
    return value

def _manifest_items(
    manifest: Mapping[str, Any], *, lane_id: str | None = None
) -> dict[str, dict[str, Any]]:
    """Expand selected/reserve declarations into one identity-indexed policy."""

    values: dict[str, dict[str, Any]] = {}
    for lane in manifest["lanes"]:
        if lane_id is not None and lane["id"] != lane_id:
            continue
        for selection in lane["selections"]:
            for role, prefix, paired_prefix in (
                ("selected", "", "reserve_"),
                ("reserve", "reserve_", ""),
            ):
                task_id = str(selection[f"{prefix}task_id"])
                values[task_id] = {
                    "lane_id": lane["id"],
                    "role": role,
                    "task_id": task_id,
                    "paired_task_id": selection[f"{paired_prefix}task_id"],
                    "behavior_family": selection[f"{prefix}behavior_family"],
                    "task_sha256": selection[f"{prefix}task_sha256"],
                    "private_label_sha256": selection[
                        f"{prefix}private_label_sha256"
                    ],
                    "resource_sha256": selection[f"{prefix}resource_sha256"],
                    "resource_target": selection[f"{prefix}resource_target"],
                }
    return values


_POOL_KEYS = (
    "lane_id",
    "task_id",
    "role",
    "behavior_family",
    "task_sha256",
    "prompt_fingerprint",
    "input_fingerprint",
    "resource_fingerprint",
    "content_fingerprint",
)


def _pool_from_lanes(lanes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pool = [
        {key: item[key] for key in _POOL_KEYS}
        for lane in lanes
        for item in lane["items"]
    ]
    return sorted(pool, key=lambda item: (item["lane_id"], item["task_id"]))


def _exposure_conclusion(
    manifest: Mapping[str, Any], matches: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, str]], str, set[str]]:
    selected = {str(item["task_id"]) for item in matches if item["role"] == "selected"}
    reserves = {str(item["task_id"]) for item in matches if item["role"] == "reserve"}
    declared = _manifest_items(manifest)
    replacements = [
        {
            "exposed_task_id": task_id,
            "reserve_task_id": declared[task_id]["paired_task_id"],
            "behavior_family": declared[task_id]["behavior_family"],
        }
        for task_id in sorted(selected)
    ]
    status = (
        "blocked_exposed_reserve"
        if reserves
        else "reviewed_replacements_required"
        if selected
        else "reviewed_clear"
    )
    return replacements, status, reserves


def build_historical_holdout_exposure_receipt(
    *,
    manifest_path: Path,
    operator_source: Path,
    project_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    project_coverage: Mapping[str, Mapping[str, Any]],
    reviewer_identity: str,
    trace_endpoint: str,
    reviewed_at: datetime | None = None,
) -> dict[str, Any]:
    """Build a private, content-free human review of historical task identities."""

    manifest = load_sealed_holdout_manifest(manifest_path)
    source = operator_source.resolve()
    if source.is_symlink() or not source.is_dir():
        raise ValueError("operator holdout source must be a regular directory")
    pool = _operator_pool_fingerprints(manifest=manifest, operator_source=source)
    projects = tuple(manifest["audit"]["historical_projects"])
    safe_rows, coverage = _validated_exposure_projection(
        projects=projects,
        project_rows=project_rows,
        project_coverage=project_coverage,
    )
    matches = _match_pool_fingerprints(pool=pool, project_rows=safe_rows)
    replacements, status, _ = _exposure_conclusion(manifest, matches)
    instant = (reviewed_at or datetime.now(UTC)).astimezone(UTC)
    reviewer = _text(reviewer_identity, "historical exposure reviewer identity")
    unsigned = {
        "schema_version": 1,
        "kind": "historical_holdout_exposure_receipt",
        "manifest_sha256": _sha256(manifest_path),
        "manifest_digest": stable_digest(manifest),
        "audit_policy_digest": stable_digest(manifest["audit"]),
        "pool_fingerprint_digest": stable_digest(pool),
        "pool_fingerprints": pool,
        "searched_project_refs": list(projects),
        "searched_call_count": sum(len(rows) for rows in safe_rows.values()),
        "queried_fields": list(_QUERY_FIELDS),
        "project_coverage": coverage,
        "project_coverage_complete": True,
        "projection_digest": _coverage_projection_digest(coverage),
        "trace_endpoint_digest": stable_digest(
            {"trace_endpoint": _text(trace_endpoint, "historical trace endpoint")}
        ),
        "matches": matches,
        "required_replacements": replacements,
        "outcome_data_consulted": False,
        "reviewer_identity_digest": hashlib.sha256(
            reviewer.strip().casefold().encode()
        ).hexdigest(),
        "reviewed_at": instant.isoformat(),
        "status": status,
        "private_content_in_receipt": False,
    }
    return {**unsigned, "receipt_digest": stable_digest(unsigned)}


_HISTORICAL_FIELDS = {
    "schema_version",
    "kind",
    "manifest_sha256",
    "manifest_digest",
    "audit_policy_digest",
    "pool_fingerprint_digest",
    "pool_fingerprints",
    "searched_project_refs",
    "searched_call_count",
    "queried_fields",
    "project_coverage",
    "project_coverage_complete",
    "projection_digest",
    "trace_endpoint_digest",
    "matches",
    "required_replacements",
    "outcome_data_consulted",
    "reviewer_identity_digest",
    "reviewed_at",
    "status",
    "private_content_in_receipt",
    "receipt_digest",
}


def validate_historical_holdout_exposure_receipt(
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    pool_fingerprints: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate one private historical review without reopening raw prompts."""

    receipt = dict(value)
    if set(receipt) != _HISTORICAL_FIELDS:
        raise ValueError("historical exposure receipt fields do not match")
    supplied = receipt.pop("receipt_digest", None)
    if supplied != stable_digest(receipt):
        raise ValueError("historical exposure receipt digest does not match")
    receipt["receipt_digest"] = supplied
    for key in (
        "manifest_sha256",
        "manifest_digest",
        "audit_policy_digest",
        "pool_fingerprint_digest",
        "projection_digest",
        "trace_endpoint_digest",
        "reviewer_identity_digest",
        "receipt_digest",
    ):
        _digest(receipt[key], key)

    pool = [dict(item) for item in pool_fingerprints]
    projects = tuple(manifest["audit"]["historical_projects"])
    invariants = {
        "schema_version": 1,
        "kind": "historical_holdout_exposure_receipt",
        "manifest_digest": stable_digest(dict(manifest)),
        "audit_policy_digest": stable_digest(manifest["audit"]),
        "pool_fingerprint_digest": stable_digest(pool),
        "pool_fingerprints": pool,
        "searched_project_refs": list(projects),
        "queried_fields": list(_QUERY_FIELDS),
        "project_coverage_complete": True,
        "outcome_data_consulted": False,
        "private_content_in_receipt": False,
    }
    if any(receipt.get(key) != expected for key, expected in invariants.items()):
        raise ValueError("historical exposure receipt is stale or invalid")
    _instant(receipt["reviewed_at"], "historical exposure reviewed_at")

    coverage = _validate_project_coverage(
        projects=projects,
        project_coverage={
            str(item.get("project_ref")): item
            for item in _sequence(receipt["project_coverage"], "historical project coverage")
            if isinstance(item, Mapping)
        },
    )
    returned = sum(int(item["returned_call_count"]) for item in coverage)
    if (
        coverage != receipt["project_coverage"]
        or receipt["projection_digest"] != _coverage_projection_digest(coverage)
        or type(receipt["searched_call_count"]) is not int
        or receipt["searched_call_count"] != returned
    ):
        raise ValueError("historical exposure project coverage changed")

    matches = _validate_recorded_matches(receipt["matches"], pool=pool)
    if any(not set(item["project_refs"]) <= set(projects) for item in matches):
        raise ValueError("historical exposure match references an undeclared project")
    replacements, status, exposed_reserves = _exposure_conclusion(manifest, matches)
    if receipt["required_replacements"] != replacements or receipt["status"] != status:
        raise ValueError("historical exposure conclusion changed")
    if exposed_reserves:
        raise ValueError("a preregistered reserve was already exposed historically")
    return receipt


def read_historical_holdout_exposure_receipt(
    repo_root: Path, *, preparation_receipt: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    root = repo_root.resolve()
    manifest_path = root / _CAMPAIGN / "sealed-holdouts.json"
    manifest = load_sealed_holdout_manifest(manifest_path)
    preparation = _validate_sealed_holdout_preparation(
        preparation_receipt
        or _json_mapping(root / _PRIVATE_ROOT / "preparation-receipt.json"),
        manifest=manifest,
    )
    receipt = validate_historical_holdout_exposure_receipt(
        _json_mapping(root / _HISTORICAL_RECEIPT),
        manifest=manifest,
        pool_fingerprints=_pool_from_lanes(preparation["lanes"]),
    )
    if preparation["historical_exposure_receipt_digest"] != receipt["receipt_digest"]:
        raise ValueError("sealed preparation does not bind the historical review")
    return receipt


def write_historical_holdout_exposure_receipt(
    path: Path, receipt: Mapping[str, Any]
) -> None:
    _write_private_json(path, dict(receipt))


def _operator_pool_fingerprints(
    *, manifest: Mapping[str, Any], operator_source: Path
) -> list[dict[str, Any]]:
    """Derive safe selected/reserve fingerprints from the restricted packet."""

    values: list[dict[str, Any]] = []
    for lane in manifest["lanes"]:
        lane_id = str(lane["id"])
        source_lane = operator_source / lane_id
        tasks = _jsonl_by_id(source_lane / "tasks.jsonl")
        declared = _manifest_items(manifest, lane_id=lane_id)
        if set(tasks) != set(declared):
            raise ValueError(f"operator packet task pool changed for {lane_id}")
        for task_id, task in sorted(tasks.items()):
            item = declared[task_id]
            if _canonical_digest(task) != item["task_sha256"]:
                raise ValueError(f"sealed task content changed: {task_id}")
            resources = _sequence(task.get("resources"), f"{task_id} resources")
            if len(resources) != 1:
                raise ValueError(f"sealed task must bind one resource: {task_id}")
            resource = _mapping(resources[0], f"{task_id} resource")
            relative = _safe_relative(resource.get("path"), "operator resource")
            resource_digest = _sha256(_inside(source_lane, source_lane / relative))
            if (
                set(resource) != {"path", "target"}
                or resource_digest != item["resource_sha256"]
                or resource["target"] != item["resource_target"]
            ):
                raise ValueError(f"sealed task resource changed: {task_id}")
            values.append(
                {
                    **{key: item[key] for key in ("lane_id", "task_id", "role", "behavior_family", "task_sha256")},
                    **_task_fingerprints(task=task, resource_digest=resource_digest),
                }
            )
    values.sort(key=lambda item: (item["lane_id"], item["task_id"]))
    if len(values) != len(_manifest_items(manifest)):
        raise ValueError("historical exposure review requires the complete holdout pool")
    return values
def _task_fingerprints(
    *, task: Mapping[str, Any], resource_digest: str
) -> dict[str, str]:
    input_value = _mapping(task.get("input"), "sealed holdout input")
    prompt = _text(input_value.get("question"), "sealed holdout prompt")
    input_fingerprint = stable_digest(input_value)
    prompt_fingerprint = stable_digest({"prompt": prompt})
    resource_fingerprint = _digest(resource_digest, "sealed resource fingerprint")
    return {
        "prompt_fingerprint": prompt_fingerprint,
        "input_fingerprint": input_fingerprint,
        "resource_fingerprint": resource_fingerprint,
        "content_fingerprint": stable_digest(
            {
                "prompt_fingerprint": prompt_fingerprint,
                "input_fingerprint": input_fingerprint,
                "resource_fingerprint": resource_fingerprint,
            }
        ),
    }


_PREPARATION_FIELDS = {
    "schema_version",
    "kind",
    "manifest_sha256",
    "manifest_digest",
    "historical_exposure_receipt_digest",
    "pool_fingerprint_digest",
    "lanes",
    "selected_task_count",
    "reserve_task_count",
    "private_content_in_receipt",
    "receipt_digest",
}
_LANE_RECEIPT_FILES = (
    ("tasks_path", "tasks_sha256"),
    ("private_labels_path", "private_labels_sha256"),
    ("reserve_tasks_path", "reserve_tasks_sha256"),
    ("reserve_private_labels_path", "reserve_private_labels_sha256"),
)
_LANE_RECEIPT_FIELDS = {
    "lane_id",
    "study_id",
    "holdout_suite_digest",
    "reserve_pool_digest",
    "items",
    *(key for pair in _LANE_RECEIPT_FILES for key in pair),
}
_PREPARED_ITEM_FIELDS = {
    "lane_id",
    "task_id",
    "role",
    "behavior_family",
    "paired_task_id",
    "source_task_digest",
    "private_label_digest",
    "resource_digest",
    "prepared_task_digest",
    "task_sha256",
    "prompt_fingerprint",
    "input_fingerprint",
    "resource_fingerprint",
    "content_fingerprint",
}


def _prepared_identity(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "role": item["role"],
        "behavior_family": item["behavior_family"],
        "paired_task_id": item["paired_task_id"],
        "source_task_digest": item["task_sha256"],
        "private_label_digest": item["private_label_sha256"],
        "resource_digest": item["resource_sha256"],
    }


def prepare_sealed_holdouts(
    *,
    manifest_path: Path,
    operator_source: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Copy a digest-matched operator packet into ignored private state."""

    manifest = load_sealed_holdout_manifest(manifest_path)
    root, source = repo_root.resolve(), operator_source.resolve()
    if source.is_symlink() or not source.is_dir():
        raise ValueError("operator holdout source must be a regular directory")
    pool = _operator_pool_fingerprints(manifest=manifest, operator_source=source)
    historical_source = source / str(
        manifest["historical_exposure_receipt"]["operator_filename"]
    )
    historical = validate_historical_holdout_exposure_receipt(
        _json_mapping(historical_source),
        manifest=manifest,
        pool_fingerprints=pool,
    )
    if historical["manifest_sha256"] != _sha256(manifest_path):
        raise ValueError("historical exposure receipt manifest bytes changed")
    _copy_private(historical_source, root / _HISTORICAL_RECEIPT)
    pool_by_id = {item["task_id"]: item for item in pool}
    private_root = root / _PRIVATE_ROOT
    lane_receipts: list[dict[str, Any]] = []

    for lane in manifest["lanes"]:
        lane_id, source_lane = str(lane["id"]), source / str(lane["id"])
        declared = _manifest_items(manifest, lane_id=lane_id)
        tasks = _jsonl_by_id(source_lane / "tasks.jsonl")
        labels = _jsonl_by_id(source_lane / "private-labels.jsonl")
        if set(tasks) != set(declared) or set(labels) != set(declared):
            raise ValueError(f"operator packet pool changed for {lane_id}")

        target = private_root / lane_id
        prepared: dict[str, dict[str, dict[str, Any]]] = {
            "selected": {"tasks": {}, "labels": {}},
            "reserve": {"tasks": {}, "labels": {}},
        }
        receipts: list[dict[str, Any]] = []
        for task_id, item in sorted(declared.items()):
            task, label = tasks[task_id], labels[task_id]
            if (
                _canonical_digest(task) != item["task_sha256"]
                or _canonical_digest(label) != item["private_label_sha256"]
            ):
                raise ValueError(f"sealed task or private label changed: {task_id}")
            if task.get("partition") != "holdout":
                raise ValueError(f"sealed task is not a holdout: {task_id}")
            resources = _sequence(task.get("resources"), f"{task_id} resources")
            resource = _mapping(
                resources[0] if len(resources) == 1 else None, f"{task_id} resource"
            )
            source_resource = _inside(
                source_lane,
                source_lane / _safe_relative(resource.get("path"), "operator resource"),
            )
            if (
                set(resource) != {"path", "target"}
                or resource.get("target") != item["resource_target"]
                or _sha256(source_resource) != item["resource_sha256"]
            ):
                raise ValueError(f"sealed task resource changed: {task_id}")
            target_resource = target / "resources" / f"{task_id}.tar"
            _copy_private(source_resource, target_resource)
            prepared_task = {
                **task,
                "resources": [
                    {
                        "path": target_resource.relative_to(root).as_posix(),
                        "target": item["resource_target"],
                    }
                ],
            }
            prepared[item["role"]]["tasks"][task_id] = prepared_task
            prepared[item["role"]]["labels"][task_id] = label
            fingerprint = pool_by_id[task_id]
            receipts.append(
                {
                    "lane_id": lane_id,
                    "task_id": task_id,
                    **_prepared_identity(item),
                    "prepared_task_digest": _canonical_digest(prepared_task),
                    **{key: fingerprint[key] for key in _POOL_KEYS[4:]},
                }
            )

        paths = {
            "tasks_path": target / "tasks.jsonl",
            "private_labels_path": target / "private-labels.jsonl",
            "reserve_tasks_path": target / "reserve-tasks.jsonl",
            "reserve_private_labels_path": target / "reserve-private-labels.jsonl",
        }
        for key, path in paths.items():
            role = "reserve" if key.startswith("reserve_") else "selected"
            kind = "labels" if "labels" in key else "tasks"
            _write_private_jsonl(path, list(prepared[role][kind].values()))
        selected_ids = set(prepared["selected"]["tasks"])
        reserve_ids = set(prepared["reserve"]["tasks"])
        lane_receipts.append(
            {
                "lane_id": lane_id,
                "study_id": lane["study_id"],
                "holdout_suite_digest": _prepared_suite_digest(
                    receipts, selected_task_ids=selected_ids
                ),
                "reserve_pool_digest": _prepared_suite_digest(
                    receipts, selected_task_ids=reserve_ids
                ),
                "items": receipts,
                **{
                    field: (
                        path.relative_to(root).as_posix()
                        if field.endswith("_path")
                        else _sha256(paths[path_field])
                    )
                    for path_field, digest_field in _LANE_RECEIPT_FILES
                    for field, path in (
                        (path_field, paths[path_field]),
                        (digest_field, paths[path_field]),
                    )
                },
            }
        )

    selected_count = sum(
        item["role"] == "selected" for lane in lane_receipts for item in lane["items"]
    )
    unsigned = {
        "schema_version": 1,
        "kind": "sealed_holdout_preparation_receipt",
        "manifest_sha256": _sha256(manifest_path),
        "manifest_digest": stable_digest(manifest),
        "historical_exposure_receipt_digest": historical["receipt_digest"],
        "pool_fingerprint_digest": stable_digest(pool),
        "lanes": lane_receipts,
        "selected_task_count": selected_count,
        "reserve_task_count": len(pool) - selected_count,
        "private_content_in_receipt": False,
    }
    receipt = {**unsigned, "receipt_digest": stable_digest(unsigned)}
    _write_private_json(private_root / "preparation-receipt.json", receipt)
    return receipt


def read_sealed_holdout_preparation(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    manifest_path = root / _CAMPAIGN / "sealed-holdouts.json"
    value = _json_mapping(root / _PRIVATE_ROOT / "preparation-receipt.json")
    if value.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError("sealed holdout preparation manifest bytes changed")
    return _validate_sealed_holdout_preparation(
        value,
        manifest=load_sealed_holdout_manifest(manifest_path),
        repo_root=root,
    )


def _validate_sealed_holdout_preparation(  # noqa: C901 - strict persisted V1 reader
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Validate the content-free receipt for selected and reserve holdout pools."""

    receipt = dict(value)
    if set(receipt) != _PREPARATION_FIELDS:
        raise ValueError("sealed holdout preparation receipt fields do not match")
    supplied = receipt.pop("receipt_digest", None)
    if supplied != stable_digest(receipt):
        raise ValueError("sealed holdout preparation receipt digest does not match")
    receipt["receipt_digest"] = supplied
    for key in (
        "manifest_sha256",
        "historical_exposure_receipt_digest",
        "pool_fingerprint_digest",
        "receipt_digest",
    ):
        _digest(receipt[key], key)

    declared = _manifest_items(manifest)
    selected_count = sum(item["role"] == "selected" for item in declared.values())
    invariants = {
        "schema_version": 1,
        "kind": "sealed_holdout_preparation_receipt",
        "manifest_digest": stable_digest(dict(manifest)),
        "selected_task_count": selected_count,
        "reserve_task_count": len(declared) - selected_count,
        "private_content_in_receipt": False,
    }
    if any(receipt.get(key) != expected for key, expected in invariants.items()):
        raise ValueError("sealed holdout preparation receipt is stale or invalid")

    manifest_lanes = {str(item["id"]): item for item in manifest["lanes"]}
    lanes = _sequence(receipt["lanes"], "sealed holdout preparation lanes")
    if {
        str(item.get("lane_id")) for item in lanes if isinstance(item, Mapping)
    } != set(manifest_lanes):
        raise ValueError("sealed holdout preparation lanes changed")
    for raw_lane in lanes:
        lane = _mapping(raw_lane, "sealed holdout preparation lane")
        if set(lane) != _LANE_RECEIPT_FIELDS:
            raise ValueError("sealed holdout lane preparation fields do not match")
        lane_id = str(lane["lane_id"])
        lane_declared = _manifest_items(manifest, lane_id=lane_id)
        if lane["study_id"] != manifest_lanes[lane_id]["study_id"]:
            raise ValueError("sealed holdout preparation Study changed")
        items = _sequence(lane["items"], "sealed holdout prepared items")
        by_id = {
            str(item.get("task_id")): _mapping(item, "sealed holdout prepared item")
            for item in items
            if isinstance(item, Mapping)
        }
        if set(by_id) != set(lane_declared) or len(items) != len(by_id):
            raise ValueError("sealed holdout prepared pool identities changed")
        for task_id, item in by_id.items():
            source = lane_declared[task_id]
            if (
                set(item) != _PREPARED_ITEM_FIELDS
                or item["lane_id"] != lane_id
                or {
                    key: item[key] for key in _prepared_identity(source)
                }
                != _prepared_identity(source)
                or item["task_sha256"] != item["source_task_digest"]
            ):
                raise ValueError("sealed holdout prepared item changed from manifest")
            for key in _POOL_KEYS[4:]:
                _digest(item[key], f"prepared holdout {key}")
        for role, digest_key in (
            ("selected", "holdout_suite_digest"),
            ("reserve", "reserve_pool_digest"),
        ):
            task_ids = {
                task_id for task_id, item in lane_declared.items() if item["role"] == role
            }
            if lane[digest_key] != _prepared_suite_digest(
                items, selected_task_ids=task_ids
            ):
                raise ValueError("sealed holdout prepared suite digest changed")
        if repo_root is not None:
            for path_key, digest_key in _LANE_RECEIPT_FILES:
                path = _inside(repo_root, repo_root / str(lane[path_key]))
                if _sha256(path) != lane[digest_key]:
                    raise ValueError("sealed holdout prepared input changed")

    pool = _pool_from_lanes(lanes)
    if receipt["pool_fingerprint_digest"] != stable_digest(pool):
        raise ValueError("sealed holdout preparation pool fingerprints changed")
    if repo_root is not None:
        historical = validate_historical_holdout_exposure_receipt(
            _json_mapping(repo_root / _HISTORICAL_RECEIPT),
            manifest=manifest,
            pool_fingerprints=pool,
        )
        if (
            historical["manifest_sha256"]
            != _sha256(repo_root / _CAMPAIGN / "sealed-holdouts.json")
            or receipt["historical_exposure_receipt_digest"]
            != historical["receipt_digest"]
        ):
            raise ValueError("sealed holdout preparation historical review changed")
    return receipt
def _prepared_suite_digest(
    items: Sequence[Mapping[str, Any]], *, selected_task_ids: set[str]
) -> str:
    selected = [dict(item) for item in items if item.get("task_id") in selected_task_ids]
    if len(selected) != len(selected_task_ids):
        raise ValueError("sealed holdout suite is missing a prepared identity")
    selected.sort(key=lambda item: str(item["task_id"]))
    return stable_digest({"items": selected})


def validate_sealed_holdouts_zero_model(
    *, repo_root: Path, preparation_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Prove all targeted bases fail and all known-good artifacts pass offline."""

    root = repo_root.resolve()
    lane_receipts = {
        item["lane_id"]: item for item in preparation_receipt.get("lanes", [])
    }
    if set(lane_receipts) != set(_LANES):
        raise ValueError("sealed holdout preparation receipt is incomplete")
    selected_results: list[dict[str, Any]] = []
    reserve_results: list[dict[str, Any]] = []
    for lane in _LANES:
        lane_receipt = lane_receipts[lane]
        selected_tasks = _jsonl_by_id(root / lane_receipt["tasks_path"])
        selected_labels = _jsonl_by_id(root / lane_receipt["private_labels_path"])
        reserve_tasks = _jsonl_by_id(root / lane_receipt["reserve_tasks_path"])
        reserve_labels = _jsonl_by_id(
            root / lane_receipt["reserve_private_labels_path"]
        )
        tasks = {**selected_tasks, **reserve_tasks}
        labels = {**selected_labels, **reserve_labels}
        if len(tasks) != 8 or set(tasks) != set(labels):
            raise ValueError(
                f"sealed holdout pool must contain four selected and four reserves: {lane}"
            )
        scorer = _load_module(root / _CAMPAIGN / lane / "scorer.py", f"sealed_{lane}")
        for task_id in sorted(tasks):
            task = tasks[task_id]
            label = labels[task_id]
            result = validate_task_zero_model(
                repo_root=root,
                lane=lane,
                task=task,
                label=label,
                scorer=scorer,
            )
            if task_id in selected_tasks:
                selected_results.append(result)
            else:
                reserve_results.append(result)
    reserve_unsigned = {
        "schema_version": 1,
        "kind": "sealed_holdout_reserve_preparation_receipt",
        "preparation_receipt_digest": preparation_receipt["receipt_digest"],
        "task_count": len(reserve_results),
        "results": reserve_results,
    }
    reserve_receipt = {
        **reserve_unsigned,
        "receipt_digest": stable_digest(reserve_unsigned),
    }
    _write_private_json(
        root / _PRIVATE_ROOT / "reserve-preparation-receipt.json", reserve_receipt
    )
    unsigned = {
        "schema_version": 1,
        "kind": "sealed_holdout_zero_model_receipt",
        "preparation_receipt_digest": preparation_receipt["receipt_digest"],
        "task_count": len(selected_results),
        "reserve_preparation_receipt_digest": reserve_receipt["receipt_digest"],
        "results": selected_results,
    }
    value = {**unsigned, "receipt_digest": stable_digest(unsigned)}
    _write_private_json(root / _PRIVATE_ROOT / "zero-model-receipt.json", value)
    return value


def validate_task_zero_model(
    *,
    repo_root: Path,
    lane: str,
    task: Mapping[str, Any],
    label: Mapping[str, Any],
    scorer: ModuleType,
) -> dict[str, Any]:
    """Prove one reviewed gold and one isolated task-specific mutant offline."""

    task_id = _text(task.get("id"), "zero-model task id")
    expected = _mapping(label.get("expected"), f"{task_id} expected values")
    gold = label.get("gold_output")
    fixture_builder = getattr(scorer, "zero_model_fixture", None)
    if not callable(fixture_builder):
        raise RuntimeError(f"zero-model fixture builder is missing: {task_id}")
    fixture = _mapping(
        fixture_builder(task, gold, expected), f"{task_id} zero-model fixture"
    )
    if set(fixture) != {"target_dimensions", "mutant"}:
        raise RuntimeError(f"zero-model fixture fields changed: {task_id}")
    targets = set(
        _sorted_unique_text(
            fixture["target_dimensions"], f"{task_id} target dimension"
        )
    )
    mutant = fixture["mutant"]
    if stable_digest(mutant) == stable_digest(gold):
        raise RuntimeError(f"zero-model mutant is identical to gold: {task_id}")
    gold_evidence: dict[str, Any] = {"expected": expected}
    mutant_evidence: dict[str, Any] = {"expected": expected}
    if lane != "superpowers-writing-plans":
        mutant_receipt, mutant_status = _host_verifier_receipt(
            repo_root, lane, task, mutant, expected
        )
        gold_receipt, gold_status = _host_verifier_receipt(
            repo_root, lane, task, gold, expected
        )
        if gold_status != 0 or (
            lane == "vercel-react-best-practices" and mutant_status == 0
        ):
            raise RuntimeError(f"host verifier disagrees with fixture: {task_id}")
        mutant_evidence.update(
            host_verifier=mutant_receipt,
            changed_paths=sorted(mutant["files"]),
        )
        gold_evidence.update(
            host_verifier=gold_receipt,
            changed_paths=sorted(gold["files"]),
        )
    mutant_scores = scorer.score(task, mutant, mutant_evidence)
    gold_scores = scorer.score(task, gold, gold_evidence)
    if set(mutant_scores) != set(gold_scores) or not targets <= set(gold_scores):
        raise RuntimeError(f"zero-model dimensions changed: {task_id}")
    failed = {name for name, passed in mutant_scores.items() if not passed}
    if not all(gold_scores.values()) or failed != targets:
        raise RuntimeError(
            f"zero-model fixture is not isolated: {task_id}: "
            f"target={sorted(targets)}, failed={sorted(failed)}, gold={gold_scores}"
        )
    inventory_digest = _required_path_inventory_digest(
        repo_root=repo_root,
        task=task,
        expected=expected,
        required=lane == "superpowers-writing-plans",
    )
    return {
        "lane_id": lane,
        "task_id": task_id,
        "target_dimensions": sorted(targets),
        "mutant_status": "target_dimensions_failed_only",
        "gold_status": "all_dimensions_passed",
        "mutant_scores_digest": stable_digest(mutant_scores),
        "gold_scores_digest": stable_digest(gold_scores),
        "source_inventory_digest": inventory_digest,
    }


def _required_path_inventory_digest(
    *,
    repo_root: Path,
    task: Mapping[str, Any],
    expected: Mapping[str, Any],
    required: bool,
) -> str | None:
    if not required:
        return None
    resources = _sequence(task.get("resources"), "zero-model task resources")
    if len(resources) != 1:
        raise RuntimeError("Superpowers zero-model task needs one source archive")
    resource = _mapping(resources[0], "zero-model task resource")
    archive_path = _inside(repo_root, repo_root / str(resource.get("path")))
    with tarfile.open(archive_path, "r:") as archive:
        paths = sorted(
            name.removeprefix("repo/")
            for member in archive.getmembers()
            if member.isfile()
            for name in (member.name,)
        )
    required_paths = {
        _text(item, "required source path")
        for item in _sequence(expected.get("required_paths"), "required source paths")
    }
    if not required_paths or not required_paths <= set(paths):
        raise RuntimeError("reviewed source anchors are absent from the locked archive")
    return stable_digest(paths)


def fetch_task_identity_projections(
    *,
    projects: Sequence[str],
    env: Mapping[str, str],
    limit: int = 10_000,
    project_exists: Callable[[str], bool] | None = None,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    str,
    dict[str, dict[str, Any]],
]:
    """Read safe task-input fingerprints; never request outcome-bearing fields."""

    if type(limit) is not int or limit < 1:
        raise ValueError("holdout audit limit must be a positive integer")
    api_key = trace_api_key(env)
    if not api_key:
        raise ValueError("holdout exposure audit requires a W&B API key")
    destination = resolve_evidence_destination(env)
    endpoint = resolved_weave_trace_server_url(env).rstrip("/")
    if not endpoint.startswith("https://"):
        raise ValueError("holdout exposure audit requires an HTTPS trace endpoint")
    rows_by_project: dict[str, list[dict[str, Any]]] = {}
    coverage_by_project: dict[str, dict[str, Any]] = {}
    with httpx.Client(timeout=60.0, auth=httpx.BasicAuth("api", api_key)) as client:
        for project in sorted(set(projects)):
            present = (
                project_exists(project)
                if project_exists is not None
                else _wandb_project_exists(
                    client, api_base_url=destination.api_base_url, project=project
                )
            )
            if not present:
                rows_by_project[project] = []
                coverage_by_project[project] = _project_coverage(
                    project=project,
                    project_status="absent",
                    returned_call_count=0,
                    query_limit=limit,
                    rows=(),
                )
                continue
            requested_limit = limit + 1
            response = client.post(
                f"{endpoint}/calls/stream_query",
                json={
                    "project_id": project,
                    "filter": {"trace_roots_only": False},
                    "limit": requested_limit,
                    "include_costs": False,
                    "include_feedback": False,
                    "columns": list(_QUERY_FIELDS),
                },
            )
            if response.status_code == 404:
                rows_by_project[project] = []
                coverage_by_project[project] = _project_coverage(
                    project=project,
                    project_status="present",
                    returned_call_count=0,
                    query_limit=limit,
                    rows=(),
                )
                continue
            response.raise_for_status()
            rows = [
                _safe_identity_projection(row) for row in _decode_stream(response.text)
            ]
            if len(rows) > limit:
                raise RuntimeError(f"holdout audit reached its limit for {project}")
            rows.sort(key=_canonical_json)
            rows_by_project[project] = rows
            coverage_by_project[project] = _project_coverage(
                project=project,
                project_status="present",
                returned_call_count=len(rows),
                query_limit=limit,
                rows=rows,
            )
    return rows_by_project, endpoint, coverage_by_project


def _wandb_project_exists(
    client: httpx.Client, *, api_base_url: str, project: str
) -> bool:
    entity, name = project.split("/", maxsplit=1)
    response = client.post(
        f"{api_base_url.rstrip('/')}/graphql",
        json={
            "query": (
                "query Project($entity: String!, $project: String!) { "
                "project(name: $project, entityName: $entity) { id name } }"
            ),
            "variables": {"entity": entity, "project": name},
        },
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(f"W&B project lookup failed for {project}")
    value = (payload.get("data") or {}).get("project")
    return isinstance(value, Mapping)


def build_live_holdout_audits(
    *,
    manifest_path: Path,
    preparation_receipt: Mapping[str, Any],
    historical_receipt: Mapping[str, Any],
    project_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    project_coverage: Mapping[str, Mapping[str, Any]],
    replacements: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> dict[str, HoldoutExposureAuditV1]:
    manifest = load_sealed_holdout_manifest(manifest_path)
    if preparation_receipt.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError("sealed holdout preparation manifest bytes changed")
    projects = tuple(manifest["audit"]["historical_projects"])
    safe_rows, coverage = _validated_exposure_projection(
        projects=projects,
        project_rows=project_rows,
        project_coverage=project_coverage,
    )
    preparation_receipt = _validate_sealed_holdout_preparation(
        preparation_receipt, manifest=manifest
    )
    receipt_lanes = {
        item["lane_id"]: item for item in preparation_receipt.get("lanes", [])
    }
    if set(receipt_lanes) != set(_LANES):
        raise ValueError("sealed holdout preparation receipt is incomplete")
    pool = [
        {
            key: item[key]
            for key in (
                "lane_id",
                "task_id",
                "role",
                "behavior_family",
                "task_sha256",
                "prompt_fingerprint",
                "input_fingerprint",
                "resource_fingerprint",
                "content_fingerprint",
            )
        }
        for lane in preparation_receipt["lanes"]
        for item in lane["items"]
    ]
    pool.sort(key=lambda item: (item["lane_id"], item["task_id"]))
    historical = validate_historical_holdout_exposure_receipt(
        historical_receipt,
        manifest=manifest,
        pool_fingerprints=pool,
    )
    if historical["manifest_sha256"] != _sha256(manifest_path):
        raise ValueError("live audit historical receipt manifest bytes changed")
    if preparation_receipt["historical_exposure_receipt_digest"] != historical[
        "receipt_digest"
    ]:
        raise ValueError("live audit does not bind the prepared historical review")
    replacement_map = dict(replacements or {})
    selections = {
        item["task_id"]: item
        for lane in manifest["lanes"]
        for item in lane["selections"]
    }
    for exposed, reserve in replacement_map.items():
        selection = selections.get(exposed)
        if selection is None or selection["reserve_task_id"] != reserve:
            raise ValueError("holdout replacement was not preregistered")
    live_matches = _match_pool_fingerprints(pool=pool, project_rows=safe_rows)
    historical_matches = _validate_recorded_matches(
        historical["matches"], pool=pool
    )
    exposed_pool_ids = {
        item["task_id"] for item in [*historical_matches, *live_matches]
    }
    exposed_reserves = {
        item["task_id"]
        for item in [*historical_matches, *live_matches]
        if item["role"] == "reserve"
    }
    if exposed_reserves:
        raise ValueError("a preregistered reserve was exposed and cannot be activated")
    matched = set(selections) & exposed_pool_ids
    if matched != set(replacement_map):
        raise ValueError(
            "every exposed holdout requires its exact preregistered replacement"
        )
    instant = (now or datetime.now(UTC)).astimezone(UTC)
    expires = instant + timedelta(seconds=manifest["audit"]["maximum_age_seconds"])
    audits: dict[str, HoldoutExposureAuditV1] = {}
    for lane in manifest["lanes"]:
        lane_id = str(lane["id"])
        lane_matches = sorted(
            item["task_id"] for item in lane["selections"] if item["task_id"] in matched
        )
        lane_replacements = [
            {
                "exposed_task_id": task_id,
                "reserve_task_id": replacement_map[task_id],
                "behavior_family": selections[task_id]["behavior_family"],
            }
            for task_id in lane_matches
        ]
        final_ids = [
            replacement_map.get(item["task_id"], item["task_id"])
            for item in lane["selections"]
        ]
        suite_digest = _prepared_suite_digest(
            receipt_lanes[lane_id]["items"], selected_task_ids=set(final_ids)
        )
        audits[lane_id] = build_holdout_exposure_audit(
            study_id=lane["study_id"],
            holdout_suite_digest=suite_digest,
            selected_task_ids=final_ids,
            searched_project_refs=projects,
            project_rows=safe_rows,
            prior_evidence_digest=historical["receipt_digest"],
            historical_exposure_receipt_digest=historical["receipt_digest"],
            pool_fingerprint_digest=preparation_receipt[
                "pool_fingerprint_digest"
            ],
            project_coverage_digest=stable_digest(coverage),
            audited_at=instant.isoformat(),
            expires_at=expires.isoformat(),
            queried_fields=_QUERY_FIELDS,
            matched_task_ids=lane_matches,
            replacements=lane_replacements,
        )
    return audits


def _materialize_followup_spec(
    *,
    raw: Mapping[str, Any],
    source_path: Path,
    target: Path,
    root: Path,
    study_id: str,
    question: str,
    tasks_path: Path,
    labels_path: Path,
    task_ids: Sequence[str],
    attempts: int,
    stage_id: str,
    retry_allowance: int,
    baseline: Mapping[str, Any] | None = None,
    authorization_path: Path | None = None,
) -> tuple[Any, Path]:
    """Derive either allowed follow-up without duplicating execution policy."""

    followup = json.loads(json.dumps(raw))
    followup.update(
        id=study_id,
        question=question,
        taskset={
            "tasks": os.path.relpath(tasks_path, target),
            "private_labels": os.path.relpath(labels_path, target),
        },
    )
    if baseline is not None:
        followup["baseline"] = dict(baseline)
    _relativize_evaluator_inputs(
        followup, source_path=source_path, target=target, root=root
    )
    allocation_path = target / "campaign-allocation.json"
    execution = _mapping(followup["execution"], "follow-up execution")
    execution.update(
        research_id=study_id,
        attempts=attempts,
        concurrency=2,
        campaign_allocation_receipt=allocation_path.name,
    )
    if authorization_path is not None:
        execution["holdout_authorization"] = authorization_path.name
    judge_reserve = sum(
        float(item.get("reserve_cost_usd") or 0.0)
        for item in followup["evaluators"]
        if item.get("type") == "llm_judge"
    )
    per_cell = float(execution["reserve_per_attempt_usd"]) + judge_reserve
    logical_cells = len(task_ids) * 2 * attempts
    coordination = _mapping(
        _mapping(raw["execution"], "source execution")["schedule"],
        "source schedule",
    )["coordination"]
    execution["max_cost_usd"] = round(
        (logical_cells + retry_allowance) * per_cell, 6
    )
    execution["schedule"] = {
        "stages": [
            {
                "id": stage_id,
                "pair_complete": False,
                "task_ids": list(task_ids),
                "trial_indexes": list(range(attempts)),
            }
        ],
        "worker_limit": 2,
        "wave_size": 4,
        "infrastructure_retry_limit": retry_allowance,
        "maximum_physical_executions": logical_cells + retry_allowance,
        "maximum_in_flight_cost_usd": round(2 * per_cell, 6),
        "coordination": coordination,
    }
    followup["execution"] = execution
    spec_path = target / "comparison.yaml"
    _write_private_text(spec_path, yaml.safe_dump(followup, sort_keys=False))
    return load_comparison(spec_path, repo_root=root), spec_path


def build_sealed_holdout_comparison(
    *,
    comparison_path: Path,
    advancement_decision_path: Path,
    exposure_audit_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Build one separately approved 16-cell holdout comparison."""

    root = repo_root.resolve()
    source_path = _inside(root, comparison_path.resolve())
    raw = _yaml_mapping(source_path)
    source_spec = load_comparison(source_path, repo_root=root)
    lane = source_path.parent.name
    if lane not in _LANES:
        raise ValueError("sealed holdout source is not a campaign lane")
    decision = read_study_advancement_decision(advancement_decision_path)
    audit = read_holdout_exposure_audit(exposure_audit_path)
    if decision.status != "advance_holdout":
        raise ValueError("sealed holdout requires an advancing development decision")
    if decision.study_id != raw.get("id") or audit.study_id != raw.get("id"):
        raise ValueError("holdout decision or audit belongs to another Study")
    if (
        decision.holdout_suite_digest != audit.holdout_suite_digest
        or decision.holdout_exposure_audit_digest != audit.audit_digest
    ):
        raise ValueError("advancement decision does not bind the exact holdout audit")
    verify_fresh_holdout_exposure_audit(
        audit,
        study_id=decision.study_id,
        holdout_suite_digest=str(decision.holdout_suite_digest),
    )
    preparation = read_sealed_holdout_preparation(root)
    if (
        audit.historical_exposure_receipt_digest
        != preparation["historical_exposure_receipt_digest"]
        or audit.pool_fingerprint_digest
        != preparation["pool_fingerprint_digest"]
    ):
        raise ValueError("holdout audit does not bind the prepared exposure inputs")
    zero_model = _json_mapping(root / _PRIVATE_ROOT / "zero-model-receipt.json")
    reserve_preparation = _json_mapping(
        root / _PRIVATE_ROOT / "reserve-preparation-receipt.json"
    )
    if (
        zero_model.get("preparation_receipt_digest") != preparation["receipt_digest"]
        or zero_model.get("task_count") != 12
        or reserve_preparation.get("preparation_receipt_digest")
        != preparation["receipt_digest"]
        or reserve_preparation.get("task_count") != 12
        or zero_model.get("reserve_preparation_receipt_digest")
        != reserve_preparation.get("receipt_digest")
    ):
        raise ValueError("sealed holdout zero-model receipt is missing or stale")
    lane_receipts = {item["lane_id"]: item for item in preparation["lanes"]}
    lane_receipt = _mapping(lane_receipts.get(lane), "sealed lane preparation")
    selected_tasks = _jsonl_by_id(root / str(lane_receipt["tasks_path"]))
    selected_labels = _jsonl_by_id(root / str(lane_receipt["private_labels_path"]))
    reserve_tasks = _jsonl_by_id(root / str(lane_receipt["reserve_tasks_path"]))
    reserve_labels = _jsonl_by_id(
        root / str(lane_receipt["reserve_private_labels_path"])
    )
    pool_tasks = {**selected_tasks, **reserve_tasks}
    pool_labels = {**selected_labels, **reserve_labels}
    selected_ids = tuple(audit.selected_task_ids)
    if (
        len(selected_ids) != 4
        or not set(selected_ids) <= set(pool_tasks)
        or not set(selected_ids) <= set(pool_labels)
    ):
        raise ValueError(
            "audited holdout identities do not match prepared private inputs"
        )
    if audit.holdout_suite_digest != _prepared_suite_digest(
        lane_receipt["items"], selected_task_ids=set(selected_ids)
    ):
        raise ValueError(
            "replacement holdout content is not prepared under the audited suite"
        )
    selected_zero_model_results = {
        str(item.get("task_id")): item for item in zero_model.get("results", [])
    }
    reserve_zero_model_results = {
        str(item.get("task_id")): item
        for item in reserve_preparation.get("results", [])
    }
    zero_model_results = {
        **selected_zero_model_results,
        **reserve_zero_model_results,
    }
    if not set(selected_ids) <= set(zero_model_results) or any(
        zero_model_results[task_id].get("gold_status") != "all_dimensions_passed"
        or zero_model_results[task_id].get("mutant_status")
        != "target_dimensions_failed_only"
        for task_id in selected_ids
    ):
        raise ValueError("audited holdout did not pass its locked zero-model proof")

    target = (
        root / ".fugue/private/community-skill-selected-v1" / lane / "sealed-holdout"
    )
    tasks_path = target / "tasks.jsonl"
    labels_path = target / "private-labels.jsonl"
    _write_private_jsonl(tasks_path, [pool_tasks[item] for item in selected_ids])
    _write_private_jsonl(labels_path, [pool_labels[item] for item in selected_ids])
    authorization_path = target / "holdout-authorization.json"
    allocation_path = target / "campaign-allocation.json"
    retry_allowance = int(lane == _LANES[0])
    spec, spec_path = _materialize_followup_spec(
        raw=raw,
        source_path=source_path,
        target=target,
        root=root,
        study_id=f"{raw['id']}-sealed-holdout-v1",
        question=(
            "Does the exact candidate Skill preserve its development improvement "
            f"on the four sealed {lane} holdouts under unchanged fixed conditions?"
        ),
        tasks_path=tasks_path,
        labels_path=labels_path,
        task_ids=selected_ids,
        attempts=2,
        stage_id="sealed-holdout",
        retry_allowance=retry_allowance,
        authorization_path=authorization_path,
    )
    allocation = allocate_campaign_branch(
        repo_root=root,
        lane=lane,
        branch="holdout",
        source_study_id=decision.study_id,
        followup_study_id=spec.id,
        followup_spec_digest=spec.spec_digest,
        logical_cells=16,
        receipt_path=allocation_path,
    )
    selected_items = _holdout_authorization_items(
        root=root,
        lane_receipt=lane_receipt,
        tasks={item: pool_tasks[item] for item in selected_ids},
        labels={item: pool_labels[item] for item in selected_ids},
    )
    unsigned = {
        "schema_version": 1,
        "kind": "sealed_holdout_authorization",
        "source_study_id": decision.study_id,
        "source_spec_digest": source_spec.spec_digest,
        "holdout_study_id": spec.id,
        "holdout_spec_digest": spec.spec_digest,
        "development_result_digest": decision.development_result_digest,
        "development_qualification_digest": decision.development_qualification_digest,
        "development_preview_digest": decision.preview_digest,
        "advancement_decision_digest": decision.decision_digest,
        "holdout_suite_digest": audit.holdout_suite_digest,
        "holdout_exposure_audit_digest": audit.audit_digest,
        "holdout_exposure_projection_digest": audit.projection_digest,
        "historical_exposure_receipt_digest": (
            audit.historical_exposure_receipt_digest
        ),
        "holdout_pool_fingerprint_digest": audit.pool_fingerprint_digest,
        "holdout_project_coverage_digest": audit.project_coverage_digest,
        "outcome_data_consulted": False,
        "audited_at": audit.audited_at,
        "expires_at": audit.expires_at,
        "sealed_preparation_receipt_digest": preparation["receipt_digest"],
        "zero_model_receipt_digest": zero_model["receipt_digest"],
        "reserve_preparation_receipt_digest": reserve_preparation["receipt_digest"],
        "campaign_allocation_receipt": allocation_path.relative_to(root).as_posix(),
        "campaign_allocation_receipt_digest": allocation["receipt_digest"],
        "selected_task_ids": list(selected_ids),
        "selected_items": selected_items,
        "selected_zero_model_results_digest": stable_digest(
            [zero_model_results[item] for item in sorted(selected_ids)]
        ),
        "activated_reserve_task_ids": sorted(set(selected_ids) & set(reserve_tasks)),
        "tasks_sha256": _sha256(tasks_path),
        "private_labels_sha256": _sha256(labels_path),
        "attempts": 2,
        "logical_cells": 16,
        "arm_labels": [spec.baseline.label, spec.candidate.label],
    }
    authorization = {**unsigned, "receipt_digest": stable_digest(unsigned)}
    _write_private_json(authorization_path, authorization)
    validate_holdout_authorization(
        authorization_path,
        comparison_id=spec.id,
        spec_digest=spec.spec_digest,
        tasks_path=tasks_path,
        private_labels_path=labels_path,
        attempts=spec.execution.attempts,
        repo_root=root,
    )
    return {
        "spec_path": spec_path.relative_to(root).as_posix(),
        "spec_digest": spec.spec_digest,
        "authorization_path": authorization_path.relative_to(root).as_posix(),
        "authorization_digest": authorization["receipt_digest"],
        "allocation_path": allocation_path.relative_to(root).as_posix(),
        "allocation_digest": allocation["receipt_digest"],
        "logical_cells": 16,
    }


def _holdout_authorization_items(
    *,
    root: Path,
    lane_receipt: Mapping[str, Any],
    tasks: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {
        str(item["task_id"]): _mapping(item, "sealed holdout prepared item")
        for item in lane_receipt["items"]
    }
    values: list[dict[str, Any]] = []
    for task_id in sorted(tasks):
        item = by_id.get(task_id)
        if item is None or task_id not in labels:
            raise ValueError("holdout authorization item was not prepared")
        task = tasks[task_id]
        label = labels[task_id]
        if _canonical_digest(task) != item["prepared_task_digest"]:
            raise ValueError("prepared holdout task digest changed")
        if _canonical_digest(label) != item["private_label_digest"]:
            raise ValueError("prepared holdout private label digest changed")
        resources = _sequence(task.get("resources"), "authorized holdout resources")
        if len(resources) != 1:
            raise ValueError("authorized holdout task must have one resource")
        resource = _mapping(resources[0], "authorized holdout resource")
        path = _inside(root, root / str(resource.get("path")))
        if _sha256(path) != item["resource_digest"]:
            raise ValueError("authorized holdout resource digest changed")
        values.append(
            {
                "task_id": task_id,
                "role": item["role"],
                "behavior_family": item["behavior_family"],
                "paired_task_id": item["paired_task_id"],
                "source_task_digest": item["source_task_digest"],
                "prepared_task_digest": item["prepared_task_digest"],
                "private_label_digest": item["private_label_digest"],
                "resource_digest": item["resource_digest"],
                "resource_target": resource.get("target"),
            }
        )
    return values


def allocate_campaign_branch(
    *,
    repo_root: Path,
    lane: str,
    branch: str,
    source_study_id: str,
    followup_study_id: str,
    followup_spec_digest: str,
    logical_cells: int,
    receipt_path: Path,
) -> dict[str, Any]:
    """Allocate exactly one manifest-declared follow-up branch for a lane."""

    if lane not in _LANES or branch not in {"holdout", "no_skill_diagnostic"}:
        raise ValueError("campaign lane or branch is invalid")
    expected_cells = 16 if branch == "holdout" else 4
    if logical_cells != expected_cells:
        raise ValueError("campaign branch logical cell allocation changed")
    root = repo_root.resolve()
    ledger_path = root / _ALLOCATION_LEDGER
    lock_path = root / _ALLOCATION_LOCK
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(lock_path, timeout=120):
        ledger = (
            _json_mapping(ledger_path)
            if ledger_path.is_file()
            else _new_campaign_allocation_ledger()
        )
        _validate_campaign_allocation_ledger(ledger)
        lanes = _mapping(ledger["lanes"], "campaign allocation lanes")
        state = _mapping(lanes[lane], "campaign lane allocation")
        existing = state.get("followup")
        if existing is not None:
            current = _mapping(existing, "campaign follow-up allocation")
            if (
                current.get("branch") != branch
                or current.get("source_study_id") != source_study_id
                or current.get("followup_study_id") != followup_study_id
                or current.get("followup_spec_digest") != followup_spec_digest
            ):
                raise ValueError(
                    "campaign lane already allocated to the mutually exclusive branch"
                )
            receipt = _json_mapping(receipt_path)
            validate_campaign_allocation_receipt(
                receipt,
                comparison_id=followup_study_id,
                spec_digest=followup_spec_digest,
                logical_cells=logical_cells,
                repo_root=root,
            )
            return receipt

        # The campaign has four replacement executions total: one for each
        # development lane and one deterministic contingency assigned by manifest
        # lane order. Diagnostics replace holdout and never receive a retry.
        allowance = int(branch == "holdout" and lane == _LANES[0])
        followup = {
            "branch": branch,
            "source_study_id": source_study_id,
            "followup_study_id": followup_study_id,
            "followup_spec_digest": followup_spec_digest,
            "logical_cells": logical_cells,
            "infrastructure_replacement_allowance": allowance,
        }
        followup["allocation_digest"] = stable_digest(followup)
        state["followup"] = followup
        lanes[lane] = state
        ledger["lanes"] = lanes
        allocated = [
            _mapping(item["followup"], "campaign follow-up")
            for item in lanes.values()
            if isinstance(item, Mapping) and item.get("followup") is not None
        ]
        ledger["logical_cells"] = 48 + sum(int(item["logical_cells"]) for item in allocated)
        ledger["infrastructure_replacement_allowance"] = 3 + sum(
            int(item["infrastructure_replacement_allowance"]) for item in allocated
        )
        ledger["maximum_physical_executions"] = (
            int(ledger["logical_cells"])
            + int(ledger["infrastructure_replacement_allowance"])
        )
        ledger.pop("ledger_digest", None)
        ledger["ledger_digest"] = stable_digest(ledger)
        _validate_campaign_allocation_ledger(ledger)
        _write_private_json(ledger_path, ledger)

        unsigned = {
            "schema_version": 1,
            "kind": "campaign_branch_allocation",
            "campaign_id": "community-skill-selected-v1",
            "lane": lane,
            "branch": branch,
            "source_study_id": source_study_id,
            "followup_study_id": followup_study_id,
            "followup_spec_digest": followup_spec_digest,
            "development_logical_cells": 16,
            "followup_logical_cells": logical_cells,
            "holdout_logical_cells_replaced": (
                16 if branch == "no_skill_diagnostic" else 0
            ),
            "infrastructure_replacement_allowance": allowance,
            "campaign_logical_cells_after": ledger["logical_cells"],
            "campaign_replacement_allowance_after": ledger[
                "infrastructure_replacement_allowance"
            ],
            "campaign_maximum_physical_executions_after": ledger[
                "maximum_physical_executions"
            ],
            "ledger_digest": ledger["ledger_digest"],
            "lane_allocation_digest": followup["allocation_digest"],
        }
        receipt = {**unsigned, "receipt_digest": stable_digest(unsigned)}
        _write_private_json(receipt_path, receipt)
        return receipt


def validate_campaign_allocation_receipt(
    value: Mapping[str, Any],
    *,
    comparison_id: str,
    spec_digest: str,
    logical_cells: int,
    maximum_physical_executions: int | None = None,
    infrastructure_retry_limit: int | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Apply campaign ledger freshness after Fugue's generic receipt validation."""

    receipt = validate_followup_allocation_receipt(
        value,
        comparison_id=comparison_id,
        spec_digest=spec_digest,
        logical_cells=logical_cells,
        maximum_physical_executions=maximum_physical_executions,
        infrastructure_retry_limit=infrastructure_retry_limit,
    )
    branch = receipt["branch"]
    expected_cells = 16 if branch == "holdout" else 4 if branch == "no_skill_diagnostic" else -1
    if (
        logical_cells != expected_cells
        or receipt["development_logical_cells"] != 16
        or receipt["holdout_logical_cells_replaced"]
        != (16 if branch == "no_skill_diagnostic" else 0)
    ):
        raise ValueError("campaign allocation branch semantics changed")
    expected_allowance = int(branch == "holdout" and receipt["lane"] == _LANES[0])
    if receipt["infrastructure_replacement_allowance"] != expected_allowance:
        raise ValueError("campaign allocation replacement allowance changed")
    if repo_root is not None:
        ledger = _json_mapping(repo_root.resolve() / _ALLOCATION_LEDGER)
        _validate_campaign_allocation_ledger(ledger)
        lane = _mapping(ledger["lanes"], "campaign lanes").get(receipt["lane"])
        current = _mapping(
            _mapping(lane, "campaign lane").get("followup"),
            "campaign follow-up",
        )
        if current.get("allocation_digest") != receipt["lane_allocation_digest"]:
            raise ValueError("campaign allocation receipt is stale")
        for field, receipt_field in (
            ("branch", "branch"),
            ("source_study_id", "source_study_id"),
            ("followup_study_id", "followup_study_id"),
            ("followup_spec_digest", "followup_spec_digest"),
            ("logical_cells", "followup_logical_cells"),
            (
                "infrastructure_replacement_allowance",
                "infrastructure_replacement_allowance",
            ),
        ):
            if current.get(field) != receipt[receipt_field]:
                raise ValueError("campaign allocation receipt is not current")
    return receipt


def _new_campaign_allocation_ledger() -> dict[str, Any]:
    unsigned = {
        "schema_version": 1,
        "kind": "campaign_allocation_ledger",
        "campaign_id": "community-skill-selected-v1",
        "logical_cell_ceiling": 96,
        "infrastructure_replacement_ceiling": 4,
        "logical_cells": 48,
        "infrastructure_replacement_allowance": 3,
        "maximum_physical_executions": 51,
        "lanes": {
            lane: {
                "development_logical_cells": 16,
                "development_infrastructure_replacement_allowance": 1,
                "followup": None,
            }
            for lane in _LANES
        },
    }
    return {**unsigned, "ledger_digest": stable_digest(unsigned)}


def _validate_campaign_allocation_ledger(value: Mapping[str, Any]) -> None:
    ledger = dict(value)
    supplied = ledger.pop("ledger_digest", None)
    if supplied != stable_digest(ledger):
        raise ValueError("campaign allocation ledger digest does not match")
    if (
        ledger.get("schema_version") != 1
        or ledger.get("kind") != "campaign_allocation_ledger"
        or ledger.get("campaign_id") != "community-skill-selected-v1"
    ):
        raise ValueError("unsupported campaign allocation ledger")
    lanes = _mapping(ledger.get("lanes"), "campaign allocation lanes")
    if set(lanes) != set(_LANES):
        raise ValueError("campaign allocation lanes changed")
    followups = [
        _mapping(state["followup"], "campaign follow-up")
        for state in lanes.values()
        if isinstance(state, Mapping) and state.get("followup") is not None
    ]
    for lane, state in lanes.items():
        lane_state = _mapping(state, "campaign lane")
        if lane_state.get("development_logical_cells") != 16:
            raise ValueError("campaign development allocation changed")
        followup = lane_state.get("followup")
        if followup is None:
            continue
        branch = _mapping(followup, "campaign follow-up")
        digest = branch.pop("allocation_digest", None)
        if digest != stable_digest(branch):
            raise ValueError("campaign lane allocation digest does not match")
        expected = 16 if branch.get("branch") == "holdout" else 4
        allowance = int(
            branch.get("branch") == "holdout" and lane == _LANES[0]
        )
        if (
            branch.get("logical_cells") != expected
            or branch.get("infrastructure_replacement_allowance") != allowance
        ):
            raise ValueError("campaign follow-up allocation changed")
    logical = 48 + sum(int(item["logical_cells"]) for item in followups)
    replacements = 3 + sum(
        int(item["infrastructure_replacement_allowance"]) for item in followups
    )
    if (
        ledger.get("logical_cell_ceiling") != 96
        or ledger.get("infrastructure_replacement_ceiling") != 4
        or ledger.get("logical_cells") != logical
        or ledger.get("infrastructure_replacement_allowance") != replacements
        or ledger.get("maximum_physical_executions") != logical + replacements
        or logical > 96
        or replacements > 4
    ):
        raise ValueError("campaign allocation ledger exceeds or misstates its ceilings")



def build_no_skill_diagnostic(
    *,
    comparison_path: Path,
    advancement_decision_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Build a four-cell diagnostic that replaces, never adds to, holdout."""

    root = repo_root.resolve()
    source_path = _inside(root, comparison_path.resolve())
    decision = read_study_advancement_decision(advancement_decision_path)
    raw = _yaml_mapping(source_path)
    if decision.status != "run_no_skill_diagnostic":
        raise ValueError("no-Skill diagnostic requires a non-discriminating decision")
    if decision.study_id != raw.get("id"):
        raise ValueError("advancement decision belongs to another Study")
    schedule = _mapping(_mapping(raw["execution"], "execution")["schedule"], "schedule")
    stages = _sequence(schedule["stages"], "schedule stages")
    checkpoint = next(
        (
            _mapping(stage, "checkpoint stage")
            for stage in stages
            if stage.get("id") == "checkpoint"
        ),
        None,
    )
    if checkpoint is None or len(checkpoint.get("task_ids", [])) != 2:
        raise ValueError("diagnostic source must have exactly two checkpoint tasks")
    checkpoint_ids = tuple(checkpoint["task_ids"])
    source_tasks = _jsonl_by_id(
        root / _portable_source(raw["taskset"]["tasks"], source_path, root)
    )
    source_labels = _jsonl_by_id(
        root / _portable_source(raw["taskset"]["private_labels"], source_path, root)
    )
    if not set(checkpoint_ids) <= set(source_tasks) or not set(checkpoint_ids) <= set(
        source_labels
    ):
        raise ValueError("diagnostic checkpoint tasks or labels are missing")
    lane = source_path.parent.name
    target = (
        root
        / ".fugue/private/community-skill-selected-v1"
        / lane
        / "no-skill-diagnostic"
    )
    tasks_path = target / "tasks.jsonl"
    labels_path = target / "private-labels.jsonl"
    _write_private_jsonl(tasks_path, [source_tasks[item] for item in checkpoint_ids])
    _write_private_jsonl(labels_path, [source_labels[item] for item in checkpoint_ids])
    spec, spec_path = _materialize_followup_spec(
        raw=raw,
        source_path=source_path,
        target=target,
        root=root,
        study_id=f"{lane}-candidate-vs-no-skill-diagnostic-v1",
        question=(
            "Does the exact candidate Skill outperform the same fixed agent with "
            "no assigned Skill on the two preregistered checkpoint tasks?"
        ),
        tasks_path=tasks_path,
        labels_path=labels_path,
        task_ids=checkpoint_ids,
        attempts=1,
        stage_id="candidate-vs-no-skill-diagnostic",
        retry_allowance=0,
        baseline={"label": "No Skill control", "skills": []},
    )
    allocation_path = target / "campaign-allocation.json"
    allocation = allocate_campaign_branch(
        repo_root=root,
        lane=lane,
        branch="no_skill_diagnostic",
        source_study_id=decision.study_id,
        followup_study_id=spec.id,
        followup_spec_digest=spec.spec_digest,
        logical_cells=4,
        receipt_path=allocation_path,
    )
    unsigned = {
        "schema_version": 1,
        "kind": "no_skill_diagnostic_lock",
        "source_study_id": decision.study_id,
        "advancement_decision_digest": decision.decision_digest,
        "diagnostic_study_id": spec.id,
        "diagnostic_spec_digest": spec.spec_digest,
        "task_ids": list(checkpoint_ids),
        "logical_cells": 4,
        "holdout_logical_cells_replaced": 16,
        "holdout_logical_cells_admitted": 0,
        "allocation_action": "replaces_holdout",
        "campaign_allocation_receipt_digest": allocation["receipt_digest"],
    }
    receipt = {**unsigned, "lock_digest": stable_digest(unsigned)}
    _write_private_json(target / "diagnostic.lock.json", receipt)
    return receipt


def write_live_holdout_audits(
    *, repo_root: Path, audits: Mapping[str, HoldoutExposureAuditV1]
) -> dict[str, str]:
    root = repo_root.resolve()
    paths: dict[str, str] = {}
    for lane, audit in audits.items():
        path = (
            root
            / ".fugue/private/community-skill-selected-v1"
            / lane
            / "holdout-exposure-audit.json"
        )
        write_holdout_exposure_audit(path, audit)
        path.chmod(0o600)
        paths[lane] = path.relative_to(root).as_posix()
    return paths


_TASK_ID_PATHS = (
    ("attributes", "fugue", "task_id"),
    ("attributes", "fugue.task_id"),
    ("inputs", "task_id"),
    ("inputs", "task_name"),
    ("inputs", "example", "task_id"),
    ("attributes.fugue.task_id",),
    ("inputs.task_id",),
    ("inputs.task_name",),
    ("inputs.example.task_id",),
)
_INPUT_PATHS = (
    ("inputs", "example", "input"),
    ("inputs", "input"),
    ("inputs.example.input",),
    ("inputs.input",),
)
_RESOURCE_PATHS = (
    ("inputs", "example", "resources"),
    ("inputs", "resources"),
    ("inputs.example.resources",),
    ("inputs.resources",),
)


def _safe_identity_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    _reject_outcome_bearing_projection(row)
    task_ids = {
        value
        for path in _TASK_ID_PATHS
        for value in (_nested_text(row, *path),)
        if value
    }
    attributes = row.get("attributes")
    fingerprints = {
        "prompt_fingerprints": _declared_digests(
            row,
            attributes,
            nested="task_prompt_sha256",
            flattened="attributes.fugue.task_prompt_sha256",
        ),
        "input_fingerprints": _declared_digests(
            row,
            attributes,
            nested="task_input_sha256",
            flattened="attributes.fugue.task_input_sha256",
        ),
        "resource_fingerprints": _declared_digests(
            row,
            attributes,
            nested="task_resource_sha256",
            flattened="attributes.fugue.task_resource_sha256",
        ),
    }
    for path in _INPUT_PATHS:
        candidate = _nested_value(row, *path)
        if isinstance(candidate, Mapping):
            safe_input = dict(candidate)
            fingerprints["input_fingerprints"].add(stable_digest(safe_input))
            question = safe_input.get("question")
            if isinstance(question, str) and question.strip():
                fingerprints["prompt_fingerprints"].add(
                    stable_digest({"prompt": question.strip()})
                )
    for path in _RESOURCE_PATHS:
        candidate = _nested_value(row, *path)
        if isinstance(candidate, list):
            fingerprints["resource_fingerprints"].add(stable_digest(candidate))
    return {
        "id": str(row.get("id") or ""),
        "op_name": str(row.get("op_name") or ""),
        "task_ids": sorted(task_ids),
        **{key: sorted(values) for key, values in fingerprints.items()},
    }


def _declared_digests(
    row: Mapping[str, Any],
    attributes: Any,
    *,
    nested: str,
    flattened: str,
) -> set[str]:
    values = {
        _nested_text(attributes, "fugue", nested),
        _nested_text(attributes, f"fugue.{nested}"),
        _nested_text(row, flattened),
    } - {""}
    for item in values:
        _digest(item, f"safe exposure projection {nested}")
    return values


def _nested_value(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _reject_outcome_bearing_projection(value: Any, *, path: str = "row") -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            tokens = set(
                str(raw_key).casefold().replace("-", "_").replace(".", "_").split("_")
            )
            if tokens & _FORBIDDEN_EXPOSURE_KEYS:
                raise ValueError(
                    f"holdout exposure projection retained outcome field: {path}.{raw_key}"
                )
            _reject_outcome_bearing_projection(item, path=f"{path}.{raw_key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_outcome_bearing_projection(item, path=f"{path}[{index}]")


_COVERAGE_FIELDS = {
    "project_ref",
    "project_status",
    "returned_call_count",
    "query_limit",
    "projection_digest",
    "truncated",
    "complete",
}


def _project_coverage(
    *,
    project: str,
    project_status: str,
    returned_call_count: int,
    query_limit: int,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "project_ref": project,
        "project_status": project_status,
        "returned_call_count": returned_call_count,
        "query_limit": query_limit,
        "projection_digest": stable_digest([dict(item) for item in rows]),
        "truncated": False,
        "complete": True,
    }


def _validate_project_coverage(
    *,
    projects: Sequence[str],
    project_coverage: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if set(project_coverage) != set(projects):
        raise ValueError("holdout audit did not cover every historical project")
    values: list[dict[str, Any]] = []
    for project in projects:
        row = _mapping(project_coverage[project], "historical project coverage")
        returned, limit = row.get("returned_call_count"), row.get("query_limit")
        if (
            set(row) != _COVERAGE_FIELDS
            or row.get("project_ref") != project
            or row.get("project_status") not in {"present", "absent"}
            or type(returned) is not int
            or returned < 0
            or type(limit) is not int
            or not 1 <= limit
            or returned > limit
            or row.get("truncated") is not False
            or row.get("complete") is not True
            or (row.get("project_status") == "absent" and returned != 0)
        ):
            raise ValueError("historical project query was incomplete or truncated")
        _digest(row.get("projection_digest"), "historical project projection digest")
        values.append(row)
    return values


def _coverage_projection_digest(coverage: Sequence[Mapping[str, Any]]) -> str:
    return stable_digest(
        {str(item["project_ref"]): str(item["projection_digest"]) for item in coverage}
    )


def _validated_exposure_projection(
    *,
    projects: Sequence[str],
    project_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    project_coverage: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    if set(project_rows) != set(projects):
        raise ValueError("live holdout audit did not query every historical project")
    coverage = _validate_project_coverage(
        projects=projects, project_coverage=project_coverage
    )
    coverage_by_project = {item["project_ref"]: item for item in coverage}
    safe_rows: dict[str, list[dict[str, Any]]] = {}
    list_fields = tuple(_SAFE_PROJECTION_FIELDS - {"id", "op_name"})
    digest_fields = tuple(field for field in list_fields if field != "task_ids")
    for project in projects:
        rows = sorted((dict(row) for row in project_rows[project]), key=_canonical_json)
        for row in rows:
            if set(row) != _SAFE_PROJECTION_FIELDS or any(
                not isinstance(row.get(key), list) for key in list_fields
            ):
                raise ValueError(f"holdout audit retained a non-safe field for {project}")
            if tuple(row["task_ids"]) != _sorted_unique_text(
                row["task_ids"], "holdout audit task id"
            ):
                raise ValueError("holdout audit task identities must be canonical")
            for key in digest_fields:
                if row[key] != sorted(set(row[key])):
                    raise ValueError("holdout audit fingerprints must be canonical")
                for digest in row[key]:
                    _digest(digest, f"holdout audit {key}")
            for key in ("id", "op_name"):
                if row[key]:
                    _text(row[key], f"holdout audit {key}")
        recorded = coverage_by_project[project]
        if (
            len(rows) != recorded["returned_call_count"]
            or stable_digest(rows) != recorded["projection_digest"]
        ):
            raise ValueError("holdout audit coverage disagrees with projection")
        safe_rows[project] = rows
    return safe_rows, coverage


def _match_pool_fingerprints(
    *,
    pool: Sequence[Mapping[str, Any]],
    project_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for item in pool:
        reasons: set[str] = set()
        projects: set[str] = set()
        for project, rows in project_rows.items():
            for row in rows:
                row_reasons = {
                    reason
                    for reason, matched in (
                        ("task_id", item["task_id"] in row["task_ids"]),
                        (
                            "prompt_fingerprint",
                            item["prompt_fingerprint"] in row["prompt_fingerprints"],
                        ),
                        (
                            "input_fingerprint",
                            item["input_fingerprint"] in row["input_fingerprints"],
                        ),
                    )
                    if matched
                }
                if row_reasons:
                    reasons.update(row_reasons)
                    projects.add(project)
        if reasons:
            matches.append(
                {
                    **{
                        key: item[key]
                        for key in ("lane_id", "task_id", "role", "behavior_family")
                    },
                    "matched_by": sorted(reasons),
                    "project_refs": sorted(projects),
                }
            )
    return sorted(matches, key=lambda item: item["task_id"])


_MATCH_FIELDS = {
    "lane_id",
    "task_id",
    "role",
    "behavior_family",
    "matched_by",
    "project_refs",
}


def _validate_recorded_matches(
    value: Any, *, pool: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {str(item["task_id"]): item for item in pool}
    canonical: list[dict[str, Any]] = []
    for raw in _sequence(value, "historical exposure matches"):
        item = _mapping(raw, "historical exposure match")
        task_id = _text(item.get("task_id"), "historically exposed task id")
        source = by_id.get(task_id)
        reasons = _sorted_unique_text(item.get("matched_by"), "exposure match reason")
        projects = _sorted_unique_text(
            item.get("project_refs"), "exposure match project"
        )
        if (
            set(item) != _MATCH_FIELDS
            or source is None
            or any(existing["task_id"] == task_id for existing in canonical)
            or not reasons
            or not projects
            or not set(reasons)
            <= {"task_id", "prompt_fingerprint", "input_fingerprint"}
            or any(
                item.get(key) != source.get(key)
                for key in ("lane_id", "role", "behavior_family")
            )
        ):
            raise ValueError("historical exposure match is invalid or changed")
        canonical.append(item)
    if canonical != sorted(canonical, key=lambda item: item["task_id"]):
        raise ValueError("historical exposure matches must be sorted")
    return canonical
def _decode_stream(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return []
    if stripped[:1] in {"[", "{"}:
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, Mapping)]
        if isinstance(value, Mapping) and isinstance(value.get("calls"), list):
            return [dict(item) for item in value["calls"] if isinstance(item, Mapping)]
    return [
        dict(item)
        for line in stripped.splitlines()
        for item in (json.loads(line),)
        if isinstance(item, Mapping)
    ]


def _relativize_evaluator_inputs(
    raw: dict[str, Any],
    *,
    source_path: Path,
    target: Path,
    root: Path,
) -> None:
    for evaluator in raw["evaluators"]:
        for key in ("scorer", "calibration", "calibration_rubric"):
            if evaluator.get(key):
                evaluator[key] = os.path.relpath(
                    root / _portable_source(evaluator[key], source_path, root),
                    target,
                )
        verifier = evaluator.get("verifier")
        if isinstance(verifier, dict):
            for key in ("source", "runtime_lock"):
                verifier[key] = os.path.relpath(
                    root / _portable_source(verifier[key], source_path, root),
                    target,
                )


def _portable_source(value: Any, source_path: Path, root: Path) -> str:
    path = Path(_text(value, "portable source path"))
    resolved = (source_path.parent / path).resolve()
    return _inside(root, resolved).relative_to(root).as_posix()


def _nested_text(value: Any, *keys: str) -> str:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return ""
        current = current.get(key)
    return current.strip() if isinstance(current, str) else ""


def _jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing regular JSONL input: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(
        isinstance(row, dict) and isinstance(row.get("id"), str) for row in rows
    ):
        raise ValueError(f"invalid JSONL rows: {path}")
    values = {row["id"]: row for row in rows}
    if len(values) != len(rows):
        raise ValueError(f"duplicate JSONL identity: {path}")
    return values


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_private_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_private_text(
        path,
        "".join(
            json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
    )


def _write_private_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_private_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_private_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        temporary.chmod(0o600)
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_private(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(source, temporary)
        temporary.chmod(0o600)
        os.replace(temporary, target)
        target.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _host_verifier_receipt(
    root: Path,
    lane: str,
    task: Mapping[str, Any],
    output: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    lane_root = root / _CAMPAIGN / lane
    comparison = _yaml_mapping(lane_root / "comparison.yaml")
    deterministic = next(
        item for item in comparison["evaluators"] if item["type"] == "deterministic"
    )
    declaration = _mapping(deterministic.get("verifier"), "host verifier")
    verifier = _inside(lane_root, lane_root / declaration["source"])
    runtime_lock = _json_mapping(
        _inside(lane_root, lane_root / declaration["runtime_lock"])
    )
    if _sha256(verifier) != runtime_lock.get("verifier_source_sha256"):
        raise RuntimeError(f"host verifier source lock changed: {lane}")
    runtime_digest = stable_digest(runtime_lock)
    declared_digest = expected.get("verifier_runtime_lock_digest")
    if declared_digest is not None and declared_digest != runtime_digest:
        raise RuntimeError(f"host verifier runtime lock changed: {task['id']}")
    archive = root / task["resources"][0]["path"]
    with tempfile.TemporaryDirectory(prefix="fugue-sealed-verifier-") as raw:
        temporary = Path(raw)
        input_root = temporary / "input"
        workspace = temporary / "work"
        input_root.mkdir()
        workspace.mkdir()
        archive_input = input_root / "task.tar"
        output_input = input_root / "agent-output.json"
        archive_input.write_bytes(archive.read_bytes())
        payload = (_canonical_json(output) + "\n").encode()
        output_input.write_bytes(payload)
        config = {
            "schema_version": 1,
            "task_id": task["id"],
            "task_archive": {
                "path": str(archive_input),
                "sha256": _sha256(archive_input),
            },
            "agent_output": {
                "path": str(output_input),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            "runtime_lock_digest": runtime_digest,
            "workspace": str(workspace),
            "allowed_paths": expected["allowed_paths"],
        }
        config_path = input_root / "input.json"
        config_path.write_text(_canonical_json(config), encoding="utf-8")
        run = subprocess.run(
            ("node", str(verifier), str(config_path)),
            check=False,
            capture_output=True,
            text=True,
            timeout=40,
        )
    receipt = json.loads(run.stdout)
    if not isinstance(receipt, dict):
        raise RuntimeError(f"Node verifier returned invalid receipt: {task['id']}")
    return receipt, run.returncode


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load scorer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _safe_relative(value: Any, label: str) -> str:
    text = _text(value, label)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or text != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be a safe relative path")
    return text


def _inside(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("operator input escapes its declared root")
    return resolved


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_mapping(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular JSON file: {path}")
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _yaml_mapping(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), str(path))


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _sorted_unique_text(value: Any, label: str) -> tuple[str, ...]:
    items = tuple(_text(item, label) for item in _sequence(value, label))
    if items != tuple(sorted(items)) or len(items) != len(set(items)):
        raise ValueError(f"{label} values must be sorted and unique")
    return items


def _digest(value: Any, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or any(char not in _DIGEST for char in text):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return text


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 2_000:
        raise ValueError(f"{label} must be nonempty bounded text")
    return value.strip()


def _instant(value: Any, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(_text(value, label).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 instant") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed.astimezone(UTC)
