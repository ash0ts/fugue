from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fugue.bench.candidates import stable_digest

_DIGEST = frozenset("0123456789abcdef")


def validate_followup_allocation_receipt(
    value: Mapping[str, Any],
    *,
    comparison_id: str,
    spec_digest: str,
    logical_cells: int,
    maximum_physical_executions: int | None = None,
    infrastructure_retry_limit: int | None = None,
) -> dict[str, Any]:
    """Validate a legacy-compatible allocation receipt without campaign policy.

    The persisted V1 wire names include ``campaign_*`` for compatibility.  Core
    validates their integrity and their agreement with the approved comparison;
    lane membership, branch choice, and campaign-wide ceilings belong to the
    operator that issued the receipt.
    """

    receipt = dict(value)
    expected = {
        "schema_version",
        "kind",
        "campaign_id",
        "lane",
        "branch",
        "source_study_id",
        "followup_study_id",
        "followup_spec_digest",
        "development_logical_cells",
        "followup_logical_cells",
        "holdout_logical_cells_replaced",
        "infrastructure_replacement_allowance",
        "campaign_logical_cells_after",
        "campaign_replacement_allowance_after",
        "campaign_maximum_physical_executions_after",
        "ledger_digest",
        "lane_allocation_digest",
        "receipt_digest",
    }
    if set(receipt) != expected:
        raise ValueError("follow-up allocation receipt fields do not match")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind") != "campaign_branch_allocation"
    ):
        raise ValueError("unsupported follow-up allocation receipt")
    supplied = receipt.pop("receipt_digest", None)
    if supplied != stable_digest(receipt):
        raise ValueError("follow-up allocation receipt digest does not match")
    receipt["receipt_digest"] = supplied
    for key in (
        "campaign_id",
        "lane",
        "branch",
        "source_study_id",
        "followup_study_id",
    ):
        _text(receipt.get(key), key)
    for key in (
        "followup_spec_digest",
        "ledger_digest",
        "lane_allocation_digest",
    ):
        _digest(receipt.get(key), key)
    if (
        receipt["followup_study_id"] != comparison_id
        or receipt["followup_spec_digest"] != spec_digest
    ):
        raise ValueError("follow-up allocation belongs to another comparison")
    numeric = {
        key: _non_negative_int(receipt.get(key), key)
        for key in (
            "development_logical_cells",
            "followup_logical_cells",
            "holdout_logical_cells_replaced",
            "infrastructure_replacement_allowance",
            "campaign_logical_cells_after",
            "campaign_replacement_allowance_after",
            "campaign_maximum_physical_executions_after",
        )
    }
    if numeric["followup_logical_cells"] != logical_cells:
        raise ValueError("follow-up allocation logical cells changed")
    if (
        numeric["holdout_logical_cells_replaced"]
        > numeric["development_logical_cells"]
    ):
        raise ValueError("follow-up allocation replaces unavailable logical cells")
    if (
        numeric["campaign_maximum_physical_executions_after"]
        != numeric["campaign_logical_cells_after"]
        + numeric["campaign_replacement_allowance_after"]
    ):
        raise ValueError("follow-up allocation campaign totals are inconsistent")
    allowance = numeric["infrastructure_replacement_allowance"]
    if maximum_physical_executions is not None and (
        maximum_physical_executions != logical_cells + allowance
    ):
        raise ValueError("comparison physical ceiling disagrees with its allocation")
    if infrastructure_retry_limit is not None and (
        infrastructure_retry_limit != allowance
    ):
        raise ValueError("comparison retry policy disagrees with its allocation")
    return receipt


def validate_holdout_authorization(
    path: Path,
    *,
    comparison_id: str,
    spec_digest: str,
    tasks_path: Path,
    private_labels_path: Path,
    attempts: int,
    repo_root: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate an immutable follow-up authorization at Fugue's trust boundary.

    This validator deliberately knows nothing about repositories, lanes, reserve
    selection policy, historical projects, or fixed matrix sizes.  It verifies
    the supplied task/label/spec/decision/freshness digests and derives the
    logical cell count from the materialized inputs and declared arm labels.
    """

    value = _json_mapping(path)
    expected = {
        "schema_version",
        "kind",
        "source_study_id",
        "source_spec_digest",
        "holdout_study_id",
        "holdout_spec_digest",
        "development_result_digest",
        "development_qualification_digest",
        "development_preview_digest",
        "advancement_decision_digest",
        "holdout_suite_digest",
        "holdout_exposure_audit_digest",
        "holdout_exposure_projection_digest",
        "historical_exposure_receipt_digest",
        "holdout_pool_fingerprint_digest",
        "holdout_project_coverage_digest",
        "outcome_data_consulted",
        "audited_at",
        "expires_at",
        "sealed_preparation_receipt_digest",
        "zero_model_receipt_digest",
        "reserve_preparation_receipt_digest",
        "campaign_allocation_receipt",
        "campaign_allocation_receipt_digest",
        "selected_task_ids",
        "selected_items",
        "selected_zero_model_results_digest",
        "activated_reserve_task_ids",
        "tasks_sha256",
        "private_labels_sha256",
        "attempts",
        "logical_cells",
        "arm_labels",
        "receipt_digest",
    }
    if set(value) != expected:
        raise ValueError("holdout authorization fields do not match")
    if value.get("schema_version") != 1 or value.get("kind") != (
        "sealed_holdout_authorization"
    ):
        raise ValueError("unsupported holdout authorization")
    supplied = value.pop("receipt_digest", None)
    if supplied != stable_digest(value):
        raise ValueError("holdout authorization digest does not match")
    value["receipt_digest"] = supplied
    if (
        value.get("holdout_study_id") != comparison_id
        or value.get("holdout_spec_digest") != spec_digest
    ):
        raise ValueError("holdout authorization belongs to another comparison")
    for key in (
        "source_spec_digest",
        "holdout_spec_digest",
        "development_result_digest",
        "development_qualification_digest",
        "development_preview_digest",
        "advancement_decision_digest",
        "holdout_suite_digest",
        "holdout_exposure_audit_digest",
        "holdout_exposure_projection_digest",
        "historical_exposure_receipt_digest",
        "holdout_pool_fingerprint_digest",
        "holdout_project_coverage_digest",
        "sealed_preparation_receipt_digest",
        "zero_model_receipt_digest",
        "reserve_preparation_receipt_digest",
        "campaign_allocation_receipt_digest",
        "selected_zero_model_results_digest",
        "tasks_sha256",
        "private_labels_sha256",
    ):
        _digest(value.get(key), key)
    _text(value.get("source_study_id"), "source study id")
    _text(value.get("holdout_study_id"), "holdout study id")
    _text(value.get("campaign_allocation_receipt"), "allocation receipt path")
    if value.get("outcome_data_consulted") is not False:
        raise ValueError("holdout authorization consulted treatment outcomes")

    selected = _unique_text(value.get("selected_task_ids"), "holdout task id")
    tasks = _jsonl_by_id(tasks_path)
    labels = _jsonl_by_id(private_labels_path)
    if not selected or set(selected) != set(tasks) or set(selected) != set(labels):
        raise ValueError("holdout authorization task identities changed")
    if value["tasks_sha256"] != _sha256(tasks_path) or value[
        "private_labels_sha256"
    ] != _sha256(private_labels_path):
        raise ValueError("holdout authorization task inputs changed")
    _validate_selected_items(
        root=repo_root.resolve(),
        tasks=tasks,
        labels=labels,
        supplied=value.get("selected_items"),
    )
    _unique_text(
        value.get("activated_reserve_task_ids"),
        "activated task id",
        allow_empty=True,
    )

    bound_attempts = _positive_int(value.get("attempts"), "holdout attempts")
    if bound_attempts != attempts:
        raise ValueError("holdout authorization attempt count changed")
    arm_labels = _unique_text(value.get("arm_labels"), "holdout arm label")
    expected_cells = len(selected) * len(arm_labels) * attempts
    if _positive_int(value.get("logical_cells"), "logical cells") != expected_cells:
        raise ValueError("holdout authorization logical cells changed")
    audited_at = _instant(value.get("audited_at"), "holdout authorization audited_at")
    expires_at = _instant(value.get("expires_at"), "holdout authorization expires_at")
    if expires_at <= audited_at:
        raise ValueError("holdout authorization freshness window is invalid")
    instant = (now or datetime.now(UTC)).astimezone(UTC)
    if instant >= expires_at:
        raise ValueError("holdout exposure audit expired before preview")
    return value


def _validate_selected_items(
    *,
    root: Path,
    tasks: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    supplied: Any,
) -> None:
    values = _sequence(supplied, "holdout selected items")
    by_id: dict[str, Mapping[str, Any]] = {}
    required = {
        "task_id",
        "role",
        "behavior_family",
        "paired_task_id",
        "source_task_digest",
        "prepared_task_digest",
        "private_label_digest",
        "resource_digest",
        "resource_target",
    }
    for raw in values:
        item = _mapping(raw, "authorized selected item")
        if set(item) != required:
            raise ValueError("authorized selected item fields do not match")
        task_id = _text(item.get("task_id"), "authorized selected task id")
        if task_id in by_id:
            raise ValueError("authorized selected task identity is duplicated")
        by_id[task_id] = item
    if set(by_id) != set(tasks) or set(by_id) != set(labels):
        raise ValueError("authorized selected item identities changed")
    for task_id, task in tasks.items():
        item = by_id[task_id]
        if item.get("prepared_task_digest") != stable_digest(task):
            raise ValueError("selected item digests changed")
        if item.get("private_label_digest") != stable_digest(labels[task_id]):
            raise ValueError("selected item digests changed")
        resources = _sequence(task.get("resources"), "authorized holdout resources")
        if len(resources) != 1:
            raise ValueError("authorized holdout task must have one resource")
        resource = _mapping(resources[0], "authorized holdout resource")
        path = _inside(root, root / _text(resource.get("path"), "resource path"))
        if item.get("resource_digest") != _sha256(path):
            raise ValueError("selected item digests changed")
        if item.get("resource_target") != resource.get("target"):
            raise ValueError("authorized holdout resource target changed")


def _json_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(value, str(path))


def _jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = _mapping(json.loads(line), f"{path}:{line_number}")
        identity = _text(value.get("id"), f"{path}:{line_number} id")
        if identity in values:
            raise ValueError(f"duplicate identity in {path}: {identity}")
        values[identity] = value
    if not values:
        raise ValueError(f"{path} is empty")
    return values


def _inside(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("authorized resource escapes the repository") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError("authorized resource must be a regular file")
    return resolved


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be an array")
    return list(value)


def _unique_text(value: Any, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    values = tuple(_text(item, label) for item in _sequence(value, label))
    if (not values and not allow_empty) or len(values) != len(set(values)):
        raise ValueError(f"{label} values must be non-empty and unique")
    return values


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 2_000:
        raise ValueError(f"{label} must be non-empty text")
    return value


def _digest(value: Any, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or any(character not in _DIGEST for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _non_negative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _positive_int(value: Any, label: str) -> int:
    result = _non_negative_int(value, label)
    if result < 1:
        raise ValueError(f"{label} must be positive")
    return result


def _instant(value: Any, label: str) -> datetime:
    text = _text(value, label)
    try:
        instant = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be ISO-8601") from exc
    if instant.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return instant.astimezone(UTC)
