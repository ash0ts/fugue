"""Fail-closed validation for the powered Superpowers sampling frame.

This module validates a *completed* prospective sampling-frame bundle.  It does
not discover repositories, author tasks, or infer public provenance.  Those
trusted preparation activities must happen before this validator is invoked.
Trials consume only the immutable archives and digests accepted here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

SCHEMA_VERSION = 1
FRAME_ID = "superpowers-writing-plans-conference-sampling-frame-v1"
DEVELOPMENT_STUDY_ID = "superpowers-writing-plans-conference-development-v1"
HOLDOUT_STUDY_ID = "superpowers-writing-plans-conference-holdout-v1"
EXPECTED_SKILL_REPOSITORY = "https://github.com/obra/superpowers"
EXPECTED_BASELINE_COMMIT = "de4672b171213a6ff6960228d8b95c46ea0b09f4"
EXPECTED_CANDIDATE_COMMIT = "8e1262a3bae92b640d87fa81c51c53b65e490590"

EXPECTED_STRATA: dict[str, dict[str, int]] = {
    "development": {
        "security_privacy": 8,
        "identity_migration": 8,
        "lifecycle_recovery": 8,
        "integration_runtime": 8,
    },
    "target_holdout": {
        "security_privacy": 48,
        "identity_migration": 48,
        "lifecycle_recovery": 48,
        "integration_runtime": 48,
    },
    "safety_control": {
        "docs_only": 16,
        "unsafe_request": 16,
        "no_change_required": 16,
        "single_surface": 16,
    },
}
EXPECTED_COUNTS = {
    partition: sum(strata.values()) for partition, strata in EXPECTED_STRATA.items()
}
EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_TASK_ID = re.compile(r"^sp-(?:dev|target|control)-[a-z0-9][a-z0-9-]{2,95}$")
_REPOSITORY_URL = re.compile(
    r"^https://github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)$"
)
_SPDX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+-]{0,63}$")
_RECORD_KINDS = {"issue", "pull_request", "commit"}

_TOP_LEVEL_KEYS = {
    "schema_version",
    "frame_id",
    "frame_digest",
    "reserved_studies",
    "skill_treatment",
    "population_snapshot",
    "selection_protocol",
    "public_tasks",
    "private_labels",
    "entries",
}
_STUDY_KEYS = {"development", "holdout"}
_SKILL_KEYS = {
    "repository",
    "path",
    "lock_path",
    "lock_sha256",
    "baseline_commit",
    "baseline_bundle_sha256",
    "candidate_commit",
    "candidate_bundle_sha256",
}
_POPULATION_KEYS = {
    "selection_cutoff_utc",
    "records_path",
    "records_sha256",
    "record_count",
    "query_receipt_path",
    "query_receipt_sha256",
    "eligibility_protocol_path",
    "eligibility_protocol_sha256",
    "selection_code_path",
    "selection_code_sha256",
}
_SELECTION_KEYS = {
    "seed_sha256",
    "algorithm",
    "cluster_unit",
    "unique_clusters_across_partitions",
    "strata",
}
_FILE_MANIFEST_KEYS = {"path", "sha256", "count"}
_PRIVATE_MANIFEST_KEYS = {
    "path",
    "sha256",
    "count",
    "host_only",
    "required_mode",
}
_ENTRY_KEYS = {
    "task_id",
    "partition",
    "stratum",
    "selection_rank",
    "repository_url",
    "repository_owner",
    "repository_name",
    "commit_sha",
    "tree_sha",
    "public_record_url",
    "public_record_kind",
    "public_record_id",
    "provenance_cluster_id",
    "population_record_digest",
    "selection_score_sha256",
    "source_archive_path",
    "source_archive_sha256",
    "source_manifest_path",
    "source_manifest_sha256",
    "provenance_receipt_path",
    "provenance_receipt_sha256",
    "public_task_digest",
    "private_label_digest",
    "license",
}
_LICENSE_KEYS = {"spdx_id", "path", "sha256"}
_POPULATION_RECORD_KEYS = {
    "schema_version",
    "record_id",
    "repository_url",
    "repository_owner",
    "repository_name",
    "commit_sha",
    "tree_sha",
    "public_record_url",
    "public_record_kind",
    "public_record_id",
    "public_record_snapshot_sha256",
    "provenance_cluster_id",
    "sampling_family",
    "stratum",
    "selection_score_sha256",
    "source_archive_path",
    "source_archive_sha256",
    "source_manifest_path",
    "source_manifest_sha256",
    "provenance_receipt_path",
    "provenance_receipt_sha256",
    "eligibility_receipt_sha256",
    "license",
}


class SamplingFrameError(ValueError):
    """Raised when a completed frame is not safe to use."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def stable_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SamplingFrameError(f"{field} must be an object")
    return value


def _require_exact_keys(value: dict[str, Any], expected: set[str], *, field: str) -> None:
    keys = set(value)
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        raise SamplingFrameError(
            f"{field} has invalid keys; missing={missing}, unknown={unknown}"
        )


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SamplingFrameError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_git_sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _GIT_SHA.fullmatch(value) is None:
        raise SamplingFrameError(f"{field} must be a full lowercase Git SHA")
    return value


def _require_safe_relative_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SamplingFrameError(f"{field} must be a non-empty repository-relative path")
    path = PurePosixPath(value)
    raw_parts = value.split("/")
    if (
        path.is_absolute()
        or "\\" in value
        or "\x00" in value
        or "\n" in value
        or "\r" in value
        or any(part in {"", ".", ".."} for part in raw_parts)
    ):
        raise SamplingFrameError(f"{field} is not a safe repository-relative path")
    return value


def _require_positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SamplingFrameError(f"{field} must be a positive integer")
    return value


def _normalized_repository_url(value: object, *, field: str) -> tuple[str, str, str]:
    if not isinstance(value, str):
        raise SamplingFrameError(f"{field} must be a GitHub repository URL")
    match = _REPOSITORY_URL.fullmatch(value)
    if match is None or value.endswith(".git"):
        raise SamplingFrameError(f"{field} must be canonical https://github.com/OWNER/REPO")
    owner = match.group("owner")
    repository = match.group("repo")
    return value, owner, repository


def _validate_public_record(
    *, repository_url: str, kind: object, record_url: object, record_id: object
) -> None:
    if kind not in _RECORD_KINDS:
        raise SamplingFrameError("entry.public_record_kind is unsupported")
    if not isinstance(record_url, str) or not isinstance(record_id, str) or not record_id:
        raise SamplingFrameError("entry public record URL and ID must be non-empty strings")
    parsed = urlsplit(record_url)
    if parsed.scheme != "https" or parsed.netloc != "github.com":
        raise SamplingFrameError("entry.public_record_url must be an HTTPS GitHub URL")
    prefix = f"{repository_url}/"
    if not record_url.startswith(prefix) or parsed.query or parsed.fragment:
        raise SamplingFrameError("entry.public_record_url must belong to its repository")
    suffix = record_url[len(prefix) :]
    if kind == "issue":
        expected = f"issues/{record_id}"
        if not record_id.isdigit() or suffix != expected:
            raise SamplingFrameError("issue provenance URL/ID mismatch")
    elif kind == "pull_request":
        expected = f"pull/{record_id}"
        if not record_id.isdigit() or suffix != expected:
            raise SamplingFrameError("pull-request provenance URL/ID mismatch")
    else:
        _require_git_sha(record_id, field="entry.public_record_id")
        if suffix != f"commit/{record_id}":
            raise SamplingFrameError("commit provenance URL/ID mismatch")


def _selection_score(seed: str, repository_url: str, public_record_url: str) -> str:
    payload = f"{seed}\n{repository_url.casefold()}\n{public_record_url}\n".encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_population_record(record: object, *, seed: str) -> dict[str, Any]:
    row = _require_mapping(record, field="population record")
    _require_exact_keys(row, _POPULATION_RECORD_KEYS, field="population record")
    if row["schema_version"] != 1:
        raise SamplingFrameError("population record schema version is unsupported")
    if (
        not isinstance(row["record_id"], str)
        or re.fullmatch(r"sp-pop-[0-9a-f]{16}", row["record_id"]) is None
    ):
        raise SamplingFrameError("population record identity is invalid")
    repository_url, owner, repository = _normalized_repository_url(
        row["repository_url"], field="population_record.repository_url"
    )
    if row["repository_owner"] != owner or row["repository_name"] != repository:
        raise SamplingFrameError("population record owner/name do not match repository_url")
    cluster = f"github:{owner.casefold()}/{repository.casefold()}"
    if row["provenance_cluster_id"] != cluster:
        raise SamplingFrameError("population record cluster is not its canonical repository")
    _require_git_sha(row["commit_sha"], field="population_record.commit_sha")
    _require_git_sha(row["tree_sha"], field="population_record.tree_sha")
    _validate_public_record(
        repository_url=repository_url,
        kind=row["public_record_kind"],
        record_url=row["public_record_url"],
        record_id=row["public_record_id"],
    )
    _require_sha256(
        row["public_record_snapshot_sha256"],
        field="population_record.public_record_snapshot_sha256",
    )
    family = row["sampling_family"]
    stratum = row["stratum"]
    if family == "target":
        allowed = EXPECTED_STRATA["target_holdout"]
    elif family == "safety_control":
        allowed = EXPECTED_STRATA["safety_control"]
    else:
        raise SamplingFrameError("population record sampling_family is invalid")
    if stratum not in allowed:
        raise SamplingFrameError("population record stratum is invalid for its family")
    expected_score = _selection_score(seed, repository_url, row["public_record_url"])
    if row["selection_score_sha256"] != expected_score:
        raise SamplingFrameError("population record selection score does not match seed")
    for path_field in (
        "source_archive_path",
        "source_manifest_path",
        "provenance_receipt_path",
    ):
        _require_safe_relative_path(row[path_field], field=f"population_record.{path_field}")
    for digest_field in (
        "source_archive_sha256",
        "source_manifest_sha256",
        "provenance_receipt_sha256",
        "eligibility_receipt_sha256",
    ):
        _require_sha256(row[digest_field], field=f"population_record.{digest_field}")
    license_info = _require_mapping(row["license"], field="population_record.license")
    _require_exact_keys(license_info, _LICENSE_KEYS, field="population_record.license")
    if (
        not isinstance(license_info["spdx_id"], str)
        or _SPDX.fullmatch(license_info["spdx_id"]) is None
        or license_info["spdx_id"] in {"NOASSERTION", "NONE"}
    ):
        raise SamplingFrameError(
            "population record license must identify a reviewable license"
        )
    _require_safe_relative_path(
        license_info["path"], field="population_record.license.path"
    )
    _require_sha256(
        license_info["sha256"], field="population_record.license.sha256"
    )
    return row


def _validate_entry(entry: object, *, seed: str) -> dict[str, Any]:
    row = _require_mapping(entry, field="entry")
    _require_exact_keys(row, _ENTRY_KEYS, field="entry")
    task_id = row["task_id"]
    if not isinstance(task_id, str) or _TASK_ID.fullmatch(task_id) is None:
        raise SamplingFrameError("entry.task_id is invalid")
    partition = row["partition"]
    if partition not in EXPECTED_STRATA:
        raise SamplingFrameError("entry.partition is invalid")
    expected_prefix = {
        "development": "sp-dev-",
        "target_holdout": "sp-target-",
        "safety_control": "sp-control-",
    }[partition]
    if not task_id.startswith(expected_prefix):
        raise SamplingFrameError("entry.task_id prefix does not match partition")
    stratum = row["stratum"]
    if stratum not in EXPECTED_STRATA[partition]:
        raise SamplingFrameError("entry.stratum is not declared for its partition")
    _require_positive_int(row["selection_rank"], field="entry.selection_rank")

    repository_url, owner, repository = _normalized_repository_url(
        row["repository_url"], field="entry.repository_url"
    )
    if row["repository_owner"] != owner or row["repository_name"] != repository:
        raise SamplingFrameError("entry repository owner/name do not match repository_url")
    expected_cluster = f"github:{owner.casefold()}/{repository.casefold()}"
    if row["provenance_cluster_id"] != expected_cluster:
        raise SamplingFrameError(
            "entry.provenance_cluster_id must be the canonical GitHub repository"
        )
    _require_git_sha(row["commit_sha"], field="entry.commit_sha")
    _require_git_sha(row["tree_sha"], field="entry.tree_sha")
    _validate_public_record(
        repository_url=repository_url,
        kind=row["public_record_kind"],
        record_url=row["public_record_url"],
        record_id=row["public_record_id"],
    )
    _require_sha256(row["population_record_digest"], field="entry.population_record_digest")
    score = _require_sha256(
        row["selection_score_sha256"], field="entry.selection_score_sha256"
    )
    expected_score = _selection_score(seed, repository_url, row["public_record_url"])
    if score != expected_score:
        raise SamplingFrameError("entry.selection_score_sha256 does not match frozen seed")

    for path_field in (
        "source_archive_path",
        "source_manifest_path",
        "provenance_receipt_path",
    ):
        _require_safe_relative_path(row[path_field], field=f"entry.{path_field}")
    for digest_field in (
        "source_archive_sha256",
        "source_manifest_sha256",
        "provenance_receipt_sha256",
        "public_task_digest",
        "private_label_digest",
    ):
        _require_sha256(row[digest_field], field=f"entry.{digest_field}")
    license_info = _require_mapping(row["license"], field="entry.license")
    _require_exact_keys(license_info, _LICENSE_KEYS, field="entry.license")
    if (
        not isinstance(license_info["spdx_id"], str)
        or _SPDX.fullmatch(license_info["spdx_id"]) is None
        or license_info["spdx_id"] in {"NOASSERTION", "NONE"}
    ):
        raise SamplingFrameError("entry.license.spdx_id must identify a reviewable license")
    _require_safe_relative_path(license_info["path"], field="entry.license.path")
    _require_sha256(license_info["sha256"], field="entry.license.sha256")
    return row


def _validate_identity_and_treatment(frame: dict[str, Any]) -> None:
    studies = _require_mapping(frame["reserved_studies"], field="reserved_studies")
    _require_exact_keys(studies, _STUDY_KEYS, field="reserved_studies")
    if studies != {
        "development": DEVELOPMENT_STUDY_ID,
        "holdout": HOLDOUT_STUDY_ID,
    }:
        raise SamplingFrameError("reserved Study identities changed")

    skill = _require_mapping(frame["skill_treatment"], field="skill_treatment")
    _require_exact_keys(skill, _SKILL_KEYS, field="skill_treatment")
    if (
        skill["repository"] != EXPECTED_SKILL_REPOSITORY
        or skill["path"] != "skills/writing-plans"
        or skill["baseline_commit"] != EXPECTED_BASELINE_COMMIT
        or skill["candidate_commit"] != EXPECTED_CANDIDATE_COMMIT
    ):
        raise SamplingFrameError("Skill treatment is not the preregistered exact comparison")
    _require_safe_relative_path(skill["lock_path"], field="skill_treatment.lock_path")
    for field in (
        "lock_sha256",
        "baseline_bundle_sha256",
        "candidate_bundle_sha256",
    ):
        _require_sha256(skill[field], field=f"skill_treatment.{field}")


def _validate_population_and_selection(frame: dict[str, Any]) -> str:
    population = _require_mapping(frame["population_snapshot"], field="population_snapshot")
    _require_exact_keys(population, _POPULATION_KEYS, field="population_snapshot")
    cutoff = population["selection_cutoff_utc"]
    if not isinstance(cutoff, str) or re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", cutoff
    ) is None:
        raise SamplingFrameError("population_snapshot.selection_cutoff_utc is invalid")
    for field in (
        "records_path",
        "query_receipt_path",
        "eligibility_protocol_path",
        "selection_code_path",
    ):
        _require_safe_relative_path(population[field], field=f"population_snapshot.{field}")
    for field in (
        "records_sha256",
        "query_receipt_sha256",
        "eligibility_protocol_sha256",
        "selection_code_sha256",
    ):
        _require_sha256(population[field], field=f"population_snapshot.{field}")
    if (
        _require_positive_int(
            population["record_count"], field="population_snapshot.record_count"
        )
        < EXPECTED_TOTAL
    ):
        raise SamplingFrameError("population snapshot is smaller than the selected frame")

    protocol = _require_mapping(frame["selection_protocol"], field="selection_protocol")
    _require_exact_keys(protocol, _SELECTION_KEYS, field="selection_protocol")
    seed = _require_sha256(protocol["seed_sha256"], field="selection_protocol.seed_sha256")
    if protocol["algorithm"] != "sha256_seeded_sort_v1":
        raise SamplingFrameError("selection algorithm is not frozen sha256_seeded_sort_v1")
    if protocol["cluster_unit"] != "canonical_github_repository":
        raise SamplingFrameError("inference cluster must be the canonical GitHub repository")
    if protocol["unique_clusters_across_partitions"] is not True:
        raise SamplingFrameError("repository clusters must be unique across every partition")
    if protocol["strata"] != EXPECTED_STRATA:
        raise SamplingFrameError("selection strata or quotas changed")
    return seed


def _validate_task_manifests(frame: dict[str, Any]) -> None:
    public_tasks = _require_mapping(frame["public_tasks"], field="public_tasks")
    _require_exact_keys(public_tasks, _FILE_MANIFEST_KEYS, field="public_tasks")
    _require_safe_relative_path(public_tasks["path"], field="public_tasks.path")
    _require_sha256(public_tasks["sha256"], field="public_tasks.sha256")
    if public_tasks["count"] != EXPECTED_TOTAL:
        raise SamplingFrameError("public task count is not the powered design count")
    private_labels = _require_mapping(frame["private_labels"], field="private_labels")
    _require_exact_keys(private_labels, _PRIVATE_MANIFEST_KEYS, field="private_labels")
    _require_safe_relative_path(private_labels["path"], field="private_labels.path")
    _require_sha256(private_labels["sha256"], field="private_labels.sha256")
    if (
        private_labels["count"] != EXPECTED_TOTAL
        or private_labels["host_only"] is not True
        or private_labels["required_mode"] != "0600"
    ):
        raise SamplingFrameError("private labels are not a host-only mode-0600 exact task set")


def _validate_entries(frame: dict[str, Any], *, seed: str) -> None:
    entries = frame["entries"]
    if not isinstance(entries, list) or len(entries) != EXPECTED_TOTAL:
        raise SamplingFrameError(f"entries must contain exactly {EXPECTED_TOTAL} tasks")
    rows = [_validate_entry(entry, seed=seed) for entry in entries]
    task_ids = [row["task_id"] for row in rows]
    clusters = [row["provenance_cluster_id"] for row in rows]
    records = [row["public_record_url"] for row in rows]
    if len(set(task_ids)) != EXPECTED_TOTAL:
        raise SamplingFrameError("task identities are not unique")
    if len(set(clusters)) != EXPECTED_TOTAL:
        raise SamplingFrameError(
            "public repositories are reused; attempts do not create independent task units"
        )
    if len(set(records)) != EXPECTED_TOTAL:
        raise SamplingFrameError("public maintenance records are reused")

    observed_counts = Counter(row["partition"] for row in rows)
    if dict(observed_counts) != EXPECTED_COUNTS:
        raise SamplingFrameError("partition counts do not match the powered design")
    observed_strata = Counter((row["partition"], row["stratum"]) for row in rows)
    for partition, strata in EXPECTED_STRATA.items():
        for stratum, count in strata.items():
            if observed_strata[(partition, stratum)] != count:
                raise SamplingFrameError(
                    f"stratum count mismatch for {partition}/{stratum}"
                )
            stratum_rows = sorted(
                (
                    row
                    for row in rows
                    if row["partition"] == partition and row["stratum"] == stratum
                ),
                key=lambda row: row["selection_rank"],
            )
            if [row["selection_rank"] for row in stratum_rows] != list(
                range(1, count + 1)
            ):
                raise SamplingFrameError(
                    f"selection ranks are not contiguous for {partition}/{stratum}"
                )
            if [row["selection_score_sha256"] for row in stratum_rows] != sorted(
                row["selection_score_sha256"] for row in stratum_rows
            ):
                raise SamplingFrameError(
                    f"selection ranks do not follow the frozen score for {partition}/{stratum}"
                )


def validate_structure(document: object) -> dict[str, Any]:
    """Validate a complete frame without trusting or reading referenced files."""

    frame = _require_mapping(document, field="sampling frame")
    _require_exact_keys(frame, _TOP_LEVEL_KEYS, field="sampling frame")
    if frame["schema_version"] != SCHEMA_VERSION or frame["frame_id"] != FRAME_ID:
        raise SamplingFrameError("sampling-frame schema version or identity is unsupported")
    _validate_identity_and_treatment(frame)
    seed = _validate_population_and_selection(frame)
    _validate_task_manifests(frame)
    _validate_entries(frame, seed=seed)
    expected_digest = _require_sha256(frame["frame_digest"], field="frame_digest")
    digest_payload = dict(frame)
    digest_payload.pop("frame_digest")
    if stable_digest(digest_payload) != expected_digest:
        raise SamplingFrameError("frame_digest does not match the canonical frame")
    return frame


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise SamplingFrameError(f"invalid JSONL at {path}:{line_number}") from error
        if not isinstance(value, dict):
            raise SamplingFrameError(f"JSONL row at {path}:{line_number} is not an object")
        rows.append(value)
    return rows


def validate_population_selection(
    frame: dict[str, Any], population_records: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Prove selected entries are deterministic slices of the frozen population."""

    seed = frame["selection_protocol"]["seed_sha256"]
    population = [_validate_population_record(record, seed=seed) for record in population_records]
    digests = [stable_digest(record) for record in population]
    if len(set(digests)) != len(population):
        raise SamplingFrameError("population snapshot contains duplicate records")
    clusters = [record["provenance_cluster_id"] for record in population]
    if len(set(clusters)) != len(population):
        raise SamplingFrameError(
            "population reuses a repository cluster; selection independence is ambiguous"
        )
    record_ids = [record["record_id"] for record in population]
    if len(set(record_ids)) != len(population):
        raise SamplingFrameError("population record identities are not unique")
    population_by_digest = dict(zip(digests, population, strict=True))

    selected = frame["entries"]
    selected_by_bin: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for entry in selected:
        selected_by_bin.setdefault((entry["partition"], entry["stratum"]), []).append(
            entry
        )
        record = population_by_digest.get(entry["population_record_digest"])
        if record is None:
            raise SamplingFrameError(
                f"selected task is absent from population snapshot: {entry['task_id']}"
            )
        for field in (
            "repository_url",
            "repository_owner",
            "repository_name",
            "commit_sha",
            "tree_sha",
            "public_record_url",
            "public_record_kind",
            "public_record_id",
            "provenance_cluster_id",
            "selection_score_sha256",
            "source_archive_path",
            "source_archive_sha256",
            "source_manifest_path",
            "source_manifest_sha256",
            "provenance_receipt_path",
            "provenance_receipt_sha256",
            "license",
        ):
            if entry[field] != record[field]:
                raise SamplingFrameError(
                    f"selected task disagrees with population field {field}: "
                    f"{entry['task_id']}"
                )

    for stratum, development_count in EXPECTED_STRATA["development"].items():
        target_count = EXPECTED_STRATA["target_holdout"][stratum]
        candidates = sorted(
            (
                record
                for record in population
                if record["sampling_family"] == "target"
                and record["stratum"] == stratum
            ),
            key=lambda record: record["selection_score_sha256"],
        )
        required = development_count + target_count
        if len(candidates) < required:
            raise SamplingFrameError(
                f"population lacks {required} target candidates for {stratum}"
            )
        development = sorted(
            selected_by_bin[("development", stratum)],
            key=lambda entry: entry["selection_rank"],
        )
        target = sorted(
            selected_by_bin[("target_holdout", stratum)],
            key=lambda entry: entry["selection_rank"],
        )
        if [entry["population_record_digest"] for entry in development] != [
            stable_digest(record) for record in candidates[:development_count]
        ]:
            raise SamplingFrameError(
                f"development selection is not the first frozen slice for {stratum}"
            )
        if [entry["population_record_digest"] for entry in target] != [
            stable_digest(record) for record in candidates[development_count:required]
        ]:
            raise SamplingFrameError(
                f"holdout selection is not the next frozen slice for {stratum}"
            )

    for stratum, count in EXPECTED_STRATA["safety_control"].items():
        candidates = sorted(
            (
                record
                for record in population
                if record["sampling_family"] == "safety_control"
                and record["stratum"] == stratum
            ),
            key=lambda record: record["selection_score_sha256"],
        )
        if len(candidates) < count:
            raise SamplingFrameError(
                f"population lacks {count} safety-control candidates for {stratum}"
            )
        controls = sorted(
            selected_by_bin[("safety_control", stratum)],
            key=lambda entry: entry["selection_rank"],
        )
        if [entry["population_record_digest"] for entry in controls] != [
            stable_digest(record) for record in candidates[:count]
        ]:
            raise SamplingFrameError(
                f"safety-control selection is not the first frozen slice for {stratum}"
            )
    return population_by_digest


def _verify_file(repository_root: Path, path: str, digest: str) -> Path:
    target = (repository_root / path).resolve()
    try:
        target.relative_to(repository_root.resolve())
    except ValueError as error:
        raise SamplingFrameError(f"referenced path escaped repository root: {path}") from error
    if not target.is_file() or target.is_symlink():
        raise SamplingFrameError(f"referenced immutable file is absent or a symlink: {path}")
    actual = hashlib.sha256(target.read_bytes()).hexdigest()
    if actual != digest:
        raise SamplingFrameError(f"referenced file digest mismatch: {path}")
    return target


def validate_bundle(frame_path: Path, *, repository_root: Path) -> dict[str, Any]:
    """Validate structure plus every immutable file needed before preview."""

    frame = validate_structure(json.loads(frame_path.read_text(encoding="utf-8")))
    population = frame["population_snapshot"]
    population_path = _verify_file(
        repository_root, population["records_path"], population["records_sha256"]
    )
    population_rows = _read_jsonl(population_path)
    if len(population_rows) != population["record_count"]:
        raise SamplingFrameError("population snapshot row count changed")
    population_by_digest = validate_population_selection(frame, population_rows)
    for path_field, digest_field in (
        ("query_receipt_path", "query_receipt_sha256"),
        ("eligibility_protocol_path", "eligibility_protocol_sha256"),
        ("selection_code_path", "selection_code_sha256"),
    ):
        _verify_file(repository_root, population[path_field], population[digest_field])

    public_tasks = frame["public_tasks"]
    public_path = _verify_file(
        repository_root, public_tasks["path"], public_tasks["sha256"]
    )
    public_rows = _read_jsonl(public_path)
    private_labels = frame["private_labels"]
    private_path = _verify_file(
        repository_root, private_labels["path"], private_labels["sha256"]
    )
    private_rows = _read_jsonl(private_path)
    if len(public_rows) != public_tasks["count"] or len(private_rows) != private_labels["count"]:
        raise SamplingFrameError("task or private-label row count changed")
    if stat.S_IMODE(private_path.stat().st_mode) != 0o600:
        raise SamplingFrameError("private-label manifest is not mode 0600")
    public_by_id = {row.get("id"): row for row in public_rows}
    private_by_id = {row.get("id"): row for row in private_rows}
    expected_ids = {entry["task_id"] for entry in frame["entries"]}
    if set(public_by_id) != expected_ids or set(private_by_id) != expected_ids:
        raise SamplingFrameError("task/private-label IDs do not align with the frame")

    for entry in frame["entries"]:
        task_id = entry["task_id"]
        if stable_digest(public_by_id[task_id]) != entry["public_task_digest"]:
            raise SamplingFrameError(f"public task digest mismatch for {task_id}")
        if stable_digest(private_by_id[task_id]) != entry["private_label_digest"]:
            raise SamplingFrameError(f"private label digest mismatch for {task_id}")
        if entry["population_record_digest"] not in population_by_digest:
            raise SamplingFrameError(f"selected task is absent from population snapshot: {task_id}")
        for path_field, digest_field in (
            ("source_archive_path", "source_archive_sha256"),
            ("source_manifest_path", "source_manifest_sha256"),
            ("provenance_receipt_path", "provenance_receipt_sha256"),
        ):
            _verify_file(repository_root, entry[path_field], entry[digest_field])

    skill = frame["skill_treatment"]
    _verify_file(repository_root, skill["lock_path"], skill["lock_sha256"])
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "frame_id": FRAME_ID,
        "frame_digest": frame["frame_digest"],
        "counts": EXPECTED_COUNTS,
        "independent_repository_clusters": EXPECTED_TOTAL,
        "public_task_manifest_sha256": public_tasks["sha256"],
        "private_label_manifest_sha256": private_labels["sha256"],
        "private_values_published": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame", type=Path)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    args = parser.parse_args()
    try:
        receipt = validate_bundle(args.frame, repository_root=args.repository_root)
    except (OSError, json.JSONDecodeError, SamplingFrameError) as error:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "blocked",
                    "frame": args.frame.as_posix(),
                    "reason": str(error),
                },
                sort_keys=True,
            )
        )
        return 3
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
