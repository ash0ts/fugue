"""Fail-closed validation for the powered Skill Creator sampling frame.

This validator accepts only a complete, prospectively selected bundle of real
public Skill packages.  It never discovers repositories, authors tasks, or
fills quotas.  Those trusted preparation operations happen before preview;
Agent trials consume only the immutable archives accepted here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

SCHEMA_VERSION = 1
FRAME_ID = "anthropic-skill-creator-conference-sampling-frame-v1"
DEVELOPMENT_STUDY_ID = "anthropic-skill-creator-conference-development-v1"
HOLDOUT_STUDY_ID = "anthropic-skill-creator-conference-holdout-v1"
EXPECTED_SKILL_REPOSITORY = "https://github.com/anthropics/skills"
EXPECTED_BASELINE_COMMIT = "a5bcdd7e58cdff48566bf876f0a72a2008dcefbc"
EXPECTED_CANDIDATE_COMMIT = "1ed29a03dc852d30fa6ef2ca53a67dc2c2c2c563"

EXPECTED_STRATA: dict[str, dict[str, int]] = {
    "development": {
        "missing_declared_compatibility": 8,
        "valid_declared_compatibility": 8,
        "invalid_declared_compatibility": 8,
        "long_valid_name": 8,
    },
    "target_holdout": {
        "missing_declared_compatibility": 48,
        "valid_declared_compatibility": 48,
        "invalid_declared_compatibility": 48,
        "long_valid_name": 48,
    },
    "safety_control": {
        "compatibility_not_applicable": 16,
        "unrelated_metadata_preservation": 16,
        "unsupported_exact_name": 16,
        "no_change_required": 16,
    },
}
EXPECTED_COUNTS = {
    partition: sum(strata.values()) for partition, strata in EXPECTED_STRATA.items()
}
EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_TASK_ID = re.compile(
    r"^as-conf-(?:dev|target|control)-[a-z0-9][a-z0-9-]{2,87}$"
)
_REPOSITORY_URL = re.compile(
    r"^https://github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)$"
)
_SKILL_PATH = re.compile(r"^[A-Za-z0-9._/-]+$")
_SPDX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+-]{0,63}$")
_ISO_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_BASIS_KINDS = {"skill_blob", "issue", "pull_request", "commit"}

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
    "skill_path",
    "skill_tree_sha",
    "skill_blob_url",
    "skill_content_sha256",
    "public_basis_url",
    "public_basis_kind",
    "public_basis_id",
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
_RECEIPT_KEYS = {
    "schema_version",
    "repository_url",
    "commit_sha",
    "tree_sha",
    "skill_path",
    "skill_tree_sha",
    "skill_blob_url",
    "skill_content_sha256",
    "public_basis_url",
    "public_basis_kind",
    "public_basis_id",
    "observed_at_utc",
    "verified_remote",
    "source_archive_sha256",
    "source_manifest_sha256",
    "license",
}
_SOURCE_MANIFEST_KEYS = {"schema_version", "archive_root", "files"}
_SOURCE_FILE_KEYS = {"path", "sha256", "bytes"}
_FORBIDDEN_PUBLIC_KEYS = {
    "answer_key",
    "expected",
    "expected_answer",
    "expected_files",
    "gold",
    "private_label",
    "reference_answer",
    "scorer_inputs",
}
_FORBIDDEN_ARCHIVE_PARTS = {
    ".git",
    ".env",
    ".venv",
    "__pycache__",
    "node_modules",
    "private-labels",
    "private_labels",
}


class SamplingFrameError(ValueError):
    """Raised when a completed frame is not safe or scientifically valid."""


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


def _require_exact_keys(
    value: dict[str, Any], expected: set[str], *, field: str
) -> None:
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
        raise SamplingFrameError(f"{field} must be a non-empty relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or "\\" in value
        or "\x00" in value
        or "\n" in value
        or "\r" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise SamplingFrameError(f"{field} is not a safe relative path")
    return value


def _require_skill_path(value: object, *, field: str) -> str:
    path = _require_safe_relative_path(value, field=field)
    if (
        _SKILL_PATH.fullmatch(path) is None
        or path.endswith("/")
        or path.endswith("/SKILL.md")
    ):
        raise SamplingFrameError(f"{field} must name a Skill package directory")
    return path


def _require_positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SamplingFrameError(f"{field} must be a positive integer")
    return value


def _normalized_repository_url(value: object, *, field: str) -> tuple[str, str, str]:
    if not isinstance(value, str):
        raise SamplingFrameError(f"{field} must be a canonical GitHub repository URL")
    match = _REPOSITORY_URL.fullmatch(value)
    if match is None or value.endswith(".git"):
        raise SamplingFrameError(
            f"{field} must be canonical https://github.com/OWNER/REPO"
        )
    return value, match.group("owner"), match.group("repo")


def _validate_https_github_url(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise SamplingFrameError(f"{field} must be an HTTPS GitHub URL")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "github.com"
        or parsed.query
        or parsed.fragment
    ):
        raise SamplingFrameError(f"{field} must be an unparameterized GitHub URL")
    return value


def _validate_public_basis(
    *,
    repository_url: str,
    commit_sha: str,
    skill_path: str,
    skill_blob_url: object,
    basis_kind: object,
    basis_url: object,
    basis_id: object,
) -> tuple[str, str]:
    blob_url = _validate_https_github_url(skill_blob_url, field="entry.skill_blob_url")
    expected_blob = f"{repository_url}/blob/{commit_sha}/{skill_path}/SKILL.md"
    if blob_url != expected_blob:
        raise SamplingFrameError("entry.skill_blob_url does not bind the exact Skill blob")
    if basis_kind not in _BASIS_KINDS:
        raise SamplingFrameError("entry.public_basis_kind is unsupported")
    public_url = _validate_https_github_url(basis_url, field="entry.public_basis_url")
    if not isinstance(basis_id, str) or not basis_id:
        raise SamplingFrameError("entry.public_basis_id must be a non-empty string")
    if basis_kind == "skill_blob":
        expected_id = f"{commit_sha}:{skill_path}/SKILL.md"
        if public_url != expected_blob or basis_id != expected_id:
            raise SamplingFrameError("Skill-blob basis does not bind the exact public blob")
        return blob_url, public_url
    if not public_url.startswith(f"{repository_url}/"):
        raise SamplingFrameError("entry.public_basis_url belongs to another repository")
    suffix = public_url[len(repository_url) + 1 :]
    if basis_kind == "issue":
        if not basis_id.isdigit() or suffix != f"issues/{basis_id}":
            raise SamplingFrameError("issue basis URL/ID mismatch")
    elif basis_kind == "pull_request":
        if not basis_id.isdigit() or suffix != f"pull/{basis_id}":
            raise SamplingFrameError("pull-request basis URL/ID mismatch")
    else:
        _require_git_sha(basis_id, field="entry.public_basis_id")
        if suffix != f"commit/{basis_id}":
            raise SamplingFrameError("commit basis URL/ID mismatch")
    return blob_url, public_url


def _selection_score(
    seed: str, repository_url: str, skill_path: str, public_basis_url: str
) -> str:
    payload = (
        f"{seed}\n{repository_url.casefold()}\n{skill_path}\n{public_basis_url}\n"
    ).encode()
    return hashlib.sha256(payload).hexdigest()


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
        "development": "as-conf-dev-",
        "target_holdout": "as-conf-target-",
        "safety_control": "as-conf-control-",
    }[partition]
    if not task_id.startswith(expected_prefix):
        raise SamplingFrameError("entry.task_id prefix does not match partition")
    if row["stratum"] not in EXPECTED_STRATA[partition]:
        raise SamplingFrameError("entry.stratum is not declared for its partition")
    _require_positive_int(row["selection_rank"], field="entry.selection_rank")

    repository_url, owner, repository = _normalized_repository_url(
        row["repository_url"], field="entry.repository_url"
    )
    if row["repository_owner"] != owner or row["repository_name"] != repository:
        raise SamplingFrameError("entry owner/name do not match repository_url")
    expected_cluster = f"github:{owner.casefold()}/{repository.casefold()}"
    if row["provenance_cluster_id"] != expected_cluster:
        raise SamplingFrameError(
            "entry.provenance_cluster_id must be the canonical GitHub repository"
        )
    commit_sha = _require_git_sha(row["commit_sha"], field="entry.commit_sha")
    _require_git_sha(row["tree_sha"], field="entry.tree_sha")
    skill_path = _require_skill_path(row["skill_path"], field="entry.skill_path")
    _require_git_sha(row["skill_tree_sha"], field="entry.skill_tree_sha")
    _, basis_url = _validate_public_basis(
        repository_url=repository_url,
        commit_sha=commit_sha,
        skill_path=skill_path,
        skill_blob_url=row["skill_blob_url"],
        basis_kind=row["public_basis_kind"],
        basis_url=row["public_basis_url"],
        basis_id=row["public_basis_id"],
    )
    _require_sha256(row["skill_content_sha256"], field="entry.skill_content_sha256")
    _require_sha256(row["population_record_digest"], field="entry.population_record_digest")
    score = _require_sha256(
        row["selection_score_sha256"], field="entry.selection_score_sha256"
    )
    if score != _selection_score(seed, repository_url, skill_path, basis_url):
        raise SamplingFrameError("entry selection score does not match the frozen seed")

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
        or license_info["spdx_id"] in {"NONE", "NOASSERTION"}
    ):
        raise SamplingFrameError("entry.license.spdx_id must identify a license")
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

    treatment = _require_mapping(frame["skill_treatment"], field="skill_treatment")
    _require_exact_keys(treatment, _SKILL_KEYS, field="skill_treatment")
    if (
        treatment["repository"] != EXPECTED_SKILL_REPOSITORY
        or treatment["path"] != "skills/skill-creator"
        or treatment["baseline_commit"] != EXPECTED_BASELINE_COMMIT
        or treatment["candidate_commit"] != EXPECTED_CANDIDATE_COMMIT
    ):
        raise SamplingFrameError("Skill treatment is not the exact locked comparison")
    _require_safe_relative_path(treatment["lock_path"], field="skill_treatment.lock_path")
    for field in (
        "lock_sha256",
        "baseline_bundle_sha256",
        "candidate_bundle_sha256",
    ):
        _require_sha256(treatment[field], field=f"skill_treatment.{field}")


def _validate_population(population_value: object) -> dict[str, Any]:
    population = _require_mapping(population_value, field="population_snapshot")
    _require_exact_keys(population, _POPULATION_KEYS, field="population_snapshot")
    cutoff = population["selection_cutoff_utc"]
    if not isinstance(cutoff, str) or _ISO_UTC.fullmatch(cutoff) is None:
        raise SamplingFrameError("population selection cutoff is invalid")
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
        raise SamplingFrameError("population snapshot is smaller than the powered frame")
    return population


def _validate_selection(selection_value: object) -> str:
    selection = _require_mapping(selection_value, field="selection_protocol")
    _require_exact_keys(selection, _SELECTION_KEYS, field="selection_protocol")
    seed = _require_sha256(selection["seed_sha256"], field="selection_protocol.seed_sha256")
    if selection["algorithm"] != "sha256_seeded_sort_v1":
        raise SamplingFrameError("selection algorithm is not frozen")
    if selection["cluster_unit"] != "canonical_github_repository":
        raise SamplingFrameError("inference cluster must be the GitHub repository")
    if selection["unique_clusters_across_partitions"] is not True:
        raise SamplingFrameError("repository clusters must be globally unique")
    if selection["strata"] != EXPECTED_STRATA:
        raise SamplingFrameError("selection strata or quotas changed")
    return seed


def _validate_task_manifests(frame: dict[str, Any]) -> None:
    public_tasks = _require_mapping(frame["public_tasks"], field="public_tasks")
    _require_exact_keys(public_tasks, _FILE_MANIFEST_KEYS, field="public_tasks")
    _require_safe_relative_path(public_tasks["path"], field="public_tasks.path")
    _require_sha256(public_tasks["sha256"], field="public_tasks.sha256")
    if public_tasks["count"] != EXPECTED_TOTAL:
        raise SamplingFrameError("public task count is not the powered count")
    private_labels = _require_mapping(frame["private_labels"], field="private_labels")
    _require_exact_keys(private_labels, _PRIVATE_MANIFEST_KEYS, field="private_labels")
    _require_safe_relative_path(private_labels["path"], field="private_labels.path")
    _require_sha256(private_labels["sha256"], field="private_labels.sha256")
    if (
        private_labels["count"] != EXPECTED_TOTAL
        or private_labels["host_only"] is not True
        or private_labels["required_mode"] != "0600"
    ):
        raise SamplingFrameError("private labels are not a mode-0600 exact task set")


def _validate_strata(rows: list[dict[str, Any]]) -> None:
    if Counter(row["partition"] for row in rows) != Counter(EXPECTED_COUNTS):
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
            observed_scores = [row["selection_score_sha256"] for row in stratum_rows]
            if observed_scores != sorted(observed_scores):
                raise SamplingFrameError(
                    f"selection ranks do not follow the frozen score for {partition}/{stratum}"
                )


def _validate_entries(entries_value: object, *, seed: str) -> list[dict[str, Any]]:
    if not isinstance(entries_value, list) or len(entries_value) != EXPECTED_TOTAL:
        raise SamplingFrameError(f"entries must contain exactly {EXPECTED_TOTAL} tasks")
    rows = [_validate_entry(entry, seed=seed) for entry in entries_value]
    unique_fields = {
        "task identities": [row["task_id"] for row in rows],
        "repository clusters": [row["provenance_cluster_id"] for row in rows],
        "Skill blobs": [row["skill_blob_url"] for row in rows],
        "Skill contents": [row["skill_content_sha256"] for row in rows],
        "source archives": [row["source_archive_path"] for row in rows],
        "source archive contents": [row["source_archive_sha256"] for row in rows],
        "source manifests": [row["source_manifest_path"] for row in rows],
        "provenance receipts": [row["provenance_receipt_path"] for row in rows],
    }
    for label, values in unique_fields.items():
        if len(set(values)) != EXPECTED_TOTAL:
            raise SamplingFrameError(f"{label} are reused across independent tasks")
    _validate_strata(rows)
    return rows


def validate_structure(document: object) -> dict[str, Any]:
    """Validate a complete frame without trusting or reading referenced files."""

    frame = _require_mapping(document, field="sampling frame")
    _require_exact_keys(frame, _TOP_LEVEL_KEYS, field="sampling frame")
    if frame["schema_version"] != SCHEMA_VERSION or frame["frame_id"] != FRAME_ID:
        raise SamplingFrameError("sampling-frame version or identity is unsupported")
    _validate_identity_and_treatment(frame)
    _validate_population(frame["population_snapshot"])
    seed = _validate_selection(frame["selection_protocol"])
    _validate_task_manifests(frame)
    _validate_entries(frame["entries"], seed=seed)

    expected_digest = _require_sha256(frame["frame_digest"], field="frame_digest")
    digest_payload = dict(frame)
    digest_payload.pop("frame_digest")
    if stable_digest(digest_payload) != expected_digest:
        raise SamplingFrameError("frame_digest does not match the canonical frame")
    return frame


def _read_json(path: Path) -> dict[str, Any]:
    return _require_mapping(json.loads(path.read_text(encoding="utf-8")), field=str(path))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise SamplingFrameError(f"invalid JSONL at {path}:{line_number}") from error
        rows.append(_require_mapping(value, field=f"{path}:{line_number}"))
    return rows


def _verify_file(repository_root: Path, path: str, digest: str) -> Path:
    target = (repository_root / path).resolve()
    try:
        target.relative_to(repository_root.resolve())
    except ValueError as error:
        raise SamplingFrameError(f"referenced path escaped repository root: {path}") from error
    if not target.is_file() or target.is_symlink():
        raise SamplingFrameError(f"immutable file is absent or a symlink: {path}")
    if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        raise SamplingFrameError(f"referenced file digest mismatch: {path}")
    return target


def _assert_no_private_keys(value: object, *, path: str = "public task") -> None:
    if isinstance(value, dict):
        leaked = _FORBIDDEN_PUBLIC_KEYS.intersection(value)
        if leaked:
            raise SamplingFrameError(f"{path} contains private keys: {sorted(leaked)}")
        for key, child in value.items():
            _assert_no_private_keys(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_private_keys(child, path=f"{path}[{index}]")


def _validate_source_archive(
    archive_path: Path,
    manifest_path: Path,
    *,
    skill_path: str,
    skill_content_sha256: str,
    license_info: dict[str, Any],
) -> None:
    manifest = _read_json(manifest_path)
    _require_exact_keys(manifest, _SOURCE_MANIFEST_KEYS, field="source manifest")
    if manifest["schema_version"] != 1 or manifest["archive_root"] != ".":
        raise SamplingFrameError("source manifest version or archive root is unsupported")
    raw_files = manifest["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise SamplingFrameError("source manifest must contain at least one file")
    declared: dict[str, tuple[str, int]] = {}
    for raw_file in raw_files:
        source_file = _require_mapping(raw_file, field="source manifest file")
        _require_exact_keys(source_file, _SOURCE_FILE_KEYS, field="source manifest file")
        path = _require_safe_relative_path(source_file["path"], field="source file path")
        digest = _require_sha256(source_file["sha256"], field="source file sha256")
        size = source_file["bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise SamplingFrameError("source file bytes must be a non-negative integer")
        if path in declared:
            raise SamplingFrameError("source manifest contains duplicate paths")
        declared[path] = (digest, size)

    observed: dict[str, tuple[str, int]] = {}
    try:
        archive = tarfile.open(archive_path, mode="r:*")
    except tarfile.TarError as error:
        raise SamplingFrameError("source archive is not a readable tar archive") from error
    with archive:
        for member in archive.getmembers():
            member_path = _require_safe_relative_path(member.name, field="archive member")
            parts = set(PurePosixPath(member_path).parts)
            if parts.intersection(_FORBIDDEN_ARCHIVE_PARTS):
                raise SamplingFrameError("source archive contains a forbidden path")
            if member.issym() or member.islnk() or member.isdev():
                raise SamplingFrameError("source archive contains an unsafe link or device")
            if member.isdir():
                continue
            if not member.isfile():
                raise SamplingFrameError("source archive contains an unsupported member")
            if member_path in observed:
                raise SamplingFrameError("source archive contains duplicate member paths")
            handle = archive.extractfile(member)
            if handle is None:
                raise SamplingFrameError("source archive file could not be read")
            payload = handle.read()
            observed[member_path] = (hashlib.sha256(payload).hexdigest(), len(payload))
    if observed != declared:
        raise SamplingFrameError("source archive contents do not match its file manifest")
    skill_file = f"{skill_path}/SKILL.md"
    if observed.get(skill_file, (None, None))[0] != skill_content_sha256:
        raise SamplingFrameError("source archive does not contain the exact public SKILL.md")
    if observed[skill_file][1] == 0:
        raise SamplingFrameError("source archive contains an empty public SKILL.md")
    license_path = license_info["path"]
    if observed.get(license_path, (None, None))[0] != license_info["sha256"]:
        raise SamplingFrameError("source archive does not contain the locked license")


def _validate_provenance_receipt(
    receipt_path: Path,
    *,
    entry: dict[str, Any],
    cutoff: str,
) -> None:
    receipt = _read_json(receipt_path)
    _require_exact_keys(receipt, _RECEIPT_KEYS, field="provenance receipt")
    expected = {
        "schema_version": 1,
        "repository_url": entry["repository_url"],
        "commit_sha": entry["commit_sha"],
        "tree_sha": entry["tree_sha"],
        "skill_path": entry["skill_path"],
        "skill_tree_sha": entry["skill_tree_sha"],
        "skill_blob_url": entry["skill_blob_url"],
        "skill_content_sha256": entry["skill_content_sha256"],
        "public_basis_url": entry["public_basis_url"],
        "public_basis_kind": entry["public_basis_kind"],
        "public_basis_id": entry["public_basis_id"],
        "verified_remote": True,
        "source_archive_sha256": entry["source_archive_sha256"],
        "source_manifest_sha256": entry["source_manifest_sha256"],
        "license": entry["license"],
    }
    for field, value in expected.items():
        if receipt[field] != value:
            raise SamplingFrameError(f"provenance receipt disagrees on {field}")
    observed_at = receipt["observed_at_utc"]
    if (
        not isinstance(observed_at, str)
        or _ISO_UTC.fullmatch(observed_at) is None
        or observed_at > cutoff
    ):
        raise SamplingFrameError("provenance receipt is not bounded by selection cutoff")


def _population_record_matches(entry: dict[str, Any], record: dict[str, Any]) -> bool:
    fields = (
        "repository_url",
        "commit_sha",
        "tree_sha",
        "skill_path",
        "skill_tree_sha",
        "skill_blob_url",
        "skill_content_sha256",
        "public_basis_url",
        "public_basis_kind",
        "public_basis_id",
    )
    return all(record.get(field) == entry[field] for field in fields)


def validate_bundle(frame_path: Path, *, repository_root: Path) -> dict[str, Any]:
    """Validate structure and every immutable artifact required before preview."""

    frame = validate_structure(json.loads(frame_path.read_text(encoding="utf-8")))
    population = frame["population_snapshot"]
    population_path = _verify_file(
        repository_root, population["records_path"], population["records_sha256"]
    )
    population_rows = _read_jsonl(population_path)
    if len(population_rows) != population["record_count"]:
        raise SamplingFrameError("population snapshot row count changed")
    population_by_digest = {stable_digest(row): row for row in population_rows}
    if len(population_by_digest) != len(population_rows):
        raise SamplingFrameError("population snapshot contains duplicate records")
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
    if len(public_rows) != EXPECTED_TOTAL or len(private_rows) != EXPECTED_TOTAL:
        raise SamplingFrameError("task or private-label row count changed")
    if stat.S_IMODE(private_path.stat().st_mode) != 0o600:
        raise SamplingFrameError("private-label manifest is not mode 0600")
    public_by_id = {row.get("id"): row for row in public_rows}
    private_by_id = {row.get("id"): row for row in private_rows}
    expected_ids = {entry["task_id"] for entry in frame["entries"]}
    if set(public_by_id) != expected_ids or set(private_by_id) != expected_ids:
        raise SamplingFrameError("task/private-label IDs do not align with the frame")

    cutoff = population["selection_cutoff_utc"]
    for entry in frame["entries"]:
        task_id = entry["task_id"]
        public_task = public_by_id[task_id]
        _assert_no_private_keys(public_task, path=f"public task {task_id}")
        if public_task.get("partition") != entry["partition"]:
            raise SamplingFrameError(f"public task partition mismatch for {task_id}")
        if stable_digest(public_task) != entry["public_task_digest"]:
            raise SamplingFrameError(f"public task digest mismatch for {task_id}")
        if stable_digest(private_by_id[task_id]) != entry["private_label_digest"]:
            raise SamplingFrameError(f"private label digest mismatch for {task_id}")
        population_record = population_by_digest.get(entry["population_record_digest"])
        if population_record is None or not _population_record_matches(
            entry, population_record
        ):
            raise SamplingFrameError(
                f"selected task lacks matching population provenance: {task_id}"
            )

        archive_path = _verify_file(
            repository_root,
            entry["source_archive_path"],
            entry["source_archive_sha256"],
        )
        manifest_path = _verify_file(
            repository_root,
            entry["source_manifest_path"],
            entry["source_manifest_sha256"],
        )
        receipt_path = _verify_file(
            repository_root,
            entry["provenance_receipt_path"],
            entry["provenance_receipt_sha256"],
        )
        _validate_source_archive(
            archive_path,
            manifest_path,
            skill_path=entry["skill_path"],
            skill_content_sha256=entry["skill_content_sha256"],
            license_info=entry["license"],
        )
        _validate_provenance_receipt(receipt_path, entry=entry, cutoff=cutoff)

    treatment = frame["skill_treatment"]
    _verify_file(repository_root, treatment["lock_path"], treatment["lock_sha256"])
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "frame_id": FRAME_ID,
        "frame_digest": frame["frame_digest"],
        "reserved_studies": frame["reserved_studies"],
        "counts": EXPECTED_COUNTS,
        "independent_repository_clusters": EXPECTED_TOTAL,
        "public_task_manifest_sha256": public_tasks["sha256"],
        "private_label_manifest_sha256": private_labels["sha256"],
        "private_values_published": False,
        "trials_may_resolve_network_sources": False,
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
