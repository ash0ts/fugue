"""Fail-closed validation for the powered Vercel Skill sampling frame.

This validator accepts only a complete, prospectively sampled population of
real public TypeScript/Next.js repositories. It does not discover repositories,
author tasks, or generate fixtures. Those trusted preparation actions must
finish before a development or holdout comparison spec can exist.
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
FRAME_ID = "vercel-react-best-practices-conference-sampling-frame-v1"
DEVELOPMENT_STUDY_ID = "vercel-react-best-practices-conference-development-v1"
HOLDOUT_STUDY_ID = "vercel-react-best-practices-conference-holdout-v1"
EXPECTED_SKILL_REPOSITORY = "https://github.com/vercel-labs/agent-skills"
EXPECTED_SKILL_PATH = "skills/react-best-practices"
EXPECTED_BASELINE_COMMIT = "ac6a79af08f6d32c34ee03c829824990f3de0a6d"
EXPECTED_BASELINE_BUNDLE = "042dce52998aa6288b4b5eac3fae325113559a666d959a10dd164a981b8ab797"
EXPECTED_CANDIDATE_COMMIT = "20987af2f1bc17857b55e7758af8bed91c364ff5"
EXPECTED_CANDIDATE_BUNDLE = "c9a31361925582718024d9ed69078e4dc3a41293f18642436a05a396d91134de"

EXPECTED_STRATA: dict[str, dict[str, int]] = {
    "development": {
        "server_action_authorization": 16,
        "rsc_serialization": 16,
    },
    "target_holdout": {
        "server_action_authorization": 96,
        "rsc_serialization": 96,
    },
    "safety_control": {
        "dom_batching": 16,
        "large_array_iteration": 16,
        "hook_timing": 16,
        "event_handler_reference": 16,
    },
}
EXPECTED_COUNTS = {
    partition: sum(strata.values()) for partition, strata in EXPECTED_STRATA.items()
}
EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_TASK_ID = re.compile(r"^vrb-(?:dev|target|control)-[a-z0-9][a-z0-9-]{2,95}$")
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
_PUBLIC_MANIFEST_KEYS = {"path", "sha256", "count"}
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
    "target_stack_evidence",
    "source_archive_path",
    "source_archive_sha256",
    "source_manifest_path",
    "source_manifest_sha256",
    "provenance_receipt_path",
    "provenance_receipt_sha256",
    "public_test",
    "host_verifier",
    "qualification_receipt_path",
    "qualification_receipt_sha256",
    "public_task_digest",
    "private_label_digest",
    "license",
}
_STACK_KEYS = {
    "package_manifest_path",
    "package_manifest_sha256",
    "dependency_lock_path",
    "dependency_lock_sha256",
    "react_dependency",
    "next_dependency",
    "typescript_paths",
    "target_paths",
}
_PUBLIC_TEST_KEYS = {"path", "sha256", "command_profile_id"}
_HOST_VERIFIER_KEYS = {
    "path",
    "sha256",
    "runtime_profile_id",
    "host_only",
    "required_mode",
}
_LICENSE_KEYS = {"spdx_id", "path", "sha256"}
_SOURCE_MANIFEST_KEYS = {
    "schema_version",
    "repository_url",
    "commit_sha",
    "tree_sha",
    "archive_sha256",
    "files",
}
_SOURCE_FILE_KEYS = {"path", "sha256", "size"}
_PROVENANCE_RECEIPT_KEYS = {
    "schema_version",
    "repository_url",
    "commit_sha",
    "tree_sha",
    "public_record_url",
    "remote_verified",
    "license_verified",
    "receipt_digest",
}
_QUALIFICATION_RECEIPT_KEYS = {
    "schema_version",
    "task_id",
    "public_test_sha256",
    "host_verifier_sha256",
    "runtime_lock_sha256",
    "base_public_test",
    "gold_public_test",
    "base_host_verifier",
    "gold_host_verifier",
    "receipt_digest",
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
    if set(value) != expected:
        raise SamplingFrameError(
            f"{field} has invalid keys; "
            f"missing={sorted(expected - set(value))}, unknown={sorted(set(value) - expected)}"
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
    if (
        path.is_absolute()
        or "\\" in value
        or "\x00" in value
        or "\n" in value
        or "\r" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise SamplingFrameError(f"{field} is not a safe repository-relative path")
    return value


def _require_positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SamplingFrameError(f"{field} must be a positive integer")
    return value


def _require_dependency(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 128
        or any(character in value for character in "\x00\r\n")
    ):
        raise SamplingFrameError(f"{field} must be the frozen dependency declaration")
    return value


def _normalized_repository_url(value: object, *, field: str) -> tuple[str, str, str]:
    if not isinstance(value, str):
        raise SamplingFrameError(f"{field} must be a GitHub repository URL")
    match = _REPOSITORY_URL.fullmatch(value)
    if match is None or value.endswith(".git"):
        raise SamplingFrameError(f"{field} must be canonical https://github.com/OWNER/REPO")
    return value, match.group("owner"), match.group("repo")


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


def _validate_stack(value: object) -> dict[str, Any]:
    stack = _require_mapping(value, field="entry.target_stack_evidence")
    _require_exact_keys(stack, _STACK_KEYS, field="entry.target_stack_evidence")
    package_path = _require_safe_relative_path(
        stack["package_manifest_path"], field="entry.target_stack_evidence.package_manifest_path"
    )
    if PurePosixPath(package_path).name != "package.json":
        raise SamplingFrameError("target stack package manifest must be package.json")
    _require_sha256(
        stack["package_manifest_sha256"],
        field="entry.target_stack_evidence.package_manifest_sha256",
    )
    lock_path = _require_safe_relative_path(
        stack["dependency_lock_path"], field="entry.target_stack_evidence.dependency_lock_path"
    )
    if PurePosixPath(lock_path).name not in {
        "package-lock.json",
        "npm-shrinkwrap.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "bun.lock",
        "bun.lockb",
    }:
        raise SamplingFrameError("target stack dependency lock is unsupported")
    _require_sha256(
        stack["dependency_lock_sha256"],
        field="entry.target_stack_evidence.dependency_lock_sha256",
    )
    _require_dependency(stack["react_dependency"], field="target_stack_evidence.react_dependency")
    _require_dependency(stack["next_dependency"], field="target_stack_evidence.next_dependency")
    type_paths = stack["typescript_paths"]
    target_paths = stack["target_paths"]
    if (
        not isinstance(type_paths, list)
        or not type_paths
        or not isinstance(target_paths, list)
        or not target_paths
    ):
        raise SamplingFrameError("target stack needs TypeScript and task target paths")
    normalized_types = [
        _require_safe_relative_path(item, field="target_stack_evidence.typescript_paths")
        for item in type_paths
    ]
    normalized_targets = [
        _require_safe_relative_path(item, field="target_stack_evidence.target_paths")
        for item in target_paths
    ]
    if len(set(normalized_types)) != len(normalized_types) or len(set(normalized_targets)) != len(
        normalized_targets
    ):
        raise SamplingFrameError("target stack paths must be unique")
    if not all(path.endswith((".ts", ".tsx")) for path in normalized_types):
        raise SamplingFrameError("typescript_paths must contain only .ts or .tsx files")
    if not set(normalized_targets).issubset(normalized_types):
        raise SamplingFrameError("every target path must be proven TypeScript/TSX")
    return stack


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
        "development": "vrb-dev-",
        "target_holdout": "vrb-target-",
        "safety_control": "vrb-control-",
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
        raise SamplingFrameError("entry repository owner/name do not match repository_url")
    if row["provenance_cluster_id"] != f"github:{owner.casefold()}/{repository.casefold()}":
        raise SamplingFrameError("entry provenance cluster must be the canonical repository")
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
    if score != _selection_score(seed, repository_url, row["public_record_url"]):
        raise SamplingFrameError("entry selection score does not match the frozen seed")
    _validate_stack(row["target_stack_evidence"])

    for path_field in (
        "source_archive_path",
        "source_manifest_path",
        "provenance_receipt_path",
        "qualification_receipt_path",
    ):
        _require_safe_relative_path(row[path_field], field=f"entry.{path_field}")
    for digest_field in (
        "source_archive_sha256",
        "source_manifest_sha256",
        "provenance_receipt_sha256",
        "qualification_receipt_sha256",
        "public_task_digest",
        "private_label_digest",
    ):
        _require_sha256(row[digest_field], field=f"entry.{digest_field}")

    public_test = _require_mapping(row["public_test"], field="entry.public_test")
    _require_exact_keys(public_test, _PUBLIC_TEST_KEYS, field="entry.public_test")
    _require_safe_relative_path(public_test["path"], field="entry.public_test.path")
    _require_sha256(public_test["sha256"], field="entry.public_test.sha256")
    if public_test["command_profile_id"] != "prepared-node-public-test-v1":
        raise SamplingFrameError("entry public-test command profile changed")

    verifier = _require_mapping(row["host_verifier"], field="entry.host_verifier")
    _require_exact_keys(verifier, _HOST_VERIFIER_KEYS, field="entry.host_verifier")
    _require_safe_relative_path(verifier["path"], field="entry.host_verifier.path")
    _require_sha256(verifier["sha256"], field="entry.host_verifier.sha256")
    if (
        verifier["runtime_profile_id"] != "node22-verifier-v1"
        or verifier["host_only"] is not True
        or verifier["required_mode"] != "0600"
    ):
        raise SamplingFrameError("entry host verifier is not the locked host-only profile")

    license_info = _require_mapping(row["license"], field="entry.license")
    _require_exact_keys(license_info, _LICENSE_KEYS, field="entry.license")
    if (
        not isinstance(license_info["spdx_id"], str)
        or _SPDX.fullmatch(license_info["spdx_id"]) is None
        or license_info["spdx_id"] in {"NONE", "NOASSERTION"}
    ):
        raise SamplingFrameError("entry license must be reviewable")
    _require_safe_relative_path(license_info["path"], field="entry.license.path")
    _require_sha256(license_info["sha256"], field="entry.license.sha256")
    return row


def _validate_header_and_treatment(frame: dict[str, Any]) -> None:
    if frame["schema_version"] != SCHEMA_VERSION or frame["frame_id"] != FRAME_ID:
        raise SamplingFrameError("sampling-frame version or identity is unsupported")

    studies = _require_mapping(frame["reserved_studies"], field="reserved_studies")
    if studies != {"development": DEVELOPMENT_STUDY_ID, "holdout": HOLDOUT_STUDY_ID}:
        raise SamplingFrameError("reserved Study identities changed")

    skill = _require_mapping(frame["skill_treatment"], field="skill_treatment")
    _require_exact_keys(skill, _SKILL_KEYS, field="skill_treatment")
    expected_skill = {
        "repository": EXPECTED_SKILL_REPOSITORY,
        "path": EXPECTED_SKILL_PATH,
        "baseline_commit": EXPECTED_BASELINE_COMMIT,
        "baseline_bundle_sha256": EXPECTED_BASELINE_BUNDLE,
        "candidate_commit": EXPECTED_CANDIDATE_COMMIT,
        "candidate_bundle_sha256": EXPECTED_CANDIDATE_BUNDLE,
    }
    if any(skill[key] != value for key, value in expected_skill.items()):
        raise SamplingFrameError("Skill treatment is not the exact preregistered comparison")
    _require_safe_relative_path(skill["lock_path"], field="skill_treatment.lock_path")
    _require_sha256(skill["lock_sha256"], field="skill_treatment.lock_sha256")


def _validate_population_and_selection(frame: dict[str, Any]) -> str:
    population = _require_mapping(frame["population_snapshot"], field="population_snapshot")
    _require_exact_keys(population, _POPULATION_KEYS, field="population_snapshot")
    if not isinstance(population["selection_cutoff_utc"], str) or re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
        population["selection_cutoff_utc"],
    ) is None:
        raise SamplingFrameError("population selection cutoff is invalid")
    for path_field in (
        "records_path",
        "query_receipt_path",
        "eligibility_protocol_path",
        "selection_code_path",
    ):
        _require_safe_relative_path(population[path_field], field=f"population.{path_field}")
    for digest_field in (
        "records_sha256",
        "query_receipt_sha256",
        "eligibility_protocol_sha256",
        "selection_code_sha256",
    ):
        _require_sha256(population[digest_field], field=f"population.{digest_field}")
    if _require_positive_int(population["record_count"], field="population.record_count") < EXPECTED_TOTAL:
        raise SamplingFrameError("population snapshot is smaller than the selected frame")

    selection = _require_mapping(frame["selection_protocol"], field="selection_protocol")
    _require_exact_keys(selection, _SELECTION_KEYS, field="selection_protocol")
    seed = _require_sha256(selection["seed_sha256"], field="selection_protocol.seed_sha256")
    if (
        selection["algorithm"] != "sha256_seeded_sort_v1"
        or selection["cluster_unit"] != "canonical_github_repository"
        or selection["unique_clusters_across_partitions"] is not True
        or selection["strata"] != EXPECTED_STRATA
    ):
        raise SamplingFrameError("selection protocol differs from the powered design")
    return seed


def _validate_task_manifests(frame: dict[str, Any]) -> None:
    public_tasks = _require_mapping(frame["public_tasks"], field="public_tasks")
    _require_exact_keys(public_tasks, _PUBLIC_MANIFEST_KEYS, field="public_tasks")
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
        raise SamplingFrameError("private labels are not an exact mode-0600 host-only set")


def _validate_selected_rows(frame: dict[str, Any], *, seed: str) -> None:
    entries = frame["entries"]
    if not isinstance(entries, list) or len(entries) != EXPECTED_TOTAL:
        raise SamplingFrameError(f"entries must contain exactly {EXPECTED_TOTAL} tasks")
    rows = [_validate_entry(entry, seed=seed) for entry in entries]
    task_ids = [row["task_id"] for row in rows]
    clusters = [row["provenance_cluster_id"] for row in rows]
    records = [row["public_record_url"] for row in rows]
    source_paths = [row["source_archive_path"] for row in rows]
    if len(set(task_ids)) != EXPECTED_TOTAL:
        raise SamplingFrameError("task identities are not unique")
    if len(set(clusters)) != EXPECTED_TOTAL:
        raise SamplingFrameError("repository clusters are reused; the frame is pseudo-replicated")
    if len(set(records)) != EXPECTED_TOTAL:
        raise SamplingFrameError("public maintenance records are reused")
    if len(set(source_paths)) != EXPECTED_TOTAL:
        raise SamplingFrameError("source archives are reused across independent task units")

    if Counter(row["partition"] for row in rows) != Counter(EXPECTED_COUNTS):
        raise SamplingFrameError("partition counts do not match the powered design")
    observed_strata = Counter((row["partition"], row["stratum"]) for row in rows)
    for partition, strata in EXPECTED_STRATA.items():
        for stratum, count in strata.items():
            if observed_strata[(partition, stratum)] != count:
                raise SamplingFrameError(f"stratum count mismatch for {partition}/{stratum}")
            stratum_rows = sorted(
                (
                    row
                    for row in rows
                    if row["partition"] == partition and row["stratum"] == stratum
                ),
                key=lambda row: row["selection_rank"],
            )
            if [row["selection_rank"] for row in stratum_rows] != list(range(1, count + 1)):
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
    """Validate a complete frame without reading referenced files."""

    frame = _require_mapping(document, field="sampling frame")
    _require_exact_keys(frame, _TOP_LEVEL_KEYS, field="sampling frame")
    _validate_header_and_treatment(frame)
    seed = _validate_population_and_selection(frame)
    _validate_task_manifests(frame)
    _validate_selected_rows(frame, seed=seed)
    expected_digest = _require_sha256(frame["frame_digest"], field="frame_digest")
    digest_payload = dict(frame)
    digest_payload.pop("frame_digest")
    if stable_digest(digest_payload) != expected_digest:
        raise SamplingFrameError("frame_digest does not match the canonical frame")
    return frame


def _read_json(path: Path) -> dict[str, Any]:
    return _require_mapping(json.loads(path.read_text(encoding="utf-8")), field=path.as_posix())


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


def _verify_file(repository_root: Path, path: str, digest: str, *, mode: int | None = None) -> Path:
    target = (repository_root / path).resolve()
    try:
        target.relative_to(repository_root.resolve())
    except ValueError as error:
        raise SamplingFrameError(f"referenced path escaped repository root: {path}") from error
    if not target.is_file() or target.is_symlink():
        raise SamplingFrameError(f"referenced immutable file is absent or a symlink: {path}")
    if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        raise SamplingFrameError(f"referenced file digest mismatch: {path}")
    if mode is not None and stat.S_IMODE(target.stat().st_mode) != mode:
        raise SamplingFrameError(f"referenced private file mode mismatch: {path}")
    return target


def _validate_digested_receipt(
    value: dict[str, Any], *, expected_keys: set[str], field: str
) -> None:
    _require_exact_keys(value, expected_keys, field=field)
    supplied = _require_sha256(value["receipt_digest"], field=f"{field}.receipt_digest")
    payload = dict(value)
    payload.pop("receipt_digest")
    if stable_digest(payload) != supplied:
        raise SamplingFrameError(f"{field} receipt digest mismatch")


def _validate_entry_bundle(repository_root: Path, entry: dict[str, Any]) -> None:
    archive = _verify_file(
        repository_root, entry["source_archive_path"], entry["source_archive_sha256"]
    )
    manifest_path = _verify_file(
        repository_root, entry["source_manifest_path"], entry["source_manifest_sha256"]
    )
    source_manifest = _read_json(manifest_path)
    _require_exact_keys(source_manifest, _SOURCE_MANIFEST_KEYS, field="source manifest")
    if (
        source_manifest["schema_version"] != 1
        or source_manifest["repository_url"] != entry["repository_url"]
        or source_manifest["commit_sha"] != entry["commit_sha"]
        or source_manifest["tree_sha"] != entry["tree_sha"]
        or source_manifest["archive_sha256"] != entry["source_archive_sha256"]
        or hashlib.sha256(archive.read_bytes()).hexdigest() != source_manifest["archive_sha256"]
    ):
        raise SamplingFrameError(f"source manifest identity mismatch for {entry['task_id']}")
    files = source_manifest["files"]
    if not isinstance(files, list) or not files:
        raise SamplingFrameError("source manifest files must be nonempty")
    file_rows: dict[str, dict[str, Any]] = {}
    for item in files:
        row = _require_mapping(item, field="source manifest file")
        _require_exact_keys(row, _SOURCE_FILE_KEYS, field="source manifest file")
        path = _require_safe_relative_path(row["path"], field="source manifest file.path")
        _require_sha256(row["sha256"], field="source manifest file.sha256")
        _require_positive_int(row["size"], field="source manifest file.size")
        if path in file_rows:
            raise SamplingFrameError("source manifest contains duplicate file paths")
        file_rows[path] = row

    stack = entry["target_stack_evidence"]
    expected_source_files = {
        stack["package_manifest_path"]: stack["package_manifest_sha256"],
        stack["dependency_lock_path"]: stack["dependency_lock_sha256"],
        entry["public_test"]["path"]: entry["public_test"]["sha256"],
        entry["license"]["path"]: entry["license"]["sha256"],
    }
    for path in stack["typescript_paths"]:
        if path not in file_rows:
            raise SamplingFrameError(f"TypeScript evidence path absent from source: {path}")
    for path, digest in expected_source_files.items():
        if path not in file_rows or file_rows[path]["sha256"] != digest:
            raise SamplingFrameError(f"source manifest evidence mismatch: {path}")

    provenance_path = _verify_file(
        repository_root,
        entry["provenance_receipt_path"],
        entry["provenance_receipt_sha256"],
    )
    provenance = _read_json(provenance_path)
    _validate_digested_receipt(
        provenance,
        expected_keys=_PROVENANCE_RECEIPT_KEYS,
        field="provenance receipt",
    )
    if (
        provenance["schema_version"] != 1
        or provenance["repository_url"] != entry["repository_url"]
        or provenance["commit_sha"] != entry["commit_sha"]
        or provenance["tree_sha"] != entry["tree_sha"]
        or provenance["public_record_url"] != entry["public_record_url"]
        or provenance["remote_verified"] is not True
        or provenance["license_verified"] is not True
    ):
        raise SamplingFrameError(f"unverified public provenance for {entry['task_id']}")

    verifier = entry["host_verifier"]
    _verify_file(repository_root, verifier["path"], verifier["sha256"], mode=0o600)
    qualification_path = _verify_file(
        repository_root,
        entry["qualification_receipt_path"],
        entry["qualification_receipt_sha256"],
    )
    qualification = _read_json(qualification_path)
    _validate_digested_receipt(
        qualification,
        expected_keys=_QUALIFICATION_RECEIPT_KEYS,
        field="qualification receipt",
    )
    if (
        qualification["schema_version"] != 1
        or qualification["task_id"] != entry["task_id"]
        or qualification["public_test_sha256"] != entry["public_test"]["sha256"]
        or qualification["host_verifier_sha256"] != verifier["sha256"]
        or _SHA256.fullmatch(str(qualification["runtime_lock_sha256"])) is None
        or qualification["base_public_test"] != "failed"
        or qualification["gold_public_test"] != "passed"
        or qualification["base_host_verifier"] != "failed"
        or qualification["gold_host_verifier"] != "passed"
    ):
        raise SamplingFrameError(f"base/gold qualification failed for {entry['task_id']}")


def validate_bundle(frame_path: Path, *, repository_root: Path) -> dict[str, Any]:
    """Validate structure and every immutable input before preview."""

    frame = validate_structure(_read_json(frame_path))
    population = frame["population_snapshot"]
    population_path = _verify_file(
        repository_root, population["records_path"], population["records_sha256"]
    )
    population_rows = _read_jsonl(population_path)
    if len(population_rows) != population["record_count"]:
        raise SamplingFrameError("population snapshot row count changed")
    population_digests = {stable_digest(row) for row in population_rows}
    if len(population_digests) != len(population_rows):
        raise SamplingFrameError("population snapshot contains duplicate records")
    for path_field, digest_field in (
        ("query_receipt_path", "query_receipt_sha256"),
        ("eligibility_protocol_path", "eligibility_protocol_sha256"),
        ("selection_code_path", "selection_code_sha256"),
    ):
        _verify_file(repository_root, population[path_field], population[digest_field])

    public_tasks = frame["public_tasks"]
    public_path = _verify_file(repository_root, public_tasks["path"], public_tasks["sha256"])
    public_rows = _read_jsonl(public_path)
    private_labels = frame["private_labels"]
    private_path = _verify_file(
        repository_root,
        private_labels["path"],
        private_labels["sha256"],
        mode=0o600,
    )
    private_rows = _read_jsonl(private_path)
    if len(public_rows) != EXPECTED_TOTAL or len(private_rows) != EXPECTED_TOTAL:
        raise SamplingFrameError("task or private-label row count changed")
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
        if entry["population_record_digest"] not in population_digests:
            raise SamplingFrameError(f"task absent from population snapshot: {task_id}")
        _validate_entry_bundle(repository_root, entry)

    skill = frame["skill_treatment"]
    _verify_file(repository_root, skill["lock_path"], skill["lock_sha256"])
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "frame_id": FRAME_ID,
        "frame_digest": frame["frame_digest"],
        "counts": EXPECTED_COUNTS,
        "independent_repository_clusters": EXPECTED_TOTAL,
        "public_test_base_fail_gold_pass": EXPECTED_TOTAL,
        "host_verifier_base_fail_gold_pass": EXPECTED_TOTAL,
        "private_values_published": False,
        "development_spec_generation_allowed": True,
        "holdout_spec_generation_allowed": True,
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
