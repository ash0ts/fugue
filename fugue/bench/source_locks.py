from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from fugue.bench.analysis_contracts import EvidenceDriftCheckV1
from fugue.bench.candidates import stable_digest

LOCAL_SOURCE_LOCK_KIND = "local-comparison-source-lock"
LOCAL_SOURCE_LOCK_SCHEMA_VERSION = 1
_HEX = frozenset("0123456789abcdef")


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value


def _project(value: Any, label: str) -> str:
    if not isinstance(value, str) or value.count("/") != 1:
        raise ValueError(f"{label} must be an entity/project slug")
    entity, project = value.split("/", 1)
    if (
        not entity
        or not project
        or any(part in {".", ".."} for part in (entity, project))
    ):
        raise ValueError(f"{label} must be an entity/project slug")
    return value


def _relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("local source file path must be nonempty text")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("local source file path must be repository-relative")
    return path.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_file(repo_root: Path, relative: str) -> Path:
    root = repo_root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("local source file escapes the repository")
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"local source file is not a regular file: {relative}")
    return path


def build_local_source_lock(
    *,
    repo_root: Path,
    source_project: str,
    result_project: str,
    files: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a secret-free lock for already prepared local source artifacts."""

    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in files:
        unknown = set(raw) - {"path", "role"}
        if unknown:
            raise ValueError(
                "unknown local source file fields: " + ", ".join(sorted(unknown))
            )
        relative = _relative_path(raw.get("path"))
        role = raw.get("role")
        if not isinstance(role, str) or not role.strip():
            raise ValueError("local source file role must be nonempty text")
        if relative in seen:
            raise ValueError(f"duplicate local source file: {relative}")
        seen.add(relative)
        records.append(
            {
                "path": relative,
                "role": role.strip(),
                "sha256": _sha256(_resolve_file(repo_root, relative)),
            }
        )
    if not records:
        raise ValueError("local source lock requires at least one file")
    records.sort(key=lambda item: (item["path"], item["role"]))
    unsigned: dict[str, Any] = {
        "schema_version": LOCAL_SOURCE_LOCK_SCHEMA_VERSION,
        "kind": LOCAL_SOURCE_LOCK_KIND,
        "source_project": _project(source_project, "source project"),
        "result_project": _project(result_project, "result project"),
        "files": records,
        "source_snapshot_digest": stable_digest(records),
    }
    return {**unsigned, "source_lock_digest": stable_digest(unsigned)}


def validate_local_source_lock(
    value: Mapping[str, Any],
    *,
    repo_root: Path,
    expected_source_project: str | None = None,
    expected_result_project: str | None = None,
    verify_files: bool = True,
) -> dict[str, Any]:
    allowed = {
        "schema_version",
        "kind",
        "source_project",
        "result_project",
        "files",
        "source_snapshot_digest",
        "source_lock_digest",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(
            "unknown local source lock fields: " + ", ".join(sorted(unknown))
        )
    if value.get("schema_version") != LOCAL_SOURCE_LOCK_SCHEMA_VERSION:
        raise ValueError("unsupported local source lock schema version")
    if value.get("kind") != LOCAL_SOURCE_LOCK_KIND:
        raise ValueError("local source lock kind does not match")
    source_project = _project(value.get("source_project"), "source project")
    result_project = _project(value.get("result_project"), "result project")
    if expected_source_project and source_project != expected_source_project:
        raise ValueError("local source lock source project does not match")
    if expected_result_project and result_project != expected_result_project:
        raise ValueError("local source lock result project does not match")
    raw_files = value.get("files")
    if not isinstance(raw_files, Sequence) or isinstance(raw_files, str | bytes):
        raise ValueError("local source lock files must be an array")
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in raw_files:
        if not isinstance(raw, Mapping) or set(raw) != {"path", "role", "sha256"}:
            raise ValueError("local source lock file record has invalid fields")
        relative = _relative_path(raw.get("path"))
        role = raw.get("role")
        if not isinstance(role, str) or not role.strip():
            raise ValueError("local source file role must be nonempty text")
        digest = _digest(raw.get("sha256"), "local source file digest")
        if relative in seen:
            raise ValueError(f"duplicate local source file: {relative}")
        seen.add(relative)
        if verify_files and _sha256(_resolve_file(repo_root, relative)) != digest:
            raise ValueError(f"local source file digest changed: {relative}")
        records.append({"path": relative, "role": role.strip(), "sha256": digest})
    if not records:
        raise ValueError("local source lock requires at least one file")
    if records != sorted(records, key=lambda item: (item["path"], item["role"])):
        raise ValueError("local source lock files must be canonically ordered")
    snapshot_digest = _digest(
        value.get("source_snapshot_digest"), "local source snapshot digest"
    )
    if snapshot_digest != stable_digest(records):
        raise ValueError("local source snapshot digest does not match")
    unsigned: dict[str, Any] = {
        "schema_version": LOCAL_SOURCE_LOCK_SCHEMA_VERSION,
        "kind": LOCAL_SOURCE_LOCK_KIND,
        "source_project": source_project,
        "result_project": result_project,
        "files": records,
        "source_snapshot_digest": snapshot_digest,
    }
    lock_digest = _digest(value.get("source_lock_digest"), "local source lock digest")
    if lock_digest != stable_digest(unsigned):
        raise ValueError("local source lock digest does not match")
    return {**unsigned, "source_lock_digest": lock_digest}


def read_local_source_lock(
    path: Path,
    *,
    repo_root: Path,
    expected_source_project: str | None = None,
    expected_result_project: str | None = None,
    verify_files: bool = True,
) -> dict[str, Any]:
    root = repo_root.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("local source lock escapes the repository")
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError("local source lock must be a regular file")
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("local source lock must be an object")
    return validate_local_source_lock(
        raw,
        repo_root=repo_root,
        expected_source_project=expected_source_project,
        expected_result_project=expected_result_project,
        verify_files=verify_files,
    )


def verify_local_source_drift(
    path: Path,
    *,
    repo_root: Path,
    expected_source_project: str,
    expected_result_project: str,
    expected_digest: str,
) -> EvidenceDriftCheckV1:
    try:
        lock = read_local_source_lock(
            path,
            repo_root=repo_root,
            expected_source_project=expected_source_project,
            expected_result_project=expected_result_project,
            verify_files=True,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return EvidenceDriftCheckV1(
            status="unavailable",
            expected_digest=expected_digest,
            reason="the approved local source lock could not be verified",
        )
    observed = str(lock["source_lock_digest"])
    if observed == expected_digest:
        return EvidenceDriftCheckV1(
            status="matched",
            expected_digest=expected_digest,
            observed_digest=observed,
        )
    return EvidenceDriftCheckV1(
        status="drifted",
        expected_digest=expected_digest,
        observed_digest=observed,
        reason="the prepared local source inputs differ from the approved lock",
    )
