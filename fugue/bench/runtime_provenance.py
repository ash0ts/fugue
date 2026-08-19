from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from importlib.metadata import Distribution, PackageNotFoundError, distribution
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path, PurePosixPath
from typing import Any

STUDY_WORKSPACE_PROVENANCE_SCHEMA_VERSION = 1
# Kept for callers that imported the original constant. New artifacts use the
# ``study_workspace`` field and the canonical name above.
SOURCE_PROVENANCE_SCHEMA_VERSION = STUDY_WORKSPACE_PROVENANCE_SCHEMA_VERSION
DISTRIBUTION_PROVENANCE_SCHEMA_VERSION = 2
_INSTALLED_DISTRIBUTION_DIGEST_KIND = "installed_distribution_contract_v1"
_PACKAGE_CONTENT_FALLBACK_DIGEST_KIND = "package_content_fallback_v1"
_DIST_INFO_CONTRACT_FILES = {
    "METADATA",
    "PKG-INFO",
    "WHEEL",
    "entry_points.txt",
    "namespace_packages.txt",
    "top_level.txt",
}
_DIST_INFO_EXCLUDED_FILES = {
    "direct_url.json",
    "INSTALLER",
    "RECORD",
    "REQUESTED",
}
_DIST_INFO_EXCLUDED_FILE_CASEFOLDS = {
    name.casefold() for name in _DIST_INFO_EXCLUDED_FILES
}
_FALLBACK_EXCLUDED_ROOTS = {
    ".fugue",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "artifacts",
    "build",
    "dist",
    "jobs",
}


def resolve_fugue_source_provenance(repo_root: Path) -> dict[str, Any]:
    """Resolve study-workspace state through the legacy public name."""

    return resolve_study_workspace_provenance(repo_root)


def resolve_workspace_source_provenance(repo_root: Path) -> dict[str, Any]:
    """Resolve study-workspace state through the legacy generic name."""

    return resolve_study_workspace_provenance(repo_root)


def resolve_study_workspace_provenance(repo_root: Path) -> dict[str, Any]:
    """Resolve the user's study workspace independently of Fugue itself."""

    root = repo_root.resolve()
    commit = _git(root, "rev-parse", "--verify", "HEAD")
    if commit is None:
        digest, files = _fallback_tree_digest(root)
        return {
            "schema_version": STUDY_WORKSPACE_PROVENANCE_SCHEMA_VERSION,
            "kind": "unversioned",
            "dirty": True,
            "digest": digest,
            "files": files,
        }
    tree = _git(root, "rev-parse", "--verify", "HEAD^{tree}")
    if tree is None:
        raise ValueError(f"unable to resolve study workspace tree: {root}")
    status = _git_bytes(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    if status is None:
        raise ValueError(f"unable to inspect study workspace state: {root}")
    provenance: dict[str, Any] = {
        "schema_version": STUDY_WORKSPACE_PROVENANCE_SCHEMA_VERSION,
        "kind": "git",
        "commit": commit.decode().strip(),
        "tree": tree.decode().strip(),
        "dirty": bool(status),
    }
    if status:
        provenance["dirty_digest"] = _dirty_tree_digest(root, status)
    return provenance


def study_workspace_provenance(
    value: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Read canonical or legacy study-workspace provenance.

    New execution and snapshot artifacts write ``study_workspace``. Persisted
    V1 artifacts may still contain the historical ``fugue_source`` field. If
    both fields are present they must describe the same workspace; accepting
    conflicting aliases would make the execution identity ambiguous.
    """

    current = value.get("study_workspace")
    legacy = value.get("fugue_source")
    if current is not None and not isinstance(current, Mapping):
        raise ValueError("study workspace provenance must be an object")
    if legacy is not None and not isinstance(legacy, Mapping):
        raise ValueError("legacy study workspace provenance must be an object")
    if current is not None and legacy is not None and dict(current) != dict(legacy):
        raise ValueError(
            "study_workspace and legacy fugue_source provenance disagree"
        )
    selected = current if current is not None else legacy
    return dict(selected) if isinstance(selected, Mapping) else None


def resolve_fugue_distribution_provenance() -> dict[str, Any]:
    """Resolve Fugue's executable installed-distribution contract.

    The digest covers the importable package (including bundled resources) and
    the deterministic distribution metadata that controls installation and
    entry points.  Installer-local records are deliberately excluded: their
    contents describe where or how a particular environment installed the
    wheel, not which Fugue distribution contract is executing.
    """

    entries = _package_contract_entries()
    try:
        installed_distribution = distribution("fugue")
    except PackageNotFoundError:
        installed_distribution = None
        installed_version = "0+uninstalled"
        provenance_kind = "package_content_fallback"
        digest_kind = _PACKAGE_CONTENT_FALLBACK_DIGEST_KIND
        metadata_count = 0
    else:
        installed_version = installed_distribution.version
        provenance_kind = "installed_distribution"
        metadata_entries = _distribution_metadata_contract_entries(
            installed_distribution
        )
        entries.extend(metadata_entries)
        digest_kind = _INSTALLED_DISTRIBUTION_DIGEST_KIND
        metadata_count = len(metadata_entries)
    entries = _unique_contract_entries(entries)
    build = _embedded_build_provenance()
    return {
        "schema_version": DISTRIBUTION_PROVENANCE_SCHEMA_VERSION,
        "kind": provenance_kind,
        "name": "fugue",
        "version": installed_version,
        "source_commit": build.get("source_commit"),
        "digest_kind": digest_kind,
        "digest": _contract_digest(entries, digest_kind=digest_kind),
        "files": len(entries),
        "package_files": len(entries) - metadata_count,
        "metadata_files": metadata_count,
    }


def _package_contract_entries() -> list[tuple[str, bytes]]:
    root = files("fugue")
    return [
        (f"package/fugue/{relative}", item.read_bytes())
        for relative, item in _distribution_files(root)
    ]


def _distribution_metadata_contract_entries(
    installed: Distribution,
) -> list[tuple[str, bytes]]:
    selected: list[tuple[str, bytes]] = []
    declared_files = installed.files
    if declared_files is not None:
        for declared in declared_files:
            relative = PurePosixPath(str(declared))
            contract_path = _metadata_contract_path(relative)
            if contract_path is None:
                continue
            located = Path(installed.locate_file(declared))
            if not located.is_file() or located.is_symlink():
                raise ValueError(
                    "installed Fugue distribution metadata is missing or unsafe: "
                    f"{relative}"
                )
            selected.append((contract_path, located.read_bytes()))
    # Some Distribution providers do not expose ``files``.  Preserve the
    # semantic metadata contract through their standard text API.  It also
    # fills a missing core file from an incomplete provider manifest.
    for source_name, contract_name in (
        ("METADATA", "METADATA"),
        ("PKG-INFO", "METADATA"),
        ("WHEEL", "WHEEL"),
        ("entry_points.txt", "entry_points.txt"),
        ("namespace_packages.txt", "namespace_packages.txt"),
        ("top_level.txt", "top_level.txt"),
    ):
        value = installed.read_text(source_name)
        if value is None:
            continue
        entry = (f"metadata/{contract_name}", value.encode("utf-8"))
        if any(path == entry[0] for path, _ in selected):
            continue
        selected.append(entry)
    normalized = _unique_contract_entries(selected)
    if not any(path == "metadata/METADATA" for path, _ in normalized):
        raise ValueError("installed Fugue distribution exposes no core metadata")
    return normalized


def _metadata_contract_path(relative: PurePosixPath) -> str | None:
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        return None
    metadata_root = relative.parts[0]
    if not metadata_root.endswith((".dist-info", ".egg-info")):
        return None
    metadata_relative = relative.parts[1:]
    if not metadata_relative:
        return None
    name = metadata_relative[-1]
    if name.casefold() in _DIST_INFO_EXCLUDED_FILE_CASEFOLDS:
        return None
    if len(metadata_relative) == 1 and name in _DIST_INFO_CONTRACT_FILES:
        normalized = "METADATA" if name == "PKG-INFO" else name
        return f"metadata/{normalized}"
    if metadata_relative[0] == "licenses" and len(metadata_relative) > 1:
        return "metadata/" + PurePosixPath(*metadata_relative).as_posix()
    return None


def _unique_contract_entries(
    entries: list[tuple[str, bytes]],
) -> list[tuple[str, bytes]]:
    selected: dict[str, bytes] = {}
    for path, content in entries:
        previous = selected.get(path)
        if previous is not None and previous != content:
            raise ValueError(f"conflicting installed distribution content: {path}")
        selected[path] = content
    return sorted(selected.items())


def _contract_digest(
    entries: list[tuple[str, bytes]],
    *,
    digest_kind: str,
) -> str:
    manifest = [
        {
            "path": path,
            "sha256": hashlib.sha256(content).hexdigest(),
            "size": len(content),
        }
        for path, content in entries
    ]
    encoded = json.dumps(
        {
            "digest_kind": digest_kind,
            "files": manifest,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _embedded_build_provenance() -> dict[str, Any]:
    item = files("fugue").joinpath("resources", "build-provenance.json")
    if not item.is_file():
        return {}
    try:
        value = json.loads(item.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _distribution_files(
    root: Traversable,
    prefix: str = "",
) -> list[tuple[str, Traversable]]:
    selected: list[tuple[str, Traversable]] = []
    for item in sorted(root.iterdir(), key=lambda candidate: candidate.name):
        if item.name == "__pycache__" or item.name.endswith((".pyc", ".pyo")):
            continue
        relative = f"{prefix}/{item.name}" if prefix else item.name
        if item.is_dir():
            selected.extend(_distribution_files(item, relative))
        elif item.is_file():
            selected.append((relative, item))
    return selected


def _dirty_tree_digest(root: Path, status: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(b"status\0")
    digest.update(status)
    diff = _git_bytes(root, "diff", "--binary", "HEAD", "--")
    if diff is None:
        raise ValueError(f"unable to hash dirty study workspace: {root}")
    digest.update(b"diff\0")
    digest.update(diff)
    untracked = _git_bytes(
        root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    if untracked is None:
        raise ValueError(f"unable to hash untracked study workspace: {root}")
    for raw_path in sorted(item for item in untracked.split(b"\0") if item):
        relative = PurePosixPath(os.fsdecode(raw_path))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe untracked study workspace path: {relative}")
        source = root / Path(*relative.parts)
        digest.update(b"untracked\0")
        digest.update(raw_path)
        digest.update(b"\0")
        if source.is_symlink():
            digest.update(os.fsencode(os.readlink(source)))
        else:
            digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _fallback_tree_digest(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = 0
    if not root.is_dir():
        return digest.hexdigest(), files
    for source in sorted(root.rglob("*")):
        relative = source.relative_to(root)
        if not relative.parts or relative.parts[0] in _FALLBACK_EXCLUDED_ROOTS:
            continue
        if source.name == ".env" or source.name.startswith(".env."):
            continue
        if source.is_dir() or (not source.is_file() and not source.is_symlink()):
            continue
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        if source.is_symlink():
            digest.update(os.fsencode(os.readlink(source)))
        else:
            digest.update(source.read_bytes())
        digest.update(b"\0")
        files += 1
    return digest.hexdigest(), files


def _git(root: Path, *args: str) -> bytes | None:
    value = _git_bytes(root, *args)
    return value.strip() if value is not None and value.strip() else None


def _git_bytes(root: Path, *args: str) -> bytes | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout if result.returncode == 0 else None
