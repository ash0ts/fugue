from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

SOURCE_PROVENANCE_SCHEMA_VERSION = 1
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
    """Resolve the executing source state once for an execution plan."""

    root = repo_root.resolve()
    commit = _git(root, "rev-parse", "--verify", "HEAD")
    if commit is None:
        digest, files = _fallback_tree_digest(root)
        return {
            "schema_version": SOURCE_PROVENANCE_SCHEMA_VERSION,
            "kind": "unversioned",
            "dirty": True,
            "digest": digest,
            "files": files,
        }
    status = _git_bytes(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    if status is None:
        raise ValueError(f"unable to inspect Fugue source state: {root}")
    provenance: dict[str, Any] = {
        "schema_version": SOURCE_PROVENANCE_SCHEMA_VERSION,
        "kind": "git",
        "commit": commit.decode().strip(),
        "dirty": bool(status),
    }
    if status:
        provenance["dirty_digest"] = _dirty_tree_digest(root, status)
    return provenance


def _dirty_tree_digest(root: Path, status: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(b"status\0")
    digest.update(status)
    diff = _git_bytes(root, "diff", "--binary", "HEAD", "--")
    if diff is None:
        raise ValueError(f"unable to hash dirty Fugue source: {root}")
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
        raise ValueError(f"unable to hash untracked Fugue source: {root}")
    for raw_path in sorted(item for item in untracked.split(b"\0") if item):
        relative = PurePosixPath(os.fsdecode(raw_path))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe untracked source path: {relative}")
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
