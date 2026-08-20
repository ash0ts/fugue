from __future__ import annotations

import hashlib
import json
import os
import subprocess
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path, PurePosixPath
from typing import Any

SOURCE_PROVENANCE_SCHEMA_VERSION = 1
DISTRIBUTION_PROVENANCE_SCHEMA_VERSION = 1
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
    """Resolve user-workspace source state (legacy public name)."""

    return resolve_workspace_source_provenance(repo_root)


def resolve_workspace_source_provenance(repo_root: Path) -> dict[str, Any]:
    """Resolve the user workspace independently of Fugue's installation."""

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
    tree = _git(root, "rev-parse", "--verify", "HEAD^{tree}")
    if tree is None:
        raise ValueError(f"unable to resolve Fugue source tree: {root}")
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
        "tree": tree.decode().strip(),
        "dirty": bool(status),
    }
    if status:
        provenance["dirty_digest"] = _dirty_tree_digest(root, status)
    return provenance


def resolve_fugue_distribution_provenance() -> dict[str, Any]:
    """Resolve the installed Fugue code and bundled assets without using cwd."""

    digest = hashlib.sha256()
    count = 0
    for package in ("fugue",):
        root = files(package)
        for relative, item in _distribution_files(root):
            digest.update(package.encode())
            digest.update(b"\0")
            digest.update(relative.encode())
            digest.update(b"\0")
            digest.update(item.read_bytes())
            digest.update(b"\0")
            count += 1
    try:
        installed_version = version("fugue")
    except PackageNotFoundError:
        installed_version = "0+uninstalled"
    build = _embedded_build_provenance()
    return {
        "schema_version": DISTRIBUTION_PROVENANCE_SCHEMA_VERSION,
        "kind": "installed_distribution",
        "name": "fugue",
        "version": installed_version,
        "source_commit": build.get("source_commit"),
        "digest": digest.hexdigest(),
        "files": count,
    }


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
