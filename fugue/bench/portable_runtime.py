from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import uuid
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path, PurePosixPath
from typing import Any

from filelock import FileLock

from fugue.bench.files import atomic_write_json, docker_build_command
from fugue.bench.files import inspect_docker_image as _inspect_image
from fugue.bench.runtime_provenance import resolve_fugue_distribution_provenance

RUNTIME_ROOT = Path(".fugue/runtime/portable-context-runtime")
_BUILD_RESOURCE_ROOT = ("resources", "runtime", "fugue-context")


def recipe_sha256(_repo_root: Path | None = None) -> str:
    """Hash the immutable context runtime bundled with this distribution.

    ``_repo_root`` remains accepted for persisted callers, but it is deliberately
    not read. The user's study workspace is not a Fugue runtime build input.
    """

    digest = hashlib.sha256()
    for relative, body in _build_context_entries():
        encoded = relative.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(body).to_bytes(8, "big"))
        digest.update(body)
    return digest.hexdigest()


def materialize_build_context(destination: Path) -> None:
    """Materialize only installed-distribution assets into a Docker context."""

    if destination.is_symlink():
        raise ValueError("portable context runtime build directory cannot be a symlink")
    destination.mkdir(parents=True, exist_ok=False)
    for relative, body in _build_context_entries():
        path = PurePosixPath(relative)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError(f"unsafe portable context runtime asset: {relative}")
        output = destination.joinpath(*path.parts)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(body)


def _build_context_entries() -> tuple[tuple[str, bytes], ...]:
    package_root = files("fugue")
    build_root = package_root.joinpath(*_BUILD_RESOURCE_ROOT)
    required = ("Dockerfile", "requirements.lock")
    if not build_root.is_dir():
        raise FileNotFoundError("packaged Fugue context runtime build assets are missing")
    entries: list[tuple[str, bytes]] = []
    for name in required:
        asset = build_root.joinpath(name)
        if not asset.is_file():
            raise FileNotFoundError(
                f"packaged Fugue context runtime asset is missing: {name}"
            )
        entries.append((name, asset.read_bytes()))
    entries.extend(
        (f"fugue/{relative}", item.read_bytes())
        for relative, item in _runtime_python_files(package_root)
    )
    return tuple(sorted(entries))


def _runtime_python_files(
    root: Traversable,
    prefix: str = "",
) -> tuple[tuple[str, Traversable], ...]:
    """Select executable sources without bundled authoring/private evidence.

    The installed distribution contains scaffold labels, reference-study
    inputs, and replay fixtures. Those are trusted-preparation assets and must
    not enter an active context sidecar image. The selected context config and
    prepared corpus arrive through separate, read-only mounts.
    """

    selected: list[tuple[str, Traversable]] = []
    for item in sorted(root.iterdir(), key=lambda candidate: candidate.name):
        if item.name == "__pycache__" or item.name.endswith((".pyc", ".pyo")):
            continue
        relative = f"{prefix}/{item.name}" if prefix else item.name
        if item.is_dir():
            if relative == "resources" or relative.startswith("resources/"):
                continue
            selected.extend(_runtime_python_files(item, relative))
        elif item.is_file() and item.name.endswith(".py"):
            selected.append((relative, item))
    return tuple(selected)


def prepare_runtime(repo_root: Path, *, rebuild: bool = False) -> dict[str, Any]:
    if shutil.which("docker") is None:
        raise RuntimeError("docker is required to prepare portable context runtime")
    root = repo_root / RUNTIME_ROOT
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(root / ".prepare.lock", timeout=1800):
        existing = read_runtime_lock(repo_root)
        if not rebuild and existing is not None:
            try:
                inspected = _inspect_image(str(existing["image_id"]))
            except (OSError, RuntimeError, subprocess.SubprocessError):
                pass
            else:
                if inspected.get("Id") == existing.get("image_id"):
                    return existing
        recipe = recipe_sha256()
        image = f"fugue-context-runtime:{recipe[:12]}"
        build = root / f"build-{uuid.uuid4().hex}"
        try:
            materialize_build_context(build)
            subprocess.run(
                docker_build_command(
                    "--pull",
                    "-f",
                    "Dockerfile",
                    "-t",
                    image,
                    build.as_posix(),
                ),
                cwd=root,
                check=True,
                timeout=1800,
            )
        finally:
            shutil.rmtree(build, ignore_errors=True)
        inspected = _inspect_image(image)
        lock = {
            "schema_version": 1,
            "kind": "portable_context",
            "recipe_sha256": recipe,
            "image": image,
            "image_id": inspected["Id"],
            "architecture": inspected.get("Architecture"),
            "os": inspected.get("Os"),
            "fugue_distribution": resolve_fugue_distribution_provenance(),
        }
        atomic_write_json(root / "runtime-lock.json", lock)
        return lock


def read_runtime_lock(repo_root: Path) -> dict[str, Any] | None:
    path = repo_root / RUNTIME_ROOT / "runtime-lock.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        return None
    expected = {
        "schema_version": 1,
        "kind": "portable_context",
        "recipe_sha256": recipe_sha256(),
    }
    if any(value.get(key) != item for key, item in expected.items()):
        return None
    return value


def runtime_ready(repo_root: Path) -> tuple[bool, str]:
    lock = read_runtime_lock(repo_root)
    if lock is None:
        return False, "run fugue setup --prepare to build the portable runtime"
    try:
        inspected = _inspect_image(str(lock["image_id"]))
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        return False, f"portable runtime image is unavailable: {exc}"
    if inspected.get("Id") != lock.get("image_id"):
        return False, "portable runtime image does not match runtime-lock.json"
    return True, f"{lock['image']} matches {str(lock['image_id'])[:19]}"
