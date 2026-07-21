from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from filelock import FileLock

from fugue.bench.files import atomic_write_json, docker_build_command
from fugue.bench.files import inspect_docker_image as _inspect_image

RUNTIME_ROOT = Path(".fugue/runtime/portable-context-runtime")


def recipe_sha256(repo_root: Path) -> str:
    paths = [
        repo_root / "Dockerfile.context",
        repo_root / "pyproject.toml",
        repo_root / "uv.lock",
        *sorted((repo_root / "fugue").rglob("*.py")),
        *sorted((repo_root / "configs/fugue/context-systems").glob("*.yaml")),
    ]
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(repo_root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        body = path.read_bytes()
        digest.update(len(body).to_bytes(8, "big"))
        digest.update(body)
    return digest.hexdigest()


def runtime_identity(repo_root: Path) -> dict[str, Any]:
    """Return the immutable build recipe without requiring a prepared image."""
    recipe = recipe_sha256(repo_root)
    return {
        "schema_version": 1,
        "kind": "portable_context",
        "recipe_sha256": recipe,
        "image": f"fugue-context-runtime:{recipe[:12]}",
    }


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
        identity = runtime_identity(repo_root)
        image = str(identity["image"])
        subprocess.run(
            docker_build_command(
                "--pull",
                "-f",
                "Dockerfile.context",
                "-t",
                image,
                ".",
            ),
            cwd=repo_root,
            check=True,
            timeout=1800,
        )
        inspected = _inspect_image(image)
        lock = {
            **identity,
            "image_id": inspected["Id"],
            "architecture": inspected.get("Architecture"),
            "os": inspected.get("Os"),
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
        "recipe_sha256": recipe_sha256(repo_root),
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
