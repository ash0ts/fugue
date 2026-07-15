from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

import toml
from filelock import FileLock

from fugue.bench.manifest import BenchmarkManifest, TaskSpec

TASK_RUNTIME_ROOT = Path(".fugue/runtime/task-images")
TASK_RUNTIME_CONTRACT_VERSION = 2


def prepare_task_runtime(
    manifest: BenchmarkManifest,
    task: TaskSpec,
    *,
    repo_root: Path,
    rebuild: bool = False,
) -> dict[str, Any]:
    """Build a task once and publish a local dataset pinned to its image ID."""
    if shutil.which("docker") is None:
        raise RuntimeError("docker is required to prepare task images")
    architecture = task_architecture(task)
    root = _lock_root(manifest, task, repo_root)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(root / f".prepare-{architecture}.lock", timeout=3600):
        source = _resolve_task_source(manifest, task, repo_root)
        source_digest = _tree_digest(source)
        dockerfile = source / "environment" / "Dockerfile"
        if not dockerfile.is_file():
            raise RuntimeError(f"task {task.id} has no environment/Dockerfile")
        recipe_sha256 = _digest(
            {
                "contract_version": TASK_RUNTIME_CONTRACT_VERSION,
                "dataset": manifest.dataset.harbor_ref,
                "task_id": task.id,
                "architecture": architecture,
                "source_sha256": source_digest,
                "trial_policy": {
                    "network_mode": "no-network",
                    "allowed_hosts": [],
                    "image_pull_policy": "never",
                },
            }
        )
        existing = read_task_runtime_lock(manifest, task, repo_root)
        if not rebuild and existing and existing.get("recipe_sha256") == recipe_sha256:
            ready, _ = task_runtime_ready(manifest, task, repo_root)
            if ready:
                return existing

        image = f"fugue-task-{_slug(task.id)}-{architecture}:{recipe_sha256[:12]}"
        subprocess.run(
            [
                "docker",
                "build",
                "--provenance=false",
                "--platform",
                f"linux/{architecture}",
                "--pull",
                "-t",
                image,
                (source / "environment").as_posix(),
            ],
            cwd=repo_root,
            check=True,
            timeout=3600,
        )
        inspected = _inspect_image(image)
        if inspected.get("Architecture") != architecture:
            raise RuntimeError(
                f"task {task.id} built for {inspected.get('Architecture') or 'unknown'}, "
                f"expected {architecture}"
            )

        dataset_root = root / f"dataset-{recipe_sha256[:16]}"
        temporary = root / f".dataset-{uuid.uuid4().hex}.tmp"
        task_root = temporary / task.id
        shutil.copytree(source, task_root, symlinks=True)
        _reject_escaping_symlinks(task_root)
        task_toml = task_root / "task.toml"
        value = toml.loads(task_toml.read_text())
        environment = value.setdefault("environment", {})
        environment.pop("allow_internet", None)
        environment["docker_image"] = str(inspected["Id"])
        environment["network_mode"] = "no-network"
        environment["allowed_hosts"] = []
        task_toml.write_text(toml.dumps(value))
        if dataset_root.exists():
            shutil.rmtree(dataset_root)
        os.replace(temporary, dataset_root)

        lock = {
            "schema_version": 1,
            "contract_version": TASK_RUNTIME_CONTRACT_VERSION,
            "dataset": manifest.dataset.harbor_ref,
            "task_id": task.id,
            "architecture": architecture,
            "source_sha256": source_digest,
            "recipe_sha256": recipe_sha256,
            "image": image,
            "image_id": str(inspected["Id"]),
            "os": str(inspected.get("Os") or "linux"),
            "dataset_path": dataset_root.as_posix(),
        }
        _atomic_json(root / f"runtime-lock-{architecture}.json", lock)
        return lock


def read_task_runtime_lock(
    manifest: BenchmarkManifest,
    task: TaskSpec,
    repo_root: Path,
) -> dict[str, Any] | None:
    architecture = task_architecture(task)
    path = _lock_root(manifest, task, repo_root) / f"runtime-lock-{architecture}.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    required = {
        "schema_version": 1,
        "contract_version": TASK_RUNTIME_CONTRACT_VERSION,
        "dataset": manifest.dataset.harbor_ref,
        "task_id": task.id,
        "architecture": architecture,
    }
    if not isinstance(value, dict) or any(
        value.get(key) != item for key, item in required.items()
    ):
        return None
    image_id = str(value.get("image_id") or "")
    dataset_path = Path(str(value.get("dataset_path") or ""))
    if not image_id.startswith("sha256:") or not dataset_path.is_dir():
        return None
    return value


def task_runtime_ready(
    manifest: BenchmarkManifest,
    task: TaskSpec,
    repo_root: Path,
) -> tuple[bool, str]:
    lock = read_task_runtime_lock(manifest, task, repo_root)
    if lock is None:
        return False, "run fugue setup --prepare to build the locked task image"
    try:
        inspected = _inspect_image(str(lock["image_id"]))
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        return False, f"prepared task image is unavailable: {exc}"
    if inspected.get("Id") != lock.get("image_id"):
        return False, "prepared task image drifted from its lock"
    return True, f"{lock['image_id']} is available"


def task_architecture(task: TaskSpec) -> str:
    architecture = str(task.metadata.get("architecture") or "amd64")
    if architecture not in {"amd64", "arm64"}:
        raise ValueError(
            f"task {task.id} has unsupported architecture {architecture!r}"
        )
    return architecture


def _resolve_task_source(
    manifest: BenchmarkManifest,
    task: TaskSpec,
    repo_root: Path,
) -> Path:
    from harbor.models.job.config import DatasetConfig
    from harbor.tasks.client import TaskClient

    if manifest.dataset.path:
        dataset = manifest.dataset.path
        dataset_path = dataset if dataset.is_absolute() else repo_root / dataset
        config = DatasetConfig(path=dataset_path, task_names=[task.id])
    else:
        task_name = task.id
        if (
            manifest.dataset.ref
            and "/" in manifest.dataset.ref
            and "/" not in task_name
        ):
            task_name = f"{manifest.dataset.ref.split('/', 1)[0]}/{task_name}"
        config = DatasetConfig(
            name=manifest.dataset.ref,
            ref=manifest.dataset.version,
            task_names=[task_name],
        )

    async def resolve() -> Path:
        task_configs = await config.get_task_configs()
        if len(task_configs) != 1:
            raise RuntimeError(f"task resolution returned {len(task_configs)} entries")
        task_id = task_configs[0].get_task_id()
        result = await TaskClient().download_tasks([task_id])
        if len(result.results) != 1:
            raise RuntimeError("task download did not return exactly one result")
        return result.results[0].path.resolve()

    return asyncio.run(resolve())


def _lock_root(manifest: BenchmarkManifest, task: TaskSpec, repo_root: Path) -> Path:
    dataset_key = hashlib.sha256(manifest.dataset.harbor_ref.encode()).hexdigest()[:16]
    return repo_root / TASK_RUNTIME_ROOT / dataset_key / _slug(task.id)


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            digest.update(f"L\0{relative}\0{os.readlink(path)}\0".encode())
        elif path.is_file():
            digest.update(f"F\0{relative}\0".encode())
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _reject_escaping_symlinks(root: Path) -> None:
    resolved_root = root.resolve()
    for path in root.rglob("*"):
        if not path.is_symlink():
            continue
        target = (path.parent / os.readlink(path)).resolve()
        if not target.is_relative_to(resolved_root):
            raise RuntimeError(
                f"task contains escaping symlink: {path.relative_to(root)}"
            )


def _inspect_image(image: str) -> dict[str, Any]:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout or "image missing").strip())
    values = json.loads(result.stdout)
    if not isinstance(values, list) or len(values) != 1:
        raise RuntimeError("docker image inspect returned invalid JSON")
    return values[0]


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value).strip("-").lower()
