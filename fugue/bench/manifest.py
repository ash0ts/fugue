from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetSpec:
    ref: str
    version: str | None = None

    @property
    def harbor_ref(self) -> str:
        return f"{self.ref}@{self.version}" if self.version else self.ref


@dataclass(frozen=True)
class HarnessSpec:
    name: str
    agent: str
    model: str | None = None


@dataclass(frozen=True)
class TaskSpec:
    id: str
    repo: str | None = None
    base_commit: str | None = None
    notes: str | None = None

    @property
    def repo_slug(self) -> str:
        if self.repo:
            return self.repo
        return self.id.split("__", 1)[0]


@dataclass(frozen=True)
class BenchmarkManifest:
    dataset: DatasetSpec
    tasks: list[TaskSpec]
    conditions: list[str]
    harnesses: list[HarnessSpec]
    model: str | None = None
    k: int = 1
    jobs_dir: Path = Path("jobs/pilot")
    artifact_root: Path = Path("artifacts/memory")
    lock_path: Path = Path("artifacts/lock.json")
    n_concurrent: int = 2

    def select_conditions(self, names: list[str] | None) -> list[str]:
        return _select("condition", self.conditions, names)

    def select_harnesses(self, names: list[str] | None) -> list[HarnessSpec]:
        selected = _select("harness", [h.name for h in self.harnesses], names)
        return [h for h in self.harnesses if h.name in selected]

    def task_by_id(self) -> dict[str, TaskSpec]:
        return {task.id: task for task in self.tasks}


def _select(kind: str, available: list[str], requested: list[str] | None) -> list[str]:
    if not requested:
        return available
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"unknown {kind}(s): {', '.join(missing)}")
    return [item for item in available if item in requested]


def _as_path(value: str | Path | None, default: str) -> Path:
    return Path(value) if value is not None else Path(default)


def load_manifest(path: Path | str) -> BenchmarkManifest:
    manifest_path = Path(path)
    raw = yaml.safe_load(manifest_path.read_text()) or {}
    dataset_raw = raw.get("dataset") or {}
    if isinstance(dataset_raw, str):
        dataset_raw = {"ref": dataset_raw}

    harnesses = [
        HarnessSpec(
            name=str(item["name"]),
            agent=str(item["agent"]),
            model=str(item["model"]) if item.get("model") else None,
        )
        for item in raw.get("harnesses", [])
    ]
    tasks = [
        TaskSpec(
            id=str(item["id"]),
            repo=item.get("repo"),
            base_commit=item.get("base_commit"),
            notes=item.get("notes"),
        )
        for item in raw.get("tasks", [])
    ]

    if not dataset_raw.get("ref"):
        raise ValueError(f"{manifest_path}: dataset.ref is required")
    if not harnesses:
        raise ValueError(f"{manifest_path}: at least one harness is required")
    if not tasks:
        raise ValueError(f"{manifest_path}: at least one task is required")

    return BenchmarkManifest(
        dataset=DatasetSpec(
            ref=str(dataset_raw["ref"]),
            version=dataset_raw.get("version"),
        ),
        tasks=tasks,
        conditions=[str(c) for c in raw.get("conditions", ["none"])],
        harnesses=harnesses,
        model=str(raw["model"]) if raw.get("model") else None,
        k=int(raw.get("k", 1)),
        jobs_dir=_as_path(raw.get("jobs_dir"), "jobs/pilot"),
        artifact_root=_as_path(raw.get("artifact_root"), "artifacts/memory"),
        lock_path=_as_path(raw.get("lock_path"), "artifacts/lock.json"),
        n_concurrent=int(raw.get("n_concurrent", 2)),
    )


def manifest_digest(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    if not path.exists():
        return hasher.hexdigest()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = file.relative_to(path).as_posix()
        hasher.update(rel.encode())
        hasher.update(b"\0")
        hasher.update(file.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def file_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    files: list[dict[str, str]] = []
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        files.append(
            {
                "path": file.relative_to(path).as_posix(),
                "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
            }
        )
    return files


def write_lock(
    *,
    path: Path,
    manifest_path: Path,
    manifest: BenchmarkManifest,
    artifacts: list[dict[str, Any]],
) -> None:
    lock = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": manifest_digest(manifest_path),
        "dataset": asdict(manifest.dataset),
        "model": manifest.model,
        "tasks": [asdict(task) for task in manifest.tasks],
        "conditions": manifest.conditions,
        "harnesses": [asdict(harness) for harness in manifest.harnesses],
        "artifacts": artifacts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")


@dataclass
class PreparedArtifact:
    condition: str
    task_id: str
    path: Path
    builder: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_lock_record(self, root: Path) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "task_id": self.task_id,
            "path": self.path.relative_to(root).as_posix()
            if self.path.is_relative_to(root)
            else self.path.as_posix(),
            "sha256": tree_digest(self.path),
            "files": file_manifest(self.path),
            "builder": self.builder,
            "metadata": self.metadata,
        }
