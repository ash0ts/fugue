from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetSpec:
    ref: str | None = None
    version: str | None = None
    path: Path | None = None
    materializer: str | None = None
    source: dict[str, Any] = field(default_factory=dict)

    @property
    def harbor_ref(self) -> str:
        if not self.ref:
            return self.path.as_posix() if self.path else ""
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
    expected_paths: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def repo_slug(self) -> str:
        if self.repo:
            return self.repo
        return self.id.split("__", 1)[0]


@dataclass(frozen=True)
class BenchmarkManifest:
    dataset: DatasetSpec
    tasks: list[TaskSpec]
    harnesses: list[HarnessSpec]
    model: str | None = None
    k: int = 1
    jobs_dir: Path = Path("jobs/pilot")
    n_concurrent: int = 2

    def select_harnesses(self, names: list[str] | None) -> list[HarnessSpec]:
        selected = _select("harness", [h.name for h in self.harnesses], names)
        return [h for h in self.harnesses if h.name in selected]

def _select(kind: str, available: list[str], requested: list[str] | None) -> list[str]:
    if not requested:
        return available
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"unknown {kind}(s): {', '.join(missing)}")
    return [item for item in available if item in requested]


def _as_path(value: str | Path | None, default: str) -> Path:
    return Path(value) if value is not None else Path(default)


def load_manifest(path: Path | str, *, text: str | None = None) -> BenchmarkManifest:
    manifest_path = Path(path)
    raw = yaml.safe_load(text if text is not None else manifest_path.read_text()) or {}
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
            expected_paths=tuple(str(path) for path in item.get("expected_paths", [])),
            metadata=dict(item.get("metadata") or {}),
        )
        for item in raw.get("tasks", [])
    ]

    if not dataset_raw.get("ref") and not dataset_raw.get("path"):
        raise ValueError(f"{manifest_path}: dataset.ref or dataset.path is required")
    if not harnesses:
        raise ValueError(f"{manifest_path}: at least one harness is required")
    if not tasks:
        raise ValueError(f"{manifest_path}: at least one task is required")
    _require_unique([harness.name for harness in harnesses], "harness", manifest_path)
    _require_unique([task.id for task in tasks], "task", manifest_path)
    k = _positive_int(raw.get("k", 1), "k", manifest_path)
    n_concurrent = _positive_int(
        raw.get("n_concurrent", 2), "n_concurrent", manifest_path
    )

    return BenchmarkManifest(
        dataset=DatasetSpec(
            ref=str(dataset_raw["ref"]) if dataset_raw.get("ref") else None,
            version=dataset_raw.get("version"),
            path=Path(dataset_raw["path"]) if dataset_raw.get("path") else None,
            materializer=(
                str(dataset_raw["materializer"])
                if dataset_raw.get("materializer")
                else None
            ),
            source=dict(dataset_raw.get("source") or {}),
        ),
        tasks=tasks,
        harnesses=harnesses,
        model=str(raw["model"]) if raw.get("model") else None,
        k=k,
        jobs_dir=_as_path(raw.get("jobs_dir"), "jobs/pilot"),
        n_concurrent=n_concurrent,
    )


def _positive_int(value: Any, name: str, path: Path) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{path}: {name} must be positive")
    return parsed


def _require_unique(values: list[str], kind: str, path: Path) -> None:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"{path}: duplicate {kind} id(s): {', '.join(duplicates)}")
