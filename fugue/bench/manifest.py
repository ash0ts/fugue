from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from fugue.bench.sources import (
    GitSourceSpec,
    canonical_git_url,
    validate_relative_source_path,
)


@dataclass(frozen=True)
class HttpSourceSpec:
    type: str
    url: str
    sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        parsed = urlparse(self.url)
        if self.type != "http" or parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("HTTP dataset sources require an HTTPS URL")
        if len(self.sha256) != 64 or any(ch not in "0123456789abcdef" for ch in self.sha256):
            raise ValueError("HTTP dataset sources require a lowercase SHA-256")

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "url": self.url, "sha256": self.sha256, **self.metadata}


@dataclass(frozen=True)
class RepositorySpec:
    type: str
    url: str
    commit: str
    path: str | None = None

    def __post_init__(self) -> None:
        if self.type != "git":
            raise ValueError("repository type must be 'git'")
        canonical_git_url(self.url)
        if len(self.commit) != 40 or any(ch not in "0123456789abcdef" for ch in self.commit):
            raise ValueError("repository commit must be a full lowercase Git SHA")
        if self.path:
            validate_relative_source_path(self.path)

    @property
    def slug(self) -> str:
        return canonical_git_url(self.url).removeprefix("https://github.com/")

    def to_dict(self) -> dict[str, str]:
        value = {"type": self.type, "url": canonical_git_url(self.url), "commit": self.commit}
        if self.path:
            value["path"] = self.path
        return value


@dataclass(frozen=True)
class DatasetSpec:
    ref: str | None = None
    version: str | None = None
    path: Path | None = None
    materializer: str | None = None
    source: dict[str, Any] = field(default_factory=dict)
    source_spec: HttpSourceSpec | GitSourceSpec | None = None

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
    repository: RepositorySpec | None = None

    @property
    def repo_slug(self) -> str:
        if self.repository:
            return self.repository.slug
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
    tasks = [_task_spec(item, manifest_path) for item in raw.get("tasks", [])]

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

    source, source_spec = _dataset_source(dataset_raw.get("source"), manifest_path)
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
            source=source,
            source_spec=source_spec,
        ),
        tasks=tasks,
        harnesses=harnesses,
        model=str(raw["model"]) if raw.get("model") else None,
        k=k,
        jobs_dir=_as_path(raw.get("jobs_dir"), "jobs/pilot"),
        n_concurrent=n_concurrent,
    )


def _task_spec(item: Any, manifest_path: Path) -> TaskSpec:
    if not isinstance(item, dict):
        raise ValueError(f"{manifest_path}: task must be a mapping")
    repository_raw = item.get("repository")
    if repository_raw is not None and (item.get("repo") or item.get("base_commit")):
        raise ValueError(
            f"{manifest_path}: task {item.get('id')} may not mix repository with repo/base_commit"
        )
    repository = None
    if repository_raw is not None:
        if not isinstance(repository_raw, dict):
            raise ValueError(f"{manifest_path}: task repository must be a mapping")
        unknown = sorted(set(repository_raw) - {"type", "url", "commit", "path"})
        if unknown:
            raise ValueError(
                f"{manifest_path}: unknown repository field(s): {', '.join(unknown)}"
            )
        repository = RepositorySpec(
            type=str(repository_raw.get("type") or ""),
            url=str(repository_raw.get("url") or ""),
            commit=str(repository_raw.get("commit") or ""),
            path=str(repository_raw["path"]) if repository_raw.get("path") else None,
        )
    return TaskSpec(
        id=str(item["id"]),
        repo=repository.slug if repository else item.get("repo"),
        base_commit=repository.commit if repository else item.get("base_commit"),
        notes=item.get("notes"),
        expected_paths=tuple(str(path) for path in item.get("expected_paths", [])),
        metadata=dict(item.get("metadata") or {}),
        repository=repository,
    )


def _dataset_source(
    value: Any, manifest_path: Path
) -> tuple[dict[str, Any], HttpSourceSpec | GitSourceSpec | None]:
    if value is None:
        return {}, None
    if not isinstance(value, dict):
        raise ValueError(f"{manifest_path}: dataset.source must be a mapping")
    raw = dict(value)
    source_type = raw.get("type")
    if source_type is None:
        return raw, None
    if source_type == "http":
        metadata = {key: item for key, item in raw.items() if key not in {"type", "url", "sha256"}}
        spec = HttpSourceSpec(
            type="http",
            url=str(raw.get("url") or ""),
            sha256=str(raw.get("sha256") or ""),
            metadata=metadata,
        )
        return spec.to_dict(), spec
    if source_type == "git":
        unknown = sorted(set(raw) - {"type", "url", "ref", "path"})
        if unknown:
            raise ValueError(
                f"{manifest_path}: unknown Git dataset source field(s): {', '.join(unknown)}"
            )
        spec = GitSourceSpec(
            type="git",
            url=str(raw.get("url") or ""),
            ref=str(raw.get("ref") or ""),
            path=str(raw["path"]) if raw.get("path") else None,
        )
        if len(spec.ref) != 40 or any(
            char not in "0123456789abcdef" for char in spec.ref
        ):
            raise ValueError(
                f"{manifest_path}: Git dataset source ref must be a full commit SHA"
            )
        return {
            key: item
            for key, item in {
                "type": spec.type,
                "url": canonical_git_url(spec.url),
                "ref": spec.ref,
                "path": spec.path,
            }.items()
            if item is not None
        }, spec
    raise ValueError(f"{manifest_path}: unsupported dataset source type {source_type!r}")


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
