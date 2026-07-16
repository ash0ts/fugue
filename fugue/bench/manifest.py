from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from fugue.bench.files import require_unique
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
        if len(self.sha256) != 64 or any(
            ch not in "0123456789abcdef" for ch in self.sha256
        ):
            raise ValueError("HTTP dataset sources require a lowercase SHA-256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "url": self.url,
            "sha256": self.sha256,
            **self.metadata,
        }


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
        if len(self.commit) != 40 or any(
            ch not in "0123456789abcdef" for ch in self.commit
        ):
            raise ValueError("repository commit must be a full lowercase Git SHA")
        if self.path:
            validate_relative_source_path(self.path)

    @property
    def slug(self) -> str:
        return canonical_git_url(self.url).removeprefix("https://github.com/")

    def to_dict(self) -> dict[str, str]:
        value = {
            "type": self.type,
            "url": canonical_git_url(self.url),
            "commit": self.commit,
        }
        if self.path:
            value["path"] = self.path
        return value


@dataclass(frozen=True)
class FixtureRepositorySpec:
    type: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        if self.type != "fixture":
            raise ValueError("fixture repository type must be 'fixture'")
        validate_relative_source_path(self.path)
        if len(self.sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.sha256
        ):
            raise ValueError("fixture repository requires a lowercase SHA-256")

    @property
    def slug(self) -> str:
        return f"fixture/{Path(self.path).name}"

    def to_dict(self) -> dict[str, str]:
        return {"type": self.type, "path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class DatasetSpec:
    ref: str | None = None
    version: str | None = None
    path: Path | None = None
    materializer: str | None = None
    source: dict[str, Any] = field(default_factory=dict)
    source_spec: HttpSourceSpec | GitSourceSpec | None = None
    verifier_runtime: DatasetVerifierRuntimeSpec | None = None

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
class VerifierRuntimeSpec:
    python_packages: tuple[str, ...]
    test_script_sha256: str

    def __post_init__(self) -> None:
        _validate_exact_python_packages(self.python_packages, "verifier runtime")
        if len(self.test_script_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.test_script_sha256
        ):
            raise ValueError(
                "verifier runtime test script requires a lowercase SHA-256"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_packages": list(self.python_packages),
            "test_script_sha256": self.test_script_sha256,
        }


@dataclass(frozen=True)
class DatasetVerifierRuntimeSpec:
    profile: str
    python_interpreter: str
    python_packages: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.profile != "swebench-v4-offline":
            raise ValueError(f"unsupported dataset verifier profile: {self.profile}")
        _validate_exact_python_packages(
            self.python_packages, "dataset verifier runtime"
        )
        expected_packages = (
            "swebench==4.0.3",
            "datasets==2.16.1",
            "fastcore==1.10.5",
        )
        if self.python_packages != expected_packages:
            raise ValueError(
                "swebench-v4-offline requires its exact pinned Python packages"
            )
        interpreter = Path(self.python_interpreter)
        if not interpreter.is_absolute() or any(
            part == ".." for part in interpreter.parts
        ):
            raise ValueError("dataset verifier Python interpreter must be absolute")
        if interpreter.name != "python" or interpreter.parent.name != "bin":
            raise ValueError(
                "dataset verifier Python interpreter must be a venv bin/python"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "python_interpreter": self.python_interpreter,
            "python_packages": list(self.python_packages),
        }


def _validate_exact_python_packages(packages: tuple[str, ...], label: str) -> None:
    if not packages:
        raise ValueError(f"{label} requires at least one Python package")
    for package in packages:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.+!-]+", package):
            raise ValueError(
                f"{label} Python packages must use exact name==version pins"
            )


@dataclass(frozen=True)
class TaskSpec:
    id: str
    notes: str | None = None
    expected_paths: tuple[str, ...] = ()
    artifacts: tuple[Any, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    repository: RepositorySpec | FixtureRepositorySpec | None = None
    verifier_runtime: VerifierRuntimeSpec | None = None

    @property
    def repo_slug(self) -> str:
        if self.repository:
            return self.repository.slug
        return self.id.split("__", 1)[0]

    @property
    def repo(self) -> str | None:
        return self.repository.slug if self.repository else None

    @property
    def base_commit(self) -> str | None:
        if isinstance(self.repository, RepositorySpec):
            return self.repository.commit
        if isinstance(self.repository, FixtureRepositorySpec):
            return self.repository.sha256
        return None


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
    tasks = [_task_spec(item, manifest_path) for item in raw.get("tasks", [])]

    if not dataset_raw.get("ref") and not dataset_raw.get("path"):
        raise ValueError(f"{manifest_path}: dataset.ref or dataset.path is required")
    if not harnesses:
        raise ValueError(f"{manifest_path}: at least one harness is required")
    if not tasks:
        raise ValueError(f"{manifest_path}: at least one task is required")
    require_unique([harness.name for harness in harnesses], "harness", manifest_path)
    require_unique([task.id for task in tasks], "task", manifest_path)
    k = _positive_int(raw.get("k", 1), "k", manifest_path)
    n_concurrent = _positive_int(
        raw.get("n_concurrent", 2), "n_concurrent", manifest_path
    )

    source, source_spec = _dataset_source(dataset_raw.get("source"), manifest_path)
    dataset_verifier_runtime = _dataset_verifier_runtime(
        dataset_raw.get("verifier_runtime"), manifest_path
    )
    if dataset_verifier_runtime and any(task.verifier_runtime for task in tasks):
        raise ValueError(
            f"{manifest_path}: dataset and task verifier runtimes may not be mixed"
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
            source=source,
            source_spec=source_spec,
            verifier_runtime=dataset_verifier_runtime,
        ),
        tasks=tasks,
        harnesses=harnesses,
        model=str(raw["model"]) if raw.get("model") else None,
        k=k,
        jobs_dir=_as_path(raw.get("jobs_dir"), "jobs/pilot"),
        n_concurrent=n_concurrent,
    )


def _dataset_verifier_runtime(
    value: Any, manifest_path: Path
) -> DatasetVerifierRuntimeSpec | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{manifest_path}: dataset.verifier_runtime must be a mapping")
    unknown = sorted(set(value) - {"profile", "python_interpreter", "python_packages"})
    if unknown:
        raise ValueError(
            f"{manifest_path}: unknown dataset.verifier_runtime field(s): "
            + ", ".join(unknown)
        )
    packages = value.get("python_packages")
    if not isinstance(packages, list) or not all(
        isinstance(package, str) for package in packages
    ):
        raise ValueError(
            f"{manifest_path}: dataset.verifier_runtime.python_packages must be a list"
        )
    return DatasetVerifierRuntimeSpec(
        profile=str(value.get("profile") or ""),
        python_interpreter=str(value.get("python_interpreter") or ""),
        python_packages=tuple(packages),
    )


def _task_spec(item: Any, manifest_path: Path) -> TaskSpec:
    if not isinstance(item, dict):
        raise ValueError(f"{manifest_path}: task must be a mapping")
    artifacts = item.get("artifacts") or []
    if not isinstance(artifacts, list):
        raise ValueError(f"{manifest_path}: task artifacts must be a list")
    if "repo" in item or "base_commit" in item:
        raise ValueError(
            f"{manifest_path}: task {item.get('id')} must use repository instead of "
            "repo/base_commit"
        )
    repository_raw = item.get("repository")
    repository = None
    if repository_raw is not None:
        if not isinstance(repository_raw, dict):
            raise ValueError(f"{manifest_path}: task repository must be a mapping")
        repository_type = str(repository_raw.get("type") or "")
        if repository_type == "fixture":
            unknown = sorted(set(repository_raw) - {"type", "path", "sha256"})
            if unknown:
                raise ValueError(
                    f"{manifest_path}: unknown fixture repository field(s): "
                    + ", ".join(unknown)
                )
            repository = FixtureRepositorySpec(
                type=repository_type,
                path=str(repository_raw.get("path") or ""),
                sha256=str(repository_raw.get("sha256") or ""),
            )
        else:
            unknown = sorted(set(repository_raw) - {"type", "url", "commit", "path"})
            if unknown:
                raise ValueError(
                    f"{manifest_path}: unknown repository field(s): "
                    + ", ".join(unknown)
                )
            repository = RepositorySpec(
                type=repository_type,
                url=str(repository_raw.get("url") or ""),
                commit=str(repository_raw.get("commit") or ""),
                path=(
                    str(repository_raw["path"]) if repository_raw.get("path") else None
                ),
            )
    verifier_runtime_raw = item.get("verifier_runtime")
    verifier_runtime = None
    if verifier_runtime_raw is not None:
        if not isinstance(verifier_runtime_raw, dict):
            raise ValueError(f"{manifest_path}: verifier_runtime must be a mapping")
        unknown = sorted(
            set(verifier_runtime_raw) - {"python_packages", "test_script_sha256"}
        )
        if unknown:
            raise ValueError(
                f"{manifest_path}: unknown verifier_runtime field(s): "
                + ", ".join(unknown)
            )
        packages = verifier_runtime_raw.get("python_packages")
        if not isinstance(packages, list) or not all(
            isinstance(package, str) for package in packages
        ):
            raise ValueError(
                f"{manifest_path}: verifier_runtime.python_packages must be a list"
            )
        verifier_runtime = VerifierRuntimeSpec(
            python_packages=tuple(packages),
            test_script_sha256=str(
                verifier_runtime_raw.get("test_script_sha256") or ""
            ),
        )
    return TaskSpec(
        id=str(item["id"]),
        notes=item.get("notes"),
        expected_paths=tuple(str(path) for path in item.get("expected_paths", [])),
        artifacts=tuple(artifacts),
        metadata=dict(item.get("metadata") or {}),
        repository=repository,
        verifier_runtime=verifier_runtime,
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
        metadata = {
            key: item
            for key, item in raw.items()
            if key not in {"type", "url", "sha256"}
        }
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
    raise ValueError(
        f"{manifest_path}: unsupported dataset source type {source_type!r}"
    )


def _positive_int(value: Any, name: str, path: Path) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{path}: {name} must be positive")
    return parsed


def fixture_repository_digest(root: Path) -> str:
    if not root.is_dir():
        raise ValueError(f"fixture repository does not exist: {root}")
    digest = hashlib.sha256()
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"fixture repository may not contain symlinks: {relative}")
        if not path.is_file():
            continue
        name = relative.encode()
        body = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(body).to_bytes(8, "big"))
        digest.update(body)
    return digest.hexdigest()
