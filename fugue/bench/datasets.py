from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Protocol

import httpx

from fugue.bench.manifest import BenchmarkManifest, TaskSpec

DATASET_MANIFEST = "fugue-dataset.json"
_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class DatasetMaterializer(Protocol):
    def materialize(
        self,
        manifest: BenchmarkManifest,
        destination: Path,
        source_path: Path,
    ) -> dict[str, Any]: ...


def materialize_manifest_dataset(
    manifest: BenchmarkManifest,
    repo_root: Path,
    *,
    rebuild: bool = False,
) -> Path | None:
    dataset = manifest.dataset
    if not dataset.materializer:
        return None
    if not dataset.path:
        raise ValueError("materialized datasets require dataset.path")
    destination = _resolve(repo_root, dataset.path)
    expected = _dataset_fingerprint(manifest)
    marker = destination / DATASET_MANIFEST
    with _directory_lock(destination):
        if marker.is_file() and not rebuild:
            current = json.loads(marker.read_text())
            if current.get("fingerprint") == expected:
                return destination
            raise ValueError(
                f"cached dataset at {destination} does not match its manifest; "
                "rerun preparation with --rebuild"
            )
        if destination.exists():
            raise ValueError(
                f"refusing to replace existing dataset at {destination}; "
                "remove it explicitly or choose a content-addressed path"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
        )
        try:
            source_path = staging / "_source.jsonl"
            _download_source(dataset.source, source_path)
            materializer = _load_materializer(dataset.materializer)
            metrics = materializer.materialize(manifest, staging, source_path)
            marker_payload = {
                "fingerprint": expected,
                "materializer": dataset.materializer,
                "source": _public_source_metadata(dataset.source),
                "metrics": metrics,
            }
            (staging / DATASET_MANIFEST).write_text(
                json.dumps(marker_payload, indent=2, sort_keys=True) + "\n"
            )
            os.replace(staging, destination)
        except Exception:
            _remove_staging(staging)
            raise
    return destination


class SweQaProMaterializer:
    def materialize(
        self,
        manifest: BenchmarkManifest,
        destination: Path,
        source_path: Path,
    ) -> dict[str, Any]:
        rows = [json.loads(line) for line in source_path.read_text().splitlines() if line]
        selected: list[dict[str, Any]] = []
        for task in manifest.tasks:
            index = task.metadata.get("source_index")
            if not isinstance(index, int) or not 0 <= index < len(rows):
                raise ValueError(f"{task.id}: invalid SWE-QA-Pro source_index")
            row = rows[index]
            _validate_source_row(task, row)
            _write_swe_qa_task(destination / task.id, task, row, index)
            selected.append(
                {
                    "task_id": task.id,
                    "source_index": index,
                    "repo": task.repo,
                    "commit": task.base_commit,
                }
            )
        (destination / "selection.json").write_text(
            json.dumps(selected, indent=2, sort_keys=True) + "\n"
        )
        return {"tasks": len(selected), "source_rows": len(rows)}


def _write_swe_qa_task(
    root: Path,
    task: TaskSpec,
    row: dict[str, Any],
    source_index: int,
) -> None:
    repo = str(task.repo)
    commit = str(task.base_commit)
    root.mkdir(parents=True)
    for name in ("environment", "solution", "tests"):
        (root / name).mkdir()
    (root / "task.toml").write_text(
        "\n".join(
            [
                'schema_version = "1.3"',
                "",
                "[task]",
                f'name = "fugue/{task.id}"',
                'description = "Repository-grounded question answering"',
                "",
                "[metadata]",
                'benchmark = "TIGER-Lab/SWE-QA-Pro-Bench"',
                f"source_index = {source_index}",
                f'repository = "{repo}"',
                f'commit = "{commit}"',
                'license = "MIT"',
                "",
                "[agent]",
                "timeout_sec = 900.0",
                "",
                "[verifier]",
                "timeout_sec = 60.0",
                "",
                "[environment]",
                "build_timeout_sec = 1800.0",
                "cpus = 2",
                "memory_mb = 4096",
                "storage_mb = 10240",
                "",
            ]
        )
    )
    (root / "instruction.md").write_text(
        "\n".join(
            [
                "# Repository question",
                "",
                str(row["question"]).strip(),
                "",
                "Explore the checked-out repository before answering. Write a grounded "
                "answer to `/logs/artifacts/fugue-answer.md`. Fugue records inspected "
                "and changed files automatically.",
                "",
            ]
        )
    )
    (root / "environment" / "Dockerfile").write_text(
        "\n".join(
            [
                "FROM python:3.12.10-slim-bookworm",
                "RUN apt-get update && apt-get install -y --no-install-recommends "
                "ca-certificates git ripgrep && rm -rf /var/lib/apt/lists/*",
                "WORKDIR /workspace/repo",
                f"RUN git clone https://github.com/{repo}.git . && "
                f"git checkout --detach {commit} && rm -rf .git",
                "",
            ]
        )
    )
    (root / "solution" / "reference.md").write_text(str(row["answer"]).strip() + "\n")
    (root / "solution" / "solve.sh").write_text(
        "#!/bin/sh\nmkdir -p /logs/artifacts\n"
        "cp /solution/reference.md /logs/artifacts/fugue-answer.md\n"
    )
    (root / "tests" / "test.sh").write_text(
        "#!/bin/sh\n"
        "mkdir -p /logs/verifier\n"
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "answer = Path('/logs/artifacts/fugue-answer.md')\n"
        "ok = answer.is_file() and bool(answer.read_text().strip())\n"
        "Path('/logs/verifier/reward.json').write_text(" 
        "__import__('json').dumps({'format_completion': float(ok)}))\n"
        "raise SystemExit(0 if ok else 1)\n"
        "PY\n"
    )
    for path in (root / "solution" / "solve.sh", root / "tests" / "test.sh"):
        path.chmod(0o755)


def _validate_source_row(task: TaskSpec, row: dict[str, Any]) -> None:
    repo = str(row.get("repo") or "")
    commit = str(row.get("commit_id") or "")
    if repo != task.repo or commit != task.base_commit:
        raise ValueError(
            f"{task.id}: source row repo/commit changed ({repo}@{commit})"
        )
    if not _REPO.fullmatch(repo) or not _COMMIT.fullmatch(commit):
        raise ValueError(f"{task.id}: unsafe repository or commit metadata")
    if not str(row.get("question") or "").strip() or not str(row.get("answer") or "").strip():
        raise ValueError(f"{task.id}: source row lacks a question or reference answer")


def _download_source(source: dict[str, Any], destination: Path) -> None:
    if source.get("type") not in (None, "http"):
        raise ValueError(
            "this dataset materializer requires an HTTP source; Git dataset sources "
            "must use a Git-aware materializer"
        )
    url = str(source.get("url") or "")
    expected = str(source.get("sha256") or "")
    if not url.startswith("https://") or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("dataset source requires an HTTPS URL and SHA-256")
    digest = hashlib.sha256()
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_bytes():
                digest.update(chunk)
                handle.write(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        destination.unlink(missing_ok=True)
        raise ValueError(f"dataset source checksum mismatch: expected {expected}, got {actual}")


def _load_materializer(import_path: str) -> DatasetMaterializer:
    module_name, separator, object_name = import_path.partition(":")
    if not separator:
        raise ValueError("dataset materializer must use module:Class syntax")
    cls = getattr(importlib.import_module(module_name), object_name)
    return cls()


def _dataset_fingerprint(manifest: BenchmarkManifest) -> str:
    payload = {
        "materializer": manifest.dataset.materializer,
        "path": manifest.dataset.path.as_posix() if manifest.dataset.path else None,
        "source": manifest.dataset.source,
        "tasks": [
            {
                "id": task.id,
                "repo": task.repo,
                "commit": task.base_commit,
                "metadata": task.metadata,
            }
            for task in manifest.tasks
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _public_source_metadata(source: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in source.items()
        if not any(token in key.lower() for token in ("key", "secret", "token"))
    }


@contextlib.contextmanager
def _directory_lock(destination: Path):
    lock = destination.with_name(destination.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 120
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for dataset lock {lock}") from None
            time.sleep(0.1)
    try:
        yield
    finally:
        lock.rmdir()


def _remove_staging(path: Path) -> None:
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), reverse=True):
        child.unlink() if child.is_file() or child.is_symlink() else child.rmdir()
    path.rmdir()


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path
