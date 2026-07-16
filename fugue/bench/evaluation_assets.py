from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx

from fugue.bench.candidates import stable_digest
from fugue.bench.files import atomic_write_json
from fugue.bench.manifest import BenchmarkManifest

SWE_BENCH_VERIFIED_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
SWE_BENCH_VERIFIED_PARQUET_SHA256 = (
    "a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd"
)
_SOURCE_FILE = "data/test-00000-of-00001.parquet"
_DIFF_PATH = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)


def prepare_evaluation_assets(
    manifest: BenchmarkManifest, repo_root: Path
) -> Path | None:
    if not _uses_pinned_swe_bench_source(manifest):
        return None
    source = _source_path(repo_root)
    _ensure_source(source)
    destination = _lock_path(repo_root)
    if destination.is_file():
        _read_lock(destination)
        return destination
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "SWE-bench evaluation preparation requires the Fugue context extra"
        ) from exc
    rows = parquet.read_table(
        source,
        columns=["instance_id", "patch", "test_patch"],
    ).to_pylist()
    predictions = {
        str(row["instance_id"]): {
            "expected_evidence_paths": _changed_paths(
                str(row.get("patch") or ""),
                str(row.get("test_patch") or ""),
            )
        }
        for row in rows
    }
    base = {
        "schema_version": 1,
        "source": {
            "dataset": "princeton-nlp/SWE-bench_Verified",
            "revision": SWE_BENCH_VERIFIED_REVISION,
            "parquet_sha256": SWE_BENCH_VERIFIED_PARQUET_SHA256,
        },
        "tasks": predictions,
        "lock_sha256": "",
    }
    payload = {**base, "lock_sha256": stable_digest(base)}
    atomic_write_json(destination, payload)
    return destination


def attach_evaluation_assets(
    manifest: BenchmarkManifest,
    repo_root: Path,
    *,
    required: bool,
) -> BenchmarkManifest:
    if not _uses_pinned_swe_bench_source(manifest):
        return manifest
    path = _lock_path(repo_root)
    if not path.is_file():
        if required:
            raise ValueError(
                "host evaluation assets are not prepared; run fugue setup --prepare"
            )
        return manifest
    payload = _read_lock(path)
    tasks = dict(payload["tasks"])
    missing = [task.id for task in manifest.tasks if task.id not in tasks]
    if missing:
        raise ValueError(
            "host evaluation assets are missing task(s): " + ", ".join(missing)
        )
    return replace(
        manifest,
        tasks=[
            replace(
                task,
                expected_paths=tuple(
                    str(value)
                    for value in tasks[task.id].get("expected_evidence_paths") or ()
                ),
            )
            for task in manifest.tasks
        ],
    )


def _uses_pinned_swe_bench_source(manifest: BenchmarkManifest) -> bool:
    return (
        manifest.dataset.ref == "swe-bench/swe-bench-verified"
        and manifest.dataset.source.get("parquet_sha256")
        == SWE_BENCH_VERIFIED_PARQUET_SHA256
        and manifest.dataset.source.get("revision") == SWE_BENCH_VERIFIED_REVISION
    )


def _source_path(repo_root: Path) -> Path:
    return (
        repo_root
        / ".fugue"
        / "cache"
        / "evaluation-sources"
        / SWE_BENCH_VERIFIED_REVISION
        / "verified.parquet"
    )


def _lock_path(repo_root: Path) -> Path:
    return (
        repo_root
        / ".fugue"
        / "evaluation-assets"
        / f"swe-bench-verified-{SWE_BENCH_VERIFIED_PARQUET_SHA256}.json"
    )


def _ensure_source(path: Path) -> None:
    if path.is_file() and _file_sha256(path) == SWE_BENCH_VERIFIED_PARQUET_SHA256:
        return
    if path.exists():
        raise ValueError(f"cached SWE-bench evaluation source digest changed: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/resolve/"
        f"{SWE_BENCH_VERIFIED_REVISION}/{_SOURCE_FILE}"
    )
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        if _file_sha256(temporary) != SWE_BENCH_VERIFIED_PARQUET_SHA256:
            raise ValueError("downloaded SWE-bench evaluation source digest changed")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_lock(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = str(payload.get("lock_sha256") or "")
    if int(payload.get("schema_version") or 0) != 1:
        raise ValueError("unsupported host evaluation asset schema")
    if not expected or stable_digest({**payload, "lock_sha256": ""}) != expected:
        raise ValueError("host evaluation asset lock digest does not match its content")
    source = payload.get("source") or {}
    if source.get("parquet_sha256") != SWE_BENCH_VERIFIED_PARQUET_SHA256:
        raise ValueError("host evaluation assets use a different source")
    return payload


def _changed_paths(patch: str, test_patch: str) -> list[str]:
    return list(
        dict.fromkeys(
            match.group(2)
            for value in (patch, test_patch)
            for match in _DIFF_PATH.finditer(value)
            if match.group(2) != "/dev/null"
        )
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
