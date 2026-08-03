from __future__ import annotations

import json
from pathlib import Path

import pytest

from fugue.bench.source_locks import (
    build_local_source_lock,
    read_local_source_lock,
    validate_local_source_lock,
    verify_local_source_drift,
)


def _write_lock(tmp_path: Path) -> tuple[Path, dict]:
    (tmp_path / "tasks.jsonl").write_text('{"id":"task-a"}\n', encoding="utf-8")
    (tmp_path / "scorer.py").write_text("def score(): return True\n", encoding="utf-8")
    lock = build_local_source_lock(
        repo_root=tmp_path,
        source_project="wandb/source-project",
        result_project="wandb/result-project",
        files=[
            {"path": "tasks.jsonl", "role": "public_tasks"},
            {"path": "scorer.py", "role": "host_scorer"},
        ],
    )
    path = tmp_path / "source.lock.json"
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, lock


def test_local_source_lock_round_trip_and_drift(tmp_path: Path) -> None:
    path, lock = _write_lock(tmp_path)
    loaded = read_local_source_lock(
        path,
        repo_root=tmp_path,
        expected_source_project="wandb/source-project",
        expected_result_project="wandb/result-project",
    )
    assert loaded == lock
    matched = verify_local_source_drift(
        path,
        repo_root=tmp_path,
        expected_source_project="wandb/source-project",
        expected_result_project="wandb/result-project",
        expected_digest=lock["source_lock_digest"],
    )
    assert matched.status == "matched"
    assert matched.observed_digest == lock["source_lock_digest"]

    (tmp_path / "tasks.jsonl").write_text('{"id":"drifted"}\n', encoding="utf-8")
    unavailable = verify_local_source_drift(
        path,
        repo_root=tmp_path,
        expected_source_project="wandb/source-project",
        expected_result_project="wandb/result-project",
        expected_digest=lock["source_lock_digest"],
    )
    assert unavailable.status == "unavailable"
    assert unavailable.reason == "the approved local source lock could not be verified"


def test_local_source_lock_rejects_path_escape_and_unknown_fields(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / "outside-source-lock.txt"
    outside.write_text("outside\n", encoding="utf-8")
    with pytest.raises(ValueError, match="repository-relative"):
        build_local_source_lock(
            repo_root=tmp_path,
            source_project="wandb/source-project",
            result_project="wandb/result-project",
            files=[{"path": "../outside-source-lock.txt", "role": "source"}],
        )
    _, lock = _write_lock(tmp_path)
    with pytest.raises(ValueError, match="unknown local source lock fields"):
        validate_local_source_lock(
            {**lock, "credential": "secret"},
            repo_root=tmp_path,
        )


def test_local_source_lock_detects_digest_and_project_mismatch(tmp_path: Path) -> None:
    path, lock = _write_lock(tmp_path)
    with pytest.raises(ValueError, match="source project does not match"):
        read_local_source_lock(
            path,
            repo_root=tmp_path,
            expected_source_project="wandb/other-project",
        )
    drifted = verify_local_source_drift(
        path,
        repo_root=tmp_path,
        expected_source_project="wandb/source-project",
        expected_result_project="wandb/result-project",
        expected_digest="f" * 64,
    )
    assert drifted.status == "drifted"
    assert drifted.observed_digest == lock["source_lock_digest"]
