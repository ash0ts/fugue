from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from fugue.bench.scorer_runtime import (
    prepare_scorer_runtime,
    read_scorer_runtime_lock,
    scorer_runtime_lock_digest,
    scorer_runtime_ready,
)
from fugue.bench.task_authoring import ScorerRuntimeProfileV1

_DIGEST = "a" * 64
_IMAGE = f"python:3.12-slim@sha256:{_DIGEST}"


def _profile() -> ScorerRuntimeProfileV1:
    return ScorerRuntimeProfileV1(
        id="python-scorer-v1",
        title="Pinned Python scorer",
        image=_IMAGE,
        platform="linux/amd64",
        command=("python", "/input/scorer.py", "/input/input.json"),
        profile_digest="b" * 64,
    )


def _inspection(*, image_id: str = "sha256:" + "c" * 64) -> dict[str, object]:
    return {
        "Id": image_id,
        "Architecture": "amd64",
        "Os": "linux",
        "RepoDigests": [f"python@sha256:{_DIGEST}"],
    }


def test_prepare_scorer_runtime_pulls_exact_image_and_reuses_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = Mock(return_value=subprocess.CompletedProcess([], 0))
    inspect = Mock(return_value=_inspection())
    monkeypatch.setattr("fugue.bench.scorer_runtime.shutil.which", lambda _name: "docker")
    monkeypatch.setattr("fugue.bench.scorer_runtime.subprocess.run", run)
    monkeypatch.setattr("fugue.bench.scorer_runtime.inspect_docker_image", inspect)
    profile = _profile()

    lock = prepare_scorer_runtime(profile, repo_root=tmp_path)

    run.assert_called_once_with(
        ["docker", "pull", "--platform", "linux/amd64", _IMAGE],
        cwd=tmp_path,
        check=True,
        timeout=900,
    )
    assert lock["profile_digest"] == profile.profile_digest
    assert lock["command"] == list(profile.command)
    assert lock["repo_digests"] == [f"python@sha256:{_DIGEST}"]
    assert len(scorer_runtime_lock_digest(lock)) == 64
    assert read_scorer_runtime_lock(profile, repo_root=tmp_path) == lock
    lock_path = next(tmp_path.glob(".fugue/runtime/scorer-runtimes/**/runtime-lock.json"))
    assert lock_path.stat().st_mode & 0o777 == 0o600

    inspect.return_value = {
        **_inspection(),
        "RepoDigests": [
            f"mirror/python@sha256:{_DIGEST}",
            f"python@sha256:{_DIGEST}",
        ],
    }
    assert prepare_scorer_runtime(profile, repo_root=tmp_path) == lock
    assert run.call_count == 1


def test_scorer_runtime_readiness_detects_missing_and_image_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile()
    assert scorer_runtime_ready(profile, repo_root=tmp_path)[0] is False
    monkeypatch.setattr("fugue.bench.scorer_runtime.shutil.which", lambda _name: "docker")
    monkeypatch.setattr(
        "fugue.bench.scorer_runtime.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0),
    )
    inspection = {"value": _inspection()}
    monkeypatch.setattr(
        "fugue.bench.scorer_runtime.inspect_docker_image",
        lambda _image: inspection["value"],
    )
    prepare_scorer_runtime(profile, repo_root=tmp_path)

    inspection["value"] = _inspection(image_id="sha256:" + "d" * 64)
    ready, detail = scorer_runtime_ready(profile, repo_root=tmp_path)

    assert ready is False
    assert "drifted" in detail


def test_prepare_scorer_runtime_rejects_wrong_platform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("fugue.bench.scorer_runtime.shutil.which", lambda _name: "docker")
    monkeypatch.setattr(
        "fugue.bench.scorer_runtime.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0),
    )
    wrong = {**_inspection(), "Architecture": "arm64"}
    monkeypatch.setattr(
        "fugue.bench.scorer_runtime.inspect_docker_image", lambda _image: wrong
    )

    with pytest.raises(RuntimeError, match="expected amd64"):
        prepare_scorer_runtime(_profile(), repo_root=tmp_path)


def test_scorer_runtime_profile_drift_invalidates_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("fugue.bench.scorer_runtime.shutil.which", lambda _name: "docker")
    monkeypatch.setattr(
        "fugue.bench.scorer_runtime.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0),
    )
    monkeypatch.setattr(
        "fugue.bench.scorer_runtime.inspect_docker_image",
        lambda _image: _inspection(),
    )
    profile = _profile()
    prepare_scorer_runtime(profile, repo_root=tmp_path)

    changed = replace(profile, profile_digest="e" * 64)

    assert read_scorer_runtime_lock(changed, repo_root=tmp_path) is None
    assert scorer_runtime_ready(changed, repo_root=tmp_path)[0] is False
