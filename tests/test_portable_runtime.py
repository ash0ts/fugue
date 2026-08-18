from __future__ import annotations

import json
from pathlib import Path

import pytest

from fugue.bench import portable_runtime


def test_packaged_context_recipe_ignores_user_study_checkout(tmp_path: Path) -> None:
    first = tmp_path / "first-study"
    second = tmp_path / "second-study"
    for root, marker in ((first, "first"), (second, "second")):
        (root / "fugue").mkdir(parents=True)
        (root / "Dockerfile.context").write_text(f"FROM {marker}\n")
        (root / "pyproject.toml").write_text(f"name = {marker!r}\n")
        (root / "uv.lock").write_text(f"revision = {marker!r}\n")
        (root / "fugue/context_server.py").write_text(f"{marker!r}\n")

    assert portable_runtime.recipe_sha256(first) == portable_runtime.recipe_sha256(
        second
    )

    build = tmp_path / "materialized"
    portable_runtime.materialize_build_context(build)

    dockerfile = (build / "Dockerfile").read_text()
    assert "COPY fugue ./fugue" in dockerfile
    assert "--require-hashes" in dockerfile
    assert "Dockerfile.context" not in dockerfile
    assert (build / "requirements.lock").is_file()
    assert (build / "fugue/context_server.py").is_file()
    packaged_paths = {
        path.relative_to(build).as_posix()
        for path in build.rglob("*")
        if path.is_file()
    }
    assert not any("private-labels" in path for path in packaged_paths)
    assert not any(path.startswith("fugue/resources/") for path in packaged_paths)
    assert not any("reference-studies" in path for path in packaged_paths)
    assert not any("source-use-replay" in path for path in packaged_paths)
    assert not any("templates/" in path for path in packaged_paths)
    assert "FROM first" not in dockerfile
    assert "FROM second" not in dockerfile


def test_prepare_runtime_builds_only_the_materialized_distribution_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = tmp_path / "user-study"
    study.mkdir()
    (study / "Dockerfile.context").write_text("FROM user-controlled\n")
    (study / "private-labels.jsonl").write_text('{"secret":"host-only"}\n')
    observed: dict[str, object] = {}

    monkeypatch.setattr(portable_runtime.shutil, "which", lambda _name: "/bin/docker")
    monkeypatch.setattr(
        portable_runtime,
        "docker_build_command",
        lambda *arguments: ["docker", "build", *arguments],
    )
    monkeypatch.setattr(
        portable_runtime,
        "resolve_fugue_distribution_provenance",
        lambda: {"schema_version": 2, "digest": "d" * 64},
    )

    def fake_run(command: list[str], *, cwd: Path, **_kwargs: object) -> None:
        build = Path(command[-1])
        observed["command"] = command
        observed["cwd"] = cwd
        observed["dockerfile"] = (build / "Dockerfile").read_text()
        observed["has_package"] = (build / "fugue/context_server.py").is_file()
        observed["files"] = {
            path.relative_to(build).as_posix()
            for path in build.rglob("*")
            if path.is_file()
        }

    monkeypatch.setattr(portable_runtime.subprocess, "run", fake_run)
    monkeypatch.setattr(
        portable_runtime,
        "_inspect_image",
        lambda _image: {
            "Id": "sha256:" + "a" * 64,
            "Architecture": "arm64",
            "Os": "linux",
        },
    )

    lock = portable_runtime.prepare_runtime(study)

    assert observed["cwd"] == study / portable_runtime.RUNTIME_ROOT
    assert observed["has_package"] is True
    assert "FROM user-controlled" not in str(observed["dockerfile"])
    assert "private-labels.jsonl" not in observed["files"]
    assert lock["recipe_sha256"] == portable_runtime.recipe_sha256()
    assert lock["fugue_distribution"]["digest"] == "d" * 64
    stored = json.loads(
        (study / portable_runtime.RUNTIME_ROOT / "runtime-lock.json").read_text()
    )
    assert stored == lock
    assert not list((study / portable_runtime.RUNTIME_ROOT).glob("build-*"))


def test_materialized_context_rejects_existing_or_symlink_destination(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        portable_runtime.materialize_build_context(existing)

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="cannot be a symlink"):
        portable_runtime.materialize_build_context(link)
