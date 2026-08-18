from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.fugue_build import (
    SOURCE_COMMIT_RELATIVE,
    carry_source_commit,
    repository_source_commit,
    resolve_source_commit,
    write_source_commit,
)


def test_source_commit_is_carried_into_a_git_free_release_tree(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    release = tmp_path / "release"
    source.mkdir()
    release.mkdir()
    commit = "a" * 40

    carried = carry_source_commit(
        source,
        release,
        env={"FUGUE_BUILD_SOURCE_COMMIT": commit},
    )

    assert carried == commit
    assert (release / SOURCE_COMMIT_RELATIVE).read_text() == commit + "\n"
    assert resolve_source_commit(release, env={}) == commit


def test_git_free_source_without_a_carried_identity_fails_closed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()

    assert repository_source_commit(source) is None
    with pytest.raises(RuntimeError, match="source commit is unavailable"):
        resolve_source_commit(source, env={})


def test_nested_unrelated_git_repository_is_never_used_as_fugue_identity(
    tmp_path: Path,
) -> None:
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    _git(unrelated, "init")
    _git(unrelated, "config", "user.email", "fugue-build@example.test")
    _git(unrelated, "config", "user.name", "Fugue build test")
    (unrelated / "README.md").write_text("unrelated\n", encoding="utf-8")
    _git(unrelated, "add", "README.md")
    _git(unrelated, "commit", "-m", "unrelated parent")
    parent_commit = _git(unrelated, "rev-parse", "HEAD").stdout.strip()

    source = unrelated / "unpacked-fugue-sdist"
    source.mkdir()
    discovered_parent = _git(source, "rev-parse", "HEAD").stdout.strip()
    assert discovered_parent == parent_commit
    assert repository_source_commit(source) is None

    carried_commit = "b" * 40
    write_source_commit(source, carried_commit)
    assert resolve_source_commit(source, env={}) == carried_commit
    assert carried_commit != parent_commit


def test_invalid_or_disagreeing_source_commit_inputs_are_rejected(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(RuntimeError, match="full lowercase Git commit"):
        resolve_source_commit(
            source,
            env={"FUGUE_BUILD_SOURCE_COMMIT": "not-a-commit"},
        )

    write_source_commit(source, "c" * 40)
    with pytest.raises(RuntimeError, match="inputs disagree"):
        resolve_source_commit(
            source,
            env={"FUGUE_BUILD_SOURCE_COMMIT": "d" * 40},
        )


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
