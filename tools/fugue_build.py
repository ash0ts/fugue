from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path

SOURCE_COMMIT_RELATIVE = Path("fugue/resources/source-commit.txt")

_COMMIT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


def validate_source_commit(value: str, *, source: str) -> str:
    commit = value.strip()
    if not _COMMIT.fullmatch(commit):
        raise RuntimeError(
            f"{source} must contain one full lowercase Git commit identity"
        )
    return commit


def carried_source_commit(source_root: Path) -> str | None:
    path = source_root / SOURCE_COMMIT_RELATIVE
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("carried Fugue source commit must be a regular file")
    try:
        value = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RuntimeError("unable to read carried Fugue source commit") from exc
    return validate_source_commit(value, source=path.as_posix())


def repository_source_commit(source_root: Path) -> str | None:
    """Read HEAD only when ``source_root`` itself owns the Git metadata.

    Git normally searches parent directories. Build directories and unpacked
    sdists are frequently nested below an unrelated checkout, so accepting that
    discovery would attach another repository's identity to the Fugue wheel.
    """

    metadata = source_root / ".git"
    if not metadata.exists():
        return None
    if metadata.is_symlink() or not (metadata.is_dir() or metadata.is_file()):
        raise RuntimeError("Fugue source .git metadata is not a regular path")
    top_level = _git(source_root, "rev-parse", "--show-toplevel")
    if top_level is None:
        raise RuntimeError("unable to resolve the Fugue source Git root")
    try:
        observed_root = Path(top_level).resolve(strict=True)
        expected_root = source_root.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("unable to validate the Fugue source Git root") from exc
    if observed_root != expected_root:
        raise RuntimeError("Fugue source Git metadata resolves to another repository")
    commit = _git(source_root, "rev-parse", "--verify", "HEAD^{commit}")
    if commit is None:
        raise RuntimeError("unable to resolve the Fugue source commit")
    return validate_source_commit(commit, source="Fugue source Git HEAD")


def resolve_source_commit(
    source_root: Path,
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    values = os.environ if env is None else env
    configured = values.get("FUGUE_BUILD_SOURCE_COMMIT")
    configured_commit = (
        validate_source_commit(
            configured,
            source="FUGUE_BUILD_SOURCE_COMMIT",
        )
        if configured is not None
        else None
    )
    carried = carried_source_commit(source_root)
    repository = repository_source_commit(source_root)
    observed = {
        value
        for value in (configured_commit, carried, repository)
        if value is not None
    }
    if len(observed) > 1:
        raise RuntimeError("Fugue build source commit inputs disagree")
    if not observed:
        raise RuntimeError(
            "Fugue build source commit is unavailable; build from the exact Git "
            "checkout, a Fugue sdist, or set FUGUE_BUILD_SOURCE_COMMIT"
        )
    return observed.pop()


def write_source_commit(root: Path, commit: str) -> Path:
    selected = validate_source_commit(commit, source="Fugue source commit")
    target = root / SOURCE_COMMIT_RELATIVE
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise RuntimeError("refusing to replace a symlinked Fugue source commit")
    target.write_text(selected + "\n", encoding="utf-8")
    return target


def carry_source_commit(
    source_root: Path,
    release_root: Path,
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    commit = resolve_source_commit(source_root, env=env)
    write_source_commit(release_root, commit)
    return commit


def _git(root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None
