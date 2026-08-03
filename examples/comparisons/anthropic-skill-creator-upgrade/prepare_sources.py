"""Prepare exact Skill sources and deterministic task fixtures without network I/O."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = Path(__file__).resolve().parent
DECLARATION = EXAMPLE / "skill-revisions.lock.json"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / ".fugue/comparison-resources/anthropic-skill-creator-upgrade"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_record(
    repo: Path,
    output: Path,
    *,
    role: str,
    declared: dict[str, object],
    skill_path: str,
) -> dict[str, object]:
    commit = str(declared["commit"])
    resolved = _git(repo, "rev-parse", f"{commit}^{{commit}}")
    if resolved != commit:
        raise RuntimeError(f"{role} did not resolve to exact commit {commit}")
    paths = tuple(
        value
        for value in _git(
            repo,
            "ls-tree",
            "-r",
            "--name-only",
            commit,
            "--",
            skill_path,
        ).splitlines()
        if value
    )
    expected_count = int(declared["file_count"])
    if len(paths) != expected_count:
        raise RuntimeError(
            f"{role} expected {expected_count} files, observed {len(paths)}"
        )
    tree = _git(repo, "rev-parse", f"{commit}:{skill_path}")
    archive = output / f"skill-creator-{role}-{commit}.tar"
    subprocess.run(
        (
            "git",
            "archive",
            "--format=tar",
            "--prefix=source/",
            f"--output={archive}",
            commit,
            skill_path,
        ),
        cwd=repo,
        check=True,
    )
    return {
        "role": role,
        "id": declared["id"],
        "commit": commit,
        "tree": tree,
        "path": skill_path,
        "file_count": len(paths),
        "paths_digest": hashlib.sha256("\n".join(paths).encode()).hexdigest(),
        "archive": _display_path(archive),
        "archive_sha256": _sha256(archive),
        "fugue_bundle_digest": declared["bundle_digest"],
    }


def _fixture_archive(source: Path, destination: Path) -> dict[str, object]:
    paths = tuple(
        sorted(
            path
            for path in source.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        )
    )
    with tarfile.open(destination, "w") as archive:
        for path in paths:
            relative = path.relative_to(source)
            info = archive.gettarinfo(
                path,
                arcname=(Path("workspace") / relative).as_posix(),
            )
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            info.mode = 0o755 if path.name == "quick_validate.py" else 0o644
            with path.open("rb") as handle:
                archive.addfile(info, handle)
    return {
        "id": source.name,
        "archive": _display_path(destination),
        "archive_sha256": _sha256(destination),
        "file_count": len(paths),
        "paths_digest": hashlib.sha256(
            "\n".join(path.relative_to(source).as_posix() for path in paths).encode()
        ).hexdigest(),
    }


def prepare(anthropic_repo: Path, output: Path) -> dict[str, object]:
    declaration = json.loads(DECLARATION.read_text(encoding="utf-8"))
    if declaration.get("repository") != "https://github.com/anthropics/skills":
        raise RuntimeError("unexpected Skill source repository")
    if declaration.get("path") != "skills/skill-creator":
        raise RuntimeError("unexpected Skill source path")
    output.mkdir(parents=True, exist_ok=True)
    sources = [
        _source_record(
            anthropic_repo,
            output,
            role=role,
            declared=declaration[role],
            skill_path=declaration["path"],
        )
        for role in ("baseline", "candidate")
    ]
    fixtures = [
        _fixture_archive(
            EXAMPLE / "fixtures" / fixture,
            output / f"{fixture}.tar",
        )
        for fixture in ("create-skill-workspace", "update-skill-workspace")
    ]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "repository": declaration["repository"],
        "path": declaration["path"],
        "sources": sources,
        "task_fixtures": fixtures,
    }
    manifest["manifest_digest"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (output / "preparation.lock.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--anthropic-repo",
        type=Path,
        required=True,
        help="local anthropics/skills clone containing both exact commits",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.anthropic_repo.resolve(), args.output.resolve())))


if __name__ == "__main__":
    main()
