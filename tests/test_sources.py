from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from fugue.bench.sources import (
    SkillSetupRequired,
    _bare_repo_path,
    approve_skill_source,
    canonical_git_url,
    digest_local_skill,
    list_skill_source_ids,
    load_skill_source,
    prepare_skill_source,
    resolve_skill,
    skill_cache_path,
    validate_relative_source_path,
)


def test_git_source_policy_accepts_only_public_github_and_safe_subdirs() -> None:
    assert canonical_git_url("https://github.com/owner/repo.git") == (
        "https://github.com/owner/repo"
    )
    assert validate_relative_source_path("skills/demo") == "skills/demo"
    with pytest.raises(ValueError, match="public"):
        canonical_git_url("ssh://git@github.com/owner/repo")
    with pytest.raises(ValueError, match="without"):
        validate_relative_source_path("../skills/demo")


def test_local_skill_digest_covers_the_entire_bundle(tmp_path: Path) -> None:
    skill = tmp_path / "skill"
    (skill / "references").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\n---\n# Demo\n")
    reference = skill / "references" / "rules.md"
    reference.write_text("one\n")

    first, name = digest_local_skill(skill, fallback_name="fallback")
    reference.write_text("two\n")
    second, _ = digest_local_skill(skill, fallback_name="fallback")

    assert name == "demo"
    assert first.startswith("sha256:")
    assert first != second


def test_checked_in_skill_sources_are_commit_pinned() -> None:
    root = Path(__file__).parents[1]

    assert list_skill_source_ids(root) == [
        "emil-design-eng",
        "hallmark",
        "superpowers-brainstorming",
        "taste-frontend",
    ]
    for skill_id in list_skill_source_ids(root):
        source = load_skill_source(skill_id, root).source
        assert len(source.ref) == 40
        assert set(source.ref) <= set("0123456789abcdef")
        assert source.path


def test_remote_skill_is_inert_until_exact_digest_is_approved(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    skill = upstream / "skills" / "demo"
    (skill / "references").mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: remote-demo\nlicense: MIT\n---\n# Remote demo\n"
    )
    (skill / "references" / "guide.md").write_text("Use the smallest change.\n")
    (upstream / "LICENSE").write_text("MIT\n")
    _git(upstream, "init")
    _git(upstream, "add", ".")
    _git(
        upstream,
        "-c",
        "user.name=Fugue Test",
        "-c",
        "user.email=fugue@example.test",
        "commit",
        "-m",
        "fixture",
    )
    commit = _git(upstream, "rev-parse", "HEAD").stdout.strip()

    source_url = "https://github.com/example/fugue-skill-fixture"
    bare = _bare_repo_path(tmp_path, source_url)
    bare.parent.mkdir(parents=True)
    subprocess.run(
        ["git", "clone", "--bare", upstream.as_posix(), bare.as_posix()],
        check=True,
        capture_output=True,
        text=True,
    )
    source_file = (
        tmp_path / "configs" / "fugue" / "skill-sources" / "remote.yaml"
    )
    source_file.parent.mkdir(parents=True)
    source_file.write_text(
        "\n".join(
            [
                "id: remote",
                "source:",
                "  type: git",
                f"  url: {source_url}",
                f"  ref: {commit}",
                "  path: skills/demo",
                "",
            ]
        )
    )

    inspection = prepare_skill_source("remote", tmp_path)
    with pytest.raises(SkillSetupRequired, match="not approved"):
        resolve_skill("remote", tmp_path)

    review_path = tmp_path / ".fugue" / "runtime" / "skill-reviews" / "remote.yaml"
    review_path.write_text(
        review_path.read_text().replace(
            f"requested_ref: {commit}", f"requested_ref: {'0' * 40}"
        )
    )
    with pytest.raises(ValueError, match="source declaration"):
        approve_skill_source("remote", inspection.digest, tmp_path)

    inspection = prepare_skill_source("remote", tmp_path)
    cached_guide = (
        skill_cache_path(tmp_path, inspection.digest, inspection.declared_name)
        / "references"
        / "guide.md"
    )
    cached_guide.chmod(0o644)
    cached_guide.write_text("tampered\n")
    inspection = prepare_skill_source("remote", tmp_path)
    assert cached_guide.read_text().startswith("Use")

    with pytest.raises(ValueError, match="digest mismatch"):
        approve_skill_source("remote", "sha256:wrong", tmp_path)

    entry = approve_skill_source("remote", inspection.digest, tmp_path)
    resolved = resolve_skill("remote", tmp_path)

    assert entry.resolved_commit == commit
    assert resolved.declared_name == "remote-demo"
    assert resolved.digest == inspection.digest
    assert resolved.source_path == "skills/demo"
    assert (resolved.path / "references" / "guide.md").read_text().startswith("Use")


def _git(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
