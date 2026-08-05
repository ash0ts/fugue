from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from fugue.bench.intervention_provenance import (
    build_intervention_component_lock,
    intervention_component_lock_from_dict,
    read_intervention_component_lock,
    verify_intervention_component_checkout,
    write_intervention_component_lock,
)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str, str]:
    root = tmp_path / "component"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.name", "Fugue Tests")
    _git(root, "config", "user.email", "fugue@example.invalid")
    _git(
        root,
        "remote",
        "add",
        "origin",
        "git@github.com:wandb/wandb-mcp-server.git",
    )
    (root / "component.txt").write_text("qualified\n", encoding="utf-8")
    _git(root, "add", "component.txt")
    _git(root, "commit", "-m", "qualified component")
    return root, _git(root, "rev-parse", "HEAD"), _git(
        root, "rev-parse", "HEAD^{tree}"
    )


def test_component_lock_verifies_exact_clean_pr_tree(tmp_path: Path) -> None:
    root, commit, tree = _repo(tmp_path)
    lock = build_intervention_component_lock(
        kind="mcp",
        component_id="wandb-mcp-0-4-loop-fix",
        lock_digest="1" * 64,
        repository="https://github.com/wandb/wandb-mcp-server",
        source_commit=commit,
        source_tree=tree,
        release_target="wandb-mcp-server Python package 0.4",
        superseded_release_candidate_sha="2" * 40,
        release_requalification_required=True,
    )
    path = write_intervention_component_lock(
        tmp_path / "component-lock.json",
        lock,
    )

    assert read_intervention_component_lock(path) == lock
    receipt = verify_intervention_component_checkout(lock, root)
    assert receipt["verified"] is True
    assert receipt["pr_tree_matches_qualified_tree"] is True
    assert receipt["release_requalification_required"] is True
    assert receipt["superseded_release_candidate_sha"] == "2" * 40


def test_component_checkout_rejects_dirty_or_different_tree(
    tmp_path: Path,
) -> None:
    root, commit, tree = _repo(tmp_path)
    lock = build_intervention_component_lock(
        kind="skill",
        component_id="bounded-evidence-skill",
        lock_digest="3" * 64,
        repository="https://github.com/wandb/wandb-mcp-server",
        source_commit=commit,
        source_tree=tree,
    )
    (root / "component.txt").write_text("changed\n", encoding="utf-8")

    receipt = verify_intervention_component_checkout(lock, root)

    assert receipt["verified"] is False
    assert "component worktree is not clean" in receipt["blockers"]


def test_component_lock_fails_closed_on_schema_and_release_impact(
    tmp_path: Path,
) -> None:
    root, commit, tree = _repo(tmp_path)
    with pytest.raises(ValueError, match="only an MCP"):
        build_intervention_component_lock(
            kind="skill",
            component_id="skill",
            lock_digest="4" * 64,
            repository="https://github.com/wandb/wandb-mcp-server",
            source_commit=commit,
            source_tree=tree,
            release_target="wandb-mcp-server Python package 0.4",
            superseded_release_candidate_sha="5" * 40,
            release_requalification_required=True,
        )
    with pytest.raises(ValueError, match="must invalidate"):
        build_intervention_component_lock(
            kind="mcp",
            component_id="unreviewed-release-impact",
            lock_digest="8" * 64,
            repository="https://github.com/wandb/wandb-mcp-server",
            source_commit=commit,
            source_tree=tree,
        )
    with pytest.raises(ValueError, match="must invalidate"):
        build_intervention_component_lock(
            kind="mcp",
            component_id="forked-release-impact",
            lock_digest="9" * 64,
            repository="https://github.com/ash0ts/wandb-mcp-server",
            source_commit=commit,
            source_tree=tree,
        )

    valid = build_intervention_component_lock(
        kind="memory",
        component_id="rag-dense",
        lock_digest="6" * 64,
        repository="https://github.com/wandb/wandb-mcp-server",
        source_commit=commit,
        source_tree=tree,
    ).to_dict()
    valid["unexpected"] = True
    with pytest.raises(ValueError, match="unknown=unexpected"):
        intervention_component_lock_from_dict(valid)

    path = tmp_path / "tampered.json"
    valid.pop("unexpected")
    valid["source_tree"] = "7" * 40
    path.write_text(json.dumps(valid), encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        read_intervention_component_lock(path)
