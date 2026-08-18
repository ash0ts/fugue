from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from fugue.bench.candidates import stable_digest
from fugue.bench.files import atomic_write_json

ComponentKind = Literal["skill", "mcp", "memory"]

_COMPONENT_FIELDS = {
    "schema_version",
    "kind",
    "component_id",
    "lock_digest",
    "repository",
    "source_commit",
    "source_tree",
    "release_target",
    "superseded_release_candidate_sha",
    "release_requalification_required",
    "component_digest",
}
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class InterventionComponentLockV1:
    schema_version: int
    kind: ComponentKind
    component_id: str
    lock_digest: str
    repository: str
    source_commit: str
    source_tree: str
    release_target: str = ""
    superseded_release_candidate_sha: str = ""
    release_requalification_required: bool = False
    component_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_intervention_component_lock(
    *,
    kind: str,
    component_id: str,
    lock_digest: str,
    repository: str,
    source_commit: str,
    source_tree: str,
    release_target: str = "",
    superseded_release_candidate_sha: str = "",
    release_requalification_required: bool = False,
) -> InterventionComponentLockV1:
    normalized_kind = str(kind).strip()
    identifier = str(component_id).strip()
    digest = str(lock_digest).removeprefix("sha256:")
    repo = _canonical_repository(repository)
    commit = str(source_commit).strip()
    tree = str(source_tree).strip()
    target = str(release_target).strip()
    superseded = str(superseded_release_candidate_sha).strip()

    if normalized_kind not in {"skill", "mcp", "memory"}:
        raise ValueError("intervention component kind must be skill, mcp, or memory")
    if not identifier:
        raise ValueError("intervention component requires a stable id")
    if not _HEX_64.fullmatch(digest):
        raise ValueError("intervention component lock digest must be SHA-256")
    if not repo:
        raise ValueError("intervention component requires a source repository")
    if not _HEX_40.fullmatch(commit) or not _HEX_40.fullmatch(tree):
        raise ValueError(
            "intervention component requires full source commit and tree identities"
        )
    # Repository-specific release policy belongs to the reference study or
    # user-authored workflow that owns that release.  The generic lock records
    # an explicitly declared impact; it does not infer one from a repository
    # name or URL.
    if release_requalification_required:
        if normalized_kind != "mcp":
            raise ValueError(
                "only an MCP intervention can invalidate an MCP release lock"
            )
        if not target or not _HEX_40.fullmatch(superseded):
            raise ValueError(
                "release requalification requires a target and superseded SHA"
            )
        if superseded == commit:
            raise ValueError(
                "release requalification requires a changed MCP source commit"
            )
    elif target or superseded:
        raise ValueError(
            "release-impact fields require release requalification to be enabled"
        )

    base = InterventionComponentLockV1(
        schema_version=1,
        kind=normalized_kind,  # type: ignore[arg-type]
        component_id=identifier,
        lock_digest=digest,
        repository=repo,
        source_commit=commit,
        source_tree=tree,
        release_target=target,
        superseded_release_candidate_sha=superseded,
        release_requalification_required=release_requalification_required,
    )
    return InterventionComponentLockV1(
        **{**asdict(base), "component_digest": stable_digest(base.to_dict())}
    )


def intervention_component_lock_from_dict(
    value: Mapping[str, Any],
) -> InterventionComponentLockV1:
    unknown = set(value) - _COMPONENT_FIELDS
    missing = _COMPONENT_FIELDS - set(value)
    if unknown or missing:
        detail = []
        if unknown:
            detail.append("unknown=" + ",".join(sorted(unknown)))
        if missing:
            detail.append("missing=" + ",".join(sorted(missing)))
        raise ValueError(
            "invalid intervention component fields: " + "; ".join(detail)
        )
    if int(value.get("schema_version") or 0) != 1:
        raise ValueError("unsupported intervention component schema")
    rebuilt = build_intervention_component_lock(
        kind=str(value.get("kind") or ""),
        component_id=str(value.get("component_id") or ""),
        lock_digest=str(value.get("lock_digest") or ""),
        repository=str(value.get("repository") or ""),
        source_commit=str(value.get("source_commit") or ""),
        source_tree=str(value.get("source_tree") or ""),
        release_target=str(value.get("release_target") or ""),
        superseded_release_candidate_sha=str(
            value.get("superseded_release_candidate_sha") or ""
        ),
        release_requalification_required=bool(
            value.get("release_requalification_required")
        ),
    )
    if rebuilt.component_digest != str(value.get("component_digest") or ""):
        raise ValueError("intervention component digest does not match its content")
    return rebuilt


def write_intervention_component_lock(
    path: Path,
    lock: InterventionComponentLockV1,
) -> Path:
    payload = lock.to_dict()
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"existing component lock is not JSON: {path}") from exc
        if current != payload:
            raise ValueError(f"intervention component lock already differs: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    return atomic_write_json(path, payload, mode=0o600)


def read_intervention_component_lock(path: Path) -> InterventionComponentLockV1:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"intervention component lock is not JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("intervention component lock must be an object")
    return intervention_component_lock_from_dict(value)


def verify_intervention_component_checkout(
    lock: InterventionComponentLockV1,
    worktree: Path,
) -> dict[str, Any]:
    root = worktree.resolve()
    if not root.is_dir():
        raise ValueError(f"intervention component worktree does not exist: {root}")
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    repository = _canonical_repository(_git(root, "remote", "get-url", "origin"))
    blockers: list[str] = []
    if status:
        blockers.append("component worktree is not clean")
    if commit != lock.source_commit:
        blockers.append("component worktree commit differs from the qualified lock")
    if tree != lock.source_tree:
        blockers.append("component worktree tree differs from the qualified lock")
    if repository != lock.repository:
        blockers.append(
            "component worktree repository differs from the qualified lock"
        )
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "kind": "intervention-component-checkout-verification",
        "component_id": lock.component_id,
        "component_kind": lock.kind,
        "component_digest": lock.component_digest,
        "repository": repository,
        "source_commit": commit,
        "source_tree": tree,
        "clean": not status,
        "pr_tree_matches_qualified_tree": tree == lock.source_tree,
        "release_requalification_required": (
            lock.release_requalification_required
        ),
        "release_target": lock.release_target,
        "superseded_release_candidate_sha": (
            lock.superseded_release_candidate_sha
        ),
        "verified": not blockers,
        "blockers": blockers,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = stable_digest(receipt)
    return receipt


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(
            f"cannot inspect intervention component worktree: {message}"
        )
    return completed.stdout.strip()


def _canonical_repository(value: str) -> str:
    text = str(value).strip().removesuffix("/")
    if not text:
        return ""
    if text.startswith("git@github.com:"):
        text = "https://github.com/" + text.removeprefix("git@github.com:")
    elif text.startswith("ssh://git@github.com/"):
        text = "https://github.com/" + text.removeprefix(
            "ssh://git@github.com/"
        )
    return text.removesuffix(".git")
