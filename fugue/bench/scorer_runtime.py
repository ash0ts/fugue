from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from filelock import FileLock

from fugue.bench.candidates import stable_digest
from fugue.bench.files import atomic_write_json, inspect_docker_image
from fugue.bench.task_authoring import ScorerRuntimeProfileV1

SCORER_RUNTIME_ROOT = Path(".fugue/runtime/scorer-runtimes")
SCORER_RUNTIME_CONTRACT_VERSION = 1


def prepare_scorer_runtime(
    profile: ScorerRuntimeProfileV1,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Pull and lock one digest-pinned scorer image during preparation."""

    if shutil.which("docker") is None:
        raise RuntimeError("docker is required to prepare scorer runtimes")
    root = _runtime_root(profile, repo_root)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(root / ".prepare.lock", timeout=900):
        existing = read_scorer_runtime_lock(profile, repo_root=repo_root)
        if existing is not None:
            ready, _detail = scorer_runtime_ready(profile, repo_root=repo_root)
            if ready:
                return existing

        subprocess.run(
            ["docker", "pull", "--platform", profile.platform, profile.image],
            cwd=repo_root,
            check=True,
            timeout=900,
        )
        inspected = inspect_docker_image(profile.image)
        _validate_image(profile, inspected)
        lock = {
            "schema_version": 1,
            "contract_version": SCORER_RUNTIME_CONTRACT_VERSION,
            "profile_id": profile.id,
            "profile_digest": profile.profile_digest,
            "image": profile.image,
            "platform": profile.platform,
            "command": list(profile.command),
            "image_id": str(inspected["Id"]),
            "architecture": str(inspected.get("Architecture") or ""),
            "os": str(inspected.get("Os") or ""),
            "repo_digests": sorted(
                str(item) for item in inspected.get("RepoDigests") or ()
            ),
        }
        atomic_write_json(_lock_path(profile, repo_root), lock)
        return lock


def read_scorer_runtime_lock(
    profile: ScorerRuntimeProfileV1,
    *,
    repo_root: Path,
) -> dict[str, Any] | None:
    path = _lock_path(profile, repo_root)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    required = {
        "schema_version": 1,
        "contract_version": SCORER_RUNTIME_CONTRACT_VERSION,
        "profile_id": profile.id,
        "profile_digest": profile.profile_digest,
        "image": profile.image,
        "platform": profile.platform,
        "command": list(profile.command),
        "architecture": profile.platform.rsplit("/", 1)[-1],
        "os": "linux",
    }
    if not isinstance(value, dict) or any(
        value.get(key) != expected for key, expected in required.items()
    ):
        return None
    image_id = str(value.get("image_id") or "")
    repo_digests = value.get("repo_digests")
    expected_digest = "@sha256:" + profile.image.rsplit("@sha256:", 1)[1]
    if (
        not _is_sha256_image_id(image_id)
        or not isinstance(repo_digests, list)
        or not any(str(item).endswith(expected_digest) for item in repo_digests)
    ):
        return None
    return value


def scorer_runtime_ready(
    profile: ScorerRuntimeProfileV1,
    *,
    repo_root: Path,
) -> tuple[bool, str]:
    lock = read_scorer_runtime_lock(profile, repo_root=repo_root)
    if lock is None:
        return False, "run comparison preparation to pull the pinned scorer image"
    try:
        inspected = inspect_docker_image(profile.image)
        _validate_image(profile, inspected)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        return False, f"prepared scorer runtime is unavailable: {exc}"
    if inspected.get("Id") != lock.get("image_id"):
        return False, "prepared scorer runtime image drifted from its lock"
    return True, f"{profile.image} matches {str(lock['image_id'])[:19]}"


def scorer_runtime_lock_digest(lock: dict[str, Any]) -> str:
    return stable_digest(lock)


def _validate_image(
    profile: ScorerRuntimeProfileV1,
    inspected: dict[str, Any],
) -> None:
    expected_architecture = profile.platform.rsplit("/", 1)[-1]
    architecture = str(inspected.get("Architecture") or "")
    if architecture != expected_architecture:
        raise RuntimeError(
            f"scorer runtime {profile.id} resolved {architecture or 'unknown'}; "
            f"expected {expected_architecture}"
        )
    if str(inspected.get("Os") or "") != "linux":
        raise RuntimeError(f"scorer runtime {profile.id} must resolve a Linux image")
    expected_digest = "@sha256:" + profile.image.rsplit("@sha256:", 1)[1]
    repo_digests = tuple(str(item) for item in inspected.get("RepoDigests") or ())
    if not any(item.endswith(expected_digest) for item in repo_digests):
        raise RuntimeError(
            f"scorer runtime {profile.id} did not resolve its pinned registry digest"
        )


def _runtime_root(profile: ScorerRuntimeProfileV1, repo_root: Path) -> Path:
    architecture = profile.platform.rsplit("/", 1)[-1]
    return repo_root / SCORER_RUNTIME_ROOT / profile.id / architecture


def _lock_path(profile: ScorerRuntimeProfileV1, repo_root: Path) -> Path:
    return _runtime_root(profile, repo_root) / "runtime-lock.json"


def _is_sha256_image_id(value: str) -> bool:
    digest = value.removeprefix("sha256:")
    return (
        value.startswith("sha256:")
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )
