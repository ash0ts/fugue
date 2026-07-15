from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

HARBOR_ARTIFACT_ROOT = PurePosixPath("/logs/artifacts")


@dataclass(frozen=True)
class ArtifactRecovery:
    target: str
    candidates: tuple[str, ...]


def artifact_source_paths(values: list[Any]) -> list[str]:
    paths: list[str] = []
    for value in values:
        source = (
            value
            if isinstance(value, str)
            else value.get("source")
            if isinstance(value, dict)
            else None
        )
        if isinstance(source, str) and source.strip() and source.strip() not in paths:
            paths.append(source.strip())
    return paths


def harbor_artifacts(values: list[Any]) -> list[Any]:
    normalized: list[Any] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, str):
            source = PurePosixPath(value.strip())
            if source != HARBOR_ARTIFACT_ROOT and HARBOR_ARTIFACT_ROOT in source.parents:
                value = HARBOR_ARTIFACT_ROOT.as_posix()
        key = json.dumps(value, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            normalized.append(value)
    return normalized


def artifact_recoveries(
    expected_paths: list[Any], repo_root: str
) -> tuple[ArtifactRecovery, ...]:
    roots = tuple(dict.fromkeys((repo_root, "/workspace", "/root")))
    recoveries: list[ArtifactRecovery] = []
    for value in expected_paths:
        target = PurePosixPath(str(value))
        if (
            not target.is_absolute()
            or target == HARBOR_ARTIFACT_ROOT
            or HARBOR_ARTIFACT_ROOT not in target.parents
            or ".." in target.parts
        ):
            continue
        candidates = tuple(
            candidate.as_posix()
            for root in roots
            if (candidate := PurePosixPath(root) / target.relative_to("/")) != target
        )
        if candidates:
            recoveries.append(
                ArtifactRecovery(target=target.as_posix(), candidates=candidates)
            )
    return tuple(recoveries)
