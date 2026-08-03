"""Prepare and verify the immutable full-tree source for the confirmatory Study.

This trusted preparation step uses local Git only.  It deliberately archives a
commit from before this taskset existed, so neither prompts nor private oracles
can be discovered inside the Agent-visible repository tree.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_ROOT = Path(__file__).resolve().parent
OUTPUT = (
    REPO_ROOT / ".fugue/comparison-resources/superpowers-writing-plans-conference-v1"
)
SOURCE_REPOSITORY = "https://github.com/ash0ts/fugue"
SOURCE_COMMIT = "faa60280841bad8c1a301bd14006d486a86dde5e"
SOURCE_TREE = "b301da496caa8894534c29c43df6b59d60815a57"
ARCHIVE_NAME = f"fugue-full-{SOURCE_COMMIT}.tar"
ARCHIVE_RELATIVE = OUTPUT.relative_to(REPO_ROOT) / ARCHIVE_NAME
TASKS = EXAMPLE_ROOT / "tasks-conference-v1.jsonl"
PRIVATE_LABELS = EXAMPLE_ROOT / "private-labels-conference-v1.jsonl"
SCORER = EXAMPLE_ROOT / "plan_quality_scorer_v2.py"
PREREGISTRATION = EXAMPLE_ROOT / "preregistration-confirmatory-v1.json"
SKILL_LOCK = EXAMPLE_ROOT / "skill-revisions.lock.json"
EXPECTED_DEVELOPMENT_TASKS = 8
EXPECTED_HOLDOUT_TASKS = 16


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _validate_design() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    tasks = _jsonl(TASKS)
    labels = _jsonl(PRIVATE_LABELS)
    task_ids = [str(item.get("id") or "") for item in tasks]
    label_ids = [str(item.get("id") or "") for item in labels]
    if len(task_ids) != len(set(task_ids)) or set(task_ids) != set(label_ids):
        raise RuntimeError("public tasks and private labels must align exactly")
    development = [item for item in tasks if item.get("partition") == "qualification"]
    holdout = [item for item in tasks if item.get("partition") == "holdout"]
    if (len(development), len(holdout)) != (
        EXPECTED_DEVELOPMENT_TASKS,
        EXPECTED_HOLDOUT_TASKS,
    ):
        raise RuntimeError("confirmatory task partition count changed")
    expected_resource = ARCHIVE_RELATIVE.as_posix()
    for task in tasks:
        resources = task.get("resources")
        if not isinstance(resources, list) or resources != [
            {
                "path": expected_resource,
                "target": "/workspace/resources/fugue-source.tar",
            }
        ]:
            raise RuntimeError(f"task {task.get('id')} has an unbound source resource")
    return tasks, labels


def _validate_private_oracles(
    labels: list[dict[str, object]], *, source_paths: set[str]
) -> None:
    for label in labels:
        task_id = str(label.get("id") or "")
        expected = label.get("expected")
        if not isinstance(expected, dict):
            raise RuntimeError(f"task {task_id} has no private oracle")
        allowed = expected.get("allowed_paths")
        if not isinstance(allowed, list) or not allowed:
            raise RuntimeError(f"task {task_id} has no exact allowed path set")
        missing = sorted(str(path) for path in allowed if str(path) not in source_paths)
        if missing:
            raise RuntimeError(
                f"task {task_id} oracle paths are absent from the source tree: {missing}"
            )
        bindings = expected.get("repository_bindings")
        if not isinstance(bindings, list) or not bindings:
            raise RuntimeError(f"task {task_id} has no repository bindings")
        for raw in bindings:
            if not isinstance(raw, dict):
                raise RuntimeError(f"task {task_id} has a malformed binding")
            path = str(raw.get("path") or "")
            symbols = raw.get("symbols")
            if path not in source_paths or not isinstance(symbols, list) or not symbols:
                raise RuntimeError(f"task {task_id} binding is incomplete: {path}")
            content = _git("show", f"{SOURCE_COMMIT}:{path}").casefold()
            present = sum(str(symbol).casefold() in content for symbol in symbols)
            minimum = int(raw.get("minimum_symbols") or len(symbols))
            if present < minimum:
                raise RuntimeError(
                    f"task {task_id} binding {path} has {present}/{minimum} symbols"
                )


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    content = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(content)
        temporary.chmod(0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    resolved_commit = _git("rev-parse", f"{SOURCE_COMMIT}^{{commit}}")
    resolved_tree = _git("rev-parse", f"{SOURCE_COMMIT}^{{tree}}")
    if resolved_commit != SOURCE_COMMIT or resolved_tree != SOURCE_TREE:
        raise RuntimeError("confirmatory source commit or tree changed")
    source_paths = {
        value
        for value in _git("ls-tree", "-r", "--name-only", SOURCE_COMMIT).splitlines()
        if value
    }
    if any(
        path.startswith("examples/comparisons/superpowers-writing-plans-upgrade/")
        for path in source_paths
    ):
        raise RuntimeError("Agent-visible source unexpectedly contains this taskset")
    tasks, labels = _validate_design()
    _validate_private_oracles(labels, source_paths=source_paths)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    archive = OUTPUT / ARCHIVE_NAME
    temporary = archive.with_name(f".{archive.name}.{os.getpid()}.tmp")
    try:
        subprocess.run(
            (
                "git",
                "archive",
                "--format=tar",
                "--prefix=repo/",
                f"--output={temporary}",
                SOURCE_COMMIT,
            ),
            cwd=REPO_ROOT,
            check=True,
        )
        temporary.chmod(0o444)
        os.replace(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "commit": SOURCE_COMMIT,
            "tree": SOURCE_TREE,
            "scope": "complete_repository_tree",
            "archive": ARCHIVE_RELATIVE.as_posix(),
            "archive_sha256": _sha256(archive),
            "file_count": len(source_paths),
            "paths_digest": hashlib.sha256(
                "\n".join(sorted(source_paths)).encode()
            ).hexdigest(),
            "contains_task_oracle": False,
        },
        "design": {
            "development_tasks": EXPECTED_DEVELOPMENT_TASKS,
            "holdout_tasks": EXPECTED_HOLDOUT_TASKS,
            "arms": 2,
            "attempts": 4,
            "planned_cells": len(tasks) * 2 * 4,
            "task_ids_digest": hashlib.sha256(
                "\n".join(sorted(str(task["id"]) for task in tasks)).encode()
            ).hexdigest(),
        },
        "artifacts": {
            "tasks_sha256": _sha256(TASKS),
            "private_labels_sha256": _sha256(PRIVATE_LABELS),
            "scorer_sha256": _sha256(SCORER),
            "preregistration_sha256": _sha256(PREREGISTRATION),
            "skill_lock_sha256": _sha256(SKILL_LOCK),
        },
    }
    manifest["manifest_digest"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _atomic_json(OUTPUT / "preparation.receipt.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
