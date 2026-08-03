"""Prepare and verify the immutable source tree for the confirmatory Study.

This trusted preparation step uses local Git only.  It deliberately archives a
commit from before this taskset existed and removes host-only evaluation
artifacts from every historical Study.  The Agent receives the implementation,
tests, and public repository evidence needed for planning, never private labels,
reference solutions, answer keys, or recorded answers.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tarfile
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
SCORER = EXAMPLE_ROOT / "plan_quality_scorer_v3.py"
PREREGISTRATION = EXAMPLE_ROOT / "preregistration-confirmatory-v1.json"
SKILL_LOCK = EXAMPLE_ROOT / "skill-revisions.lock.json"
EXPECTED_DEVELOPMENT_TASKS = 8
EXPECTED_HOLDOUT_TASKS = 16
PRIVATE_ORACLE_FILTER_VERSION = 1

_MACHINE_READABLE_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".toml",
    ".tsv",
    ".yaml",
    ".yml",
}
_STUDY_DATA_PREFIXES = (
    "datasets/",
    "examples/comparisons/",
    "fugue/resources/",
)
_PRIVATE_ORACLE_NAME = re.compile(
    r"(?:^|[-_.])(?:answer[-_]?key|gold(?:en)?[-_]?(?:answer|output)|"
    r"oracle|private)(?:[-_.]|$)",
    re.IGNORECASE,
)
_PRIVATE_ORACLE_PAYLOAD = re.compile(
    rb'"(?:adjudicated_label|answer|authored_reference|base_output|expected|'
    rb'gold_output)"\s*:|^\s*expected(?:_paths)?\s*:',
    re.MULTILINE,
)


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


def _git_blobs(paths: tuple[str, ...]) -> dict[str, bytes]:
    if not paths:
        return {}
    payload = "".join(f"{SOURCE_COMMIT}:{path}\n" for path in paths).encode()
    output = subprocess.run(
        ("git", "cat-file", "--batch"),
        cwd=REPO_ROOT,
        check=True,
        input=payload,
        capture_output=True,
    ).stdout
    blobs: dict[str, bytes] = {}
    cursor = 0
    for path in paths:
        newline = output.find(b"\n", cursor)
        if newline < 0:
            raise RuntimeError(f"missing Git object header for source path {path}")
        header = output[cursor:newline].split()
        if len(header) != 3 or header[1] != b"blob":
            raise RuntimeError(f"source path is not a Git blob: {path}")
        try:
            size = int(header[2])
        except ValueError as error:
            raise RuntimeError(f"invalid Git object size for source path {path}") from error
        start = newline + 1
        end = start + size
        if end >= len(output) or output[end : end + 1] != b"\n":
            raise RuntimeError(f"truncated Git object for source path {path}")
        blobs[path] = output[start:end]
        cursor = end + 1
    if cursor != len(output):
        raise RuntimeError("unexpected trailing data from Git source object batch")
    return blobs


def _validate_source_path(path: str) -> None:
    parts = path.split("/")
    if (
        not path
        or path.startswith("/")
        or "\\" in path
        or "\x00" in path
        or "\n" in path
        or "\r" in path
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise RuntimeError(f"unsafe path in confirmatory source tree: {path!r}")


def _private_oracle_reason(path: str, *, content: bytes) -> str | None:
    """Classify host-only evaluation material without inspecting live state.

    The pinned source commit is repository code for the task, so ordinary tests
    and schemas remain visible even when their prose discusses privacy.  The
    filter is limited to Study/evaluation data roots and explicit reference
    solution directories; within those roots it rejects both established path
    conventions and answer-bearing machine-readable payloads.
    """

    _validate_source_path(path)
    folded = path.casefold()
    parts = tuple(part.casefold() for part in path.split("/"))
    suffix = Path(path).suffix.casefold()

    if folded.startswith("configs/fugue/evaluations/"):
        return "evaluation_definition"
    if parts[0] == "datasets" and "solution" in parts:
        return "reference_solution"
    if parts[0] == "datasets" and "tests" in parts:
        return "private_verifier"
    if ".demo_credentials" in parts:
        return "demo_credential_fixture"
    if not folded.startswith(_STUDY_DATA_PREFIXES):
        return None
    if suffix not in _MACHINE_READABLE_SUFFIXES:
        return None
    if _PRIVATE_ORACLE_NAME.search(Path(path).name):
        return "private_oracle_name"
    if _PRIVATE_ORACLE_PAYLOAD.search(content):
        return "answer_bearing_payload"
    return None


def _needs_private_oracle_payload_scan(path: str) -> bool:
    folded = path.casefold()
    return folded.startswith(_STUDY_DATA_PREFIXES) and (
        Path(path).suffix.casefold() in _MACHINE_READABLE_SUFFIXES
    )


def _partition_source_paths(
    source_paths: set[str],
) -> tuple[set[str], dict[str, str]]:
    visible: set[str] = set()
    excluded: dict[str, str] = {}
    for path in source_paths:
        _validate_source_path(path)
    scan_paths = tuple(
        path
        for path in sorted(source_paths)
        if _needs_private_oracle_payload_scan(path)
    )
    blobs = _git_blobs(scan_paths)
    for path in sorted(source_paths):
        reason = _private_oracle_reason(path, content=b"")
        if reason is None and path in blobs:
            reason = _private_oracle_reason(path, content=blobs[path])
        if reason is None:
            visible.add(path)
        else:
            excluded[path] = reason
    if not visible or not excluded:
        raise RuntimeError("private-oracle source filter produced an invalid partition")
    if visible & excluded.keys() or visible | excluded.keys() != source_paths:
        raise RuntimeError("private-oracle source filter did not partition the source")
    return visible, excluded


def _write_filtered_archive(path: Path, *, source_paths: set[str]) -> None:
    """Write and verify one deterministic archive from the allowlisted blobs."""

    if not source_paths:
        raise RuntimeError("cannot archive an empty confirmatory source tree")
    ordered = tuple(sorted(source_paths))
    for source_path in ordered:
        _validate_source_path(source_path)
    subprocess.run(
        (
            "git",
            "archive",
            "--format=tar",
            "--prefix=repo/",
            f"--output={path}",
            SOURCE_COMMIT,
            "--",
            *ordered,
        ),
        cwd=REPO_ROOT,
        check=True,
    )
    _verify_filtered_archive(path, source_paths=set(ordered))


def _verify_filtered_archive(path: Path, *, source_paths: set[str]) -> None:
    with tarfile.open(path, mode="r:") as handle:
        archived = {
            member.name.removeprefix("repo/")
            for member in handle.getmembers()
            if member.isfile()
        }
        unsafe = [
            member.name
            for member in handle.getmembers()
            if not member.name.rstrip("/") == "repo"
            and not member.name.startswith("repo/")
        ]
        unsupported = [
            member.name
            for member in handle.getmembers()
            if not member.isfile() and not member.isdir()
        ]
    if unsafe or unsupported or archived != source_paths:
        raise RuntimeError("filtered confirmatory source archive did not verify")


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
    repository_paths = {
        value
        for value in _git("ls-tree", "-r", "--name-only", SOURCE_COMMIT).splitlines()
        if value
    }
    if any(
        path.startswith("examples/comparisons/superpowers-writing-plans-upgrade/")
        for path in repository_paths
    ):
        raise RuntimeError("Agent-visible source unexpectedly contains this taskset")
    source_paths, excluded_private_oracles = _partition_source_paths(repository_paths)
    tasks, labels = _validate_design()
    _validate_private_oracles(labels, source_paths=source_paths)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    archive = OUTPUT / ARCHIVE_NAME
    temporary = archive.with_name(f".{archive.name}.{os.getpid()}.tmp")
    try:
        _write_filtered_archive(temporary, source_paths=source_paths)
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
            "scope": "repository_tree_without_private_evaluation_artifacts",
            "archive": ARCHIVE_RELATIVE.as_posix(),
            "archive_sha256": _sha256(archive),
            "file_count": len(source_paths),
            "paths_digest": hashlib.sha256(
                "\n".join(sorted(source_paths)).encode()
            ).hexdigest(),
            "contains_task_oracle": False,
            "private_oracle_filter": {
                "version": PRIVATE_ORACLE_FILTER_VERSION,
                "repository_file_count": len(repository_paths),
                "excluded_file_count": len(excluded_private_oracles),
                "excluded_paths": sorted(excluded_private_oracles),
                "excluded_paths_digest": hashlib.sha256(
                    "\n".join(sorted(excluded_private_oracles)).encode()
                ).hexdigest(),
                "reason_counts": {
                    reason: sum(
                        value == reason for value in excluded_private_oracles.values()
                    )
                    for reason in sorted(set(excluded_private_oracles.values()))
                },
            },
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
