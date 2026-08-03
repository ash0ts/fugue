"""Prepare and preflight the dependency-free Vercel confirmation fixtures."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import re
import runpy
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_ROOT = Path(__file__).resolve().parent
CATALOG_PATH = EXAMPLE_ROOT / "conference_fixture_catalog.py"
SOURCE_LOCK = EXAMPLE_ROOT / "confirmatory-fixtures.lock.json"
TASKS_PATH = EXAMPLE_ROOT / "confirmatory-tasks.jsonl"
PRIVATE_PATH = EXAMPLE_ROOT / "confirmatory-private-labels.jsonl"
SCORER_PATH = EXAMPLE_ROOT / "vercel_confirmatory_scorer.py"
HOST_VERIFIER_PATH = EXAMPLE_ROOT / "host_node_verifier.cjs"
OUTPUT_ROOT = (
    REPO_ROOT
    / ".fugue/comparison-resources/vercel-react-best-practices-confirmatory-v1"
)
NODE_IMAGE = "node:22-bookworm-slim@sha256:53ada149d435c38b14476cb57e4a7da73c15595aba79bd6971b547ceb6d018bf"
_FROZEN_SOURCE_NAMES = (
    "conference_fixture_catalog.py",
    "confirmatory-tasks.jsonl",
    "confirmatory-private-labels.jsonl",
    "vercel_confirmatory_scorer.py",
    "host_node_verifier.cjs",
    "confirmatory-preregistration.json",
    "prepare_confirmatory_fixtures.py",
)
_RESERVED_PUBLIC_PATHS = frozenset(
    {"readme.md", "package.json", "tests/task.test.mjs"}
)


def _stable_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_catalog() -> tuple[dict[str, Any], ...]:
    spec = importlib.util.spec_from_file_location(
        "vercel_confirmatory_catalog", CATALOG_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load fixture catalog")
    module = importlib.util.module_from_spec(spec)
    assert isinstance(module, ModuleType)
    spec.loader.exec_module(module)
    fixtures = tuple(module.FIXTURES)
    if len(fixtures) != 24 or len({item["id"] for item in fixtures}) != 24:
        raise RuntimeError("confirmatory catalog must contain 24 unique fixtures")
    if sum(item["split"] == "discovery" for item in fixtures) != 8:
        raise RuntimeError(
            "confirmatory catalog must contain eight development fixtures"
        )
    if sum(item["split"] == "holdout" for item in fixtures) != 16:
        raise RuntimeError("confirmatory catalog must contain sixteen holdouts")
    for fixture in fixtures:
        _validated_target_files(fixture)
    return fixtures


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _expected_contract(fixture: dict[str, Any]) -> dict[str, Any]:
    archive = _archive_contract(fixture)
    return {
        "task_id": str(fixture["id"]),
        "required_file_paths": list(fixture["target_files"]),
        "allowed_file_paths": list(fixture["target_files"]),
        "required_inspected_paths": list(fixture["required_inspected_paths"]),
        "public_test_name": str(fixture["public_test_name"]),
        **archive,
        "verifier": dict(fixture["verifier"]),
    }


def refresh_frozen_sources() -> dict[str, Any]:
    """Regenerate host-only labels and their self-authenticating source lock."""

    fixtures = _load_catalog()
    labels = _load_jsonl(PRIVATE_PATH)
    labels_by_id = {str(item["id"]): item for item in labels}
    refreshed = []
    for fixture in fixtures:
        task_id = str(fixture["id"])
        label = dict(labels_by_id[task_id])
        label["expected"] = _expected_contract(fixture)
        label["base_output"] = _frozen_label_output(fixture, gold=False)
        label["gold_output"] = _frozen_label_output(fixture, gold=True)
        refreshed.append(label)
    PRIVATE_PATH.write_text(
        "".join(
            json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
            for item in refreshed
        ),
        encoding="utf-8",
    )
    lock = {
        "schema_version": 1,
        "sources": [
            {
                "path": name,
                "sha256": _sha256(EXAMPLE_ROOT / name),
                "size": (EXAMPLE_ROOT / name).stat().st_size,
            }
            for name in _FROZEN_SOURCE_NAMES
        ],
    }
    lock["manifest_digest"] = _stable_digest(lock)
    SOURCE_LOCK.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return lock


def _load_and_verify_source_lock() -> dict[str, Any]:
    value = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
    supplied = str(value.pop("manifest_digest", ""))
    if (
        set(value) != {"schema_version", "sources"}
        or value.get("schema_version") != 1
        or supplied != _stable_digest(value)
    ):
        raise RuntimeError("confirmatory fixture lock digest does not match")
    records = value.get("sources")
    if not isinstance(records, list) or not records:
        raise RuntimeError("confirmatory fixture lock is empty")
    if not all(isinstance(record, dict) for record in records):
        raise RuntimeError("confirmatory fixture lock record is invalid")
    record_paths = tuple(str(record.get("path") or "") for record in records)
    if record_paths != _FROZEN_SOURCE_NAMES or len(set(record_paths)) != len(
        record_paths
    ):
        raise RuntimeError(
            "confirmatory fixture lock source records are not exact and unique"
        )
    for record in records:
        if set(record) != {"path", "sha256", "size"}:
            raise RuntimeError("confirmatory fixture lock record is invalid")
        relative = str(record.get("path") or "")
        path = EXAMPLE_ROOT / relative
        if (
            not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or not path.is_file()
            or path.is_symlink()
        ):
            raise RuntimeError(f"unsafe confirmatory source path: {relative}")
        if path.stat().st_size != record.get("size") or _sha256(path) != record.get(
            "sha256"
        ):
            raise RuntimeError(f"confirmatory source changed: {relative}")
    return {**value, "manifest_digest": supplied}


def _validated_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RuntimeError(f"unsafe confirmatory fixture path: {value}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RuntimeError(f"unsafe confirmatory fixture path: {value}")
    return path.as_posix()


def _validated_target_files(fixture: dict[str, Any]) -> dict[str, dict[str, str]]:
    raw_targets = fixture.get("target_files")
    if not isinstance(raw_targets, dict) or not raw_targets:
        raise RuntimeError("confirmatory fixture target_files must be a nonempty object")
    targets: dict[str, dict[str, str]] = {}
    casefolded = set(_RESERVED_PUBLIC_PATHS)
    for raw_path, raw_sources in raw_targets.items():
        relative = _validated_relative_path(raw_path)
        folded = relative.casefold()
        if folded in casefolded:
            raise RuntimeError(
                f"confirmatory fixture target path collides with another input: {raw_path}"
            )
        if (
            not isinstance(raw_sources, dict)
            or set(raw_sources) != {"base", "gold"}
            or not all(isinstance(value, str) for value in raw_sources.values())
        ):
            raise RuntimeError(
                f"confirmatory fixture sources are invalid for target: {raw_path}"
            )
        casefolded.add(folded)
        targets[relative] = {
            "base": raw_sources["base"],
            "gold": raw_sources["gold"],
        }
    return targets


def _public_files(fixture: dict[str, Any], *, gold: bool) -> dict[str, bytes]:
    task_id = str(fixture["id"])
    targets = _validated_target_files(fixture)
    files: dict[str, bytes] = {
        "package.json": (
            json.dumps(
                {
                    "name": task_id,
                    "private": True,
                    "type": "module",
                    "scripts": {"test": "node --test"},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode(),
        "README.md": (
            f"# {fixture['title']}\n\n"
            f"Reported symptom: {fixture['symptom']}\n\n"
            "Repair the behavior without adding dependencies or changing unrelated files. "
            "Run `node --test` before returning the complete changed source files.\n"
        ).encode(),
        "tests/task.test.mjs": str(fixture["public_test_source"]).encode(),
    }
    side = "gold" if gold else "base"
    for path, sources in targets.items():
        files[path] = sources[side].encode()
    return files


def _archive_bytes(fixture: dict[str, Any]) -> bytes:
    files = _public_files(fixture, gold=False)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for relative in sorted(files):
            data = files[relative]
            info = tarfile.TarInfo(f"repo/{relative}")
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = "root"
            info.gname = "root"
            archive.addfile(info, io.BytesIO(data))
    return stream.getvalue()


def _archive_contract(fixture: dict[str, Any]) -> dict[str, Any]:
    files = _public_files(fixture, gold=False)
    archive = _archive_bytes(fixture)
    manifest = [
        {
            "path": relative,
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        for relative, data in sorted(files.items())
    ]
    public_test = files["tests/task.test.mjs"]
    return {
        "base_archive_format": "ustar",
        "base_archive_sha256": hashlib.sha256(archive).hexdigest(),
        "base_archive_size": len(archive),
        "base_archive_file_count": len(files),
        "base_archive_files": manifest,
        "base_archive_manifest_digest": _stable_digest(manifest),
        "public_test_path": "tests/task.test.mjs",
        "public_test_sha256": hashlib.sha256(public_test).hexdigest(),
    }


def _write_tree(root: Path, files: dict[str, bytes]) -> None:
    normalized: dict[str, bytes] = {}
    casefolded: set[str] = set()
    for relative, data in files.items():
        safe_relative = _validated_relative_path(relative)
        if not isinstance(data, bytes):
            raise RuntimeError(f"confirmatory fixture is not bytes: {relative}")
        folded = safe_relative.casefold()
        if folded in casefolded:
            raise RuntimeError(f"colliding confirmatory fixture path: {relative}")
        casefolded.add(folded)
        normalized[safe_relative] = data
    resolved_root = root.resolve()
    for relative, data in normalized.items():
        target = (resolved_root / relative).resolve()
        try:
            target.relative_to(resolved_root)
        except ValueError as exc:
            raise RuntimeError(
                f"confirmatory fixture path escapes the workspace: {relative}"
            ) from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def _node_runtime_receipt() -> dict[str, str]:
    if shutil.which("docker") is None:
        raise RuntimeError(
            "Docker with the digest-pinned Node 22 image is required for fixture preflight"
        )
    return {
        "mode": "docker",
        "identity": NODE_IMAGE,
        "version": "22",
        "platform": "linux/arm64",
    }


def _run_public_tests(root: Path) -> subprocess.CompletedProcess[str]:
    _node_runtime_receipt()
    docker = shutil.which("docker")
    assert docker is not None
    command = [
        docker,
        "run",
        "--rm",
        "--pull",
        "never",
        "--platform",
        "linux/arm64",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "64",
        "--cpus",
        "1",
        "--memory",
        "256m",
        "--mount",
        f"type=bind,src={root.resolve()},dst=/workspace,readonly",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=16m",
        "--workdir",
        "/workspace",
        NODE_IMAGE,
        "node",
        "--test",
        "tests/task.test.mjs",
    ]
    return subprocess.run(
        command,
        cwd=None,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env={"PATH": str(Path(docker).parent)},
    )


def _validate_node_test_receipt(
    fixture: dict[str, Any],
    completed: subprocess.CompletedProcess[str],
    *,
    should_pass: bool,
) -> None:
    detail = completed.stdout + completed.stderr
    public_name = str(fixture["public_test_name"])
    common = bool(
        "TAP version 13" in detail
        and public_name in detail
        and re.search(r"# tests\s+1\b", detail)
    )
    if should_pass:
        valid = bool(
            common
            and completed.returncode == 0
            and re.search(r"# pass\s+1\b", detail)
            and re.search(r"# fail\s+0\b", detail)
        )
        expectation = "passing"
    else:
        valid = bool(
            common
            and completed.returncode == 1
            and re.search(r"# pass\s+0\b", detail)
            and re.search(r"# fail\s+1\b", detail)
        )
        expectation = "failing"
    if not valid:
        raise RuntimeError(
            f"{expectation} Node test receipt was not produced for "
            f"{fixture['id']} (exit {completed.returncode})\n{detail[-4000:]}"
        )


def _tree_digest(files: dict[str, bytes | str]) -> str:
    records = [
        [
            relative,
            len(data if isinstance(data, bytes) else data.encode()),
            hashlib.sha256(
                data if isinstance(data, bytes) else data.encode()
            ).hexdigest(),
        ]
        for relative, data in sorted(files.items())
    ]
    return _stable_digest(records)


def _host_verifier_receipt(
    fixture: dict[str, Any],
    expected: dict[str, Any],
    output: dict[str, Any],
    completed: subprocess.CompletedProcess[str],
    *,
    attempt_id: str,
) -> dict[str, Any]:
    passed = completed.returncode == 0
    detail = completed.stdout + completed.stderr
    submitted = dict(output["files"])
    final_tree = _public_files(fixture, gold=False)
    final_tree.update({path: body.encode() for path, body in submitted.items()})
    runtime = _node_runtime_receipt()
    runtime_profile = {
        "id": "node22-verifier-v1",
        "image": NODE_IMAGE,
        "platform": runtime["platform"],
    }
    receipt = {
        "schema_version": 2,
        "kind": "post_trial_verifier_receipt",
        "evaluator_id": "vercel-confirmatory",
        "task_id": str(fixture["id"]),
        "attempt_id": attempt_id,
        "status": "passed" if passed else "failed",
        "failure_kind": None if passed else "public_test_failed",
        "runtime": f"node-{runtime['version']}",
        "command": ["node", "--test", "tests/task.test.mjs"],
        "exit_code": completed.returncode,
        "test_count": 1,
        "pass_count": 1 if passed else 0,
        "fail_count": 0 if passed else 1,
        "output_sha256": hashlib.sha256(detail.encode()).hexdigest(),
        "base_archive_sha256": expected["base_archive_sha256"],
        "public_test_sha256": expected["public_test_sha256"],
        "submitted_artifact_sha256": _stable_digest(output),
        "final_tree_sha256": _tree_digest(final_tree),
        "verifier_source_sha256": _sha256(HOST_VERIFIER_PATH),
        "runtime_profile_id": runtime_profile["id"],
        "runtime_profile_digest": _stable_digest(runtime_profile),
        "runtime_image": NODE_IMAGE,
        "runtime_platform": runtime["platform"],
        "runtime_image_id": runtime["identity"],
        "runtime_lock_digest": _stable_digest(runtime),
    }
    receipt["receipt_digest"] = _stable_digest(receipt)
    return receipt


def _output(
    fixture: dict[str, Any],
    *,
    gold: bool,
    completed: subprocess.CompletedProcess[str],
) -> dict[str, Any]:
    side = "gold" if gold else "base"
    return {
        "schema_version": 1,
        "task_id": fixture["id"],
        "status": "completed",
        "files": {
            path: sources[side] for path, sources in fixture["target_files"].items()
        },
        "inspected_paths": fixture["required_inspected_paths"],
        "verification": [
            {
                "command": "node --test",
                "exit_code": completed.returncode,
                "stdout": completed.stdout + completed.stderr,
            }
        ],
        "summary": (
            "Frozen gold fixture after host-side semantic and executable verification."
            if gold
            else "Frozen base fixture before the requested repair was applied."
        ),
        "limitations": ["This receipt is generated outside Agent execution."],
    }


def _frozen_label_output(fixture: dict[str, Any], *, gold: bool) -> dict[str, Any]:
    passed = gold
    status = "ok" if passed else "not ok"
    completed = subprocess.CompletedProcess(
        args=["node", "--test", "tests/task.test.mjs"],
        returncode=0 if passed else 1,
        stdout=(
            "TAP version 13\n"
            f"# Subtest: {fixture['public_test_name']}\n"
            f"{status} 1 - {fixture['public_test_name']}\n"
            "1..1\n"
            "# tests 1\n"
            f"# pass {1 if passed else 0}\n"
            f"# fail {0 if passed else 1}\n"
        ),
        stderr="",
    )
    return _output(fixture, gold=gold, completed=completed)


def _preflight(fixture: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    scorer = runpy.run_path(SCORER_PATH.as_posix())["score"]
    with tempfile.TemporaryDirectory(prefix=f"fugue-vercel-{fixture['id']}-") as value:
        root = Path(value)
        base_root = root / "base"
        gold_root = root / "gold"
        _write_tree(base_root, _public_files(fixture, gold=False))
        _write_tree(gold_root, _public_files(fixture, gold=True))
        base_run = _run_public_tests(base_root)
        gold_run = _run_public_tests(gold_root)
        _validate_node_test_receipt(fixture, base_run, should_pass=False)
        _validate_node_test_receipt(fixture, gold_run, should_pass=True)
        base_output = _output(fixture, gold=False, completed=base_run)
        gold_output = _output(fixture, gold=True, completed=gold_run)
        base_scores = scorer(
            {"id": fixture["id"]},
            base_output,
            {
                "expected": expected,
                "host_verifier_receipt": _host_verifier_receipt(
                    fixture,
                    expected,
                    base_output,
                    base_run,
                    attempt_id=f"preflight:{fixture['id']}:base",
                ),
            },
        )
        gold_scores = scorer(
            {"id": fixture["id"]},
            gold_output,
            {
                "expected": expected,
                "host_verifier_receipt": _host_verifier_receipt(
                    fixture,
                    expected,
                    gold_output,
                    gold_run,
                    attempt_id=f"preflight:{fixture['id']}:gold",
                ),
            },
        )
    if all(base_scores.values()):
        raise RuntimeError(f"base scorer unexpectedly passed: {fixture['id']}")
    if not all(gold_scores.values()):
        raise RuntimeError(f"gold scorer failed: {fixture['id']}: {gold_scores}")
    return {
        "task_id": fixture["id"],
        "split": fixture["split"],
        "base_public_test_exit_code": base_run.returncode,
        "gold_public_test_exit_code": gold_run.returncode,
        "base_scores": base_scores,
        "gold_scores": gold_scores,
        "base_tree_digest": _stable_digest(
            {
                key: hashlib.sha256(data).hexdigest()
                for key, data in _public_files(fixture, gold=False).items()
            }
        ),
        "gold_tree_digest": _stable_digest(
            {
                key: hashlib.sha256(data).hexdigest()
                for key, data in _public_files(fixture, gold=True).items()
            }
        ),
        "base_archive_sha256": expected["base_archive_sha256"],
        "archive_manifest_digest": expected["base_archive_manifest_digest"],
        "public_test_sha256": expected["public_test_sha256"],
    }


def _build_archive(fixture: dict[str, Any], destination: Path) -> dict[str, Any]:
    files = _public_files(fixture, gold=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    contract = _archive_contract(fixture)
    destination.write_bytes(_archive_bytes(fixture))
    try:
        archive_path = destination.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        archive_path = destination.as_posix()
    return {
        "id": fixture["id"],
        "archive": archive_path,
        "sha256": _sha256(destination),
        "file_count": len(files),
        "archive_manifest_digest": contract["base_archive_manifest_digest"],
        "public_test_sha256": contract["public_test_sha256"],
    }


def prepare(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    source_lock = _load_and_verify_source_lock()
    fixtures = _load_catalog()
    tasks = _load_jsonl(TASKS_PATH)
    labels = _load_jsonl(PRIVATE_PATH)
    expected_by_id = {str(item["id"]): item["expected"] for item in labels}
    fixture_ids = [str(item["id"]) for item in fixtures]
    if [str(item["id"]) for item in tasks] != fixture_ids:
        raise RuntimeError("public task order differs from the frozen fixture catalog")
    if [str(item["id"]) for item in labels] != fixture_ids:
        raise RuntimeError(
            "private-label order differs from the frozen fixture catalog"
        )
    receipts = [
        _preflight(fixture, expected_by_id[str(fixture["id"])]) for fixture in fixtures
    ]
    archives = [
        _build_archive(fixture, output_root / f"{fixture['id']}.tar")
        for fixture in fixtures
    ]
    receipt = {
        "schema_version": 1,
        "source_lock_digest": source_lock["manifest_digest"],
        "preflight_runtime": _node_runtime_receipt(),
        "tasks": receipts,
    }
    receipt["receipt_digest"] = _stable_digest(receipt)
    manifest = {
        "schema_version": 1,
        "source_lock_digest": source_lock["manifest_digest"],
        "preflight_receipt_digest": receipt["receipt_digest"],
        "archives": archives,
    }
    manifest["manifest_digest"] = _stable_digest(manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "preflight.receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "fixtures.lock.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-frozen-sources", action="store_true")
    args = parser.parse_args()
    result = refresh_frozen_sources() if args.refresh_frozen_sources else prepare()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
