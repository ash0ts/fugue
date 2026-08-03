#!/usr/bin/env python3
"""Lock every prepared local input used by one V3 Skill comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from fugue.bench.source_locks import build_local_source_lock

CAMPAIGN_SUPPORT_FILES = {
    "analyze_confirmatory.py": "confirmatory_analysis_implementation",
    "freeze_trace_audit.py": "trace_audit_selection_implementation",
    "prepare_local_source_lock.py": "source_lock_implementation",
}


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"source input escapes the repository: {path}")
    return resolved.relative_to(root.resolve()).as_posix()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"task row {line_number} must be an object")
        rows.append(value)
    return rows


def _append_file(records: dict[str, str], path: Path, *, root: Path, role: str) -> None:
    relative = _relative(path, root)
    prior = records.setdefault(relative, role)
    if prior != role:
        records[relative] = "+".join(sorted({*prior.split("+"), role}))


def collect_source_files(  # noqa: C901 - one bounded cross-input lock collector.
    *, spec_path: Path, repo_root: Path, extras: list[Path]
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    raw = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("comparison spec must be a mapping")
    source = spec_path.parent
    records: dict[str, str] = {}
    campaign_root = Path(__file__).resolve().parent
    for name, role in CAMPAIGN_SUPPORT_FILES.items():
        _append_file(records, campaign_root / name, root=repo_root, role=role)
    taskset = raw.get("taskset")
    if not isinstance(taskset, dict):
        raise ValueError("comparison taskset is unavailable")
    tasks_path = source / str(taskset.get("tasks") or "")
    labels_path = source / str(taskset.get("private_labels") or "")
    _append_file(records, tasks_path, root=repo_root, role="public_tasks")
    _append_file(records, labels_path, root=repo_root, role="host_private_labels")
    for task in _load_jsonl(tasks_path):
        resources = task.get("resources") or []
        if not isinstance(resources, list):
            raise ValueError("task resources must be an array")
        for resource in resources:
            if not isinstance(resource, dict) or not resource.get("path"):
                raise ValueError("task resource must declare a path")
            _append_file(
                records,
                repo_root / str(resource["path"]),
                root=repo_root,
                role="immutable_task_resource",
            )
    evaluators = raw.get("evaluators") or []
    if not isinstance(evaluators, list):
        raise ValueError("comparison evaluators must be an array")
    for evaluator in evaluators:
        if not isinstance(evaluator, dict):
            raise ValueError("comparison evaluator must be an object")
        for field, role in (
            ("scorer", "host_scorer"),
            ("calibration", "judge_calibration"),
        ):
            if evaluator.get(field):
                _append_file(
                    records,
                    source / str(evaluator[field]),
                    root=repo_root,
                    role=role,
                )
        verifier = evaluator.get("verifier")
        if verifier is not None:
            if not isinstance(verifier, dict) or not verifier.get("source"):
                raise ValueError("comparison verifier must declare a source")
            _append_file(
                records,
                source / str(verifier["source"]),
                root=repo_root,
                role="host_post_trial_verifier",
            )
    for arm in ("baseline", "candidate"):
        candidate = raw.get(arm)
        if not isinstance(candidate, dict):
            raise ValueError(f"comparison {arm} candidate is unavailable")
        skills = candidate.get("skills") or []
        if not isinstance(skills, list) or not skills:
            raise ValueError(f"comparison {arm} must declare a Skill")
        for skill_id in skills:
            lock_path = repo_root / ".fugue/imports/skills/locks" / f"{skill_id}.json"
            _append_file(records, lock_path, root=repo_root, role="skill_import_lock")
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            digest = str(lock.get("digest") or "").removeprefix("sha256:")
            if len(digest) != 64:
                raise ValueError(f"Skill {skill_id} has no exact bundle digest")
            cache_root = repo_root / ".fugue/cache/skills/v1" / digest
            cache_files = sorted(
                path for path in cache_root.rglob("*") if path.is_file()
            )
            if not cache_files:
                raise ValueError(f"Skill {skill_id} cache is unavailable")
            for path in cache_files:
                _append_file(
                    records,
                    path,
                    root=repo_root,
                    role="skill_bundle_file",
                )
    revision_lock = source / "skill-revisions.lock.json"
    if revision_lock.is_file():
        _append_file(records, revision_lock, root=repo_root, role="skill_revision_lock")
    for path in extras:
        _append_file(records, path, root=repo_root, role="preparation_receipt")
    execution = raw.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("comparison execution policy is unavailable")
    qualification_inputs = execution.get("qualification_inputs") or {}
    if not isinstance(qualification_inputs, dict):
        raise ValueError("comparison qualification inputs must be a mapping")
    for name, relative in sorted(qualification_inputs.items()):
        if not isinstance(name, str) or not isinstance(relative, str):
            raise ValueError("comparison qualification input is malformed")
        _append_file(
            records,
            source / relative,
            root=repo_root,
            role=f"qualification_input.{name}",
        )
    return raw, [{"path": path, "role": role} for path, role in sorted(records.items())]


def prepare(
    *,
    spec_path: Path,
    repo_root: Path,
    extras: list[Path],
    output: Path,
) -> dict[str, Any]:
    raw, files = collect_source_files(
        spec_path=spec_path, repo_root=repo_root, extras=extras
    )
    execution = raw["execution"]
    lock = build_local_source_lock(
        repo_root=repo_root,
        source_project=str(execution["source_evidence_project"]),
        result_project=str(execution["evidence_project"]),
        files=files,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return lock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--extra", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    lock = prepare(
        spec_path=args.spec.resolve(),
        repo_root=root,
        extras=[path.resolve() for path in args.extra],
        output=args.output.resolve(),
    )
    print(json.dumps({"source_lock_digest": lock["source_lock_digest"]}))


if __name__ == "__main__":
    main()
