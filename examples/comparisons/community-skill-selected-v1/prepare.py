#!/usr/bin/env python3
"""Prepare the selected public Skill studies without exposing answer data.

The generated task archives are public inputs.  They live under ignored Fugue
state, are content locked by ``task-resources.lock.json``, and are created only
at the trusted preparation boundary.  Reviewed known-good outputs remain in
the restricted private packet and are never copied into the public archive or
receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from holdout_support import (  # noqa: E402
    load_sealed_holdout_manifest,
    prepare_sealed_holdouts,
    read_sealed_holdout_preparation,
    validate_sealed_holdouts_zero_model,
    validate_task_zero_model,
)

from fugue.bench.candidates import stable_digest  # noqa: E402
from fugue.bench.comparison import load_comparison  # noqa: E402
from fugue.bench.component_imports import (  # noqa: E402
    import_skill,
    inspect_skill_import,
    lock_skill_import,
    resolve_imported_skill,
)
from fugue.bench.files import atomic_write_json  # noqa: E402
from fugue.bench.tasksets import prepare_locked_private_labels  # noqa: E402

CAMPAIGN = Path("examples/comparisons/community-skill-selected-v1")
RESOURCE_ROOT = Path(".fugue/comparison-resources/community-skill-selected-v1")
LOCK_FILE = CAMPAIGN / "task-resources.lock.json"
DEVELOPMENT_LABEL_LOCK_FILE = CAMPAIGN / "development-labels.lock.json"
VERIFIER_CONTRACTS = {
    "anthropic-skill-creator": {
        "type": "skill_package",
        "source": "host_skill_package_verifier.cjs",
        "runtime_lock": "host-skill-package-verifier.lock.json",
        "verifier_id": "fugue-skill-package-validator-v1",
        "command": ["node", "skill-package-validate"],
        "dimension": "artifact_validity",
    },
    "vercel-react-best-practices": {
        "type": "node_test",
        "source": "host_node_verifier.cjs",
        "runtime_lock": "host-verifier.lock.json",
        "verifier_id": "fugue-node-test-v1",
        "command": ["node", "--test"],
        "dimension": "verification_passed",
    },
}
LANES = (
    "superpowers-writing-plans",
    "anthropic-skill-creator",
    "vercel-react-best-practices",
)
TARGETS = {
    "superpowers-writing-plans": "/workspace/resources/fugue-source.tar",
    "anthropic-skill-creator": "/workspace/resources/task-source.tar",
    "vercel-react-best-practices": "/workspace/resources/task-repository.tar",
}
JUDGE_CONTRACTS = {
    "superpowers-writing-plans": (
        "implementation-plan",
        ("actionability", "repository_grounding", "reviewability", "risk_calibration"),
    ),
    "anthropic-skill-creator": (
        "skill-package",
        (
            "instruction_usefulness",
            "compatibility_guidance",
            "boundedness",
            "maintainability",
        ),
    ),
    "vercel-react-best-practices": (
        "code-change",
        ("reviewability", "evidence_use", "bounded_scope", "risk_communication"),
    ),
}

SUPERPOWERS_SOURCE = {
    "repository": "https://github.com/ash0ts/fugue",
    "commit": "faa60280841bad8c1a301bd14006d486a86dde5e",
    "tree": "b301da496caa8894534c29c43df6b59d60815a57",
}
_VISIBLE_ROOTS = (
    ".codex/skills/fugue-dev/",
    ".github/workflows/",
    "configs/",
    "docs/",
    "fugue/",
    "tests/",
)
_VISIBLE_FILES = frozenset({"LICENSE", "README.md", "pyproject.toml", "uv.lock"})
_PRIVATE_ROOTS = (
    "configs/fugue/evaluations/",
    "datasets/",
    "examples/comparisons/",
    "fugue/resources/",
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} rows must be objects")
    return rows


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe archive path: {value!r}")
    return value


def _deterministic_archive(
    path: Path, files: Mapping[str, bytes | str], *, prefix: str
) -> dict[str, Any]:
    normalized: dict[str, bytes] = {}
    for name, raw in files.items():
        relative = _safe_relative(name)
        content = raw.encode() if isinstance(raw, str) else raw
        if relative.casefold() in {item.casefold() for item in normalized}:
            raise ValueError(f"case-colliding archive path: {relative}")
        normalized[relative] = content
    if not normalized:
        raise ValueError("prepared archive cannot be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tarfile.open(temporary, "w", format=tarfile.USTAR_FORMAT) as archive:
            for relative, content in sorted(normalized.items()):
                info = tarfile.TarInfo(f"{prefix}/{relative}")
                info.size = len(content)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(content))
        temporary.chmod(0o444)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": path.as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "file_count": len(normalized),
        "paths_digest": hashlib.sha256(
            "\n".join(sorted(normalized)).encode()
        ).hexdigest(),
    }


def _superpowers_files(repo_root: Path) -> tuple[dict[str, bytes], int]:
    commit = SUPERPOWERS_SOURCE["commit"]
    resolved = subprocess.run(
        ["git", "rev-parse", f"{commit}^{{commit}}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", f"{commit}^{{tree}}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if resolved != commit or tree != SUPERPOWERS_SOURCE["tree"]:
        raise ValueError("historical Fugue source identity changed")
    entries = subprocess.run(
        ["git", "ls-tree", "-r", "-z", commit],
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    files: dict[str, bytes] = {}
    excluded = 0
    for entry in entries:
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode, kind, _object_id = metadata.decode().split()
        path = raw_path.decode()
        visible = path in _VISIBLE_FILES or path.startswith(_VISIBLE_ROOTS)
        private = path.startswith(_PRIVATE_ROOTS)
        if not visible:
            continue
        if private:
            excluded += 1
            continue
        if mode not in {"100644", "100755"} or kind != "blob":
            raise ValueError(f"unsupported historical source entry: {path}")
        files[path] = subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        ).stdout
    if "fugue/bench/comparison.py" not in files or not files:
        raise ValueError("historical Fugue source archive is incomplete")
    return files, excluded


def _resource_records(repo_root: Path) -> dict[str, Any]:
    root = repo_root / RESOURCE_ROOT
    fixtures = _json(repo_root / CAMPAIGN / "task-fixtures.json")
    if set(fixtures) != {"anthropic-skill-creator", "vercel-react-best-practices"}:
        raise ValueError("public task fixture families changed")
    superpowers_files, excluded = _superpowers_files(repo_root)
    shared = _deterministic_archive(
        root / "superpowers/fugue-source-faa60280.tar",
        superpowers_files,
        prefix="repo",
    )
    lane_records: dict[str, list[dict[str, Any]]] = {
        "superpowers-writing-plans": [
            {"task_id": task_id, **shared}
            for task_id in (
                "sp-dev-credential-rotation",
                "sp-dev-evidence-destination",
                "sp-dev-package-tree-qualification",
                "sp-dev-single-file-validation-fix",
            )
        ],
        "anthropic-skill-creator": [],
        "vercel-react-best-practices": [],
    }
    fixture_lanes = (
        ("anthropic-skill-creator", "anthropic", "workspace"),
        ("vercel-react-best-practices", "vercel", "repo"),
    )
    for lane, folder, prefix in fixture_lanes:
        selected = fixtures[lane]
        ids = [row["id"] for row in _rows(repo_root / CAMPAIGN / lane / "tasks.jsonl")]
        if set(selected) != set(ids):
            raise ValueError(f"{lane} public task fixtures changed")
        for task_id in ids:
            record = _deterministic_archive(
                root / f"{folder}/{task_id}.tar", selected[task_id], prefix=prefix
            )
            lane_records[lane].append({"task_id": task_id, **record})
    for records in lane_records.values():
        for record in records:
            record["path"] = Path(record["path"]).relative_to(repo_root).as_posix()
    return {
        "schema_version": 1,
        "generator": "community-skill-public-resources-v1",
        "superpowers_source": {
            **SUPERPOWERS_SOURCE,
            "excluded_private_roots": list(_PRIVATE_ROOTS),
            "excluded_private_file_count": excluded,
        },
        "lanes": lane_records,
    }


def _load_scorer(path: Path, lane: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"community_zero_model_{lane}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load scorer for {lane}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_development_zero_model(repo_root: Path) -> list[dict[str, Any]]:
    """Prove one reviewed gold and isolated mutant for every development task."""

    results: list[dict[str, Any]] = []
    for lane in LANES:
        lane_root = repo_root / CAMPAIGN / lane
        tasks = {row["id"]: row for row in _rows(lane_root / "tasks.jsonl")}
        labels_path = (
            repo_root
            / ".fugue/private/community-skill-selected-v1"
            / lane
            / "private-labels.jsonl"
        )
        labels = {row["id"]: row for row in _rows(labels_path)}
        if set(labels) != set(tasks):
            raise ValueError(f"private zero-model labels do not match {lane} tasks")
        scorer = _load_scorer(lane_root / "scorer.py", lane)
        for task_id in sorted(tasks):
            results.append(
                validate_task_zero_model(
                    repo_root=repo_root,
                    lane=lane,
                    task=tasks[task_id],
                    label=labels[task_id],
                    scorer=scorer,
                )
            )
    if len(results) != 12:
        raise ValueError("development zero-model proof must cover exactly 12 tasks")
    return results


def _prepare_development_labels(
    repo_root: Path,
    operator_source: Path,
    *,
    source_name: str = "development-private-labels.jsonl",
) -> dict[str, Any]:
    lock = _json(repo_root / DEVELOPMENT_LABEL_LOCK_FILE)
    if (
        lock.get("schema_version") != 1
        or lock.get("kind") != "community_skill_development_private_label_lock"
        or set(lock.get("lanes") or {}) != set(LANES)
    ):
        raise ValueError("development private-label lock changed")
    receipts = []
    for lane in LANES:
        receipts.append(
            {
                "lane": lane,
                **prepare_locked_private_labels(
                    source_path=operator_source / lane / source_name,
                    destination_path=(
                        repo_root
                        / ".fugue/private/community-skill-selected-v1"
                        / lane
                        / "private-labels.jsonl"
                    ),
                    expected=lock["lanes"][lane],
                ),
            }
        )
    unsigned = {
        "schema_version": 1,
        "kind": "community_skill_development_labels_preparation",
        "lock_digest": stable_digest(lock),
        "task_count": sum(int(item["task_count"]) for item in receipts),
        "lanes": receipts,
    }
    receipt = {**unsigned, "receipt_digest": stable_digest(unsigned)}
    atomic_write_json(
        repo_root
        / ".fugue/private/community-skill-selected-v1/development-labels-receipt.json",
        receipt,
    )
    return receipt


def _campaign_zero_model_receipt(
    repo_root: Path, development: list[dict[str, Any]]
) -> dict[str, Any]:
    sealed_path = (
        repo_root
        / ".fugue/private/community-skill-selected-v1/sealed-holdouts/zero-model-receipt.json"
    )
    sealed = _json(sealed_path)
    sealed_unsigned = dict(sealed)
    sealed_digest = sealed_unsigned.pop("receipt_digest", None)
    if (
        sealed_digest != stable_digest(sealed_unsigned)
        or sealed.get("task_count") != 12
        or len(development) != 12
    ):
        raise ValueError("24-task zero-model prerequisites are missing or stale")
    results = [*development, *sealed["results"]]
    task_ids = [str(row["task_id"]) for row in results]
    if len(results) != 24 or len(set(task_ids)) != 24:
        raise ValueError("zero-model receipt must cover 24 unique selected tasks")
    unsigned = {
        "schema_version": 1,
        "kind": "community_skill_campaign_zero_model_receipt",
        "task_count": 24,
        "development_task_count": 12,
        "holdout_task_count": 12,
        "sealed_zero_model_receipt_digest": sealed_digest,
        "results_digest": stable_digest(results),
        "status": "known_good_passed_and_targeted_mutants_failed",
    }
    receipt = {**unsigned, "receipt_digest": stable_digest(unsigned)}
    atomic_write_json(
        repo_root
        / ".fugue/private/community-skill-selected-v1/campaign-zero-model-receipt.json",
        receipt,
    )
    return receipt


def _validate_supersession_registry(repo_root: Path) -> str:
    registry = _json(repo_root / CAMPAIGN / "supersession-registry.json")
    if (
        registry.get("schema_version") != 2
        or registry.get("kind") != "community_skill_supersession_registry"
        or not isinstance(registry.get("entries"), list)
    ):
        raise ValueError("supersession registry contract changed")
    required = {
        "study_id",
        "project",
        "wandb_project_url",
        "study_console_query",
        "status",
        "relationship",
        "pooled",
        "reason",
        "preview_digest",
        "result_digest",
        "invalidation_digest",
        "record_digest",
    }
    study_ids: set[str] = set()
    statuses = {"valid-historical-negative", "invalid", "superseded-design"}
    for entry in registry["entries"]:
        if not isinstance(entry, dict) or set(entry) != required:
            raise ValueError("supersession registry entry fields changed")
        unsigned = {key: value for key, value in entry.items() if key != "record_digest"}
        if entry["record_digest"] != stable_digest(unsigned):
            raise ValueError("supersession registry record digest changed")
        study_id = str(entry["study_id"])
        if (
            study_id in study_ids
            or entry["status"] not in statuses
            or not str(entry["relationship"]).strip()
            or not str(entry["reason"]).strip()
            or str(entry["project"]).count("/") != 1
        ):
            raise ValueError("supersession registry Study ids must be unique")
        study_ids.add(study_id)
        entity, project = str(entry["project"]).split("/", maxsplit=1)
        if entry["wandb_project_url"] != f"https://wandb.ai/{entity}/{project}":
            raise ValueError("supersession registry W&B URL changed")
        if entry["study_console_query"] != (
            f"?research_id={study_id}&study_id={study_id}"
        ):
            raise ValueError("supersession registry Study URL changed")
        digests = (entry["preview_digest"], entry["result_digest"])
        if any(value is not None and not _is_sha256(value) for value in digests):
            raise ValueError("supersession registry artifact digest changed")
        if entry["status"] == "valid-historical-negative":
            if not all(_is_sha256(value) for value in digests) or entry["invalidation_digest"] is not None:
                raise ValueError("valid historical result requires preview and result digests")
        else:
            invalidation = {
                key: entry[key]
                for key in ("study_id", "status", "relationship", "reason")
            }
            if entry["result_digest"] is not None or entry["invalidation_digest"] != stable_digest(invalidation):
                raise ValueError("invalid historical result requires an invalidation digest")
        if entry["pooled"] is not False:
            raise ValueError("historical Studies must never be pooled")
    unsigned_registry = {
        key: value for key, value in registry.items() if key != "registry_digest"
    }
    if registry.get("registry_digest") != stable_digest(unsigned_registry):
        raise ValueError("supersession registry digest changed")
    return str(registry["registry_digest"])


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def validate_public_campaign(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    campaign_root = root / CAMPAIGN
    manifest = _json(campaign_root / "campaign-manifest.json")
    sealed_manifest_path = campaign_root / "sealed-holdouts.json"
    sealed_manifest = load_sealed_holdout_manifest(sealed_manifest_path)
    resource_lock = _json(root / LOCK_FILE)
    label_lock = _json(root / DEVELOPMENT_LABEL_LOCK_FILE)
    if manifest.get("schema_version") != 1 or manifest.get("id") != CAMPAIGN.name:
        raise ValueError("campaign manifest identity changed")
    declared_lanes = tuple(item["id"] for item in manifest.get("lanes", []))
    if declared_lanes != LANES or resource_lock.get("schema_version") != 1:
        raise ValueError("campaign lanes or resource lock changed")
    if (
        label_lock.get("schema_version") != 1
        or label_lock.get("kind")
        != "community_skill_development_private_label_lock"
        or set(label_lock.get("lanes") or {}) != set(LANES)
    ):
        raise ValueError("development private-label lock changed")
    case_set = _json(campaign_root / "judge/case-set-manifest.json")
    rubric = _json(campaign_root / "judge/rubric.json")
    verifier_locks = {}
    for lane, contract in VERIFIER_CONTRACTS.items():
        lane_root = campaign_root / lane
        verifier_lock = _json(lane_root / contract["runtime_lock"])
        verifier_source = lane_root / contract["source"]
        if (
            verifier_lock.get("schema_version") != 1
            or verifier_lock.get("id") != contract["verifier_id"]
            or verifier_lock.get("verifier_source_sha256")
            != _sha256(verifier_source)
            or verifier_lock.get("command") != contract["command"]
            or "@sha256:"
            not in str((verifier_lock.get("runtime") or {}).get("image"))
            or (verifier_lock.get("runtime") or {}).get("network") != "none"
            or (verifier_lock.get("runtime") or {}).get("read_only_base") is not True
        ):
            raise ValueError(f"trusted {lane} verifier lock changed")
        verifier_locks[lane] = verifier_lock
    supersession_registry_digest = _validate_supersession_registry(root)
    modality_counts = case_set.get("modalities", {})
    if (
        case_set.get("case_count") != 48
        or set(modality_counts) != set(rubric.get("modalities", {}))
        or sum(
            int(value.get("acceptable", 0)) + int(value.get("defective", 0))
            for value in modality_counts.values()
        )
        != 48
    ):
        raise ValueError("judge case manifest is not balanced across 48 cases")

    lane_receipts = []
    locked_lanes = resource_lock.get("lanes")
    if not isinstance(locked_lanes, dict) or set(locked_lanes) != set(LANES):
        raise ValueError("resource lock does not cover exactly the selected lanes")
    for lane in LANES:
        lane_root = campaign_root / lane
        spec = load_comparison(lane_root / "comparison.yaml", repo_root=root)
        tasks = _rows(lane_root / "tasks.jsonl")
        ids = [str(row.get("id") or "") for row in tasks]
        if len(tasks) != 4 or len(set(ids)) != 4 or not all(ids):
            raise ValueError(f"{lane} must contain four unique public tasks")
        if any(row.get("partition") != "qualification" for row in tasks):
            raise ValueError(f"{lane} may contain public development tasks only")
        locked = {
            str(item["task_id"]): item
            for item in locked_lanes[lane]
            if isinstance(item, dict)
        }
        if set(locked) != set(ids):
            raise ValueError(f"{lane} task resources are incomplete")
        locked_labels = label_lock["lanes"][lane]
        if (
            not isinstance(locked_labels, list)
            or [item.get("task_id") for item in locked_labels] != sorted(ids)
        ):
            raise ValueError(f"{lane} development private-label lock is incomplete")
        for row in tasks:
            resources = row.get("resources")
            expected = locked[row["id"]]
            if resources != [{"path": expected["path"], "target": TARGETS[lane]}]:
                raise ValueError(f"{lane}/{row['id']} resource binding changed")
        lock = _json(lane_root / "skill-revisions.lock.json")
        revisions = [lock["baseline"], lock["candidate"]]
        expected_skills = {revisions[0]["id"], revisions[1]["id"]}
        actual_skills = set(spec.baseline.skills) | set(spec.candidate.skills)
        if actual_skills != expected_skills or spec.changed != ("skills",):
            raise ValueError(f"{lane} comparison disagrees with its revision lock")
        _validate_lane_evaluators(spec.evaluators, lane=lane)
        private_path = root / spec.taskset.private_labels
        if not private_path.is_relative_to(root / ".fugue/private"):
            raise ValueError(f"{lane} private labels must stay under .fugue/private")
        lane_receipts.append(
            {
                "id": lane,
                "comparison_digest": spec.spec_digest,
                "tasks_sha256": _sha256(lane_root / "tasks.jsonl"),
                "scorer_sha256": _sha256(lane_root / "scorer.py"),
                "revision_lock_sha256": _sha256(lane_root / "skill-revisions.lock.json"),
                "resource_lock_digest": stable_digest(locked_lanes[lane]),
                "host_verifier_lock_digest": (
                    stable_digest(verifier_locks[lane])
                    if lane in verifier_locks
                    else None
                ),
                "task_ids": ids,
            }
        )
    return {
        "schema_version": 1,
        "campaign_id": manifest["id"],
        "campaign_manifest_sha256": _sha256(campaign_root / "campaign-manifest.json"),
        "sealed_holdout_manifest_sha256": _sha256(sealed_manifest_path),
        "sealed_holdout_audit_policy_digest": stable_digest(
            sealed_manifest["audit"]
        ),
        "resource_lock_sha256": _sha256(root / LOCK_FILE),
        "development_private_label_lock_sha256": _sha256(
            root / DEVELOPMENT_LABEL_LOCK_FILE
        ),
        "rubric_sha256": _sha256(campaign_root / "judge/rubric.json"),
        "cases_digest": case_set["cases_digest"],
        "supersession_registry_digest": supersession_registry_digest,
        "lanes": lane_receipts,
    }


def _validate_lane_evaluators(evaluators: tuple[Any, ...], *, lane: str) -> None:
    deterministic = tuple(item for item in evaluators if item.type == "deterministic")
    judges = tuple(item for item in evaluators if item.type == "llm_judge")
    if len(deterministic) != 1 or not deterministic[0].required or len(judges) > 1:
        raise ValueError(
            f"{lane} requires one authoritative deterministic evaluator and at most "
            "one optional judge"
        )
    verifier = deterministic[0].verifier
    if lane in VERIFIER_CONTRACTS:
        contract = VERIFIER_CONTRACTS[lane]
        expected_root = CAMPAIGN / lane
        if verifier is None or verifier.to_dict() != {
            "type": contract["type"],
            "source": (expected_root / contract["source"]).as_posix(),
            "runtime_lock": (expected_root / contract["runtime_lock"]).as_posix(),
            "runtime": "node22-verifier-v1",
            "dimension": contract["dimension"],
        }:
            raise ValueError(
                f"{lane} deterministic scoring requires its locked host verifier"
            )
    elif verifier is not None:
        raise ValueError(f"{lane} must not declare a host verifier")
    if not judges:
        return
    judge = judges[0]
    modality, dimensions = JUDGE_CONTRACTS[lane]
    expected_rubric = (CAMPAIGN / "judge/rubric.json").as_posix()
    valid = (
        judge.required is False
        and judge.profile == "anthropic/claude-sonnet-5"
        and judge.calibration_modality == modality
        and judge.calibration_rubric == expected_rubric
        and judge.calibration is not None
        and judge.calibration.startswith(".fugue/private/")
        and judge.dimensions == dimensions
        and set(judge.dimension_roles.values()) == {"outcome"}
        and judge.evidence
        == ("artifact_paths", "inspected_paths", "changed_paths")
        and judge.reserve_cost_usd == 0.1
        and bool(judge.rubric)
    )
    if not valid:
        raise ValueError(f"{lane} optional judge does not match its advisory contract")


def _prepare_lane(repo_root: Path, lane: str) -> dict[str, Any]:
    selected = _json(repo_root / CAMPAIGN / lane / "skill-revisions.lock.json")
    prepared = []
    for arm in ("baseline", "candidate"):
        expected = selected[arm]
        skill_id = str(expected["id"])
        resolved = resolve_imported_skill(skill_id, repo_root)
        if resolved is None or (
            resolved.digest != expected["bundle_digest"]
            or resolved.resolved_commit != expected["commit"]
        ):
            source = (
                f"git+{selected['repository']}@{expected['commit']}"
                f"#path={selected['path']}"
            )
            import_skill(source, repo_root=repo_root, import_id=skill_id)
            inspection = inspect_skill_import(skill_id, repo_root)
            if (
                inspection["digest"] != expected["bundle_digest"]
                or inspection["total_files"] != expected["file_count"]
            ):
                raise ValueError(f"{skill_id} inspection differs from reviewed lock")
            locked = lock_skill_import(skill_id, repo_root)
            if (
                locked.digest != expected["bundle_digest"]
                or locked.resolved_commit != expected["commit"]
                or locked.total_files != expected["file_count"]
            ):
                raise ValueError(f"{skill_id} materialization differs from reviewed lock")
            resolved = resolve_imported_skill(skill_id, repo_root)
        if resolved is None:
            raise ValueError(f"{skill_id} did not resolve after preparation")
        prepared.append(
            {"id": skill_id, "commit": resolved.resolved_commit, "bundle_digest": resolved.digest}
        )
    return {"lane": lane, "skills": prepared}


def prepare_public_campaign(
    repo_root: Path,
    *,
    fetch: bool,
    prepare_resources: bool = False,
    operator_source: Path | None = None,
) -> dict[str, Any]:
    public = validate_public_campaign(repo_root)
    resources: dict[str, Any] | None = None
    verifier: list[dict[str, Any]] | None = None
    zero_model: list[dict[str, Any]] | None = None
    campaign_zero_model: dict[str, Any] | None = None
    sealed_preparation: dict[str, Any] | None = None
    sealed_zero_model: dict[str, Any] | None = None
    development_labels: dict[str, Any] | None = None
    if operator_source is not None and not prepare_resources:
        raise ValueError("--operator-source requires --prepare-resources")
    if prepare_resources:
        if operator_source is not None:
            selected_source = (
                operator_source
                if operator_source.is_absolute()
                else repo_root / operator_source
            )
            development_labels = _prepare_development_labels(
                repo_root, selected_source
            )
            sealed_preparation = prepare_sealed_holdouts(
                manifest_path=repo_root / CAMPAIGN / "sealed-holdouts.json",
                operator_source=selected_source,
                repo_root=repo_root,
            )
        else:
            try:
                development_labels = _prepare_development_labels(
                    repo_root,
                    repo_root / ".fugue/private/community-skill-selected-v1",
                    source_name="private-labels.jsonl",
                )
                sealed_preparation = read_sealed_holdout_preparation(repo_root)
            except (FileNotFoundError, ValueError) as error:
                raise ValueError(
                    "first-time 24-task preparation requires --operator-source"
                ) from error
        sealed_zero_model = validate_sealed_holdouts_zero_model(
            repo_root=repo_root,
            preparation_receipt=sealed_preparation,
        )
        resources = _resource_records(repo_root)
        expected = _json(repo_root / LOCK_FILE)
        if resources != expected:
            raise ValueError(
                "prepared task resources differ from task-resources.lock.json; "
                "review and update the public lock explicitly"
            )
        zero_model = _validate_development_zero_model(repo_root)
        campaign_zero_model = _campaign_zero_model_receipt(repo_root, zero_model)
        verifier = [
            {
                "task_id": row["task_id"],
                "base_status": row["mutant_status"],
                "known_good_status": "passed",
                "base_receipt_digest": row["mutant_scores_digest"],
                "known_good_receipt_digest": row["gold_scores_digest"],
            }
            for row in zero_model
            if row["lane_id"] == "vercel-react-best-practices"
        ]
    prepared: list[dict[str, Any]] = []
    if fetch:
        with ThreadPoolExecutor(max_workers=3) as workers:
            prepared = list(workers.map(lambda lane: _prepare_lane(repo_root, lane), LANES))
    status = "public-inputs-valid"
    if prepare_resources:
        status = (
            "public-resources-verified"
            if verifier and all(row["known_good_status"] == "passed" for row in verifier)
            else "public-resources-base-verified-known-good-pending"
        )
    if fetch and status == "public-resources-verified":
        status = "ready-for-private-label-and-preview"
    unsigned = {
        **public,
        "status": status,
        "writes": bool(fetch or prepare_resources),
        "prepared_skills": prepared,
        "resource_lock_digest": stable_digest(resources) if resources else None,
        "vercel_preflight": verifier,
        "development_zero_model": zero_model,
        "campaign_zero_model_receipt_digest": (
            campaign_zero_model["receipt_digest"] if campaign_zero_model else None
        ),
        "campaign_zero_model_task_count": (
            campaign_zero_model["task_count"] if campaign_zero_model else 0
        ),
        "sealed_preparation_receipt_digest": (
            sealed_preparation["receipt_digest"] if sealed_preparation else None
        ),
        "sealed_zero_model_receipt_digest": (
            sealed_zero_model["receipt_digest"] if sealed_zero_model else None
        ),
        "development_labels_receipt_digest": (
            development_labels["receipt_digest"] if development_labels else None
        ),
        "private_labels_included": False,
        "known_good_content_included": False,
    }
    receipt = {**unsigned, "preparation_digest": stable_digest(unsigned)}
    if fetch or prepare_resources:
        atomic_write_json(
            repo_root / ".fugue/qualification/community-skill-selected-v1/public-preparation.json",
            receipt,
        )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--fetch-skills", action="store_true")
    parser.add_argument("--prepare-resources", action="store_true")
    parser.add_argument("--operator-source", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_public_campaign(
                args.repo_root.resolve(),
                fetch=args.fetch_skills,
                prepare_resources=args.prepare_resources,
                operator_source=args.operator_source,
            ),
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
