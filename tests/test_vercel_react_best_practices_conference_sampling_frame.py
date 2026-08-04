from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = (
    ROOT / "examples/comparisons/vercel-react-best-practices-upgrade"
)
VALIDATOR_PATH = EXAMPLE / "validate_conference_sampling_frame_v1.py"
SCHEMA_PATH = EXAMPLE / "conference-sampling-frame-v1.schema.json"
PROTOCOL_PATH = EXAMPLE / "conference-sampling-frame-protocol-v1.json"
BLOCKERS_PATH = EXAMPLE / "conference-sampling-frame-blockers-v1.json"


def _load_validator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "vercel_conference_sampling_frame", VALIDATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(value: object) -> str:
    if isinstance(value, bytes):
        payload = value
    else:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    return hashlib.sha256(payload).hexdigest()


def _hex(value: str, length: int) -> str:
    digest = hashlib.sha256(value.encode()).hexdigest()
    if length <= len(digest):
        return digest[:length]
    return (digest * ((length // len(digest)) + 1))[:length]


def _fake_frame(module: ModuleType) -> dict[str, object]:
    seed = "1" * 64
    rows: list[dict[str, object]] = []
    global_index = 0
    prefix = {
        "development": "vrb-dev",
        "target_holdout": "vrb-target",
        "safety_control": "vrb-control",
    }
    for partition, strata in module.EXPECTED_STRATA.items():
        for stratum, count in strata.items():
            stratum_rows: list[dict[str, object]] = []
            for _ in range(count):
                global_index += 1
                repository_name = f"next-app-{global_index:03d}"
                repository_url = f"https://github.com/example/{repository_name}"
                commit = _hex(f"commit-{global_index}", 40)
                public_record_url = f"{repository_url}/commit/{commit}"
                task_id = f"{prefix[partition]}-{global_index:03d}"
                stratum_rows.append(
                    {
                        "task_id": task_id,
                        "partition": partition,
                        "stratum": stratum,
                        "selection_rank": 0,
                        "repository_url": repository_url,
                        "repository_owner": "example",
                        "repository_name": repository_name,
                        "commit_sha": commit,
                        "tree_sha": _hex(f"tree-{global_index}", 40),
                        "public_record_url": public_record_url,
                        "public_record_kind": "commit",
                        "public_record_id": commit,
                        "provenance_cluster_id": f"github:example/{repository_name}",
                        "population_record_digest": _hex(
                            f"population-{global_index}", 64
                        ),
                        "selection_score_sha256": module._selection_score(
                            seed, repository_url, public_record_url
                        ),
                        "target_stack_evidence": {
                            "package_manifest_path": "package.json",
                            "package_manifest_sha256": _hex(
                                f"package-{global_index}", 64
                            ),
                            "dependency_lock_path": "pnpm-lock.yaml",
                            "dependency_lock_sha256": _hex(
                                f"lock-{global_index}", 64
                            ),
                            "react_dependency": "19.1.0",
                            "next_dependency": "15.4.2",
                            "typescript_paths": [
                                "app/page.tsx",
                                "tests/fugue-regression.test.ts",
                            ],
                            "target_paths": ["app/page.tsx"],
                        },
                        "source_archive_path": (
                            f"private/vercel-conference/sources/{task_id}.tar"
                        ),
                        "source_archive_sha256": _hex(
                            f"archive-{global_index}", 64
                        ),
                        "source_manifest_path": (
                            f"private/vercel-conference/manifests/{task_id}.json"
                        ),
                        "source_manifest_sha256": _hex(
                            f"manifest-{global_index}", 64
                        ),
                        "provenance_receipt_path": (
                            f"private/vercel-conference/provenance/{task_id}.json"
                        ),
                        "provenance_receipt_sha256": _hex(
                            f"provenance-{global_index}", 64
                        ),
                        "public_test": {
                            "path": "tests/fugue-regression.test.ts",
                            "sha256": _hex(f"public-test-{global_index}", 64),
                            "command_profile_id": "prepared-node-public-test-v1",
                        },
                        "host_verifier": {
                            "path": (
                                f"private/vercel-conference/verifiers/{task_id}.cjs"
                            ),
                            "sha256": _hex(f"verifier-{global_index}", 64),
                            "runtime_profile_id": "node22-verifier-v1",
                            "host_only": True,
                            "required_mode": "0600",
                        },
                        "qualification_receipt_path": (
                            f"private/vercel-conference/qualification/{task_id}.json"
                        ),
                        "qualification_receipt_sha256": _hex(
                            f"qualification-{global_index}", 64
                        ),
                        "public_task_digest": _hex(f"public-task-{global_index}", 64),
                        "private_label_digest": _hex(
                            f"private-label-{global_index}", 64
                        ),
                        "license": {
                            "spdx_id": "MIT",
                            "path": "LICENSE",
                            "sha256": _hex(f"license-{global_index}", 64),
                        },
                    }
                )
            stratum_rows.sort(key=lambda row: str(row["selection_score_sha256"]))
            for rank, row in enumerate(stratum_rows, 1):
                row["selection_rank"] = rank
            rows.extend(stratum_rows)

    frame: dict[str, object] = {
        "schema_version": 1,
        "frame_id": module.FRAME_ID,
        "frame_digest": "0" * 64,
        "reserved_studies": {
            "development": module.DEVELOPMENT_STUDY_ID,
            "holdout": module.HOLDOUT_STUDY_ID,
        },
        "skill_treatment": {
            "repository": module.EXPECTED_SKILL_REPOSITORY,
            "path": module.EXPECTED_SKILL_PATH,
            "lock_path": (
                "examples/comparisons/vercel-react-best-practices-upgrade/"
                "skill-revisions.lock.json"
            ),
            "lock_sha256": _hex("skill-lock", 64),
            "baseline_commit": module.EXPECTED_BASELINE_COMMIT,
            "baseline_bundle_sha256": module.EXPECTED_BASELINE_BUNDLE,
            "candidate_commit": module.EXPECTED_CANDIDATE_COMMIT,
            "candidate_bundle_sha256": module.EXPECTED_CANDIDATE_BUNDLE,
        },
        "population_snapshot": {
            "selection_cutoff_utc": "2026-08-03T12:00:00Z",
            "records_path": "private/vercel-conference/population.jsonl",
            "records_sha256": _hex("population", 64),
            "record_count": module.EXPECTED_TOTAL,
            "query_receipt_path": "private/vercel-conference/query.json",
            "query_receipt_sha256": _hex("query", 64),
            "eligibility_protocol_path": (
                "examples/comparisons/vercel-react-best-practices-upgrade/"
                "conference-sampling-frame-protocol-v1.json"
            ),
            "eligibility_protocol_sha256": _hex("protocol", 64),
            "selection_code_path": "private/vercel-conference/select.py",
            "selection_code_sha256": _hex("selection", 64),
        },
        "selection_protocol": {
            "seed_sha256": seed,
            "algorithm": "sha256_seeded_sort_v1",
            "cluster_unit": "canonical_github_repository",
            "unique_clusters_across_partitions": True,
            "strata": module.EXPECTED_STRATA,
        },
        "public_tasks": {
            "path": "private/vercel-conference/tasks.jsonl",
            "sha256": _hex("tasks", 64),
            "count": module.EXPECTED_TOTAL,
        },
        "private_labels": {
            "path": "private/vercel-conference/private-labels.jsonl",
            "sha256": _hex("labels", 64),
            "count": module.EXPECTED_TOTAL,
            "host_only": True,
            "required_mode": "0600",
        },
        "entries": rows,
    }
    payload = dict(frame)
    payload.pop("frame_digest")
    frame["frame_digest"] = module.stable_digest(payload)
    return frame


def _redigest(module: ModuleType, frame: dict[str, object]) -> None:
    payload = dict(frame)
    payload.pop("frame_digest")
    frame["frame_digest"] = module.stable_digest(payload)


def test_blocker_receipt_is_exact_and_no_powered_specs_exist() -> None:
    value = json.loads(BLOCKERS_PATH.read_text(encoding="utf-8"))
    supplied = value.pop("receipt_digest")
    assert supplied == _digest(value)
    assert value["eligible_reviewed_sources_present"] == 0
    assert value["exact_missing_sources"] == {
        "development": 32,
        "target_holdout": 192,
        "safety_control": 64,
        "total": 288,
    }
    assert not (EXAMPLE / "vercel-react-best-practices-conference-development-v1.yaml").exists()
    assert not (EXAMPLE / "vercel-react-best-practices-conference-holdout-v1.yaml").exists()


def test_protocol_and_json_schema_reserve_real_public_repository_design() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert protocol["sampling"]["quotas"]["development"]["total"] == 32
    assert protocol["sampling"]["quotas"]["target_holdout"]["total"] == 192
    assert protocol["sampling"]["quotas"]["safety_control"]["total"] == 64
    assert protocol["estimand"]["cluster"] == "canonical GitHub repository"
    assert schema["properties"]["entries"]["minItems"] == 288
    assert schema["properties"]["entries"]["maxItems"] == 288


def test_complete_structural_frame_accepts_288_unique_repository_clusters() -> None:
    module = _load_validator()
    frame = _fake_frame(module)
    accepted = module.validate_structure(frame)
    assert len(accepted["entries"]) == 288


def test_structure_rejects_repository_pseudoreplication() -> None:
    module = _load_validator()
    frame = _fake_frame(module)
    rows = frame["entries"]
    assert isinstance(rows, list)
    first = rows[0]
    second = rows[1]
    assert isinstance(first, dict) and isinstance(second, dict)
    second["repository_url"] = first["repository_url"]
    second["repository_owner"] = first["repository_owner"]
    second["repository_name"] = first["repository_name"]
    second["provenance_cluster_id"] = first["provenance_cluster_id"]
    record_id = second["public_record_id"]
    second["public_record_url"] = f"{first['repository_url']}/commit/{record_id}"
    second["selection_score_sha256"] = module._selection_score(
        "1" * 64, second["repository_url"], second["public_record_url"]
    )
    _redigest(module, frame)
    with pytest.raises(module.SamplingFrameError, match="repository clusters are reused"):
        module.validate_structure(frame)


def test_structure_rejects_non_typescript_target() -> None:
    module = _load_validator()
    frame = _fake_frame(module)
    rows = frame["entries"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    rows[0]["target_stack_evidence"]["typescript_paths"] = ["app/page.jsx"]
    rows[0]["target_stack_evidence"]["target_paths"] = ["app/page.jsx"]
    _redigest(module, frame)
    with pytest.raises(module.SamplingFrameError, match="only .ts or .tsx"):
        module.validate_structure(frame)


def test_structure_rejects_unqualified_or_mutable_selection() -> None:
    module = _load_validator()
    frame = _fake_frame(module)
    rows = frame["entries"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    rows[0]["selection_score_sha256"] = "f" * 64
    _redigest(module, frame)
    with pytest.raises(module.SamplingFrameError, match="selection score"):
        module.validate_structure(frame)


def test_bundle_fails_closed_before_missing_sources_can_create_specs(
    tmp_path: Path,
) -> None:
    module = _load_validator()
    frame = _fake_frame(module)
    frame_path = tmp_path / "frame.json"
    frame_path.write_text(json.dumps(frame), encoding="utf-8")
    with pytest.raises(module.SamplingFrameError, match="referenced immutable file is absent"):
        module.validate_bundle(frame_path, repository_root=tmp_path)


def test_mutating_blocker_receipt_breaks_its_digest() -> None:
    value = copy.deepcopy(json.loads(BLOCKERS_PATH.read_text(encoding="utf-8")))
    supplied = value.pop("receipt_digest")
    value["eligible_reviewed_sources_present"] = 1
    assert supplied != _digest(value)
