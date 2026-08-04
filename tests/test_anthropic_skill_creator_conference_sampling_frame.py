from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE = Path("examples/comparisons/anthropic-skill-creator-upgrade")
VALIDATOR = EXAMPLE / "validate_conference_sampling_frame_v1.py"
SCHEMA = EXAMPLE / "conference-sampling-frame-v1.schema.json"
PROTOCOL = EXAMPLE / "conference-sampling-frame-protocol-v1.json"
BLOCKERS = EXAMPLE / "conference-sampling-frame-blockers-v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("anthropic_sampling_frame", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _git_sha(text: str) -> str:
    return hashlib.sha1(text.encode(), usedforsecurity=False).hexdigest()


def _synthetic_complete_structure(module) -> dict:
    """Build an in-memory validator fixture, never a campaign source artifact."""

    seed = _sha256("validator-only-seed")
    entries: list[dict] = []
    repository_index = 0
    prefixes = {
        "development": "dev",
        "target_holdout": "target",
        "safety_control": "control",
    }
    for partition, strata in module.EXPECTED_STRATA.items():
        for stratum, count in strata.items():
            stratum_entries: list[dict] = []
            for _local_index in range(count):
                repository_index += 1
                repository_name = f"validator-fixture-{repository_index:03d}"
                repository_url = (
                    f"https://github.com/fugue-sampling-fixture/{repository_name}"
                )
                commit_sha = _git_sha(f"commit-{repository_index}")
                skill_path = f"skills/public-skill-{repository_index:03d}"
                skill_blob_url = (
                    f"{repository_url}/blob/{commit_sha}/{skill_path}/SKILL.md"
                )
                population_record = {
                    "repository_url": repository_url,
                    "commit_sha": commit_sha,
                    "tree_sha": _git_sha(f"tree-{repository_index}"),
                    "skill_path": skill_path,
                    "skill_tree_sha": _git_sha(f"skill-tree-{repository_index}"),
                    "skill_blob_url": skill_blob_url,
                    "skill_content_sha256": _sha256(f"skill-{repository_index}"),
                    "public_basis_url": skill_blob_url,
                    "public_basis_kind": "skill_blob",
                    "public_basis_id": f"{commit_sha}:{skill_path}/SKILL.md",
                }
                selection_score = module._selection_score(  # noqa: SLF001
                    seed, repository_url, skill_path, skill_blob_url
                )
                task_id = (
                    f"as-conf-{prefixes[partition]}-fixture-{repository_index:03d}"
                )
                stratum_entries.append(
                    {
                        "task_id": task_id,
                        "partition": partition,
                        "stratum": stratum,
                        "selection_rank": 0,
                        "repository_url": repository_url,
                        "repository_owner": "fugue-sampling-fixture",
                        "repository_name": repository_name,
                        "commit_sha": commit_sha,
                        "tree_sha": population_record["tree_sha"],
                        "skill_path": skill_path,
                        "skill_tree_sha": population_record["skill_tree_sha"],
                        "skill_blob_url": skill_blob_url,
                        "skill_content_sha256": population_record[
                            "skill_content_sha256"
                        ],
                        "public_basis_url": skill_blob_url,
                        "public_basis_kind": "skill_blob",
                        "public_basis_id": population_record["public_basis_id"],
                        "provenance_cluster_id": (
                            f"github:fugue-sampling-fixture/{repository_name}"
                        ),
                        "population_record_digest": module.stable_digest(
                            population_record
                        ),
                        "selection_score_sha256": selection_score,
                        "source_archive_path": f"fixtures/{task_id}.tar",
                        "source_archive_sha256": _sha256(f"archive-{task_id}"),
                        "source_manifest_path": f"fixtures/{task_id}.manifest.json",
                        "source_manifest_sha256": _sha256(f"manifest-{task_id}"),
                        "provenance_receipt_path": (
                            f"fixtures/{task_id}.provenance.json"
                        ),
                        "provenance_receipt_sha256": _sha256(
                            f"provenance-{task_id}"
                        ),
                        "public_task_digest": _sha256(f"public-{task_id}"),
                        "private_label_digest": _sha256(f"private-{task_id}"),
                        "license": {
                            "spdx_id": "MIT",
                            "path": "LICENSE",
                            "sha256": _sha256(f"license-{repository_index}"),
                        },
                    }
                )
            stratum_entries.sort(key=lambda row: row["selection_score_sha256"])
            for rank, row in enumerate(stratum_entries, 1):
                row["selection_rank"] = rank
            entries.extend(stratum_entries)

    frame = {
        "schema_version": 1,
        "frame_id": module.FRAME_ID,
        "frame_digest": "0" * 64,
        "reserved_studies": {
            "development": module.DEVELOPMENT_STUDY_ID,
            "holdout": module.HOLDOUT_STUDY_ID,
        },
        "skill_treatment": {
            "repository": module.EXPECTED_SKILL_REPOSITORY,
            "path": "skills/skill-creator",
            "lock_path": "fixtures/treatment.lock.json",
            "lock_sha256": _sha256("lock"),
            "baseline_commit": module.EXPECTED_BASELINE_COMMIT,
            "baseline_bundle_sha256": _sha256("baseline"),
            "candidate_commit": module.EXPECTED_CANDIDATE_COMMIT,
            "candidate_bundle_sha256": _sha256("candidate"),
        },
        "population_snapshot": {
            "selection_cutoff_utc": "2026-08-03T00:00:00Z",
            "records_path": "fixtures/population.jsonl",
            "records_sha256": _sha256("population"),
            "record_count": module.EXPECTED_TOTAL,
            "query_receipt_path": "fixtures/query.json",
            "query_receipt_sha256": _sha256("query"),
            "eligibility_protocol_path": "fixtures/eligibility.json",
            "eligibility_protocol_sha256": _sha256("eligibility"),
            "selection_code_path": "fixtures/select.py",
            "selection_code_sha256": _sha256("selection"),
        },
        "selection_protocol": {
            "seed_sha256": seed,
            "algorithm": "sha256_seeded_sort_v1",
            "cluster_unit": "canonical_github_repository",
            "unique_clusters_across_partitions": True,
            "strata": module.EXPECTED_STRATA,
        },
        "public_tasks": {
            "path": "fixtures/public-tasks.jsonl",
            "sha256": _sha256("public tasks"),
            "count": module.EXPECTED_TOTAL,
        },
        "private_labels": {
            "path": "fixtures/private-labels.jsonl",
            "sha256": _sha256("private labels"),
            "count": module.EXPECTED_TOTAL,
            "host_only": True,
            "required_mode": "0600",
        },
        "entries": entries,
    }
    unsigned = dict(frame)
    unsigned.pop("frame_digest")
    frame["frame_digest"] = module.stable_digest(unsigned)
    return frame


def _resign(module, frame: dict) -> None:
    unsigned = dict(frame)
    unsigned.pop("frame_digest")
    frame["frame_digest"] = module.stable_digest(unsigned)


def test_protocol_reserves_powered_studies_but_authorizes_no_execution() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    blockers = json.loads(BLOCKERS.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))

    assert protocol["status"] == "prospective_not_yet_sampled"
    assert protocol["reserved_studies"] == {
        "development": "anthropic-skill-creator-conference-development-v1",
        "holdout": "anthropic-skill-creator-conference-holdout-v1",
    }
    assert "authorizes no execution" in protocol["sealing"]["approval"]
    assert protocol["estimand"]["cluster"] == "canonical GitHub repository"
    assert schema["properties"]["entries"]["minItems"] == 288
    assert blockers["prepared_eligible_public_sources_present"] == 0
    assert blockers["required_independent_repository_clusters"]["total"] == 288
    assert blockers["exact_missing_sources"]["development"]["total"] == 32
    assert blockers["exact_missing_sources"]["target_holdout"]["total"] == 192
    assert blockers["exact_missing_sources"]["safety_control"]["total"] == 64


def test_powered_study_and_task_artifacts_fail_closed_until_sources_exist() -> None:
    names = {
        "anthropic-skill-creator-conference-development-v1.yaml",
        "anthropic-skill-creator-conference-holdout-v1.yaml",
        "conference-development-tasks-v1.jsonl",
        "conference-holdout-tasks-v1.jsonl",
        "conference-private-labels-v1.jsonl",
        "conference-sampling-frame-v1.json",
    }
    assert names.isdisjoint(path.name for path in EXAMPLE.iterdir())


def test_complete_structure_requires_288_independent_public_repositories() -> None:
    module = _module()
    frame = _synthetic_complete_structure(module)
    validated = module.validate_structure(frame)

    assert len(validated["entries"]) == 288
    assert len(
        {entry["provenance_cluster_id"] for entry in validated["entries"]}
    ) == 288


def test_repository_reuse_is_rejected_as_pseudoreplication() -> None:
    module = _module()
    frame = _synthetic_complete_structure(module)
    first, second = frame["entries"][:2]
    for field in (
        "repository_url",
        "repository_owner",
        "repository_name",
        "commit_sha",
        "tree_sha",
        "skill_path",
        "skill_tree_sha",
        "skill_blob_url",
        "skill_content_sha256",
        "public_basis_url",
        "public_basis_kind",
        "public_basis_id",
        "provenance_cluster_id",
        "population_record_digest",
        "selection_score_sha256",
    ):
        second[field] = first[field]
    _resign(module, frame)

    with pytest.raises(module.SamplingFrameError, match="reused"):
        module.validate_structure(frame)


def test_cross_repository_public_basis_is_rejected() -> None:
    module = _module()
    frame = _synthetic_complete_structure(module)
    entry = frame["entries"][0]
    entry["public_basis_kind"] = "issue"
    entry["public_basis_id"] = "1"
    entry["public_basis_url"] = "https://github.com/another/repository/issues/1"
    entry["selection_score_sha256"] = module._selection_score(  # noqa: SLF001
        frame["selection_protocol"]["seed_sha256"],
        entry["repository_url"],
        entry["skill_path"],
        entry["public_basis_url"],
    )
    _resign(module, frame)

    with pytest.raises(module.SamplingFrameError, match="another repository"):
        module.validate_structure(frame)


def test_cli_returns_blocked_for_absent_sampling_frame(tmp_path: Path) -> None:
    missing = tmp_path / "not-prepared.json"
    completed = subprocess.run(
        [sys.executable, VALIDATOR.as_posix(), missing.as_posix()],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 3
    receipt = json.loads(completed.stdout)
    assert receipt["status"] == "blocked"
    assert "not-prepared.json" in receipt["frame"]


def test_unknown_frame_fields_fail_closed() -> None:
    module = _module()
    frame = _synthetic_complete_structure(module)
    frame["invented_population_claim"] = True

    with pytest.raises(module.SamplingFrameError, match="unknown"):
        module.validate_structure(frame)


def test_schema_protocol_and_blocker_documents_are_valid_json() -> None:
    for path in (SCHEMA, PROTOCOL, BLOCKERS):
        value = json.loads(path.read_text(encoding="utf-8"))
        assert value.get("schema_version") == 1 or value.get("$schema", "").endswith(
            "2020-12/schema"
        )
