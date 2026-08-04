from __future__ import annotations

import copy
import hashlib
import json
import runpy
from pathlib import Path

import pytest

EXAMPLE = Path("examples/comparisons/superpowers-writing-plans-upgrade")
VALIDATOR_PATH = EXAMPLE / "validate_conference_sampling_frame_v1.py"
SCHEMA_PATH = EXAMPLE / "conference-sampling-frame-v1.schema.json"
PROTOCOL_PATH = EXAMPLE / "conference-sampling-frame-protocol-v1.json"
BLOCKERS_PATH = EXAMPLE / "conference-sampling-frame-blockers-v1.json"

VALIDATOR = runpy.run_path(VALIDATOR_PATH.as_posix())
SamplingFrameError = VALIDATOR["SamplingFrameError"]
validate_structure = VALIDATOR["validate_structure"]
validate_bundle = VALIDATOR["validate_bundle"]
validate_population_selection = VALIDATOR["validate_population_selection"]
stable_digest = VALIDATOR["stable_digest"]
EXPECTED_STRATA = VALIDATOR["EXPECTED_STRATA"]


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _synthetic_structure_fixture() -> dict:
    """Build non-source synthetic rows used only to exercise the validator."""

    seed = _sha("unit-test-selection-seed")
    rows: list[dict] = []
    global_index = 0
    prefix = {
        "development": "sp-dev",
        "target_holdout": "sp-target",
        "safety_control": "sp-control",
    }
    for partition, strata in EXPECTED_STRATA.items():
        for stratum, count in strata.items():
            stratum_rows: list[dict] = []
            for _local_index in range(count):
                global_index += 1
                owner = f"synthetic-owner-{global_index:03d}"
                repository = f"synthetic-repo-{global_index:03d}"
                repository_url = f"https://github.com/{owner}/{repository}"
                record_id = str(global_index)
                public_record_url = f"{repository_url}/issues/{record_id}"
                score = hashlib.sha256(
                    (
                        f"{seed}\n{repository_url.casefold()}\n"
                        f"{public_record_url}\n"
                    ).encode()
                ).hexdigest()
                stratum_rows.append(
                    {
                        "task_id": f"{prefix[partition]}-{global_index:03d}",
                        "partition": partition,
                        "stratum": stratum,
                        "selection_rank": 0,
                        "repository_url": repository_url,
                        "repository_owner": owner,
                        "repository_name": repository,
                        "commit_sha": f"{global_index:040x}",
                        "tree_sha": f"{global_index + 1000:040x}",
                        "public_record_url": public_record_url,
                        "public_record_kind": "issue",
                        "public_record_id": record_id,
                        "provenance_cluster_id": (
                            f"github:{owner.casefold()}/{repository.casefold()}"
                        ),
                        "population_record_digest": _sha(
                            f"population-record-{global_index}"
                        ),
                        "selection_score_sha256": score,
                        "source_archive_path": (
                            f".fugue/comparison-resources/superpowers-conference/"
                            f"{global_index:03d}/source.tar"
                        ),
                        "source_archive_sha256": _sha(
                            f"source-archive-{global_index}"
                        ),
                        "source_manifest_path": (
                            f".fugue/comparison-resources/superpowers-conference/"
                            f"{global_index:03d}/source.json"
                        ),
                        "source_manifest_sha256": _sha(
                            f"source-manifest-{global_index}"
                        ),
                        "provenance_receipt_path": (
                            f".fugue/comparison-resources/superpowers-conference/"
                            f"{global_index:03d}/provenance.json"
                        ),
                        "provenance_receipt_sha256": _sha(
                            f"provenance-receipt-{global_index}"
                        ),
                        "public_task_digest": _sha(f"public-task-{global_index}"),
                        "private_label_digest": _sha(f"private-label-{global_index}"),
                        "license": {
                            "spdx_id": "Apache-2.0",
                            "path": "LICENSE",
                            "sha256": _sha(f"license-{global_index}"),
                        },
                    }
                )
            stratum_rows.sort(key=lambda row: row["selection_score_sha256"])
            for rank, row in enumerate(stratum_rows, 1):
                row["selection_rank"] = rank
                rows.append(row)

    frame = {
        "schema_version": 1,
        "frame_id": "superpowers-writing-plans-conference-sampling-frame-v1",
        "frame_digest": "",
        "reserved_studies": {
            "development": "superpowers-writing-plans-conference-development-v1",
            "holdout": "superpowers-writing-plans-conference-holdout-v1",
        },
        "skill_treatment": {
            "repository": "https://github.com/obra/superpowers",
            "path": "skills/writing-plans",
            "lock_path": (
                "examples/comparisons/superpowers-writing-plans-upgrade/"
                "skill-revisions.lock.json"
            ),
            "lock_sha256": _sha("skill-lock"),
            "baseline_commit": "de4672b171213a6ff6960228d8b95c46ea0b09f4",
            "baseline_bundle_sha256": _sha("baseline-bundle"),
            "candidate_commit": "8e1262a3bae92b640d87fa81c51c53b65e490590",
            "candidate_bundle_sha256": _sha("candidate-bundle"),
        },
        "population_snapshot": {
            "selection_cutoff_utc": "2026-08-03T12:00:00Z",
            "records_path": "population.jsonl",
            "records_sha256": _sha("population"),
            "record_count": 512,
            "query_receipt_path": "query-receipt.json",
            "query_receipt_sha256": _sha("query-receipt"),
            "eligibility_protocol_path": "eligibility.json",
            "eligibility_protocol_sha256": _sha("eligibility"),
            "selection_code_path": "select.py",
            "selection_code_sha256": _sha("selection-code"),
        },
        "selection_protocol": {
            "seed_sha256": seed,
            "algorithm": "sha256_seeded_sort_v1",
            "cluster_unit": "canonical_github_repository",
            "unique_clusters_across_partitions": True,
            "strata": EXPECTED_STRATA,
        },
        "public_tasks": {
            "path": "public-tasks.jsonl",
            "sha256": _sha("public-tasks"),
            "count": 288,
        },
        "private_labels": {
            "path": "private-labels.jsonl",
            "sha256": _sha("private-labels"),
            "count": 288,
            "host_only": True,
            "required_mode": "0600",
        },
        "entries": rows,
    }
    digest_payload = dict(frame)
    digest_payload.pop("frame_digest")
    frame["frame_digest"] = stable_digest(digest_payload)
    return frame


def _redigest(frame: dict) -> None:
    payload = dict(frame)
    payload.pop("frame_digest")
    frame["frame_digest"] = stable_digest(payload)


def _attach_synthetic_population(frame: dict) -> list[dict]:
    task_serial = 0
    for stratum in EXPECTED_STRATA["development"]:
        selected = [
            entry
            for entry in frame["entries"]
            if entry["stratum"] == stratum
            and entry["partition"] in {"development", "target_holdout"}
        ]
        selected.sort(key=lambda entry: entry["selection_score_sha256"])
        for index, entry in enumerate(selected):
            task_serial += 1
            if index < EXPECTED_STRATA["development"][stratum]:
                entry["partition"] = "development"
                entry["task_id"] = f"sp-dev-selected-{task_serial:03d}"
                entry["selection_rank"] = index + 1
            else:
                entry["partition"] = "target_holdout"
                entry["task_id"] = f"sp-target-selected-{task_serial:03d}"
                entry["selection_rank"] = (
                    index - EXPECTED_STRATA["development"][stratum] + 1
                )
    for stratum in EXPECTED_STRATA["safety_control"]:
        selected = sorted(
            (
                entry
                for entry in frame["entries"]
                if entry["stratum"] == stratum
                and entry["partition"] == "safety_control"
            ),
            key=lambda entry: entry["selection_score_sha256"],
        )
        for index, entry in enumerate(selected, 1):
            task_serial += 1
            entry["task_id"] = f"sp-control-selected-{task_serial:03d}"
            entry["selection_rank"] = index

    population: list[dict] = []
    for entry in frame["entries"]:
        record = {
            "schema_version": 1,
            "record_id": f"sp-pop-{_sha(entry['repository_url'])[:16]}",
            "repository_url": entry["repository_url"],
            "repository_owner": entry["repository_owner"],
            "repository_name": entry["repository_name"],
            "commit_sha": entry["commit_sha"],
            "tree_sha": entry["tree_sha"],
            "public_record_url": entry["public_record_url"],
            "public_record_kind": entry["public_record_kind"],
            "public_record_id": entry["public_record_id"],
            "public_record_snapshot_sha256": _sha(
                f"public-record-snapshot-{entry['task_id']}"
            ),
            "provenance_cluster_id": entry["provenance_cluster_id"],
            "sampling_family": (
                "safety_control"
                if entry["partition"] == "safety_control"
                else "target"
            ),
            "stratum": entry["stratum"],
            "selection_score_sha256": entry["selection_score_sha256"],
            "source_archive_path": entry["source_archive_path"],
            "source_archive_sha256": entry["source_archive_sha256"],
            "source_manifest_path": entry["source_manifest_path"],
            "source_manifest_sha256": entry["source_manifest_sha256"],
            "provenance_receipt_path": entry["provenance_receipt_path"],
            "provenance_receipt_sha256": entry["provenance_receipt_sha256"],
            "eligibility_receipt_sha256": _sha(
                f"eligibility-{entry['repository_url']}"
            ),
            "license": entry["license"],
        }
        entry["population_record_digest"] = stable_digest(record)
        population.append(record)
    _redigest(frame)
    return population


def test_sampling_frame_schema_and_protocol_reserve_only_new_studies() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert schema["properties"]["entries"]["minItems"] == 288
    assert schema["properties"]["entries"]["maxItems"] == 288
    assert protocol["status"] == "prospective_not_yet_sampled"
    assert protocol["sampling"]["quotas"]["development"]["total"] == 32
    assert protocol["sampling"]["quotas"]["target_holdout"]["total"] == 192
    assert protocol["sampling"]["quotas"]["safety_control"]["total"] == 64
    assert "No clone, fetch, install, download" in protocol["preparation"][
        "trial_restrictions"
    ][0]
    assert protocol["readiness"]["required_status"] == "ready"


def test_sampling_frame_blocker_does_not_claim_pseudo_independent_sources() -> None:
    blockers = json.loads(BLOCKERS_PATH.read_text(encoding="utf-8"))
    assert blockers["status"] == "blocked_missing_reviewed_public_sampling_frame"
    assert blockers["eligible_reviewed_sources_present"] == 0
    assert blockers["exact_missing_sources"] == {
        "development": 32,
        "target_holdout": 192,
        "safety_control": 64,
        "total": 288,
    }
    assert blockers["known_historical_source"]["eligibility"] == (
        "not_eligible_for_powered_frame"
    )
    assert all("comparison spec" in value for value in blockers["not_generated"][:2])


def test_structural_validator_accepts_exact_powered_shape() -> None:
    frame = _synthetic_structure_fixture()
    assert validate_structure(frame)["frame_digest"] == frame["frame_digest"]


def test_population_selection_uses_first_frozen_independent_slices() -> None:
    frame = _synthetic_structure_fixture()
    population = _attach_synthetic_population(frame)
    validate_structure(frame)
    resolved = validate_population_selection(frame, population)
    assert len(resolved) == 288


def test_population_selection_rejects_a_hand_picked_holdout() -> None:
    frame = _synthetic_structure_fixture()
    population = _attach_synthetic_population(frame)
    target = next(
        entry for entry in frame["entries"] if entry["partition"] == "target_holdout"
    )
    development = next(
        entry
        for entry in frame["entries"]
        if entry["partition"] == "development" and entry["stratum"] == target["stratum"]
    )
    target["population_record_digest"], development["population_record_digest"] = (
        development["population_record_digest"],
        target["population_record_digest"],
    )
    with pytest.raises(SamplingFrameError, match="disagrees with population field"):
        validate_population_selection(frame, population)


def test_structural_validator_rejects_repository_pseudoreplication() -> None:
    frame = _synthetic_structure_fixture()
    frame["entries"][1]["repository_url"] = frame["entries"][0]["repository_url"]
    frame["entries"][1]["repository_owner"] = frame["entries"][0][
        "repository_owner"
    ]
    frame["entries"][1]["repository_name"] = frame["entries"][0][
        "repository_name"
    ]
    frame["entries"][1]["provenance_cluster_id"] = frame["entries"][0][
        "provenance_cluster_id"
    ]
    frame["entries"][1]["public_record_url"] = (
        f"{frame['entries'][1]['repository_url']}/issues/9999"
    )
    frame["entries"][1]["public_record_id"] = "9999"
    frame["entries"][1]["selection_score_sha256"] = hashlib.sha256(
        (
            f"{frame['selection_protocol']['seed_sha256']}\n"
            f"{frame['entries'][1]['repository_url'].casefold()}\n"
            f"{frame['entries'][1]['public_record_url']}\n"
        ).encode()
    ).hexdigest()
    _redigest(frame)
    with pytest.raises(SamplingFrameError, match="repositories are reused"):
        validate_structure(frame)


def test_structural_validator_rejects_post_selection_cherry_pick() -> None:
    frame = _synthetic_structure_fixture()
    frame["entries"][0]["selection_score_sha256"] = "0" * 64
    _redigest(frame)
    with pytest.raises(SamplingFrameError, match="does not match frozen seed"):
        validate_structure(frame)


def test_structural_validator_rejects_private_label_exposure_contract() -> None:
    frame = _synthetic_structure_fixture()
    frame["private_labels"]["host_only"] = False
    _redigest(frame)
    with pytest.raises(SamplingFrameError, match="host-only"):
        validate_structure(frame)


@pytest.mark.parametrize(
    "unsafe_path",
    ["/absolute/source.tar", "../source.tar", "source//archive.tar", "source/./archive.tar", "source\\archive.tar"],
)
def test_structural_validator_rejects_unsafe_resource_paths(unsafe_path: str) -> None:
    frame = _synthetic_structure_fixture()
    frame["entries"][0]["source_archive_path"] = unsafe_path
    _redigest(frame)
    with pytest.raises(SamplingFrameError, match="safe repository-relative path"):
        validate_structure(frame)


def test_bundle_validator_fails_closed_when_materialized_sources_are_absent(
    tmp_path: Path,
) -> None:
    frame = _synthetic_structure_fixture()
    path = tmp_path / "frame.json"
    path.write_text(json.dumps(frame), encoding="utf-8")
    with pytest.raises(SamplingFrameError, match="absent or a symlink"):
        validate_bundle(path, repository_root=tmp_path)


def test_frame_digest_binds_every_entry() -> None:
    frame = _synthetic_structure_fixture()
    changed = copy.deepcopy(frame)
    changed["entries"][0]["private_label_digest"] = _sha("changed-label")
    with pytest.raises(SamplingFrameError, match="frame_digest"):
        validate_structure(changed)
