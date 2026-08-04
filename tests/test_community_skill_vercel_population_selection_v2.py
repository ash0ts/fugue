from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "examples"
    / "comparisons"
    / "community-skill-upgrades"
    / "select_vercel_population_v2.py"
)
PROTOCOL_PATH = (
    ROOT
    / "examples"
    / "comparisons"
    / "vercel-react-best-practices-upgrade"
    / "conference-sampling-frame-protocol-v2.json"
)
QUERY_PLAN_PATH = PROTOCOL_PATH.with_name("conference-query-plan-v2.json")
SPEC = importlib.util.spec_from_file_location(
    "select_vercel_population_v2", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _identity(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


REVIEWER_A = _identity("reviewer-a")
REVIEWER_B = _identity("reviewer-b")
ADJUDICATOR = _identity("adjudicator")


def _receipt(value: dict[str, Any]) -> dict[str, Any]:
    value["receipt_sha256"] = MODULE.stable_digest(value)
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(_json_bytes(row) for row in rows)


def _detached_authentication(
    private_key: Any,
    *,
    purpose: str,
    payload: dict[str, Any],
    authenticated_at_utc: str = "2026-01-01T00:00:00Z",
) -> dict[str, Any]:
    from cryptography.hazmat.primitives import serialization

    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    receipt = {
        "schema_version": 2,
        "purpose": purpose,
        "algorithm": MODULE.AUTHENTICATION_ALGORITHM,
        "key_id_sha256": hashlib.sha256(public_key).hexdigest(),
        "public_key_hex": public_key.hex(),
        "payload_sha256": MODULE.stable_digest(payload),
        "signature_hex": private_key.sign(
            MODULE._authentication_message(  # noqa: SLF001
                purpose,
                payload,
                authenticated_at_utc=authenticated_at_utc,
            )
        ).hex(),
        "authenticated_at_utc": authenticated_at_utc,
    }
    receipt["receipt_sha256"] = MODULE.stable_digest(receipt)
    return receipt


def _role_separation() -> dict[str, Any]:
    return _receipt(
        {
            "schema_version": 2,
            "eligibility_reviewer_id_hashes": sorted([REVIEWER_A, REVIEWER_B]),
            "adjudicator_id_hashes": [ADJUDICATOR],
            "public_task_author_id_hashes": [_identity("task-author")],
            "public_test_author_id_hashes": [_identity("public-test-author")],
            "host_verifier_author_id_hashes": [_identity("host-verifier-author")],
            "pairwise_disjoint": True,
        }
    )


def _calibration(role_separation: dict[str, Any]) -> dict[str, Any]:
    del role_separation
    return _receipt(
        {
            "schema_version": 2,
            "calibration_id": "vercel-reviewer-calibration-v2",
            "completed_at_utc": "2026-01-01T12:00:00Z",
            "reviewer_id_hashes": sorted([REVIEWER_A, REVIEWER_B, ADJUDICATOR]),
            "family_examples": {
                family: {"acceptable": 4, "unacceptable": 4}
                for family in MODULE.FAMILIES
            },
            "record_count": 48,
            "cohens_kappa_micros": 850_000,
            "critical_false_inclusions": 0,
            "treatment_outputs_blinded": True,
        }
    )


def _population_record(
    number: int,
    *,
    owner_id: str,
    repository_id: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": 2,
        "record_id": f"record-{number:04d}",
        "repository_id": repository_id or f"repository-{number:04d}",
        "owner_id": owner_id,
        "canonical_pull_request_url": (
            f"https://github.com/public-owner/repo-{number}/pull/1"
        ),
        "acquisition_candidate_sha256": _identity(f"acquisition-{number}"),
        "pre_treatment": True,
        "source_facts_sha256": _identity(f"facts-{number}"),
        "repository_receipt_sha256": _identity(f"repository-receipt-{number}"),
        "pull_request_receipt_sha256": _identity(f"pull-receipt-{number}"),
        "source_receipt_sha256": _identity(f"source-receipt-{number}"),
        "stack_receipt_sha256": _identity(f"stack-receipt-{number}"),
    }
    value["source_record_sha256"] = MODULE.stable_digest(value)
    return value


def _lineage_row(
    population: dict[str, Any],
    *,
    roots: list[str] | None = None,
) -> dict[str, Any]:
    canonical_roots = sorted(roots or [_identity(f"root-{population['record_id']}")])
    value: dict[str, Any] = {
        "schema_version": 2,
        "record_id": population["record_id"],
        "repository_id": population["repository_id"],
        "owner_id": population["owner_id"],
        "canonical_roots": canonical_roots,
        "lineage_id": MODULE._lineage_digest(canonical_roots),  # noqa: SLF001
    }
    value["lineage_row_sha256"] = MODULE.stable_digest(value)
    return value


def _reviewer(
    reviewer_id_hash: str,
    population: dict[str, Any],
    lineage: dict[str, Any],
    *,
    family: str,
    calibration_sha256: str,
    roles_sha256: str,
    role: str = "eligibility_reviewer",
    decision: str = "eligible",
    reason_codes: list[str] | None = None,
) -> dict[str, Any]:
    eligible = decision == "eligible"
    value: dict[str, Any] = {
        "reviewer_id_hash": reviewer_id_hash,
        "role": role,
        "record_id": population["record_id"],
        "source_record_sha256": population["source_record_sha256"],
        "lineage_row_sha256": lineage["lineage_row_sha256"],
        "decision": decision,
        "family": family if eligible else None,
        "lineage_id": lineage["lineage_id"] if eligible else None,
        "reason_codes": reason_codes or [],
        "natural_repair_confirmed": eligible,
        "task_authorable_without_gold": eligible,
        "decision_timestamp_utc": "2026-01-01T18:00:00Z",
        "baseline_candidate_skills_blinded": True,
        "treatment_arm_labels_blinded": True,
        "selection_seed_and_score_blinded": True,
        "partition_blinded": True,
        "agent_judge_outputs_blinded": True,
        "other_reviewer_decision_blinded": True,
        "calibration_receipt_sha256": calibration_sha256,
        "role_separation_receipt_sha256": roles_sha256,
    }
    value["decision_sha256"] = MODULE.stable_digest(value)
    return value


def _empty_adjudication(
    population: dict[str, Any],
    lineage: dict[str, Any],
    *,
    calibration_sha256: str,
    roles_sha256: str,
) -> dict[str, Any]:
    value = {key: None for key in MODULE._ADJUDICATION_KEYS}  # noqa: SLF001
    value.update(
        {
            "required": False,
            "record_id": population["record_id"],
            "source_record_sha256": population["source_record_sha256"],
            "lineage_row_sha256": lineage["lineage_row_sha256"],
            "calibration_receipt_sha256": calibration_sha256,
            "role_separation_receipt_sha256": roles_sha256,
        }
    )
    return value


def _agreed_review(
    population: dict[str, Any],
    lineage: dict[str, Any],
    *,
    family: str,
    calibration_sha256: str,
    roles_sha256: str,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": 2,
        "record_id": population["record_id"],
        "source_record_sha256": population["source_record_sha256"],
        "reviewer_a": _reviewer(
            REVIEWER_A,
            population,
            lineage,
            family=family,
            calibration_sha256=calibration_sha256,
            roles_sha256=roles_sha256,
        ),
        "reviewer_b": _reviewer(
            REVIEWER_B,
            population,
            lineage,
            family=family,
            calibration_sha256=calibration_sha256,
            roles_sha256=roles_sha256,
        ),
        "adjudication": _empty_adjudication(
            population,
            lineage,
            calibration_sha256=calibration_sha256,
            roles_sha256=roles_sha256,
        ),
        "exclusion_reasons": [],
    }
    value["review_row_sha256"] = MODULE.stable_digest(value)
    return value


def _verifier_lock(provider: str, endpoint: str) -> dict[str, Any]:
    drand = provider == "drand-mainnet"
    return _receipt(
        {
            "schema_version": 2,
            "provider": provider,
            "verifier_id": "locked-public-beacon-verifier-v2",
            "verifier_code_sha256": _identity("verifier-code-v2"),
            "canonical_endpoint": endpoint,
            "trust_root_kind": "drand_chain_hash"
            if drand
            else "nist_certificate_bundle",
            "trust_root_sha256": _identity("drand-chain" if drand else "nist-certs"),
            "signature_algorithm": "bls12-381-g2" if drand else "nist-signed-pulse-v2",
            "locked_at_utc": "2026-01-01T06:00:00Z",
        }
    )


def _solver_runtime() -> dict[str, Any]:
    return _receipt(
        {
            "schema_version": 2,
            "runtime_id": "vercel-selector-runtime-v2",
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "selector_algorithm": "exact_min_cost_flow_owner_branching_v2",
        }
    )


def _frame(
    *,
    provider: str = "drand-mainnet",
    family_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    counts = dict(MODULE.MINIMUM_ELIGIBLE_LINEAGES)
    if family_counts:
        counts.update(family_counts)
    roles = _role_separation()
    calibration = _calibration(roles)
    roles_raw = _json_bytes(roles)
    calibration_raw = _json_bytes(calibration)
    roles_sha = hashlib.sha256(roles_raw).hexdigest()
    calibration_sha = hashlib.sha256(calibration_raw).hexdigest()
    population: list[dict[str, Any]] = []
    lineages: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    number = 0
    for family in MODULE.FAMILIES:
        for _index in range(counts[family]):
            number += 1
            record = _population_record(number, owner_id=f"owner-{number:04d}")
            lineage = _lineage_row(record)
            population.append(record)
            lineages.append(lineage)
            reviews.append(
                _agreed_review(
                    record,
                    lineage,
                    family=family,
                    calibration_sha256=calibration_sha,
                    roles_sha256=roles_sha,
                )
            )

    protocol_raw = PROTOCOL_PATH.read_bytes()
    query_raw = QUERY_PLAN_PATH.read_bytes()
    population_raw = _jsonl_bytes(population)
    reviews_raw = _jsonl_bytes(reviews)
    lineages_raw = _jsonl_bytes(lineages)
    code_raw = MODULE_PATH.read_bytes()
    runtime = _solver_runtime()
    runtime_raw = _json_bytes(runtime)
    pulse: str | int
    random_value: str
    if provider == "drand-mainnet":
        pulse = 123456
        endpoint = "https://api.drand.sh/public/123456"
        random_value = "a" * 64
    else:
        pulse = "1767398400000"
        endpoint = "https://beacon.nist.gov/beacon/2.0/pulse/time/1767398400000"
        random_value = "a" * 128
    verifier = _verifier_lock(provider, endpoint)
    verifier_raw = _json_bytes(verifier)
    plan: dict[str, Any] = {
        "schema_version": 2,
        "selection_id": "vercel-react-public-population-v2",
        "protocol_id": MODULE.PROTOCOL_ID,
        "protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(),
        "population_sha256": hashlib.sha256(population_raw).hexdigest(),
        "review_ledger_sha256": hashlib.sha256(reviews_raw).hexdigest(),
        "lineage_ledger_sha256": hashlib.sha256(lineages_raw).hexdigest(),
        "query_plan_sha256": hashlib.sha256(query_raw).hexdigest(),
        "calibration_receipt_sha256": calibration_sha,
        "role_separation_receipt_sha256": roles_sha,
        "precommit_receipt_sha256": "0" * 64,
        "verifier_lock_sha256": hashlib.sha256(verifier_raw).hexdigest(),
        "randomness_receipt_sha256": "0" * 64,
        "selection_code_sha256": hashlib.sha256(code_raw).hexdigest(),
        "solver_runtime_sha256": hashlib.sha256(runtime_raw).hexdigest(),
        "acquisition_candidates_sha256": "0" * 64,
        "repository_receipts_sha256": "0" * 64,
        "pull_request_receipts_sha256": "0" * 64,
        "source_receipts_sha256": "0" * 64,
        "stack_receipts_sha256": "0" * 64,
        "calibration_ledger_sha256": "0" * 64,
        "review_authentication_sha256": "0" * 64,
        "source_authentication_sha256": "0" * 64,
        "beacon_response_sha256": "0" * 64,
        "beacon_authentication_sha256": "0" * 64,
        "power_receipt_sha256": "0" * 64,
        "population_frozen_at_utc": "2026-01-01T00:00:00Z",
        "beacon_target": {
            "provider": provider,
            "pulse_or_round": pulse,
            "canonical_endpoint": endpoint,
        },
        "precommit_commitment_sha256": "0" * 64,
        "partition_order": list(MODULE.PARTITION_ORDER),
        "family_order_by_partition": {
            partition: list(MODULE.FAMILY_ORDER_BY_PARTITION[partition])
            for partition in MODULE.PARTITION_ORDER
        },
        "slot_manifest_sha256": MODULE.stable_digest(MODULE._slot_manifest()),  # noqa: SLF001
        "quotas": copy.deepcopy(MODULE.EXPECTED_QUOTAS),
        "owner_caps": {"all_partitions": 4, "target_holdout": 2},
        "reserve_fraction": {"numerator": 1, "denominator": 4},
        "minimum_eligible_lineages": dict(MODULE.MINIMUM_ELIGIBLE_LINEAGES),
        "minimum_unique_lineages": MODULE.MINIMUM_UNIQUE_LINEAGES,
        "bounds": dict(MODULE.EXPECTED_BOUNDS),
    }
    commitment = MODULE.selection_commitment_digest(plan)
    plan["precommit_commitment_sha256"] = commitment
    precommit = _receipt(
        {
            "schema_version": 2,
            "selection_id": plan["selection_id"],
            "commitment_sha256": commitment,
            "commitment_artifact_sha256": MODULE.stable_digest(
                MODULE._commitment_artifact(plan)  # noqa: SLF001
            ),
            "committed_at_utc": "2026-01-02T00:00:00Z",
            "public_commit_url": (
                "https://github.com/example/public-frame/commit/" + "e" * 40
            ),
            "public_commit_sha": "e" * 40,
            "publication_observed_at_utc": "2026-01-02T00:01:00Z",
            "publication_evidence_sha256": _identity("precommit-publication-evidence"),
            "publication_authentication": {},
            "beacon_target": copy.deepcopy(plan["beacon_target"]),
        }
    )
    precommit_raw = _json_bytes(precommit)
    randomness = _receipt(
        {
            "schema_version": 2,
            "provider": provider,
            "canonical_endpoint": endpoint,
            "pulse_or_round": pulse,
            "published_at_utc": "2026-01-03T00:00:00Z",
            "retrieved_at_utc": "2026-01-03T00:01:00Z",
            "verified_at_utc": "2026-01-03T00:02:00Z",
            "random_value": random_value,
            "signature_sha256": _identity("beacon-signature"),
            "signature_verified": True,
            "verification_material_sha256": _identity("verification-material"),
            "response_sha256": _identity("beacon-response"),
            "commitment_sha256": commitment,
            "verifier_lock_sha256": hashlib.sha256(verifier_raw).hexdigest(),
            "verifier_code_sha256": verifier["verifier_code_sha256"],
            "trust_root_sha256": verifier["trust_root_sha256"],
            "population_sha256": plan["population_sha256"],
            "review_ledger_sha256": plan["review_ledger_sha256"],
            "lineage_ledger_sha256": plan["lineage_ledger_sha256"],
            "query_plan_sha256": plan["query_plan_sha256"],
            "calibration_receipt_sha256": calibration_sha,
            "role_separation_receipt_sha256": roles_sha,
            "selection_code_sha256": plan["selection_code_sha256"],
        }
    )
    randomness_raw = _json_bytes(randomness)
    plan["precommit_receipt_sha256"] = hashlib.sha256(precommit_raw).hexdigest()
    plan["randomness_receipt_sha256"] = hashlib.sha256(randomness_raw).hexdigest()
    # The excluded circular receipt fields do not change the public commitment.
    assert MODULE.selection_commitment_digest(plan) == commitment
    artifacts = MODULE.SelectionArtifactBytes(
        plan=_json_bytes(plan),
        protocol=protocol_raw,
        population=population_raw,
        review_ledger=reviews_raw,
        lineage_ledger=lineages_raw,
        query_plan=query_raw,
        calibration_receipt=calibration_raw,
        role_separation_receipt=roles_raw,
        precommit_receipt=precommit_raw,
        verifier_lock=verifier_raw,
        randomness_receipt=randomness_raw,
        solver_runtime=runtime_raw,
        selection_code=code_raw,
    )
    return {
        "artifacts": artifacts,
        "plan": plan,
        "population": population,
        "lineages": lineages,
        "reviews": reviews,
        "calibration": calibration,
        "roles": roles,
        "precommit": precommit,
        "verifier": verifier,
        "randomness": randomness,
    }


def _candidate(number: int, *, owner: str, family: str, score: int) -> Any:
    record = MODULE.PopulationRecord(
        record_id=f"record-{number}",
        repository_id=f"repository-{number}",
        owner_id=owner,
        canonical_pull_request_url=f"https://github.com/public-owner/repo-{number}/pull/1",
        acquisition_candidate_sha256=_identity(f"acquisition-{number}"),
        source_facts_sha256=_identity(f"facts-{number}"),
        repository_receipt_sha256=_identity(f"repo-receipt-{number}"),
        pull_request_receipt_sha256=_identity(f"pr-receipt-{number}"),
        source_receipt_sha256=_identity(f"source-receipt-{number}"),
        stack_receipt_sha256=_identity(f"stack-receipt-{number}"),
        source_record_sha256=_identity(f"record-{number}"),
    )
    return MODULE.Candidate(
        record=record,
        lineage_id=_identity(f"lineage-{number}"),
        family=family,
        score_sha256=f"{score:064x}",
    )


def _source_evidence_fixture() -> dict[str, Any]:
    record_id = "record-source-0001"
    pull_url = "https://github.com/public-owner/repo-1/pull/1"
    query_id = "server-action-auth-action-mutation"
    repository = _receipt(
        {
            "schema_version": 2,
            "record_id": record_id,
            "canonical_repository_url": "https://github.com/public-owner/repo-1",
            "repository_node_id": "repository-node-1",
            "owner_node_id": "owner-node-1",
            "owner_login": "public-owner",
            "visibility": "public",
            "private": False,
            "fork_network_root_sha256": _identity("fork-root"),
            "template_root_sha256": None,
            "fork": False,
            "mirror_url": None,
            "archived": False,
            "disabled": False,
            "default_branch": "main",
            "license_spdx": "MIT",
            "license_file_sha256": _identity("license"),
            "api_response_sha256": _identity("repository-api-response"),
            "etag": '"repository-etag"',
            "fetched_at_utc": "2026-01-17T23:59:00Z",
        }
    )
    candidate_identity = MODULE.stable_digest(
        {
            "pull_request_node_id": "pull-node-1",
            "canonical_pull_request_url": pull_url,
            "repository_node_id": "repository-node-1",
            "owner_node_id": "owner-node-1",
            "merged_at_utc": "2025-06-01T00:00:00Z",
        }
    )

    def acquisition_observation(index: int, hour: str) -> dict[str, Any]:
        requested_url = (
            "https://api.github.com/search/issues?q=server-action+is%3Apr+"
            "is%3Amerged+is%3Apublic"
        )
        return {
            "lane_id": MODULE.QUERY_PLAN_ID,
            "credential_profile_id": "github-public-metadata-readonly-v2",
            "credential_env_name": "GITHUB_TOKEN",
            "credential_present_without_value": True,
            "collector_source_commit": "7" * 40,
            "collector_source_tree": "8" * 40,
            "collector_sha256": _identity("collector-code"),
            "compiler_sha256": _identity("compiler-code"),
            "requested_url": requested_url,
            "final_url": requested_url,
            "redirect_chain": [requested_url],
            "request_started_at_utc": f"2026-01-18T{hour}:00:00Z",
            "request_completed_at_utc": f"2026-01-18T{hour}:00:10Z",
            "github_api_version": "2022-11-28",
            "response_date": f"2026-01-18T{hour}:00:05Z",
            "x_github_request_id": f"request-{index}",
            "x_ratelimit_limit": 30,
            "x_ratelimit_remaining": 29,
            "x_ratelimit_reset": 1_768_694_400 + index,
            "x_ratelimit_resource": "search",
            "x_ratelimit_used": 1,
            "request_url_sha256": hashlib.sha256(requested_url.encode()).hexdigest(),
            "headers_sha256": _identity(f"headers-{index}"),
            "body_sha256": _identity(f"acquisition-response-{index}"),
            "repository_public_visibility_receipt_sha256": (
                MODULE._repository_public_visibility_digest(repository)  # noqa: SLF001
            ),
            "terminal_shard_manifest_sha256": _identity("terminal-shards"),
            "ordered_identity_sha256": _identity("ordered-identities"),
            "candidate_identity_sha256": candidate_identity,
            "checkpoint_sha256": _identity(f"checkpoint-{index}"),
        }

    acquisition = {
        "schema_version": 2,
        "record_id": record_id,
        "query_ids": [query_id],
        "pull_request_node_id": "pull-node-1",
        "canonical_pull_request_url": pull_url,
        "repository_node_id": "repository-node-1",
        "owner_node_id": "owner-node-1",
        "merged_at_utc": "2025-06-01T00:00:00Z",
        "acquisition_one": acquisition_observation(1, "00"),
        "acquisition_two": acquisition_observation(2, "01"),
    }
    acquisition["acquisition_row_sha256"] = MODULE.stable_digest(acquisition)
    pull_request = _receipt(
        {
            "schema_version": 2,
            "record_id": record_id,
            "pull_request_url": pull_url,
            "pull_request_node_id": "pull-node-1",
            "pull_request_number": 1,
            "merged_at_utc": "2025-06-01T00:00:00Z",
            "base_repository_node_id": "repository-node-1",
            "base_commit_sha": "1" * 40,
            "base_tree_sha": "2" * 40,
            "head_repository_node_id": "head-repository-node-1",
            "head_commit_sha": "3" * 40,
            "head_tree_sha": "4" * 40,
            "merge_commit_sha": "5" * 40,
            "gold_tree_sha": "6" * 40,
            "api_response_sha256": _identity("pull-api-response"),
            "diff_sha256": _identity("diff"),
            "patch_sha256": _identity("patch"),
            "changed_files_manifest_sha256": _identity("changed-files"),
            "review_manifest_sha256": _identity("reviews"),
            "linked_issue_manifest_sha256": _identity("linked-issues"),
        }
    )
    source = _receipt(
        {
            "schema_version": 2,
            "record_id": record_id,
            "base_archive_sha256": _identity("base-archive"),
            "gold_archive_sha256": _identity("gold-archive"),
            "base_tree_listing_sha256": _identity("base-tree-listing"),
            "gold_tree_listing_sha256": _identity("gold-tree-listing"),
            "archive_file_manifest_sha256": _identity("file-manifest"),
            "source_fingerprint_sha256": _identity("source-fingerprint"),
            "submodule_check": "passed",
            "unsafe_link_check": "passed",
            "path_traversal_check": "passed",
            "secret_path_check": "passed",
            "generated_dependency_tree_check": "passed",
            "source_size_bytes": 1024,
        }
    )
    stack = _receipt(
        {
            "schema_version": 2,
            "record_id": record_id,
            "package_boundary_path": "app",
            "package_manifest_path": "app/package.json",
            "package_manifest_sha256": _identity("package-manifest"),
            "dependency_lock_path": "pnpm-lock.yaml",
            "dependency_lock_sha256": _identity("dependency-lock"),
            "lock_format": "pnpm-v9",
            "react_declared_requirement": "^19.0.0",
            "react_resolved_version": "19.0.0",
            "next_declared_requirement": "^15.0.0",
            "next_resolved_version": "15.0.0",
            "typescript_target_paths": ["app/action.ts"],
            "target_file_manifest_sha256": _identity("target-files"),
            "parser_runtime_sha256": _identity("parser-runtime"),
        }
    )
    source_facts = {
        "acquisition_candidate_sha256": acquisition["acquisition_row_sha256"],
        "repository_receipt_sha256": repository["receipt_sha256"],
        "pull_request_receipt_sha256": pull_request["receipt_sha256"],
        "source_receipt_sha256": source["receipt_sha256"],
        "stack_receipt_sha256": stack["receipt_sha256"],
    }
    population: dict[str, Any] = {
        "schema_version": 2,
        "record_id": record_id,
        "repository_id": "repository-node-1",
        "owner_id": "owner-node-1",
        "canonical_pull_request_url": pull_url,
        "acquisition_candidate_sha256": acquisition["acquisition_row_sha256"],
        "pre_treatment": True,
        "source_facts_sha256": MODULE.stable_digest(source_facts),
        **{key: value for key, value in source_facts.items() if key != "acquisition_candidate_sha256"},
    }
    population["source_record_sha256"] = MODULE.stable_digest(population)
    roots = sorted(
        [
            repository["fork_network_root_sha256"],
            source["source_fingerprint_sha256"],
            pull_request["patch_sha256"],
        ]
    )
    lineage = _lineage_row(population, roots=roots)
    family_queries = {family: {f"query-{family}"} for family in MODULE.FAMILIES}
    family_queries["server_action_authorization"] = {query_id}
    return {
        "population": population,
        "lineage": lineage,
        "acquisition": acquisition,
        "repository": repository,
        "pull_request": pull_request,
        "source": source,
        "stack": stack,
        "family_queries": family_queries,
    }
def test_current_protocol_blocks_before_randomization_or_assignment() -> None:
    frame = _frame()
    started = time.perf_counter()
    result = MODULE.select_vercel_population(frame["artifacts"])
    elapsed = time.perf_counter() - started

    assert elapsed < 12
    assert result["status"] == "blocked_authenticated_evidence_required"
    assert result["assignment_emitted"] is False
    assert result["blockers"] == [
        "no_randomness_or_assignment_was_computed",
        "protocol_trust_roots_not_registered",
    ]
    for forbidden in ("selected", "seed_receipt", "optimization", "replacement"):
        assert forbidden not in result
    unsigned = dict(result)
    digest = unsigned.pop("receipt_sha256")
    assert digest == MODULE.stable_digest(unsigned)


def test_selection_is_deterministic_on_the_same_frozen_bytes() -> None:
    frame = _frame()
    assert MODULE.select_vercel_population(
        frame["artifacts"]
    ) == MODULE.select_vercel_population(frame["artifacts"])


def test_cross_owner_lineage_is_canonical_but_repository_owner_drift_fails() -> None:
    first = _population_record(1, owner_id="owner-one")
    second = _population_record(2, owner_id="owner-two")
    roots = [_identity("shared-fork-root"), _identity("shared-source-fingerprint")]
    lineage_rows = [_lineage_row(first, roots=roots), _lineage_row(second, roots=roots)]
    normalized = MODULE._validate_lineage_rows(  # noqa: SLF001
        lineage_rows,
        MODULE._validate_population_rows([first, second]),  # noqa: SLF001
    )
    assert normalized[first["record_id"]][0] == normalized[second["record_id"]][0]

    duplicate_repo = _population_record(
        2,
        owner_id="owner-two",
        repository_id=first["repository_id"],
    )
    with pytest.raises(MODULE.VercelPopulationSelectionError, match="multiple owner"):
        MODULE._validate_population_rows([first, duplicate_repo])  # noqa: SLF001


def test_lineage_ids_must_be_digest_of_sorted_roots_and_repo_is_single_lineage() -> (
    None
):
    row = _population_record(1, owner_id="owner-one")
    lineage = _lineage_row(row)
    lineage["lineage_id"] = _identity("fabricated-lineage")
    unsigned = dict(lineage)
    unsigned.pop("lineage_row_sha256")
    lineage["lineage_row_sha256"] = MODULE.stable_digest(unsigned)
    with pytest.raises(
        MODULE.VercelPopulationSelectionError, match="canonical roots digest"
    ):
        MODULE._validate_lineage_rows(  # noqa: SLF001
            [lineage],
            MODULE._validate_population_rows([row]),  # noqa: SLF001
        )


def test_actual_artifact_bytes_are_recomputed_and_duplicate_keys_rejected() -> None:
    frame = _frame()
    artifacts = frame["artifacts"]
    tampered = copy.copy(artifacts)
    object.__setattr__(tampered, "selection_code", artifacts.selection_code + b"\n")
    with pytest.raises(MODULE.VercelPopulationSelectionError, match="executing selector"):
        MODULE.select_vercel_population(tampered)

    duplicate_plan = artifacts.plan.replace(
        b'"schema_version":2', b'"schema_version":2,"schema_version":2', 1
    )
    duplicate = copy.copy(artifacts)
    object.__setattr__(duplicate, "plan", duplicate_plan)
    with pytest.raises(
        MODULE.VercelPopulationSelectionError, match="duplicate JSON key"
    ):
        MODULE.select_vercel_population(duplicate)


def test_reviews_require_calibration_roles_blinding_and_precommit_timing() -> None:
    frame = _frame()
    result = MODULE.select_vercel_population(frame["artifacts"])
    assert result["assignment_emitted"] is False

    bad_roles = copy.deepcopy(frame["roles"])
    bad_roles["public_test_author_id_hashes"] = [REVIEWER_A]
    unsigned = dict(bad_roles)
    unsigned.pop("receipt_sha256")
    bad_roles["receipt_sha256"] = MODULE.stable_digest(unsigned)
    with pytest.raises(MODULE.VercelPopulationSelectionError, match="not disjoint"):
        MODULE._validate_role_separation_receipt(bad_roles)  # noqa: SLF001

    decision = copy.deepcopy(frame["reviews"][0]["reviewer_a"])
    decision["selection_seed_and_score_blinded"] = False
    unsigned_decision = dict(decision)
    unsigned_decision.pop("decision_sha256")
    decision["decision_sha256"] = MODULE.stable_digest(unsigned_decision)
    with pytest.raises(MODULE.VercelPopulationSelectionError, match="must be true"):
        MODULE._reviewer_decision(  # noqa: SLF001
            decision,
            field="reviewer",
            record_id=frame["population"][0]["record_id"],
            source_record_sha256=frame["population"][0]["source_record_sha256"],
            lineage_row_sha256=frame["lineages"][0]["lineage_row_sha256"],
            expected_lineage_id=frame["lineages"][0]["lineage_id"],
            calibration_receipt_sha256=hashlib.sha256(
                frame["artifacts"].calibration_receipt
            ).hexdigest(),
            role_separation_receipt_sha256=hashlib.sha256(
                frame["artifacts"].role_separation_receipt
            ).hexdigest(),
            calibrated_at=MODULE._utc(  # noqa: SLF001
                frame["calibration"]["completed_at_utc"], "calibrated"
            ),
            committed_at=MODULE._utc(  # noqa: SLF001
                frame["precommit"]["committed_at_utc"], "committed"
            ),
            registered_role_ids=MODULE._validate_role_separation_receipt(  # noqa: SLF001
                frame["roles"]
            ),
        )


def test_public_precommit_and_locked_trust_root_fail_closed() -> None:
    frame = _frame()
    bad_precommit = copy.deepcopy(frame["precommit"])
    bad_precommit["public_commit_url"] = "https://example.com/not-public"
    unsigned = dict(bad_precommit)
    unsigned.pop("receipt_sha256")
    bad_precommit["receipt_sha256"] = MODULE.stable_digest(unsigned)
    with pytest.raises(MODULE.VercelPopulationSelectionError, match="URL"):
        MODULE._validate_precommit_receipt(  # noqa: SLF001
            bad_precommit,
            plan=frame["plan"],
            source_key_id=_identity("source-publication-key"),
        )

    bad_lock = copy.deepcopy(frame["verifier"])
    bad_lock["trust_root_kind"] = "operator_chosen_key"
    unsigned_lock = dict(bad_lock)
    unsigned_lock.pop("receipt_sha256")
    bad_lock["receipt_sha256"] = MODULE.stable_digest(unsigned_lock)
    with pytest.raises(MODULE.VercelPopulationSelectionError, match="trust-root"):
        MODULE._validate_verifier_lock(  # noqa: SLF001
            bad_lock,
            plan=frame["plan"],
            committed_at=MODULE._utc(  # noqa: SLF001
                frame["precommit"]["committed_at_utc"], "committed"
            ),
        )


def test_nist_64_byte_output_is_supported_and_bound_to_canonical_endpoint() -> None:
    frame = _frame(provider="nist-beacon-v2")
    result = MODULE.select_vercel_population(frame["artifacts"])
    assert result["assignment_emitted"] is False

    plan = copy.deepcopy(frame["plan"])
    plan["beacon_target"]["canonical_endpoint"] = "https://example.com/1767398400000"
    with pytest.raises(
        MODULE.VercelPopulationSelectionError, match="canonical provider"
    ):
        MODULE.validate_plan(plan)


def test_exact_order_bounds_and_graph_limits_fail_closed() -> None:
    frame = _frame()
    bad = copy.deepcopy(frame["plan"])
    bad["partition_order"] = list(reversed(MODULE.PARTITION_ORDER))
    with pytest.raises(MODULE.VercelPopulationSelectionError, match="partition order"):
        MODULE.validate_plan(bad)

    oversized = copy.copy(frame["artifacts"])
    object.__setattr__(
        oversized,
        "plan",
        b"{" + b" " * MODULE.EXPECTED_BOUNDS["max_plan_bytes"] + b"}",
    )
    with pytest.raises(MODULE.VercelPopulationSelectionError, match="1.."):
        MODULE.select_vercel_population(oversized)

    family = MODULE.PRIMARY_FAMILIES[0]
    with pytest.raises(MODULE.VercelPopulationSelectionError, match="residual graph"):
        MODULE._solve_constrained(  # noqa: SLF001
            [_candidate(1, owner="one", family=family, score=1)],
            {f"development/{family}": 1, f"target_holdout/{family}": 1},
            max_search_states=10,
            max_graph_edges=1,
        )


def test_exact_solver_preserves_optimum_while_enforcing_owner_caps() -> None:
    server = MODULE.PRIMARY_FAMILIES[0]
    dom = MODULE.CONTROL_FAMILIES[0]
    candidates = [
        *[
            _candidate(index, owner="shared-owner", family=server, score=index)
            for index in range(1, 6)
        ],
        _candidate(6, owner="fallback-owner", family=server, score=100),
        _candidate(7, owner="control-owner", family=dom, score=1),
    ]
    quotas = {
        f"development/{server}": 2,
        f"target_holdout/{server}": 3,
        f"safety_control/{dom}": 1,
    }
    assignments, _objective, _states = MODULE._solve_constrained(  # noqa: SLF001
        candidates,
        quotas,
        max_search_states=1_000,
    )
    shared = [
        assignment
        for assignment in assignments
        if assignment.candidate.record.owner_id == "shared-owner"
    ]
    assert {assignment.candidate.record.record_id for assignment in shared} == {
        "record-1",
        "record-2",
        "record-3",
        "record-4",
    }
    assert any(
        assignment.candidate.record.owner_id == "fallback-owner"
        for assignment in assignments
    )


def test_family_minimum_and_calibration_quality_fail_before_selection() -> None:
    frame = _frame(family_counts={"server_action_authorization": 139})
    assert MODULE.select_vercel_population(frame["artifacts"])[
        "assignment_emitted"
    ] is False
    candidates = [
        _candidate(index, owner=f"owner-{index}", family=family, score=index)
        for family in MODULE.FAMILIES
        for index in range(
            1,
            (139 if family == "server_action_authorization" else 140) + 1,
        )
    ]
    with pytest.raises(MODULE.VercelPopulationSelectionError, match="minimum is 140"):
        MODULE._validate_frame_minimums(candidates)  # noqa: SLF001

    bad = copy.deepcopy(frame["calibration"])
    bad["cohens_kappa_micros"] = 799_999
    unsigned = dict(bad)
    unsigned.pop("receipt_sha256")
    bad["receipt_sha256"] = MODULE.stable_digest(unsigned)
    roles = MODULE._validate_role_separation_receipt(frame["roles"])  # noqa: SLF001
    with pytest.raises(MODULE.VercelPopulationSelectionError, match="below 0.8"):
        MODULE._validate_calibration_receipt(bad, role_ids=roles)  # noqa: SLF001


def test_forged_self_asserted_randomness_never_creates_an_assignment() -> None:
    frame = _frame()
    original = MODULE.select_vercel_population(frame["artifacts"])
    forged_value = copy.deepcopy(frame["randomness"])
    forged_value["random_value"] = "f" * 64
    unsigned = dict(forged_value)
    unsigned.pop("receipt_sha256")
    forged_value["receipt_sha256"] = MODULE.stable_digest(unsigned)
    forged = copy.copy(frame["artifacts"])
    object.__setattr__(forged, "randomness_receipt", _json_bytes(forged_value))

    result = MODULE.select_vercel_population(forged)
    assert result == original
    assert result["assignment_emitted"] is False
    assert "master_seed_sha256" not in json.dumps(result)


def test_query_amendment_cannot_self_attest_as_the_locked_protocol() -> None:
    frame = _frame()
    query = json.loads(frame["artifacts"].query_plan)
    query["treatment_preperiod"]["record_cutoff_utc"] = "2035-01-17T23:59:59Z"
    query["treatment_preperiod"]["github_merged_qualifier"] = (
        "merged:2023-01-01..2035-01-17"
    )
    unsigned = dict(query)
    unsigned.pop("artifact_digest")
    query["artifact_digest"] = MODULE.stable_digest(unsigned)
    amended = copy.copy(frame["artifacts"])
    object.__setattr__(amended, "query_plan", _json_bytes(query))
    with pytest.raises(MODULE.VercelPopulationSelectionError, match="exact Vercel V2"):
        MODULE.select_vercel_population(amended)


def test_detached_ed25519_authentication_is_publicly_verifiable() -> None:
    cryptography = pytest.importorskip("cryptography")
    del cryptography
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private_key = Ed25519PrivateKey.generate()
    payload = {"schema_version": 2, "artifact_sha256": _identity("artifact")}
    receipt = _detached_authentication(
        private_key, purpose="source_verification", payload=payload
    )
    MODULE._validate_authentication(  # noqa: SLF001
        receipt,
        purpose="source_verification",
        payload=payload,
        expected_key_id=receipt["key_id_sha256"],
    )

    forged = copy.deepcopy(receipt)
    forged["signature_hex"] = "00" * 64
    unsigned = dict(forged)
    unsigned.pop("receipt_sha256")
    forged["receipt_sha256"] = MODULE.stable_digest(unsigned)
    with pytest.raises(MODULE.VercelPopulationSelectionError, match="signature is invalid"):
        MODULE._validate_authentication(  # noqa: SLF001
            forged,
            purpose="source_verification",
            payload=payload,
            expected_key_id=receipt["key_id_sha256"],
        )

    retimed = copy.deepcopy(receipt)
    retimed["authenticated_at_utc"] = "2025-12-31T23:59:59Z"
    unsigned = dict(retimed)
    unsigned.pop("receipt_sha256")
    retimed["receipt_sha256"] = MODULE.stable_digest(unsigned)
    with pytest.raises(MODULE.VercelPopulationSelectionError, match="signature is invalid"):
        MODULE._validate_authentication(  # noqa: SLF001
            retimed,
            purpose="source_verification",
            payload=payload,
            expected_key_id=receipt["key_id_sha256"],
        )


def test_ed25519_runtime_unavailable_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    cryptography = pytest.importorskip("cryptography")
    del cryptography
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private_key = Ed25519PrivateKey.generate()
    payload = {"schema_version": 2, "artifact_sha256": _identity("artifact")}
    receipt = _detached_authentication(
        private_key,
        purpose="source_verification",
        payload=payload,
    )
    monkeypatch.setattr(MODULE, "Ed25519PublicKey", None)
    with pytest.raises(
        MODULE.VercelPopulationSelectionError,
        match="Ed25519 verification runtime is unavailable",
    ):
        MODULE._validate_authentication(  # noqa: SLF001
            receipt,
            purpose="source_verification",
            payload=payload,
            expected_key_id=receipt["key_id_sha256"],
        )


def test_public_precommit_requires_pinned_source_verifier_signature() -> None:
    cryptography = pytest.importorskip("cryptography")
    del cryptography
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    frame = _frame()
    private_key = Ed25519PrivateKey.generate()
    receipt = copy.deepcopy(frame["precommit"])
    receipt["publication_authentication"] = _detached_authentication(
        private_key,
        purpose="precommit_publication_verification",
        payload=MODULE._precommit_publication_payload(receipt),  # noqa: SLF001
        authenticated_at_utc=receipt["publication_observed_at_utc"],
    )
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256")
    receipt["receipt_sha256"] = MODULE.stable_digest(unsigned)
    normalized, observed = MODULE._validate_precommit_receipt(  # noqa: SLF001
        receipt,
        plan=frame["plan"],
        source_key_id=receipt["publication_authentication"]["key_id_sha256"],
    )
    assert normalized == receipt
    assert observed == MODULE._utc(  # noqa: SLF001
        receipt["publication_observed_at_utc"], "observed"
    )

    self_asserted = copy.deepcopy(receipt)
    self_asserted["publication_authentication"] = {}
    unsigned = dict(self_asserted)
    unsigned.pop("receipt_sha256")
    self_asserted["receipt_sha256"] = MODULE.stable_digest(unsigned)
    with pytest.raises(MODULE.VercelPopulationSelectionError):
        MODULE._validate_precommit_receipt(  # noqa: SLF001
            self_asserted,
            plan=frame["plan"],
            source_key_id=receipt["publication_authentication"]["key_id_sha256"],
        )


def test_calibration_is_balanced_recomputed_and_rejects_boolean_counts() -> None:
    roles_receipt = _role_separation()
    role_ids = MODULE._validate_role_separation_receipt(roles_receipt)  # noqa: SLF001
    rows: list[dict[str, Any]] = []
    for family in MODULE.FAMILIES:
        for index in range(8):
            acceptable = index < 4
            row = {
                "schema_version": 2,
                "calibration_record_id": f"{family}-{index}",
                "family": family,
                "expected_acceptable": acceptable,
                "reviewer_decisions": {
                    identity: acceptable
                    for identity in sorted(
                        role_ids["eligibility_reviewer_id_hashes"]
                        | role_ids["adjudicator_id_hashes"]
                    )
                },
            }
            row["row_sha256"] = MODULE.stable_digest(row)
            rows.append(row)
    calibration = _calibration(roles_receipt)
    calibration["cohens_kappa_micros"] = 1_000_000
    unsigned = dict(calibration)
    unsigned.pop("receipt_sha256")
    calibration["receipt_sha256"] = MODULE.stable_digest(unsigned)
    assert MODULE._validate_calibration_ledger(  # noqa: SLF001
        rows,
        calibration_receipt=calibration,
        role_ids=role_ids,
    ) == MODULE.stable_digest(rows)

    calibration["critical_false_inclusions"] = False
    unsigned = dict(calibration)
    unsigned.pop("receipt_sha256")
    calibration["receipt_sha256"] = MODULE.stable_digest(unsigned)
    with pytest.raises(MODULE.VercelPopulationSelectionError, match="false inclusion"):
        MODULE._validate_calibration_ledger(  # noqa: SLF001
            rows,
            calibration_receipt=calibration,
            role_ids=role_ids,
        )


def test_nonfinite_json_and_symlinked_io_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(MODULE.VercelPopulationSelectionError, match="non-finite"):
        MODULE._json_bytes(b'{"value":NaN}', "artifact", 100)  # noqa: SLF001

    real_input = tmp_path / "input.json"
    real_input.write_text("{}", encoding="utf-8")
    linked_input = tmp_path / "linked-input.json"
    linked_input.symlink_to(real_input)
    with pytest.raises(MODULE.VercelPopulationSelectionError, match="cannot read"):
        MODULE._read_regular_file(linked_input, "linked input")  # noqa: SLF001

    ancestor_target = tmp_path / "ancestor-target"
    ancestor_target.mkdir(mode=0o700)
    private_child = ancestor_target / "private-child"
    private_child.mkdir(mode=0o700)
    ancestor_input = private_child / "input.json"
    ancestor_input.write_text("{}", encoding="utf-8")
    ancestor_alias = tmp_path / "ancestor-alias"
    ancestor_alias.symlink_to(ancestor_target, target_is_directory=True)
    with pytest.raises(MODULE.VercelPopulationSelectionError, match="no-follow parent"):
        MODULE._read_regular_file(  # noqa: SLF001
            ancestor_alias / "private-child" / "input.json",
            "ancestor-linked input",
        )
    with pytest.raises(MODULE.VercelPopulationSelectionError, match="no-follow parent"):
        MODULE._write_private_json(  # noqa: SLF001
            ancestor_alias / "private-child" / "output.json",
            {"status": "blocked"},
        )

    output = tmp_path / "private" / "selection.json"
    MODULE._write_private_json(output, {"status": "blocked"})  # noqa: SLF001
    assert output.stat().st_mode & 0o777 == 0o600
    assert output.parent.stat().st_mode & 0o777 == 0o700
    with pytest.raises(MODULE.VercelPopulationSelectionError, match="new non-symlink"):
        MODULE._write_private_json(output, {"status": "selected"})  # noqa: SLF001


def test_pending_owner_search_has_a_locked_memory_bound() -> None:
    family = MODULE.PRIMARY_FAMILIES[0]
    candidates = [
        *[
            _candidate(index, owner="shared-owner", family=family, score=index)
            for index in range(1, 6)
        ],
        _candidate(6, owner="fallback-owner", family=family, score=100),
    ]
    quotas = {
        f"development/{family}": 2,
        f"target_holdout/{family}": 3,
    }
    with pytest.raises(MODULE.VercelPopulationSelectionError, match="pending-state"):
        MODULE._solve_constrained(  # noqa: SLF001
            candidates,
            quotas,
            max_search_states=1_000,
            max_pending_search_states=0,
        )


def test_source_provenance_uses_authoritative_cross_receipt_identities() -> None:
    fixture = _source_evidence_fixture()
    records = MODULE._validate_population_rows([fixture["population"]])  # noqa: SLF001
    normalized, queries = MODULE._validate_source_evidence(  # noqa: SLF001
        records=records,
        lineage_rows=[fixture["lineage"]],
        acquisition_rows=[fixture["acquisition"]],
        repository_rows=[fixture["repository"]],
        pull_request_rows=[fixture["pull_request"]],
        source_rows=[fixture["source"]],
        stack_rows=[fixture["stack"]],
        family_queries=fixture["family_queries"],
        population_frozen_at=MODULE._utc(  # noqa: SLF001
            "2026-01-18T02:00:00Z", "population frozen"
        ),
    )
    assert normalized[fixture["population"]["record_id"]][0] == fixture["lineage"][
        "lineage_id"
    ]
    assert queries[fixture["population"]["record_id"]] == {
        "server-action-auth-action-mutation"
    }

    mismatched_repository = copy.deepcopy(fixture["repository"])
    mismatched_repository["owner_node_id"] = "invented-owner-cap-bucket"
    unsigned = dict(mismatched_repository)
    unsigned.pop("receipt_sha256")
    mismatched_repository["receipt_sha256"] = MODULE.stable_digest(unsigned)
    with pytest.raises(MODULE.VercelPopulationSelectionError, match="owner_node_id differs"):
        MODULE._validate_source_evidence(  # noqa: SLF001
            records=records,
            lineage_rows=[fixture["lineage"]],
            acquisition_rows=[fixture["acquisition"]],
            repository_rows=[mismatched_repository],
            pull_request_rows=[fixture["pull_request"]],
            source_rows=[fixture["source"]],
            stack_rows=[fixture["stack"]],
            family_queries=fixture["family_queries"],
            population_frozen_at=MODULE._utc(  # noqa: SLF001
                "2026-01-18T02:00:00Z", "population frozen"
            ),
        )


def test_power_gate_is_recomputed_and_signed_by_the_pinned_key() -> None:
    cryptography = pytest.importorskip("cryptography")
    del cryptography
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private_key = Ed25519PrivateKey.generate()
    candidate_probabilities = [200_000, 250_000, 300_000, 350_000]
    baseline_probabilities = [20_000, 50_000, 100_000, 150_000]
    scenarios = [
        {
            "candidate_only_probability_micros": candidate,
            "baseline_only_probability_micros": baseline,
            "power_micros": 950_000,
        }
        for candidate in candidate_probabilities
        for baseline in baseline_probabilities
        if candidate - baseline >= 150_000
    ]
    receipt: dict[str, Any] = {
        "schema_version": 2,
        "analysis_code_sha256": _identity("power-code"),
        "runtime_sha256": _identity("power-runtime"),
        "simulation_seed": "power-seed-v2",
        "simulation_repetitions": 100_000,
        "assumption_grid": {
            "candidate_only_probability_micros": candidate_probabilities,
            "baseline_only_probability_micros": baseline_probabilities,
            "minimum_difference_micros": 150_000,
            "holm_familywise_alpha_micros": 50_000,
        },
        "family_sample_sizes": {
            **{family: 96 for family in MODULE.PRIMARY_FAMILIES},
            **{family: 16 for family in MODULE.CONTROL_FAMILIES},
        },
        "scenario_power_results": scenarios,
        "minimum_observed_power_micros": 950_000,
        "supported_effect_boundary_micros": 150_000,
        "safety_control_power_results": {
            family: {
                "margin_micros": -50_000,
                "power_micros": 950_000,
                "critical_regressions_allowed": 0,
            }
            for family in MODULE.CONTROL_FAMILIES
        },
    }
    receipt["authentication"] = _detached_authentication(
        private_key, purpose="power_verification", payload=receipt
    )
    receipt["receipt_sha256"] = MODULE.stable_digest(receipt)
    MODULE._validate_power_receipt(  # noqa: SLF001
        receipt,
        power_key_id=receipt["authentication"]["key_id_sha256"],
    )

    too_high = copy.deepcopy(receipt)
    too_high["safety_control_power_results"][MODULE.CONTROL_FAMILIES[0]][
        "power_micros"
    ] = 1_000_001
    unsigned = dict(too_high)
    unsigned.pop("receipt_sha256")
    too_high["receipt_sha256"] = MODULE.stable_digest(unsigned)
    with pytest.raises(MODULE.VercelPopulationSelectionError, match="below 0.9"):
        MODULE._validate_power_receipt(  # noqa: SLF001
            too_high,
            power_key_id=receipt["authentication"]["key_id_sha256"],
        )


def test_every_registered_reviewer_signs_a_public_key_bound_manifest() -> None:
    cryptography = pytest.importorskip("cryptography")
    del cryptography
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    reviewer_keys = [Ed25519PrivateKey.generate() for _ in range(3)]
    reviewer_receipts = [
        _detached_authentication(
            key,
            purpose="unused-key-discovery",
            payload={"schema_version": 2},
        )
        for key in reviewer_keys
    ]
    reviewer_ids = [receipt["key_id_sha256"] for receipt in reviewer_receipts]
    role_ids = {
        "eligibility_reviewer_id_hashes": set(reviewer_ids[:2]),
        "adjudicator_id_hashes": {reviewer_ids[2]},
    }
    review_ledger_sha256 = _identity("empty-review-ledger")
    calibration_ledger_sha256 = _identity("calibration-ledger")
    role_separation_sha256 = _identity("role-separation")
    signatures: list[dict[str, Any]] = []
    for index, (private_key, reviewer_id) in enumerate(
        zip(reviewer_keys, reviewer_ids, strict=True)
    ):
        role = "eligibility_reviewer" if index < 2 else "eligibility_adjudicator"
        payload = {
            "schema_version": 2,
            "reviewer_id_hash": reviewer_id,
            "role": role,
            "decision_sha256s": [],
            "review_ledger_sha256": review_ledger_sha256,
            "calibration_ledger_sha256": calibration_ledger_sha256,
            "role_separation_receipt_sha256": role_separation_sha256,
        }
        signature_receipt = _detached_authentication(
            private_key, purpose="reviewer_manifest", payload=payload
        )
        signatures.append(
            {
                "reviewer_id_hash": reviewer_id,
                "role": role,
                "public_key_hex": signature_receipt["public_key_hex"],
                "key_id_sha256": reviewer_id,
                "decision_manifest_sha256": MODULE.stable_digest(payload),
                "signature_hex": private_key.sign(
                    MODULE._authentication_message(  # noqa: SLF001
                        "reviewer_manifest", payload
                    )
                ).hex(),
            }
        )
    governance_key = Ed25519PrivateKey.generate()
    governance_payload = {
        "schema_version": 2,
        "review_ledger_sha256": review_ledger_sha256,
        "calibration_ledger_sha256": calibration_ledger_sha256,
        "role_separation_receipt_sha256": role_separation_sha256,
        "reviewer_registry": [
            {
                "reviewer_id_hash": signature["reviewer_id_hash"],
                "role": signature["role"],
                "public_key_hex": signature["public_key_hex"],
                "key_id_sha256": signature["key_id_sha256"],
            }
            for signature in sorted(signatures, key=lambda item: item["reviewer_id_hash"])
        ],
    }
    governance = _detached_authentication(
        governance_key, purpose="review_governance", payload=governance_payload
    )
    receipt = {
        "schema_version": 2,
        "purpose": "review_authentication",
        "algorithm": MODULE.AUTHENTICATION_ALGORITHM,
        "reviewer_signatures": signatures,
        "governance_authentication": governance,
    }
    receipt["receipt_sha256"] = MODULE.stable_digest(receipt)
    MODULE._validate_review_authentication(  # noqa: SLF001
        receipt,
        review_rows=[],
        review_ledger_sha256=review_ledger_sha256,
        calibration_ledger_sha256=calibration_ledger_sha256,
        role_separation_receipt_sha256=role_separation_sha256,
        role_ids=role_ids,
        governance_key_id=governance["key_id_sha256"],
    )

    forged = copy.deepcopy(receipt)
    forged["reviewer_signatures"][0]["reviewer_id_hash"] = reviewer_ids[1]
    unsigned = dict(forged)
    unsigned.pop("receipt_sha256")
    forged["receipt_sha256"] = MODULE.stable_digest(unsigned)
    with pytest.raises(MODULE.VercelPopulationSelectionError):
        MODULE._validate_review_authentication(  # noqa: SLF001
            forged,
            review_rows=[],
            review_ledger_sha256=review_ledger_sha256,
            calibration_ledger_sha256=calibration_ledger_sha256,
            role_separation_receipt_sha256=role_separation_sha256,
            role_ids=role_ids,
            governance_key_id=governance["key_id_sha256"],
        )
