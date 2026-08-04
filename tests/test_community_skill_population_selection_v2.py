from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "examples"
    / "comparisons"
    / "community-skill-upgrades"
    / "select_population_v2.py"
)
SUPERPOWERS_PROTOCOL = (
    ROOT
    / "examples"
    / "comparisons"
    / "superpowers-writing-plans-upgrade"
    / "conference-sampling-frame-protocol-v2.json"
)
ANTHROPIC_PROTOCOL = (
    ROOT
    / "examples"
    / "comparisons"
    / "anthropic-skill-creator-upgrade"
    / "conference-sampling-frame-protocol-v2.json"
)
SUPERPOWERS_QUERY_PLAN = (
    ROOT
    / "examples"
    / "comparisons"
    / "superpowers-writing-plans-upgrade"
    / "conference-query-plan-v2.json"
)
ANTHROPIC_QUERY_PLAN = (
    ROOT
    / "examples"
    / "comparisons"
    / "anthropic-skill-creator-upgrade"
    / "conference-github-candidate-query-plan-v2.json"
)
SPEC = importlib.util.spec_from_file_location("select_population_v2", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

SHA = {
    "facts": "1" * 64,
    "lineage": "2" * 64,
    "basis": "3" * 64,
    "signature_a": "4" * 64,
    "signature_b": "5" * 64,
    "signature_c": "6" * 64,
    "review_key_a": "7" * 64,
    "review_key_b": "8" * 64,
    "review_key_c": "9" * 64,
    "code": "a" * 64,
    "runtime": "b" * 64,
    "trust": "c" * 64,
    "response": "d" * 64,
    "beacon_signature": "e" * 64,
    "randomness": "f" * 64,
    "chain": "0" * 64,
}


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(_json_bytes(row) + b"\n" for row in rows)


def _verifier(identity: str) -> dict[str, Any]:
    return {
        "identity": identity,
        "method": f"{identity}-external-verification-v1",
        "code_sha256": SHA["code"],
        "runtime_sha256": SHA["runtime"],
        "trust_root_sha256": SHA["trust"],
    }


def _registry() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "reviewers": [
            {
                "reviewer_id": "reviewer-a",
                "role": "eligibility_reviewer",
                "signing_key_sha256": SHA["review_key_a"],
            },
            {
                "reviewer_id": "reviewer-b",
                "role": "eligibility_reviewer",
                "signing_key_sha256": SHA["review_key_b"],
            },
            {
                "reviewer_id": "adjudicator-c",
                "role": "eligibility_adjudicator",
                "signing_key_sha256": SHA["review_key_c"],
            },
        ],
    }


def _attestation(
    *,
    reviewer_id: str,
    role: str,
    record_id: str,
    repository_id: str,
    stratum: str,
    lineage: str,
    cluster: str,
    temporal_relation: str,
    decision: str = "eligible",
    adjudication_reason: str | None = None,
) -> dict[str, Any]:
    key_suffix = {"reviewer-a": "a", "reviewer-b": "b", "adjudicator-c": "c"}[
        reviewer_id
    ]
    value: dict[str, Any] = {
        "reviewer_id": reviewer_id,
        "role": role,
        "signing_key_sha256": SHA[f"review_key_{key_suffix}"],
        "record_id": record_id,
        "repository_id": repository_id,
        "decision": decision,
        "source_facts_sha256": SHA["facts"],
        "lineage_receipt_sha256": SHA["lineage"],
        "stratum": stratum,
        "source_lineage_id": lineage,
        "independence_cluster_id": cluster,
        "temporal_relation": temporal_relation,
        "reviewed_at_utc": "2026-06-30T12:00:00Z",
        "reason_codes": [] if decision == "eligible" else ["review-disagreement"],
        "adjudication_reason": adjudication_reason,
        "treatment_output_blinded": True,
        "selection_rank_blinded": True,
        "signature_sha256": SHA[f"signature_{key_suffix}"],
    }
    unsigned = dict(value)
    del unsigned["signature_sha256"]
    value["attestation_sha256"] = MODULE.stable_digest(unsigned)
    return value


def _row(
    number: int,
    *,
    cohort: str,
    temporal_relation: str,
    stratum: str,
    repository_number: int | None = None,
    cluster_number: int | None = None,
    record_in_repository: int = 1,
    disagree: bool = False,
    empty_adjudication_reason: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repository_number = repository_number or number
    cluster_number = cluster_number or repository_number
    repository_id = f"R_repo_{repository_number:04d}"
    record_id = f"record-{number:04d}-{record_in_repository}"
    cluster = f"cluster-{cluster_number:04d}"
    lineage = f"lineage-{cluster_number:04d}"
    reviewer_a = _attestation(
        reviewer_id="reviewer-a",
        role="eligibility_reviewer",
        record_id=record_id,
        repository_id=repository_id,
        stratum=stratum,
        lineage=lineage,
        cluster=cluster,
        temporal_relation=temporal_relation,
    )
    reviewer_b = _attestation(
        reviewer_id="reviewer-b",
        role="eligibility_reviewer",
        record_id=record_id,
        repository_id=repository_id,
        stratum="wrong-stratum" if disagree else stratum,
        lineage=lineage,
        cluster=cluster,
        temporal_relation=temporal_relation,
        decision="uncertain" if disagree else "eligible",
    )
    adjudication = None
    if disagree:
        adjudication = _attestation(
            reviewer_id="adjudicator-c",
            role="eligibility_adjudicator",
            record_id=record_id,
            repository_id=repository_id,
            stratum=stratum,
            lineage=lineage,
            cluster=cluster,
            temporal_relation=temporal_relation,
            adjudication_reason=""
            if empty_adjudication_reason
            else "Resolved against frozen source facts.",
        )
    population: dict[str, Any] = {
        "schema_version": 2,
        "record_id": record_id,
        "repository_id": repository_id,
        "skill_path": f"skills/example-{record_in_repository}",
        "cohort_id": cohort,
        "source_facts_sha256": SHA["facts"],
        "lineage_receipt_sha256": SHA["lineage"],
        "public_basis_bundle_sha256": SHA["basis"],
    }
    population["population_record_sha256"] = MODULE.stable_digest(population)
    ledger: dict[str, Any] = {
        "schema_version": 2,
        "record_id": record_id,
        "repository_id": repository_id,
        "skill_path": population["skill_path"],
        "cohort_id": cohort,
        "independence_cluster_id": cluster,
        "source_lineage_id": lineage,
        "stratum": stratum,
        "temporal_relation": temporal_relation,
        "source_facts_sha256": SHA["facts"],
        "lineage_receipt_sha256": SHA["lineage"],
        "public_basis_bundle_sha256": SHA["basis"],
        "review": {
            "reviewer_a": reviewer_a,
            "reviewer_b": reviewer_b,
            "adjudication": adjudication,
        },
        "exclusion_reasons": [],
    }
    ledger["record_sha256"] = MODULE.stable_digest(ledger)
    return population, ledger


def _superpowers_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    population: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    number = 1
    for stratum, count in {
        "security_privacy": 168,
        "identity_migration": 168,
        "lifecycle_recovery": 168,
        "integration_runtime": 168,
        "docs_only": 48,
        "unsafe_request": 48,
        "no_change_required": 48,
        "single_surface": 48,
    }.items():
        for _ in range(count):
            source, review = _row(
                number,
                cohort="post_treatment_maintenance_primary",
                temporal_relation="post_treatment",
                stratum=stratum,
            )
            population.append(source)
            ledger.append(review)
            number += 1
    return population, ledger


def _anthropic_rows(
    *, disagree: bool = False, empty_adjudication_reason: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    population: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    number = 1
    for stratum, cluster_count in (("maintenance", 4), ("repair", 2)):
        for cluster_number in range(1, cluster_count + 1):
            absolute_cluster = (
                cluster_number if stratum == "maintenance" else 100 + cluster_number
            )
            for repository_offset in range(2):
                repository_number = absolute_cluster * 10 + repository_offset
                for record_in_repository in range(1, 3):
                    source, review = _row(
                        number,
                        cohort="maintenance_primary",
                        temporal_relation="pre_treatment",
                        stratum=stratum,
                        repository_number=repository_number,
                        cluster_number=absolute_cluster,
                        record_in_repository=record_in_repository,
                        disagree=disagree and number == 1,
                        empty_adjudication_reason=empty_adjudication_reason,
                    )
                    population.append(source)
                    ledger.append(review)
                    number += 1
    return population, ledger


def _review_bundle(
    ledger: list[dict[str, Any]], registry_bytes: bytes
) -> dict[str, Any]:
    verifications: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in ledger:
        for name in ("reviewer_a", "reviewer_b", "adjudication"):
            attestation = row["review"][name]
            if attestation is None or attestation["attestation_sha256"] in seen:
                continue
            seen.add(attestation["attestation_sha256"])
            verifications.append(
                {
                    "attestation_sha256": attestation["attestation_sha256"],
                    "reviewer_id": attestation["reviewer_id"],
                    "signing_key_sha256": attestation["signing_key_sha256"],
                    "signature_sha256": attestation["signature_sha256"],
                    "verification_passed": True,
                    "verified_at_utc": "2026-06-30T13:00:00Z",
                }
            )
    return {
        "schema_version": 2,
        "reviewer_registry_sha256": MODULE._sha256_bytes(registry_bytes),  # noqa: SLF001
        "verifier": _verifier("review-signature-verifier-v1"),
        "verifications": verifications,
    }


def _lane(lane: str, protocol_bytes: bytes) -> dict[str, Any]:
    if lane == "superpowers":
        return {
            "protocol_id": MODULE.SUPERPOWERS_PROTOCOL_ID,
            "protocol_sha256": MODULE._sha256_bytes(protocol_bytes),  # noqa: SLF001
            "cohort_id": "post_treatment_maintenance_primary",
            "cohort_role": "primary_inference",
            "temporal_relation": "post_treatment",
            "treatment": copy.deepcopy(MODULE._SUPERPOWERS_TREATMENT),  # noqa: SLF001
        }
    return {
        "protocol_id": MODULE.ANTHROPIC_PROTOCOL_ID,
        "protocol_sha256": MODULE._sha256_bytes(protocol_bytes),  # noqa: SLF001
        "cohort_id": "maintenance_primary",
        "cohort_role": "primary_inference",
        "temporal_relation": "pre_treatment",
        "treatment": copy.deepcopy(MODULE._ANTHROPIC_TREATMENT),  # noqa: SLF001
    }


def _seal_case(
    lane: str,
    *,
    population: list[dict[str, Any]] | None = None,
    ledger: list[dict[str, Any]] | None = None,
    review_bundle_mutator: Any = None,
    beacon_mutator: Any = None,
) -> dict[str, bytes]:
    protocol_bytes = (
        SUPERPOWERS_PROTOCOL.read_bytes()
        if lane == "superpowers"
        else ANTHROPIC_PROTOCOL.read_bytes()
    )
    query_plan_bytes = (
        SUPERPOWERS_QUERY_PLAN.read_bytes()
        if lane == "superpowers"
        else ANTHROPIC_QUERY_PLAN.read_bytes()
    )
    if population is None or ledger is None:
        population, ledger = (
            _superpowers_rows() if lane == "superpowers" else _anthropic_rows()
        )

    candidates: list[dict[str, Any]] = []
    candidate_by_record: dict[str, dict[str, Any]] = {}
    for source in population:
        item = {"id": source["record_id"], "repository_id": source["repository_id"]}
        item_sha256 = MODULE.stable_digest(item)
        candidate = {
            "candidate_identity": f"candidate-{source['record_id']}",
            "query_ids": ["query-1"],
            "item_sha256": item_sha256,
            "observed_item_sha256s": [item_sha256],
            "item": item,
            "privacy_review": {
                "status": "clear",
                "pattern_ids": [],
                "matched_values_serialized": False,
                "raw_source_redacted": False,
                "downstream_authoring_export_allowed": True,
            },
        }
        candidates.append(candidate)
        candidate_by_record[source["record_id"]] = candidate
    candidate_bytes = _jsonl_bytes(candidates)
    query_plan_sha256 = MODULE._sha256_bytes(query_plan_bytes)  # noqa: SLF001
    query_receipt: dict[str, Any] = {
        "schema_version": 2,
        "receipt_id": f"{lane}-frozen-query-v2",
        "lane_id": lane,
        "status": "frozen_candidate_discovery",
        "candidate_discovery_only": True,
        "eligibility_or_population_claim": False,
        "public_only_repository_visibility_verified": True,
        "plan_sha256": "a" * 64,
        "api_origin": "https://api.github.com",
        "api_version": "2022-11-28",
        "source_query_plan_sha256": query_plan_sha256,
        "temporal_scope": {},
        "selection_source_cutoff_utc": (
            "2026-07-31T23:59:59Z" if lane == "superpowers" else "2026-02-06T14:59:12Z"
        ),
        "acquisition_repetitions": 2,
        "index_stabilization_seconds": 86400,
        "repeat_separation_seconds": 3600,
        "acquisition_windows": [],
        "acquisition_receipts": [],
        "credential_profile": {},
        "collector_implementation": {},
        "public_visibility": {},
        "query_count": 1,
        "shards": [],
        "responses": [],
        "response_count": 0,
        "repository_visibility_receipts": [],
        "repository_visibility_receipt_count": 0,
        "candidate_count": len(candidates),
        "candidates_path": "candidates.jsonl",
        "candidates_sha256": MODULE._sha256_bytes(candidate_bytes),  # noqa: SLF001
        "privacy_scan": {
            "status": "clear",
            "downstream_authoring_export_blocked": False,
            "matched_values_serialized": False,
        },
        "limitations": [],
    }
    query_receipt["receipt_sha256"] = MODULE.stable_digest(query_receipt)
    query_bytes = _json_bytes(query_receipt)
    query_receipt_sha256 = MODULE._sha256_bytes(query_bytes)  # noqa: SLF001

    provenance: list[dict[str, Any]] = []
    ledger_by_id = {row["record_id"]: row for row in ledger}
    for source in population:
        review = ledger_by_id[source["record_id"]]
        record_number = int(source["record_id"].split("-")[1])
        source_facts = {
            "record_id": source["record_id"],
            "repository_id": source["repository_id"],
            "skill_path": source["skill_path"],
            "cohort_id": source["cohort_id"],
            "record_published_at_utc": (
                "2026-06-17T12:00:00Z"
                if lane == "superpowers"
                else "2026-02-05T12:00:00Z"
            ),
            "source_committed_at_utc": (
                "2026-06-16T12:00:00Z"
                if lane == "superpowers"
                else "2026-02-04T12:00:00Z"
            ),
            "query_plan_sha256": query_plan_sha256,
        }
        lineage = {
            "repository_id": source["repository_id"],
            "owner_node_id": f"O_owner_{record_number:04d}",
            "source_root_repository_id": source["repository_id"],
            "template_root_repository_id": None,
            "source_lineage_id": review["source_lineage_id"],
            "independence_cluster_id": review["independence_cluster_id"],
        }
        candidate = candidate_by_record[source["record_id"]]
        basis = {
            "record_id": source["record_id"],
            "query_candidate_identity": candidate["candidate_identity"],
            "query_receipt_sha256": query_receipt_sha256,
            "query_plan_sha256": query_plan_sha256,
            "candidate_item_sha256": candidate["item_sha256"],
        }
        source["source_facts_sha256"] = MODULE.stable_digest(source_facts)
        source["lineage_receipt_sha256"] = MODULE.stable_digest(lineage)
        source["public_basis_bundle_sha256"] = MODULE.stable_digest(basis)
        for key in (
            "source_facts_sha256",
            "lineage_receipt_sha256",
            "public_basis_bundle_sha256",
        ):
            review[key] = source[key]
        for name in ("reviewer_a", "reviewer_b", "adjudication"):
            attestation = review["review"][name]
            if attestation is None:
                continue
            attestation["source_facts_sha256"] = source["source_facts_sha256"]
            attestation["lineage_receipt_sha256"] = source["lineage_receipt_sha256"]
            unsigned_attestation = dict(attestation)
            unsigned_attestation.pop("signature_sha256")
            unsigned_attestation.pop("attestation_sha256")
            attestation["attestation_sha256"] = MODULE.stable_digest(
                unsigned_attestation
            )
        source_unsigned = dict(source)
        source_unsigned.pop("population_record_sha256")
        source["population_record_sha256"] = MODULE.stable_digest(source_unsigned)
        review_unsigned = dict(review)
        review_unsigned.pop("record_sha256")
        review["record_sha256"] = MODULE.stable_digest(review_unsigned)
        provenance_row: dict[str, Any] = {
            "schema_version": 2,
            "record_id": source["record_id"],
            "query_candidate_identity": candidate["candidate_identity"],
            "source_facts": source_facts,
            "lineage": lineage,
            "public_basis": basis,
        }
        provenance_row["provenance_record_sha256"] = MODULE.stable_digest(
            provenance_row
        )
        provenance.append(provenance_row)

    population_bytes = _jsonl_bytes(population)
    provenance_bytes = _jsonl_bytes(provenance)
    ledger_bytes = _jsonl_bytes(ledger)
    registry_bytes = _json_bytes(_registry())
    review_bundle = _review_bundle(ledger, registry_bytes)
    if review_bundle_mutator is not None:
        review_bundle_mutator(review_bundle)
    review_bytes = _json_bytes(review_bundle)
    amendment_bytes = _json_bytes(
        {
            "schema_version": 2,
            "artifact_kind": "sampling_design_amendment",
            "status": "pending_authenticated_verification",
        }
    )
    power_bytes = _json_bytes(
        {
            "schema_version": 2,
            "artifact_kind": "prospective_power_receipt",
            "status": "pending_authenticated_verification",
        }
    )
    git_freeze = {
        "commit_sha": "1" * 40,
        "commit_url": f"https://github.com/wandb/fugue/commit/{'1' * 40}",
        "committed_at_utc": "2026-07-01T00:00:00Z",
    }
    if lane == "superpowers":
        partition_order = ["development", "target_holdout", "safety_control"]
        partition_roles = {
            "development": "development",
            "target_holdout": "primary_inference",
            "safety_control": "safety_gate",
        }
        quotas = {
            "development": {
                "security_privacy": 8,
                "identity_migration": 8,
                "lifecycle_recovery": 8,
                "integration_runtime": 8,
            },
            "target_holdout": {
                "security_privacy": 48,
                "identity_migration": 48,
                "lifecycle_recovery": 48,
                "integration_runtime": 48,
            },
            "safety_control": {
                "docs_only": 16,
                "unsafe_request": 16,
                "no_change_required": 16,
                "single_surface": 16,
            },
        }
        provider = "nist-randomness-beacon-2.0"
        pulse_or_round: str | int = "2026-07-02T00:00:00Z"
        chain_hash = None
        endpoint = "https://beacon.nist.gov/beacon/2.0/pulse/time/1782950400000"
        replacement = "invalidate_frame_on_preexecution_failure"
    else:
        partition_order = ["target_holdout", "development", "safety_control"]
        partition_roles = {
            "target_holdout": "primary_inference",
            "development": "development",
            "safety_control": "safety_gate",
        }
        quotas = {
            "target_holdout": {"maintenance": 1},
            "development": {"maintenance": 1},
            "safety_control": {"repair": 1},
        }
        provider = "drand-mainnet"
        pulse_or_round = 123456
        chain_hash = SHA["chain"]
        endpoint = "https://api.drand.sh/public/123456"
        replacement = "next_frozen_rank_same_cohort_stratum"
    plan: dict[str, Any] = {
        "schema_version": 2,
        "selection_id": f"{lane}-population-selection-v2",
        "lane": _lane(lane, protocol_bytes),
        "artifacts": {
            "query_plan_sha256": query_plan_sha256,
            "population_sha256": MODULE._sha256_bytes(population_bytes),  # noqa: SLF001
            "query_receipt_sha256": MODULE._sha256_bytes(query_bytes),  # noqa: SLF001
            "candidate_records_sha256": MODULE._sha256_bytes(candidate_bytes),  # noqa: SLF001
            "provenance_bundle_sha256": MODULE._sha256_bytes(provenance_bytes),  # noqa: SLF001
            "eligibility_ledger_sha256": MODULE._sha256_bytes(ledger_bytes),  # noqa: SLF001
            "reviewer_registry_sha256": MODULE._sha256_bytes(registry_bytes),  # noqa: SLF001
            "review_verification_sha256": MODULE._sha256_bytes(review_bytes),  # noqa: SLF001
            "sampling_design_amendment_sha256": MODULE._sha256_bytes(amendment_bytes),  # noqa: SLF001
            "power_receipt_sha256": MODULE._sha256_bytes(power_bytes),  # noqa: SLF001
            "selection_code_sha256": MODULE._sha256_bytes(MODULE_PATH.read_bytes()),  # noqa: SLF001
        },
        "population_frozen_at_utc": "2026-06-30T14:00:00Z",
        "public_git_freeze": git_freeze,
        "precommit": {"commitment_sha256": "0" * 64, "receipt_sha256": "0" * 64},
        "beacon": {
            "provider": provider,
            "pulse_or_round": pulse_or_round,
            "chain_hash": chain_hash,
            "canonical_endpoint": endpoint,
            "receipt_sha256": "0" * 64,
        },
        "verifier_policy": {
            "precommit": _verifier("public-git-freeze-verifier-v1"),
            "beacon": _verifier(f"{lane}-beacon-verifier-v1"),
        },
        "reviewer_policy": {
            "reviewer_ids": ["reviewer-a", "reviewer-b"],
            "adjudicator_ids": ["adjudicator-c"],
            "registry_sha256": MODULE._sha256_bytes(registry_bytes),  # noqa: SLF001
            "verification_bundle_sha256": MODULE._sha256_bytes(review_bytes),  # noqa: SLF001
            "verifier": _verifier("review-signature-verifier-v1"),
        },
        "partition_order": partition_order,
        "partition_roles": partition_roles,
        "quotas": quotas,
        "minimum_pool_multiplier": 3 if lane == "superpowers" else 2,
        "cluster_policy": {
            "repository_id_type": "github_node_id",
            "record_within_repository": True,
            "repository_within_cluster": True,
            "one_cluster_per_partition_global": True,
            "repository_cluster_consistency_required": True,
            "lineage_cluster_consistency_required": True,
        },
        "weighting": {
            "method": "three_stage_cluster_design",
            "estimand": "Design-weighted paired candidate-minus-baseline effect.",
            "analysis_unit": "independence_cluster",
        },
        "replacement_policy": replacement,
    }
    commitment = MODULE.selection_commitment_digest(plan)
    plan["precommit"]["commitment_sha256"] = commitment
    precommit = {
        "schema_version": 2,
        "commitment_sha256": commitment,
        "public_git_freeze": git_freeze,
        "verification_passed": True,
        "verified_at_utc": "2026-07-01T00:01:00Z",
        "verifier": plan["verifier_policy"]["precommit"],
    }
    precommit_bytes = _json_bytes(precommit)
    beacon = {
        "schema_version": 2,
        "provider": provider,
        "pulse_or_round": pulse_or_round,
        "chain_hash": chain_hash,
        "canonical_endpoint": endpoint,
        "published_at_utc": (
            "2026-07-02T00:00:00Z" if lane == "superpowers" else "2026-07-01T01:00:00Z"
        ),
        "retrieved_at_utc": (
            "2026-07-02T00:01:00Z" if lane == "superpowers" else "2026-07-01T01:01:00Z"
        ),
        "verified_at_utc": (
            "2026-07-02T00:02:00Z" if lane == "superpowers" else "2026-07-01T01:02:00Z"
        ),
        "random_value": SHA["randomness"],
        "response_sha256": SHA["response"],
        "signature_sha256": SHA["beacon_signature"],
        "precommit_commitment_sha256": commitment,
        "public_git_freeze_commit_sha": git_freeze["commit_sha"],
        "signature_verification_passed": True,
        "target_rule_verified": True,
        "verification_passed": True,
        "verifier": copy.deepcopy(plan["verifier_policy"]["beacon"]),
    }
    if beacon_mutator is not None:
        beacon_mutator(beacon)
    beacon_bytes = _json_bytes(beacon)
    plan["precommit"]["receipt_sha256"] = MODULE._sha256_bytes(precommit_bytes)  # noqa: SLF001
    plan["beacon"]["receipt_sha256"] = MODULE._sha256_bytes(beacon_bytes)  # noqa: SLF001
    return {
        "selection_plan_bytes": _json_bytes(plan),
        "lane_protocol_bytes": protocol_bytes,
        "query_plan_bytes": query_plan_bytes,
        "query_receipt_bytes": query_bytes,
        "candidate_jsonl_bytes": candidate_bytes,
        "population_jsonl_bytes": population_bytes,
        "provenance_jsonl_bytes": provenance_bytes,
        "eligibility_jsonl_bytes": ledger_bytes,
        "reviewer_registry_bytes": registry_bytes,
        "review_verification_bytes": review_bytes,
        "precommit_receipt_bytes": precommit_bytes,
        "beacon_receipt_bytes": beacon_bytes,
        "sampling_design_amendment_bytes": amendment_bytes,
        "power_receipt_bytes": power_bytes,
    }


def _select(case: dict[str, bytes]) -> dict[str, Any]:
    return MODULE.select_population_from_bytes(**case)


def _validated_context(case: dict[str, bytes]) -> dict[str, Any]:
    protocol = json.loads(case["lane_protocol_bytes"])
    plan = MODULE._validate_plan(  # noqa: SLF001
        json.loads(case["selection_plan_bytes"]),
        protocol,
        MODULE._sha256_bytes(case["lane_protocol_bytes"]),  # noqa: SLF001
    )
    registry_value = MODULE._validate_registry(  # noqa: SLF001
        json.loads(case["reviewer_registry_bytes"]), plan
    )
    registry = {row["reviewer_id"]: row for row in registry_value["reviewers"]}
    rows, _ = MODULE._validate_rows(  # noqa: SLF001
        [json.loads(line) for line in case["eligibility_jsonl_bytes"].splitlines()],
        plan,
        registry,
    )
    population = MODULE._validate_population(  # noqa: SLF001
        [json.loads(line) for line in case["population_jsonl_bytes"].splitlines()],
        rows,
    )
    candidates = MODULE._validate_candidates(  # noqa: SLF001
        [json.loads(line) for line in case["candidate_jsonl_bytes"].splitlines()]
    )
    query_plan = MODULE._validate_query_plan(  # noqa: SLF001
        json.loads(case["query_plan_bytes"]),
        protocol_id=plan["lane"]["protocol_id"],
        query_plan_sha256=MODULE._sha256_bytes(case["query_plan_bytes"]),  # noqa: SLF001
    )
    return {
        "plan": plan,
        "registry": registry,
        "rows": rows,
        "population": population,
        "candidates": candidates,
        "query_plan": query_plan,
    }


def test_superpowers_adapter_binds_exact_protocol_nist_policy_and_no_replacement() -> (
    None
):
    case = _seal_case("superpowers")
    receipt = _select(case)

    assert receipt["lane"]["protocol_id"] == MODULE.SUPERPOWERS_PROTOCOL_ID
    assert receipt["status"] == "blocked"
    assert receipt["selected_count"] == 0
    assert sum(receipt["frame_independent_cluster_counts"].values()) == 864
    assert receipt["replacement_policy"] == "invalidate_frame_on_preexecution_failure"
    assert "authenticated_beacon_provider_response_missing" in receipt["blockers"]
    assert receipt["selected"] == []


def test_anthropic_adapter_uses_record_repository_cluster_stages_and_primary_weights() -> (
    None
):
    case = _seal_case("anthropic")
    first = _select(case)
    second = _select(case)

    assert first == second
    assert first["lane"]["cohort_id"] == "maintenance_primary"
    assert first["replacement_policy"] == "next_frozen_rank_same_cohort_stratum"
    assert first["frame_independent_cluster_counts"] == {"maintenance": 4, "repair": 2}
    assert first["status"] == "blocked"
    assert first["selected_count"] == 0
    protocol = json.loads(case["lane_protocol_bytes"])
    plan = MODULE._validate_plan(  # noqa: SLF001
        json.loads(case["selection_plan_bytes"]),
        protocol,
        MODULE._sha256_bytes(case["lane_protocol_bytes"]),  # noqa: SLF001
    )
    registry_value = MODULE._validate_registry(  # noqa: SLF001
        json.loads(case["reviewer_registry_bytes"]), plan
    )
    registry = {row["reviewer_id"]: row for row in registry_value["reviewers"]}
    rows, _ = MODULE._validate_rows(  # noqa: SLF001
        [json.loads(line) for line in case["eligibility_jsonl_bytes"].splitlines()],
        plan,
        registry,
    )
    seed = MODULE._selection_seed(plan, SHA["randomness"])  # noqa: SLF001
    frame = MODULE._eligible_frame(rows, plan)  # noqa: SLF001
    choices = MODULE._repository_and_cluster_choices(frame, seed, plan)  # noqa: SLF001
    selected, _, _ = MODULE._allocate(plan, choices, seed)  # noqa: SLF001
    assert len(selected) == 3
    target = next(row for row in selected if row["partition"] == "target_holdout")
    assert target["record_within_repository_probability"] == {
        "numerator": 1,
        "denominator": 2,
    }
    assert target["repository_within_cluster_probability"] == {
        "numerator": 1,
        "denominator": 2,
    }
    assert target["cluster_into_partition_probability"] == {
        "numerator": 1,
        "denominator": 4,
    }
    assert target["total_record_inclusion_probability"] == {
        "numerator": 1,
        "denominator": 16,
    }
    assert target["analysis_weight"] == {"numerator": 4, "denominator": 1}
    assert target["finite_population_correction"] == {"numerator": 1, "denominator": 1}
    assert len(target["frozen_replacement_order"]) == 2
    assert all(
        row["analysis_weight"] is None and row["finite_population_correction"] is None
        for row in selected
        if row["partition"] != "target_holdout"
    )
    unsigned = dict(first)
    digest = unsigned.pop("receipt_sha256")
    assert digest == MODULE.stable_digest(unsigned)


def test_duplicate_json_keys_fail_closed_before_digest_validation() -> None:
    case = _seal_case("anthropic")
    case["selection_plan_bytes"] = case["selection_plan_bytes"].replace(
        b'{"artifacts"', b'{"schema_version":2,"artifacts"', 1
    )
    with pytest.raises(
        MODULE.PopulationSelectionError, match="duplicate JSON key 'schema_version'"
    ):
        _select(case)


def test_supplied_bytes_and_current_selector_code_are_rehashed() -> None:
    case = _seal_case("anthropic")
    case["query_receipt_bytes"] += b" "
    with pytest.raises(MODULE.PopulationSelectionError, match="query_receipt_sha256"):
        _select(case)

    case = _seal_case("anthropic")
    plan = json.loads(case["selection_plan_bytes"])
    plan["artifacts"]["selection_code_sha256"] = "9" * 64
    plan["precommit"]["commitment_sha256"] = MODULE.selection_commitment_digest(plan)
    case["selection_plan_bytes"] = _json_bytes(plan)
    with pytest.raises(MODULE.PopulationSelectionError, match="selection_code_sha256"):
        _select(case)


def test_reviewer_signatures_are_bound_to_registry_and_external_verifier() -> None:
    def mutate(bundle: dict[str, Any]) -> None:
        bundle["verifications"][0]["signature_sha256"] = "0" * 64

    case = _seal_case("anthropic", review_bundle_mutator=mutate)
    with pytest.raises(
        MODULE.PopulationSelectionError, match="does not match signed attestation"
    ):
        _select(case)


def test_every_disagreement_requires_signed_adjudication_reason() -> None:
    population, ledger = _anthropic_rows(disagree=True, empty_adjudication_reason=True)
    case = _seal_case("anthropic", population=population, ledger=ledger)
    with pytest.raises(MODULE.PopulationSelectionError, match="adjudication_reason"):
        _select(case)


def test_nist_pulse_must_be_first_verified_target_at_least_24_hours_later() -> None:
    def early(beacon: dict[str, Any]) -> None:
        beacon["published_at_utc"] = "2026-07-01T23:59:59Z"
        beacon["retrieved_at_utc"] = "2026-07-02T00:01:00Z"
        beacon["verified_at_utc"] = "2026-07-02T00:02:00Z"

    case = _seal_case("superpowers", beacon_mutator=early)
    with pytest.raises(MODULE.PopulationSelectionError, match="at least 24h"):
        _select(case)


def test_drand_chain_and_external_verifier_trust_root_are_locked() -> None:
    def wrong_chain(beacon: dict[str, Any]) -> None:
        beacon["chain_hash"] = "1" * 64

    case = _seal_case("anthropic", beacon_mutator=wrong_chain)
    with pytest.raises(
        MODULE.PopulationSelectionError, match="chain_hash is not locked"
    ):
        _select(case)

    def wrong_trust(beacon: dict[str, Any]) -> None:
        beacon["verifier"]["trust_root_sha256"] = "1" * 64

    case = _seal_case("anthropic", beacon_mutator=wrong_trust)
    with pytest.raises(
        MODULE.PopulationSelectionError, match="verifier identity is not locked"
    ):
        _select(case)


def test_seed_formulas_match_each_locked_protocol() -> None:
    anthropic_case = _seal_case("anthropic")
    anthropic_plan = json.loads(anthropic_case["selection_plan_bytes"])
    bundle = {
        key: anthropic_plan["artifacts"][key]
        for key in (
            "query_plan_sha256",
            "query_receipt_sha256",
            "candidate_records_sha256",
            "population_sha256",
            "provenance_bundle_sha256",
            "eligibility_ledger_sha256",
            "reviewer_registry_sha256",
            "review_verification_sha256",
            "sampling_design_amendment_sha256",
            "power_receipt_sha256",
        )
    }
    expected_drand = MODULE._sha256_bytes(  # noqa: SLF001
        "\n".join(
            [
                MODULE.stable_digest(bundle),
                SHA["chain"],
                "123456",
                SHA["randomness"],
            ]
        ).encode()
    )
    assert MODULE._selection_seed(anthropic_plan, SHA["randomness"]) == expected_drand  # noqa: SLF001

    superpowers_case = _seal_case("superpowers")
    superpowers_plan = json.loads(superpowers_case["selection_plan_bytes"])
    expected_nist = MODULE._sha256_bytes(  # noqa: SLF001
        (
            f"{MODULE.SUPERPOWERS_PROTOCOL_ID}\n"
            f"{superpowers_plan['artifacts']['population_sha256']}\n"
            f"{SHA['randomness']}\n"
        ).encode()
    )
    assert MODULE._selection_seed(superpowers_plan, SHA["randomness"]) == expected_nist  # noqa: SLF001


def test_repository_ids_and_repository_lineage_assignments_are_canonical() -> None:
    with pytest.raises(
        MODULE.PopulationSelectionError, match="canonical string GitHub"
    ):
        MODULE._repository_id(123, "repository_id")  # noqa: SLF001

    population, ledger = _anthropic_rows()
    ledger[1]["independence_cluster_id"] = "different-cluster"
    ledger[1]["source_lineage_id"] = "different-lineage"
    for name in ("reviewer_a", "reviewer_b"):
        attestation = ledger[1]["review"][name]
        attestation["independence_cluster_id"] = "different-cluster"
        attestation["source_lineage_id"] = "different-lineage"
        unsigned_attestation = dict(attestation)
        del unsigned_attestation["signature_sha256"]
        del unsigned_attestation["attestation_sha256"]
        attestation["attestation_sha256"] = MODULE.stable_digest(unsigned_attestation)
    unsigned = dict(ledger[1])
    del unsigned["record_sha256"]
    ledger[1]["record_sha256"] = MODULE.stable_digest(unsigned)
    case = _seal_case("anthropic", population=population, ledger=ledger)
    with pytest.raises(
        MODULE.PopulationSelectionError, match="inconsistent stratum or lineage"
    ):
        _select(case)


def test_protocol_bytes_and_exact_treatment_cannot_drift() -> None:
    case = _seal_case("anthropic")
    case["lane_protocol_bytes"] += b" "
    with pytest.raises(MODULE.PopulationSelectionError, match="pinned protocol"):
        _select(case)

    case = _seal_case("anthropic")
    plan = json.loads(case["selection_plan_bytes"])
    plan["lane"]["treatment"]["candidate_revision"] = "0" * 40
    plan["precommit"]["commitment_sha256"] = MODULE.selection_commitment_digest(plan)
    case["selection_plan_bytes"] = _json_bytes(plan)
    with pytest.raises(
        MODULE.PopulationSelectionError, match="treatment does not match"
    ):
        _select(case)


def test_missing_authenticated_inputs_returns_typed_block_not_selection() -> None:
    case = _seal_case("anthropic")
    case["query_plan_bytes"] = None  # type: ignore[assignment]
    receipt = _select(case)
    assert receipt["status"] == "blocked"
    assert receipt["blocker_code"] == "selection_inputs_missing"
    assert receipt["selected_count"] == 0


def test_self_asserted_beacon_boolean_cannot_authorize_a_selected_population() -> None:
    case = _seal_case("anthropic")
    original_plan = json.loads(case["selection_plan_bytes"])
    beacon = json.loads(case["beacon_receipt_bytes"])
    beacon["random_value"] = "a" * 64
    case["beacon_receipt_bytes"] = _json_bytes(beacon)
    original_plan["beacon"]["receipt_sha256"] = MODULE._sha256_bytes(  # noqa: SLF001
        case["beacon_receipt_bytes"]
    )
    case["selection_plan_bytes"] = _json_bytes(original_plan)

    receipt = _select(case)
    assert receipt["status"] == "blocked"
    assert receipt["selected"] == []
    assert "authenticated_beacon_provider_response_missing" in receipt["blockers"]


def test_query_receipt_and_query_plan_are_strict_and_exact() -> None:
    case = _seal_case("anthropic")
    with pytest.raises(MODULE.PopulationSelectionError, match="invalid keys"):
        MODULE._validate_query_receipt(  # noqa: SLF001
            {"schema_version": 2, "query_id": "dummy"},
            query_plan_sha256=MODULE._sha256_bytes(case["query_plan_bytes"]),  # noqa: SLF001
            candidate_bytes=case["candidate_jsonl_bytes"],
        )

    mutated = case["query_plan_bytes"] + b" "
    with pytest.raises(MODULE.PopulationSelectionError, match="not pinned"):
        MODULE._validate_query_plan(  # noqa: SLF001
            json.loads(mutated),
            protocol_id=MODULE.ANTHROPIC_PROTOCOL_ID,
            query_plan_sha256=MODULE._sha256_bytes(mutated),  # noqa: SLF001
        )


def test_superpowers_exact_eligible_cluster_minimum_is_864() -> None:
    case = _seal_case("superpowers")
    context = _validated_context(case)
    limits = {
        "security_privacy": 56,
        "identity_migration": 56,
        "lifecycle_recovery": 56,
        "integration_runtime": 56,
        "docs_only": 16,
        "unsafe_request": 16,
        "no_change_required": 16,
        "single_surface": 16,
    }
    counts: dict[str, int] = {}
    underfilled: list[dict[str, Any]] = []
    for row in context["rows"]:
        stratum = row["stratum"]
        if counts.get(stratum, 0) < limits[stratum]:
            counts[stratum] = counts.get(stratum, 0) + 1
            underfilled.append(row)
    with pytest.raises(MODULE.PopulationSelectionError, match="minima are unmet"):
        MODULE._eligible_frame(underfilled, context["plan"])  # noqa: SLF001


def test_reviewer_aliases_cannot_share_one_signing_key() -> None:
    case = _seal_case("anthropic")
    context = _validated_context(case)
    registry = _registry()
    registry["reviewers"][1]["signing_key_sha256"] = registry["reviewers"][0][
        "signing_key_sha256"
    ]
    with pytest.raises(MODULE.PopulationSelectionError, match="unique signing keys"):
        MODULE._validate_registry(registry, context["plan"])  # noqa: SLF001


def test_review_after_population_freeze_fails() -> None:
    case = _seal_case("anthropic")
    context = _validated_context(case)
    raw_rows = [
        json.loads(line) for line in case["eligibility_jsonl_bytes"].splitlines()
    ]
    attestation = raw_rows[0]["review"]["reviewer_a"]
    attestation["reviewed_at_utc"] = "2026-06-30T15:00:00Z"
    unsigned = dict(attestation)
    unsigned.pop("signature_sha256")
    unsigned.pop("attestation_sha256")
    attestation["attestation_sha256"] = MODULE.stable_digest(unsigned)
    with pytest.raises(
        MODULE.PopulationSelectionError, match="after population freeze"
    ):
        MODULE._validate_rows(  # noqa: SLF001
            raw_rows, context["plan"], context["registry"]
        )


def test_lineage_cluster_cannot_inflate_multiple_strata() -> None:
    case = _seal_case("anthropic")
    context = _validated_context(case)
    rows = copy.deepcopy(context["rows"])
    for row in rows:
        if row["independence_cluster_id"] == "cluster-0101":
            row["independence_cluster_id"] = "cluster-0001"
            row["source_lineage_id"] = "lineage-0001"
    with pytest.raises(MODULE.PopulationSelectionError, match="occupy one stratum"):
        MODULE._eligible_frame(rows, context["plan"])  # noqa: SLF001


def test_nonfinite_json_fails_closed() -> None:
    with pytest.raises(MODULE.PopulationSelectionError, match="non-finite"):
        MODULE._parse_json_bytes(b'{"value":NaN}', "test")  # noqa: SLF001


def test_provenance_must_be_a_discovered_candidate_and_obey_cutoff() -> None:
    case = _seal_case("anthropic")
    context = _validated_context(case)
    provenance = [
        json.loads(line) for line in case["provenance_jsonl_bytes"].splitlines()
    ]
    provenance[0]["query_candidate_identity"] = "candidate-not-discovered"
    with pytest.raises(MODULE.PopulationSelectionError, match="absent from discovery"):
        MODULE._validate_provenance(  # noqa: SLF001
            provenance,
            population=context["population"],
            eligibility_rows={row["record_id"]: row for row in context["rows"]},
            candidates=context["candidates"],
            plan=context["plan"],
            query_plan=context["query_plan"],
            query_plan_sha256=MODULE._sha256_bytes(case["query_plan_bytes"]),  # noqa: SLF001
            query_receipt_sha256=MODULE._sha256_bytes(case["query_receipt_bytes"]),  # noqa: SLF001
        )

    provenance = [
        json.loads(line) for line in case["provenance_jsonl_bytes"].splitlines()
    ]
    provenance[0]["source_facts"]["record_published_at_utc"] = "2026-02-07T00:00:00Z"
    with pytest.raises(MODULE.PopulationSelectionError, match="not pre-treatment"):
        MODULE._validate_provenance(  # noqa: SLF001
            provenance,
            population=context["population"],
            eligibility_rows={row["record_id"]: row for row in context["rows"]},
            candidates=context["candidates"],
            plan=context["plan"],
            query_plan=context["query_plan"],
            query_plan_sha256=MODULE._sha256_bytes(case["query_plan_bytes"]),  # noqa: SLF001
            query_receipt_sha256=MODULE._sha256_bytes(case["query_receipt_bytes"]),  # noqa: SLF001
        )


def test_anthropic_post_treatment_requires_source_and_record_after_window_start() -> (
    None
):
    plan = {
        "lane": {
            "protocol_id": MODULE.ANTHROPIC_PROTOCOL_ID,
            "temporal_relation": "post_treatment",
            "treatment": {
                "candidate_public_at_utc": "2026-02-06T14:59:13Z",
                "source_window_start_utc": "2026-02-06T14:59:14Z",
            },
        }
    }
    query_plan = {
        "temporal_contract": {
            "primary_pre_treatment_cutoff_utc": "2026-02-06T14:59:12Z"
        }
    }
    with pytest.raises(MODULE.PopulationSelectionError, match="not post-treatment"):
        MODULE._validate_provenance_times(  # noqa: SLF001
            {
                "record_published_at_utc": "2026-02-07T00:00:00Z",
                "source_committed_at_utc": "2026-02-01T00:00:00Z",
            },
            plan=plan,
            query_plan=query_plan,
        )
    with pytest.raises(MODULE.PopulationSelectionError, match="not post-treatment"):
        MODULE._validate_provenance_times(  # noqa: SLF001
            {
                "record_published_at_utc": "2026-02-06T14:59:13Z",
                "source_committed_at_utc": "2026-02-06T14:59:13Z",
            },
            plan=plan,
            query_plan=query_plan,
        )
    MODULE._validate_provenance_times(  # noqa: SLF001
        {
            "record_published_at_utc": "2026-02-06T14:59:14Z",
            "source_committed_at_utc": "2026-02-06T14:59:14Z",
        },
        plan=plan,
        query_plan=query_plan,
    )


def test_lane_exact_score_golden_vectors() -> None:
    seed = "0" * 64
    assert MODULE._domain_seed(seed, "record-choice-v2") == (  # noqa: SLF001
        "c97ede6cb682a518765df336e9a9e45c386c604fbc7ce2d3b6cfb728cd0fcc64"
    )
    assert MODULE._superpowers_record_score(seed, "record-0001") == (  # noqa: SLF001
        "a27077c1888cb6035cc07043ed78d2a25fb012d3e3cd76cfa28bdb8be817940c"
    )
    assert (
        MODULE._superpowers_repository_record_score(  # noqa: SLF001
            seed, "cluster-0001", "R_repo_0001", "record-0001"
        )
        == "8a94761ad96c08f3056fd94af94cc960bce81e32296171e3a9a794be96f443c4"
    )
    assert (
        MODULE._superpowers_partition_score(  # noqa: SLF001
            seed, "cluster-0001", "record-0001"
        )
        == "ac464dbbf22cc9b258ccd84121d2e714dac4d102f77c0235cc755641b64caeba"
    )
    assert (
        MODULE._anthropic_record_score(  # noqa: SLF001
            seed,
            {
                "source_lineage_id": "lineage-0001",
                "repository_id": "R_repo_0001",
                "skill_path": "skills/example",
                "public_basis_bundle_sha256": "1" * 64,
            },
        )
        == "ceefa4d77fb5367b535c08d4645537903fbaeb852614acc1bd6cd36605235367"
    )


@pytest.mark.parametrize("lane", ["superpowers", "anthropic"])
def test_record_score_collision_uses_stable_record_id_tie_break(
    lane: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_id = (
        MODULE.SUPERPOWERS_PROTOCOL_ID
        if lane == "superpowers"
        else MODULE.ANTHROPIC_PROTOCOL_ID
    )
    if lane == "superpowers":
        monkeypatch.setattr(MODULE, "_superpowers_record_score", lambda *_: "0" * 64)
    else:
        monkeypatch.setattr(MODULE, "_anthropic_record_score", lambda *_: "0" * 64)
    common = {
        "repository_id": "R_repo_0001",
        "source_lineage_id": "lineage-0001",
        "independence_cluster_id": "cluster-0001",
        "skill_path": "skills/example",
        "public_basis_bundle_sha256": "1" * 64,
    }
    frame = {
        "maintenance": {
            "R_repo_0001": [
                {**common, "record_id": "record-b"},
                {**common, "record_id": "record-a"},
            ]
        }
    }
    choices = MODULE._repository_and_cluster_choices(  # noqa: SLF001
        frame, "2" * 64, {"lane": {"protocol_id": protocol_id}}
    )
    choice = choices["maintenance"]["cluster-0001"]["choice"]
    assert choice["row"]["record_id"] == "record-a"
    assert [row["record_id"] for _score, row in choice["record_order"]] == [
        "record-a",
        "record-b",
    ]


def test_policy_digests_must_equal_actual_registry_and_review_artifacts() -> None:
    case = _seal_case("anthropic")
    plan = json.loads(case["selection_plan_bytes"])
    plan["reviewer_policy"]["registry_sha256"] = "0" * 64
    with pytest.raises(MODULE.PopulationSelectionError, match="policy digest"):
        MODULE._validate_plan(  # noqa: SLF001
            plan,
            json.loads(case["lane_protocol_bytes"]),
            MODULE._sha256_bytes(case["lane_protocol_bytes"]),  # noqa: SLF001
        )


def test_private_output_is_atomic_mode_0600_and_symlinks_fail(
    tmp_path: Path,
) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    os.chmod(private, 0o700)
    output = private / "receipt.json"
    MODULE._write_private_atomic(output, b"{}\n")  # noqa: SLF001
    assert output.read_bytes() == b"{}\n"
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(MODULE.PopulationSelectionError, match="overwrite"):
        MODULE._write_private_atomic(output, b"changed")  # noqa: SLF001

    target = private / "target.json"
    link = private / "link.json"
    link.symlink_to(target)
    with pytest.raises(MODULE.PopulationSelectionError, match="overwrite"):
        MODULE._write_private_atomic(link, b"{}")  # noqa: SLF001

    source = private / "source.json"
    source.write_bytes(b"{}")
    os.chmod(source, 0o600)
    source_link = private / "source-link.json"
    source_link.symlink_to(source)
    with pytest.raises(MODULE.PopulationSelectionError, match="non-symlink"):
        MODULE._read_regular_file(  # noqa: SLF001
            source_link, field="private source", private=True
        )

    public_parent = tmp_path / "public-parent"
    public_parent.mkdir(mode=0o755)
    os.chmod(public_parent, 0o755)
    exposed_source = public_parent / "private.json"
    exposed_source.write_bytes(b"{}")
    os.chmod(exposed_source, 0o600)
    with pytest.raises(MODULE.PopulationSelectionError, match="mode 0700"):
        MODULE._read_regular_file(  # noqa: SLF001
            exposed_source, field="private source", private=True
        )
    with pytest.raises(MODULE.PopulationSelectionError, match="mode 0700"):
        MODULE._write_private_atomic(  # noqa: SLF001
            public_parent / "output.json", b"{}"
        )

    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    os.chmod(real, 0o700)
    nested = real / "nested"
    nested.mkdir(mode=0o700)
    os.chmod(nested, 0o700)
    nested_source = nested / "source.json"
    nested_source.write_bytes(b"{}")
    os.chmod(nested_source, 0o600)
    ancestor_link = tmp_path / "ancestor-link"
    ancestor_link.symlink_to(real, target_is_directory=True)
    with pytest.raises(MODULE.PopulationSelectionError, match="ancestor"):
        MODULE._read_regular_file(  # noqa: SLF001
            ancestor_link / "nested" / "source.json",
            field="private source",
            private=True,
        )
    with pytest.raises(MODULE.PopulationSelectionError, match="ancestor"):
        MODULE._write_private_atomic(  # noqa: SLF001
            ancestor_link / "nested" / "output.json", b"{}"
        )
