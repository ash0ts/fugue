"""Select a sealed public-task population with externally verified randomness.

The public entry point is deliberately bytes-first. Every input digest is
recomputed from the bytes that are parsed, the selector hashes its own source,
and duplicate JSON keys are rejected. Cryptographic signature verification is
performed by separately locked verifier programs; this selector only accepts
their receipts when code, runtime, trust-root, and signed-object identities all
match the precommitted plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_RANDOM_HEX = re.compile(r"^(?:[0-9a-f]{64}|[0-9a-f]{128})$")
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._:-]{2,255}$")
_GITHUB_REPOSITORY_ID = re.compile(r"^R_[A-Za-z0-9_-]{4,255}$")
_ISO_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

SUPERPOWERS_PROTOCOL_ID = (
    "superpowers-writing-plans-conference-sampling-frame-protocol-v2"
)
ANTHROPIC_PROTOCOL_ID = "anthropic-skill-creator-conference-sampling-frame-protocol-v2"

_PINNED_PROTOCOL_SHA256 = {
    SUPERPOWERS_PROTOCOL_ID: "5cfb8c4e4e97658eed2d69dd2f0bb4a126ea5f23ec1db0064a9ad67ea3407b3c",
    ANTHROPIC_PROTOCOL_ID: "f2c91e5b4d7a51ecf3ff268821f05b0b2c183e0ad3f397d95e05bac7785f5bde",
}
_PINNED_QUERY_PLAN_SHA256 = {
    SUPERPOWERS_PROTOCOL_ID: "9bfc0c3ca8042f8b55f38c561daa1c600f998392740007612da235e048d753fe",
    ANTHROPIC_PROTOCOL_ID: "c93a0baf99a889ff06a82d81eafe6028e91a81dbbb9762a9e7069063a862933d",
}
_SUPERPOWERS_MINIMUM_ELIGIBLE_CLUSTERS = {
    "security_privacy": 168,
    "identity_migration": 168,
    "lifecycle_recovery": 168,
    "integration_runtime": 168,
    "docs_only": 48,
    "unsafe_request": 48,
    "no_change_required": 48,
    "single_surface": 48,
}

_SUPERPOWERS_TREATMENT = {
    "repository": "https://github.com/obra/superpowers",
    "skill_path": "skills/writing-plans",
    "baseline_revision": "de4672b171213a6ff6960228d8b95c46ea0b09f4",
    "candidate_revision": "8e1262a3bae92b640d87fa81c51c53b65e490590",
    "candidate_public_at_utc": "2026-06-16T17:09:47Z",
    "source_window_start_utc": "2026-06-16T17:09:48Z",
    "source_window_end_utc": None,
}
_ANTHROPIC_TREATMENT = {
    "repository": "https://github.com/anthropics/skills",
    "skill_path": "skills/skill-creator",
    "baseline_revision": "a5bcdd7e58cdff48566bf876f0a72a2008dcefbc",
    "candidate_revision": "1ed29a03dc852d30fa6ef2ca53a67dc2c2c2c563",
    "candidate_public_at_utc": "2026-02-06T14:59:13Z",
    "source_window_start_utc": None,
    "source_window_end_utc": "2026-02-06T14:59:12Z",
}
_ANTHROPIC_COHORTS = {
    "maintenance_primary": ("primary_inference", "pre_treatment"),
    "creation_secondary": ("secondary_inference", "pre_treatment"),
    "post_treatment_deployment_sensitivity": ("sensitivity", "post_treatment"),
}

_PLAN_KEYS = {
    "schema_version",
    "selection_id",
    "lane",
    "artifacts",
    "population_frozen_at_utc",
    "public_git_freeze",
    "precommit",
    "beacon",
    "verifier_policy",
    "reviewer_policy",
    "partition_order",
    "partition_roles",
    "quotas",
    "minimum_pool_multiplier",
    "cluster_policy",
    "weighting",
    "replacement_policy",
}
_LANE_KEYS = {
    "protocol_id",
    "protocol_sha256",
    "cohort_id",
    "cohort_role",
    "temporal_relation",
    "treatment",
}
_TREATMENT_KEYS = set(_SUPERPOWERS_TREATMENT)
_ARTIFACT_KEYS = {
    "query_plan_sha256",
    "population_sha256",
    "query_receipt_sha256",
    "candidate_records_sha256",
    "provenance_bundle_sha256",
    "eligibility_ledger_sha256",
    "reviewer_registry_sha256",
    "review_verification_sha256",
    "sampling_design_amendment_sha256",
    "power_receipt_sha256",
    "selection_code_sha256",
}
_GIT_FREEZE_KEYS = {"commit_sha", "commit_url", "committed_at_utc"}
_PRECOMMIT_PLAN_KEYS = {"commitment_sha256", "receipt_sha256"}
_BEACON_PLAN_KEYS = {
    "provider",
    "pulse_or_round",
    "chain_hash",
    "canonical_endpoint",
    "receipt_sha256",
}
_VERIFIER_KEYS = {
    "identity",
    "method",
    "code_sha256",
    "runtime_sha256",
    "trust_root_sha256",
}
_VERIFIER_POLICY_KEYS = {"precommit", "beacon"}
_REVIEWER_POLICY_KEYS = {
    "reviewer_ids",
    "adjudicator_ids",
    "registry_sha256",
    "verification_bundle_sha256",
    "verifier",
}
_CLUSTER_POLICY_KEYS = {
    "repository_id_type",
    "record_within_repository",
    "repository_within_cluster",
    "one_cluster_per_partition_global",
    "repository_cluster_consistency_required",
    "lineage_cluster_consistency_required",
}
_WEIGHTING_KEYS = {"method", "estimand", "analysis_unit"}
_PRECOMMIT_RECEIPT_KEYS = {
    "schema_version",
    "commitment_sha256",
    "public_git_freeze",
    "verification_passed",
    "verified_at_utc",
    "verifier",
}
_BEACON_RECEIPT_KEYS = {
    "schema_version",
    "provider",
    "pulse_or_round",
    "chain_hash",
    "canonical_endpoint",
    "published_at_utc",
    "retrieved_at_utc",
    "verified_at_utc",
    "random_value",
    "response_sha256",
    "signature_sha256",
    "precommit_commitment_sha256",
    "public_git_freeze_commit_sha",
    "signature_verification_passed",
    "target_rule_verified",
    "verification_passed",
    "verifier",
}
_REGISTRY_KEYS = {"schema_version", "reviewers"}
_REGISTRY_REVIEWER_KEYS = {"reviewer_id", "role", "signing_key_sha256"}
_VERIFICATION_BUNDLE_KEYS = {
    "schema_version",
    "reviewer_registry_sha256",
    "verifier",
    "verifications",
}
_VERIFICATION_KEYS = {
    "attestation_sha256",
    "reviewer_id",
    "signing_key_sha256",
    "signature_sha256",
    "verification_passed",
    "verified_at_utc",
}
_FINAL_QUERY_RECEIPT_KEYS = {
    "schema_version",
    "receipt_id",
    "lane_id",
    "status",
    "candidate_discovery_only",
    "eligibility_or_population_claim",
    "public_only_repository_visibility_verified",
    "plan_sha256",
    "api_origin",
    "api_version",
    "source_query_plan_sha256",
    "temporal_scope",
    "selection_source_cutoff_utc",
    "acquisition_repetitions",
    "index_stabilization_seconds",
    "repeat_separation_seconds",
    "acquisition_windows",
    "acquisition_receipts",
    "credential_profile",
    "collector_implementation",
    "public_visibility",
    "query_count",
    "shards",
    "responses",
    "response_count",
    "repository_visibility_receipts",
    "repository_visibility_receipt_count",
    "candidate_count",
    "candidates_path",
    "candidates_sha256",
    "privacy_scan",
    "limitations",
    "receipt_sha256",
}
_CANDIDATE_KEYS = {
    "candidate_identity",
    "query_ids",
    "item_sha256",
    "observed_item_sha256s",
    "item",
    "privacy_review",
}
_CANDIDATE_PRIVACY_KEYS = {
    "status",
    "pattern_ids",
    "matched_values_serialized",
    "raw_source_redacted",
    "downstream_authoring_export_allowed",
}
_PROVENANCE_KEYS = {
    "schema_version",
    "record_id",
    "query_candidate_identity",
    "source_facts",
    "lineage",
    "public_basis",
    "provenance_record_sha256",
}
_SOURCE_FACT_KEYS = {
    "record_id",
    "repository_id",
    "skill_path",
    "cohort_id",
    "record_published_at_utc",
    "source_committed_at_utc",
    "query_plan_sha256",
}
_LINEAGE_KEYS = {
    "repository_id",
    "owner_node_id",
    "source_root_repository_id",
    "template_root_repository_id",
    "source_lineage_id",
    "independence_cluster_id",
}
_PUBLIC_BASIS_KEYS = {
    "record_id",
    "query_candidate_identity",
    "query_receipt_sha256",
    "query_plan_sha256",
    "candidate_item_sha256",
}
_POPULATION_KEYS = {
    "schema_version",
    "record_id",
    "repository_id",
    "skill_path",
    "cohort_id",
    "source_facts_sha256",
    "lineage_receipt_sha256",
    "public_basis_bundle_sha256",
    "population_record_sha256",
}
_LEDGER_KEYS = {
    "schema_version",
    "record_id",
    "repository_id",
    "skill_path",
    "cohort_id",
    "independence_cluster_id",
    "source_lineage_id",
    "stratum",
    "temporal_relation",
    "source_facts_sha256",
    "lineage_receipt_sha256",
    "public_basis_bundle_sha256",
    "review",
    "exclusion_reasons",
    "record_sha256",
}
_REVIEW_KEYS = {"reviewer_a", "reviewer_b", "adjudication"}
_ATTESTATION_KEYS = {
    "reviewer_id",
    "role",
    "signing_key_sha256",
    "record_id",
    "repository_id",
    "decision",
    "source_facts_sha256",
    "lineage_receipt_sha256",
    "stratum",
    "source_lineage_id",
    "independence_cluster_id",
    "temporal_relation",
    "reviewed_at_utc",
    "reason_codes",
    "adjudication_reason",
    "treatment_output_blinded",
    "selection_rank_blinded",
    "signature_sha256",
    "attestation_sha256",
}
_REVIEWER_DECISIONS = {"eligible", "excluded", "uncertain"}
_FINAL_DECISIONS = {"eligible", "excluded"}
_PARTITION_ROLES = {
    "primary_inference",
    "secondary_inference",
    "sensitivity",
    "development",
    "safety_gate",
}


class PopulationSelectionError(ValueError):
    """Raised when selection cannot preserve the frozen design."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def stable_digest(value: object) -> str:
    """Return the canonical JSON SHA-256 digest used inside signed records."""

    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PopulationSelectionError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise PopulationSelectionError(f"non-finite JSON constant {value!r} is forbidden")


def _parse_json_bytes(raw: bytes, field: str) -> object:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PopulationSelectionError(f"invalid {field}: {exc}") from exc


def _parse_jsonl_bytes(raw: bytes, field: str) -> list[object]:
    rows: list[object] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise PopulationSelectionError(f"{field} contains a blank row")
        rows.append(_parse_json_bytes(line, f"{field} line {line_number}"))
    if not rows:
        raise PopulationSelectionError(f"{field} must not be empty")
    return rows


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PopulationSelectionError(f"{field} must be an object")
    return value


def _exact(value: Mapping[str, object], keys: set[str], field: str) -> None:
    if set(value) != keys:
        raise PopulationSelectionError(
            f"{field} has invalid keys; missing={sorted(keys - set(value))}, "
            f"unknown={sorted(set(value) - keys)}"
        )


def _text(value: object, field: str, *, maximum: int = 4096) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum
        or any(character in value for character in "\x00\r\n")
    ):
        raise PopulationSelectionError(f"{field} must be non-empty single-line text")
    return value


def _safe_id(value: object, field: str) -> str:
    text = _text(value, field, maximum=256)
    if _SAFE_ID.fullmatch(text) is None:
        raise PopulationSelectionError(f"{field} must be a safe identifier")
    return text


def _sha(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PopulationSelectionError(f"{field} must be a lowercase SHA-256")
    return value


def _git_sha(value: object, field: str) -> str:
    if not isinstance(value, str) or _GIT_SHA.fullmatch(value) is None:
        raise PopulationSelectionError(f"{field} must be a lowercase Git SHA")
    return value


def _repository_id(value: object, field: str) -> str:
    if not isinstance(value, str) or _GITHUB_REPOSITORY_ID.fullmatch(value) is None:
        raise PopulationSelectionError(
            f"{field} must be a canonical string GitHub repository node ID"
        )
    return value


def _utc(value: object, field: str) -> datetime:
    if not isinstance(value, str) or _ISO_UTC.fullmatch(value) is None:
        raise PopulationSelectionError(f"{field} must use YYYY-MM-DDTHH:MM:SSZ")
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def _true(value: object, field: str) -> None:
    if value is not True:
        raise PopulationSelectionError(f"{field} must be true")


def _fraction(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _verifier(value: object, field: str) -> dict[str, Any]:
    verifier = dict(_mapping(value, field))
    _exact(verifier, _VERIFIER_KEYS, field)
    _safe_id(verifier["identity"], f"{field}.identity")
    _text(verifier["method"], f"{field}.method", maximum=256)
    for key in ("code_sha256", "runtime_sha256", "trust_root_sha256"):
        _sha(verifier[key], f"{field}.{key}")
    return verifier


def _git_freeze(value: object, field: str) -> dict[str, Any]:
    freeze = dict(_mapping(value, field))
    _exact(freeze, _GIT_FREEZE_KEYS, field)
    commit_sha = _git_sha(freeze["commit_sha"], f"{field}.commit_sha")
    url = _text(freeze["commit_url"], f"{field}.commit_url")
    if not url.startswith("https://github.com/") or not url.endswith(
        f"/commit/{commit_sha}"
    ):
        raise PopulationSelectionError(
            f"{field}.commit_url must be the exact public GitHub commit URL"
        )
    _utc(freeze["committed_at_utc"], f"{field}.committed_at_utc")
    return freeze


def selection_commitment_digest(plan: Mapping[str, Any]) -> str:
    """Digest every pre-randomness field without circular receipt hashes."""

    payload = {key: plan[key] for key in _PLAN_KEYS - {"precommit", "beacon"}}
    beacon = _mapping(plan.get("beacon"), "selection plan.beacon")
    payload["beacon"] = {
        "provider": beacon.get("provider"),
        "pulse_or_round": beacon.get("pulse_or_round"),
        "chain_hash": beacon.get("chain_hash"),
        "canonical_endpoint": beacon.get("canonical_endpoint"),
    }
    payload["commitment_contract"] = "public-population-selection-v2"
    return stable_digest(payload)


def _validate_superpowers_lane(
    *,
    treatment: Mapping[str, Any],
    cohort_id: str,
    cohort_role: str,
    temporal: object,
    protocol_treatment: Mapping[str, Any],
) -> None:
    if treatment != _SUPERPOWERS_TREATMENT:
        raise PopulationSelectionError("Superpowers treatment window is not exact")
    if (
        cohort_id != "post_treatment_maintenance_primary"
        or cohort_role != "primary_inference"
        or temporal != "post_treatment"
    ):
        raise PopulationSelectionError("Superpowers cohort policy is not exact")
    if (
        protocol_treatment.get("candidate_committer_time_utc")
        != treatment["candidate_public_at_utc"]
        or protocol_treatment.get("task_record_window_starts_utc")
        != treatment["source_window_start_utc"]
    ):
        raise PopulationSelectionError("Superpowers protocol treatment times drifted")


def _validate_anthropic_lane(
    *,
    treatment: Mapping[str, Any],
    cohort_id: str,
    cohort_role: str,
    temporal: object,
    protocol: Mapping[str, Any],
    protocol_treatment: Mapping[str, Any],
) -> None:
    expected = _ANTHROPIC_COHORTS.get(cohort_id)
    if expected != (cohort_role, temporal):
        raise PopulationSelectionError(
            "Anthropic cohort role or temporal policy is wrong"
        )
    expected_treatment = dict(_ANTHROPIC_TREATMENT)
    if temporal == "post_treatment":
        expected_treatment["source_window_start_utc"] = "2026-02-06T14:59:14Z"
        expected_treatment["source_window_end_utc"] = None
    if treatment != expected_treatment:
        raise PopulationSelectionError("Anthropic treatment window is not exact")
    protocol_cohorts = _mapping(protocol.get("cohorts"), "protocol.cohorts")
    if cohort_id not in protocol_cohorts:
        raise PopulationSelectionError("Anthropic cohort is absent from protocol")
    if (
        protocol_treatment.get("candidate_public_at_utc")
        != treatment["candidate_public_at_utc"]
        or protocol_treatment.get("primary_source_cutoff_utc") != "2026-02-06T14:59:12Z"
    ):
        raise PopulationSelectionError("Anthropic protocol treatment times drifted")


def _validate_protocol_lane(
    lane: Mapping[str, Any], protocol: Mapping[str, Any], protocol_sha256: str
) -> None:
    _exact(lane, _LANE_KEYS, "selection plan.lane")
    protocol_id = _safe_id(lane["protocol_id"], "selection plan.lane.protocol_id")
    if protocol.get("protocol_id") != protocol_id:
        raise PopulationSelectionError("lane protocol_id does not match protocol bytes")
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise PopulationSelectionError("lane protocol schema_version must be 2")
    if protocol.get("status") != "prospective_not_yet_sampled":
        raise PopulationSelectionError("lane protocol status is not prospective")
    expected_protocol_sha256 = _PINNED_PROTOCOL_SHA256.get(protocol_id)
    if expected_protocol_sha256 is None or protocol_sha256 != expected_protocol_sha256:
        raise PopulationSelectionError(
            "lane protocol bytes are not the pinned protocol"
        )
    if lane["protocol_sha256"] != protocol_sha256:
        raise PopulationSelectionError(
            "lane protocol digest does not match supplied bytes"
        )
    query_plan = _mapping(protocol.get("query_plan"), "protocol.query_plan")
    if query_plan.get("sha256") != _PINNED_QUERY_PLAN_SHA256[protocol_id]:
        raise PopulationSelectionError("protocol does not bind the pinned query plan")
    treatment = dict(_mapping(lane["treatment"], "selection plan.lane.treatment"))
    _exact(treatment, _TREATMENT_KEYS, "selection plan.lane.treatment")
    for key in ("repository", "skill_path"):
        _text(treatment[key], f"selection plan.lane.treatment.{key}")
    for key in ("baseline_revision", "candidate_revision"):
        _git_sha(treatment[key], f"selection plan.lane.treatment.{key}")
    _utc(
        treatment["candidate_public_at_utc"],
        "selection plan.lane.treatment.candidate_public_at_utc",
    )
    for key in ("source_window_start_utc", "source_window_end_utc"):
        if treatment[key] is not None:
            _utc(treatment[key], f"selection plan.lane.treatment.{key}")
    cohort_id = _safe_id(lane["cohort_id"], "selection plan.lane.cohort_id")
    cohort_role = _safe_id(lane["cohort_role"], "selection plan.lane.cohort_role")
    temporal = lane["temporal_relation"]
    if temporal not in {"pre_treatment", "post_treatment"}:
        raise PopulationSelectionError("lane temporal_relation is unsupported")

    protocol_treatment = _mapping(protocol.get("treatment"), "protocol.treatment")
    common = {
        "repository": protocol_treatment.get("repository"),
        "skill_path": protocol_treatment.get("path"),
        "baseline_revision": protocol_treatment.get("baseline_commit"),
        "candidate_revision": protocol_treatment.get("candidate_commit"),
    }
    if any(treatment[key] != value for key, value in common.items()):
        raise PopulationSelectionError("lane treatment does not match locked protocol")

    if protocol_id == SUPERPOWERS_PROTOCOL_ID:
        _validate_superpowers_lane(
            treatment=treatment,
            cohort_id=cohort_id,
            cohort_role=cohort_role,
            temporal=temporal,
            protocol_treatment=protocol_treatment,
        )
    elif protocol_id == ANTHROPIC_PROTOCOL_ID:
        _validate_anthropic_lane(
            treatment=treatment,
            cohort_id=cohort_id,
            cohort_role=cohort_role,
            temporal=temporal,
            protocol=protocol,
            protocol_treatment=protocol_treatment,
        )
    else:
        raise PopulationSelectionError("unsupported lane protocol")


def _validate_partitions(plan: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
    order = plan["partition_order"]
    if (
        not isinstance(order, list)
        or not order
        or len(order) != len(set(order))
        or not all(isinstance(item, str) and _SAFE_ID.fullmatch(item) for item in order)
    ):
        raise PopulationSelectionError("partition_order must contain unique safe IDs")
    roles = _mapping(plan["partition_roles"], "selection plan.partition_roles")
    if set(roles) != set(order) or any(
        role not in _PARTITION_ROLES for role in roles.values()
    ):
        raise PopulationSelectionError(
            "partition roles must exactly classify every partition"
        )
    quotas = _mapping(plan["quotas"], "selection plan.quotas")
    if set(quotas) != set(order):
        raise PopulationSelectionError("quotas must exactly match partition_order")
    for partition, raw in quotas.items():
        stratum_quotas = _mapping(raw, f"selection plan.quotas.{partition}")
        if not stratum_quotas:
            raise PopulationSelectionError(f"partition {partition} has no quota")
        for stratum, quota in stratum_quotas.items():
            _safe_id(stratum, f"selection plan.quotas.{partition}.stratum")
            if isinstance(quota, bool) or not isinstance(quota, int) or quota < 1:
                raise PopulationSelectionError("all quotas must be positive integers")

    protocol_id = plan["lane"]["protocol_id"]
    if protocol_id == SUPERPOWERS_PROTOCOL_ID:
        if order != ["development", "target_holdout", "safety_control"]:
            raise PopulationSelectionError("Superpowers partition order is not exact")
        expected_roles = {
            "development": "development",
            "target_holdout": "primary_inference",
            "safety_control": "safety_gate",
        }
        if dict(roles) != expected_roles:
            raise PopulationSelectionError("Superpowers partition roles are not exact")
        protocol_quotas = _mapping(
            _mapping(protocol.get("selection"), "protocol.selection").get("quotas"),
            "protocol.selection.quotas",
        )
        expected_quotas = {
            partition: {
                stratum: count
                for stratum, count in _mapping(values, "protocol quota").items()
                if stratum != "total"
            }
            for partition, values in protocol_quotas.items()
        }
        if {key: dict(value) for key, value in quotas.items()} != expected_quotas:
            raise PopulationSelectionError("Superpowers quotas drift from protocol")
    else:
        inference_role = str(plan["lane"]["cohort_role"])
        if inference_role not in roles.values():
            raise PopulationSelectionError(
                "Anthropic cohort has no matching inference partition"
            )
        first_role = roles[order[0]]
        if first_role != inference_role:
            raise PopulationSelectionError(
                "Anthropic inference partition must be allocated first"
            )


def _validate_randomness_plan(plan: dict[str, Any], lane: Mapping[str, Any]) -> None:
    precommit = dict(_mapping(plan["precommit"], "selection plan.precommit"))
    _exact(precommit, _PRECOMMIT_PLAN_KEYS, "selection plan.precommit")
    _sha(precommit["commitment_sha256"], "selection plan.precommit.commitment_sha256")
    _sha(precommit["receipt_sha256"], "selection plan.precommit.receipt_sha256")
    plan["precommit"] = precommit

    beacon = dict(_mapping(plan["beacon"], "selection plan.beacon"))
    _exact(beacon, _BEACON_PLAN_KEYS, "selection plan.beacon")
    if beacon["provider"] not in {"nist-randomness-beacon-2.0", "drand-mainnet"}:
        raise PopulationSelectionError("unsupported beacon provider")
    if beacon["provider"] == "drand-mainnet":
        if (
            isinstance(beacon["pulse_or_round"], bool)
            or not isinstance(beacon["pulse_or_round"], int)
            or beacon["pulse_or_round"] < 1
        ):
            raise PopulationSelectionError("drand round must be positive")
        _sha(beacon["chain_hash"], "selection plan.beacon.chain_hash")
    else:
        _text(beacon["pulse_or_round"], "selection plan.beacon.pulse_or_round")
        if beacon["chain_hash"] is not None:
            raise PopulationSelectionError(
                "NIST beacon must not have a drand chain hash"
            )
    endpoint = _text(
        beacon["canonical_endpoint"], "selection plan.beacon.canonical_endpoint"
    )
    if beacon["provider"] == "drand-mainnet" and not endpoint.startswith(
        "https://api.drand.sh/"
    ):
        raise PopulationSelectionError("drand canonical endpoint is invalid")
    if beacon["provider"] == "nist-randomness-beacon-2.0" and not endpoint.startswith(
        "https://beacon.nist.gov/beacon/2.0/"
    ):
        raise PopulationSelectionError("NIST canonical endpoint is invalid")
    _sha(beacon["receipt_sha256"], "selection plan.beacon.receipt_sha256")
    plan["beacon"] = beacon

    expected_provider = (
        "nist-randomness-beacon-2.0"
        if lane["protocol_id"] == SUPERPOWERS_PROTOCOL_ID
        else "drand-mainnet"
    )
    if beacon["provider"] != expected_provider:
        raise PopulationSelectionError("lane uses the wrong randomness provider")
    policies = dict(_mapping(plan["verifier_policy"], "selection plan.verifier_policy"))
    _exact(policies, _VERIFIER_POLICY_KEYS, "selection plan.verifier_policy")
    plan["verifier_policy"] = {
        key: _verifier(value, f"verifier_policy.{key}")
        for key, value in policies.items()
    }


def _validate_plan(
    value: object, protocol: Mapping[str, Any], protocol_sha256: str
) -> dict[str, Any]:
    plan = dict(_mapping(value, "selection plan"))
    _exact(plan, _PLAN_KEYS, "selection plan")
    if plan["schema_version"] != SCHEMA_VERSION:
        raise PopulationSelectionError("selection plan schema_version must be 2")
    _safe_id(plan["selection_id"], "selection plan.selection_id")
    lane = dict(_mapping(plan["lane"], "selection plan.lane"))
    _validate_protocol_lane(lane, protocol, protocol_sha256)
    plan["lane"] = lane
    artifacts = dict(_mapping(plan["artifacts"], "selection plan.artifacts"))
    _exact(artifacts, _ARTIFACT_KEYS, "selection plan.artifacts")
    for key, digest in artifacts.items():
        _sha(digest, f"selection plan.artifacts.{key}")
    plan["artifacts"] = artifacts
    frozen_at = _utc(plan["population_frozen_at_utc"], "population_frozen_at_utc")
    freeze = _git_freeze(plan["public_git_freeze"], "selection plan.public_git_freeze")
    if _utc(freeze["committed_at_utc"], "public git freeze") < frozen_at:
        raise PopulationSelectionError(
            "public Git freeze cannot precede population freeze"
        )
    plan["public_git_freeze"] = freeze

    _validate_randomness_plan(plan, lane)

    reviewer_policy = dict(
        _mapping(plan["reviewer_policy"], "selection plan.reviewer_policy")
    )
    _exact(reviewer_policy, _REVIEWER_POLICY_KEYS, "selection plan.reviewer_policy")
    for key in ("reviewer_ids", "adjudicator_ids"):
        values = reviewer_policy[key]
        if (
            not isinstance(values, list)
            or not values
            or len(values) != len(set(values))
        ):
            raise PopulationSelectionError(
                f"reviewer_policy.{key} must be unique and nonempty"
            )
        for reviewer_id in values:
            _safe_id(reviewer_id, f"reviewer_policy.{key}")
    if set(reviewer_policy["reviewer_ids"]) & set(reviewer_policy["adjudicator_ids"]):
        raise PopulationSelectionError("reviewers and adjudicators must be independent")
    for key in ("registry_sha256", "verification_bundle_sha256"):
        _sha(reviewer_policy[key], f"reviewer_policy.{key}")
    reviewer_policy["verifier"] = _verifier(
        reviewer_policy["verifier"], "reviewer_policy.verifier"
    )
    if reviewer_policy["registry_sha256"] != artifacts["reviewer_registry_sha256"]:
        raise PopulationSelectionError(
            "reviewer registry policy digest does not match artifact bytes"
        )
    if (
        reviewer_policy["verification_bundle_sha256"]
        != artifacts["review_verification_sha256"]
    ):
        raise PopulationSelectionError(
            "review verification policy digest does not match artifact bytes"
        )
    plan["reviewer_policy"] = reviewer_policy
    _validate_partitions(plan, protocol)
    multiplier = plan["minimum_pool_multiplier"]
    if (
        isinstance(multiplier, bool)
        or not isinstance(multiplier, int)
        or multiplier < 1
    ):
        raise PopulationSelectionError("minimum_pool_multiplier must be positive")
    cluster_policy = dict(
        _mapping(plan["cluster_policy"], "selection plan.cluster_policy")
    )
    _exact(cluster_policy, _CLUSTER_POLICY_KEYS, "selection plan.cluster_policy")
    if cluster_policy["repository_id_type"] != "github_node_id" or not all(
        cluster_policy[key] is True
        for key in _CLUSTER_POLICY_KEYS - {"repository_id_type"}
    ):
        raise PopulationSelectionError("cluster policy must fail closed")
    weighting = dict(_mapping(plan["weighting"], "selection plan.weighting"))
    _exact(weighting, _WEIGHTING_KEYS, "selection plan.weighting")
    if (
        weighting["method"] != "three_stage_cluster_design"
        or weighting["analysis_unit"] != "independence_cluster"
    ):
        raise PopulationSelectionError("weighting policy is unsupported")
    _text(weighting["estimand"], "selection plan.weighting.estimand")
    expected_replacement = (
        "invalidate_frame_on_preexecution_failure"
        if lane["protocol_id"] == SUPERPOWERS_PROTOCOL_ID
        else "next_frozen_rank_same_cohort_stratum"
    )
    if plan["replacement_policy"] != expected_replacement:
        raise PopulationSelectionError(
            "replacement policy does not match lane protocol"
        )
    if plan["precommit"]["commitment_sha256"] != selection_commitment_digest(plan):
        raise PopulationSelectionError("precommit does not bind the frozen design")
    return plan


def _validate_precommit(value: object, plan: Mapping[str, Any]) -> dict[str, Any]:
    receipt = dict(_mapping(value, "precommit receipt"))
    _exact(receipt, _PRECOMMIT_RECEIPT_KEYS, "precommit receipt")
    if receipt["schema_version"] != SCHEMA_VERSION:
        raise PopulationSelectionError("precommit receipt schema_version must be 2")
    if receipt["commitment_sha256"] != plan["precommit"]["commitment_sha256"]:
        raise PopulationSelectionError("precommit receipt has the wrong commitment")
    if receipt["commitment_sha256"] != selection_commitment_digest(plan):
        raise PopulationSelectionError("precommit receipt does not bind the design")
    if (
        _git_freeze(receipt["public_git_freeze"], "precommit public_git_freeze")
        != plan["public_git_freeze"]
    ):
        raise PopulationSelectionError("precommit receipt has the wrong Git freeze")
    verifier = _verifier(receipt["verifier"], "precommit receipt.verifier")
    if verifier != plan["verifier_policy"]["precommit"]:
        raise PopulationSelectionError("precommit verifier identity is not locked")
    _true(receipt["verification_passed"], "precommit verification_passed")
    verified_at = _utc(receipt["verified_at_utc"], "precommit verified_at_utc")
    committed_at = _utc(
        plan["public_git_freeze"]["committed_at_utc"], "Git freeze time"
    )
    if verified_at < committed_at:
        raise PopulationSelectionError("precommit verification predates the Git freeze")
    return receipt


def _validate_beacon(
    value: object, plan: Mapping[str, Any], precommit: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = dict(_mapping(value, "beacon receipt"))
    _exact(receipt, _BEACON_RECEIPT_KEYS, "beacon receipt")
    if receipt["schema_version"] != SCHEMA_VERSION:
        raise PopulationSelectionError("beacon receipt schema_version must be 2")
    for key in ("provider", "pulse_or_round", "chain_hash", "canonical_endpoint"):
        if receipt[key] != plan["beacon"][key]:
            raise PopulationSelectionError(f"beacon receipt {key} is not locked")
    if receipt["precommit_commitment_sha256"] != precommit["commitment_sha256"]:
        raise PopulationSelectionError("beacon receipt has the wrong precommit")
    if (
        receipt["public_git_freeze_commit_sha"]
        != plan["public_git_freeze"]["commit_sha"]
    ):
        raise PopulationSelectionError("beacon receipt has the wrong Git freeze")
    verifier = _verifier(receipt["verifier"], "beacon receipt.verifier")
    if verifier != plan["verifier_policy"]["beacon"]:
        raise PopulationSelectionError("beacon verifier identity is not locked")
    for key in ("response_sha256", "signature_sha256"):
        _sha(receipt[key], f"beacon receipt.{key}")
    random_value = receipt["random_value"]
    if not isinstance(random_value, str) or _RANDOM_HEX.fullmatch(random_value) is None:
        raise PopulationSelectionError("beacon random_value is invalid")
    for key in (
        "signature_verification_passed",
        "target_rule_verified",
        "verification_passed",
    ):
        _true(receipt[key], f"beacon receipt.{key}")
    published = _utc(receipt["published_at_utc"], "beacon published_at_utc")
    retrieved = _utc(receipt["retrieved_at_utc"], "beacon retrieved_at_utc")
    verified = _utc(receipt["verified_at_utc"], "beacon verified_at_utc")
    committed = _utc(plan["public_git_freeze"]["committed_at_utc"], "Git freeze time")
    if published <= committed or retrieved < published or verified < retrieved:
        raise PopulationSelectionError("beacon chronology is invalid")
    if receipt[
        "provider"
    ] == "nist-randomness-beacon-2.0" and published < committed + timedelta(hours=24):
        raise PopulationSelectionError(
            "NIST pulse must be at least 24h after public Git freeze"
        )
    return receipt


def _validate_registry(value: object, plan: Mapping[str, Any]) -> dict[str, Any]:
    registry = dict(_mapping(value, "reviewer registry"))
    _exact(registry, _REGISTRY_KEYS, "reviewer registry")
    if registry["schema_version"] != SCHEMA_VERSION or not isinstance(
        registry["reviewers"], list
    ):
        raise PopulationSelectionError("reviewer registry is invalid")
    seen: set[str] = set()
    expected_roles = {
        **{
            reviewer: "eligibility_reviewer"
            for reviewer in plan["reviewer_policy"]["reviewer_ids"]
        },
        **{
            reviewer: "eligibility_adjudicator"
            for reviewer in plan["reviewer_policy"]["adjudicator_ids"]
        },
    }
    signing_keys: set[str] = set()
    for index, raw in enumerate(registry["reviewers"]):
        reviewer = _mapping(raw, f"reviewer registry row {index}")
        _exact(reviewer, _REGISTRY_REVIEWER_KEYS, f"reviewer registry row {index}")
        reviewer_id = _safe_id(
            reviewer["reviewer_id"], f"registry row {index}.reviewer_id"
        )
        if reviewer_id in seen or expected_roles.get(reviewer_id) != reviewer["role"]:
            raise PopulationSelectionError(
                "reviewer registry identity or role is not locked"
            )
        seen.add(reviewer_id)
        signing_key = _sha(
            reviewer["signing_key_sha256"],
            f"registry row {index}.signing_key_sha256",
        )
        if signing_key in signing_keys:
            raise PopulationSelectionError(
                "reviewers and adjudicators must use unique signing keys"
            )
        signing_keys.add(signing_key)
    if seen != set(expected_roles):
        raise PopulationSelectionError(
            "reviewer registry does not exactly match the plan"
        )
    return registry


def _validate_attestation(
    value: object, field: str, role: str, registry: Mapping[str, Mapping[str, str]]
) -> dict[str, Any]:
    attestation = dict(_mapping(value, field))
    _exact(attestation, _ATTESTATION_KEYS, field)
    reviewer_id = _safe_id(attestation["reviewer_id"], f"{field}.reviewer_id")
    if attestation["role"] != role or registry.get(reviewer_id, {}).get("role") != role:
        raise PopulationSelectionError(f"{field} reviewer role is not registered")
    if attestation["signing_key_sha256"] != registry[reviewer_id]["signing_key_sha256"]:
        raise PopulationSelectionError(f"{field} signing key is not registered")
    _repository_id(attestation["repository_id"], f"{field}.repository_id")
    _safe_id(attestation["record_id"], f"{field}.record_id")
    decisions = (
        _FINAL_DECISIONS if role == "eligibility_adjudicator" else _REVIEWER_DECISIONS
    )
    if attestation["decision"] not in decisions:
        raise PopulationSelectionError(f"{field}.decision is invalid")
    for key in (
        "source_facts_sha256",
        "lineage_receipt_sha256",
        "signing_key_sha256",
        "signature_sha256",
    ):
        _sha(attestation[key], f"{field}.{key}")
    for key in (
        "stratum",
        "source_lineage_id",
        "independence_cluster_id",
        "temporal_relation",
    ):
        _safe_id(attestation[key], f"{field}.{key}")
    _utc(attestation["reviewed_at_utc"], f"{field}.reviewed_at_utc")
    if not isinstance(attestation["reason_codes"], list) or not all(
        isinstance(reason, str) and reason.strip()
        for reason in attestation["reason_codes"]
    ):
        raise PopulationSelectionError(f"{field}.reason_codes is invalid")
    if attestation["decision"] != "eligible" and not attestation["reason_codes"]:
        raise PopulationSelectionError(f"{field} requires reason codes")
    if role == "eligibility_adjudicator":
        _text(attestation["adjudication_reason"], f"{field}.adjudication_reason")
    elif attestation["adjudication_reason"] is not None:
        raise PopulationSelectionError(f"{field} reviewer cannot adjudicate")
    _true(attestation["treatment_output_blinded"], f"{field}.treatment_output_blinded")
    _true(attestation["selection_rank_blinded"], f"{field}.selection_rank_blinded")
    digest = _sha(attestation["attestation_sha256"], f"{field}.attestation_sha256")
    unsigned = dict(attestation)
    del unsigned["attestation_sha256"]
    del unsigned["signature_sha256"]
    if digest != stable_digest(unsigned):
        raise PopulationSelectionError(f"{field} attestation digest does not match")
    return attestation


def _assignment(attestation: Mapping[str, Any]) -> tuple[object, ...]:
    return tuple(
        attestation[key]
        for key in (
            "record_id",
            "repository_id",
            "decision",
            "source_facts_sha256",
            "lineage_receipt_sha256",
            "stratum",
            "source_lineage_id",
            "independence_cluster_id",
            "temporal_relation",
        )
    )


def _matches_row(
    attestation: Mapping[str, Any], row: Mapping[str, Any], field: str
) -> None:
    for key in (
        "record_id",
        "repository_id",
        "source_facts_sha256",
        "lineage_receipt_sha256",
        "stratum",
        "source_lineage_id",
        "independence_cluster_id",
        "temporal_relation",
    ):
        if attestation[key] != row[key]:
            raise PopulationSelectionError(f"{field}.{key} does not match ledger row")


def _validate_rows(
    raw_rows: Sequence[object],
    plan: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    attestations: dict[str, dict[str, Any]] = {}
    record_ids: set[str] = set()
    population_frozen_at = _utc(
        plan["population_frozen_at_utc"], "population_frozen_at_utc"
    )
    for index, raw in enumerate(raw_rows):
        row = dict(_mapping(raw, f"eligibility row {index}"))
        _exact(row, _LEDGER_KEYS, f"eligibility row {index}")
        if row["schema_version"] != SCHEMA_VERSION:
            raise PopulationSelectionError("eligibility rows must use schema_version 2")
        record_id = _safe_id(row["record_id"], f"row {index}.record_id")
        if record_id in record_ids:
            raise PopulationSelectionError(f"duplicate record_id {record_id}")
        record_ids.add(record_id)
        _repository_id(row["repository_id"], f"row {index}.repository_id")
        _text(row["skill_path"], f"row {index}.skill_path")
        if row["cohort_id"] != plan["lane"]["cohort_id"]:
            raise PopulationSelectionError(f"row {record_id} has the wrong cohort")
        if row["temporal_relation"] != plan["lane"]["temporal_relation"]:
            raise PopulationSelectionError(
                f"row {record_id} has the wrong treatment period"
            )
        for key in ("independence_cluster_id", "source_lineage_id", "stratum"):
            _safe_id(row[key], f"row {index}.{key}")
        for key in (
            "source_facts_sha256",
            "lineage_receipt_sha256",
            "public_basis_bundle_sha256",
        ):
            _sha(row[key], f"row {index}.{key}")
        review = _mapping(row["review"], f"row {index}.review")
        _exact(review, _REVIEW_KEYS, f"row {index}.review")
        reviewer_a = _validate_attestation(
            review["reviewer_a"],
            f"row {index}.reviewer_a",
            "eligibility_reviewer",
            registry,
        )
        reviewer_b = _validate_attestation(
            review["reviewer_b"],
            f"row {index}.reviewer_b",
            "eligibility_reviewer",
            registry,
        )
        if reviewer_a["reviewer_id"] == reviewer_b["reviewer_id"]:
            raise PopulationSelectionError("two independent reviewers are required")
        agreement = _assignment(reviewer_a) == _assignment(reviewer_b)
        if agreement and reviewer_a["decision"] in _FINAL_DECISIONS:
            if review["adjudication"] is not None:
                raise PopulationSelectionError(
                    "agreement must not fabricate adjudication"
                )
            final = reviewer_a
        else:
            if review["adjudication"] is None:
                raise PopulationSelectionError(
                    "every reviewer disagreement requires adjudication"
                )
            final = _validate_attestation(
                review["adjudication"],
                f"row {index}.adjudication",
                "eligibility_adjudicator",
                registry,
            )
            if final["reviewer_id"] in {
                reviewer_a["reviewer_id"],
                reviewer_b["reviewer_id"],
            }:
                raise PopulationSelectionError("adjudicator must be independent")
        for name, attestation in (
            ("reviewer_a", reviewer_a),
            ("reviewer_b", reviewer_b),
            ("final", final),
        ):
            _matches_row(attestation, row, f"row {index}.{name}")
            if (
                _utc(attestation["reviewed_at_utc"], "reviewed_at_utc")
                > population_frozen_at
            ):
                raise PopulationSelectionError(
                    f"row {record_id} review occurred after population freeze"
                )
            attestations[attestation["attestation_sha256"]] = attestation
        reasons = row["exclusion_reasons"]
        if not isinstance(reasons, list) or not all(
            isinstance(reason, str) and reason.strip() for reason in reasons
        ):
            raise PopulationSelectionError(f"row {record_id} exclusions are invalid")
        if final["decision"] == "eligible" and reasons:
            raise PopulationSelectionError(f"eligible row {record_id} has exclusions")
        if final["decision"] == "excluded" and not reasons:
            raise PopulationSelectionError(f"excluded row {record_id} needs a reason")
        row["_final_decision"] = final["decision"]
        digest = _sha(row["record_sha256"], f"row {index}.record_sha256")
        unsigned = {
            key: value
            for key, value in row.items()
            if key not in {"record_sha256", "_final_decision"}
        }
        if digest != stable_digest(unsigned):
            raise PopulationSelectionError(f"row {record_id} digest does not match")
        rows.append(row)
    return rows, attestations


def _validate_verification_bundle(
    value: object,
    plan: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, str]],
    attestations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    bundle = dict(_mapping(value, "review verification bundle"))
    _exact(bundle, _VERIFICATION_BUNDLE_KEYS, "review verification bundle")
    if bundle["schema_version"] != SCHEMA_VERSION:
        raise PopulationSelectionError("review verification schema_version must be 2")
    if bundle["reviewer_registry_sha256"] != plan["reviewer_policy"]["registry_sha256"]:
        raise PopulationSelectionError("review verification has the wrong registry")
    if (
        _verifier(bundle["verifier"], "review verification.verifier")
        != plan["reviewer_policy"]["verifier"]
    ):
        raise PopulationSelectionError("review signature verifier is not locked")
    verifications = bundle["verifications"]
    if not isinstance(verifications, list):
        raise PopulationSelectionError("review verifications must be a list")
    seen: set[str] = set()
    for index, raw in enumerate(verifications):
        verification = _mapping(raw, f"review verification {index}")
        _exact(verification, _VERIFICATION_KEYS, f"review verification {index}")
        digest = _sha(verification["attestation_sha256"], "verified attestation")
        if digest in seen or digest not in attestations:
            raise PopulationSelectionError(
                "review verification is duplicate or unrelated"
            )
        seen.add(digest)
        attestation = attestations[digest]
        reviewer_id = str(attestation["reviewer_id"])
        if (
            verification["reviewer_id"] != reviewer_id
            or verification["signing_key_sha256"]
            != registry[reviewer_id]["signing_key_sha256"]
            or verification["signature_sha256"] != attestation["signature_sha256"]
        ):
            raise PopulationSelectionError(
                "review verification does not match signed attestation"
            )
        _true(
            verification["verification_passed"], "review signature verification_passed"
        )
        verified_at = _utc(
            verification["verified_at_utc"], "review signature verified_at_utc"
        )
        reviewed_at = _utc(attestation["reviewed_at_utc"], "reviewed_at_utc")
        frozen_at = _utc(
            plan["public_git_freeze"]["committed_at_utc"],
            "public Git freeze committed_at_utc",
        )
        if verified_at < reviewed_at or verified_at > frozen_at:
            raise PopulationSelectionError(
                "review signature verification must follow review and precede freeze"
            )
    if seen != set(attestations):
        raise PopulationSelectionError(
            "every signed attestation must be externally verified"
        )
    return bundle


def _validate_population(
    raw_rows: Sequence[object], rows: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    population: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_rows):
        row = dict(_mapping(raw, f"population row {index}"))
        _exact(row, _POPULATION_KEYS, f"population row {index}")
        if row["schema_version"] != SCHEMA_VERSION:
            raise PopulationSelectionError("population rows must use schema_version 2")
        record_id = _safe_id(row["record_id"], f"population row {index}.record_id")
        if record_id in population:
            raise PopulationSelectionError(f"duplicate population record {record_id}")
        _repository_id(row["repository_id"], f"population row {index}.repository_id")
        _text(row["skill_path"], f"population row {index}.skill_path")
        _safe_id(row["cohort_id"], f"population row {index}.cohort_id")
        for key in (
            "source_facts_sha256",
            "lineage_receipt_sha256",
            "public_basis_bundle_sha256",
        ):
            _sha(row[key], f"population row {index}.{key}")
        digest = _sha(
            row["population_record_sha256"],
            f"population row {index}.population_record_sha256",
        )
        unsigned = dict(row)
        del unsigned["population_record_sha256"]
        if digest != stable_digest(unsigned):
            raise PopulationSelectionError(
                f"population row {record_id} digest does not match"
            )
        population[record_id] = row
    if set(population) != {str(row["record_id"]) for row in rows}:
        raise PopulationSelectionError(
            "eligibility ledger must cover the frozen population exactly"
        )
    for row in rows:
        source = population[str(row["record_id"])]
        for key in (
            "repository_id",
            "skill_path",
            "cohort_id",
            "source_facts_sha256",
            "lineage_receipt_sha256",
            "public_basis_bundle_sha256",
        ):
            if row[key] != source[key]:
                raise PopulationSelectionError(
                    f"ledger row {row['record_id']} disagrees on {key}"
                )
    return population


def _validate_query_plan(
    value: object, *, protocol_id: str, query_plan_sha256: str
) -> dict[str, Any]:
    plan = dict(_mapping(value, "query plan"))
    if query_plan_sha256 != _PINNED_QUERY_PLAN_SHA256[protocol_id]:
        raise PopulationSelectionError("query plan bytes are not pinned")
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise PopulationSelectionError("query plan schema_version must be 2")
    expected_status = (
        "prospective_query_plan_only_no_population_acquired"
        if protocol_id == SUPERPOWERS_PROTOCOL_ID
        else "prospective_no_queries_executed"
    )
    if plan.get("status") != expected_status:
        raise PopulationSelectionError("query plan status is not prospective")
    return plan


def _validate_query_receipt(
    value: object,
    *,
    query_plan_sha256: str,
    candidate_bytes: bytes,
) -> dict[str, Any]:
    receipt = dict(_mapping(value, "final collector receipt"))
    _exact(receipt, _FINAL_QUERY_RECEIPT_KEYS, "final collector receipt")
    claimed = _sha(receipt["receipt_sha256"], "final collector receipt.receipt_sha256")
    unsigned = dict(receipt)
    del unsigned["receipt_sha256"]
    if claimed != stable_digest(unsigned):
        raise PopulationSelectionError("final collector receipt digest disagrees")
    required_values = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_candidate_discovery",
        "candidate_discovery_only": True,
        "eligibility_or_population_claim": False,
        "public_only_repository_visibility_verified": True,
        "api_origin": "https://api.github.com",
        "source_query_plan_sha256": query_plan_sha256,
        "candidates_path": "candidates.jsonl",
        "candidates_sha256": _sha256_bytes(candidate_bytes),
    }
    for key, expected in required_values.items():
        if receipt[key] != expected:
            raise PopulationSelectionError(f"final collector receipt has invalid {key}")
    if receipt["candidate_count"] != len(candidate_bytes.splitlines()):
        raise PopulationSelectionError("final collector candidate count disagrees")
    privacy = _mapping(receipt["privacy_scan"], "final collector privacy_scan")
    if (
        privacy.get("status") != "clear"
        or privacy.get("downstream_authoring_export_blocked") is not False
        or privacy.get("matched_values_serialized") is not False
    ):
        raise PopulationSelectionError("collector privacy scan is not clear")
    return receipt


def _validate_candidates(raw_rows: Sequence[object]) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_rows):
        row = dict(_mapping(raw, f"candidate row {index}"))
        _exact(row, _CANDIDATE_KEYS, f"candidate row {index}")
        identity = _text(
            row["candidate_identity"],
            f"candidate row {index}.candidate_identity",
            maximum=320,
        )
        if identity in candidates:
            raise PopulationSelectionError("candidate identities must be unique")
        memberships = row["query_ids"]
        if (
            not isinstance(memberships, list)
            or not memberships
            or not all(isinstance(item, str) and item for item in memberships)
            or memberships != sorted(set(memberships))
        ):
            raise PopulationSelectionError("candidate query memberships are invalid")
        item = _mapping(row["item"], f"candidate row {index}.item")
        item_sha256 = _sha(row["item_sha256"], f"candidate row {index}.item_sha256")
        observed = row["observed_item_sha256s"]
        if observed != [item_sha256] or not item:
            raise PopulationSelectionError("candidate item evidence is invalid")
        privacy = _mapping(row["privacy_review"], f"candidate row {index}.privacy")
        _exact(privacy, _CANDIDATE_PRIVACY_KEYS, f"candidate row {index}.privacy")
        if (
            privacy["status"] != "clear"
            or privacy["pattern_ids"] != []
            or privacy["matched_values_serialized"] is not False
            or privacy["raw_source_redacted"] is not False
            or privacy["downstream_authoring_export_allowed"] is not True
        ):
            raise PopulationSelectionError("candidate privacy review is not clear")
        candidates[identity] = row
    if not candidates:
        raise PopulationSelectionError("candidate records must not be empty")
    return candidates


def _optional_repository_id(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _repository_id(value, field)


def _validate_provenance_times(
    source: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    query_plan: Mapping[str, Any],
) -> None:
    record_time = _utc(source["record_published_at_utc"], "record_published_at_utc")
    source_time = _utc(source["source_committed_at_utc"], "source_committed_at_utc")
    if source_time > record_time:
        raise PopulationSelectionError(
            "source commit cannot postdate its public record"
        )
    if plan["lane"]["protocol_id"] == SUPERPOWERS_PROTOCOL_ID:
        window = _mapping(
            query_plan.get("observation_window"), "query observation_window"
        )
        start = _utc(window.get("start_inclusive_utc"), "query window start")
        end = _utc(window.get("end_inclusive_utc"), "query window end")
        if not start <= record_time <= end:
            raise PopulationSelectionError(
                "Superpowers record is outside observation window"
            )
        return
    temporal = str(plan["lane"]["temporal_relation"])
    cutoff = _utc(
        _mapping(query_plan.get("temporal_contract"), "query temporal_contract").get(
            "primary_pre_treatment_cutoff_utc"
        ),
        "Anthropic source cutoff",
    )
    candidate_public = _utc(
        plan["lane"]["treatment"]["candidate_public_at_utc"],
        "candidate_public_at_utc",
    )
    if temporal == "pre_treatment" and (record_time > cutoff or source_time > cutoff):
        raise PopulationSelectionError("Anthropic source is not pre-treatment")
    if temporal == "post_treatment":
        source_window_start = _utc(
            plan["lane"]["treatment"]["source_window_start_utc"],
            "Anthropic post-treatment source window start",
        )
        if (
            source_window_start <= candidate_public
            or record_time < source_window_start
            or source_time < source_window_start
        ):
            raise PopulationSelectionError("Anthropic source is not post-treatment")


def _validate_lineage_binding(
    lineage: Mapping[str, Any],
    *,
    population_row: Mapping[str, Any],
    eligibility_row: Mapping[str, Any],
) -> None:
    if lineage["repository_id"] != population_row["repository_id"]:
        raise PopulationSelectionError("lineage repository does not match population")
    for key in ("source_lineage_id", "independence_cluster_id"):
        _safe_id(lineage[key], f"lineage.{key}")
        if lineage[key] != eligibility_row[key]:
            raise PopulationSelectionError(f"lineage disagrees with review on {key}")
    _text(lineage["owner_node_id"], "lineage.owner_node_id")
    _optional_repository_id(
        lineage["source_root_repository_id"], "lineage.source_root_repository_id"
    )
    _optional_repository_id(
        lineage["template_root_repository_id"], "lineage.template_root_repository_id"
    )


def _validate_provenance(
    raw_rows: Sequence[object],
    *,
    population: Mapping[str, Mapping[str, Any]],
    eligibility_rows: Mapping[str, Mapping[str, Any]],
    candidates: Mapping[str, Mapping[str, Any]],
    plan: Mapping[str, Any],
    query_plan: Mapping[str, Any],
    query_plan_sha256: str,
    query_receipt_sha256: str,
) -> dict[str, dict[str, Any]]:
    provenance: dict[str, dict[str, Any]] = {}
    source_digest_records: dict[str, str] = {}
    basis_digest_records: dict[str, str] = {}
    for index, raw in enumerate(raw_rows):
        row = dict(_mapping(raw, f"provenance row {index}"))
        _exact(row, _PROVENANCE_KEYS, f"provenance row {index}")
        if row["schema_version"] != SCHEMA_VERSION:
            raise PopulationSelectionError("provenance schema_version must be 2")
        record_id = _safe_id(row["record_id"], f"provenance row {index}.record_id")
        if record_id in provenance or record_id not in population:
            raise PopulationSelectionError("provenance is duplicate or unrelated")
        candidate_identity = _text(
            row["query_candidate_identity"],
            f"provenance row {index}.query_candidate_identity",
            maximum=320,
        )
        if candidate_identity not in candidates:
            raise PopulationSelectionError(
                "provenance candidate is absent from discovery"
            )

        source = dict(
            _mapping(row["source_facts"], f"provenance {record_id}.source_facts")
        )
        _exact(source, _SOURCE_FACT_KEYS, f"provenance {record_id}.source_facts")
        lineage = dict(_mapping(row["lineage"], f"provenance {record_id}.lineage"))
        _exact(lineage, _LINEAGE_KEYS, f"provenance {record_id}.lineage")
        basis = dict(
            _mapping(row["public_basis"], f"provenance {record_id}.public_basis")
        )
        _exact(basis, _PUBLIC_BASIS_KEYS, f"provenance {record_id}.public_basis")

        population_row = population[record_id]
        eligibility_row = eligibility_rows[record_id]
        expected_source = {
            "record_id": record_id,
            "repository_id": population_row["repository_id"],
            "skill_path": population_row["skill_path"],
            "cohort_id": population_row["cohort_id"],
            "query_plan_sha256": query_plan_sha256,
        }
        for key, expected in expected_source.items():
            if source[key] != expected:
                raise PopulationSelectionError(f"provenance source disagrees on {key}")
        _repository_id(source["repository_id"], "provenance source.repository_id")
        _validate_provenance_times(source, plan=plan, query_plan=query_plan)
        _validate_lineage_binding(
            lineage, population_row=population_row, eligibility_row=eligibility_row
        )
        if basis != {
            "record_id": record_id,
            "query_candidate_identity": candidate_identity,
            "query_receipt_sha256": query_receipt_sha256,
            "query_plan_sha256": query_plan_sha256,
            "candidate_item_sha256": candidates[candidate_identity]["item_sha256"],
        }:
            raise PopulationSelectionError("public basis does not bind query evidence")
        if population_row["source_facts_sha256"] != stable_digest(source):
            raise PopulationSelectionError("source facts digest disagrees")
        if population_row["lineage_receipt_sha256"] != stable_digest(lineage):
            raise PopulationSelectionError("lineage receipt digest disagrees")
        if population_row["public_basis_bundle_sha256"] != stable_digest(basis):
            raise PopulationSelectionError("public basis digest disagrees")
        for digest, seen, name in (
            (
                str(population_row["source_facts_sha256"]),
                source_digest_records,
                "source facts",
            ),
            (
                str(population_row["public_basis_bundle_sha256"]),
                basis_digest_records,
                "public basis",
            ),
        ):
            previous = seen.setdefault(digest, record_id)
            if previous != record_id:
                raise PopulationSelectionError(
                    f"{name} bytes cannot represent multiple independent records"
                )
        claimed = _sha(
            row["provenance_record_sha256"],
            f"provenance row {index}.provenance_record_sha256",
        )
        unsigned = dict(row)
        del unsigned["provenance_record_sha256"]
        if claimed != stable_digest(unsigned):
            raise PopulationSelectionError("provenance record digest disagrees")
        provenance[record_id] = row
    if set(provenance) != set(population):
        raise PopulationSelectionError("provenance must cover the population exactly")
    return provenance


def _selection_seed(plan: Mapping[str, Any], random_value: str) -> str:
    lane = plan["lane"]
    if lane["protocol_id"] == SUPERPOWERS_PROTOCOL_ID:
        material = f"{lane['protocol_id']}\n{plan['artifacts']['population_sha256']}\n{random_value}\n"
    else:
        bundle = {
            key: plan["artifacts"][key]
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
        population_bundle_sha256 = stable_digest(bundle)
        material = "\n".join(
            [
                population_bundle_sha256,
                plan["beacon"]["chain_hash"],
                str(plan["beacon"]["pulse_or_round"]),
                random_value,
            ]
        )
    return hashlib.sha256(material.encode()).hexdigest()


def _domain_seed(master_seed: str, label: str) -> str:
    return hashlib.sha256(f"{master_seed}\n{label}\n".encode()).hexdigest()


def _superpowers_record_score(seed: str, record_id: str) -> str:
    domain = _domain_seed(seed, "record-choice-v2")
    return hashlib.sha256(f"{domain}\n{record_id}\n".encode()).hexdigest()


def _superpowers_repository_record_score(
    seed: str, cluster: str, repository: str, record_id: str
) -> str:
    domain = _domain_seed(seed, "independence-cluster-choice-v2")
    return hashlib.sha256(
        f"{domain}\n{cluster}\n{repository}\n{record_id}\n".encode()
    ).hexdigest()


def _superpowers_partition_score(seed: str, cluster: str, record_id: str) -> str:
    domain = _domain_seed(seed, "partition-selection-v2")
    return hashlib.sha256(f"{domain}\n{cluster}\n{record_id}\n".encode()).hexdigest()


def _anthropic_record_score(seed: str, row: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        "\n".join(
            [
                seed,
                str(row["source_lineage_id"]),
                str(row["repository_id"]),
                str(row["skill_path"]),
                str(row["public_basis_bundle_sha256"]),
            ]
        ).encode()
    ).hexdigest()


def _eligible_frame(
    rows: Sequence[dict[str, Any]], plan: Mapping[str, Any]
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    eligible = [row for row in rows if row["_final_decision"] == "eligible"]
    if not eligible:
        raise PopulationSelectionError("reviewed frame has no eligible records")
    repository_assignments: dict[str, tuple[str, str, str]] = {}
    lineage_clusters: dict[str, set[str]] = defaultdict(set)
    cluster_lineages: dict[str, set[str]] = defaultdict(set)
    lineage_strata: dict[str, set[str]] = defaultdict(set)
    cluster_strata: dict[str, set[str]] = defaultdict(set)
    by_stratum_repo: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in eligible:
        repository = str(row["repository_id"])
        assignment = (
            str(row["stratum"]),
            str(row["independence_cluster_id"]),
            str(row["source_lineage_id"]),
        )
        previous = repository_assignments.setdefault(repository, assignment)
        if previous != assignment:
            raise PopulationSelectionError(
                f"repository {repository} has inconsistent stratum or lineage"
            )
        lineage_clusters[str(row["source_lineage_id"])].add(
            str(row["independence_cluster_id"])
        )
        cluster_lineages[str(row["independence_cluster_id"])].add(
            str(row["source_lineage_id"])
        )
        lineage_strata[str(row["source_lineage_id"])].add(str(row["stratum"]))
        cluster_strata[str(row["independence_cluster_id"])].add(str(row["stratum"]))
        by_stratum_repo[str(row["stratum"])][repository].append(row)
    if any(len(clusters) != 1 for clusters in lineage_clusters.values()) or any(
        len(lineages) != 1 for lineages in cluster_lineages.values()
    ):
        raise PopulationSelectionError(
            "source lineage and independence cluster must map one-to-one"
        )
    if any(len(strata) != 1 for strata in lineage_strata.values()) or any(
        len(strata) != 1 for strata in cluster_strata.values()
    ):
        raise PopulationSelectionError(
            "source lineage and independence cluster must each occupy one stratum"
        )
    design_strata = {
        stratum for partition in plan["quotas"].values() for stratum in partition
    }
    if set(by_stratum_repo) != design_strata:
        raise PopulationSelectionError(
            "eligible strata must exactly match the sampling design"
        )
    if plan["lane"]["protocol_id"] == SUPERPOWERS_PROTOCOL_ID:
        actual = {
            stratum: len(
                {
                    str(row["independence_cluster_id"])
                    for rows_in_repository in repositories.values()
                    for row in rows_in_repository
                }
            )
            for stratum, repositories in by_stratum_repo.items()
        }
        short = {
            stratum: {"actual": actual.get(stratum, 0), "required": minimum}
            for stratum, minimum in _SUPERPOWERS_MINIMUM_ELIGIBLE_CLUSTERS.items()
            if actual.get(stratum, 0) < minimum
        }
        if short:
            raise PopulationSelectionError(
                f"Superpowers eligible-cluster minima are unmet: {short}"
            )
    return by_stratum_repo


def _repository_and_cluster_choices(
    frame: Mapping[str, Mapping[str, Sequence[dict[str, Any]]]],
    seed: str,
    plan: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    superpowers = plan["lane"]["protocol_id"] == SUPERPOWERS_PROTOCOL_ID
    choices: dict[str, dict[str, dict[str, Any]]] = {}
    for stratum, repositories in frame.items():
        repository_choices: dict[str, dict[str, Any]] = {}
        for repository, records in repositories.items():
            ordered_records_with_ids = sorted(
                (
                    (
                        _superpowers_record_score(seed, str(row["record_id"]))
                        if superpowers
                        else _anthropic_record_score(seed, row)
                    ),
                    str(row["record_id"]),
                    row,
                )
                for row in records
            )
            ordered_records = [
                (score, row) for score, _record_id, row in ordered_records_with_ids
            ]
            repository_choices[repository] = {
                "row": ordered_records_with_ids[0][2],
                "record_score": ordered_records[0][0],
                "record_order": ordered_records,
            }
        by_cluster: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        for repository, choice in repository_choices.items():
            by_cluster[str(choice["row"]["independence_cluster_id"])].append(
                (repository, choice)
            )
        choices[stratum] = {}
        for cluster, repository_rows in by_cluster.items():
            ordered_repositories = sorted(
                (
                    (
                        _superpowers_repository_record_score(
                            seed,
                            cluster,
                            repository,
                            str(choice["row"]["record_id"]),
                        )
                        if superpowers
                        else str(choice["record_score"])
                    ),
                    repository,
                    choice,
                )
                for repository, choice in repository_rows
            )
            choices[stratum][cluster] = {
                "repository": ordered_repositories[0][1],
                "choice": ordered_repositories[0][2],
                "repository_score": ordered_repositories[0][0],
                "repository_order": ordered_repositories,
            }
    return choices


def _allocate(
    plan: Mapping[str, Any],
    choices: Mapping[str, Mapping[str, Mapping[str, Any]]],
    seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    selected: list[dict[str, Any]] = []
    ranks: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    superpowers = plan["lane"]["protocol_id"] == SUPERPOWERS_PROTOCOL_ID
    for stratum in sorted(choices):
        cluster_choices = choices[stratum]
        pool_size = len(cluster_choices)
        total_quota = sum(
            int(plan["quotas"][partition].get(stratum, 0))
            for partition in plan["partition_order"]
        )
        required = total_quota * int(plan["minimum_pool_multiplier"])
        if pool_size < required:
            raise PopulationSelectionError(
                f"stratum {stratum} has {pool_size} clusters; at least {required} are required"
            )
        counts[stratum] = pool_size
        ordered_clusters = sorted(
            cluster_choices,
            key=lambda cluster: (
                (
                    _superpowers_partition_score(
                        seed,
                        cluster,
                        str(cluster_choices[cluster]["choice"]["row"]["record_id"]),
                    )
                    if superpowers
                    else str(cluster_choices[cluster]["repository_score"])
                ),
                cluster,
            ),
        )
        assignments: dict[str, tuple[str, int, int]] = {}
        cursor = 0
        for partition in plan["partition_order"]:
            quota = int(plan["quotas"][partition].get(stratum, 0))
            for partition_rank, cluster in enumerate(
                ordered_clusters[cursor : cursor + quota], start=1
            ):
                assignments[cluster] = (partition, partition_rank, quota)
            cursor += quota
        residual = [
            str(cluster_choices[cluster]["choice"]["row"]["record_id"])
            for cluster in ordered_clusters[cursor:]
        ]
        for stratum_rank, cluster in enumerate(ordered_clusters, start=1):
            cluster_choice = cluster_choices[cluster]
            repository_order = cluster_choice["repository_order"]
            record_order = cluster_choice["choice"]["record_order"]
            assignment = assignments.get(cluster)
            rank = {
                "cohort_id": plan["lane"]["cohort_id"],
                "stratum": stratum,
                "stratum_rank": stratum_rank,
                "independence_cluster_id": cluster,
                "cluster_order_score_sha256": (
                    _superpowers_partition_score(
                        seed,
                        cluster,
                        str(cluster_choice["choice"]["row"]["record_id"]),
                    )
                    if superpowers
                    else str(cluster_choice["repository_score"])
                ),
                "eligible_repository_count": len(repository_order),
                "repository_order": [
                    {
                        "repository_rank": index,
                        "repository_id": repository,
                        "repository_score_sha256": score,
                        "selected_as_cluster_representative": index == 1,
                        "record_order": [
                            {
                                "record_rank": record_index,
                                "record_id": row["record_id"],
                                "record_score_sha256": record_score,
                                "selected_as_repository_representative": record_index
                                == 1,
                            }
                            for record_index, (record_score, row) in enumerate(
                                choice["record_order"], start=1
                            )
                        ],
                    }
                    for index, (score, repository, choice) in enumerate(
                        repository_order, start=1
                    )
                ],
                "selected_record_id": cluster_choice["choice"]["row"]["record_id"],
                "partition": assignment[0] if assignment else None,
                "partition_rank": assignment[1] if assignment else None,
            }
            ranks.append(rank)
            if assignment is None:
                continue
            partition, partition_rank, quota = assignment
            row = cluster_choice["choice"]["row"]
            record_probability = Fraction(1, len(record_order))
            repository_probability = Fraction(1, len(repository_order))
            cluster_probability = Fraction(quota, pool_size)
            total_probability = (
                record_probability * repository_probability * cluster_probability
            )
            role = plan["partition_roles"][partition]
            primary = role == "primary_inference"
            fpc = (
                Fraction(pool_size - quota, pool_size - 1)
                if pool_size > 1
                else Fraction(0, 1)
            )
            selected.append(
                {
                    "record_id": row["record_id"],
                    "repository_id": row["repository_id"],
                    "source_lineage_id": row["source_lineage_id"],
                    "independence_cluster_id": cluster,
                    "cohort_id": plan["lane"]["cohort_id"],
                    "partition": partition,
                    "analysis_role": role,
                    "stratum": stratum,
                    "selection_rank": partition_rank,
                    "stratum_rank": stratum_rank,
                    "eligible_records_in_repository": len(record_order),
                    "eligible_repositories_in_cluster": len(repository_order),
                    "eligible_clusters_N_h": pool_size,
                    "selected_clusters_n_h": quota,
                    "record_within_repository_probability": _fraction(
                        record_probability
                    ),
                    "repository_within_cluster_probability": _fraction(
                        repository_probability
                    ),
                    "cluster_into_partition_probability": _fraction(
                        cluster_probability
                    ),
                    "total_record_inclusion_probability": _fraction(total_probability),
                    "analysis_weight": _fraction(Fraction(pool_size, quota))
                    if primary
                    else None,
                    "finite_population_correction": _fraction(fpc) if primary else None,
                    "replacement_policy": plan["replacement_policy"],
                    "frozen_replacement_order": residual
                    if plan["replacement_policy"]
                    == "next_frozen_rank_same_cohort_stratum"
                    else [],
                }
            )
    return selected, ranks, counts


def select_population_from_bytes(
    *,
    selection_plan_bytes: bytes,
    lane_protocol_bytes: bytes,
    query_plan_bytes: bytes | None = None,
    query_receipt_bytes: bytes,
    candidate_jsonl_bytes: bytes | None = None,
    population_jsonl_bytes: bytes,
    provenance_jsonl_bytes: bytes | None = None,
    eligibility_jsonl_bytes: bytes,
    reviewer_registry_bytes: bytes,
    review_verification_bytes: bytes,
    precommit_receipt_bytes: bytes,
    beacon_receipt_bytes: bytes,
    sampling_design_amendment_bytes: bytes | None = None,
    power_receipt_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Validate the prospective frame and fail closed at authentication.

    V2 receipts contain digests and self-asserted verification booleans but no
    authenticatable signature/public-key material. Consequently this public
    boundary must never emit a selected population. It returns a typed blocked
    receipt after all available structural and scientific checks succeed.
    """

    protocol = _mapping(
        _parse_json_bytes(lane_protocol_bytes, "lane protocol"), "lane protocol"
    )
    raw_plan = _mapping(
        _parse_json_bytes(selection_plan_bytes, "selection plan"), "selection plan"
    )
    required_inputs = {
        "query_plan_bytes": query_plan_bytes,
        "candidate_jsonl_bytes": candidate_jsonl_bytes,
        "provenance_jsonl_bytes": provenance_jsonl_bytes,
        "sampling_design_amendment_bytes": sampling_design_amendment_bytes,
        "power_receipt_bytes": power_receipt_bytes,
    }
    missing = sorted(key for key, value in required_inputs.items() if value is None)
    if missing:
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "blocker_code": "selection_inputs_missing",
            "missing_inputs": missing,
            "selection_plan_sha256": _sha256_bytes(selection_plan_bytes),
            "lane_protocol_sha256": _sha256_bytes(lane_protocol_bytes),
            "selected_count": 0,
        }
        receipt["receipt_sha256"] = stable_digest(receipt)
        return receipt
    assert query_plan_bytes is not None
    assert candidate_jsonl_bytes is not None
    assert provenance_jsonl_bytes is not None
    assert sampling_design_amendment_bytes is not None
    assert power_receipt_bytes is not None
    plan = _validate_plan(
        raw_plan,
        protocol,
        _sha256_bytes(lane_protocol_bytes),
    )
    actual_artifacts = {
        "query_plan_sha256": _sha256_bytes(query_plan_bytes),
        "population_sha256": _sha256_bytes(population_jsonl_bytes),
        "query_receipt_sha256": _sha256_bytes(query_receipt_bytes),
        "candidate_records_sha256": _sha256_bytes(candidate_jsonl_bytes),
        "provenance_bundle_sha256": _sha256_bytes(provenance_jsonl_bytes),
        "eligibility_ledger_sha256": _sha256_bytes(eligibility_jsonl_bytes),
        "reviewer_registry_sha256": _sha256_bytes(reviewer_registry_bytes),
        "review_verification_sha256": _sha256_bytes(review_verification_bytes),
        "sampling_design_amendment_sha256": _sha256_bytes(
            sampling_design_amendment_bytes
        ),
        "power_receipt_sha256": _sha256_bytes(power_receipt_bytes),
        "selection_code_sha256": _sha256_bytes(Path(__file__).read_bytes()),
    }
    if actual_artifacts != plan["artifacts"]:
        mismatches = sorted(
            key
            for key, value in actual_artifacts.items()
            if plan["artifacts"].get(key) != value
        )
        raise PopulationSelectionError(
            f"artifact bytes do not match plan: {mismatches}"
        )
    protocol_id = str(plan["lane"]["protocol_id"])
    query_plan = _validate_query_plan(
        _parse_json_bytes(query_plan_bytes, "query plan"),
        protocol_id=protocol_id,
        query_plan_sha256=actual_artifacts["query_plan_sha256"],
    )
    _validate_query_receipt(
        _parse_json_bytes(query_receipt_bytes, "query receipt"),
        query_plan_sha256=actual_artifacts["query_plan_sha256"],
        candidate_bytes=candidate_jsonl_bytes,
    )
    candidates = _validate_candidates(
        _parse_jsonl_bytes(candidate_jsonl_bytes, "candidate records")
    )
    precommit = _validate_precommit(
        _parse_json_bytes(precommit_receipt_bytes, "precommit receipt"), plan
    )
    _validate_beacon(
        _parse_json_bytes(beacon_receipt_bytes, "beacon receipt"), plan, precommit
    )
    if _sha256_bytes(precommit_receipt_bytes) != plan["precommit"]["receipt_sha256"]:
        raise PopulationSelectionError("precommit receipt bytes do not match plan")
    if _sha256_bytes(beacon_receipt_bytes) != plan["beacon"]["receipt_sha256"]:
        raise PopulationSelectionError("beacon receipt bytes do not match plan")
    registry_value = _validate_registry(
        _parse_json_bytes(reviewer_registry_bytes, "reviewer registry"), plan
    )
    registry = {
        str(row["reviewer_id"]): dict(row) for row in registry_value["reviewers"]
    }
    rows, attestations = _validate_rows(
        _parse_jsonl_bytes(eligibility_jsonl_bytes, "eligibility ledger"),
        plan,
        registry,
    )
    _validate_verification_bundle(
        _parse_json_bytes(review_verification_bytes, "review verification bundle"),
        plan,
        registry,
        attestations,
    )
    population = _validate_population(
        _parse_jsonl_bytes(population_jsonl_bytes, "population"), rows
    )
    eligibility_by_id = {str(row["record_id"]): row for row in rows}
    _validate_provenance(
        _parse_jsonl_bytes(provenance_jsonl_bytes, "provenance bundle"),
        population=population,
        eligibility_rows=eligibility_by_id,
        candidates=candidates,
        plan=plan,
        query_plan=query_plan,
        query_plan_sha256=actual_artifacts["query_plan_sha256"],
        query_receipt_sha256=actual_artifacts["query_receipt_sha256"],
    )
    for name, raw in (
        ("sampling design amendment", sampling_design_amendment_bytes),
        ("prospective power receipt", power_receipt_bytes),
    ):
        pending = _mapping(_parse_json_bytes(raw, name), name)
        if pending.get("schema_version") != SCHEMA_VERSION:
            raise PopulationSelectionError(f"{name} schema_version must be 2")
    committed_at = _utc(
        plan["public_git_freeze"]["committed_at_utc"], "Git freeze time"
    )
    for attestation in attestations.values():
        if _utc(attestation["reviewed_at_utc"], "reviewed_at_utc") > committed_at:
            raise PopulationSelectionError(
                "review occurred after public selection freeze"
            )
    frame = _eligible_frame(rows, plan)
    counts = {
        stratum: len(
            {
                str(row["independence_cluster_id"])
                for repository_rows in repositories.values()
                for row in repository_rows
            }
        )
        for stratum, repositories in frame.items()
    }
    blockers = [
        "authenticated_precommit_verifier_receipt_missing",
        "authenticated_beacon_provider_response_missing",
        "authenticated_reviewer_signatures_missing",
    ]
    if protocol_id == ANTHROPIC_PROTOCOL_ID:
        blockers.extend(
            [
                "authenticated_sampling_design_amendment_missing",
                "authenticated_prospective_power_receipt_missing",
            ]
        )
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "selection_id": plan["selection_id"],
        "status": "blocked",
        "blocker_code": "authenticated_verification_boundary_unavailable",
        "blockers": blockers,
        "selection_plan_sha256": _sha256_bytes(selection_plan_bytes),
        "selection_commitment_sha256": plan["precommit"]["commitment_sha256"],
        "lane": plan["lane"],
        "artifact_digests": actual_artifacts,
        "public_git_freeze": plan["public_git_freeze"],
        "precommit_verifier": plan["verifier_policy"]["precommit"],
        "beacon_verifier": plan["verifier_policy"]["beacon"],
        "review_signature_verifier": plan["reviewer_policy"]["verifier"],
        "external_signature_verification": "not_authenticated_selection_forbidden",
        "partition_order": plan["partition_order"],
        "partition_roles": plan["partition_roles"],
        "quotas": plan["quotas"],
        "weighting": plan["weighting"],
        "replacement_policy": plan["replacement_policy"],
        "frame_independent_cluster_counts": counts,
        "rank_ledger": [],
        "selected_count": 0,
        "selected": [],
        "limitations": [
            "V2 verifier receipts do not carry enough cryptographic material for independent authentication.",
            "No population selection, preview, approval, model call, or behavioral claim is authorized by this blocked receipt.",
        ],
    }
    receipt["receipt_sha256"] = stable_digest(receipt)
    return receipt


def _absolute_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _validate_path_ancestors(
    path: Path, *, field: str, require_private_parent: bool
) -> Path:
    absolute = _absolute_without_symlink_resolution(path)
    for ancestor in reversed(absolute.parents):
        try:
            metadata = ancestor.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise PopulationSelectionError(
                f"{field} ancestor must be a non-symlink directory"
            )
    if require_private_parent:
        try:
            parent_metadata = absolute.parent.lstat()
        except FileNotFoundError as exc:
            raise PopulationSelectionError(
                f"{field} private parent must already exist with mode 0700"
            ) from exc
        if (
            stat.S_ISLNK(parent_metadata.st_mode)
            or not stat.S_ISDIR(parent_metadata.st_mode)
            or parent_metadata.st_mode & 0o077
        ):
            raise PopulationSelectionError(
                f"{field} private parent must be a non-symlink directory with mode 0700"
            )
    return absolute


def _read_regular_file(path: Path, *, field: str, private: bool) -> bytes:
    absolute = _validate_path_ancestors(
        path, field=field, require_private_parent=private
    )
    metadata = absolute.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise PopulationSelectionError(f"{field} must be a regular non-symlink file")
    if private and metadata.st_mode & 0o077:
        raise PopulationSelectionError(f"{field} must not be group/world accessible")
    return absolute.read_bytes()


def _write_private_atomic(path: Path, payload: bytes) -> None:
    absolute = _absolute_without_symlink_resolution(path)
    _validate_path_ancestors(absolute, field="output", require_private_parent=False)
    parent = absolute.parent
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _validate_path_ancestors(absolute, field="output", require_private_parent=True)
    if absolute.is_symlink() or absolute.exists():
        raise PopulationSelectionError(f"refusing to overwrite {absolute}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, absolute, follow_symlinks=False)
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select a locked public population")
    parser.add_argument("selection_plan", type=Path)
    parser.add_argument("lane_protocol", type=Path)
    parser.add_argument("query_plan", type=Path)
    parser.add_argument("query_receipt", type=Path)
    parser.add_argument("candidate_records", type=Path)
    parser.add_argument("population_records", type=Path)
    parser.add_argument("provenance_bundle", type=Path)
    parser.add_argument("eligibility_ledger", type=Path)
    parser.add_argument("reviewer_registry", type=Path)
    parser.add_argument("review_verification", type=Path)
    parser.add_argument("precommit_receipt", type=Path)
    parser.add_argument("beacon_receipt", type=Path)
    parser.add_argument("sampling_design_amendment", type=Path)
    parser.add_argument("power_receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = select_population_from_bytes(
            selection_plan_bytes=_read_regular_file(
                args.selection_plan, field="selection plan", private=True
            ),
            lane_protocol_bytes=_read_regular_file(
                args.lane_protocol, field="lane protocol", private=False
            ),
            query_plan_bytes=_read_regular_file(
                args.query_plan, field="query plan", private=False
            ),
            query_receipt_bytes=_read_regular_file(
                args.query_receipt, field="query receipt", private=True
            ),
            candidate_jsonl_bytes=_read_regular_file(
                args.candidate_records, field="candidate records", private=True
            ),
            population_jsonl_bytes=_read_regular_file(
                args.population_records, field="population records", private=True
            ),
            provenance_jsonl_bytes=_read_regular_file(
                args.provenance_bundle, field="provenance bundle", private=True
            ),
            eligibility_jsonl_bytes=_read_regular_file(
                args.eligibility_ledger, field="eligibility ledger", private=True
            ),
            reviewer_registry_bytes=_read_regular_file(
                args.reviewer_registry, field="reviewer registry", private=True
            ),
            review_verification_bytes=_read_regular_file(
                args.review_verification, field="review verification", private=True
            ),
            precommit_receipt_bytes=_read_regular_file(
                args.precommit_receipt, field="precommit receipt", private=True
            ),
            beacon_receipt_bytes=_read_regular_file(
                args.beacon_receipt, field="beacon receipt", private=True
            ),
            sampling_design_amendment_bytes=_read_regular_file(
                args.sampling_design_amendment,
                field="sampling design amendment",
                private=True,
            ),
            power_receipt_bytes=_read_regular_file(
                args.power_receipt, field="power receipt", private=True
            ),
        )
        _write_private_atomic(
            args.output,
            (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
    except (OSError, PopulationSelectionError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {"status": receipt["status"], "receipt_sha256": receipt["receipt_sha256"]}
        )
    )
    return 2 if receipt["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
