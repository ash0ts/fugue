"""Select the preregistered Vercel React population without weakening its design.

This adapter is intentionally lane-specific.  It consumes a frozen public
population plus a treatment-blind, two-reviewer ledger and implements the V2
protocol's global lineage-to-slot allocation, owner caps, and frozen reserve
order.  It does not discover candidates, author tasks, inspect treatment
outputs, or authorize an experiment.

The optimization is an exact min-cost bipartite flow with deterministic
branching for the two owner constraints.  Any capacity, provenance, review,
randomness, or digest disagreement fails closed.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import heapq
import json
import os
import platform
import re
import stat
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
except ImportError:  # pragma: no cover - exercised by the typed blocked path.
    InvalidSignature = ValueError  # type: ignore[assignment,misc]
    Ed25519PublicKey = None  # type: ignore[assignment,misc]

SCHEMA_VERSION = 2
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_RANDOM_HEX = re.compile(r"^(?:[0-9a-f]{64}|[0-9a-f]{128})$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,255}$")
_ISO_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_PULL_URL = re.compile(
    r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/pull/[1-9][0-9]*$"
)

PRIMARY_FAMILIES = (
    "server_action_authorization",
    "rsc_serialization",
)
CONTROL_FAMILIES = (
    "dom_batching",
    "large_array_iteration",
    "hook_timing",
    "event_handler_reference",
)
FAMILIES = PRIMARY_FAMILIES + CONTROL_FAMILIES
PARTITION_ORDER = ("development", "target_holdout", "safety_control")
FAMILY_ORDER_BY_PARTITION: dict[str, tuple[str, ...]] = {
    "development": PRIMARY_FAMILIES,
    "target_holdout": PRIMARY_FAMILIES,
    "safety_control": CONTROL_FAMILIES,
}
EXPECTED_QUOTAS: dict[str, dict[str, int]] = {
    "development": {
        "server_action_authorization": 16,
        "rsc_serialization": 16,
    },
    "target_holdout": {
        "server_action_authorization": 96,
        "rsc_serialization": 96,
    },
    "safety_control": {
        "dom_batching": 16,
        "large_array_iteration": 16,
        "hook_timing": 16,
        "event_handler_reference": 16,
    },
}
MINIMUM_ELIGIBLE_LINEAGES: dict[str, int] = {
    "server_action_authorization": 140,
    "rsc_serialization": 140,
    "dom_batching": 20,
    "large_array_iteration": 20,
    "hook_timing": 20,
    "event_handler_reference": 20,
}
MINIMUM_UNIQUE_LINEAGES = 360
MAX_SELECTED_PER_OWNER = 4
MAX_TARGET_HOLDOUT_PER_OWNER = 2
RESERVE_NUMERATOR = 1
RESERVE_DENOMINATOR = 4

PROTOCOL_ID = "vercel-react-best-practices-conference-sampling-frame-protocol-v2"
QUERY_PLAN_ID = "vercel-react-best-practices-conference-query-plan-v2"
EXPECTED_PROTOCOL_ARTIFACT_DIGEST = (
    "5005f93222ee84569b2b47fdca90ea6bb37650de7c8af8ee278ce8d10e470027"
)
EXPECTED_QUERY_PLAN_ARTIFACT_DIGEST = (
    "7a653385ebbd50a7fff15ce0bb806e1424ce04042ef609a2a8cd3c0185f44844"
)
BASELINE_COMMIT = "ac6a79af08f6d32c34ee03c829824990f3de0a6d"
CANDIDATE_COMMIT = "20987af2f1bc17857b55e7758af8bed91c364ff5"
RECORD_WINDOW_START_UTC = "2023-01-01T00:00:00Z"
RECORD_CUTOFF_UTC = "2026-01-17T23:59:59Z"

AUTHENTICATION_ALGORITHM = "ed25519-detached-signature-v1"
EXPECTED_BOUNDS = {
    "max_plan_bytes": 1_000_000,
    "max_population_bytes": 64_000_000,
    "max_review_ledger_bytes": 128_000_000,
    "max_lineage_ledger_bytes": 64_000_000,
    "max_query_plan_bytes": 4_000_000,
    "max_receipt_bytes": 2_000_000,
    "max_population_records": 20_000,
    "max_graph_edges": 100_000,
    "max_search_states": 2_000,
    "max_pending_search_states": 4_000,
    "max_replacement_alternatives_per_slot": 64,
}

_SOURCE_RECEIPT_FIELDS = (
    "repository_receipt_sha256",
    "pull_request_receipt_sha256",
    "source_receipt_sha256",
    "stack_receipt_sha256",
)
_ALLOWED_EXCLUSION_REASONS = {
    "not_public",
    "archived_disabled_fork_or_mirror",
    "missing_or_unreviewable_license",
    "outside_treatment_preperiod",
    "not_a_merged_pull_request",
    "unreachable_base_head_merge_or_tree",
    "not_typescript_nextjs_react",
    "missing_supported_dependency_lock",
    "unsafe_or_nonhermetic_source",
    "skill_or_campaign_contamination",
    "no_exact_family_consensus",
    "duplicate_public_record",
    "duplicate_code_lineage",
    "public_test_not_authorable_blind",
    "host_verifier_not_authorable_independently",
    "base_did_not_fail",
    "gold_did_not_pass",
    "qualification_runtime_or_receipt_invalid",
}

_PLAN_KEYS = {
    "schema_version",
    "selection_id",
    "protocol_id",
    "protocol_sha256",
    "population_sha256",
    "review_ledger_sha256",
    "lineage_ledger_sha256",
    "query_plan_sha256",
    "calibration_receipt_sha256",
    "role_separation_receipt_sha256",
    "precommit_receipt_sha256",
    "verifier_lock_sha256",
    "randomness_receipt_sha256",
    "selection_code_sha256",
    "solver_runtime_sha256",
    "acquisition_candidates_sha256",
    "repository_receipts_sha256",
    "pull_request_receipts_sha256",
    "source_receipts_sha256",
    "stack_receipts_sha256",
    "calibration_ledger_sha256",
    "review_authentication_sha256",
    "source_authentication_sha256",
    "beacon_response_sha256",
    "beacon_authentication_sha256",
    "power_receipt_sha256",
    "population_frozen_at_utc",
    "beacon_target",
    "precommit_commitment_sha256",
    "partition_order",
    "family_order_by_partition",
    "slot_manifest_sha256",
    "quotas",
    "owner_caps",
    "reserve_fraction",
    "minimum_eligible_lineages",
    "minimum_unique_lineages",
    "bounds",
}
_POPULATION_KEYS = {
    "schema_version",
    "record_id",
    "repository_id",
    "owner_id",
    "canonical_pull_request_url",
    "acquisition_candidate_sha256",
    "pre_treatment",
    "source_facts_sha256",
    *_SOURCE_RECEIPT_FIELDS,
    "source_record_sha256",
}
_LINEAGE_ROW_KEYS = {
    "schema_version",
    "record_id",
    "repository_id",
    "owner_id",
    "canonical_roots",
    "lineage_id",
    "lineage_row_sha256",
}
_ACQUISITION_ROW_KEYS = {
    "schema_version",
    "record_id",
    "query_ids",
    "pull_request_node_id",
    "canonical_pull_request_url",
    "repository_node_id",
    "owner_node_id",
    "merged_at_utc",
    "acquisition_one",
    "acquisition_two",
    "acquisition_row_sha256",
}
_ACQUISITION_OBSERVATION_KEYS = {
    "lane_id",
    "credential_profile_id",
    "credential_env_name",
    "credential_present_without_value",
    "collector_source_commit",
    "collector_source_tree",
    "collector_sha256",
    "compiler_sha256",
    "requested_url",
    "final_url",
    "redirect_chain",
    "request_started_at_utc",
    "request_completed_at_utc",
    "github_api_version",
    "response_date",
    "x_github_request_id",
    "x_ratelimit_limit",
    "x_ratelimit_remaining",
    "x_ratelimit_reset",
    "x_ratelimit_resource",
    "x_ratelimit_used",
    "request_url_sha256",
    "headers_sha256",
    "body_sha256",
    "repository_public_visibility_receipt_sha256",
    "terminal_shard_manifest_sha256",
    "ordered_identity_sha256",
    "candidate_identity_sha256",
    "checkpoint_sha256",
}
_REPOSITORY_DETAIL_KEYS = {
    "schema_version",
    "record_id",
    "canonical_repository_url",
    "repository_node_id",
    "owner_node_id",
    "owner_login",
    "visibility",
    "private",
    "fork_network_root_sha256",
    "template_root_sha256",
    "fork",
    "mirror_url",
    "archived",
    "disabled",
    "default_branch",
    "license_spdx",
    "license_file_sha256",
    "api_response_sha256",
    "etag",
    "fetched_at_utc",
    "receipt_sha256",
}
_PULL_REQUEST_DETAIL_KEYS = {
    "schema_version",
    "record_id",
    "pull_request_url",
    "pull_request_node_id",
    "pull_request_number",
    "merged_at_utc",
    "base_repository_node_id",
    "base_commit_sha",
    "base_tree_sha",
    "head_repository_node_id",
    "head_commit_sha",
    "head_tree_sha",
    "merge_commit_sha",
    "gold_tree_sha",
    "api_response_sha256",
    "diff_sha256",
    "patch_sha256",
    "changed_files_manifest_sha256",
    "review_manifest_sha256",
    "linked_issue_manifest_sha256",
    "receipt_sha256",
}
_SOURCE_DETAIL_KEYS = {
    "schema_version",
    "record_id",
    "base_archive_sha256",
    "gold_archive_sha256",
    "base_tree_listing_sha256",
    "gold_tree_listing_sha256",
    "archive_file_manifest_sha256",
    "source_fingerprint_sha256",
    "submodule_check",
    "unsafe_link_check",
    "path_traversal_check",
    "secret_path_check",
    "generated_dependency_tree_check",
    "source_size_bytes",
    "receipt_sha256",
}
_STACK_DETAIL_KEYS = {
    "schema_version",
    "record_id",
    "package_boundary_path",
    "package_manifest_path",
    "package_manifest_sha256",
    "dependency_lock_path",
    "dependency_lock_sha256",
    "lock_format",
    "react_declared_requirement",
    "react_resolved_version",
    "next_declared_requirement",
    "next_resolved_version",
    "typescript_target_paths",
    "target_file_manifest_sha256",
    "parser_runtime_sha256",
    "receipt_sha256",
}
_CALIBRATION_ROW_KEYS = {
    "schema_version",
    "calibration_record_id",
    "family",
    "expected_acceptable",
    "reviewer_decisions",
    "row_sha256",
}
_AUTHENTICATION_KEYS = {
    "schema_version",
    "purpose",
    "algorithm",
    "key_id_sha256",
    "public_key_hex",
    "payload_sha256",
    "signature_hex",
    "authenticated_at_utc",
    "receipt_sha256",
}
_REVIEW_AUTHENTICATION_KEYS = {
    "schema_version",
    "purpose",
    "algorithm",
    "reviewer_signatures",
    "governance_authentication",
    "receipt_sha256",
}
_REVIEW_SIGNATURE_KEYS = {
    "reviewer_id_hash",
    "role",
    "public_key_hex",
    "key_id_sha256",
    "decision_manifest_sha256",
    "signature_hex",
}
_POWER_RECEIPT_KEYS = {
    "schema_version",
    "analysis_code_sha256",
    "runtime_sha256",
    "simulation_seed",
    "simulation_repetitions",
    "assumption_grid",
    "family_sample_sizes",
    "scenario_power_results",
    "minimum_observed_power_micros",
    "supported_effect_boundary_micros",
    "safety_control_power_results",
    "authentication",
    "receipt_sha256",
}
_BEACON_RESPONSE_KEYS = {
    "schema_version",
    "provider",
    "canonical_endpoint",
    "pulse_or_round",
    "published_at_utc",
    "random_value",
    "signature_sha256",
    "verification_material_sha256",
    "raw_response_base64",
    "raw_response_sha256",
    "receipt_sha256",
}
_REVIEW_ROW_KEYS = {
    "schema_version",
    "record_id",
    "source_record_sha256",
    "reviewer_a",
    "reviewer_b",
    "adjudication",
    "exclusion_reasons",
    "review_row_sha256",
}
_REVIEWER_KEYS = {
    "reviewer_id_hash",
    "role",
    "record_id",
    "source_record_sha256",
    "lineage_row_sha256",
    "decision",
    "family",
    "lineage_id",
    "reason_codes",
    "natural_repair_confirmed",
    "task_authorable_without_gold",
    "decision_timestamp_utc",
    "baseline_candidate_skills_blinded",
    "treatment_arm_labels_blinded",
    "selection_seed_and_score_blinded",
    "partition_blinded",
    "agent_judge_outputs_blinded",
    "other_reviewer_decision_blinded",
    "calibration_receipt_sha256",
    "role_separation_receipt_sha256",
    "decision_sha256",
}
_ADJUDICATION_KEYS = _REVIEWER_KEYS | {"required", "reason"}
_CALIBRATION_KEYS = {
    "schema_version",
    "calibration_id",
    "completed_at_utc",
    "reviewer_id_hashes",
    "family_examples",
    "record_count",
    "cohens_kappa_micros",
    "critical_false_inclusions",
    "treatment_outputs_blinded",
    "receipt_sha256",
}
_ROLE_SEPARATION_KEYS = {
    "schema_version",
    "eligibility_reviewer_id_hashes",
    "adjudicator_id_hashes",
    "public_task_author_id_hashes",
    "public_test_author_id_hashes",
    "host_verifier_author_id_hashes",
    "pairwise_disjoint",
    "receipt_sha256",
}
_BEACON_TARGET_KEYS = {"provider", "pulse_or_round", "canonical_endpoint"}
_PRECOMMIT_KEYS = {
    "schema_version",
    "selection_id",
    "commitment_sha256",
    "commitment_artifact_sha256",
    "committed_at_utc",
    "public_commit_url",
    "public_commit_sha",
    "publication_observed_at_utc",
    "publication_evidence_sha256",
    "publication_authentication",
    "beacon_target",
    "receipt_sha256",
}
_VERIFIER_LOCK_KEYS = {
    "schema_version",
    "provider",
    "verifier_id",
    "verifier_code_sha256",
    "canonical_endpoint",
    "trust_root_kind",
    "trust_root_sha256",
    "signature_algorithm",
    "locked_at_utc",
    "receipt_sha256",
}
_SOLVER_RUNTIME_KEYS = {
    "schema_version",
    "runtime_id",
    "python_implementation",
    "python_version",
    "selector_algorithm",
    "receipt_sha256",
}
_RANDOMNESS_KEYS = {
    "schema_version",
    "provider",
    "canonical_endpoint",
    "pulse_or_round",
    "published_at_utc",
    "retrieved_at_utc",
    "verified_at_utc",
    "random_value",
    "signature_sha256",
    "signature_verified",
    "verification_material_sha256",
    "response_sha256",
    "commitment_sha256",
    "verifier_lock_sha256",
    "verifier_code_sha256",
    "trust_root_sha256",
    "population_sha256",
    "review_ledger_sha256",
    "lineage_ledger_sha256",
    "query_plan_sha256",
    "calibration_receipt_sha256",
    "role_separation_receipt_sha256",
    "selection_code_sha256",
    "receipt_sha256",
}


class VercelPopulationSelectionError(ValueError):
    """Raised when the frozen Vercel frame cannot be selected as preregistered."""


@dataclass(frozen=True)
class PopulationRecord:
    record_id: str
    repository_id: str
    owner_id: str
    canonical_pull_request_url: str
    acquisition_candidate_sha256: str
    source_facts_sha256: str
    repository_receipt_sha256: str
    pull_request_receipt_sha256: str
    source_receipt_sha256: str
    stack_receipt_sha256: str
    source_record_sha256: str


@dataclass(frozen=True)
class SelectionArtifactBytes:
    """Exact bytes whose identities are recomputed inside selection."""

    plan: bytes
    protocol: bytes
    population: bytes
    review_ledger: bytes
    lineage_ledger: bytes
    query_plan: bytes
    calibration_receipt: bytes
    role_separation_receipt: bytes
    precommit_receipt: bytes
    verifier_lock: bytes
    randomness_receipt: bytes
    solver_runtime: bytes
    selection_code: bytes
    acquisition_candidates: bytes = b""
    repository_receipts: bytes = b""
    pull_request_receipts: bytes = b""
    source_receipts: bytes = b""
    stack_receipts: bytes = b""
    calibration_ledger: bytes = b""
    review_authentication: bytes = b""
    source_authentication: bytes = b""
    beacon_response: bytes = b""
    beacon_authentication: bytes = b""
    power_receipt: bytes = b""


@dataclass(frozen=True)
class Candidate:
    record: PopulationRecord
    lineage_id: str
    family: str
    score_sha256: str

    @property
    def key(self) -> tuple[str, str]:
        return self.lineage_id, self.family


@dataclass(frozen=True)
class Assignment:
    candidate: Candidate
    category: str

    @property
    def key(self) -> tuple[str, str]:
        return self.candidate.lineage_id, self.category


@dataclass
class _FlowEdge:
    to: int
    reverse: int
    capacity: int
    cost: int


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def stable_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VercelPopulationSelectionError(f"{field} must be an object")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], field: str) -> None:
    if set(value) != expected:
        raise VercelPopulationSelectionError(
            f"{field} has invalid keys; "
            f"missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )


def _safe_id(value: object, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise VercelPopulationSelectionError(f"{field} must be a safe identifier")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise VercelPopulationSelectionError(f"{field} must be a lowercase SHA-256")
    return value


def _utc(value: object, field: str) -> datetime:
    if not isinstance(value, str) or _ISO_UTC.fullmatch(value) is None:
        raise VercelPopulationSelectionError(f"{field} must use YYYY-MM-DDTHH:MM:SSZ")
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise VercelPopulationSelectionError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise VercelPopulationSelectionError(f"{field} must be a nonnegative integer")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise VercelPopulationSelectionError(f"duplicate JSON key {key}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> None:
    raise VercelPopulationSelectionError(f"non-finite JSON number {value} is forbidden")


def _bounded_bytes(value: object, field: str, maximum: int) -> bytes:
    if not isinstance(value, bytes):
        raise VercelPopulationSelectionError(f"{field} must be supplied as bytes")
    if not value or len(value) > maximum:
        raise VercelPopulationSelectionError(f"{field} must contain 1..{maximum} bytes")
    return value


def _json_bytes(value: bytes, field: str, maximum: int) -> dict[str, Any]:
    raw = _bounded_bytes(value, field, maximum)
    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VercelPopulationSelectionError(f"cannot parse {field}: {exc}") from exc
    return dict(_mapping(parsed, field))


def _jsonl_bytes(value: bytes, field: str, maximum: int) -> list[dict[str, Any]]:
    raw = _bounded_bytes(value, field, maximum)
    rows: list[dict[str, Any]] = []
    try:
        for line in raw.splitlines():
            if not line.strip():
                continue
            parsed = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite,
            )
            rows.append(dict(_mapping(parsed, field)))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VercelPopulationSelectionError(f"cannot parse {field}: {exc}") from exc
    return rows


def _blocked_receipt(
    selection_id: str,
    *,
    blockers: Sequence[str],
    plan_sha256: str,
) -> dict[str, Any]:
    """Return a typed, assignment-free stop artifact.

    A blocked artifact deliberately has no seed, selected rows, objective, or
    replacement order.  It is safe to persist as evidence that preparation
    stopped before randomization.
    """

    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "selection_id": selection_id,
        "status": "blocked_authenticated_evidence_required",
        "blockers": sorted(set(blockers)),
        "plan_sha256": plan_sha256,
        "assignment_emitted": False,
        "claim_boundary": (
            "No population selection or experiment claim exists because the "
            "authenticated preparation boundary did not pass."
        ),
    }
    value["receipt_sha256"] = stable_digest(value)
    return value


def _public_key_bytes(value: object, field: str) -> bytes:
    if not isinstance(value, str) or len(value) != 64:
        raise VercelPopulationSelectionError(
            f"{field} must be a 32-byte lowercase Ed25519 public key"
        )
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise VercelPopulationSelectionError(
            f"{field} must be lowercase hexadecimal"
        ) from exc
    if value != value.lower():
        raise VercelPopulationSelectionError(
            f"{field} must be lowercase hexadecimal"
        )
    return raw


def _authentication_message(
    purpose: str,
    payload: Mapping[str, Any],
    *,
    authenticated_at_utc: str | None = None,
) -> bytes:
    return (
        b"fugue-vercel-authenticated-envelope-v2\x00"
        + _canonical_json(
            {
                "purpose": purpose,
                "payload": payload,
                "authenticated_at_utc": authenticated_at_utc,
            }
        )
    )


def _validate_authentication(
    value: object,
    *,
    purpose: str,
    payload: Mapping[str, Any],
    expected_key_id: str,
) -> dict[str, Any]:
    receipt = _self_digest_receipt(
        value,
        field=f"{purpose} authentication",
        expected_keys=_AUTHENTICATION_KEYS,
    )
    if receipt["purpose"] != purpose:
        raise VercelPopulationSelectionError(
            f"{purpose} authentication has the wrong purpose"
        )
    if receipt["algorithm"] != AUTHENTICATION_ALGORITHM:
        raise VercelPopulationSelectionError(
            f"{purpose} authentication algorithm is unsupported"
        )
    key_id = _sha256(
        receipt["key_id_sha256"], f"{purpose} authentication.key_id_sha256"
    )
    public_key = _public_key_bytes(
        receipt["public_key_hex"], f"{purpose} authentication.public_key_hex"
    )
    if key_id != hashlib.sha256(public_key).hexdigest() or key_id != expected_key_id:
        raise VercelPopulationSelectionError(
            f"{purpose} authentication public key is not the protocol trust root"
        )
    if receipt["payload_sha256"] != stable_digest(payload):
        raise VercelPopulationSelectionError(
            f"{purpose} authentication payload digest differs"
        )
    signature = receipt["signature_hex"]
    if not isinstance(signature, str) or len(signature) != 128:
        raise VercelPopulationSelectionError(
            f"{purpose} authentication signature must be 64-byte Ed25519 hex"
        )
    try:
        signature_bytes = bytes.fromhex(signature)
    except ValueError as exc:
        raise VercelPopulationSelectionError(
            f"{purpose} authentication signature must be lowercase hexadecimal"
        ) from exc
    if signature != signature.lower():
        raise VercelPopulationSelectionError(
            f"{purpose} authentication signature must be lowercase hexadecimal"
        )
    if Ed25519PublicKey is None:
        raise VercelPopulationSelectionError(
            "Ed25519 verification runtime is unavailable"
        )
    authenticated_at = receipt["authenticated_at_utc"]
    if not isinstance(authenticated_at, str):
        raise VercelPopulationSelectionError(
            f"{purpose} authentication.authenticated_at_utc must be text"
        )
    _utc(
        authenticated_at,
        f"{purpose} authentication.authenticated_at_utc",
    )
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            signature_bytes,
            _authentication_message(
                purpose,
                payload,
                authenticated_at_utc=authenticated_at,
            ),
        )
    except (InvalidSignature, ValueError) as exc:
        raise VercelPopulationSelectionError(
            f"{purpose} authentication signature is invalid"
        ) from exc
    return receipt


def _signed_row(
    value: object,
    *,
    field: str,
    expected_keys: set[str],
    digest_field: str = "receipt_sha256",
) -> dict[str, Any]:
    row = dict(_mapping(value, field))
    _exact_keys(row, expected_keys, field)
    if row["schema_version"] != SCHEMA_VERSION:
        raise VercelPopulationSelectionError(f"{field} schema_version must be 2")
    digest = _sha256(row[digest_field], f"{field}.{digest_field}")
    unsigned = dict(row)
    del unsigned[digest_field]
    if digest != stable_digest(unsigned):
        raise VercelPopulationSelectionError(f"{field} digest does not match")
    return row


def _validate_exact_protocol_and_query(  # noqa: C901 - one exact policy boundary.
    protocol: Mapping[str, Any], query_plan: Mapping[str, Any]
) -> tuple[dict[str, set[str]], dict[str, str] | None]:
    if (
        protocol.get("schema_version") != SCHEMA_VERSION
        or protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("status") != "prospective_not_yet_sampled"
        or protocol.get("artifact_digest") != EXPECTED_PROTOCOL_ARTIFACT_DIGEST
    ):
        raise VercelPopulationSelectionError("protocol artifact is not exact Vercel V2")
    treatment = _mapping(protocol.get("treatment"), "protocol.treatment")
    exact_treatment = {
        "repository": "https://github.com/vercel-labs/agent-skills",
        "path": "skills/react-best-practices",
        "baseline_commit": BASELINE_COMMIT,
        "baseline_committed_at_utc": "2026-01-18T01:42:22Z",
        "baseline_bundle_sha256": (
            "042dce52998aa6288b4b5eac3fae325113559a666d959a10dd164a981b8ab797"
        ),
        "candidate_commit": CANDIDATE_COMMIT,
        "candidate_committed_at_utc": "2026-01-20T13:43:49Z",
        "candidate_bundle_sha256": (
            "c9a31361925582718024d9ed69078e4dc3a41293f18642436a05a396d91134de"
        ),
        "changed_rule_families": list(PRIMARY_FAMILIES),
        "unchanged_control_families": list(CONTROL_FAMILIES),
    }
    if treatment != exact_treatment:
        raise VercelPopulationSelectionError(
            "protocol treatment revisions or family policy differs from Vercel V2"
        )
    authorization = _mapping(
        protocol.get("authorization_gate"), "protocol.authorization_gate"
    )
    if (
        authorization.get("execution_authorized") is not False
        or authorization.get("spend_authorized") is not False
        or authorization.get("required_ready_status")
        != "ready_after_population_feasibility_power_and_qualification"
    ):
        raise VercelPopulationSelectionError(
            "protocol authorization gate differs from Vercel V2"
        )
    boundary = _mapping(
        authorization.get("authenticated_selection_boundary"),
        "protocol.authorization_gate.authenticated_selection_boundary",
    )
    _exact_keys(
        boundary,
        {
            "status",
            "algorithm",
            "source_verifier_public_key_sha256",
            "beacon_verifier_public_key_sha256",
            "review_governance_public_key_sha256",
            "power_verifier_public_key_sha256",
            "required_artifacts",
            "failure_policy",
        },
        "protocol.authorization_gate.authenticated_selection_boundary",
    )
    if boundary["algorithm"] != AUTHENTICATION_ALGORITHM:
        raise VercelPopulationSelectionError(
            "protocol authentication algorithm differs from Vercel V2"
        )
    required_artifacts = boundary["required_artifacts"]
    expected_required = [
        "acquisition candidates",
        "repository receipts",
        "pull-request receipts",
        "source receipts",
        "stack receipts",
        "lineage ledger",
        "balanced calibration ledger",
        "reviewer detached signatures",
        "source-verifier detached signature",
        "source-authenticated public precommit publication",
        "raw beacon response",
        "beacon-verifier detached signature",
        "prospective power receipt",
    ]
    if required_artifacts != expected_required:
        raise VercelPopulationSelectionError(
            "protocol authenticated artifact list differs from Vercel V2"
        )
    trust_roots: dict[str, str] | None
    if boundary["status"] == "blocked_trust_roots_not_registered":
        for field in (
            "source_verifier_public_key_sha256",
            "beacon_verifier_public_key_sha256",
            "review_governance_public_key_sha256",
            "power_verifier_public_key_sha256",
        ):
            if boundary[field] is not None:
                raise VercelPopulationSelectionError(
                    "blocked authentication boundary must not carry partial trust roots"
                )
        trust_roots = None
    elif boundary["status"] == "ready_trust_roots_registered":
        trust_roots = {
            "source_verification": _sha256(
                boundary["source_verifier_public_key_sha256"],
                "protocol source verifier key",
            ),
            "beacon_verification": _sha256(
                boundary["beacon_verifier_public_key_sha256"],
                "protocol beacon verifier key",
            ),
            "review_governance": _sha256(
                boundary["review_governance_public_key_sha256"],
                "protocol review governance key",
            ),
            "power_verification": _sha256(
                boundary["power_verifier_public_key_sha256"],
                "protocol power verifier key",
            ),
        }
        if len(set(trust_roots.values())) != len(trust_roots):
            raise VercelPopulationSelectionError(
                "protocol trust-root keys must be role separated"
            )
    else:
        raise VercelPopulationSelectionError(
            "protocol authenticated selection status is invalid"
        )

    if (
        query_plan.get("schema_version") != SCHEMA_VERSION
        or query_plan.get("query_plan_id") != QUERY_PLAN_ID
        or query_plan.get("status") != "prospective_not_executed"
        or query_plan.get("artifact_digest") != EXPECTED_QUERY_PLAN_ARTIFACT_DIGEST
    ):
        raise VercelPopulationSelectionError("query plan is not exact Vercel V2")
    protocol_query = _mapping(protocol.get("query_plan"), "protocol.query_plan")
    if protocol_query.get("artifact_digest") != query_plan.get("artifact_digest"):
        raise VercelPopulationSelectionError(
            "protocol and query-plan artifact identities disagree"
        )
    preperiod = _mapping(
        query_plan.get("treatment_preperiod"), "query plan.treatment_preperiod"
    )
    if (
        preperiod.get("baseline_commit") != BASELINE_COMMIT
        or preperiod.get("candidate_commit") != CANDIDATE_COMMIT
        or preperiod.get("record_window_start_utc") != RECORD_WINDOW_START_UTC
        or preperiod.get("record_cutoff_utc") != RECORD_CUTOFF_UTC
        or preperiod.get("github_merged_qualifier")
        != "merged:2023-01-01..2026-01-17"
    ):
        raise VercelPopulationSelectionError(
            "query treatment window or revisions differ from Vercel V2"
        )
    github = _mapping(query_plan.get("github_api"), "query plan.github_api")
    if (
        github.get("base_url") != "https://api.github.com"
        or github.get("endpoint") != "/search/issues"
        or github.get("method") != "GET"
        or github.get("api_version") != "2022-11-28"
        or github.get("fixed_parameters")
        != {"sort": "created", "order": "asc", "per_page": 100}
    ):
        raise VercelPopulationSelectionError(
            "query GitHub API policy differs from Vercel V2"
        )
    repetition = _mapping(
        github.get("acquisition_repetition"),
        "query plan.github_api.acquisition_repetition",
    )
    if (
        repetition.get("acquisitions") != 2
        or repetition.get("minimum_server_time_separation_seconds") != 3600
        or repetition.get("immediate_replay_counts_as_independent") is not False
    ):
        raise VercelPopulationSelectionError(
            "query acquisition repetition policy differs from Vercel V2"
        )
    raw_families = query_plan.get("families")
    raw_queries = query_plan.get("queries")
    if not isinstance(raw_families, list) or not isinstance(raw_queries, list):
        raise VercelPopulationSelectionError("query families and queries must be lists")
    query_ids: set[str] = set()
    family_queries: dict[str, set[str]] = {}
    expected_roles = {
        **{family: "changed_rule_primary" for family in PRIMARY_FAMILIES},
        **{family: "unchanged_rule_safety_control" for family in CONTROL_FAMILIES},
    }
    for raw_family in raw_families:
        family = _mapping(raw_family, "query family")
        family_id = family.get("id")
        if family_id not in expected_roles or family.get("role") != expected_roles[family_id]:
            raise VercelPopulationSelectionError("query family policy differs")
        ids = family.get("query_ids")
        if not isinstance(ids, list) or not ids or ids != list(dict.fromkeys(ids)):
            raise VercelPopulationSelectionError("query family IDs must be unique")
        family_queries[str(family_id)] = {str(item) for item in ids}
        query_ids.update(str(item) for item in ids)
    if set(family_queries) != set(FAMILIES):
        raise VercelPopulationSelectionError("query plan does not cover all Vercel families")
    actual_query_ids: set[str] = set()
    for raw_query in raw_queries:
        query = _mapping(raw_query, "query")
        query_id = _safe_id(query.get("id"), "query.id")
        family = query.get("family")
        if family not in FAMILIES or query_id not in family_queries[str(family)]:
            raise VercelPopulationSelectionError("query-to-family mapping differs")
        query_text = query.get("query")
        if (
            not isinstance(query_text, str)
            or " is:pr " not in f" {query_text} "
            or " is:merged " not in f" {query_text} "
            or query_text.split().count("is:public") != 1
            or "merged:2023-01-01..2026-01-17" not in query_text
            or "-repo:vercel-labs/agent-skills" not in query_text
            or " OR " in query_text
        ):
            raise VercelPopulationSelectionError("query text violates the atomic V2 policy")
        if query_id in actual_query_ids:
            raise VercelPopulationSelectionError(f"duplicate query ID {query_id}")
        actual_query_ids.add(query_id)
    if actual_query_ids != query_ids:
        raise VercelPopulationSelectionError("query family coverage does not reconcile")
    return family_queries, trust_roots


def _nonempty_text(value: object, field: str, *, maximum: int = 4096) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value.encode("utf-8")) > maximum
    ):
        raise VercelPopulationSelectionError(f"{field} must be bounded nonempty text")
    return value


def _github_api_url(value: object, field: str, *, search: bool) -> str:
    text = _nonempty_text(value, field, maximum=16_384)
    parsed = urlparse(text)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "api.github.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or (search and parsed.path != "/search/issues")
    ):
        raise VercelPopulationSelectionError(f"{field} is not a canonical GitHub API URL")
    return text


def _validate_acquisition_observation(
    value: object,
    *,
    field: str,
    expected_candidate_identity_sha256: str,
) -> tuple[dict[str, Any], datetime]:
    observation = dict(_mapping(value, field))
    _exact_keys(observation, _ACQUISITION_OBSERVATION_KEYS, field)
    if observation["lane_id"] != QUERY_PLAN_ID:
        raise VercelPopulationSelectionError(f"{field}.lane_id differs from Vercel V2")
    if (
        observation["credential_profile_id"]
        != "github-public-metadata-readonly-v2"
        or observation["credential_env_name"] != "GITHUB_TOKEN"
        or observation["credential_present_without_value"] is not True
    ):
        raise VercelPopulationSelectionError(
            f"{field} does not bind the exact value-free public GitHub credential profile"
        )
    for name in ("collector_source_commit", "collector_source_tree"):
        raw = observation[name]
        if not isinstance(raw, str) or _GIT_SHA.fullmatch(raw) is None:
            raise VercelPopulationSelectionError(f"{field}.{name} must be a Git SHA")
    for name in (
        "collector_sha256",
        "compiler_sha256",
        "request_url_sha256",
        "headers_sha256",
        "body_sha256",
        "repository_public_visibility_receipt_sha256",
        "terminal_shard_manifest_sha256",
        "ordered_identity_sha256",
        "candidate_identity_sha256",
        "checkpoint_sha256",
    ):
        _sha256(observation[name], f"{field}.{name}")
    if observation["candidate_identity_sha256"] != expected_candidate_identity_sha256:
        raise VercelPopulationSelectionError(
            f"{field}.candidate_identity_sha256 differs from the authoritative record identity"
        )
    requested = _github_api_url(observation["requested_url"], f"{field}.requested_url", search=True)
    final = _github_api_url(observation["final_url"], f"{field}.final_url", search=True)
    redirects = observation["redirect_chain"]
    if (
        not isinstance(redirects, list)
        or not redirects
        or len(redirects) > 8
        or any(not isinstance(item, str) for item in redirects)
        or redirects[0] != requested
        or redirects[-1] != final
    ):
        raise VercelPopulationSelectionError(f"{field}.redirect_chain is invalid")
    for index, item in enumerate(redirects):
        _github_api_url(item, f"{field}.redirect_chain[{index}]", search=True)
    started = _utc(observation["request_started_at_utc"], f"{field}.request_started_at_utc")
    completed = _utc(
        observation["request_completed_at_utc"],
        f"{field}.request_completed_at_utc",
    )
    response_date = _utc(observation["response_date"], f"{field}.response_date")
    if started > completed or not started <= response_date <= completed:
        raise VercelPopulationSelectionError(f"{field} request/server time ordering is invalid")
    if observation["github_api_version"] != "2022-11-28":
        raise VercelPopulationSelectionError(f"{field}.github_api_version differs")
    _nonempty_text(observation["x_github_request_id"], f"{field}.x_github_request_id", maximum=256)
    rate_limit = _positive_int(observation["x_ratelimit_limit"], f"{field}.x_ratelimit_limit")
    rate_remaining = _nonnegative_int(
        observation["x_ratelimit_remaining"], f"{field}.x_ratelimit_remaining"
    )
    rate_used = _nonnegative_int(observation["x_ratelimit_used"], f"{field}.x_ratelimit_used")
    _positive_int(observation["x_ratelimit_reset"], f"{field}.x_ratelimit_reset")
    if rate_remaining + rate_used != rate_limit:
        raise VercelPopulationSelectionError(f"{field} rate-limit counters do not reconcile")
    if observation["x_ratelimit_resource"] != "search":
        raise VercelPopulationSelectionError(f"{field}.x_ratelimit_resource must be search")
    if observation["request_url_sha256"] != _sha256_bytes(requested.encode("utf-8")):
        raise VercelPopulationSelectionError(f"{field}.request_url_sha256 differs")
    return observation, response_date


def _repository_public_visibility_digest(repository: Mapping[str, Any]) -> str:
    return stable_digest(
        {
            "canonical_repository_url": repository["canonical_repository_url"],
            "repository_node_id": repository["repository_node_id"],
            "owner_node_id": repository["owner_node_id"],
            "owner_login": repository["owner_login"],
            "visibility": repository["visibility"],
            "private": repository["private"],
            "api_response_sha256": repository["api_response_sha256"],
            "fetched_at_utc": repository["fetched_at_utc"],
        }
    )


def _receipt_rows_by_record(
    rows: Sequence[object],
    *,
    field: str,
    expected_keys: set[str],
    digest_field: str = "receipt_sha256",
) -> dict[str, dict[str, Any]]:
    if not rows:
        raise VercelPopulationSelectionError(f"{field} must not be empty")
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = _signed_row(
            raw,
            field=f"{field} row {index}",
            expected_keys=expected_keys,
            digest_field=digest_field,
        )
        record_id = _safe_id(row["record_id"], f"{field} row {index}.record_id")
        if record_id in result:
            raise VercelPopulationSelectionError(
                f"{field} contains duplicate record {record_id}"
            )
        result[record_id] = row
    return result


def _validate_source_evidence(  # noqa: C901 - one cross-receipt provenance boundary.
    *,
    records: Sequence[PopulationRecord],
    lineage_rows: Sequence[object],
    acquisition_rows: Sequence[object],
    repository_rows: Sequence[object],
    pull_request_rows: Sequence[object],
    source_rows: Sequence[object],
    stack_rows: Sequence[object],
    family_queries: Mapping[str, set[str]],
    population_frozen_at: datetime,
) -> tuple[dict[str, tuple[str, str]], dict[str, set[str]]]:
    records_by_id = {record.record_id: record for record in records}
    expected_ids = set(records_by_id)
    acquisitions = _receipt_rows_by_record(
        acquisition_rows,
        field="acquisition candidates",
        expected_keys=_ACQUISITION_ROW_KEYS,
        digest_field="acquisition_row_sha256",
    )
    repositories = _receipt_rows_by_record(
        repository_rows,
        field="repository receipts",
        expected_keys=_REPOSITORY_DETAIL_KEYS,
    )
    pull_requests = _receipt_rows_by_record(
        pull_request_rows,
        field="pull-request receipts",
        expected_keys=_PULL_REQUEST_DETAIL_KEYS,
    )
    sources = _receipt_rows_by_record(
        source_rows,
        field="source receipts",
        expected_keys=_SOURCE_DETAIL_KEYS,
    )
    stacks = _receipt_rows_by_record(
        stack_rows,
        field="stack receipts",
        expected_keys=_STACK_DETAIL_KEYS,
    )
    for name, values in (
        ("acquisition candidates", acquisitions),
        ("repository receipts", repositories),
        ("pull-request receipts", pull_requests),
        ("source receipts", sources),
        ("stack receipts", stacks),
    ):
        if set(values) != expected_ids:
            raise VercelPopulationSelectionError(
                f"{name} must exactly cover the frozen population"
            )

    all_query_ids = set().union(*family_queries.values())
    record_queries: dict[str, set[str]] = {}
    cutoff = _utc(RECORD_CUTOFF_UTC, "Vercel record cutoff")
    window_start = _utc(RECORD_WINDOW_START_UTC, "Vercel record window start")
    lineage_objects = {
        str(_mapping(row, "lineage row").get("record_id")): row
        for row in lineage_rows
    }
    if set(lineage_objects) != expected_ids:
        raise VercelPopulationSelectionError(
            "lineage ledger must exactly cover the frozen population"
        )
    for record_id in sorted(expected_ids):
        record = records_by_id[record_id]
        acquisition = acquisitions[record_id]
        repository = repositories[record_id]
        pull_request = pull_requests[record_id]
        source = sources[record_id]
        stack = stacks[record_id]

        query_ids = acquisition["query_ids"]
        if (
            not isinstance(query_ids, list)
            or not query_ids
            or query_ids != sorted(set(query_ids))
            or not all(isinstance(item, str) and item in all_query_ids for item in query_ids)
        ):
            raise VercelPopulationSelectionError(
                f"acquisition record {record_id} query IDs are invalid"
            )
        record_queries[record_id] = set(query_ids)
        merged_at = _utc(
            acquisition["merged_at_utc"],
            f"acquisition record {record_id}.merged_at_utc",
        )
        if not window_start <= merged_at <= cutoff:
            raise VercelPopulationSelectionError(
                f"acquisition record {record_id} is outside the treatment preperiod"
            )
        candidate_identity = stable_digest(
            {
                "pull_request_node_id": acquisition["pull_request_node_id"],
                "canonical_pull_request_url": acquisition[
                    "canonical_pull_request_url"
                ],
                "repository_node_id": acquisition["repository_node_id"],
                "owner_node_id": acquisition["owner_node_id"],
                "merged_at_utc": acquisition["merged_at_utc"],
            }
        )
        acquisition_one, acquired_one = _validate_acquisition_observation(
            acquisition["acquisition_one"],
            field=f"acquisition record {record_id}.acquisition_one",
            expected_candidate_identity_sha256=candidate_identity,
        )
        acquisition_two, acquired_two = _validate_acquisition_observation(
            acquisition["acquisition_two"],
            field=f"acquisition record {record_id}.acquisition_two",
            expected_candidate_identity_sha256=candidate_identity,
        )
        if (
            not merged_at <= acquired_one < acquired_two <= population_frozen_at
            or (acquired_two - acquired_one).total_seconds() < 3600
        ):
            raise VercelPopulationSelectionError(
                f"acquisition record {record_id} was not independently repeated before freeze"
            )
        invariant_observation_fields = (
            "lane_id",
            "credential_profile_id",
            "credential_env_name",
            "collector_source_commit",
            "collector_source_tree",
            "collector_sha256",
            "compiler_sha256",
            "repository_public_visibility_receipt_sha256",
            "terminal_shard_manifest_sha256",
            "ordered_identity_sha256",
            "candidate_identity_sha256",
        )
        if any(
            acquisition_one[field] != acquisition_two[field]
            for field in invariant_observation_fields
        ):
            raise VercelPopulationSelectionError(
                f"acquisition record {record_id} repeated provenance or ordered identities differ"
            )
        if acquisition["canonical_pull_request_url"] != record.canonical_pull_request_url:
            raise VercelPopulationSelectionError(
                f"acquisition record {record_id} URL differs from population"
            )
        acquisition_digest = _sha256(
            acquisition["acquisition_row_sha256"],
            f"acquisition record {record_id}.acquisition_row_sha256",
        )
        if acquisition_digest != record.acquisition_candidate_sha256:
            raise VercelPopulationSelectionError(
                f"acquisition record {record_id} digest differs from population"
            )

        parsed_pr = urlparse(record.canonical_pull_request_url)
        path_parts = parsed_pr.path.strip("/").split("/")
        if len(path_parts) != 4 or path_parts[2] != "pull":
            raise VercelPopulationSelectionError(
                f"population record {record_id} pull-request URL is not canonical"
            )
        owner_login, repository_name = path_parts[0], path_parts[1]
        expected_repository_url = f"https://github.com/{owner_login}/{repository_name}"
        if repository["canonical_repository_url"] != expected_repository_url:
            raise VercelPopulationSelectionError(
                f"repository receipt {record_id} URL differs from pull request"
            )
        if str(repository["owner_login"]).lower() != owner_login.lower():
            raise VercelPopulationSelectionError(
                f"repository receipt {record_id} owner login differs from URL"
            )
        for field in ("repository_node_id", "owner_node_id"):
            _nonempty_text(repository[field], f"repository receipt {record_id}.{field}")
            if acquisition[field] != repository[field]:
                raise VercelPopulationSelectionError(
                    f"acquisition record {record_id} {field} differs"
                )
        if (
            repository["repository_node_id"] != record.repository_id
            or repository["owner_node_id"] != record.owner_id
        ):
            raise VercelPopulationSelectionError(
                f"population record {record_id} repository or owner identity differs"
            )
        if (
            repository["visibility"] != "public"
            or repository["private"] is not False
            or repository["fork"] is not False
            or repository["mirror_url"] is not None
            or repository["archived"] is not False
            or repository["disabled"] is not False
        ):
            raise VercelPopulationSelectionError(
                f"repository receipt {record_id} is not a usable public repository"
            )
        _nonempty_text(repository["default_branch"], f"repository {record_id}.default_branch")
        _nonempty_text(repository["license_spdx"], f"repository {record_id}.license_spdx")
        for field in (
            "fork_network_root_sha256",
            "license_file_sha256",
            "api_response_sha256",
        ):
            _sha256(repository[field], f"repository receipt {record_id}.{field}")
        template_root = repository["template_root_sha256"]
        if template_root is not None:
            _sha256(template_root, f"repository receipt {record_id}.template_root_sha256")
        etag = repository["etag"]
        if etag is not None:
            _nonempty_text(etag, f"repository receipt {record_id}.etag", maximum=1024)
        fetched_at = _utc(
            repository["fetched_at_utc"],
            f"repository receipt {record_id}.fetched_at_utc",
        )
        if not merged_at <= fetched_at <= population_frozen_at:
            raise VercelPopulationSelectionError(
                f"repository receipt {record_id} was not frozen inside the acquisition boundary"
            )
        visibility_digest = _repository_public_visibility_digest(repository)
        if any(
            observation["repository_public_visibility_receipt_sha256"]
            != visibility_digest
            for observation in (acquisition_one, acquisition_two)
        ):
            raise VercelPopulationSelectionError(
                f"acquisition record {record_id} does not bind the public repository receipt"
            )

        if (
            pull_request["pull_request_url"] != record.canonical_pull_request_url
            or pull_request["pull_request_node_id"]
            != acquisition["pull_request_node_id"]
            or pull_request["merged_at_utc"] != acquisition["merged_at_utc"]
        ):
            raise VercelPopulationSelectionError(
                f"pull-request receipt {record_id} differs from acquisition"
            )
        pull_number = _positive_int(
            pull_request["pull_request_number"],
            f"pull-request receipt {record_id}.pull_request_number",
        )
        if pull_number != int(path_parts[3]):
            raise VercelPopulationSelectionError(
                f"pull-request receipt {record_id} number differs from its URL"
            )
        if pull_request["base_repository_node_id"] != record.repository_id:
            raise VercelPopulationSelectionError(
                f"pull-request receipt {record_id} base repository differs"
            )
        _nonempty_text(
            pull_request["head_repository_node_id"],
            f"pull-request receipt {record_id}.head_repository_node_id",
        )
        for field in (
            "base_commit_sha",
            "base_tree_sha",
            "head_commit_sha",
            "head_tree_sha",
            "merge_commit_sha",
            "gold_tree_sha",
        ):
            value = pull_request[field]
            if not isinstance(value, str) or _GIT_SHA.fullmatch(value) is None:
                raise VercelPopulationSelectionError(
                    f"pull-request receipt {record_id}.{field} must be a Git SHA"
                )
        for field in (
            "api_response_sha256",
            "diff_sha256",
            "patch_sha256",
            "changed_files_manifest_sha256",
            "review_manifest_sha256",
            "linked_issue_manifest_sha256",
        ):
            _sha256(pull_request[field], f"pull-request receipt {record_id}.{field}")

        for field in (
            "base_archive_sha256",
            "gold_archive_sha256",
            "base_tree_listing_sha256",
            "gold_tree_listing_sha256",
            "archive_file_manifest_sha256",
            "source_fingerprint_sha256",
        ):
            _sha256(source[field], f"source receipt {record_id}.{field}")
        for field in (
            "submodule_check",
            "unsafe_link_check",
            "path_traversal_check",
            "secret_path_check",
            "generated_dependency_tree_check",
        ):
            if source[field] != "passed":
                raise VercelPopulationSelectionError(
                    f"source receipt {record_id}.{field} did not pass"
                )
        _positive_int(source["source_size_bytes"], f"source receipt {record_id}.source_size_bytes")

        for field in (
            "package_boundary_path",
            "package_manifest_path",
            "dependency_lock_path",
            "lock_format",
            "react_declared_requirement",
            "react_resolved_version",
            "next_declared_requirement",
            "next_resolved_version",
        ):
            text = _nonempty_text(stack[field], f"stack receipt {record_id}.{field}")
            if field.endswith("_path") and (text.startswith("/") or ".." in text.split("/")):
                raise VercelPopulationSelectionError(
                    f"stack receipt {record_id}.{field} is unsafe"
                )
        target_paths = stack["typescript_target_paths"]
        if (
            not isinstance(target_paths, list)
            or not target_paths
            or target_paths != sorted(set(target_paths))
            or any(
                not isinstance(path, str)
                or path.startswith("/")
                or ".." in path.split("/")
                or not path.endswith((".ts", ".tsx"))
                for path in target_paths
            )
        ):
            raise VercelPopulationSelectionError(
                f"stack receipt {record_id} target paths are invalid"
            )
        for field in (
            "package_manifest_sha256",
            "dependency_lock_sha256",
            "target_file_manifest_sha256",
            "parser_runtime_sha256",
        ):
            _sha256(stack[field], f"stack receipt {record_id}.{field}")

        receipt_fields = {
            "acquisition_candidate_sha256": acquisition_digest,
            "repository_receipt_sha256": repository["receipt_sha256"],
            "pull_request_receipt_sha256": pull_request["receipt_sha256"],
            "source_receipt_sha256": source["receipt_sha256"],
            "stack_receipt_sha256": stack["receipt_sha256"],
        }
        if (
            record.repository_receipt_sha256
            != receipt_fields["repository_receipt_sha256"]
            or record.pull_request_receipt_sha256
            != receipt_fields["pull_request_receipt_sha256"]
            or record.source_receipt_sha256 != receipt_fields["source_receipt_sha256"]
            or record.stack_receipt_sha256 != receipt_fields["stack_receipt_sha256"]
            or record.source_facts_sha256 != stable_digest(receipt_fields)
        ):
            raise VercelPopulationSelectionError(
                f"population record {record_id} does not bind exact source receipts"
            )

        lineage = _mapping(lineage_objects[record_id], f"lineage row {record_id}")
        expected_roots = sorted(
            {
                repository["fork_network_root_sha256"],
                source["source_fingerprint_sha256"],
                pull_request["patch_sha256"],
                *(
                    [repository["template_root_sha256"]]
                    if repository["template_root_sha256"] is not None
                    else []
                ),
            }
        )
        if lineage.get("canonical_roots") != expected_roots:
            raise VercelPopulationSelectionError(
                f"lineage row {record_id} is not derived from source receipts"
            )
    return _validate_lineage_rows(lineage_rows, records), record_queries


def _category(partition: str, family: str) -> str:
    return f"{partition}/{family}"


def _split_category(category: str) -> tuple[str, str]:
    partition, family = category.split("/", 1)
    return partition, family


def _expected_categories() -> dict[str, int]:
    return {
        _category(partition, family): EXPECTED_QUOTAS[partition][family]
        for partition in PARTITION_ORDER
        for family in FAMILY_ORDER_BY_PARTITION[partition]
    }


def _slot_manifest() -> list[str]:
    return [
        slot
        for category, quota in _expected_categories().items()
        for slot in _slot_ids(category, quota)
    ]


def _commitment_artifact(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "selection_id": plan["selection_id"],
        "commitment_sha256": selection_commitment_digest(plan),
        "beacon_target": plan["beacon_target"],
    }


def _precommit_publication_payload(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "selection_id": receipt["selection_id"],
        "commitment_sha256": receipt["commitment_sha256"],
        "commitment_artifact_sha256": receipt["commitment_artifact_sha256"],
        "public_commit_url": receipt["public_commit_url"],
        "public_commit_sha": receipt["public_commit_sha"],
        "committed_at_utc": receipt["committed_at_utc"],
        "publication_observed_at_utc": receipt["publication_observed_at_utc"],
        "publication_evidence_sha256": receipt["publication_evidence_sha256"],
        "beacon_target": receipt["beacon_target"],
    }


def selection_commitment_digest(plan: Mapping[str, Any]) -> str:
    """Bind every prospective design field that exists before beacon output."""

    excluded = {
        "precommit_receipt_sha256",
        "randomness_receipt_sha256",
        "beacon_response_sha256",
        "beacon_authentication_sha256",
        "precommit_commitment_sha256",
    }
    payload = {key: plan[key] for key in _PLAN_KEYS - excluded}
    payload["commitment_contract"] = "vercel-public-population-selection-v2"
    return stable_digest(payload)


def validate_plan(value: object) -> dict[str, Any]:
    plan = dict(_mapping(value, "selection plan"))
    _exact_keys(plan, _PLAN_KEYS, "selection plan")
    if plan["schema_version"] != SCHEMA_VERSION:
        raise VercelPopulationSelectionError("selection plan schema_version must be 2")
    _safe_id(plan["selection_id"], "selection plan.selection_id")
    if plan["protocol_id"] != PROTOCOL_ID:
        raise VercelPopulationSelectionError(
            "selection plan protocol_id is not Vercel V2"
        )
    for field in (
        "protocol_sha256",
        "population_sha256",
        "review_ledger_sha256",
        "lineage_ledger_sha256",
        "query_plan_sha256",
        "calibration_receipt_sha256",
        "role_separation_receipt_sha256",
        "precommit_receipt_sha256",
        "verifier_lock_sha256",
        "randomness_receipt_sha256",
        "selection_code_sha256",
        "solver_runtime_sha256",
        "acquisition_candidates_sha256",
        "repository_receipts_sha256",
        "pull_request_receipts_sha256",
        "source_receipts_sha256",
        "stack_receipts_sha256",
        "calibration_ledger_sha256",
        "review_authentication_sha256",
        "source_authentication_sha256",
        "beacon_response_sha256",
        "beacon_authentication_sha256",
        "power_receipt_sha256",
        "precommit_commitment_sha256",
        "slot_manifest_sha256",
    ):
        _sha256(plan[field], f"selection plan.{field}")
    _utc(plan["population_frozen_at_utc"], "selection plan.population_frozen_at_utc")

    target = _mapping(plan["beacon_target"], "selection plan.beacon_target")
    _exact_keys(target, _BEACON_TARGET_KEYS, "selection plan.beacon_target")
    provider = target["provider"]
    if provider not in {"nist-beacon-v2", "drand-mainnet"}:
        raise VercelPopulationSelectionError(
            "selection plan beacon provider is unsupported"
        )
    pulse = target["pulse_or_round"]
    if provider == "drand-mainnet":
        _positive_int(pulse, "selection plan.beacon_target.pulse_or_round")
    elif not isinstance(pulse, str) or not pulse.strip() or len(pulse) > 256:
        raise VercelPopulationSelectionError("NIST pulse identifier is invalid")
    endpoint = target["canonical_endpoint"]
    if not isinstance(endpoint, str):
        raise VercelPopulationSelectionError(
            "selection plan beacon endpoint is invalid"
        )
    expected_origin = (
        "https://api.drand.sh/"
        if provider == "drand-mainnet"
        else "https://beacon.nist.gov/beacon/2.0/"
    )
    if not endpoint.startswith(expected_origin) or str(pulse) not in endpoint:
        raise VercelPopulationSelectionError(
            "selection plan beacon endpoint does not bind the canonical provider target"
        )

    if plan["partition_order"] != list(PARTITION_ORDER):
        raise VercelPopulationSelectionError(
            "selection plan partition order differs from Vercel V2"
        )
    expected_family_order = {
        partition: list(FAMILY_ORDER_BY_PARTITION[partition])
        for partition in PARTITION_ORDER
    }
    if plan["family_order_by_partition"] != expected_family_order:
        raise VercelPopulationSelectionError(
            "selection plan family order differs from Vercel V2"
        )
    if plan["slot_manifest_sha256"] != stable_digest(_slot_manifest()):
        raise VercelPopulationSelectionError(
            "selection plan slot order differs from Vercel V2"
        )

    quotas = _mapping(plan["quotas"], "selection plan.quotas")
    if quotas != EXPECTED_QUOTAS:
        raise VercelPopulationSelectionError(
            "selection plan quotas differ from Vercel V2"
        )
    caps = _mapping(plan["owner_caps"], "selection plan.owner_caps")
    if caps != {"all_partitions": 4, "target_holdout": 2}:
        raise VercelPopulationSelectionError(
            "selection plan owner caps differ from Vercel V2"
        )
    reserve = _mapping(plan["reserve_fraction"], "selection plan.reserve_fraction")
    if reserve != {
        "numerator": RESERVE_NUMERATOR,
        "denominator": RESERVE_DENOMINATOR,
    }:
        raise VercelPopulationSelectionError(
            "selection plan reserve must be exactly 25 percent"
        )
    minimums = _mapping(
        plan["minimum_eligible_lineages"],
        "selection plan.minimum_eligible_lineages",
    )
    if minimums != MINIMUM_ELIGIBLE_LINEAGES:
        raise VercelPopulationSelectionError(
            "selection plan family minimums differ from Vercel V2"
        )
    if plan["minimum_unique_lineages"] != MINIMUM_UNIQUE_LINEAGES:
        raise VercelPopulationSelectionError(
            "selection plan unique-lineage minimum differs from Vercel V2"
        )
    bounds = _mapping(plan["bounds"], "selection plan.bounds")
    if bounds != EXPECTED_BOUNDS:
        raise VercelPopulationSelectionError(
            "selection plan bounds differ from Vercel V2"
        )
    if plan["precommit_commitment_sha256"] != selection_commitment_digest(plan):
        raise VercelPopulationSelectionError(
            "selection plan precommit does not bind the exact frozen design"
        )
    return plan


def _validate_population_rows(rows: Sequence[object]) -> list[PopulationRecord]:
    if not rows:
        raise VercelPopulationSelectionError("population must not be empty")
    records: list[PopulationRecord] = []
    record_ids: set[str] = set()
    urls: set[str] = set()
    repository_owners: dict[str, str] = {}
    for index, raw in enumerate(rows):
        row = dict(_mapping(raw, f"population row {index}"))
        _exact_keys(row, _POPULATION_KEYS, f"population row {index}")
        if row["schema_version"] != SCHEMA_VERSION:
            raise VercelPopulationSelectionError(
                "population rows must use schema_version 2"
            )
        record_id = _safe_id(row["record_id"], f"population row {index}.record_id")
        if record_id in record_ids:
            raise VercelPopulationSelectionError(
                f"duplicate population record {record_id}"
            )
        record_ids.add(record_id)
        repository_id = _safe_id(
            row["repository_id"], f"population row {index}.repository_id"
        )
        owner_id = _safe_id(row["owner_id"], f"population row {index}.owner_id")
        previous_owner = repository_owners.setdefault(repository_id, owner_id)
        if previous_owner != owner_id:
            raise VercelPopulationSelectionError(
                f"repository {repository_id} maps to multiple owner identities"
            )
        url = row["canonical_pull_request_url"]
        if not isinstance(url, str) or _PULL_URL.fullmatch(url) is None:
            raise VercelPopulationSelectionError(
                f"population row {index}.canonical_pull_request_url is invalid"
            )
        if url in urls:
            raise VercelPopulationSelectionError(f"duplicate pull-request URL {url}")
        urls.add(url)
        if row["pre_treatment"] is not True:
            raise VercelPopulationSelectionError(
                f"population row {record_id} must be verified pre-treatment"
            )
        acquisition_digest = _sha256(
            row["acquisition_candidate_sha256"],
            f"population row {index}.acquisition_candidate_sha256",
        )
        source_facts = _sha256(
            row["source_facts_sha256"], f"population row {index}.source_facts_sha256"
        )
        source_receipts = {
            field: _sha256(row[field], f"population row {index}.{field}")
            for field in _SOURCE_RECEIPT_FIELDS
        }
        digest = _sha256(
            row["source_record_sha256"],
            f"population row {index}.source_record_sha256",
        )
        unsigned = dict(row)
        del unsigned["source_record_sha256"]
        if digest != stable_digest(unsigned):
            raise VercelPopulationSelectionError(
                f"population row {record_id} source digest does not match"
            )
        records.append(
            PopulationRecord(
                record_id=record_id,
                repository_id=repository_id,
                owner_id=owner_id,
                canonical_pull_request_url=url,
                acquisition_candidate_sha256=acquisition_digest,
                source_facts_sha256=source_facts,
                repository_receipt_sha256=source_receipts["repository_receipt_sha256"],
                pull_request_receipt_sha256=source_receipts[
                    "pull_request_receipt_sha256"
                ],
                source_receipt_sha256=source_receipts["source_receipt_sha256"],
                stack_receipt_sha256=source_receipts["stack_receipt_sha256"],
                source_record_sha256=digest,
            )
        )
    return records


def _lineage_digest(roots: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(roots) + "\n").encode()).hexdigest()


def _validate_lineage_rows(
    rows: Sequence[object],
    records: Sequence[PopulationRecord],
) -> dict[str, tuple[str, str]]:
    if not rows:
        raise VercelPopulationSelectionError("lineage ledger must not be empty")
    records_by_id = {record.record_id: record for record in records}
    seen: set[str] = set()
    repository_lineages: dict[str, str] = {}
    result: dict[str, tuple[str, str]] = {}
    for index, raw in enumerate(rows):
        row = dict(_mapping(raw, f"lineage row {index}"))
        _exact_keys(row, _LINEAGE_ROW_KEYS, f"lineage row {index}")
        if row["schema_version"] != SCHEMA_VERSION:
            raise VercelPopulationSelectionError(
                "lineage rows must use schema_version 2"
            )
        record_id = _safe_id(row["record_id"], f"lineage row {index}.record_id")
        if record_id in seen:
            raise VercelPopulationSelectionError(f"duplicate lineage row {record_id}")
        seen.add(record_id)
        record = records_by_id.get(record_id)
        if record is None:
            raise VercelPopulationSelectionError(
                f"lineage row {record_id} is not in the frozen population"
            )
        if (
            row["repository_id"] != record.repository_id
            or row["owner_id"] != record.owner_id
        ):
            raise VercelPopulationSelectionError(
                f"lineage row {record_id} repository or owner differs from population"
            )
        raw_roots = row["canonical_roots"]
        if (
            not isinstance(raw_roots, list)
            or not raw_roots
            or len(raw_roots) > 64
            or raw_roots != sorted(set(raw_roots))
        ):
            raise VercelPopulationSelectionError(
                f"lineage row {record_id} canonical roots must be sorted and unique"
            )
        roots = [
            _sha256(root, f"lineage row {record_id}.canonical_roots")
            for root in raw_roots
        ]
        lineage_id = _sha256(row["lineage_id"], f"lineage row {record_id}.lineage_id")
        if lineage_id != _lineage_digest(roots):
            raise VercelPopulationSelectionError(
                f"lineage row {record_id} does not use the canonical roots digest"
            )
        previous = repository_lineages.setdefault(record.repository_id, lineage_id)
        if previous != lineage_id:
            raise VercelPopulationSelectionError(
                f"repository {record.repository_id} spans multiple code lineages"
            )
        digest = _sha256(
            row["lineage_row_sha256"], f"lineage row {record_id}.lineage_row_sha256"
        )
        unsigned = dict(row)
        del unsigned["lineage_row_sha256"]
        if digest != stable_digest(unsigned):
            raise VercelPopulationSelectionError(
                f"lineage row {record_id} digest does not match"
            )
        result[record_id] = (lineage_id, digest)
    if seen != set(records_by_id):
        missing = sorted(set(records_by_id) - seen)
        raise VercelPopulationSelectionError(
            f"lineage ledger must cover every population record; missing={missing[:5]}"
        )
    return result


def _self_digest_receipt(
    value: object,
    *,
    field: str,
    expected_keys: set[str],
) -> dict[str, Any]:
    receipt = dict(_mapping(value, field))
    _exact_keys(receipt, expected_keys, field)
    if receipt["schema_version"] != SCHEMA_VERSION:
        raise VercelPopulationSelectionError(f"{field} schema_version must be 2")
    digest = _sha256(receipt["receipt_sha256"], f"{field}.receipt_sha256")
    unsigned = dict(receipt)
    del unsigned["receipt_sha256"]
    if digest != stable_digest(unsigned):
        raise VercelPopulationSelectionError(f"{field} digest does not match")
    return receipt


def _validate_role_separation_receipt(value: object) -> dict[str, set[str]]:
    receipt = _self_digest_receipt(
        value,
        field="role separation receipt",
        expected_keys=_ROLE_SEPARATION_KEYS,
    )
    role_fields = (
        "eligibility_reviewer_id_hashes",
        "adjudicator_id_hashes",
        "public_task_author_id_hashes",
        "public_test_author_id_hashes",
        "host_verifier_author_id_hashes",
    )
    roles: dict[str, set[str]] = {}
    for field in role_fields:
        raw = receipt[field]
        if not isinstance(raw, list) or not raw or raw != sorted(set(raw)):
            raise VercelPopulationSelectionError(
                f"role separation receipt.{field} must be sorted, unique, and nonempty"
            )
        roles[field] = {
            _sha256(item, f"role separation receipt.{field}") for item in raw
        }
    if receipt["pairwise_disjoint"] is not True:
        raise VercelPopulationSelectionError("role separation must fail closed")
    all_identities = [
        identity for identities in roles.values() for identity in identities
    ]
    if len(all_identities) != len(set(all_identities)):
        raise VercelPopulationSelectionError("review and author roles are not disjoint")
    return roles


def _validate_calibration_receipt(
    value: object,
    *,
    role_ids: Mapping[str, set[str]],
) -> tuple[datetime, set[str]]:
    receipt = _self_digest_receipt(
        value,
        field="reviewer calibration receipt",
        expected_keys=_CALIBRATION_KEYS,
    )
    _safe_id(receipt["calibration_id"], "reviewer calibration receipt.calibration_id")
    completed_at = _utc(
        receipt["completed_at_utc"],
        "reviewer calibration receipt.completed_at_utc",
    )
    raw_reviewers = receipt["reviewer_id_hashes"]
    if (
        not isinstance(raw_reviewers, list)
        or not raw_reviewers
        or raw_reviewers != sorted(set(raw_reviewers))
    ):
        raise VercelPopulationSelectionError(
            "reviewer calibration identities must be sorted and unique"
        )
    reviewers = {
        _sha256(item, "reviewer calibration receipt.reviewer_id_hashes")
        for item in raw_reviewers
    }
    expected_reviewers = (
        role_ids["eligibility_reviewer_id_hashes"] | role_ids["adjudicator_id_hashes"]
    )
    if reviewers != expected_reviewers:
        raise VercelPopulationSelectionError(
            "calibration must cover every registered reviewer and adjudicator"
        )
    examples = _mapping(
        receipt["family_examples"], "reviewer calibration receipt.family_examples"
    )
    if set(examples) != set(FAMILIES):
        raise VercelPopulationSelectionError(
            "calibration must cover every behavior family"
        )
    total = 0
    for family in FAMILIES:
        counts = _mapping(examples[family], f"calibration family {family}")
        _exact_keys(
            counts, {"acceptable", "unacceptable"}, f"calibration family {family}"
        )
        acceptable = _positive_int(
            counts["acceptable"], f"calibration {family}.acceptable"
        )
        unacceptable = _positive_int(
            counts["unacceptable"], f"calibration {family}.unacceptable"
        )
        if acceptable + unacceptable < 8:
            raise VercelPopulationSelectionError(
                f"calibration family {family} has fewer than eight examples"
            )
        total += acceptable + unacceptable
    if receipt["record_count"] != total or total < 48:
        raise VercelPopulationSelectionError(
            "calibration record count must reconcile and be at least 48"
        )
    kappa = receipt["cohens_kappa_micros"]
    if (
        isinstance(kappa, bool)
        or not isinstance(kappa, int)
        or not 800_000 <= kappa <= 1_000_000
    ):
        raise VercelPopulationSelectionError("reviewer calibration kappa is below 0.8")
    if receipt["critical_false_inclusions"] != 0:
        raise VercelPopulationSelectionError(
            "reviewer calibration has a critical false inclusion"
        )
    if receipt["treatment_outputs_blinded"] is not True:
        raise VercelPopulationSelectionError("calibration was not treatment blind")
    return completed_at, reviewers


def _cohens_kappa_micros(first: Sequence[bool], second: Sequence[bool]) -> int:
    if len(first) != len(second) or not first:
        raise VercelPopulationSelectionError(
            "calibration reviewer vectors must be nonempty and aligned"
        )
    total = len(first)
    observed = sum(a == b for a, b in zip(first, second, strict=True)) / total
    first_true = sum(first) / total
    second_true = sum(second) / total
    expected = first_true * second_true + (1 - first_true) * (1 - second_true)
    if expected == 1:
        return 1_000_000 if observed == 1 else 0
    return round((observed - expected) / (1 - expected) * 1_000_000)


def _validate_calibration_ledger(
    rows: Sequence[object],
    *,
    calibration_receipt: Mapping[str, Any],
    role_ids: Mapping[str, set[str]],
) -> str:
    if len(rows) != 48:
        raise VercelPopulationSelectionError(
            "balanced calibration ledger must contain exactly 48 records"
        )
    registered = (
        role_ids["eligibility_reviewer_id_hashes"]
        | role_ids["adjudicator_id_hashes"]
    )
    reviewer_ids = sorted(role_ids["eligibility_reviewer_id_hashes"])
    if len(reviewer_ids) != 2:
        raise VercelPopulationSelectionError(
            "calibration requires exactly two eligibility reviewers"
        )
    seen: set[str] = set()
    family_expected: dict[str, list[bool]] = defaultdict(list)
    decisions: dict[str, list[bool]] = {identity: [] for identity in registered}
    false_inclusions = 0
    normalized_rows: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        row = _signed_row(
            raw,
            field=f"calibration ledger row {index}",
            expected_keys=_CALIBRATION_ROW_KEYS,
            digest_field="row_sha256",
        )
        record_id = _safe_id(
            row["calibration_record_id"],
            f"calibration ledger row {index}.calibration_record_id",
        )
        if record_id in seen:
            raise VercelPopulationSelectionError(
                f"duplicate calibration record {record_id}"
            )
        seen.add(record_id)
        family = row["family"]
        if family not in FAMILIES:
            raise VercelPopulationSelectionError(
                f"calibration record {record_id} has invalid family"
            )
        expected = row["expected_acceptable"]
        if not isinstance(expected, bool):
            raise VercelPopulationSelectionError(
                f"calibration record {record_id} expected label must be boolean"
            )
        raw_decisions = _mapping(
            row["reviewer_decisions"],
            f"calibration record {record_id}.reviewer_decisions",
        )
        if set(raw_decisions) != registered or not all(
            isinstance(value, bool) for value in raw_decisions.values()
        ):
            raise VercelPopulationSelectionError(
                f"calibration record {record_id} must cover every registered reviewer"
            )
        family_expected[str(family)].append(expected)
        for identity in registered:
            decision = bool(raw_decisions[identity])
            decisions[identity].append(decision)
            if not expected and decision:
                false_inclusions += 1
        normalized_rows.append(row)
    if set(family_expected) != set(FAMILIES):
        raise VercelPopulationSelectionError(
            "calibration ledger must cover every Vercel family"
        )
    for family, labels in family_expected.items():
        if len(labels) != 8 or sum(labels) != 4:
            raise VercelPopulationSelectionError(
                f"calibration family {family} must contain four acceptable and four unacceptable records"
            )
    kappa = _cohens_kappa_micros(
        decisions[reviewer_ids[0]], decisions[reviewer_ids[1]]
    )
    if calibration_receipt["record_count"] != 48:
        raise VercelPopulationSelectionError(
            "calibration receipt record count does not match ledger"
        )
    expected_examples = {
        family: {"acceptable": 4, "unacceptable": 4} for family in FAMILIES
    }
    if calibration_receipt["family_examples"] != expected_examples:
        raise VercelPopulationSelectionError(
            "calibration receipt family counts do not match balanced ledger"
        )
    if calibration_receipt["cohens_kappa_micros"] != kappa or kappa < 800_000:
        raise VercelPopulationSelectionError(
            "calibration receipt kappa does not match recomputation or is below 0.8"
        )
    if (
        isinstance(calibration_receipt["critical_false_inclusions"], bool)
        or calibration_receipt["critical_false_inclusions"] != false_inclusions
        or false_inclusions != 0
    ):
        raise VercelPopulationSelectionError(
            "calibration has a critical false inclusion"
        )
    return stable_digest(normalized_rows)


def _reviewer_manifest_payloads(
    review_rows: Sequence[object],
    *,
    calibration_ledger_sha256: str,
    review_ledger_sha256: str,
    role_separation_receipt_sha256: str,
) -> dict[str, dict[str, Any]]:
    decisions: dict[str, list[str]] = defaultdict(list)
    roles: dict[str, str] = {}
    for index, raw in enumerate(review_rows):
        row = _mapping(raw, f"review row {index}")
        for field, expected_role in (
            ("reviewer_a", "eligibility_reviewer"),
            ("reviewer_b", "eligibility_reviewer"),
        ):
            decision = _mapping(row.get(field), f"review row {index}.{field}")
            reviewer_id = _sha256(
                decision.get("reviewer_id_hash"),
                f"review row {index}.{field}.reviewer_id_hash",
            )
            if decision.get("role") != expected_role:
                raise VercelPopulationSelectionError(
                    f"review row {index}.{field} role differs"
                )
            roles.setdefault(reviewer_id, expected_role)
            decisions[reviewer_id].append(
                _sha256(
                    decision.get("decision_sha256"),
                    f"review row {index}.{field}.decision_sha256",
                )
            )
        adjudication = _mapping(
            row.get("adjudication"), f"review row {index}.adjudication"
        )
        if adjudication.get("required") is True:
            reviewer_id = _sha256(
                adjudication.get("reviewer_id_hash"),
                f"review row {index}.adjudication.reviewer_id_hash",
            )
            if adjudication.get("role") != "eligibility_adjudicator":
                raise VercelPopulationSelectionError(
                    f"review row {index}.adjudication role differs"
                )
            roles.setdefault(reviewer_id, "eligibility_adjudicator")
            decisions[reviewer_id].append(
                _sha256(
                    adjudication.get("decision_sha256"),
                    f"review row {index}.adjudication.decision_sha256",
                )
            )
    return {
        reviewer_id: {
            "schema_version": SCHEMA_VERSION,
            "reviewer_id_hash": reviewer_id,
            "role": roles[reviewer_id],
            "decision_sha256s": sorted(values),
            "review_ledger_sha256": review_ledger_sha256,
            "calibration_ledger_sha256": calibration_ledger_sha256,
            "role_separation_receipt_sha256": role_separation_receipt_sha256,
        }
        for reviewer_id, values in decisions.items()
    }


def _verify_ed25519_signature(
    *, public_key_hex: object, signature_hex: object, message: bytes, field: str
) -> str:
    public_key = _public_key_bytes(public_key_hex, f"{field}.public_key_hex")
    if not isinstance(signature_hex, str) or len(signature_hex) != 128:
        raise VercelPopulationSelectionError(
            f"{field}.signature_hex must be 64-byte Ed25519 hex"
        )
    try:
        signature = bytes.fromhex(signature_hex)
    except ValueError as exc:
        raise VercelPopulationSelectionError(
            f"{field}.signature_hex must be hexadecimal"
        ) from exc
    if signature_hex != signature_hex.lower():
        raise VercelPopulationSelectionError(
            f"{field}.signature_hex must be lowercase"
        )
    if Ed25519PublicKey is None:
        raise VercelPopulationSelectionError(
            "Ed25519 verification runtime is unavailable"
        )
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, message)
    except (InvalidSignature, ValueError) as exc:
        raise VercelPopulationSelectionError(f"{field} signature is invalid") from exc
    return hashlib.sha256(public_key).hexdigest()


def _validate_review_authentication(
    value: object,
    *,
    review_rows: Sequence[object],
    review_ledger_sha256: str,
    calibration_ledger_sha256: str,
    role_separation_receipt_sha256: str,
    role_ids: Mapping[str, set[str]],
    governance_key_id: str,
) -> dict[str, Any]:
    receipt = _self_digest_receipt(
        value,
        field="review authentication",
        expected_keys=_REVIEW_AUTHENTICATION_KEYS,
    )
    if (
        receipt["purpose"] != "review_authentication"
        or receipt["algorithm"] != AUTHENTICATION_ALGORITHM
    ):
        raise VercelPopulationSelectionError(
            "review authentication contract differs from Vercel V2"
        )
    manifests = _reviewer_manifest_payloads(
        review_rows,
        calibration_ledger_sha256=calibration_ledger_sha256,
        review_ledger_sha256=review_ledger_sha256,
        role_separation_receipt_sha256=role_separation_receipt_sha256,
    )
    registered = (
        role_ids["eligibility_reviewer_id_hashes"]
        | role_ids["adjudicator_id_hashes"]
    )
    # An adjudicator with no disagreement still signs an empty manifest.  That
    # proves the role registry was frozen without inventing an adjudication.
    for reviewer_id in registered - set(manifests):
        manifests[reviewer_id] = {
            "schema_version": SCHEMA_VERSION,
            "reviewer_id_hash": reviewer_id,
            "role": (
                "eligibility_adjudicator"
                if reviewer_id in role_ids["adjudicator_id_hashes"]
                else "eligibility_reviewer"
            ),
            "decision_sha256s": [],
            "review_ledger_sha256": review_ledger_sha256,
            "calibration_ledger_sha256": calibration_ledger_sha256,
            "role_separation_receipt_sha256": role_separation_receipt_sha256,
        }
    raw_signatures = receipt["reviewer_signatures"]
    if not isinstance(raw_signatures, list):
        raise VercelPopulationSelectionError(
            "review authentication signatures must be a list"
        )
    signatures: dict[str, Mapping[str, Any]] = {}
    public_key_ids: set[str] = set()
    for index, raw in enumerate(raw_signatures):
        signature = _mapping(raw, f"review signature {index}")
        _exact_keys(signature, _REVIEW_SIGNATURE_KEYS, f"review signature {index}")
        reviewer_id = _sha256(
            signature["reviewer_id_hash"],
            f"review signature {index}.reviewer_id_hash",
        )
        if reviewer_id in signatures:
            raise VercelPopulationSelectionError(
                f"duplicate review signature {reviewer_id}"
            )
        payload = manifests.get(reviewer_id)
        if payload is None:
            raise VercelPopulationSelectionError(
                f"review signature {reviewer_id} has no decision manifest"
            )
        if signature["role"] != payload["role"]:
            raise VercelPopulationSelectionError(
                f"review signature {reviewer_id} role differs"
            )
        if signature["decision_manifest_sha256"] != stable_digest(payload):
            raise VercelPopulationSelectionError(
                f"review signature {reviewer_id} manifest digest differs"
            )
        key_id = _verify_ed25519_signature(
            public_key_hex=signature["public_key_hex"],
            signature_hex=signature["signature_hex"],
            message=_authentication_message("reviewer_manifest", payload),
            field=f"review signature {reviewer_id}",
        )
        if key_id != signature["key_id_sha256"] or key_id != reviewer_id:
            raise VercelPopulationSelectionError(
                f"review signature {reviewer_id} key identity differs"
            )
        if key_id in public_key_ids:
            raise VercelPopulationSelectionError(
                "reviewers and adjudicators must use distinct public keys"
            )
        public_key_ids.add(key_id)
        signatures[reviewer_id] = signature
    if set(signatures) != registered or set(manifests) != registered:
        raise VercelPopulationSelectionError(
            "every registered reviewer and adjudicator must sign their manifest"
        )
    governance_payload = {
        "schema_version": SCHEMA_VERSION,
        "review_ledger_sha256": review_ledger_sha256,
        "calibration_ledger_sha256": calibration_ledger_sha256,
        "role_separation_receipt_sha256": role_separation_receipt_sha256,
        "reviewer_registry": [
            {
                "reviewer_id_hash": reviewer_id,
                "role": signatures[reviewer_id]["role"],
                "public_key_hex": signatures[reviewer_id]["public_key_hex"],
                "key_id_sha256": signatures[reviewer_id]["key_id_sha256"],
            }
            for reviewer_id in sorted(signatures)
        ],
    }
    _validate_authentication(
        receipt["governance_authentication"],
        purpose="review_governance",
        payload=governance_payload,
        expected_key_id=governance_key_id,
    )
    return receipt


def _validate_power_receipt(value: object, *, power_key_id: str) -> dict[str, Any]:
    receipt = _self_digest_receipt(
        value,
        field="prospective power receipt",
        expected_keys=_POWER_RECEIPT_KEYS,
    )
    for field in ("analysis_code_sha256", "runtime_sha256"):
        _sha256(receipt[field], f"prospective power receipt.{field}")
    _safe_id(receipt["simulation_seed"], "prospective power receipt.simulation_seed")
    if receipt["simulation_repetitions"] != 100_000:
        raise VercelPopulationSelectionError(
            "prospective power receipt must use 100000 simulations"
        )
    expected_grid = {
        "candidate_only_probability_micros": [200_000, 250_000, 300_000, 350_000],
        "baseline_only_probability_micros": [20_000, 50_000, 100_000, 150_000],
        "minimum_difference_micros": 150_000,
        "holm_familywise_alpha_micros": 50_000,
    }
    if receipt["assumption_grid"] != expected_grid:
        raise VercelPopulationSelectionError(
            "prospective power receipt assumption grid differs from Vercel V2"
        )
    expected_sizes = {
        **{family: 96 for family in PRIMARY_FAMILIES},
        **{family: 16 for family in CONTROL_FAMILIES},
    }
    if receipt["family_sample_sizes"] != expected_sizes:
        raise VercelPopulationSelectionError(
            "prospective power receipt sample sizes differ from Vercel V2"
        )
    raw_scenarios = receipt["scenario_power_results"]
    if not isinstance(raw_scenarios, list):
        raise VercelPopulationSelectionError(
            "prospective power scenario results must be a list"
        )
    expected_pairs = {
        (candidate, baseline)
        for candidate in expected_grid["candidate_only_probability_micros"]
        for baseline in expected_grid["baseline_only_probability_micros"]
        if candidate - baseline >= 150_000
    }
    observed_pairs: set[tuple[int, int]] = set()
    observed_powers: list[int] = []
    for index, raw in enumerate(raw_scenarios):
        row = _mapping(raw, f"power scenario {index}")
        _exact_keys(
            row,
            {
                "candidate_only_probability_micros",
                "baseline_only_probability_micros",
                "power_micros",
            },
            f"power scenario {index}",
        )
        values = tuple(row[field] for field in row)
        if any(isinstance(item, bool) or not isinstance(item, int) for item in values):
            raise VercelPopulationSelectionError(
                f"power scenario {index} values must be integers"
            )
        pair = (
            row["candidate_only_probability_micros"],
            row["baseline_only_probability_micros"],
        )
        if pair not in expected_pairs or pair in observed_pairs:
            raise VercelPopulationSelectionError(
                f"power scenario {index} is missing, duplicate, or outside the grid"
            )
        observed_pairs.add(pair)
        power = row["power_micros"]
        if not 900_000 <= power <= 1_000_000:
            raise VercelPopulationSelectionError(
                f"power scenario {index} is below 0.9"
            )
        observed_powers.append(power)
    if observed_pairs != expected_pairs:
        raise VercelPopulationSelectionError(
            "prospective power receipt does not cover the complete grid"
        )
    if (
        receipt["minimum_observed_power_micros"] != min(observed_powers)
        or receipt["minimum_observed_power_micros"] < 900_000
        or receipt["supported_effect_boundary_micros"] != 150_000
    ):
        raise VercelPopulationSelectionError(
            "prospective power receipt summary does not reconcile"
        )
    safety = _mapping(
        receipt["safety_control_power_results"],
        "prospective power receipt.safety_control_power_results",
    )
    if set(safety) != set(CONTROL_FAMILIES):
        raise VercelPopulationSelectionError(
            "prospective power receipt does not cover every safety family"
        )
    for family, raw in safety.items():
        row = _mapping(raw, f"safety power {family}")
        if row != {
            "margin_micros": -50_000,
            "power_micros": row.get("power_micros"),
            "critical_regressions_allowed": 0,
        }:
            raise VercelPopulationSelectionError(
                f"safety power {family} contract differs"
            )
        power = row["power_micros"]
        if (
            isinstance(power, bool)
            or not isinstance(power, int)
            or not 900_000 <= power <= 1_000_000
        ):
            raise VercelPopulationSelectionError(
                f"safety power {family} is below 0.9"
            )
    payload = dict(receipt)
    payload.pop("authentication")
    payload.pop("receipt_sha256")
    _validate_authentication(
        receipt["authentication"],
        purpose="power_verification",
        payload=payload,
        expected_key_id=power_key_id,
    )
    return receipt


def _validate_beacon_response(
    value: object,
    *,
    plan: Mapping[str, Any],
    randomness_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the frozen raw beacon bytes against the normalized receipt."""

    receipt = _self_digest_receipt(
        value,
        field="raw beacon response",
        expected_keys=_BEACON_RESPONSE_KEYS,
    )
    target = _mapping(plan["beacon_target"], "selection plan.beacon_target")
    for field in ("provider", "canonical_endpoint", "pulse_or_round"):
        if receipt[field] != target[field] or receipt[field] != randomness_receipt[field]:
            raise VercelPopulationSelectionError(
                f"raw beacon response {field} differs from the locked target"
            )
    if receipt["published_at_utc"] != randomness_receipt["published_at_utc"]:
        raise VercelPopulationSelectionError(
            "raw beacon publication time differs from randomness receipt"
        )
    _utc(receipt["published_at_utc"], "raw beacon response.published_at_utc")
    random_value = receipt["random_value"]
    if (
        not isinstance(random_value, str)
        or _RANDOM_HEX.fullmatch(random_value) is None
        or random_value != randomness_receipt["random_value"]
    ):
        raise VercelPopulationSelectionError(
            "raw beacon output differs from randomness receipt"
        )
    for field in ("signature_sha256", "verification_material_sha256"):
        if (
            _sha256(receipt[field], f"raw beacon response.{field}")
            != randomness_receipt[field]
        ):
            raise VercelPopulationSelectionError(
                f"raw beacon response {field} differs from randomness receipt"
            )
    encoded = receipt["raw_response_base64"]
    if not isinstance(encoded, str) or not encoded:
        raise VercelPopulationSelectionError(
            "raw beacon response bytes must use canonical base64"
        )
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise VercelPopulationSelectionError(
            "raw beacon response bytes are not valid base64"
        ) from exc
    if not raw or base64.b64encode(raw).decode("ascii") != encoded:
        raise VercelPopulationSelectionError(
            "raw beacon response bytes must use canonical base64"
        )
    raw_digest = _sha256(receipt["raw_response_sha256"], "raw beacon response digest")
    if raw_digest != _sha256_bytes(raw) or raw_digest != randomness_receipt["response_sha256"]:
        raise VercelPopulationSelectionError(
            "raw beacon response bytes do not match the randomness receipt"
        )
    return receipt


def _source_verification_payload(
    *, plan: Mapping[str, Any], actual_digests: Mapping[str, str], record_count: int
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "selection_id": plan["selection_id"],
        "protocol_sha256": actual_digests["protocol_sha256"],
        "query_plan_sha256": actual_digests["query_plan_sha256"],
        "population_sha256": actual_digests["population_sha256"],
        "acquisition_candidates_sha256": actual_digests[
            "acquisition_candidates_sha256"
        ],
        "repository_receipts_sha256": actual_digests[
            "repository_receipts_sha256"
        ],
        "pull_request_receipts_sha256": actual_digests[
            "pull_request_receipts_sha256"
        ],
        "source_receipts_sha256": actual_digests["source_receipts_sha256"],
        "stack_receipts_sha256": actual_digests["stack_receipts_sha256"],
        "lineage_ledger_sha256": actual_digests["lineage_ledger_sha256"],
        "record_count": record_count,
        "record_window_start_utc": RECORD_WINDOW_START_UTC,
        "record_cutoff_utc": RECORD_CUTOFF_UTC,
    }


def _beacon_verification_payload(
    *,
    plan: Mapping[str, Any],
    precommit: Mapping[str, Any],
    verifier_lock: Mapping[str, Any],
    randomness_receipt: Mapping[str, Any],
    actual_digests: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "selection_id": plan["selection_id"],
        "commitment_sha256": precommit["commitment_sha256"],
        "precommit_receipt_sha256": actual_digests["precommit_receipt_sha256"],
        "raw_beacon_response_sha256": actual_digests["beacon_response_sha256"],
        "randomness_receipt_sha256": actual_digests["randomness_receipt_sha256"],
        "provider": randomness_receipt["provider"],
        "canonical_endpoint": randomness_receipt["canonical_endpoint"],
        "pulse_or_round": randomness_receipt["pulse_or_round"],
        "published_at_utc": randomness_receipt["published_at_utc"],
        "random_value": randomness_receipt["random_value"],
        "response_sha256": randomness_receipt["response_sha256"],
        "signature_sha256": randomness_receipt["signature_sha256"],
        "verification_material_sha256": randomness_receipt[
            "verification_material_sha256"
        ],
        "verifier_lock_sha256": actual_digests["verifier_lock_sha256"],
        "verifier_code_sha256": verifier_lock["verifier_code_sha256"],
        "trust_root_sha256": verifier_lock["trust_root_sha256"],
    }


def _validated_review_outcome(
    raw: Mapping[str, Any], *, field: str, expected_lineage_id: str
) -> tuple[str, str | None, str | None]:
    decision = raw["decision"]
    if decision not in {"eligible", "excluded"}:
        raise VercelPopulationSelectionError(f"{field}.decision is invalid")
    family = raw["family"]
    lineage = raw["lineage_id"]
    if decision == "eligible":
        if family not in FAMILIES:
            raise VercelPopulationSelectionError(f"{field}.family is invalid")
        lineage = _sha256(lineage, f"{field}.lineage_id")
        if lineage != expected_lineage_id:
            raise VercelPopulationSelectionError(
                f"{field}.lineage_id differs from the lineage ledger"
            )
    elif family is not None or lineage is not None:
        raise VercelPopulationSelectionError(
            f"{field} excluded decision must not assign family or lineage"
        )
    reasons = raw["reason_codes"]
    if (
        not isinstance(reasons, list)
        or reasons != sorted(set(reasons))
        or not all(reason in _ALLOWED_EXCLUSION_REASONS for reason in reasons)
    ):
        raise VercelPopulationSelectionError(f"{field}.reason_codes is invalid")
    if decision == "eligible":
        if reasons:
            raise VercelPopulationSelectionError(
                f"{field} eligible decision has exclusions"
            )
        if raw["natural_repair_confirmed"] is not True:
            raise VercelPopulationSelectionError(
                f"{field} did not confirm a natural repair"
            )
        if raw["task_authorable_without_gold"] is not True:
            raise VercelPopulationSelectionError(
                f"{field} did not confirm blind task authorability"
            )
    else:
        if not reasons:
            raise VercelPopulationSelectionError(
                f"{field} excluded decision requires a reason"
            )
        if not isinstance(raw["natural_repair_confirmed"], bool) or not isinstance(
            raw["task_authorable_without_gold"], bool
        ):
            raise VercelPopulationSelectionError(
                f"{field} qualification flags must be boolean"
            )
    return str(decision), family, lineage


def _validate_review_governance(
    raw: Mapping[str, Any],
    *,
    field: str,
    calibration_receipt_sha256: str,
    role_separation_receipt_sha256: str,
    calibrated_at: datetime,
    committed_at: datetime,
) -> None:
    reviewed_at = _utc(raw["decision_timestamp_utc"], f"{field}.decision_timestamp_utc")
    if reviewed_at < calibrated_at or reviewed_at > committed_at:
        raise VercelPopulationSelectionError(
            f"{field} must occur after calibration and no later than public precommit"
        )
    for blind_field in (
        "baseline_candidate_skills_blinded",
        "treatment_arm_labels_blinded",
        "selection_seed_and_score_blinded",
        "partition_blinded",
        "agent_judge_outputs_blinded",
        "other_reviewer_decision_blinded",
    ):
        if raw[blind_field] is not True:
            raise VercelPopulationSelectionError(f"{field}.{blind_field} must be true")
    if raw["calibration_receipt_sha256"] != calibration_receipt_sha256:
        raise VercelPopulationSelectionError(f"{field} calibration receipt differs")
    if raw["role_separation_receipt_sha256"] != role_separation_receipt_sha256:
        raise VercelPopulationSelectionError(f"{field} role separation receipt differs")


def _reviewer_decision(
    value: object,
    *,
    field: str,
    record_id: str,
    source_record_sha256: str,
    lineage_row_sha256: str,
    expected_lineage_id: str,
    calibration_receipt_sha256: str,
    role_separation_receipt_sha256: str,
    calibrated_at: datetime,
    committed_at: datetime,
    registered_role_ids: Mapping[str, set[str]],
    adjudication: bool = False,
) -> tuple[str, str, str | None, str | None]:
    raw = dict(_mapping(value, field))
    expected = _ADJUDICATION_KEYS if adjudication else _REVIEWER_KEYS
    _exact_keys(raw, expected, field)
    reviewer_id = _sha256(raw["reviewer_id_hash"], f"{field}.reviewer_id_hash")
    expected_role = (
        "eligibility_adjudicator" if adjudication else "eligibility_reviewer"
    )
    if raw["role"] != expected_role:
        raise VercelPopulationSelectionError(f"{field}.role must be {expected_role}")
    registry_key = (
        "adjudicator_id_hashes" if adjudication else "eligibility_reviewer_id_hashes"
    )
    if reviewer_id not in registered_role_ids[registry_key]:
        raise VercelPopulationSelectionError(
            f"{field} reviewer is not in the locked role registry"
        )
    if raw["record_id"] != record_id:
        raise VercelPopulationSelectionError(f"{field} reviewed a different record")
    if raw["source_record_sha256"] != source_record_sha256:
        raise VercelPopulationSelectionError(
            f"{field} reviewed a different source record"
        )
    if raw["lineage_row_sha256"] != lineage_row_sha256:
        raise VercelPopulationSelectionError(
            f"{field} reviewed a different lineage receipt"
        )
    decision, family, lineage = _validated_review_outcome(
        raw,
        field=field,
        expected_lineage_id=expected_lineage_id,
    )
    _validate_review_governance(
        raw,
        field=field,
        calibration_receipt_sha256=calibration_receipt_sha256,
        role_separation_receipt_sha256=role_separation_receipt_sha256,
        calibrated_at=calibrated_at,
        committed_at=committed_at,
    )
    digest = _sha256(raw["decision_sha256"], f"{field}.decision_sha256")
    unsigned = dict(raw)
    del unsigned["decision_sha256"]
    if digest != stable_digest(unsigned):
        raise VercelPopulationSelectionError(f"{field} decision digest does not match")
    return reviewer_id, decision, family, lineage


def _final_review(
    row: Mapping[str, Any],
    *,
    source_record_sha256: str,
    lineage_row_sha256: str,
    expected_lineage_id: str,
    calibration_receipt_sha256: str,
    role_separation_receipt_sha256: str,
    calibrated_at: datetime,
    committed_at: datetime,
    registered_role_ids: Mapping[str, set[str]],
    field: str,
) -> tuple[str, str | None, str | None]:
    reviewer_a = _reviewer_decision(
        row["reviewer_a"],
        field=f"{field}.reviewer_a",
        record_id=str(row["record_id"]),
        source_record_sha256=source_record_sha256,
        lineage_row_sha256=lineage_row_sha256,
        expected_lineage_id=expected_lineage_id,
        calibration_receipt_sha256=calibration_receipt_sha256,
        role_separation_receipt_sha256=role_separation_receipt_sha256,
        calibrated_at=calibrated_at,
        committed_at=committed_at,
        registered_role_ids=registered_role_ids,
    )
    reviewer_b = _reviewer_decision(
        row["reviewer_b"],
        field=f"{field}.reviewer_b",
        record_id=str(row["record_id"]),
        source_record_sha256=source_record_sha256,
        lineage_row_sha256=lineage_row_sha256,
        expected_lineage_id=expected_lineage_id,
        calibration_receipt_sha256=calibration_receipt_sha256,
        role_separation_receipt_sha256=role_separation_receipt_sha256,
        calibrated_at=calibrated_at,
        committed_at=committed_at,
        registered_role_ids=registered_role_ids,
    )
    if reviewer_a[0] == reviewer_b[0]:
        raise VercelPopulationSelectionError(f"{field} requires independent reviewers")
    same_decision = reviewer_a[1:] == reviewer_b[1:]

    adjudication_raw = dict(_mapping(row["adjudication"], f"{field}.adjudication"))
    required = adjudication_raw.get("required")
    if not isinstance(required, bool):
        raise VercelPopulationSelectionError(
            f"{field}.adjudication.required must be boolean"
        )
    if same_decision:
        expected = {key: None for key in _ADJUDICATION_KEYS}
        expected.update(
            {
                "required": False,
                "record_id": row["record_id"],
                "source_record_sha256": source_record_sha256,
                "lineage_row_sha256": lineage_row_sha256,
                "calibration_receipt_sha256": calibration_receipt_sha256,
                "role_separation_receipt_sha256": role_separation_receipt_sha256,
            }
        )
        if adjudication_raw != expected:
            raise VercelPopulationSelectionError(
                f"{field} must not fabricate adjudication for reviewer agreement"
            )
        return reviewer_a[1], reviewer_a[2], reviewer_a[3]
    if required is not True:
        raise VercelPopulationSelectionError(
            f"{field} must adjudicate every disagreement"
        )
    adjudicator = _reviewer_decision(
        adjudication_raw,
        field=f"{field}.adjudication",
        record_id=str(row["record_id"]),
        source_record_sha256=source_record_sha256,
        lineage_row_sha256=lineage_row_sha256,
        expected_lineage_id=expected_lineage_id,
        calibration_receipt_sha256=calibration_receipt_sha256,
        role_separation_receipt_sha256=role_separation_receipt_sha256,
        calibrated_at=calibrated_at,
        committed_at=committed_at,
        registered_role_ids=registered_role_ids,
        adjudication=True,
    )
    if adjudicator[0] in {reviewer_a[0], reviewer_b[0]}:
        raise VercelPopulationSelectionError(f"{field} adjudicator must be independent")
    reason = adjudication_raw["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise VercelPopulationSelectionError(f"{field}.adjudication.reason is required")
    return adjudicator[1], adjudicator[2], adjudicator[3]


def _validate_review_rows(
    rows: Sequence[object],
    records: Sequence[PopulationRecord],
    lineage_rows: Mapping[str, tuple[str, str]],
    *,
    calibration_receipt_sha256: str,
    role_separation_receipt_sha256: str,
    calibrated_at: datetime,
    committed_at: datetime,
    registered_role_ids: Mapping[str, set[str]],
) -> list[tuple[PopulationRecord, str, str]]:
    if not rows:
        raise VercelPopulationSelectionError("review ledger must not be empty")
    records_by_id = {record.record_id: record for record in records}
    seen: set[str] = set()
    eligible: list[tuple[PopulationRecord, str, str]] = []
    for index, raw in enumerate(rows):
        row = dict(_mapping(raw, f"review row {index}"))
        _exact_keys(row, _REVIEW_ROW_KEYS, f"review row {index}")
        if row["schema_version"] != SCHEMA_VERSION:
            raise VercelPopulationSelectionError(
                "review rows must use schema_version 2"
            )
        record_id = _safe_id(row["record_id"], f"review row {index}.record_id")
        if record_id in seen:
            raise VercelPopulationSelectionError(f"duplicate review row {record_id}")
        seen.add(record_id)
        record = records_by_id.get(record_id)
        if record is None:
            raise VercelPopulationSelectionError(
                f"review row {record_id} is not in population"
            )
        if row["source_record_sha256"] != record.source_record_sha256:
            raise VercelPopulationSelectionError(
                f"review row {record_id} source record digest differs"
            )
        decision, family, lineage = _final_review(
            row,
            source_record_sha256=record.source_record_sha256,
            lineage_row_sha256=lineage_rows[record_id][1],
            expected_lineage_id=lineage_rows[record_id][0],
            calibration_receipt_sha256=calibration_receipt_sha256,
            role_separation_receipt_sha256=role_separation_receipt_sha256,
            calibrated_at=calibrated_at,
            committed_at=committed_at,
            registered_role_ids=registered_role_ids,
            field=f"review row {record_id}",
        )
        reasons = row["exclusion_reasons"]
        if not isinstance(reasons, list) or not all(
            isinstance(reason, str) and reason.strip() for reason in reasons
        ):
            raise VercelPopulationSelectionError(
                f"review row {record_id}.exclusion_reasons is invalid"
            )
        if decision == "eligible":
            if reasons:
                raise VercelPopulationSelectionError(
                    f"eligible review row {record_id} may not have exclusions"
                )
            assert family is not None and lineage is not None
            eligible.append((record, family, lineage))
        elif not reasons:
            raise VercelPopulationSelectionError(
                f"excluded review row {record_id} requires an exclusion reason"
            )
        digest = _sha256(
            row["review_row_sha256"], f"review row {index}.review_row_sha256"
        )
        unsigned = dict(row)
        del unsigned["review_row_sha256"]
        if digest != stable_digest(unsigned):
            raise VercelPopulationSelectionError(
                f"review row {record_id} digest does not match"
            )
    if seen != set(records_by_id):
        missing = sorted(set(records_by_id) - seen)
        raise VercelPopulationSelectionError(
            f"review ledger must cover every population record; missing={missing[:5]}"
        )
    return eligible


def _validate_precommit_receipt(
    value: object,
    *,
    plan: Mapping[str, Any],
    source_key_id: str,
) -> tuple[dict[str, Any], datetime]:
    receipt = _self_digest_receipt(
        value,
        field="public precommit receipt",
        expected_keys=_PRECOMMIT_KEYS,
    )
    if receipt["selection_id"] != plan["selection_id"]:
        raise VercelPopulationSelectionError(
            "public precommit selection identity differs"
        )
    if receipt["commitment_sha256"] != plan["precommit_commitment_sha256"]:
        raise VercelPopulationSelectionError("public precommit commitment differs")
    if receipt["commitment_sha256"] != selection_commitment_digest(plan):
        raise VercelPopulationSelectionError(
            "public precommit does not bind the frozen design"
        )
    if receipt["commitment_artifact_sha256"] != stable_digest(
        _commitment_artifact(plan)
    ):
        raise VercelPopulationSelectionError("public precommit artifact digest differs")
    if receipt["beacon_target"] != plan["beacon_target"]:
        raise VercelPopulationSelectionError("public precommit beacon target differs")
    commit_sha = receipt["public_commit_sha"]
    if not isinstance(commit_sha, str) or _GIT_SHA.fullmatch(commit_sha) is None:
        raise VercelPopulationSelectionError("public precommit Git SHA is invalid")
    expected_suffix = f"/commit/{commit_sha}"
    url = receipt["public_commit_url"]
    parsed_url = urlparse(url) if isinstance(url, str) else None
    if (
        not isinstance(url, str)
        or parsed_url is None
        or parsed_url.scheme != "https"
        or parsed_url.hostname != "github.com"
        or parsed_url.username is not None
        or parsed_url.password is not None
        or parsed_url.query
        or parsed_url.fragment
        or not url.endswith(expected_suffix)
    ):
        raise VercelPopulationSelectionError(
            "public precommit URL does not bind its Git SHA"
        )
    committed = _utc(
        receipt["committed_at_utc"], "public precommit receipt.committed_at_utc"
    )
    frozen = _utc(
        plan["population_frozen_at_utc"], "selection plan.population_frozen_at_utc"
    )
    observed = _utc(
        receipt["publication_observed_at_utc"],
        "public precommit receipt.publication_observed_at_utc",
    )
    if committed < frozen or observed < committed:
        raise VercelPopulationSelectionError(
            "public precommit commit or publication proof precedes its frozen inputs"
        )
    _sha256(
        receipt["publication_evidence_sha256"],
        "public precommit receipt.publication_evidence_sha256",
    )
    publication_authentication = _validate_authentication(
        receipt["publication_authentication"],
        purpose="precommit_publication_verification",
        payload=_precommit_publication_payload(receipt),
        expected_key_id=source_key_id,
    )
    if publication_authentication["authenticated_at_utc"] != receipt[
        "publication_observed_at_utc"
    ]:
        raise VercelPopulationSelectionError(
            "public precommit publication signature time differs from its observation"
        )
    return receipt, observed


def _validate_verifier_lock(
    value: object,
    *,
    plan: Mapping[str, Any],
    committed_at: datetime,
) -> dict[str, Any]:
    receipt = _self_digest_receipt(
        value,
        field="beacon verifier lock",
        expected_keys=_VERIFIER_LOCK_KEYS,
    )
    target = plan["beacon_target"]
    if receipt["provider"] != target["provider"]:
        raise VercelPopulationSelectionError(
            "verifier provider differs from beacon target"
        )
    if receipt["canonical_endpoint"] != target["canonical_endpoint"]:
        raise VercelPopulationSelectionError(
            "verifier endpoint differs from beacon target"
        )
    _safe_id(receipt["verifier_id"], "beacon verifier lock.verifier_id")
    _sha256(
        receipt["verifier_code_sha256"], "beacon verifier lock.verifier_code_sha256"
    )
    _sha256(receipt["trust_root_sha256"], "beacon verifier lock.trust_root_sha256")
    expected = (
        ("drand_chain_hash", "bls12-381-g2")
        if receipt["provider"] == "drand-mainnet"
        else ("nist_certificate_bundle", "nist-signed-pulse-v2")
    )
    if (receipt["trust_root_kind"], receipt["signature_algorithm"]) != expected:
        raise VercelPopulationSelectionError("verifier trust-root contract is invalid")
    locked_at = _utc(receipt["locked_at_utc"], "beacon verifier lock.locked_at_utc")
    if locked_at > committed_at:
        raise VercelPopulationSelectionError(
            "beacon verifier was not locked before precommit"
        )
    return receipt


def _validate_solver_runtime(value: object) -> dict[str, Any]:
    receipt = _self_digest_receipt(
        value,
        field="solver runtime receipt",
        expected_keys=_SOLVER_RUNTIME_KEYS,
    )
    _safe_id(receipt["runtime_id"], "solver runtime receipt.runtime_id")
    if receipt["python_implementation"] != platform.python_implementation():
        raise VercelPopulationSelectionError("solver Python implementation differs")
    if receipt["python_version"] != platform.python_version():
        raise VercelPopulationSelectionError("solver Python version differs")
    if receipt["selector_algorithm"] != "exact_min_cost_flow_owner_branching_v2":
        raise VercelPopulationSelectionError("solver algorithm identity differs")
    return receipt


def _validate_randomness_receipt(
    value: object,
    *,
    plan: Mapping[str, Any],
    committed_at: datetime,
    verifier_lock: Mapping[str, Any],
) -> str:
    receipt = _self_digest_receipt(
        value,
        field="randomness receipt",
        expected_keys=_RANDOMNESS_KEYS,
    )
    target = plan["beacon_target"]
    for field in ("provider", "canonical_endpoint", "pulse_or_round"):
        if receipt[field] != target[field]:
            raise VercelPopulationSelectionError(
                f"randomness receipt {field} differs from the public beacon target"
            )
    published = _utc(receipt["published_at_utc"], "randomness receipt.published_at_utc")
    retrieved = _utc(receipt["retrieved_at_utc"], "randomness receipt.retrieved_at_utc")
    verified = _utc(receipt["verified_at_utc"], "randomness receipt.verified_at_utc")
    if not committed_at < published <= retrieved <= verified:
        raise VercelPopulationSelectionError(
            "randomness publication, retrieval, and verification ordering is invalid"
        )
    random_value = receipt["random_value"]
    if not isinstance(random_value, str) or _RANDOM_HEX.fullmatch(random_value) is None:
        raise VercelPopulationSelectionError(
            "randomness output must be 32-byte drand or 32/64-byte NIST hex"
        )
    if receipt["signature_verified"] is not True:
        raise VercelPopulationSelectionError("randomness signature is not verified")
    for field in (
        "signature_sha256",
        "verification_material_sha256",
        "response_sha256",
        "commitment_sha256",
        "verifier_lock_sha256",
        "verifier_code_sha256",
        "trust_root_sha256",
    ):
        _sha256(receipt[field], f"randomness receipt.{field}")
    if receipt["commitment_sha256"] != plan["precommit_commitment_sha256"]:
        raise VercelPopulationSelectionError("randomness receipt commitment differs")
    if receipt["verifier_lock_sha256"] != plan["verifier_lock_sha256"]:
        raise VercelPopulationSelectionError("randomness verifier lock differs")
    if receipt["verifier_code_sha256"] != verifier_lock["verifier_code_sha256"]:
        raise VercelPopulationSelectionError("randomness verifier code differs")
    if receipt["trust_root_sha256"] != verifier_lock["trust_root_sha256"]:
        raise VercelPopulationSelectionError("randomness trust root differs")
    for field in (
        "population_sha256",
        "review_ledger_sha256",
        "lineage_ledger_sha256",
        "query_plan_sha256",
        "calibration_receipt_sha256",
        "role_separation_receipt_sha256",
        "selection_code_sha256",
    ):
        if receipt[field] != plan[field]:
            raise VercelPopulationSelectionError(
                f"randomness receipt {field} differs from selection plan"
            )
    return random_value


def _master_seed(
    randomness: str,
    *,
    population_sha256: str,
    review_ledger_sha256: str,
    query_plan_sha256: str,
    selection_code_sha256: str,
) -> str:
    value = "\n".join(
        (
            randomness,
            population_sha256,
            review_ledger_sha256,
            query_plan_sha256,
            selection_code_sha256,
            "",
        )
    )
    return hashlib.sha256(value.encode()).hexdigest()


def _record_score(seed: str, lineage: str, family: str, url: str) -> str:
    value = "\n".join((seed, lineage, family, url, ""))
    return hashlib.sha256(value.encode()).hexdigest()


def _canonical_candidates(
    eligible: Sequence[tuple[PopulationRecord, str, str]],
    *,
    seed: str,
) -> tuple[list[Candidate], int]:
    by_lineage_family: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for record, family, lineage in eligible:
        by_lineage_family[(lineage, family)].append(
            Candidate(
                record=record,
                lineage_id=lineage,
                family=family,
                score_sha256=_record_score(
                    seed,
                    lineage,
                    family,
                    record.canonical_pull_request_url,
                ),
            )
        )
    candidates: list[Candidate] = []
    collapsed = 0
    for values in by_lineage_family.values():
        values.sort(
            key=lambda item: (
                item.score_sha256,
                item.record.canonical_pull_request_url,
                item.record.record_id,
            )
        )
        candidates.append(values[0])
        collapsed += len(values) - 1
    candidates.sort(
        key=lambda item: (
            item.family,
            item.score_sha256,
            item.record.canonical_pull_request_url,
            item.lineage_id,
        )
    )
    return candidates, collapsed


def _add_edge(
    graph: list[list[_FlowEdge]], source: int, target: int, capacity: int, cost: int
) -> int:
    forward_index = len(graph[source])
    graph[source].append(
        _FlowEdge(to=target, reverse=len(graph[target]), capacity=capacity, cost=cost)
    )
    graph[target].append(
        _FlowEdge(to=source, reverse=forward_index, capacity=0, cost=-cost)
    )
    return forward_index


def _min_cost_flow(
    graph: list[list[_FlowEdge]],
    source: int,
    sink: int,
    required: int,
) -> tuple[int, int]:
    node_count = len(graph)
    potential = [0] * node_count
    flow = 0
    cost = 0
    infinity = 10**200
    while flow < required:
        distance = [infinity] * node_count
        parent: list[tuple[int, int] | None] = [None] * node_count
        distance[source] = 0
        queue: list[tuple[int, int]] = [(0, source)]
        while queue:
            current_distance, node = heapq.heappop(queue)
            if current_distance != distance[node]:
                continue
            for edge_index, edge in enumerate(graph[node]):
                if edge.capacity <= 0:
                    continue
                candidate_distance = (
                    current_distance + edge.cost + potential[node] - potential[edge.to]
                )
                # Strict improvement preserves an acyclic predecessor tree.  Node
                # and adjacency ordering already provide deterministic tie handling;
                # replacing parents on zero-cost residual ties can create a cycle.
                if candidate_distance < distance[edge.to]:
                    distance[edge.to] = candidate_distance
                    parent[edge.to] = (node, edge_index)
                    heapq.heappush(queue, (candidate_distance, edge.to))
        if parent[sink] is None:
            return flow, cost
        for node, value in enumerate(distance):
            if value < infinity:
                potential[node] += value
        node = sink
        while node != source:
            previous, edge_index = parent[node]  # type: ignore[misc]
            edge = graph[previous][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = previous
        flow += 1
        cost += potential[sink]
    return flow, cost


def _compatible_categories(family: str, quotas: Mapping[str, int]) -> list[str]:
    return [
        category
        for category in _expected_categories()
        if category in quotas and _split_category(category)[1] == family
    ]


def _unconstrained_assignment(
    candidates: Sequence[Candidate],
    quotas: Mapping[str, int],
    *,
    forbidden: frozenset[tuple[str, str]],
) -> tuple[int, tuple[Assignment, ...]] | None:
    required = sum(quotas.values())
    lineages = sorted({candidate.lineage_id for candidate in candidates})
    candidate_by_key = {candidate.key: candidate for candidate in candidates}
    possible: list[tuple[str, str, Candidate]] = []
    for candidate in candidates:
        for category in _compatible_categories(candidate.family, quotas):
            if (candidate.lineage_id, category) not in forbidden:
                possible.append((candidate.lineage_id, category, candidate))
    possible.sort(
        key=lambda item: (
            item[2].record.canonical_pull_request_url,
            tuple(_expected_categories()).index(item[1]),
            item[0],
            item[2].record.record_id,
        )
    )
    edge_rank = {
        (lineage, category): rank
        for rank, (lineage, category, _candidate) in enumerate(possible, start=1)
    }
    tie_scale = max(2, required * max(1, len(possible)) + 1)

    source = 0
    lineage_node = {lineage: index + 1 for index, lineage in enumerate(lineages)}
    category_offset = len(lineage_node) + 1
    category_node = {
        category: category_offset + index
        for index, category in enumerate(
            category for category in _expected_categories() if category in quotas
        )
    }
    sink = category_offset + len(category_node)
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]
    for lineage in lineages:
        _add_edge(graph, source, lineage_node[lineage], 1, 0)
    assignment_edges: dict[tuple[str, str], tuple[int, int]] = {}
    for lineage, category, candidate in possible:
        edge_cost = (
            int(candidate.score_sha256, 16) * tie_scale + edge_rank[(lineage, category)]
        )
        node = lineage_node[lineage]
        edge_index = _add_edge(graph, node, category_node[category], 1, edge_cost)
        assignment_edges[(lineage, category)] = (node, edge_index)
    for category, quota in quotas.items():
        _add_edge(graph, category_node[category], sink, quota, 0)

    flow, cost = _min_cost_flow(graph, source, sink, required)
    if flow != required:
        return None
    assignments: list[Assignment] = []
    for key, (node, edge_index) in assignment_edges.items():
        if graph[node][edge_index].capacity == 0:
            lineage, category = key
            family = _split_category(category)[1]
            assignments.append(
                Assignment(
                    candidate=candidate_by_key[(lineage, family)], category=category
                )
            )
    assignments.sort(
        key=lambda item: (
            tuple(_expected_categories()).index(item.category),
            item.candidate.record.canonical_pull_request_url,
            item.candidate.lineage_id,
        )
    )
    if len(assignments) != required:
        raise VercelPopulationSelectionError(
            "internal flow receipt disagrees with selected count"
        )
    return cost, tuple(assignments)


def _assignment_signature(
    assignments: Sequence[Assignment],
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        sorted(
            [
                (
                    assignment.category,
                    assignment.candidate.record.canonical_pull_request_url,
                    assignment.candidate.lineage_id,
                )
                for assignment in assignments
            ],
            key=lambda item: (
                tuple(_expected_categories()).index(item[0]),
                item[1],
                item[2],
            ),
        )
    )


def _first_owner_violation(
    assignments: Sequence[Assignment],
) -> tuple[str, str, tuple[tuple[str, str], ...]] | None:
    all_assignments: dict[str, list[Assignment]] = defaultdict(list)
    target_assignments: dict[str, list[Assignment]] = defaultdict(list)
    for assignment in assignments:
        owner = assignment.candidate.record.owner_id
        all_assignments[owner].append(assignment)
        if _split_category(assignment.category)[0] == "target_holdout":
            target_assignments[owner].append(assignment)
    for owner in sorted(all_assignments):
        if len(all_assignments[owner]) > MAX_SELECTED_PER_OWNER:
            return (
                owner,
                "all_partitions",
                tuple(sorted(assignment.key for assignment in all_assignments[owner])),
            )
        if len(target_assignments[owner]) > MAX_TARGET_HOLDOUT_PER_OWNER:
            return (
                owner,
                "target_holdout",
                tuple(
                    sorted(assignment.key for assignment in target_assignments[owner])
                ),
            )
    return None


def _solve_constrained(
    candidates: Sequence[Candidate],
    quotas: Mapping[str, int],
    *,
    max_search_states: int,
    max_graph_edges: int = EXPECTED_BOUNDS["max_graph_edges"],
    max_pending_search_states: int = EXPECTED_BOUNDS["max_pending_search_states"],
) -> tuple[tuple[Assignment, ...], int, int]:
    possible_edges = sum(
        len(_compatible_categories(candidate.family, quotas))
        for candidate in candidates
    )
    lineage_count = len({candidate.lineage_id for candidate in candidates})
    directed_residual_edges = 2 * (lineage_count + possible_edges + len(quotas))
    if directed_residual_edges > max_graph_edges:
        raise VercelPopulationSelectionError(
            "global lineage-to-slot residual graph has "
            f"{directed_residual_edges} directed edges; limit is {max_graph_edges}"
        )
    initial_forbidden: frozenset[tuple[str, str]] = frozenset()
    initial = _unconstrained_assignment(candidates, quotas, forbidden=initial_forbidden)
    if initial is None:
        raise VercelPopulationSelectionError(
            "global lineage-to-slot graph cannot satisfy every Vercel quota"
        )
    counter = 0
    queue: list[
        tuple[
            int,
            tuple[tuple[str, str, str], ...],
            int,
            frozenset[tuple[str, str]],
            tuple[Assignment, ...],
        ]
    ] = [
        (
            initial[0],
            _assignment_signature(initial[1]),
            counter,
            initial_forbidden,
            initial[1],
        )
    ]
    seen = {initial_forbidden}
    explored = 0
    best: (
        tuple[int, tuple[tuple[str, str, str], ...], tuple[Assignment, ...]] | None
    ) = None
    while queue:
        cost, signature, _counter, forbidden, assignments = heapq.heappop(queue)
        if best is not None and cost > best[0]:
            break
        explored += 1
        if explored > max_search_states:
            raise VercelPopulationSelectionError(
                "owner-constrained search exceeded its locked max_search_states"
            )
        violation = _first_owner_violation(assignments)
        if violation is None:
            candidate_best = (cost, signature, assignments)
            if best is None or candidate_best[:2] < best[:2]:
                best = candidate_best
            continue
        _owner, _kind, violating_edges = violation
        for edge_key in violating_edges:
            child_forbidden = forbidden | {edge_key}
            if child_forbidden in seen:
                continue
            seen.add(child_forbidden)
            child = _unconstrained_assignment(
                candidates,
                quotas,
                forbidden=child_forbidden,
            )
            if child is None:
                continue
            counter += 1
            heapq.heappush(
                queue,
                (
                    child[0],
                    _assignment_signature(child[1]),
                    counter,
                    child_forbidden,
                    child[1],
                ),
            )
            if len(queue) > max_pending_search_states:
                raise VercelPopulationSelectionError(
                    "owner-constrained search exceeded its locked pending-state bound"
                )
    if best is None:
        raise VercelPopulationSelectionError(
            "no global assignment satisfies Vercel quotas and owner caps"
        )
    return best[2], best[0], explored


def _validate_frame_minimums(candidates: Sequence[Candidate]) -> dict[str, int]:
    family_lineages = {
        family: {
            candidate.lineage_id
            for candidate in candidates
            if candidate.family == family
        }
        for family in FAMILIES
    }
    counts = {family: len(lineages) for family, lineages in family_lineages.items()}
    for family, minimum in MINIMUM_ELIGIBLE_LINEAGES.items():
        if counts[family] < minimum:
            raise VercelPopulationSelectionError(
                f"family {family} has {counts[family]} lineages; minimum is {minimum}"
            )
    unique = len({candidate.lineage_id for candidate in candidates})
    if unique < MINIMUM_UNIQUE_LINEAGES:
        raise VercelPopulationSelectionError(
            f"frame has {unique} unique lineages; minimum is {MINIMUM_UNIQUE_LINEAGES}"
        )
    return {**counts, "unique": unique}


def _slot_ids(category: str, quota: int) -> list[str]:
    return [f"{category}/{index:03d}" for index in range(1, quota + 1)]


def _constraint_receipt(
    assignments: Sequence[Assignment], quotas: Mapping[str, int]
) -> dict[str, Any]:
    counts = defaultdict(int)
    lineages: set[str] = set()
    records: set[str] = set()
    for assignment in assignments:
        counts[assignment.category] += 1
        lineage = assignment.candidate.lineage_id
        record_id = assignment.candidate.record.record_id
        if lineage in lineages or record_id in records:
            raise VercelPopulationSelectionError(
                "assignment reuses a lineage or public pull request"
            )
        lineages.add(lineage)
        records.add(record_id)
    if dict(counts) != dict(quotas):
        raise VercelPopulationSelectionError(
            "assignment does not satisfy exact slot quotas"
        )
    if _first_owner_violation(assignments) is not None:
        raise VercelPopulationSelectionError("assignment violates frozen owner caps")
    value = {
        "selected_count": len(assignments),
        "category_counts": {
            category: counts[category] for category in _expected_categories()
        },
        "unique_lineage_count": len(lineages),
        "unique_record_count": len(records),
        "owner_caps_passed": True,
        "partition_order": list(PARTITION_ORDER),
        "slot_manifest_sha256": stable_digest(_slot_manifest()),
    }
    value["constraint_check_sha256"] = stable_digest(value)
    return value


def select_vercel_population(artifacts: SelectionArtifactBytes) -> dict[str, Any]:
    if not isinstance(artifacts, SelectionArtifactBytes):
        raise VercelPopulationSelectionError(
            "selection requires exact SelectionArtifactBytes, not caller-supplied digests"
        )
    plan_value = _json_bytes(
        artifacts.plan,
        "selection plan",
        EXPECTED_BOUNDS["max_plan_bytes"],
    )
    receipt_limit = EXPECTED_BOUNDS["max_receipt_bytes"]
    protocol = _json_bytes(artifacts.protocol, "protocol", receipt_limit)
    query_plan = _json_bytes(
        artifacts.query_plan,
        "query plan",
        EXPECTED_BOUNDS["max_query_plan_bytes"],
    )
    protocol_unsigned = dict(protocol)
    protocol_digest = protocol_unsigned.pop("artifact_digest", None)
    if protocol_digest != stable_digest(protocol_unsigned):
        raise VercelPopulationSelectionError("protocol artifact self-digest differs")
    query_unsigned = dict(query_plan)
    query_digest = query_unsigned.pop("artifact_digest", None)
    if query_digest != stable_digest(query_unsigned):
        raise VercelPopulationSelectionError("query plan artifact self-digest differs")
    family_queries, trust_roots = _validate_exact_protocol_and_query(
        protocol, query_plan
    )
    selection_id = _safe_id(plan_value.get("selection_id"), "selection plan.selection_id")
    code_bytes = _bounded_bytes(
        artifacts.selection_code, "selection code", receipt_limit
    )
    executing_code = Path(__file__).read_bytes()
    if code_bytes != executing_code:
        raise VercelPopulationSelectionError(
            "selection_code bytes do not equal the executing selector"
        )
    if trust_roots is None:
        return _blocked_receipt(
            selection_id,
            blockers=[
                "protocol_trust_roots_not_registered",
                "no_randomness_or_assignment_was_computed",
            ],
            plan_sha256=_sha256_bytes(artifacts.plan),
        )

    normalized_plan = validate_plan(plan_value)
    population_rows = _jsonl_bytes(
        artifacts.population,
        "population records",
        EXPECTED_BOUNDS["max_population_bytes"],
    )
    review_rows = _jsonl_bytes(
        artifacts.review_ledger,
        "review ledger",
        EXPECTED_BOUNDS["max_review_ledger_bytes"],
    )
    lineage_rows = _jsonl_bytes(
        artifacts.lineage_ledger,
        "lineage ledger",
        EXPECTED_BOUNDS["max_lineage_ledger_bytes"],
    )
    if (
        max(len(population_rows), len(review_rows), len(lineage_rows))
        > EXPECTED_BOUNDS["max_population_records"]
    ):
        raise VercelPopulationSelectionError(
            "population artifacts exceed the record bound"
        )
    calibration = _json_bytes(
        artifacts.calibration_receipt,
        "reviewer calibration receipt",
        receipt_limit,
    )
    role_separation = _json_bytes(
        artifacts.role_separation_receipt,
        "role separation receipt",
        receipt_limit,
    )
    precommit = _json_bytes(
        artifacts.precommit_receipt,
        "public precommit receipt",
        receipt_limit,
    )
    verifier_lock = _json_bytes(
        artifacts.verifier_lock,
        "beacon verifier lock",
        receipt_limit,
    )
    randomness_receipt = _json_bytes(
        artifacts.randomness_receipt,
        "randomness receipt",
        receipt_limit,
    )
    solver_runtime = _json_bytes(
        artifacts.solver_runtime,
        "solver runtime receipt",
        receipt_limit,
    )
    acquisition_rows = _jsonl_bytes(
        artifacts.acquisition_candidates,
        "acquisition candidates",
        EXPECTED_BOUNDS["max_population_bytes"],
    )
    repository_rows = _jsonl_bytes(
        artifacts.repository_receipts,
        "repository receipts",
        EXPECTED_BOUNDS["max_population_bytes"],
    )
    pull_request_rows = _jsonl_bytes(
        artifacts.pull_request_receipts,
        "pull-request receipts",
        EXPECTED_BOUNDS["max_population_bytes"],
    )
    source_rows = _jsonl_bytes(
        artifacts.source_receipts,
        "source receipts",
        EXPECTED_BOUNDS["max_population_bytes"],
    )
    stack_rows = _jsonl_bytes(
        artifacts.stack_receipts,
        "stack receipts",
        EXPECTED_BOUNDS["max_population_bytes"],
    )
    calibration_rows = _jsonl_bytes(
        artifacts.calibration_ledger,
        "calibration ledger",
        EXPECTED_BOUNDS["max_review_ledger_bytes"],
    )
    review_authentication = _json_bytes(
        artifacts.review_authentication,
        "review authentication",
        receipt_limit,
    )
    source_authentication = _json_bytes(
        artifacts.source_authentication,
        "source authentication",
        receipt_limit,
    )
    beacon_response = _json_bytes(
        artifacts.beacon_response,
        "raw beacon response",
        receipt_limit,
    )
    beacon_authentication = _json_bytes(
        artifacts.beacon_authentication,
        "beacon authentication",
        receipt_limit,
    )
    power_receipt = _json_bytes(
        artifacts.power_receipt,
        "prospective power receipt",
        receipt_limit,
    )
    _bounded_bytes(artifacts.selection_code, "selection code", receipt_limit)

    actual_digests = {
        "protocol_sha256": _sha256_bytes(artifacts.protocol),
        "population_sha256": _sha256_bytes(artifacts.population),
        "review_ledger_sha256": _sha256_bytes(artifacts.review_ledger),
        "lineage_ledger_sha256": _sha256_bytes(artifacts.lineage_ledger),
        "query_plan_sha256": _sha256_bytes(artifacts.query_plan),
        "calibration_receipt_sha256": _sha256_bytes(artifacts.calibration_receipt),
        "role_separation_receipt_sha256": _sha256_bytes(
            artifacts.role_separation_receipt
        ),
        "precommit_receipt_sha256": _sha256_bytes(artifacts.precommit_receipt),
        "verifier_lock_sha256": _sha256_bytes(artifacts.verifier_lock),
        "randomness_receipt_sha256": _sha256_bytes(artifacts.randomness_receipt),
        "selection_code_sha256": _sha256_bytes(artifacts.selection_code),
        "solver_runtime_sha256": _sha256_bytes(artifacts.solver_runtime),
        "acquisition_candidates_sha256": _sha256_bytes(
            artifacts.acquisition_candidates
        ),
        "repository_receipts_sha256": _sha256_bytes(artifacts.repository_receipts),
        "pull_request_receipts_sha256": _sha256_bytes(
            artifacts.pull_request_receipts
        ),
        "source_receipts_sha256": _sha256_bytes(artifacts.source_receipts),
        "stack_receipts_sha256": _sha256_bytes(artifacts.stack_receipts),
        "calibration_ledger_sha256": _sha256_bytes(artifacts.calibration_ledger),
        "review_authentication_sha256": _sha256_bytes(
            artifacts.review_authentication
        ),
        "source_authentication_sha256": _sha256_bytes(
            artifacts.source_authentication
        ),
        "beacon_response_sha256": _sha256_bytes(artifacts.beacon_response),
        "beacon_authentication_sha256": _sha256_bytes(
            artifacts.beacon_authentication
        ),
        "power_receipt_sha256": _sha256_bytes(artifacts.power_receipt),
    }
    for field, actual in actual_digests.items():
        if normalized_plan[field] != actual:
            raise VercelPopulationSelectionError(
                f"{field} bytes differ from selection plan"
            )

    normalized_precommit, committed_at = _validate_precommit_receipt(
        precommit,
        plan=normalized_plan,
        source_key_id=trust_roots["source_verification"],
    )
    normalized_verifier = _validate_verifier_lock(
        verifier_lock,
        plan=normalized_plan,
        committed_at=committed_at,
    )
    _validate_solver_runtime(solver_runtime)
    role_ids = _validate_role_separation_receipt(role_separation)
    calibrated_at, _calibrated_reviewers = _validate_calibration_receipt(
        calibration,
        role_ids=role_ids,
    )
    records = _validate_population_rows(population_rows)
    population_frozen_at = _utc(
        normalized_plan["population_frozen_at_utc"], "population frozen time"
    )
    normalized_lineages, record_queries = _validate_source_evidence(
        records=records,
        lineage_rows=lineage_rows,
        acquisition_rows=acquisition_rows,
        repository_rows=repository_rows,
        pull_request_rows=pull_request_rows,
        source_rows=source_rows,
        stack_rows=stack_rows,
        family_queries=family_queries,
        population_frozen_at=population_frozen_at,
    )
    calibration_ledger_digest = _validate_calibration_ledger(
        calibration_rows,
        calibration_receipt=calibration,
        role_ids=role_ids,
    )
    if calibration_ledger_digest != stable_digest(calibration_rows):
        raise VercelPopulationSelectionError(
            "calibration ledger normalization is not deterministic"
        )
    eligible = _validate_review_rows(
        review_rows,
        records,
        normalized_lineages,
        calibration_receipt_sha256=actual_digests["calibration_receipt_sha256"],
        role_separation_receipt_sha256=actual_digests["role_separation_receipt_sha256"],
        calibrated_at=calibrated_at,
        committed_at=committed_at,
        registered_role_ids=role_ids,
    )
    review_auth = _validate_review_authentication(
        review_authentication,
        review_rows=review_rows,
        review_ledger_sha256=actual_digests["review_ledger_sha256"],
        calibration_ledger_sha256=actual_digests["calibration_ledger_sha256"],
        role_separation_receipt_sha256=actual_digests[
            "role_separation_receipt_sha256"
        ],
        role_ids=role_ids,
        governance_key_id=trust_roots["review_governance"],
    )
    governance_auth = _mapping(
        review_auth["governance_authentication"], "review governance authentication"
    )
    governance_authenticated_at = _utc(
        governance_auth["authenticated_at_utc"],
        "review governance authentication time",
    )
    if not calibrated_at <= governance_authenticated_at <= committed_at:
        raise VercelPopulationSelectionError(
            "review governance authentication is outside the prospective boundary"
        )
    for record, family, _lineage in eligible:
        if not record_queries[record.record_id] & family_queries[family]:
            raise VercelPopulationSelectionError(
                f"eligible record {record.record_id} has no acquisition query for {family}"
            )

    source_auth = _validate_authentication(
        source_authentication,
        purpose="source_verification",
        payload=_source_verification_payload(
            plan=normalized_plan,
            actual_digests=actual_digests,
            record_count=len(records),
        ),
        expected_key_id=trust_roots["source_verification"],
    )
    source_authenticated_at = _utc(
        source_auth["authenticated_at_utc"],
        "source verification authentication time",
    )
    if not population_frozen_at <= source_authenticated_at <= committed_at:
        raise VercelPopulationSelectionError(
            "source evidence authentication is outside the frozen prospective boundary"
        )
    power = _validate_power_receipt(
        power_receipt,
        power_key_id=trust_roots["power_verification"],
    )
    power_auth = _mapping(power["authentication"], "power authentication")
    if _utc(
        power_auth["authenticated_at_utc"], "power authentication time"
    ) > committed_at:
        raise VercelPopulationSelectionError(
            "prospective power was authenticated after public precommit"
        )
    _validate_beacon_response(
        beacon_response,
        plan=normalized_plan,
        randomness_receipt=randomness_receipt,
    )
    beacon_auth = _validate_authentication(
        beacon_authentication,
        purpose="beacon_verification",
        payload=_beacon_verification_payload(
            plan=normalized_plan,
            precommit=normalized_precommit,
            verifier_lock=normalized_verifier,
            randomness_receipt=randomness_receipt,
            actual_digests=actual_digests,
        ),
        expected_key_id=trust_roots["beacon_verification"],
    )
    if beacon_auth["authenticated_at_utc"] != randomness_receipt["verified_at_utc"]:
        raise VercelPopulationSelectionError(
            "beacon detached authentication time differs from normalized verification"
        )
    random_value = _validate_randomness_receipt(
        randomness_receipt,
        plan=normalized_plan,
        committed_at=committed_at,
        verifier_lock=normalized_verifier,
    )
    seed = _master_seed(
        random_value,
        population_sha256=actual_digests["population_sha256"],
        review_ledger_sha256=actual_digests["review_ledger_sha256"],
        query_plan_sha256=actual_digests["query_plan_sha256"],
        selection_code_sha256=actual_digests["selection_code_sha256"],
    )
    candidates, collapsed_duplicates = _canonical_candidates(eligible, seed=seed)
    frame_counts = _validate_frame_minimums(candidates)
    quotas = _expected_categories()
    assignments, objective, explored = _solve_constrained(
        candidates,
        quotas,
        max_search_states=EXPECTED_BOUNDS["max_search_states"],
        max_graph_edges=EXPECTED_BOUNDS["max_graph_edges"],
        max_pending_search_states=EXPECTED_BOUNDS["max_pending_search_states"],
    )

    assignments_by_category: dict[str, list[Assignment]] = defaultdict(list)
    for assignment in assignments:
        assignments_by_category[assignment.category].append(assignment)
    selected_rows: list[dict[str, Any]] = []
    for category in _expected_categories():
        values = sorted(
            assignments_by_category[category],
            key=lambda item: (
                item.candidate.score_sha256,
                item.candidate.record.canonical_pull_request_url,
                item.candidate.lineage_id,
            ),
        )
        slots = _slot_ids(category, quotas[category])
        if len(values) != len(slots):
            raise VercelPopulationSelectionError(
                f"category {category} quota is incomplete"
            )
        for slot, assignment in zip(slots, values, strict=True):
            partition, family = _split_category(category)
            selected_rows.append(
                {
                    "slot_id": slot,
                    "partition": partition,
                    "family": family,
                    "record_id": assignment.candidate.record.record_id,
                    "repository_id": assignment.candidate.record.repository_id,
                    "owner_id": assignment.candidate.record.owner_id,
                    "lineage_id": assignment.candidate.lineage_id,
                    "canonical_pull_request_url": (
                        assignment.candidate.record.canonical_pull_request_url
                    ),
                    "source_record_sha256": (
                        assignment.candidate.record.source_record_sha256
                    ),
                    "acquisition_candidate_sha256": (
                        assignment.candidate.record.acquisition_candidate_sha256
                    ),
                    "repository_receipt_sha256": (
                        assignment.candidate.record.repository_receipt_sha256
                    ),
                    "pull_request_receipt_sha256": (
                        assignment.candidate.record.pull_request_receipt_sha256
                    ),
                    "source_receipt_sha256": (
                        assignment.candidate.record.source_receipt_sha256
                    ),
                    "stack_receipt_sha256": (
                        assignment.candidate.record.stack_receipt_sha256
                    ),
                    "record_score_sha256": assignment.candidate.score_sha256,
                }
            )
    selected_rows.sort(key=lambda row: _slot_manifest().index(row["slot_id"]))
    constraint_receipt = _constraint_receipt(assignments, quotas)
    seed_receipt = {
        "precommit_receipt_sha256": actual_digests["precommit_receipt_sha256"],
        "verifier_lock_sha256": actual_digests["verifier_lock_sha256"],
        "randomness_receipt_sha256": actual_digests["randomness_receipt_sha256"],
        "commitment_sha256": normalized_precommit["commitment_sha256"],
        "master_seed_sha256": seed,
    }
    seed_receipt_sha256 = stable_digest(seed_receipt)
    selected_assignment_sha256 = stable_digest(selected_rows)
    replacement_receipt: dict[str, Any] = {
        "status": "non_executable_complete_residual_augmentation_not_frozen",
        "replacement_executable": False,
        "selected_assignment_sha256": selected_assignment_sha256,
        "required_next_artifact": (
            "a separately reviewed complete residual-augmentation table covering "
            "every allowed failure set under the frozen quotas and owner caps"
        ),
        "failure_policy": "stop_without_replacement_or_new_selection",
    }
    replacement_receipt["receipt_sha256"] = stable_digest(replacement_receipt)
    required_receipts = {
        "seed_receipt_sha256": seed_receipt_sha256,
        "population_digest": actual_digests["population_sha256"],
        "eligibility_ledger_digest": actual_digests["review_ledger_sha256"],
        "lineage_ledger_digest": actual_digests["lineage_ledger_sha256"],
        "selection_code_sha256": actual_digests["selection_code_sha256"],
        "slot_manifest_sha256": normalized_plan["slot_manifest_sha256"],
        "selected_assignment_sha256": selected_assignment_sha256,
        "replacement_receipt_sha256": replacement_receipt["receipt_sha256"],
        "constraint_check_sha256": constraint_receipt["constraint_check_sha256"],
        "solver_runtime_sha256": actual_digests["solver_runtime_sha256"],
        "source_authentication_sha256": actual_digests[
            "source_authentication_sha256"
        ],
        "review_authentication_sha256": actual_digests[
            "review_authentication_sha256"
        ],
        "beacon_authentication_sha256": actual_digests[
            "beacon_authentication_sha256"
        ],
        "power_receipt_sha256": actual_digests["power_receipt_sha256"],
    }
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "selection_id": normalized_plan["selection_id"],
        "status": "selected_pending_task_authoring_power_and_qualification",
        "assignment_emitted": True,
        "claim_boundary": (
            "Selection covers only the frozen reviewed Vercel query-discoverable frame; "
            "it is not experiment evidence or a universal React or Next.js claim."
        ),
        "plan_sha256": _sha256_bytes(artifacts.plan),
        "bindings": actual_digests,
        "required_protocol_receipts": required_receipts,
        "seed_receipt": seed_receipt,
        "randomness": {
            "provider": randomness_receipt["provider"],
            "pulse_or_round": randomness_receipt["pulse_or_round"],
            "receipt_sha256": randomness_receipt["receipt_sha256"],
            "master_seed_sha256": seed,
        },
        "constraints": {
            "quotas": EXPECTED_QUOTAS,
            "owner_caps": {"all_partitions": 4, "target_holdout": 2},
            "one_selected_task_per_lineage": True,
        },
        "frame_lineage_counts": frame_counts,
        "collapsed_lineage_family_duplicates": collapsed_duplicates,
        "selected_count": len(selected_rows),
        "selected": selected_rows,
        "replacement": replacement_receipt,
        "constraint_receipt": constraint_receipt,
        "optimization": {
            "algorithm": "exact_min_cost_flow_with_owner_constraint_branching_v2",
            "search_states_explored": explored,
            "objective_sha256": hashlib.sha256(str(objective).encode()).hexdigest(),
        },
        "limitations": [
            "Selection does not establish task authorability or base/gold validity.",
            "No replacement is executable because complete residual augmentation is not frozen.",
            "Any selected-source failure stops this selection pending a new reviewed artifact.",
        ],
    }
    receipt["receipt_sha256"] = stable_digest(receipt)
    return receipt


def _open_nofollow_parent(
    path: Path,
    *,
    field: str,
    create_private_parent: bool,
) -> tuple[int, str]:
    """Open every ancestor by file descriptor without following symbolic links."""

    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise VercelPopulationSelectionError(
            f"{field} requires O_NOFOLLOW and O_DIRECTORY support"
        )
    if ".." in path.parts:
        raise VercelPopulationSelectionError(f"{field} must not contain parent traversal")
    absolute = path if path.is_absolute() else Path.cwd() / path
    parts = absolute.parts
    if not parts or parts[0] != os.sep or len(parts) < 2 or not parts[-1]:
        raise VercelPopulationSelectionError(f"{field} path is invalid")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(os.sep, directory_flags)
    try:
        parent_parts = parts[1:-1]
        for index, component in enumerate(parent_parts):
            try:
                child = os.open(component, directory_flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create_private_parent or index != len(parent_parts) - 1:
                    raise
                os.mkdir(component, mode=0o700, dir_fd=descriptor)
                child = os.open(component, directory_flags, dir_fd=descriptor)
            child_stat = os.fstat(child)
            if not stat.S_ISDIR(child_stat.st_mode):
                os.close(child)
                raise VercelPopulationSelectionError(
                    f"{field} ancestor {component} is not a directory"
                )
            os.close(descriptor)
            descriptor = child
        return descriptor, parts[-1]
    except (OSError, VercelPopulationSelectionError) as exc:
        os.close(descriptor)
        if isinstance(exc, VercelPopulationSelectionError):
            raise
        raise VercelPopulationSelectionError(
            f"cannot open no-follow parent for {field}: {exc}"
        ) from exc


def _write_private_json(path: Path, value: Mapping[str, Any]) -> None:
    parent_descriptor, leaf = _open_nofollow_parent(
        path,
        field="selection output",
        create_private_parent=True,
    )
    try:
        parent_mode = stat.S_IMODE(os.fstat(parent_descriptor).st_mode)
        if parent_mode & 0o077:
            raise VercelPopulationSelectionError(
                "selection output parent must not be accessible by group or other users"
            )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        try:
            descriptor = os.open(
                leaf,
                flags,
                stat.S_IRUSR | stat.S_IWUSR,
                dir_fd=parent_descriptor,
            )
        except OSError as exc:
            raise VercelPopulationSelectionError(
                f"selection output must be a new non-symlink path: {exc}"
            ) from exc
    finally:
        os.close(parent_descriptor)
    os.fchmod(descriptor, stat.S_IRUSR | stat.S_IWUSR)
    if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o600:
        os.close(descriptor)
        raise VercelPopulationSelectionError(
            "selection output could not be restricted to mode 0600"
        )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(
            value,
            stream,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        stream.write("\n")


def _read_regular_file(path: Path, field: str) -> bytes:
    parent_descriptor, leaf = _open_nofollow_parent(
        path,
        field=field,
        create_private_parent=False,
    )
    try:
        descriptor = os.open(
            leaf,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
    except OSError as exc:
        raise VercelPopulationSelectionError(f"cannot read {field}: {exc}") from exc
    finally:
        os.close(parent_descriptor)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise VercelPopulationSelectionError(f"{field} changed file type")
    with os.fdopen(descriptor, "rb") as stream:
        return stream.read()


def _read_optional_regular_file(path: Path | None, field: str) -> bytes:
    return b"" if path is None else _read_regular_file(path, field)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select the frozen Vercel V2 public population."
    )
    parser.add_argument("selection_plan", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("population_records", type=Path)
    parser.add_argument("review_ledger", type=Path)
    parser.add_argument("lineage_ledger", type=Path)
    parser.add_argument("query_plan", type=Path)
    parser.add_argument("calibration_receipt", type=Path)
    parser.add_argument("role_separation_receipt", type=Path)
    parser.add_argument("precommit_receipt", type=Path)
    parser.add_argument("verifier_lock", type=Path)
    parser.add_argument("randomness_receipt", type=Path)
    parser.add_argument("solver_runtime", type=Path)
    parser.add_argument("--acquisition-candidates", type=Path)
    parser.add_argument("--repository-receipts", type=Path)
    parser.add_argument("--pull-request-receipts", type=Path)
    parser.add_argument("--source-receipts", type=Path)
    parser.add_argument("--stack-receipts", type=Path)
    parser.add_argument("--calibration-ledger", type=Path)
    parser.add_argument("--review-authentication", type=Path)
    parser.add_argument("--source-authentication", type=Path)
    parser.add_argument("--beacon-response", type=Path)
    parser.add_argument("--beacon-authentication", type=Path)
    parser.add_argument("--power-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        artifacts = SelectionArtifactBytes(
            plan=_read_regular_file(args.selection_plan, "selection plan"),
            protocol=_read_regular_file(args.protocol, "protocol"),
            population=_read_regular_file(args.population_records, "population"),
            review_ledger=_read_regular_file(args.review_ledger, "review ledger"),
            lineage_ledger=_read_regular_file(args.lineage_ledger, "lineage ledger"),
            query_plan=_read_regular_file(args.query_plan, "query plan"),
            calibration_receipt=_read_regular_file(
                args.calibration_receipt, "calibration receipt"
            ),
            role_separation_receipt=_read_regular_file(
                args.role_separation_receipt, "role separation receipt"
            ),
            precommit_receipt=_read_regular_file(
                args.precommit_receipt, "precommit receipt"
            ),
            verifier_lock=_read_regular_file(args.verifier_lock, "verifier lock"),
            randomness_receipt=_read_regular_file(
                args.randomness_receipt, "randomness receipt"
            ),
            solver_runtime=_read_regular_file(args.solver_runtime, "solver runtime"),
            selection_code=Path(__file__).read_bytes(),
            acquisition_candidates=_read_optional_regular_file(
                args.acquisition_candidates, "acquisition candidates"
            ),
            repository_receipts=_read_optional_regular_file(
                args.repository_receipts, "repository receipts"
            ),
            pull_request_receipts=_read_optional_regular_file(
                args.pull_request_receipts, "pull-request receipts"
            ),
            source_receipts=_read_optional_regular_file(
                args.source_receipts, "source receipts"
            ),
            stack_receipts=_read_optional_regular_file(
                args.stack_receipts, "stack receipts"
            ),
            calibration_ledger=_read_optional_regular_file(
                args.calibration_ledger, "calibration ledger"
            ),
            review_authentication=_read_optional_regular_file(
                args.review_authentication, "review authentication"
            ),
            source_authentication=_read_optional_regular_file(
                args.source_authentication, "source authentication"
            ),
            beacon_response=_read_optional_regular_file(
                args.beacon_response, "beacon response"
            ),
            beacon_authentication=_read_optional_regular_file(
                args.beacon_authentication, "beacon authentication"
            ),
            power_receipt=_read_optional_regular_file(
                args.power_receipt, "power receipt"
            ),
        )
        receipt = select_vercel_population(artifacts)
        _write_private_json(args.output, receipt)
    except (OSError, VercelPopulationSelectionError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}), file=sys.stderr)
        return 2
    summary = {
        "status": receipt["status"],
        "selection_id": receipt["selection_id"],
        "assignment_emitted": receipt["assignment_emitted"],
        "receipt_sha256": receipt["receipt_sha256"],
    }
    if receipt["assignment_emitted"]:
        summary["selected_count"] = receipt["selected_count"]
    print(json.dumps(summary, sort_keys=True))
    return 0 if receipt["assignment_emitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
