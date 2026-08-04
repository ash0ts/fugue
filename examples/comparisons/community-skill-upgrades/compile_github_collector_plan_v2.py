"""Compile repository-specific scientific query plans into one collector ABI.

The checked-in lane plans retain their domain-specific claim boundaries and
review protocols. This compiler produces a strict, executable intermediate
plan for :mod:`github_population_v2` without copying query logic by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

COLLECTOR_PATH = Path(__file__).with_name("github_population_v2.py")
SPEC = importlib.util.spec_from_file_location(
    "github_population_v2_compiler", COLLECTOR_PATH
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - installation corruption
    raise RuntimeError(f"cannot load collector contract from {COLLECTOR_PATH}")
COLLECTOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = COLLECTOR
SPEC.loader.exec_module(COLLECTOR)


class CollectorPlanCompileError(ValueError):
    """Raised when a lane query plan cannot be compiled without ambiguity."""


class CollectorPublicationRecoveryRequired(CollectorPlanCompileError):
    """Raised when publication committed but durability must be recovered."""


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CollectorPlanCompileError(f"{field} must be an object")
    return value


def _list(value: object, field: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise CollectorPlanCompileError(f"{field} must be a non-empty list")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CollectorPlanCompileError(f"{field} must be non-empty text")
    return value


def _scope(value: object, field: str) -> str:
    if isinstance(value, str):
        return _text(value, field)
    return json.dumps(_mapping(value, field), sort_keys=True, separators=(",", ":"))


def _expect(value: object, expected: object, field: str) -> None:
    if value != expected:
        raise CollectorPlanCompileError(
            f"{field} violates the locked lane protocol; "
            f"expected {expected!r}, observed {value!r}"
        )


def _expect_fields(
    value: Mapping[str, Any], expected: Mapping[str, object], field: str
) -> None:
    for name, locked in expected.items():
        _expect(value.get(name), locked, f"{field}.{name}")


def _strict_json_loads(value: bytes, *, field: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise CollectorPlanCompileError(
                    f"{field} contains duplicate JSON key {key!r}"
                )
            result[key] = item
        return result

    try:
        return json.loads(value, object_pairs_hook=reject_duplicates)
    except CollectorPlanCompileError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CollectorPlanCompileError(f"{field} is malformed JSON") from exc


def _reject_symlink_ancestors(path: Path, *, field: str) -> None:
    absolute = path.absolute()
    for candidate in (absolute, *absolute.parents):
        try:
            if candidate.is_symlink():
                raise CollectorPlanCompileError(
                    f"{field} may not contain a symlink ancestor: {candidate}"
                )
        except OSError as exc:
            raise CollectorPlanCompileError(f"cannot inspect {field}: {exc}") from exc


def _public_query(value: str, *, query_id: str) -> str:
    tokens = value.split()
    if any(token in {"is:private", "visibility:private"} for token in tokens):
        raise CollectorPlanCompileError(
            f"query {query_id} contains a private-visibility boundary"
        )
    if tokens.count("is:public") > 1:
        raise CollectorPlanCompileError(
            f"query {query_id} contains duplicate public-visibility qualifiers"
        )
    return value if "is:public" in tokens else f"{value} is:public"


def _strip_temporal_clause(template: str, qualifier: str, tokens: Sequence[str]) -> str:
    expected = f"{qualifier}:{tokens[0]}..{tokens[1]}"
    if template.count(expected) != 1:
        raise CollectorPlanCompileError(
            f"query must contain exactly one compiler-owned temporal clause: {expected}"
        )
    return " ".join(template.replace(expected, "").split())


_PROTOCOL_IDS = {
    "superpowers": "superpowers-writing-plans-conference-sampling-frame-protocol-v2",
    "anthropic": "anthropic-skill-creator-conference-sampling-frame-protocol-v2",
    "vercel": "vercel-react-best-practices-conference-sampling-frame-protocol-v2",
}
_QUERY_REFERENCE_PATHS = {
    "superpowers": (
        "examples/comparisons/superpowers-writing-plans-upgrade/"
        "conference-query-plan-v2.json"
    ),
    "anthropic": "conference-github-candidate-query-plan-v2.json",
    "vercel": (
        "examples/comparisons/vercel-react-best-practices-upgrade/"
        "conference-query-plan-v2.json"
    ),
}
_RAW_PUBLIC_DATA_PRIVACY = {
    "source_bytes_are_immutable": True,
    "storage": (
        "Acquisition directories are mode 0700 and raw response, candidate, and "
        "receipt files are mode 0600; symbolic-link targets are rejected."
    ),
    "scan_boundary": (
        "A pinned credential-pattern scanner records only pattern IDs, affected "
        "immutable record IDs, response digests, and JSON paths. Matched values are "
        "never copied into a receipt, review ledger, task, trace, or report."
    ),
    "affected_record_policy": (
        "A token-shaped or private-value finding blocks the record from task authoring "
        "and export until a separately digested reviewer decision excludes it or a "
        "sanitized derivative is created without modifying the immutable raw source."
    ),
    "zero_finding_is_not_safety_proof": True,
}


def _canonical_artifact_digest(value: Mapping[str, Any], *, field: str) -> str:
    document = dict(value)
    claimed = document.pop("artifact_digest", None)
    if not isinstance(claimed, str) or re.fullmatch(r"[0-9a-f]{64}", claimed) is None:
        raise CollectorPlanCompileError(
            f"{field}.artifact_digest must be a lowercase SHA-256"
        )
    observed = hashlib.sha256(
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    if claimed != observed:
        raise CollectorPlanCompileError(f"{field}.artifact_digest disagrees")
    return claimed


def _validate_sampling_protocol(
    lane: str,
    *,
    source: Mapping[str, Any],
    source_sha256: str,
    protocol: Mapping[str, Any],
) -> str:
    _expect(protocol.get("schema_version"), 2, "sampling protocol schema_version")
    protocol_id = _text(protocol.get("protocol_id"), "sampling protocol protocol_id")
    _expect(protocol_id, _PROTOCOL_IDS[lane], "sampling protocol protocol_id")
    _expect(
        protocol.get("status"),
        "prospective_not_yet_sampled",
        "sampling protocol status",
    )
    _expect(
        protocol.get("raw_public_data_privacy"),
        _RAW_PUBLIC_DATA_PRIVACY,
        "sampling protocol raw_public_data_privacy",
    )
    query_reference = _mapping(
        protocol.get("query_plan"), "sampling protocol query_plan"
    )
    expected_reference_keys = (
        {"path", "artifact_digest"}
        if lane == "vercel"
        else {"path", "sha256", "required_status"}
    )
    if set(query_reference) != expected_reference_keys:
        raise CollectorPlanCompileError("sampling protocol query_plan has invalid keys")
    _expect(
        query_reference.get("path"),
        _QUERY_REFERENCE_PATHS[lane],
        "sampling protocol query_plan.path",
    )
    if lane in {"superpowers", "anthropic"}:
        _expect(
            query_reference.get("sha256"),
            source_sha256,
            "sampling protocol query_plan.sha256",
        )
        _expect(
            query_reference.get("required_status"),
            source.get("status"),
            "sampling protocol query_plan.required_status",
        )
    else:
        source_artifact_digest = _canonical_artifact_digest(
            source, field="source query plan"
        )
        _canonical_artifact_digest(protocol, field="sampling protocol")
        _expect(
            query_reference.get("artifact_digest"),
            source_artifact_digest,
            "sampling protocol query_plan.artifact_digest",
        )
    return protocol_id


def _common(
    *,
    plan_id: str,
    lane_id: str,
    source_sha256: str,
    protocol_sha256: str,
    protocol_id: str,
    population_scope: str,
    temporal_scope: str,
    cutoff: str,
    queries: list[dict[str, Any]],
    shard_granularity: str,
    repetitions: int = 2,
) -> dict[str, Any]:
    for query in queries:
        query_id = str(query["query_id"])
        query["query"] = _public_query(str(query["query"]), query_id=query_id)
        query["shard_granularity"] = shard_granularity
        query["unsplittable_overflow_policy"] = "fail_closed_new_protocol_required"
    plan = {
        "schema_version": 2,
        "plan_id": plan_id,
        "lane_id": lane_id,
        "candidate_discovery_only": True,
        "api_version": "2022-11-28",
        "source_query_plan_sha256": source_sha256,
        "source_sampling_protocol_sha256": protocol_sha256,
        "sampling_protocol_id": protocol_id,
        "population_scope": population_scope,
        "temporal_scope": temporal_scope,
        "selection_source_cutoff_utc": cutoff,
        "acquisition_repetitions": repetitions,
        "index_stabilization_seconds": 86400,
        "repeat_separation_seconds": 3600 if repetitions > 1 else 0,
        "queries": queries,
        "completeness": {
            "require_incomplete_results_false": True,
            "require_all_pages": True,
            "record_response_bytes": True,
            "fail_if_unsplittable": True,
        },
        "deduplication": {
            "primary_identity_fields": ["node_id", "id"],
            "candidate_only_not_lineage_deduplication": True,
        },
        "public_visibility": {
            "required_query_qualifier": "is:public",
            "independent_repository_lookup": True,
            "persist_search_body_before_visibility_verification": False,
            "persist_repository_response_body": False,
        },
        "credential_profile": {
            "profile_id": f"{lane_id}-github-public-read-v2",
            "token_env_name": "GITHUB_TOKEN",
            "credential_required": True,
            "credential_value_serialized": False,
        },
        "rate_limit_policy": {
            "required_header_names": [
                "x-ratelimit-limit",
                "x-ratelimit-remaining",
                "x-ratelimit-reset",
                "x-ratelimit-resource",
                "x-ratelimit-used",
            ],
            "minimum_remaining": 1,
            "maximum_wait_seconds": 60,
        },
        "max_response_bytes": 16 * 1024 * 1024,
        "collector_implementation": COLLECTOR.current_implementation_lock(
            compiler_path=Path(__file__)
        ),
    }
    try:
        return COLLECTOR.validate_query_plan(plan)
    except COLLECTOR.PopulationDiscoveryError as exc:
        raise CollectorPlanCompileError(
            f"compiled collector plan is invalid: {exc}"
        ) from exc


def _superpowers(
    source: Mapping[str, Any],
    source_sha256: str,
    *,
    protocol_sha256: str,
    protocol_id: str,
) -> dict[str, Any]:
    _expect(source.get("schema_version"), 2, "schema_version")
    _expect(
        source.get("id"),
        "superpowers-writing-plans-conference-query-plan-v2",
        "id",
    )
    _expect(
        source.get("status"),
        "prospective_query_plan_only_no_population_acquired",
        "status",
    )
    api = _mapping(source.get("api"), "api")
    _expect_fields(
        api,
        {
            "provider": "github_rest_api",
            "origin": "https://api.github.com",
            "endpoint": "/search/issues",
            "version": "2022-11-28",
            "accept": "application/vnd.github+json",
            "public_data_only": True,
            "credential_name": "GITHUB_TOKEN",
            "credential_value_serialized": False,
        },
        "api",
    )
    treatment = _mapping(
        source.get("treatment_time_boundary"), "treatment_time_boundary"
    )
    _expect_fields(
        treatment,
        {
            "repository": "https://github.com/obra/superpowers",
            "baseline_commit": "de4672b171213a6ff6960228d8b95c46ea0b09f4",
            "candidate_commit": "8e1262a3bae92b640d87fa81c51c53b65e490590",
            "candidate_is_direct_descendant_of_baseline": True,
            "candidate_committer_time_utc": "2026-06-16T17:09:47Z",
            "records_must_be_strictly_later_than_candidate": True,
        },
        "treatment_time_boundary",
    )
    window = _mapping(source.get("observation_window"), "observation_window")
    _expect_fields(
        window,
        {
            "start_inclusive_utc": "2026-06-16T17:09:48Z",
            "end_inclusive_utc": "2026-07-31T23:59:59Z",
            "precision": "one_second",
            "interval_semantics": "closed_nonoverlapping_second_ranges",
        },
        "observation_window",
    )
    sharding = _mapping(source.get("sharding"), "sharding")
    _expect_fields(
        sharding,
        {
            "initial_interval": "entire_observation_window",
            "algorithm": "recursively_bisect_at_whole_second_midpoint",
            "maximum_total_count_per_terminal_shard": 900,
            "github_search_hard_result_cap": 1000,
            "terminal_one_second_overflow_policy": "fail_closed_new_protocol_version_required",
            "boundaries": "closed, second-precise, adjacent, and nonoverlapping",
            "mutable_query_fallback_allowed": False,
        },
        "sharding",
    )
    pagination = _mapping(source.get("pagination"), "pagination")
    _expect_fields(
        pagination,
        {
            "per_page": 100,
            "sort": "created",
            "order": "asc",
            "follow_link_headers": True,
            "maximum_pages_per_terminal_shard": 9,
        },
        "pagination",
    )
    completeness = _mapping(source.get("completeness"), "completeness")
    _expect_fields(
        completeness,
        {
            "required_incomplete_results": False,
            "required_http_status": 200,
            "item_identity": "immutable GitHub node_id",
            "missing_or_truncated_response_policy": "fail_closed",
        },
        "completeness",
    )
    queries: list[dict[str, Any]] = []
    for index, raw_atom in enumerate(
        _list(source.get("atomic_queries"), "atomic_queries")
    ):
        atom = _mapping(raw_atom, f"atomic_queries[{index}]")
        qualifier = _text(atom.get("timestamp_qualifier"), "timestamp_qualifier")
        record_kind = _text(atom.get("record_kind"), "record_kind")
        expected_qualifier = {"issue": "closed", "pull_request": "merged"}.get(
            record_kind
        )
        if expected_qualifier is None or qualifier != expected_qualifier:
            raise CollectorPlanCompileError(
                f"atomic query {index} has an invalid record-kind/time-qualifier boundary"
            )
        if atom.get("sampling_family") not in {"target", "safety_control"}:
            raise CollectorPlanCompileError(
                f"atomic query {index} has an unsupported sampling family"
            )
        query = _strip_temporal_clause(
            _text(atom.get("query_template"), "query_template"),
            qualifier,
            ("{start}", "{end}"),
        )
        required_query_terms = (
            {"is:issue", "is:closed", "archived:false"}
            if record_kind == "issue"
            else {"is:pr", "is:merged", "archived:false"}
        )
        if not all(term in query.split() for term in required_query_terms):
            raise CollectorPlanCompileError(
                f"atomic query {index} lost its locked public record-kind filters"
            )
        queries.append(
            {
                "query_id": _text(atom.get("id"), "atomic query id"),
                "endpoint": "search/issues",
                "query": query,
                "date_qualifier": qualifier,
                "start_utc": _text(window.get("start_inclusive_utc"), "window start"),
                "end_utc": _text(window.get("end_inclusive_utc"), "window end"),
                "max_results_per_shard": sharding.get(
                    "maximum_total_count_per_terminal_shard"
                ),
                "per_page": pagination.get("per_page"),
                "sort": pagination.get("sort"),
                "order": pagination.get("order"),
            }
        )
    return _common(
        plan_id="superpowers-writing-plans-github-collector-v2",
        lane_id="superpowers-writing-plans-upgrade",
        source_sha256=source_sha256,
        protocol_sha256=protocol_sha256,
        protocol_id=protocol_id,
        population_scope=_scope(source.get("population_scope"), "population_scope"),
        temporal_scope=(
            "post-treatment public maintenance records observed after both locked Skill "
            "revisions; inference is limited to the frozen query-discoverable frame"
        ),
        cutoff=_text(window.get("end_inclusive_utc"), "window end"),
        queries=queries,
        shard_granularity="second",
    )


def _anthropic(
    source: Mapping[str, Any],
    source_sha256: str,
    *,
    protocol_sha256: str,
    protocol_id: str,
) -> dict[str, Any]:
    _expect(source.get("schema_version"), 2, "schema_version")
    _expect(
        source.get("id"),
        "anthropic-skill-creator-github-candidate-query-plan-v2",
        "id",
    )
    _expect(source.get("status"), "prospective_no_queries_executed", "status")
    api = _mapping(source.get("api_contract"), "api_contract")
    _expect_fields(
        api,
        {
            "origin": "https://api.github.com",
            "rest_api_version": "2022-11-28",
            "required_permissions": [
                "contents:read",
                "issues:read",
                "metadata:read",
                "pull_requests:read",
            ],
            "execution_boundary": "trusted setup --prepare",
            "network_allowed_in_agent_trials": False,
        },
        "api_contract",
    )
    temporal = _mapping(source.get("temporal_contract"), "temporal_contract")
    event = _mapping(temporal.get("treatment_publication_event"), "treatment event")
    _expect_fields(
        event,
        {
            "kind": "github_pull_request_created",
            "url": "https://github.com/anthropics/skills/pull/350",
            "baseline_commit": "a5bcdd7e58cdff48566bf876f0a72a2008dcefbc",
            "candidate_commit": "1ed29a03dc852d30fa6ef2ca53a67dc2c2c2c563",
            "created_at_utc": "2026-02-06T14:59:13Z",
            "merged_at_utc": "2026-02-06T21:19:33Z",
        },
        "treatment event",
    )
    _expect(
        temporal.get("primary_pre_treatment_cutoff_utc"),
        "2026-02-06T14:59:12Z",
        "temporal_contract.primary_pre_treatment_cutoff_utc",
    )
    readiness = _mapping(source.get("readiness"), "readiness")
    for field, expected in {
        "queries_executed": False,
        "population_records_present": 0,
        "authorizes_network_preparation": False,
        "authorizes_preview": False,
        "authorizes_model_calls_or_spend": False,
    }.items():
        _expect(readiness.get(field), expected, f"readiness.{field}")
    discovery = _mapping(source.get("repository_discovery"), "repository_discovery")
    _expect(
        discovery.get("endpoint"),
        "/search/repositories",
        "repository_discovery.endpoint",
    )
    sharding = _mapping(
        discovery.get("date_sharding"), "repository_discovery.date_sharding"
    )
    _expect_fields(
        sharding,
        {
            "initial_start_utc": "2008-01-01T00:00:00Z",
            "initial_end_utc": "2026-02-06T14:59:12Z",
            "maximum_results_per_leaf": 900,
            "deduplication_key": "GitHub repository node_id",
        },
        "repository_discovery.date_sharding",
    )
    queries: list[dict[str, Any]] = []
    for index, raw_template in enumerate(
        _list(discovery.get("query_templates"), "repository_discovery.query_templates")
    ):
        template = _mapping(raw_template, f"query_templates[{index}]")
        query = _strip_temporal_clause(
            _text(template.get("query"), "query template"),
            "created",
            ("{start_utc}", "{end_utc}"),
        )
        if "fork:false" not in query.split() or "archived:false" not in query.split():
            raise CollectorPlanCompileError(
                f"query template {index} lost its public non-fork/non-archived boundary"
            )
        queries.append(
            {
                "query_id": _text(template.get("id"), "query template id"),
                "endpoint": "search/repositories",
                "query": query,
                "date_qualifier": "created",
                "start_utc": _text(sharding.get("initial_start_utc"), "shard start"),
                "end_utc": _text(sharding.get("initial_end_utc"), "shard end"),
                "max_results_per_shard": sharding.get("maximum_results_per_leaf"),
                "per_page": 100,
                "sort": None,
                "order": None,
            }
        )
    target_frame = _mapping(source.get("target_frame"), "target_frame")
    return _common(
        plan_id="anthropic-skill-creator-github-collector-v2",
        lane_id="anthropic-skill-creator-upgrade",
        source_sha256=source_sha256,
        protocol_sha256=protocol_sha256,
        protocol_id=protocol_id,
        population_scope=json.dumps(
            target_frame, sort_keys=True, separators=(",", ":")
        ),
        temporal_scope="primary pre-treatment public Skill-package repository frame",
        cutoff=_text(
            temporal.get("primary_pre_treatment_cutoff_utc"), "pre-treatment cutoff"
        ),
        queries=queries,
        shard_granularity="second",
    )


def _vercel(
    source: Mapping[str, Any],
    source_sha256: str,
    *,
    protocol_sha256: str,
    protocol_id: str,
) -> dict[str, Any]:
    _expect(source.get("schema_version"), 2, "schema_version")
    _expect(
        source.get("query_plan_id"),
        "vercel-react-best-practices-conference-query-plan-v2",
        "query_plan_id",
    )
    _expect(source.get("status"), "prospective_not_executed", "status")
    preperiod = _mapping(source.get("treatment_preperiod"), "treatment_preperiod")
    _expect_fields(
        preperiod,
        {
            "baseline_commit": "ac6a79af08f6d32c34ee03c829824990f3de0a6d",
            "baseline_committed_at_utc": "2026-01-18T01:42:22Z",
            "candidate_commit": "20987af2f1bc17857b55e7758af8bed91c364ff5",
            "candidate_committed_at_utc": "2026-01-20T13:43:49Z",
            "record_window_start_utc": "2023-01-01T00:00:00Z",
            "record_cutoff_utc": "2026-01-17T23:59:59Z",
            "github_merged_qualifier": "merged:2023-01-01..2026-01-17",
        },
        "treatment_preperiod",
    )
    github = _mapping(source.get("github_api"), "github_api")
    _expect_fields(
        github,
        {
            "base_url": "https://api.github.com",
            "endpoint": "/search/issues",
            "method": "GET",
            "accept": "application/vnd.github+json",
            "api_version": "2022-11-28",
            "maximum_retrievable_results_per_query": 900,
        },
        "github_api",
    )
    fixed = _mapping(github.get("fixed_parameters"), "github_api.fixed_parameters")
    _expect_fields(
        fixed,
        {"sort": "created", "order": "asc", "per_page": 100},
        "github_api.fixed_parameters",
    )
    sharding = _mapping(github.get("sharding"), "github_api.sharding")
    _expect_fields(
        sharding,
        {
            "field": "merged",
            "interval_granularity": "whole_utc_day",
            "algorithm": "bisect closed adjacent nonoverlapping whole-UTC-day ranges until every terminal shard reports at most 900 results",
            "single_day_overflow": "fail_closed_new_protocol_version_required",
            "boundaries": "closed_adjacent_nonoverlapping_utc_days",
            "query_mutation_after_freeze": "forbidden",
        },
        "github_api.sharding",
    )
    repetition = _mapping(
        github.get("acquisition_repetition"),
        "github_api.acquisition_repetition",
    )
    _expect_fields(
        repetition,
        {
            "acquisitions": 2,
            "minimum_server_time_separation_seconds": 3600,
            "immediate_replay_counts_as_independent": False,
            "required_cross_acquisition_check": (
                "exact per-query terminal-shard boundaries, counts, and ordered "
                "identity digests"
            ),
        },
        "github_api.acquisition_repetition",
    )
    union = _mapping(source.get("union_and_screening"), "union_and_screening")
    _expect_fields(
        union,
        {
            "union_key": "GitHub pull-request node_id",
            "canonical_record_key": "lower-case canonical repository URL plus pull-request number",
            "query_overlap_policy": "retain every query and page receipt, then deduplicate records without changing inclusion probability metadata",
        },
        "union_and_screening",
    )
    raw_families = _list(source.get("families"), "families")
    expected_family_by_query: dict[str, str] = {}
    for family_index, raw_family in enumerate(raw_families):
        family = _mapping(raw_family, f"families[{family_index}]")
        family_id = _text(family.get("id"), f"families[{family_index}].id")
        if family.get("role") not in {
            "changed_rule_primary",
            "unchanged_rule_safety_control",
        }:
            raise CollectorPlanCompileError(f"family {family_id} has an invalid role")
        _text(
            family.get("upstream_rule_path"), f"family {family_id}.upstream_rule_path"
        )
        for query_id in _list(family.get("query_ids"), f"family {family_id}.query_ids"):
            query_id = _text(query_id, f"family {family_id} query id")
            if query_id in expected_family_by_query:
                raise CollectorPlanCompileError(
                    f"query {query_id} belongs to more than one Vercel family"
                )
            expected_family_by_query[query_id] = family_id
    queries: list[dict[str, Any]] = []
    observed_query_ids: set[str] = set()
    for index, raw_query in enumerate(_list(source.get("queries"), "queries")):
        query_value = _mapping(raw_query, f"queries[{index}]")
        query_id = _text(query_value.get("id"), "query id")
        family_id = _text(query_value.get("family"), f"query {query_id}.family")
        if expected_family_by_query.get(query_id) != family_id:
            raise CollectorPlanCompileError(
                f"query {query_id} is not aligned to exactly one locked family"
            )
        observed_query_ids.add(query_id)
        rendered = _text(query_value.get("query"), "query")
        matches = re.findall(r"(?:^|\s)merged:[^\s]+", rendered)
        if len(matches) != 1:
            raise CollectorPlanCompileError(
                "every Vercel query must have exactly one source-owned merged qualifier"
            )
        if matches[0].strip() != preperiod["github_merged_qualifier"]:
            raise CollectorPlanCompileError(
                f"query {query_id} left the locked treatment preperiod"
            )
        base_query = " ".join(rendered.replace(matches[0], " ").split())
        if (
            not {"is:pr", "is:merged"} <= set(base_query.split())
            or "-repo:vercel-labs/agent-skills" not in base_query.split()
            or " OR " in f" {base_query} "
        ):
            raise CollectorPlanCompileError(
                f"query {query_id} lost its locked public comparison boundary"
            )
        queries.append(
            {
                "query_id": query_id,
                "endpoint": "search/issues",
                "query": base_query,
                "date_qualifier": "merged",
                "start_utc": _text(
                    preperiod.get("record_window_start_utc"), "record window start"
                ),
                "end_utc": _text(preperiod.get("record_cutoff_utc"), "record cutoff"),
                "max_results_per_shard": 900,
                "per_page": fixed.get("per_page"),
                "sort": fixed.get("sort"),
                "order": fixed.get("order"),
            }
        )
    if observed_query_ids != set(expected_family_by_query):
        raise CollectorPlanCompileError(
            "Vercel family membership does not cover the query union exactly"
        )
    return _common(
        plan_id="vercel-react-best-practices-github-collector-v2",
        lane_id="vercel-react-best-practices-upgrade",
        source_sha256=source_sha256,
        protocol_sha256=protocol_sha256,
        protocol_id=protocol_id,
        population_scope=_scope(source.get("claim_boundary"), "claim_boundary"),
        temporal_scope="primary pre-treatment public TypeScript Next.js repair frame",
        cutoff=_text(preperiod.get("record_cutoff_utc"), "record cutoff"),
        queries=queries,
        shard_granularity="day",
    )


COMPILERS = {
    "superpowers": _superpowers,
    "anthropic": _anthropic,
    "vercel": _vercel,
}


def compile_plan(lane: str, source_path: Path, protocol_path: Path) -> dict[str, Any]:
    if lane not in COMPILERS:
        raise CollectorPlanCompileError(f"unsupported lane {lane}")
    _reject_symlink_ancestors(source_path, field="source query plan")
    try:
        raw = source_path.read_bytes()
        _reject_symlink_ancestors(protocol_path, field="sampling protocol")
        protocol_raw = protocol_path.read_bytes()
    except OSError as exc:
        raise CollectorPlanCompileError(
            "cannot load source query plan or sampling protocol"
        ) from exc
    source = _strict_json_loads(raw, field="source query plan")
    protocol = _strict_json_loads(protocol_raw, field="sampling protocol")
    source_mapping = _mapping(source, "source query plan")
    protocol_mapping = _mapping(protocol, "sampling protocol")
    source_sha256 = hashlib.sha256(raw).hexdigest()
    protocol_id = _validate_sampling_protocol(
        lane,
        source=source_mapping,
        source_sha256=source_sha256,
        protocol=protocol_mapping,
    )
    return COMPILERS[lane](
        source_mapping,
        source_sha256,
        protocol_sha256=hashlib.sha256(protocol_raw).hexdigest(),
        protocol_id=protocol_id,
    )


_COMPILED_PUBLICATION_MARKER_KEYS = {
    "schema_version",
    "target_name",
    "temporary_name",
    "payload_sha256",
}


def _compiled_publication_marker(path: Path) -> Path:
    return path.with_name(f".{path.name}.publication-v2.json")


def _fsync_directory(path: Path) -> None:
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, os.O_RDONLY | directory_flag)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_owner_only_exclusive(path: Path, value: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    os.chmod(path, 0o600)


def _read_owner_only(path: Path, *, field: str) -> bytes:
    _reject_symlink_ancestors(path, field=field)
    metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) & 0o077:
        raise CollectorPlanCompileError(f"{field} must be an owner-only regular file")
    return path.read_bytes()


def _remove_compiled_marker(marker: Path) -> None:
    try:
        marker.unlink(missing_ok=True)
        _fsync_directory(marker.parent)
    except OSError:
        # The target publication was already durably fsynced. A marker that
        # reappears after a crash is safe: the next invocation verifies and
        # removes it through the idempotent recovery path.
        pass


def _recover_compiled_publication(path: Path, *, payload: bytes) -> bool:
    marker = _compiled_publication_marker(path)
    if not marker.exists() and not marker.is_symlink():
        return False
    value = _strict_json_loads(
        _read_owner_only(marker, field="compiled publication marker"),
        field="compiled publication marker",
    )
    document = _mapping(value, "compiled publication marker")
    if set(document) != _COMPILED_PUBLICATION_MARKER_KEYS:
        raise CollectorPlanCompileError("compiled publication marker has invalid keys")
    expected_digest = hashlib.sha256(payload).hexdigest()
    if (
        document.get("schema_version") != 2
        or document.get("target_name") != path.name
        or document.get("payload_sha256") != expected_digest
    ):
        raise CollectorPlanCompileError(
            "compiled publication marker differs from the requested plan"
        )
    temporary_name = document.get("temporary_name")
    if (
        not isinstance(temporary_name, str)
        or Path(temporary_name).name != temporary_name
        or not temporary_name.startswith(f".{path.name}.staging-")
    ):
        raise CollectorPlanCompileError(
            "compiled publication marker has an invalid temporary name"
        )
    temporary = path.parent / temporary_name
    if path.exists() or path.is_symlink():
        if _read_owner_only(path, field="recovered compiled plan") != payload:
            raise CollectorPlanCompileError(
                "recovered compiled plan differs from the transaction marker"
            )
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(path.parent)
        if temporary.exists() or temporary.is_symlink():
            if (
                _read_owner_only(temporary, field="recovered compiled temporary")
                != payload
            ):
                raise CollectorPlanCompileError(
                    "recovered compiled temporary differs from the transaction marker"
                )
            temporary.unlink()
        _remove_compiled_marker(marker)
        return True
    if temporary.exists() or temporary.is_symlink():
        if _read_owner_only(temporary, field="recovered compiled temporary") != payload:
            raise CollectorPlanCompileError(
                "recovered compiled temporary differs from the transaction marker"
            )
        try:
            os.link(temporary, path, follow_symlinks=False)
            _fsync_directory(path.parent)
        except OSError as exc:
            raise CollectorPublicationRecoveryRequired(
                "compiled plan publication requires recovery"
            ) from exc
        temporary.unlink()
        _remove_compiled_marker(marker)
        return True
    _remove_compiled_marker(marker)
    return False


def _write_compiled_plan(path: Path, plan: Mapping[str, Any]) -> None:
    _reject_symlink_ancestors(path, field="compiled plan path")
    payload = (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary: Path | None = None
    marker = _compiled_publication_marker(path)
    linked = False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _reject_symlink_ancestors(path.parent, field="compiled plan parent")
        if _recover_compiled_publication(path, payload=payload):
            return
        if path.exists() or path.is_symlink():
            raise CollectorPlanCompileError(f"refusing to overwrite {path}")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.staging-", dir=path.parent
        )
        temporary = Path(temporary_name)
        os.chmod(temporary, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        reloaded = _strict_json_loads(
            temporary.read_bytes(), field="reloaded compiled plan"
        )
        if COLLECTOR.validate_query_plan(reloaded) != dict(plan):
            raise CollectorPlanCompileError(
                "reloaded compiled plan changed after serialization"
            )
        if path.exists() or path.is_symlink():
            raise CollectorPlanCompileError(
                "compiled plan destination appeared during preparation"
            )
        marker_document = {
            "schema_version": 2,
            "target_name": path.name,
            "temporary_name": temporary.name,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
        }
        _write_owner_only_exclusive(
            marker,
            (json.dumps(marker_document, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )
        _fsync_directory(path.parent)
        try:
            os.link(temporary, path, follow_symlinks=False)
            linked = True
            _fsync_directory(path.parent)
        except OSError as exc:
            if linked:
                raise CollectorPublicationRecoveryRequired(
                    "compiled plan publication requires recovery"
                ) from exc
            raise CollectorPlanCompileError(
                "cannot atomically publish compiled plan without replacement"
            ) from exc
        temporary.unlink()
        temporary = None
        _remove_compiled_marker(marker)
    except CollectorPublicationRecoveryRequired:
        raise
    except CollectorPlanCompileError:
        raise
    except OSError as exc:
        raise CollectorPlanCompileError("cannot write compiled plan") from exc
    finally:
        if temporary is not None and not linked:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        if not linked and (marker.exists() or marker.is_symlink()):
            try:
                marker.unlink(missing_ok=True)
            except OSError:
                pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile a lane GitHub collector plan."
    )
    parser.add_argument("--lane", choices=sorted(COMPILERS), required=True)
    parser.add_argument("source_query_plan", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        compiled = compile_plan(args.lane, args.source_query_plan, args.protocol)
        _write_compiled_plan(args.output, compiled)
    except CollectorPlanCompileError as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {"status": "compiled", "plan_sha256": COLLECTOR.stable_digest(compiled)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
