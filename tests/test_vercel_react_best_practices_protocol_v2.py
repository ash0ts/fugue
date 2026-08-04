from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/comparisons/vercel-react-best-practices-upgrade"
QUERY_PLAN_PATH = EXAMPLE / "conference-query-plan-v2.json"
PROTOCOL_PATH = EXAMPLE / "conference-sampling-frame-protocol-v2.json"

QUERY_TOP_LEVEL_KEYS = {
    "schema_version",
    "query_plan_id",
    "status",
    "artifact_digest",
    "treatment_preperiod",
    "github_api",
    "families",
    "queries",
    "union_and_screening",
    "claim_boundary",
}
PROTOCOL_TOP_LEVEL_KEYS = {
    "schema_version",
    "protocol_id",
    "status",
    "artifact_digest",
    "predecessor",
    "query_plan",
    "treatment",
    "estimand",
    "authorization_gate",
    "population_eligibility",
    "immutable_evidence_contract",
    "lineage_deduplication",
    "two_reviewer_classification",
    "task_authoring_and_qualification",
    "raw_public_data_privacy",
    "external_randomness",
    "deterministic_selection",
    "quotas_and_feasibility_stop",
    "prospective_power_stop",
    "analysis_and_trace_audit",
}
FAMILY_ROLES = {
    "server_action_authorization": "changed_rule_primary",
    "rsc_serialization": "changed_rule_primary",
    "dom_batching": "unchanged_rule_safety_control",
    "large_array_iteration": "unchanged_rule_safety_control",
    "hook_timing": "unchanged_rule_safety_control",
    "event_handler_reference": "unchanged_rule_safety_control",
}


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _artifact_digest(value: dict[str, object]) -> str:
    payload = dict(value)
    payload.pop("artifact_digest")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_v2_artifacts_have_strict_identity_and_self_verifying_digests() -> None:
    query_plan = _load(QUERY_PLAN_PATH)
    protocol = _load(PROTOCOL_PATH)

    assert set(query_plan) == QUERY_TOP_LEVEL_KEYS
    assert set(protocol) == PROTOCOL_TOP_LEVEL_KEYS
    assert query_plan["schema_version"] == 2
    assert protocol["schema_version"] == 2
    assert query_plan["status"] == "prospective_not_executed"
    assert protocol["status"] == "prospective_not_yet_sampled"
    assert query_plan["artifact_digest"] == _artifact_digest(query_plan)
    assert protocol["artifact_digest"] == _artifact_digest(protocol)

    query_lock = protocol["query_plan"]
    assert isinstance(query_lock, dict)
    assert set(query_lock) == {"path", "artifact_digest"}
    assert query_lock == {
        "path": (
            "examples/comparisons/vercel-react-best-practices-upgrade/"
            "conference-query-plan-v2.json"
        ),
        "artifact_digest": query_plan["artifact_digest"],
    }
    predecessor = protocol["predecessor"]
    assert isinstance(predecessor, dict)
    assert predecessor["v1_artifact_remains_immutable"] is True


def test_atomic_query_plan_is_treatment_preperiod_and_covers_six_families() -> None:
    query_plan = _load(QUERY_PLAN_PATH)
    preperiod = query_plan["treatment_preperiod"]
    assert isinstance(preperiod, dict)
    cutoff = datetime.fromisoformat(
        str(preperiod["record_cutoff_utc"]).replace("Z", "+00:00")
    )
    baseline = datetime.fromisoformat(
        str(preperiod["baseline_committed_at_utc"]).replace("Z", "+00:00")
    )
    candidate = datetime.fromisoformat(
        str(preperiod["candidate_committed_at_utc"]).replace("Z", "+00:00")
    )
    assert cutoff < baseline < candidate
    assert preperiod["github_merged_qualifier"] == ("merged:2023-01-01..2026-01-17")

    families = query_plan["families"]
    queries = query_plan["queries"]
    assert isinstance(families, list) and isinstance(queries, list)
    assert len(families) == 6
    assert len(queries) == 27
    assert all(
        isinstance(item, dict)
        and set(item) == {"id", "role", "upstream_rule_path", "query_ids"}
        for item in families
    )
    assert all(
        isinstance(item, dict)
        and set(item) == {"id", "family", "lexical_basis", "query"}
        for item in queries
    )

    observed_roles = {str(item["id"]): str(item["role"]) for item in families}
    assert observed_roles == FAMILY_ROLES
    query_ids = [str(item["id"]) for item in queries]
    assert len(set(query_ids)) == len(query_ids)
    family_query_ids = {
        str(query_id) for family in families for query_id in family["query_ids"]
    }
    assert family_query_ids == set(query_ids)

    required_suffix = (
        "in:title,body is:pr is:merged is:public merged:2023-01-01..2026-01-17 "
        "-repo:vercel-labs/agent-skills"
    )
    for item in queries:
        query = str(item["query"])
        assert query.endswith(required_suffix)
        assert " OR " not in query
        assert query.count("is:pr") == 1
        assert query.count("is:merged") == 1
        assert query.split().count("is:public") == 1
        assert query.count("merged:2023-01-01..2026-01-17") == 1
        assert str(item["family"]) in FAMILY_ROLES


def test_github_acquisition_fails_closed_on_truncation_or_query_mutation() -> None:
    query_plan = _load(QUERY_PLAN_PATH)
    github = query_plan["github_api"]
    assert isinstance(github, dict)
    assert set(github) == {
        "base_url",
        "endpoint",
        "method",
        "accept",
        "api_version",
        "authentication",
        "fixed_parameters",
        "maximum_retrievable_results_per_query",
        "sharding",
        "acquisition_repetition",
        "required_page_receipt_fields",
        "failure_conditions",
    }
    assert github["endpoint"] == "/search/issues"
    assert github["method"] == "GET"
    assert github["fixed_parameters"] == {
        "sort": "created",
        "order": "asc",
        "per_page": 100,
    }
    assert github["maximum_retrievable_results_per_query"] == 900
    sharding = github["sharding"]
    assert isinstance(sharding, dict)
    assert sharding["field"] == "merged"
    assert sharding == {
        "field": "merged",
        "interval_granularity": "whole_utc_day",
        "algorithm": (
            "bisect closed adjacent nonoverlapping whole-UTC-day ranges until "
            "every terminal shard reports at most 900 results"
        ),
        "single_day_overflow": "fail_closed_new_protocol_version_required",
        "boundaries": "closed_adjacent_nonoverlapping_utc_days",
        "query_mutation_after_freeze": "forbidden",
    }
    assert sharding["query_mutation_after_freeze"] == "forbidden"
    assert github["acquisition_repetition"] == {
        "acquisitions": 2,
        "minimum_server_time_separation_seconds": 3600,
        "immediate_replay_counts_as_independent": False,
        "required_cross_acquisition_check": (
            "exact per-query terminal-shard boundaries, counts, and ordered "
            "identity digests"
        ),
    }
    required_fields = set(github["required_page_receipt_fields"])
    assert {
        "requested_url",
        "final_url",
        "redirect_chain",
        "request_started_at_utc",
        "request_completed_at_utc",
        "x_ratelimit_limit",
        "x_ratelimit_remaining",
        "x_ratelimit_reset",
        "x_ratelimit_resource",
        "x_ratelimit_used",
        "request_url_sha256",
        "headers_sha256",
        "body_sha256",
        "collector_source_commit",
        "collector_source_tree",
        "collector_sha256",
        "compiler_sha256",
    }.issubset(required_fields)
    failures = github["failure_conditions"]
    assert isinstance(failures, list)
    assert any("incomplete_results" in str(item) for item in failures)
    assert any("more than 900" in str(item) for item in failures)
    assert any("3600 seconds" in str(item) for item in failures)

    screening = query_plan["union_and_screening"]
    assert isinstance(screening, dict)
    assert screening["union_key"] == "GitHub pull-request node_id"
    assert "Boolean OR queries" in screening["prohibited"]
    assert "relevance-ranked truncation" in screening["prohibited"]


def test_protocol_requires_immutable_public_pr_base_gold_and_stack_proof() -> None:
    protocol = _load(PROTOCOL_PATH)
    eligibility = protocol["population_eligibility"]
    evidence = protocol["immutable_evidence_contract"]
    assert isinstance(eligibility, dict) and isinstance(evidence, dict)
    acquisition_fields = set(
        evidence["candidate_discovery_acquisition_required_fields"]
    )
    assert {
        "credential_profile_id",
        "credential_env_name",
        "credential_present_without_value",
        "collector_source_commit",
        "collector_source_tree",
        "requested_url",
        "final_url",
        "redirect_chain",
        "request_started_at_utc",
        "request_completed_at_utc",
        "x_ratelimit_reset",
        "x_ratelimit_resource",
        "x_ratelimit_used",
        "repository_public_visibility_receipt_sha256",
        "checkpoint_sha256",
    }.issubset(acquisition_fields)
    receipt_rules = evidence["receipt_rules"]
    assert any("visibility=public" in str(rule) for rule in receipt_rules)
    assert any("3600 seconds" in str(rule) for rule in receipt_rules)
    assert any("cross-acquisition" in str(rule) for rule in receipt_rules)
    assert set(eligibility) == {
        "repository_required",
        "pull_request_required",
        "task_required",
        "exclusion_reason_codes",
    }
    assert "outside_treatment_preperiod" in eligibility["exclusion_reason_codes"]
    assert "duplicate_code_lineage" in eligibility["exclusion_reason_codes"]
    assert "gold_did_not_pass" in eligibility["exclusion_reason_codes"]

    pr_fields = set(evidence["pull_request_receipt_required_fields"])
    assert {
        "pull_request_node_id",
        "base_commit_sha",
        "base_tree_sha",
        "head_commit_sha",
        "head_tree_sha",
        "merge_commit_sha",
        "gold_tree_sha",
        "api_response_sha256",
        "diff_sha256",
        "patch_sha256",
    }.issubset(pr_fields)
    stack_fields = set(evidence["stack_receipt_required_fields"])
    assert {
        "package_boundary_path",
        "package_manifest_sha256",
        "dependency_lock_sha256",
        "react_declared_requirement",
        "react_resolved_version",
        "next_declared_requirement",
        "next_resolved_version",
        "typescript_target_paths",
    }.issubset(stack_fields)
    source_fields = set(evidence["source_receipt_required_fields"])
    assert {
        "base_archive_sha256",
        "gold_archive_sha256",
        "submodule_check",
        "unsafe_link_check",
        "path_traversal_check",
        "secret_path_check",
    }.issubset(source_fields)


def test_protocol_deduplicates_lineage_and_requires_two_blind_reviewers() -> None:
    protocol = _load(PROTOCOL_PATH)
    lineage = protocol["lineage_deduplication"]
    review = protocol["two_reviewer_classification"]
    assert isinstance(lineage, dict) and isinstance(review, dict)
    assert lineage["primary_cluster"] == "public_code_lineage"
    assert lineage["one_selected_task_per_lineage_across_all_partitions"] is True
    assert lineage["canonical_repository_url_alone_is_sufficient"] is False
    collapse = lineage["collapse_conditions"]
    assert isinstance(collapse, list)
    assert any("fork-network root" in str(item) for item in collapse)
    assert any("template source" in str(item) for item in collapse)
    assert any("MinHash Jaccard" in str(item) for item in collapse)
    assert any("patch digest" in str(item) for item in collapse)
    owner_controls = lineage["owner_controls"]
    assert owner_controls["maximum_selected_lineages_per_owner"] == 4
    assert owner_controls["maximum_target_holdout_lineages_per_owner"] == 2

    assert review["reviewers"] == 2
    assert review["independence"].startswith("Reviewers classify separately")
    assert review["calibration"] == {
        "minimum_records": 48,
        "coverage": (
            "at least eight reviewed examples per behavior family, balanced across "
            "acceptable and unacceptable records"
        ),
        "minimum_cohens_kappa": 0.8,
        "critical_false_inclusions_allowed": 0,
    }
    assert "third treatment-blind adjudicator" in review["adjudication"]
    assert "baseline and candidate Skill contents" in review["masked_from_reviewers"]


def test_base_fail_gold_pass_is_executable_and_not_a_self_attestation() -> None:
    protocol = _load(PROTOCOL_PATH)
    qualification = protocol["task_authoring_and_qualification"]
    assert isinstance(qualification, dict)
    runtime = qualification["runtime_profile"]
    assert runtime["network"] == "none"
    assert runtime["root_filesystem"] == "read_only"
    assert runtime["required_image_identity"].startswith("sha256 registry digest")
    receipt_fields = set(qualification["execution_receipt_required_fields"])
    assert {
        "tree_role",
        "tree_sha",
        "runtime_image_digest",
        "argv_digest",
        "exit_code",
        "stdout_sha256",
        "stderr_sha256",
        "structured_outcome",
        "cleanup_receipt_sha256",
    }.issubset(receipt_fields)
    conjunction = qualification["qualification_conjunction"]
    assert conjunction[:4] == [
        "base public test has a verified task-relevant failure",
        "gold public test passes",
        "base host verifier has a verified task-relevant failure",
        "gold host verifier passes",
    ]
    assert (
        "Never edit a frozen test or verifier"
        in qualification["failed_qualification_policy"]
    )


def test_external_randomness_precedes_deterministic_global_allocation() -> None:
    protocol = _load(PROTOCOL_PATH)
    randomness = protocol["external_randomness"]
    selection = protocol["deterministic_selection"]
    assert isinstance(randomness, dict) and isinstance(selection, dict)
    assert randomness["allowed_sources"] == [
        "NIST Randomness Beacon 2.0 signed pulse",
        "drand mainnet verified round",
    ]
    assert randomness["seed_derivation"].startswith("SHA256(random_value")
    assert "population_digest" in randomness["seed_derivation"]
    assert "eligibility_ledger_digest" in randomness["seed_derivation"]
    assert "selection_code_digest" in randomness["seed_derivation"]
    assert "minimum-cost matching" in selection["allocation"]
    assert selection["manual_substitution"] == "forbidden"
    assert (
        "one selected task per public code-lineage cluster across all partitions"
        in (selection["global_constraints"])
    )


def test_feasibility_and_power_fail_closed_before_any_study_spec() -> None:
    protocol = _load(PROTOCOL_PATH)
    authorization = protocol["authorization_gate"]
    feasibility = protocol["quotas_and_feasibility_stop"]
    power = protocol["prospective_power_stop"]
    assert authorization["execution_authorized"] is False
    assert authorization["spend_authorized"] is False

    selected = feasibility["selected_quotas"]
    minimum = feasibility["minimum_eligible_lineages_before_randomness"]
    assert selected["development"]["total"] == 32
    assert selected["target_holdout"]["total"] == 192
    assert selected["safety_control"]["total"] == 64
    assert selected["grand_total"] == 288
    assert minimum == {
        "server_action_authorization": 140,
        "rsc_serialization": 140,
        "dom_batching": 20,
        "large_array_iteration": 20,
        "hook_timing": 20,
        "event_handler_reference": 20,
        "unique_lineage_grand_total": 360,
    }
    assert "generate no task, spec, preview, approval" in feasibility["stop_rule"]

    assert power["independent_unit"] == "target-holdout public code-lineage task"
    assert power["minimum_power"] == 0.9
    assert power["minimum_supported_absolute_improvement"] == 0.15
    assert power["simulation_repetitions"] == 100000
    assert power["safety_control_noninferiority"] == {
        "margin": -0.05,
        "familywise_alpha": 0.05,
        "minimum_power": 0.9,
        "critical_regressions_allowed": 0,
    }
    assert "No development or holdout spec may be generated" in power["stop_rule"]

    for prohibited_path in (
        "conference-sampling-frame-v2.json",
        "vercel-react-best-practices-conference-development-v2.yaml",
        "vercel-react-best-practices-conference-holdout-v2.yaml",
    ):
        assert not (EXAMPLE / prohibited_path).exists()


def test_analysis_does_not_turn_balanced_strata_into_a_population_claim() -> None:
    protocol = _load(PROTOCOL_PATH)
    analysis = protocol["analysis_and_trace_audit"]
    estimand = protocol["estimand"]
    assert "family-specific estimates only" in analysis["population_weighting"]
    assert "no prevalence estimate" in estimand["claim_boundary"]
    assert analysis["trace_reviewers"] == 2
    audit = analysis["manual_trace_review"]
    assert audit[0].startswith("100 percent of primary-family discordant")
    assert audit[1].startswith("100 percent of critical failures")
    assert audit[3].startswith("a selection-seed-derived 10 percent sample")


def test_raw_public_data_privacy_blocks_secret_shaped_records() -> None:
    privacy = _load(PROTOCOL_PATH)["raw_public_data_privacy"]
    assert privacy["source_bytes_are_immutable"] is True
    assert "mode 0700" in privacy["storage"]
    assert "mode 0600" in privacy["storage"]
    assert "Matched values are never copied" in privacy["scan_boundary"]
    assert "blocks the record" in privacy["affected_record_policy"]
    assert privacy["zero_finding_is_not_safety_proof"] is True
