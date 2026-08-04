from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

EXAMPLE = Path("examples/comparisons/superpowers-writing-plans-upgrade")
QUERY_PLAN_PATH = EXAMPLE / "conference-query-plan-v2.json"
PROTOCOL_PATH = EXAMPLE / "conference-sampling-frame-protocol-v2.json"
SKILL_LOCK_PATH = EXAMPLE / "skill-revisions.lock.json"


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def test_query_plan_has_a_strict_prospective_public_github_boundary() -> None:
    plan = _read(QUERY_PLAN_PATH)
    assert set(plan) == {
        "schema_version",
        "id",
        "status",
        "api",
        "treatment_time_boundary",
        "observation_window",
        "population_scope",
        "atomic_queries",
        "sharding",
        "pagination",
        "completeness",
        "receipts",
        "prohibitions",
    }
    assert plan["schema_version"] == 2
    assert plan["status"] == "prospective_query_plan_only_no_population_acquired"
    assert plan["api"] == {
        "provider": "github_rest_api",
        "origin": "https://api.github.com",
        "endpoint": "/search/issues",
        "version": "2022-11-28",
        "accept": "application/vnd.github+json",
        "public_data_only": True,
        "credential_name": "GITHUB_TOKEN",
        "credential_value_serialized": False,
    }
    boundary = plan["treatment_time_boundary"]
    window = plan["observation_window"]
    assert boundary["candidate_is_direct_descendant_of_baseline"] is True
    assert boundary["records_must_be_strictly_later_than_candidate"] is True
    assert _utc(window["start_inclusive_utc"]) > _utc(
        boundary["candidate_committer_time_utc"]
    )
    assert _utc(window["end_inclusive_utc"]) > _utc(window["start_inclusive_utc"])
    assert plan["population_scope"]["task_authoring_before_population_freeze"] is False
    assert "does not represent all GitHub" in plan["population_scope"]["claim_boundary"]


def test_every_github_search_atom_is_bounded_atomic_and_auditable() -> None:
    plan = _read(QUERY_PLAN_PATH)
    atoms = plan["atomic_queries"]
    expected_keys = {
        "id",
        "sampling_family",
        "provisional_stratum",
        "record_kind",
        "timestamp_qualifier",
        "term",
        "query_template",
    }
    assert len(atoms) >= 32
    assert len({item["id"] for item in atoms}) == len(atoms)
    observed_strata = Counter(item["provisional_stratum"] for item in atoms)
    assert set(observed_strata) == {
        "security_privacy",
        "identity_migration",
        "lifecycle_recovery",
        "integration_runtime",
        "docs_only",
        "unsafe_request",
        "no_change_required",
        "single_surface",
    }
    assert all(count >= 1 for count in observed_strata.values())
    for atom in atoms:
        assert set(atom) == expected_keys
        query = atom["query_template"]
        assert query.count("{start}") == 1
        assert query.count("{end}") == 1
        assert " OR " not in query
        assert "archived:false" in query
        assert "sort:" not in query
        assert atom["timestamp_qualifier"] in {"merged", "closed"}
        if atom["record_kind"] == "pull_request":
            assert "is:pr is:merged" in query
            assert "merged:{start}..{end}" in query
        else:
            assert atom["record_kind"] == "issue"
            assert "is:issue is:closed" in query
            assert "closed:{start}..{end}" in query

    sharding = plan["sharding"]
    assert sharding["maximum_total_count_per_terminal_shard"] == 900
    assert sharding["github_search_hard_result_cap"] == 1000
    assert (
        sharding["maximum_total_count_per_terminal_shard"]
        < sharding["github_search_hard_result_cap"]
    )
    assert sharding["mutable_query_fallback_allowed"] is False
    pagination = plan["pagination"]
    assert pagination["per_page"] == 100
    assert pagination["maximum_pages_per_terminal_shard"] == 9
    completeness = plan["completeness"]
    assert completeness["required_incomplete_results"] is False
    assert completeness["missing_or_truncated_response_policy"] == "fail_closed"
    assert "two complete acquisitions" in completeness["repeat_acquisition"]
    assert "node-ID sets" in completeness["repeat_acquisition"]


def test_v2_protocol_binds_query_plan_and_exact_treatment_after_cutoff() -> None:
    protocol = _read(PROTOCOL_PATH)
    lock = _read(SKILL_LOCK_PATH)
    assert set(protocol) == {
        "schema_version",
        "protocol_id",
        "status",
        "supersedes_without_mutating",
        "query_plan",
        "treatment",
        "estimand",
        "population",
        "eligibility",
        "immutable_lineage",
        "independence",
        "reviewer_roles",
        "external_randomness",
        "selection",
        "population_weighting",
        "task_authoring_and_leakage",
        "raw_public_data_privacy",
        "preparation",
        "sealing",
        "analysis",
        "readiness",
        "claim_boundary",
        "not_generated",
    }
    assert protocol["schema_version"] == 2
    assert protocol["status"] == "prospective_not_yet_sampled"
    assert protocol["query_plan"]["sha256"] == _sha256(QUERY_PLAN_PATH)
    assert protocol["supersedes_without_mutating"]["v1_evidence_reused"] is False
    treatment = protocol["treatment"]
    assert treatment["baseline_commit"] == lock["baseline"]["commit"]
    assert treatment["candidate_commit"] == lock["candidate"]["commit"]
    assert treatment["baseline_bundle_sha256"] == lock["baseline"][
        "bundle_digest"
    ].removeprefix("sha256:")
    assert treatment["candidate_bundle_sha256"] == lock["candidate"][
        "bundle_digest"
    ].removeprefix("sha256:")
    assert treatment["candidate_is_direct_descendant_of_baseline"] is True
    assert _utc(treatment["task_record_window_starts_utc"]) > _utc(
        treatment["candidate_committer_time_utc"]
    )
    assert treatment["whole_revision_estimand_only"] is True
    assert treatment["component_attribution_allowed"] is False


def test_v2_protocol_requires_immutable_base_tree_and_real_independence() -> None:
    protocol = _read(PROTOCOL_PATH)
    lineage = protocol["immutable_lineage"]
    assert {
        "repository_rest_id",
        "repository_node_id",
        "owner_node_id",
        "source_repository_node_id",
        "template_repository_node_id",
    }.issubset(lineage["repository_fields"])
    assert {
        "base_repository_node_id",
        "base_commit_sha",
        "base_tree_sha",
        "merge_commit_sha",
    }.issubset(lineage["pull_request_fields"])
    assert {
        "task_commit_sha",
        "task_tree_sha",
        "archive_sha256",
        "tree_manifest_sha256",
        "license_blob_sha",
        "safety_scan_receipt_sha256",
    }.issubset(lineage["source_fields"])
    provenance = " ".join(lineage["provenance_checks"])
    assert "task_commit_sha equals base_commit_sha" in provenance
    assert "task_tree_sha equals the base commit's Git tree SHA" in provenance
    assert "merged diff is host-only" in provenance

    independence = protocol["independence"]
    assert independence["repository_identity"].startswith("immutable GitHub")
    assert independence["one_selected_task_per_repository"] is True
    assert independence["one_selected_repository_per_independence_cluster"] is True
    assert independence["multiple_agent_attempts_are_independent_units"] is False
    assert "owner/source/template" in independence["fallback_if_cluster_quota_is_unmet"]


def test_v2_protocol_separates_reviewers_and_seals_private_truth() -> None:
    protocol = _read(PROTOCOL_PATH)
    roles = protocol["reviewer_roles"]
    assert set(roles) == {
        "acquisition_steward",
        "eligibility_reviewers",
        "eligibility_adjudicator",
        "public_task_author",
        "private_label_author",
        "leakage_reviewer",
        "analysis_steward",
        "role_overlap_policy",
    }
    assert "may not inspect the merged diff" in roles["public_task_author"]
    assert "may not inspect treatment outputs" in roles["private_label_author"]
    assert "must be distinct" in roles["role_overlap_policy"]
    classification = protocol["eligibility"]["classification"]
    assert classification == {
        "reviewers": 2,
        "independent_first_pass": True,
        "treatment_output_blinded": True,
        "agreement_required": True,
        "adjudication_required_on_disagreement": True,
        "mutually_exclusive_final_stratum": True,
        "query_term_hit_is_sufficient": False,
    }
    leakage = protocol["task_authoring_and_leakage"]
    assert leakage["all_288_tasks_frozen_before_first_agent_output"] is True
    assert "merged diff" in leakage["host_only_inputs"]
    assert protocol["eligibility"]["task"][2].endswith("mode 0600.")


def test_v2_protocol_blocks_token_shaped_public_records_from_authoring() -> None:
    privacy = _read(PROTOCOL_PATH)["raw_public_data_privacy"]
    assert privacy["source_bytes_are_immutable"] is True
    assert "mode 0700" in privacy["storage"]
    assert "mode 0600" in privacy["storage"]
    assert "Matched values are never copied" in privacy["scan_boundary"]
    assert "blocks the record" in privacy["affected_record_policy"]
    assert privacy["zero_finding_is_not_safety_proof"] is True


def test_v2_protocol_uses_external_randomness_and_design_weights() -> None:
    protocol = _read(PROTOCOL_PATH)
    randomness = protocol["external_randomness"]
    assert randomness["provider"] == "NIST Randomness Beacon 2.0"
    assert randomness["population_freeze_before_pulse"] is True
    assert randomness["operator_chosen_seed_allowed"] is False
    assert randomness["master_seed_derivation"] == (
        "SHA256(UTF8(protocol_id + newline + lowercase_population_sha256 + newline + "
        "lowercase_pulse_output_hex + newline)); the lowercase 64-hex digest is the "
        "master_seed_hex."
    )
    assert len(randomness["domain_separated_seeds"]) == 4
    assert randomness["missing_or_invalid_pulse_policy"].startswith("fail closed")

    quotas = protocol["selection"]["quotas"]
    assert quotas["development"]["total"] == 32
    assert quotas["target_holdout"]["total"] == 192
    assert quotas["safety_control"]["total"] == 64
    assert sum(item["total"] for item in quotas.values()) == 288
    assert (
        protocol["population"]["minimum_eligible_clusters_before_selection"]["total"]
        == 864
    )
    weighting = protocol["population_weighting"]
    assert weighting["holdout_sample_count"] == "n_h equals 48 for each target stratum"
    assert weighting["inclusion_probability"] == "pi_h = n_h / N_h"
    assert weighting["analysis_weight"].startswith("w_h = N_h / n_h")
    assert weighting["development_weight"].startswith("none")
    assert weighting["safety_control_weight"].startswith("none")
    assert weighting["missing_weight_policy"].endswith(
        "invalidates confirmatory analysis."
    )


def test_v2_protocol_remains_a_zero_execution_prospective_artifact() -> None:
    protocol = _read(PROTOCOL_PATH)
    readiness = protocol["readiness"]
    assert readiness["protocol_only"] is True
    assert readiness["population_records_present"] == 0
    assert readiness["required_selected_tasks"] == 288
    assert readiness["development_or_holdout_spec_generated"] is False
    assert readiness["preview_or_approval_generated"] is False
    assert "No comparison spec" in readiness["execution_gate"]
    assert "authorizes no GitHub acquisition" in protocol["claim_boundary"]
    assert (
        "source locks, task images, previews, approvals, results, or reports"
        in (protocol["not_generated"])
    )
    assert not (EXAMPLE / "conference-sampling-frame-v2.json").exists()
    assert not (EXAMPLE / "conference-development-v2.yaml").exists()
    assert not (EXAMPLE / "conference-holdout-v2.yaml").exists()
