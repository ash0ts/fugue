from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

EXAMPLE = Path("examples/comparisons/anthropic-skill-creator-upgrade")
QUERY_PLAN = EXAMPLE / "conference-github-candidate-query-plan-v2.json"
PROTOCOL = EXAMPLE / "conference-sampling-frame-protocol-v2.json"
V1_PROTOCOL = EXAMPLE / "conference-sampling-frame-protocol-v1.json"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_strict(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )
    assert isinstance(value, dict)
    return value


@pytest.fixture(scope="module")
def query_plan() -> dict[str, Any]:
    return _load_strict(QUERY_PLAN)


@pytest.fixture(scope="module")
def protocol() -> dict[str, Any]:
    return _load_strict(PROTOCOL)


def test_query_plan_is_strict_prospective_and_pre_treatment(
    query_plan: dict[str, Any],
) -> None:
    assert set(query_plan) == {
        "schema_version",
        "id",
        "status",
        "purpose",
        "api_contract",
        "temporal_contract",
        "target_frame",
        "repository_discovery",
        "code_candidate_discovery",
        "repository_verification",
        "readiness",
    }
    assert query_plan["schema_version"] == 2
    assert query_plan["status"] == "prospective_no_queries_executed"

    temporal = query_plan["temporal_contract"]
    cutoff = datetime.fromisoformat(temporal["primary_pre_treatment_cutoff_utc"])
    public_at = datetime.fromisoformat(
        temporal["treatment_publication_event"]["created_at_utc"]
    )
    assert cutoff < public_at
    assert (public_at - cutoff).total_seconds() == 1
    assert temporal["treatment_publication_event"]["url"] == (
        "https://github.com/anthropics/skills/pull/350"
    )
    assert temporal["post_treatment_rule"].startswith("Current or post-treatment")

    readiness = query_plan["readiness"]
    assert readiness == {
        "queries_executed": False,
        "population_records_present": 0,
        "authorizes_network_preparation": False,
        "authorizes_preview": False,
        "authorizes_model_calls_or_spend": False,
        "next_gate": (
            "Implement reviewed trusted preparation, execute it with read-only "
            "GitHub credentials, and freeze complete query, eligibility, lineage, "
            "and coverage-audit receipts under a new population identity."
        ),
    }


def test_protocol_binds_the_exact_query_plan(
    query_plan: dict[str, Any], protocol: dict[str, Any]
) -> None:
    digest = hashlib.sha256(QUERY_PLAN.read_bytes()).hexdigest()
    binding = protocol["query_plan"]
    assert binding == {
        "path": QUERY_PLAN.name,
        "sha256": digest,
        "required_status": query_plan["status"],
    }
    assert protocol["schema_version"] == 2
    assert protocol["status"] == "prospective_not_yet_sampled"
    assert protocol["treatment"]["baseline_commit"] == (
        "a5bcdd7e58cdff48566bf876f0a72a2008dcefbc"
    )
    assert protocol["treatment"]["candidate_commit"] == (
        "1ed29a03dc852d30fa6ef2ca53a67dc2c2c2c563"
    )


def test_repository_queries_are_sharded_complete_and_deduplicated(
    query_plan: dict[str, Any],
) -> None:
    discovery = query_plan["repository_discovery"]
    queries = discovery["query_templates"]
    ids = [item["id"] for item in queries]
    assert len(ids) == len(set(ids)) == 7
    for item in queries:
        query = item["query"]
        assert "fork:false" in query
        assert "archived:false" in query
        assert "created:{start_utc}..{end_utc}" in query

    sharding = discovery["date_sharding"]
    assert sharding["maximum_results_per_leaf"] == 900
    assert sharding["deduplication_key"] == "GitHub repository node_id"
    assert "midpoint + 1 second" in sharding["algorithm"]
    assert "incomplete_results=false" in discovery["completeness_gate"]
    assert "continuous page ledger" in discovery["completeness_gate"]

    receipt_fields = set(discovery["required_query_receipt_fields"])
    assert {
        "rendered_query",
        "total_count",
        "incomplete_results",
        "page_count",
        "response_etags_or_explicit_nulls",
        "response_body_sha256s",
        "terminal_link_header_sha256",
    } <= receipt_fields


def test_code_search_is_candidate_expansion_not_a_classifier(
    query_plan: dict[str, Any],
) -> None:
    discovery = query_plan["code_candidate_discovery"]
    assert "never used as the population denominator" in discovery["role"]
    assert "never used" in discovery["role"]
    assert "regex can match examples" in discovery["completeness_caveat"]

    queries = {item["id"]: item["query"] for item in discovery["queries"]}
    assert set(queries) == {
        "frontmatter-candidates",
        "declared-compatibility-candidates",
        "missing-compatibility-candidates",
        "long-valid-name-candidates",
        "unsupported-name-candidates",
        "invalid-compatibility-expansion",
    }
    for query in queries.values():
        assert "(?-i)SKILL\\.md" in query
        assert "NOT is:fork" in query
        assert "NOT is:archived" in query
    assert "{41,64}" in queries["long-valid-name-candidates"]
    assert "{65,}" in queries["unsupported-name-candidates"]
    assert "NOT /^(?-i)compatibility:/" in queries["missing-compatibility-candidates"]
    assert any(
        item.startswith("Classifying YAML type")
        for item in discovery["prohibited_uses"]
    )


def test_v2_requires_typed_yaml_facts_and_human_adjudication(
    protocol: dict[str, Any],
) -> None:
    facts = protocol["typed_public_facts"]
    assert facts["artifact"] == "PublicSkillFactsV2"
    assert facts["strict_unknown_fields"] == "reject"
    assert set(facts["required_fields"]) == {
        "fact_id",
        "cohort",
        "repository_node_id",
        "repository_url",
        "owner_node_id",
        "commit_sha",
        "commit_committed_at_utc",
        "skill_path",
        "skill_blob_sha",
        "skill_content_sha256",
        "skill_first_public_commit_utc",
        "frontmatter",
        "requirement_groups",
        "public_basis_refs",
        "classifier_revision",
        "predates_treatment_publication",
    }
    frontmatter = facts["frontmatter"]
    assert "first complete top-level frontmatter" in frontmatter["parser"]
    yaml_types = "string|null|boolean|integer|number|sequence|mapping"
    assert frontmatter["fields"]["name"]["yaml_type"] == yaml_types
    assert frontmatter["fields"]["compatibility"]["yaml_type"] == yaml_types
    assert (
        "unsupported_claim"
        in frontmatter["fields"]["compatibility"]["contract_status_type"]
    )

    review = protocol["review_and_adjudication"]
    assert review["reviewers_per_candidate"] == 2
    assert review["models_may_review_or_adjudicate"] is False
    assert "third human adjudicator" in review["adjudication"]
    assert "baseline or candidate output" in review["reviewer_independence"]
    assert "chance-corrected agreement" in review["agreement_reporting"]


def test_lineage_dedup_and_one_primary_skill_per_repository(
    protocol: dict[str, Any],
) -> None:
    maintenance = protocol["cohorts"]["maintenance_primary"]
    assert maintenance["one_primary_skill_per_repository"] is True
    assert maintenance["globally_disjoint_repository_clusters"] is True

    lineage = protocol["lineage_and_independence"]
    assert lineage["primary_inference_cluster"] == "source_lineage_cluster_id"
    assert "At most one primary holdout Skill" in lineage["repository_rule"]
    assert "near-copied Skill content" in lineage["cross_repository_lineage_rule"]
    assert "fork parent" in lineage["signals"][0]
    near_copy = lineage["near_copy_rule"]
    assert near_copy["shingle_tokens"] == 5
    assert near_copy["minhash_permutations"] == 256
    assert near_copy["jaccard_threshold"] == 0.9
    assert "development" in lineage["multiple_packages_policy"]
    assert "never increase" in lineage["owner_sensitivity"]


def test_external_randomness_and_inclusion_weights_are_required(
    protocol: dict[str, Any],
) -> None:
    sampling = protocol["sampling_and_randomness"]
    randomness = sampling["external_randomness"]
    assert randomness["provider"] == "drand"
    assert randomness["future_round_committed_before_available"] is True
    assert randomness["bls_signature_verification_required"] is True
    assert randomness["seed_formula"] == (
        "SHA256(UTF8(lowercase_population_bundle_sha256 + newline + "
        "lowercase_drand_chain_hash + newline + decimal_round + newline + "
        "lowercase_randomness_hex)); no trailing newline is included."
    )
    assert "source_lineage_cluster_id" in sampling["selection_algorithm"]
    assert "No investigator choice" in sampling["replacement"]
    assert "Never synthesize" in sampling["insufficient_stratum_rule"]

    analysis = protocol["inclusion_weights_and_analysis"]
    assert analysis["primary_population_estimator"].startswith("Stratified Hajek")
    assert set(analysis["weight_record_required_fields"]) == {
        "stratum",
        "eligible_lineage_clusters_N_h",
        "selected_primary_clusters_n_h",
        "inclusion_probability_numerator",
        "inclusion_probability_denominator",
        "inverse_probability_weight",
        "finite_population_fraction",
        "selection_rank",
    }
    assert "pi_i = n_h / N_h" in analysis["inclusion_probability"]
    assert "frozen inclusion weights" in analysis["weighting_rule"]
    assert "never increase independent N" in analysis["repeated_attempts"]
    assert "prospective power" in analysis["power_gate"]


def test_creation_is_a_separate_secondary_cohort(protocol: dict[str, Any]) -> None:
    maintenance = protocol["cohorts"]["maintenance_primary"]
    creation = protocol["cohorts"]["creation_secondary"]
    assert maintenance["primary_inference"] is True
    assert creation["primary_inference"] is False
    assert creation["existing_skill_package_required"] is False
    assert creation["separate_frame_tasks_labels_specs_results_and_analysis"] is True
    assert creation["never_pooled_with_maintenance"] is True
    assert creation["repository_clusters_disjoint_from_maintenance_holdout"] is True
    assert "requesting one new Skill" in creation["population"]
    assert creation["holdout_study_id"] != maintenance["holdout_study_id"]


def test_v2_authorizes_nothing_and_v1_remains_a_distinct_contract(
    protocol: dict[str, Any],
) -> None:
    v1 = _load_strict(V1_PROTOCOL)
    assert v1["schema_version"] == 1
    assert v1["protocol_id"].endswith("protocol-v1")
    assert protocol["protocol_id"].endswith("protocol-v2")
    assert "V1 remains immutable audit history" in protocol["relationship_to_v1"]

    readiness = protocol["readiness"]
    assert readiness["population_records_present"] == 0
    assert readiness["query_receipts_present"] is False
    assert readiness["reviewed_lineage_clusters_present"] == 0
    assert readiness["sampling_design_present"] is False
    assert readiness["external_randomness_receipt_present"] is False
    assert readiness["development_or_holdout_specs_present"] is False
    assert readiness["authorizes_preview"] is False
    assert readiness["authorizes_model_calls_or_spend"] is False
    assert "may be created from this protocol alone" in readiness["fail_closed"]


def test_v2_raw_public_data_privacy_is_fail_closed(
    protocol: dict[str, Any],
) -> None:
    privacy = protocol["raw_public_data_privacy"]
    assert privacy["source_bytes_are_immutable"] is True
    assert "symbolic-link targets are rejected" in privacy["storage"]
    assert "Matched values are never copied" in privacy["scan_boundary"]
    assert "blocks the record" in privacy["affected_record_policy"]
    assert privacy["zero_finding_is_not_safety_proof"] is True
