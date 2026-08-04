from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "examples" / "comparisons" / "community-skill-upgrades"
MANIFEST_PATH = COMMON / "conference-population-acquisition-manifest-v3.json"
PILOT_PATH = COMMON / "github-collector-live-pilot-v2.json"
COMPILER_PATH = COMMON / "compile_github_collector_plan_v2.py"
SPEC = importlib.util.spec_from_file_location(
    "population_manifest_compiler_v2", COMPILER_PATH
)
assert SPEC is not None and SPEC.loader is not None
COMPILER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = COMPILER
SPEC.loader.exec_module(COMPILER)


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes(), object_pairs_hook=_reject_duplicates)
    assert isinstance(value, dict)
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _serialized_plan_sha(plan: dict[str, Any]) -> str:
    payload = (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def test_manifest_has_one_strict_non_authorizing_campaign_state() -> None:
    manifest = _load(MANIFEST_PATH)
    assert set(manifest) == {
        "schema_version",
        "campaign_id",
        "status",
        "claim_boundary",
        "implementation_commit",
        "relationship_to_descriptive_campaign",
        "shared_implementation",
        "lanes",
        "historical_live_pilot",
        "next_gates",
    }
    assert manifest["schema_version"] == 3
    assert manifest["status"] == (
        "collector_implementation_committed_live_acquisition_not_started_"
        "selection_blocked"
    )
    assert manifest["claim_boundary"] == {
        "candidate_acquisition_implementation_committed": True,
        "candidate_acquisition_executed": False,
        "population_frame_ready": False,
        "authenticated_selection_ready": False,
        "task_authoring_ready": False,
        "population_experiment_preview_generated": False,
        "population_experiment_execution_authorized": False,
        "community_or_conference_claim_supported": False,
    }
    assert "authorizes no network acquisition" in manifest["next_gates"][-1]


def test_manifest_binds_exact_implementation_commit_and_executables() -> None:
    manifest = _load(MANIFEST_PATH)
    implementation = manifest["implementation_commit"]
    assert implementation["source_commit"] == (
        "4b26d5608bac9a89f4fbdf956a722b707704f36a"
    )
    assert implementation["source_tree"] == ("bc32e2c024450d11a9faf7738995e9660a6c55e1")
    assert implementation["files_match_commit"] is True

    current_lock = COMPILER.COLLECTOR.current_implementation_lock(
        compiler_path=COMPILER_PATH
    )
    assert current_lock == {
        key: implementation[key]
        for key in (
            "collector_sha256",
            "compiler_sha256",
            "files_match_commit",
            "lock_sha256",
            "source_commit",
            "source_tree",
        )
    }

    paths = {
        "collector": "github_population_v2.py",
        "compiler": "compile_github_collector_plan_v2.py",
        "generic_selector": "select_population_v2.py",
        "vercel_selector": "select_vercel_population_v2.py",
    }
    for name, expected_path in paths.items():
        binding = manifest["shared_implementation"][name]
        assert binding["path"] == expected_path
        assert binding["sha256"] == _sha(COMMON / expected_path)

    generic = manifest["shared_implementation"]["generic_selector"]
    assert generic["status"] == "blocked"
    assert generic["blocker_code"] == (
        "authenticated_verification_boundary_unavailable"
    )
    assert generic["selected_count"] == 0
    vercel = manifest["shared_implementation"]["vercel_selector"]
    assert vercel["status"] == "blocked_authenticated_evidence_required"
    assert vercel["blocker_code"] == "protocol_trust_roots_not_registered"
    assert vercel["required_registered_ed25519_trust_roots"] == 4
    assert vercel["assignment_emitted"] is False

    selector_locks = manifest["shared_implementation"]["selection_implementation_locks"]
    assert set(selector_locks) == {"generic", "vercel"}
    for selector_name, lock_binding in selector_locks.items():
        lock = lock_binding["lock"]
        assert lock_binding["lock_sha256"] == (COMPILER.COLLECTOR.stable_digest(lock))
        selector_path = COMMON / lock["selector_path"]
        assert lock["selector_sha256"] == _sha(selector_path)
        if selector_name == "generic":
            assert lock["lanes"] == [
                "superpowers_writing_plans",
                "anthropic_skill_creator",
            ]
        else:
            assert lock["lanes"] == ["vercel_react_best_practices"]

    assert manifest["shared_implementation"]["generic_selection_invariants"] == [
        "one externally randomized record is retained per independence cluster "
        "before stratum allocation",
        "every selected partition receives its preregistered slice of one frozen "
        "stratum permutation",
        "inclusion probabilities and inverse-probability weights are emitted per "
        "selected task",
    ]
    vercel_invariants = manifest["shared_implementation"]["vercel_selection_invariants"]
    assert any("minimum-cost flow" in value for value in vercel_invariants)
    assert any("owner caps" in value for value in vercel_invariants)
    assert all("inclusion probabilities" not in value for value in vercel_invariants)


@pytest.mark.parametrize(
    ("lane", "compiler_lane"),
    [
        ("superpowers_writing_plans", "superpowers"),
        ("anthropic_skill_creator", "anthropic"),
        ("vercel_react_best_practices", "vercel"),
    ],
)
def test_lane_plan_recompilation_preserves_every_bound_identity(
    lane: str, compiler_lane: str
) -> None:
    manifest = _load(MANIFEST_PATH)
    entry = manifest["lanes"][lane]
    query_path = (COMMON / entry["query_plan"]["path"]).resolve()
    protocol_path = (COMMON / entry["protocol"]["path"]).resolve()
    assert entry["query_plan"]["sha256"] == _sha(query_path)
    assert entry["protocol"]["sha256"] == _sha(protocol_path)

    compiled = COMPILER.compile_plan(compiler_lane, query_path, protocol_path)
    compiled_binding = entry["compiled_plan"]
    assert compiled_binding["canonical_sha256"] == (
        COMPILER.COLLECTOR.stable_digest(compiled)
    )
    assert compiled_binding["serialized_bytes_sha256"] == _serialized_plan_sha(compiled)
    assert (
        compiled_binding["implementation_lock_sha256"]
        == (compiled["collector_implementation"]["lock_sha256"])
    )
    assert compiled["collector_implementation"] == (
        COMPILER.COLLECTOR.current_implementation_lock(compiler_path=COMPILER_PATH)
    )

    collector = entry["collector_readiness"]
    assert collector == {
        "status": "compiled_not_acquired",
        "required_acquisitions": 2,
        "completed_acquisitions": 0,
        "minimum_separation_seconds": 3600,
        "operator_execution_authorized": False,
    }
    selector = entry["selector_readiness"]
    assert selector["status"].startswith("blocked")
    assert selector.get("selected_count", 0) == 0
    assert selector.get("assignment_emitted", False) is False
    expected_selector = "vercel" if lane == "vercel_react_best_practices" else "generic"
    assert (
        selector["selection_implementation_lock_sha256"]
        == (
            manifest["shared_implementation"]["selection_implementation_locks"][
                expected_selector
            ]["lock_sha256"]
        )
    )


def test_vercel_protocol_registers_no_partial_trust_root() -> None:
    manifest = _load(MANIFEST_PATH)
    entry = manifest["lanes"]["vercel_react_best_practices"]
    protocol = _load((COMMON / entry["protocol"]["path"]).resolve())
    authorization = protocol["authorization_gate"]
    assert authorization["execution_authorized"] is False
    assert authorization["spend_authorized"] is False
    boundary = authorization["authenticated_selection_boundary"]
    assert boundary["status"] == "blocked_trust_roots_not_registered"
    assert boundary["algorithm"] == "ed25519-detached-signature-v1"
    assert all(
        boundary[field] is None
        for field in (
            "source_verifier_public_key_sha256",
            "beacon_verifier_public_key_sha256",
            "review_governance_public_key_sha256",
            "power_verifier_public_key_sha256",
        )
    )


def test_historical_pilot_is_superseded_transport_evidence_only() -> None:
    manifest = _load(MANIFEST_PATH)
    pilot_binding = manifest["historical_live_pilot"]
    pilot = _load(PILOT_PATH)

    assert pilot_binding["classification"] == (
        "historical_superseded_transport_diagnostic"
    )
    assert pilot_binding["sha256"] == _sha(PILOT_PATH)
    assert pilot_binding["credential_nonleakage_qualified"] is False
    assert (
        "not a global credential non-leakage qualification"
        in pilot_binding["qualified_claim"]
    )
    assert pilot["execution"] == {
        "network_operation": (
            "read_only_authenticated_github_search_public_visibility_not_proven"
        ),
        "model_calls": 0,
        "paid_cells": 0,
        "credential_value_serialized": False,
        "raw_responses_committed_to_git": False,
    }
    assert pilot["observations"]["candidate_records"] == 194
    assert pilot["query"]["acquisition_repetitions"] == 1
    assert pilot["visibility_audit"]["public_visibility_proven"] is False
    privacy = pilot["public_record_privacy_scan"]
    assert privacy["candidate_records_with_token_shaped_values"] == 1
    assert privacy["matched_values_copied_to_receipt"] is False
    assert pilot["code_locks"] == {
        "collector_path": (
            "examples/comparisons/community-skill-upgrades/github_population_v2.py"
        ),
        "collector_sha256": (
            "5cf40ddb87e6241533c21d4563a315e798eca7a1a535e5776bb78498bf2142aa"
        ),
        "compiler_path": (
            "examples/comparisons/community-skill-upgrades/"
            "compile_github_collector_plan_v2.py"
        ),
        "compiler_sha256": (
            "b24ef230620eb554ad67bc9ec747223bccc2b856b9c51833adc57c1ea2d01e44"
        ),
        "selector_path": (
            "examples/comparisons/community-skill-upgrades/select_population_v2.py"
        ),
        "selector_sha256": (
            "c3fbe36eb2f26f82046857f57888751a31e2f850f2d36738dbf40592fd3100b4"
        ),
    }
    current = manifest["shared_implementation"]
    assert pilot["code_locks"]["collector_sha256"] != current["collector"]["sha256"]
    assert pilot["code_locks"]["compiler_sha256"] != current["compiler"]["sha256"]
    assert (
        pilot["code_locks"]["selector_sha256"] != current["generic_selector"]["sha256"]
    )
    boundary = pilot["claim_boundary"]
    assert boundary["candidate_discovery_transport_qualified"] is True
    assert all(
        boundary[key] is False
        for key in (
            "public_only_candidate_discovery_qualified",
            "population_frame_ready",
            "eligibility_review_complete",
            "selection_authorized",
            "experiment_authorized",
            "conference_claim_supported",
        )
    )


def test_population_work_does_not_claim_descriptive_execution_or_approval() -> None:
    relationship = _load(MANIFEST_PATH)["relationship_to_descriptive_campaign"]
    assert relationship["descriptive_campaign_head"] == (
        "44a5a84a78944ef77290316b662a3da7a0d7da8e"
    )
    assert relationship["descriptive_campaign_tree"] == (
        "4e1c91dbaa8894380296e41bb9bf4745b42f2d3d"
    )
    assert relationship["total_planned_cells"] == 576
    assert relationship["total_maximum_usd"] == 4980
    assert relationship["executed_cells"] == 0
    assert relationship["descriptive_previews_remain_immutable"] is True
    assert relationship["population_work_changes_descriptive_preview_identity"] is False
    assert relationship["fresh_preview_and_approval_required_for_population_execution"]

    expected = {
        "superpowers_writing_plans": (
            "6a4b531b7aa96c4670924adc06fd3c88f9c5c433327fa5d56335390fa56f945e",
            192,
            1700,
            "descriptive-preview-superpowers-v6.json",
        ),
        "anthropic_skill_creator": (
            "fee496ac0b4ebf787a39e4f64bd758ffd67a15a91af8bddce049e8aa11c8d0a2",
            192,
            1640,
            "descriptive-preview-anthropic-v2.json",
        ),
        "vercel_react_best_practices": (
            "21c065bb4b122b68eec4f0d3346eba33a8214807c1abd2bf4cc27244c735dfcc",
            192,
            1640,
            "descriptive-preview-vercel-v2.json",
        ),
    }
    assert len(relationship["previews"]) == len(expected)
    for preview in relationship["previews"]:
        digest, cells, cap, projection_name = expected[preview["lane"]]
        projection_binding = preview["preview_projection"]
        assert projection_binding["path"] == projection_name
        projection_path = COMMON / projection_name
        assert projection_binding["sha256"] == _sha(projection_path)
        projection = _load(projection_path)
        assert (
            projection["source_descriptive_head"]
            == relationship["descriptive_campaign_head"]
        )
        assert (
            projection["source_descriptive_tree"]
            == relationship["descriptive_campaign_tree"]
        )
        assert projection["preview_digest"] == digest
        assert projection["estimated_cells"] == cells
        assert projection["hard_cap_usd"] == cap
        assert projection["approval"] == {
            "state": "fresh_exact_approval_required",
            "receipt_sha256": None,
        }
        assert projection["execution"] == {"started": False, "terminal_cells": 0}
        spec_path = (COMMON / projection["source_spec"]["path"]).resolve()
        assert projection["source_spec"]["sha256"] == _sha(spec_path)
        assert preview == {
            "lane": preview["lane"],
            "preview_digest": digest,
            "preview_projection": projection_binding,
            "planned_cells": cells,
            "maximum_usd": cap,
            "status": "unexecuted_authored_benchmark_preview",
            "approval_state": "fresh_exact_approval_required",
            "approval_receipt_sha256": None,
        }
