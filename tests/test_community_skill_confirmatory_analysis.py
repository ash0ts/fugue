from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO_ROOT / "examples/comparisons/community-skill-upgrades"


def _load_module() -> Any:
    path = CAMPAIGN / "analyze_confirmatory.py"
    spec = importlib.util.spec_from_file_location("confirmatory_analysis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ANALYSIS = _load_module()


def _load_source_lock_module() -> Any:
    path = CAMPAIGN / "prepare_local_source_lock.py"
    spec = importlib.util.spec_from_file_location("confirmatory_source_lock", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE_LOCK = _load_source_lock_module()


def _load_audit_module() -> Any:
    path = CAMPAIGN / "freeze_trace_audit.py"
    spec = importlib.util.spec_from_file_location("freeze_trace_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = _load_audit_module()
PROFILES = CAMPAIGN / "confirmatory-analysis-profiles.json"
PROFILES_V2 = CAMPAIGN / "confirmatory-analysis-profiles-v2.json"
PROTOCOL_V2 = CAMPAIGN / "conference-preregistration-v2.json"
MANIFEST_V2 = CAMPAIGN / "conference-campaign-manifest-v2.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _study_inputs(study_id: str) -> dict[str, Any]:
    manifest = json.loads(
        (CAMPAIGN / "conference-campaign-manifest.json").read_text(encoding="utf-8")
    )
    study = ANALYSIS._campaign_study(manifest, study_id)
    profile, profile_sha = ANALYSIS._profile_for_study(PROFILES, study_id)
    preregistration, preregistration_binding = (
        ANALYSIS._load_profile_preregistration(
            profile,
            profile_path=PROFILES,
            study_id=study_id,
        )
    )
    spec_path = (CAMPAIGN / str(study["spec"])).resolve()
    spec = ANALYSIS.load_comparison(spec_path, repo_root=REPO_ROOT)
    tasks = ANALYSIS._public_tasks(REPO_ROOT / spec.taskset.tasks)
    development, holdout, tags = ANALYSIS._partition_tasks(
        profile,
        preregistration,
        tasks,
    )
    locked_input_bindings = ANALYSIS._preregistered_input_bindings(
        profile=profile,
        preregistration=preregistration,
        profile_path=PROFILES,
        repo_root=REPO_ROOT,
    )
    return {
        "manifest": manifest,
        "study": study,
        "profile": profile,
        "profile_sha": profile_sha,
        "preregistration": preregistration,
        "preregistration_binding": preregistration_binding,
        "spec": spec,
        "tasks": tasks,
        "development": development,
        "holdout": holdout,
        "tags": tags,
        "locked_input_bindings": locked_input_bindings,
    }


def _superpowers_base_inputs() -> dict[str, Any]:
    document = json.loads(PROFILES.read_text(encoding="utf-8"))
    profile = next(
        item for item in document["profiles"] if item["id"] == "superpowers-writing-plans"
    )
    preregistration, preregistration_binding = (
        ANALYSIS._load_profile_preregistration(
            profile,
            profile_path=PROFILES,
            study_id=str(profile["study_ids"][0]),
        )
    )
    spec = ANALYSIS.load_comparison(
        REPO_ROOT
        / "examples/comparisons/superpowers-writing-plans-upgrade/confirmatory-v5.yaml",
        repo_root=REPO_ROOT,
    )
    tasks = ANALYSIS._public_tasks(REPO_ROOT / spec.taskset.tasks)
    development, holdout, tags = ANALYSIS._partition_tasks(
        profile,
        preregistration,
        tasks,
    )
    return {
        "profile": profile,
        "preregistration": preregistration,
        "preregistration_binding": preregistration_binding,
        "spec": spec,
        "tasks": tasks,
        "development": development,
        "holdout": holdout,
        "tags": tags,
    }


def _dimensions(profile: dict[str, Any]) -> tuple[str, ...]:
    dimensions = {
        str(dimension)
        for family in profile["primary_families"]
        for dimension in (
            [family["dimension"]]
            if family["aggregation"] == "dimension"
            else family["dimensions"]
        )
    }
    dimensions.update(profile["primary_composite"]["dimensions"])
    dimensions.update(profile.get("secondary_outcome_dimensions") or [])
    dimensions.update(profile.get("safety_dimensions") or [])
    dimensions.update(profile.get("candidate_all_attempt_dimensions") or [])
    return tuple(sorted(dimensions))


def _rows(
    inputs: dict[str, Any],
    score: Callable[[str, str, int, str], bool] | None = None,
) -> list[dict[str, Any]]:
    profile = inputs["profile"]
    safety = set(profile.get("safety_dimensions") or [])
    dimensions = _dimensions(profile)
    values: list[dict[str, Any]] = []
    for task in inputs["tasks"]:
        task_id = str(task["id"])
        for arm in ("baseline", "candidate"):
            for attempt in range(1, 5):
                scores = {
                    dimension: (
                        score(task_id, arm, attempt, dimension)
                        if score is not None
                        else True
                    )
                    for dimension in dimensions
                }
                identity = {
                    "task_id": task_id,
                    "variant_id": arm,
                    "harness": "claude-code",
                    "trial_index": attempt,
                }
                values.append(
                    {
                        "attempt_id": ANALYSIS.stable_digest(identity),
                        **identity,
                        "status": "passed",
                        "runtime_outcome": "completed",
                        "benchmark_outcome": "passed",
                        "comparison_deterministic_scores": scores,
                        "comparison_dimension_roles": {
                            dimension: (
                                "safety_gate" if dimension in safety else "outcome"
                            )
                            for dimension in dimensions
                        },
                        "comparison_deterministic_criticality": {
                            dimension: True for dimension in dimensions
                        },
                        "cost_usd": 0.25,
                        "latency_sec": 1.5,
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "tool_call_count": 2,
                    }
                )
    return values


def _fast_profile(profile: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(profile))
    value["bootstrap"] = {"samples": 64, "seed": 20260803}
    return value


def test_confirmatory_contract_is_frozen_before_execution() -> None:
    registration = json.loads(
        (CAMPAIGN / "conference-preregistration.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (CAMPAIGN / "conference-campaign-manifest.json").read_text(encoding="utf-8")
    )
    studies = [
        item for item in manifest["final_four"] if item["kind"] == "governed_comparison"
    ]
    assert registration["status"] == "frozen_before_execution"
    assert registration["sampling"] == {
        "arms": 2,
        "attempt_role": "within-task replication",
        "attempts_per_task_per_arm": 4,
        "cells_per_repository": 192,
        "development_tasks": 8,
        "inference_unit": "task",
        "task_replacement_after_unblinding": False,
        "tasks_per_repository": 24,
        "total_agent_cells": 576,
        "untouched_holdout_tasks": 16,
    }
    assert len(manifest["final_four"]) == 4
    assert len(studies) == 3
    assert all(item["cells"] == 192 for item in studies)


def test_historical_v1_protocol_and_manifest_remain_byte_exact() -> None:
    assert _sha(CAMPAIGN / "conference-preregistration.json") == (
        "f339962b2f24cf695a1efddc840ecd9161e0689060ce2e0ea418161b37ad45ec"
    )
    assert _sha(CAMPAIGN / "conference-campaign-manifest.json") == (
        "b4bde568c569cf736dd0ac7afeae0a6a054d0a8bfb6184315271a44b23467a35"
    )


def test_v2_protocol_is_descriptive_and_binds_recomputed_power() -> None:
    preregistration = json.loads(PROTOCOL_V2.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_V2.read_text(encoding="utf-8"))
    study = ANALYSIS._campaign_study(
        manifest, "superpowers-writing-plans-confirmatory-v6"
    )

    ANALYSIS._validate_descriptive_protocol_bindings(
        campaign_manifest_path=MANIFEST_V2,
        preregistration_path=PROTOCOL_V2,
        profile_path=PROFILES_V2,
        manifest=manifest,
        campaign_preregistration=preregistration,
        study=study,
    )
    power = ANALYSIS._prospective_power_design(
        preregistration_path=PROTOCOL_V2,
        campaign_preregistration=preregistration,
        manifest=manifest,
    )

    assert preregistration["claim_eligibility"]["conference_claim_eligible"] is False
    assert preregistration["claim_eligibility"]["population_claim_eligible"] is False
    assert power["status"] == "verified_underpowered_current_design"
    assert power["current_design"]["conference_claim_eligible"] is False
    assert power["recommended_design"]["campaign_cells"] == 6528
    assert len(preregistration["reserved_powered_studies"]) == 3
    assert all(
        item["status"] == "blocked_missing_reviewed_public_sampling_frame"
        for item in preregistration["reserved_powered_studies"]
    )
    assert all(
        item["sampling_frame"]["blocker"]["eligible_sources"] == 0
        and item["sampling_frame"]["blocker"]["required_sources"] == 288
        for item in preregistration["reserved_powered_studies"]
    )


def test_v2_vercel_profile_separates_outcome_safety_and_mechanism() -> None:
    profile, _digest = ANALYSIS._profile_for_study(
        PROFILES_V2,
        "vercel-react-best-practices-confirmatory-v2",
        campaign_id="community-skill-upgrade-measurement-development-v2",
    )
    outcome = set(profile["primary_composite"]["dimensions"])
    safety = set(profile["safety_dimensions"])
    mechanism = set(profile["mechanism_dimensions"])

    assert outcome == {
        "vercel-confirmatory.requested_change",
        "vercel-confirmatory.repository_grounding",
        "vercel-public-behavioral.public_regression_test_success",
    }
    assert safety == {
        "vercel-confirmatory.artifact_validity",
        "vercel-confirmatory.behavior_preservation",
        "vercel-confirmatory.scope_safety",
    }
    assert mechanism == {
        "vercel-mechanism.skill_material_opened",
        "vercel-mechanism.task_relevant_rule_opened",
    }
    assert not outcome & safety
    assert not outcome & mechanism
    assert not safety & mechanism
    assert all(
        family["aggregation"] == "mean_dimensions"
        and set(family["dimensions"]) == outcome
        for family in profile["primary_families"]
    )


@pytest.mark.parametrize(
    "study_id",
    (
        "superpowers-writing-plans-confirmatory-v6",
        "anthropic-skill-creator-confirmatory-v2",
        "vercel-react-best-practices-confirmatory-v2",
    ),
)
def test_v2_profiles_and_manifest_bind_exact_measurement_amendments(
    study_id: str,
) -> None:
    manifest = json.loads(MANIFEST_V2.read_text(encoding="utf-8"))
    profile, _digest = ANALYSIS._profile_for_study(
        PROFILES_V2,
        study_id,
        campaign_id=manifest["id"],
    )
    _preregistration, binding = ANALYSIS._load_profile_preregistration(
        profile,
        profile_path=PROFILES_V2,
        study_id=study_id,
    )

    ANALYSIS._validate_manifest_amendment_binding(
        study=ANALYSIS._campaign_study(manifest, study_id),
        preregistration_binding=binding,
        manifest_path=MANIFEST_V2,
    )
    assert binding["amendment"]["amendment_digest"]


def test_confirmatory_budget_policy_reconciles_active_specs_and_denies_legacy_authority() -> None:
    manifest = json.loads(
        (CAMPAIGN / "conference-campaign-manifest.json").read_text(encoding="utf-8")
    )
    binding = manifest["budget_policy"]
    policy_path = CAMPAIGN / binding["path"]
    policy_bytes = policy_path.read_bytes()
    assert hashlib.sha256(policy_bytes).hexdigest() == binding["sha256"]

    policy = json.loads(policy_bytes)
    assert policy["status"] == binding["status"] == "limits_only_not_approval"
    assert policy["authorization_contract"] == {
        "approval_unit": "one_exact_preview_per_study",
        "campaign_wide_approval_is_sufficient": False,
        "checked_in_policy_is_approval": False,
        "approval_receipts_are_checked_in": False,
        "required_preview_bindings": [
            "source",
            "taskset",
            "private_labels",
            "candidate",
            "runtime",
            "scorer",
            "judge_calibration",
            "evidence_destination",
            "cell_count",
            "maximum_cost_usd",
        ],
        "prior_approval_reuse": "forbidden_after_any_bound_input_changes",
    }
    legacy = policy["historical_canary_policy"]
    assert legacy["ceiling_usd"] == 110
    assert legacy["status"] == "audit_history_only"
    assert legacy["authorizes_confirmatory_execution"] is False
    assert policy["prerequisite_diagnostic"] == {
        "id": "community-skill-judge-calibration-v1",
        "status": "completed_advisory_only_pending_human_review",
        "new_cells": 0,
        "new_reserve_usd": 0,
        "new_cap_usd": 0,
        "authorizes_agent_studies": False,
    }

    expected = {
        "superpowers-writing-plans-confirmatory-v5": (192, 1632, 1700),
        "anthropic-skill-creator-confirmatory-v1": (192, 1632, 1640),
        "vercel-react-best-practices-confirmatory-v1": (192, 1632, 1640),
    }
    for study in policy["studies"]:
        spec = ANALYSIS.load_comparison(
            (CAMPAIGN / study["spec"]).resolve(), repo_root=REPO_ROOT
        )
        tasks = ANALYSIS._public_tasks(REPO_ROOT / spec.taskset.tasks)
        judge_reserve = sum(
            evaluator.reserve_cost_usd
            for evaluator in spec.evaluators
            if evaluator.type == "llm_judge"
        )
        cells = len(tasks) * 2 * spec.execution.attempts
        reserve_per_cell = spec.execution.reserve_per_attempt_usd + judge_reserve
        assert study["id"] == spec.id
        assert study["tasks"] == len(tasks) == 24
        assert study["arms"] == 2
        assert study["attempts_per_task_per_arm"] == spec.execution.attempts == 4
        assert study["cells"] == cells
        assert study["agent_reserve_per_cell_usd"] == 8.4
        assert study["judge_reserve_per_cell_usd"] == pytest.approx(judge_reserve)
        assert study["total_reserve_per_cell_usd"] == pytest.approx(
            reserve_per_cell
        )
        assert study["estimated_reserve_usd"] == pytest.approx(
            cells * reserve_per_cell
        )
        assert study["maximum_cost_usd"] == spec.execution.max_cost_usd
        assert spec.execution.qualification_inputs[
            "confirmatory_budget_policy_sha256"
        ] == str(policy_path.relative_to(REPO_ROOT))
        assert (
            study["cells"],
            study["estimated_reserve_usd"],
            study["maximum_cost_usd"],
        ) == expected[study["id"]]
        assert study["approval_status"] == "fresh_exact_preview_required"

    assert policy["totals"] == {
        "cells": 576,
        "estimated_reserve_usd": 4896,
        "sum_of_independent_maximum_cost_usd": 4980,
        "single_campaign_spend_authorization_exists": False,
    }


def test_v2_budget_policy_governs_exact_repaired_studies_and_current_calibration() -> None:
    manifest = json.loads(MANIFEST_V2.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_V2.read_text(encoding="utf-8"))
    binding = protocol["budget_policy"]
    policy_path = CAMPAIGN / binding["path"]
    policy = json.loads(policy_path.read_text(encoding="utf-8"))

    assert manifest["budget_policy"] == binding
    assert _sha(policy_path) == binding["sha256"]
    assert policy["schema_version"] == 2
    assert policy["id"] == binding["id"]
    assert policy["campaign_id"] == manifest["id"]
    assert policy["conference_claim_eligible"] is False
    assert policy["historical_v1_policy"] == {
        "path": "confirmatory-budget-policy-v1.json",
        "sha256": _sha(CAMPAIGN / "confirmatory-budget-policy-v1.json"),
        "status": "immutable_audit_history_only",
        "authorizes_v2_execution": False,
    }
    assert [item["id"] for item in policy["studies"]] == manifest[
        "execution_order"
    ]
    assert [item["spec"] for item in policy["studies"]] == [
        "../superpowers-writing-plans-upgrade/confirmatory-v6.yaml",
        "../anthropic-skill-creator-upgrade/confirmatory-v2.yaml",
        "../vercel-react-best-practices-upgrade/confirmatory-v2.yaml",
    ]
    assert [item["cells"] for item in policy["studies"]] == [192, 192, 192]
    calibration = json.loads(
        (CAMPAIGN / policy["judge_calibration"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    assert _sha(CAMPAIGN / policy["judge_calibration"]["path"]) == policy[
        "judge_calibration"
    ]["sha256"]
    assert policy["judge_calibration"]["status"] == calibration["review_status"]
    assert policy["judge_calibration"]["passed"] is calibration["passed"] is False
    assert policy["judge_calibration"]["reviewers_per_example"] == calibration[
        "reviewers_per_example"
    ] == 0
    assert policy["judge_calibration"]["judge_role"] == "secondary_advisory_only"

    ANALYSIS._validate_v2_budget_policy_bindings(
        campaign_manifest_path=MANIFEST_V2,
        preregistration_path=PROTOCOL_V2,
        manifest=manifest,
        campaign_preregistration=protocol,
    )


def test_v2_budget_policy_rejects_calibration_metadata_drift(monkeypatch) -> None:
    manifest = json.loads(MANIFEST_V2.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_V2.read_text(encoding="utf-8"))
    original = ANALYSIS._load_json

    def _drifted(path: Path, label: str) -> dict[str, Any]:
        value = original(path, label)
        if label == "V2 judge calibration":
            value["review_status"] = "passed"
        return value

    monkeypatch.setattr(ANALYSIS, "_load_json", _drifted)
    with pytest.raises(ValueError, match="calibration metadata drifted"):
        ANALYSIS._validate_v2_budget_policy_bindings(
            campaign_manifest_path=MANIFEST_V2,
            preregistration_path=PROTOCOL_V2,
            manifest=manifest,
            campaign_preregistration=protocol,
        )


def test_source_lock_always_binds_campaign_analysis_and_selection_code() -> None:
    assert SOURCE_LOCK.CAMPAIGN_SUPPORT_FILES == {
        "analyze_confirmatory.py": "confirmatory_analysis_implementation",
        "freeze_trace_audit.py": "trace_audit_selection_implementation",
        "generate_scientific_report.py": "scientific_report_implementation",
        "prepare_local_source_lock.py": "source_lock_implementation",
        "scientific-report-template-v3.json": "scientific_report_template",
    }
    assert all((CAMPAIGN / name).is_file() for name in SOURCE_LOCK.CAMPAIGN_SUPPORT_FILES)


@pytest.mark.parametrize(
    ("study_id", "samples", "families", "requirement"),
    [
        ("superpowers-writing-plans-confirmatory-v5", 20_000, 4, "all"),
        ("anthropic-skill-creator-confirmatory-v1", 10_000, 3, "any"),
        ("vercel-react-best-practices-confirmatory-v1", 10_000, 2, "any"),
    ],
)
def test_repository_analysis_profiles_bind_exact_frozen_contracts(
    study_id: str,
    samples: int,
    families: int,
    requirement: str,
) -> None:
    inputs = _study_inputs(study_id)
    profile = inputs["profile"]
    assert len(inputs["profile_sha"]) == 64
    assert len(inputs["preregistration_binding"]["sha256"]) == 64
    assert profile["bootstrap"] == {"samples": samples, "seed": 20260803}
    assert len(profile["primary_families"]) == families
    assert profile["decision"]["significant_family_requirement"] == requirement
    assert len(inputs["development"]) == 8
    assert len(inputs["holdout"]) == 16
    assert set(inputs["development"]).isdisjoint(inputs["holdout"])
    assert inputs["spec"].execution.attempts == 4
    assert inputs["spec"].execution.harnesses == ("claude-code",)
    if study_id.startswith("anthropic-"):
        assert inputs["locked_input_bindings"]["status"] == "matched"
        assert len(inputs["locked_input_bindings"]["inputs"]) == 5
    else:
        assert inputs["locked_input_bindings"] == {
            "status": "not_declared",
            "inputs": [],
        }


def test_preregistered_locked_input_drift_fails_closed() -> None:
    inputs = _study_inputs("anthropic-skill-creator-confirmatory-v1")
    drifted = json.loads(json.dumps(inputs["preregistration"]))
    drifted["locked_inputs"]["public_tasks_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="public_tasks_sha256"):
        ANALYSIS._preregistered_input_bindings(
            profile=inputs["profile"],
            preregistration=drifted,
            profile_path=PROFILES,
            repo_root=REPO_ROOT,
        )


def test_superpowers_v1_through_v4_are_rejected_and_v5_is_exactly_bound() -> None:
    for study_id in (
        "superpowers-writing-plans-confirmatory-v1",
        "superpowers-writing-plans-confirmatory-v2",
        "superpowers-writing-plans-confirmatory-v3",
        "superpowers-writing-plans-confirmatory-v4",
    ):
        with pytest.raises(ValueError, match="identify exactly one"):
            ANALYSIS._profile_for_study(PROFILES, study_id)
    profile, profile_sha = ANALYSIS._profile_for_study(
        PROFILES,
        "superpowers-writing-plans-confirmatory-v5",
    )
    assert len(profile_sha) == 64
    assert profile["study_ids"] == ["superpowers-writing-plans-confirmatory-v5"]
    assert profile["amendments"]["superpowers-writing-plans-confirmatory-v5"] == {
        "path": (
            "../superpowers-writing-plans-upgrade/"
            "preregistration-confirmatory-v5-amendment.json"
        ),
        "sha256": "acccc950f236102a19be1f023f6451c284a2f8b7b375788c9373f1e3ec1cce37",
        "amendment_digest": (
            "5b35d3d4ac692f6b7be4be160f1c0d799a34ecb68f33fa24f747de2305394bcd"
        ),
    }
    inputs = _superpowers_base_inputs()
    assert inputs["preregistration_binding"]["amendment"]["sha256"] == (
        "acccc950f236102a19be1f023f6451c284a2f8b7b375788c9373f1e3ec1cce37"
    )
    assert inputs["preregistration_binding"]["amendment"][
        "amendment_digest"
    ] == (
        "5b35d3d4ac692f6b7be4be160f1c0d799a34ecb68f33fa24f747de2305394bcd"
    )
    assert inputs["profile"]["historical_rejected_study_ids"] == [
        "superpowers-writing-plans-confirmatory-v1",
        "superpowers-writing-plans-confirmatory-v2",
        "superpowers-writing-plans-confirmatory-v3",
        "superpowers-writing-plans-confirmatory-v4",
    ]
    assert inputs["profile"]["sensitivity_analysis"]["restart_exposure"] == {
        "status": "required_descriptive",
        "source_execution": "superpowers-writing-plans-confirmatory-v3",
        "excluded_holdout_task_ids": ["sp-holdout-research-event-projection"],
        "claim_effect": (
            "Repeat the frozen primary summary without the one holdout task whose V3 "
            "pair was exposed. Report whether the conclusion changes; this sensitivity "
            "cannot replace the V1 primary analysis."
        ),
    }


def test_superpowers_v5_manifest_binds_amendment_bytes_and_canonical_digest() -> None:
    inputs = _study_inputs("superpowers-writing-plans-confirmatory-v5")
    expected = {
        "path": (
            "../superpowers-writing-plans-upgrade/"
            "preregistration-confirmatory-v5-amendment.json"
        ),
        "sha256": "acccc950f236102a19be1f023f6451c284a2f8b7b375788c9373f1e3ec1cce37",
        "amendment_digest": (
            "5b35d3d4ac692f6b7be4be160f1c0d799a34ecb68f33fa24f747de2305394bcd"
        ),
    }
    assert inputs["study"]["amendment"] == expected
    ANALYSIS._validate_manifest_amendment_binding(
        study=inputs["study"],
        preregistration_binding=inputs["preregistration_binding"],
        manifest_path=CAMPAIGN / "conference-campaign-manifest.json",
    )

    drifted = json.loads(json.dumps(inputs["study"]))
    drifted["amendment"]["amendment_digest"] = "0" * 64
    with pytest.raises(ValueError, match="campaign amendment binding drifted"):
        ANALYSIS._validate_manifest_amendment_binding(
            study=drifted,
            preregistration_binding=inputs["preregistration_binding"],
            manifest_path=CAMPAIGN / "conference-campaign-manifest.json",
        )


def test_superpowers_v5_amendment_canonical_drift_fails_closed(
    tmp_path: Path,
) -> None:
    document = json.loads(PROFILES.read_text(encoding="utf-8"))
    profile = next(
        item
        for item in document["profiles"]
        if item["id"] == "superpowers-writing-plans"
    )
    amendment_path = (
        REPO_ROOT
        / "examples/comparisons/superpowers-writing-plans-upgrade/"
        "preregistration-confirmatory-v5-amendment.json"
    )
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    amendment["claim_boundary"] += " Undisclosed drift."
    drifted_path = tmp_path / "drifted-amendment.json"
    drifted_path.write_text(
        json.dumps(amendment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    binding = profile["amendments"]["superpowers-writing-plans-confirmatory-v5"]
    binding["path"] = drifted_path.as_posix()
    binding["sha256"] = ANALYSIS._sha256(drifted_path)

    with pytest.raises(
        ValueError,
        match="repository preregistration amendment digest drifted",
    ):
        ANALYSIS._load_profile_preregistration(
            profile,
            profile_path=PROFILES,
            study_id="superpowers-writing-plans-confirmatory-v5",
        )


def test_primary_inference_uses_holdout_only() -> None:
    inputs = _superpowers_base_inputs()
    primary = set(inputs["profile"]["primary_composite"]["dimensions"])
    development = set(inputs["development"])

    def score(task: str, arm: str, _attempt: int, dimension: str) -> bool:
        if task in development and dimension in primary:
            return arm == "candidate"
        return True

    report = ANALYSIS._preregistered_analysis(
        rows=_rows(inputs, score),
        profile=_fast_profile(inputs["profile"]),
        development=inputs["development"],
        holdout=inputs["holdout"],
        tags=inputs["tags"],
    )
    assert report["primary_partition"] == "holdout"
    assert report["finding"]["status"] == "unchanged"
    assert report["finding"]["equivalence_established"] is True
    assert any(
        item["descriptive_difference"] == 1.0
        for item in report["development_descriptive"]["dimensions"]
        if item["dimension"] in primary
    )


def test_superpowers_restart_exposure_sensitivity_is_computed() -> None:
    inputs = _superpowers_base_inputs()
    exposed = "sp-holdout-research-event-projection"
    safety = inputs["profile"]["safety_dimensions"][0]

    def score(task: str, arm: str, _attempt: int, dimension: str) -> bool:
        if task == exposed and arm == "candidate" and dimension == safety:
            return False
        return True

    report = ANALYSIS._preregistered_analysis(
        rows=_rows(inputs, score),
        profile=_fast_profile(inputs["profile"]),
        development=inputs["development"],
        holdout=inputs["holdout"],
        tags=inputs["tags"],
    )
    restart = report["sensitivity_analysis"]["restart_exposure"]

    assert report["finding"]["status"] == "regressed"
    assert restart["analysis_status"] == "complete"
    assert restart["excluded_holdout_task_ids"] == [exposed]
    assert exposed not in restart["included_holdout_task_ids"]
    assert restart["included_holdout_task_count"] == 15
    assert restart["holdout"]["task_count"] == 15
    assert restart["sensitivity_finding"]["status"] == "unchanged"
    assert restart["conclusion_changed"] is True
    assert restart["role"] == (
        "mandatory_descriptive_sensitivity_not_primary_inference"
    )


@pytest.mark.parametrize(
    "excluded",
    [
        [],
        ["sp-holdout-research-event-projection"] * 2,
        ["sp-dev-credential-rotation"],
        ["not-a-frozen-task"],
    ],
)
def test_restart_exposure_sensitivity_rejects_invalid_exclusions(
    excluded: list[str],
) -> None:
    inputs = _superpowers_base_inputs()
    profile = _fast_profile(inputs["profile"])
    profile["sensitivity_analysis"]["restart_exposure"][
        "excluded_holdout_task_ids"
    ] = excluded

    with pytest.raises(ValueError, match="restart-exposure"):
        ANALYSIS._preregistered_analysis(
            rows=_rows(inputs),
            profile=profile,
            development=inputs["development"],
            holdout=inputs["holdout"],
            tags=inputs["tags"],
        )


def test_superpowers_requires_all_four_holm_adjusted_primary_families() -> None:
    inputs = _superpowers_base_inputs()
    primary = set(inputs["profile"]["primary_composite"]["dimensions"])
    holdout = set(inputs["holdout"])

    def score(task: str, arm: str, _attempt: int, dimension: str) -> bool:
        if task in holdout and dimension in primary:
            return arm == "candidate"
        return True

    report = ANALYSIS._preregistered_analysis(
        rows=_rows(inputs, score),
        profile=inputs["profile"],
        development=inputs["development"],
        holdout=inputs["holdout"],
        tags=inputs["tags"],
    )
    families = report["holdout"]["primary_families"]
    assert report["finding"]["status"] == "improved"
    assert len(report["finding"]["significant_primary_families"]) == 4
    assert all(item["holm_adjusted_p"] <= 0.05 for item in families)
    assert all(
        item["bootstrap"] == {"samples": 20_000, "seed": 20260803}
        for item in families
    )
    assert all(
        item["can_clear_frozen_alpha"]
        for item in report["holdout"]["primary_test_attainability"]
    )


def test_anthropic_any_primary_family_still_requires_composite_threshold() -> None:
    inputs = _study_inputs("anthropic-skill-creator-confirmatory-v1")
    improved_dimension = inputs["profile"]["primary_families"][0]["dimension"]
    holdout = set(inputs["holdout"])

    def score(task: str, arm: str, _attempt: int, dimension: str) -> bool:
        if task in holdout and dimension == improved_dimension:
            return arm == "candidate"
        return True

    report = ANALYSIS._preregistered_analysis(
        rows=_rows(inputs, score),
        profile=_fast_profile(inputs["profile"]),
        development=inputs["development"],
        holdout=inputs["holdout"],
        tags=inputs["tags"],
    )
    assert report["finding"]["status"] == "improved"
    assert report["finding"]["significant_primary_families"] == [
        "frontmatter_semantics"
    ]
    family_ids = {
        item["id"] for item in report["holdout"]["primary_families"]
    }
    assert "instruction_quality" not in family_ids


def test_vercel_uses_target_families_and_all_attempt_candidate_gate() -> None:
    inputs = _study_inputs("vercel-react-best-practices-confirmatory-v1")
    target_task = next(
        task_id
        for task_id in inputs["holdout"]
        if "server-action-security" in inputs["tags"][task_id]
    )
    requested_change = "vercel-confirmatory.requested_change"

    def score(task: str, _arm: str, attempt: int, dimension: str) -> bool:
        if (
            task == target_task
            and attempt == 1
            and dimension == requested_change
        ):
            return False
        return True

    report = ANALYSIS._preregistered_analysis(
        rows=_rows(inputs, score),
        profile=_fast_profile(inputs["profile"]),
        development=inputs["development"],
        holdout=inputs["holdout"],
        tags=inputs["tags"],
    )
    families = report["holdout"]["primary_families"]
    assert {item["id"]: item["task_count"] for item in families} == {
        "server_action_security": 4,
        "rsc_serialization": 4,
    }
    assert all(
        not item["can_clear_frozen_alpha"]
        for item in report["holdout"]["primary_test_attainability"]
    )
    assert {
        item["minimum_holm_adjusted_p_at_rank_one"]
        for item in report["holdout"]["primary_test_attainability"]
    } == {0.25}
    assert report["finding"]["status"] == "inconclusive"
    assert report["finding"]["candidate_critical_failures"] == [
        {
            "task_id": target_task,
            "attempt": 1,
            "dimension": requested_change,
        }
    ]


def test_any_paired_safety_regression_fails_closed_with_task_blocker() -> None:
    inputs = _superpowers_base_inputs()
    affected_task = inputs["holdout"][0]
    safety = inputs["profile"]["safety_dimensions"][0]

    def score(task: str, arm: str, _attempt: int, dimension: str) -> bool:
        if task == affected_task and arm == "candidate" and dimension == safety:
            return False
        return True

    report = ANALYSIS._preregistered_analysis(
        rows=_rows(inputs, score),
        profile=_fast_profile(inputs["profile"]),
        development=inputs["development"],
        holdout=inputs["holdout"],
        tags=inputs["tags"],
    )
    assert report["finding"]["status"] == "regressed"
    assert report["finding"]["safety_regressions"] == [
        {"dimension": safety, "task_ids": [affected_task]}
    ]
    assert affected_task in report["finding"]["critical_blockers"][0]


def test_shared_candidate_safety_failure_is_inconclusive_not_unchanged() -> None:
    inputs = _superpowers_base_inputs()
    affected_task = inputs["holdout"][0]
    safety = inputs["profile"]["safety_dimensions"][0]

    def score(task: str, _arm: str, _attempt: int, dimension: str) -> bool:
        return not (task == affected_task and dimension == safety)

    report = ANALYSIS._preregistered_analysis(
        rows=_rows(inputs, score),
        profile=_fast_profile(inputs["profile"]),
        development=inputs["development"],
        holdout=inputs["holdout"],
        tags=inputs["tags"],
    )
    assert report["finding"]["status"] == "inconclusive"
    assert len(report["finding"]["candidate_critical_failures"]) == 4
    assert all(
        item["task_id"] == affected_task and item["dimension"] == safety
        for item in report["finding"]["candidate_critical_failures"]
    )


def test_matrix_requires_every_unique_preregistered_coordinate() -> None:
    inputs = _study_inputs("anthropic-skill-creator-confirmatory-v1")
    rows = _rows(inputs)
    ANALYSIS._validate_matrix(
        rows=rows,
        tasks=[str(item["id"]) for item in inputs["tasks"]],
        attempts=4,
        harnesses=("claude-code",),
    )
    with pytest.raises(ValueError, match="duplicate task/arm/attempt coordinate"):
        ANALYSIS._validate_matrix(
            rows=[*rows, dict(rows[0])],
            tasks=[str(item["id"]) for item in inputs["tasks"]],
            attempts=4,
            harnesses=("claude-code",),
        )


def test_matrix_rejects_scored_agent_timeout_as_missing_attempt() -> None:
    inputs = _study_inputs("anthropic-skill-creator-confirmatory-v1")
    rows = _rows(inputs)
    rows[0] = {
        **rows[0],
        "status": "passed",
        "runtime_outcome": "timed_out",
        "benchmark_outcome": "failed",
    }

    with pytest.raises(ValueError, match="runtime outcome is not completed"):
        ANALYSIS._validate_matrix(
            rows=rows,
            tasks=[str(item["id"]) for item in inputs["tasks"]],
            attempts=4,
            harnesses=("claude-code",),
        )


def test_attempt_rows_must_share_one_digest_verified_approval_lock() -> None:
    unsigned = {
        "kind": "approved_comparison_execution",
        "comparison_id": "study-v1",
        "approval_digest": "a" * 64,
        "preview_digest": "b" * 64,
    }
    approved = {**unsigned, "lock_digest": ANALYSIS.stable_digest(unsigned)}
    assert ANALYSIS._approved_execution_lock(
        [{"approved_comparison": approved}, {"approved_comparison": approved}]
    ) == approved
    drifted_unsigned = {**unsigned, "preview_digest": "c" * 64}
    drifted = {
        **drifted_unsigned,
        "lock_digest": ANALYSIS.stable_digest(drifted_unsigned),
    }
    with pytest.raises(ValueError, match="disagree on the approved execution lock"):
        ANALYSIS._approved_execution_lock(
            [{"approved_comparison": approved}, {"approved_comparison": drifted}]
        )
    with pytest.raises(ValueError, match="lock digest does not match"):
        ANALYSIS._approved_execution_lock(
            [{"approved_comparison": {**approved, "lock_digest": "d" * 64}}]
        )


def test_attempt_rows_may_reference_one_verified_approval_sidecar(
    tmp_path: Path,
) -> None:
    unsigned = {
        "kind": "approved_comparison_execution",
        "comparison_id": "study-v1",
        "approval_digest": "a" * 64,
        "preview_digest": "b" * 64,
    }
    approved = {**unsigned, "lock_digest": ANALYSIS.stable_digest(unsigned)}
    lock_path = tmp_path / "approved-comparison.lock.json"
    lock_path.write_text(json.dumps(approved), encoding="utf-8")
    rows = [
        {"approved_comparison_lock_digest": approved["lock_digest"]},
        {"approved_comparison_lock_digest": approved["lock_digest"]},
    ]
    assert ANALYSIS._approved_execution_lock(rows, lock_path=lock_path) == approved
    with pytest.raises(ValueError, match="sidecar lock"):
        ANALYSIS._approved_execution_lock(
            [{"approved_comparison_lock_digest": "c" * 64}],
            lock_path=lock_path,
        )
    lock_path.unlink()
    with pytest.raises(ValueError, match="sidecar is missing"):
        ANALYSIS._approved_execution_lock(rows, lock_path=lock_path)


def test_trace_audit_selection_must_be_bound_by_the_approval() -> None:
    selection_digest = "a" * 64
    trace_audit = {"selection_digest": selection_digest}
    approved = {
        "approval_input_bindings": {
            "trace_audit_selection": selection_digest,
        }
    }
    ANALYSIS._validate_trace_audit_approval_binding(approved, trace_audit)
    with pytest.raises(ValueError, match="does not bind"):
        ANALYSIS._validate_trace_audit_approval_binding(
            {"approval_input_bindings": {}}, trace_audit
        )
    with pytest.raises(ValueError, match="does not bind"):
        ANALYSIS._validate_trace_audit_approval_binding(
            {
                "approval_input_bindings": {
                    "trace_audit_selection": "b" * 64,
                }
            },
            trace_audit,
        )


def test_canonical_v3_must_recompute_exactly_from_exported_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [{"attempt_id": "a" * 64}]
    result = SimpleNamespace(
        comparison_id="study-v3",
        preview_digest="b" * 64,
        source="live",
        evidence_project="wandb/result",
        evidence_topology=SimpleNamespace(
            source_destination=SimpleNamespace(project_slug="wandb/source")
        ),
        decision_policy={"policy": "frozen"},
        decision=SimpleNamespace(attestation=None),
        release_note_coverage=(),
        supersedes=(),
        aligned_analysis=SimpleNamespace(study_intent="confirmatory"),
        to_dict=lambda: {"canonical": True},
    )
    captured: dict[str, Any] = {}

    def recompute(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(to_dict=lambda: {"canonical": True})

    monkeypatch.setattr(ANALYSIS, "analyze_comparison_rows", recompute)
    ANALYSIS._recompute_canonical_result(result, rows, {"lock_digest": "c" * 64})
    assert captured["rows"] is rows
    assert captured["result_schema_version"] == 3
    assert captured["expected_evidence_project"] == "wandb/result"
    assert captured["expected_source_evidence_project"] == "wandb/source"

    monkeypatch.setattr(
        ANALYSIS,
        "analyze_comparison_rows",
        lambda **_kwargs: SimpleNamespace(to_dict=lambda: {"canonical": False}),
    )
    with pytest.raises(ValueError, match="disagrees with the exact exported"):
        ANALYSIS._recompute_canonical_result(
            result,
            rows,
            {"lock_digest": "c" * 64},
        )


def test_bindings_require_exact_topology_scorers_and_task_validity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt_id = "a" * 64
    source_lock = "b" * 64
    topology_identity = "c" * 64
    lock_digest = "d" * 64
    scorer_digest = "e" * 64
    destination = lambda slug: SimpleNamespace(  # noqa: E731
        project_slug=slug,
        to_dict=lambda: {"project_slug": slug},
    )
    drift = SimpleNamespace(
        status="matched",
        to_dict=lambda: {"status": "matched"},
    )
    topology = SimpleNamespace(
        pre_run_drift=drift,
        post_run_drift=drift,
        execution_identity=topology_identity,
        source_lock_digest=source_lock,
        source_destination=destination("wandb/source"),
        result_destination=destination("wandb/result"),
        topology_digest="f" * 64,
    )
    lineage = {
        "arms": {
            "baseline": {
                "source_revisions": [
                    {"kind": "skill", "version_identity": "git:baseline"}
                ]
            },
            "candidate": {
                "source_revisions": [
                    {"kind": "skill", "version_identity": "git:candidate"}
                ]
            },
        },
        "scorer_digests": {"scorer": scorer_digest},
    }
    scorer = SimpleNamespace(
        id="scorer",
        digest=scorer_digest,
        to_dict=lambda: {"id": "scorer", "digest": scorer_digest},
    )
    runtime = SimpleNamespace(
        to_dict=lambda: {"id": "runtime", "digest": "1" * 64}
    )
    validity = SimpleNamespace(
        task_id="task-1",
        status="valid",
        to_dict=lambda: {"task_id": "task-1", "status": "valid"},
    )
    result = SimpleNamespace(
        comparison_id="study-v3",
        preview_digest="2" * 64,
        evidence_project="wandb/result",
        rows=1,
        paired_cases=(
            SimpleNamespace(
                baseline=SimpleNamespace(attempt_id=attempt_id),
                candidate=None,
            ),
        ),
        integrity={
            "status": "reconciled",
            "approved_manifest_digest": lock_digest,
            "expected_cell_count": 1,
        },
        evidence_topology=topology,
        cohort_lineage=lineage,
        scorer_revisions=(scorer,),
        runtime_locks=(runtime,),
        task_validity=(validity,),
        candidate_source_revisions=(),
        result_digest="3" * 64,
        qualification_digest="4" * 64,
    )
    approved = {
        "preview_digest": result.preview_digest,
        "comparison_id": result.comparison_id,
        "spec_digest": "5" * 64,
        "evidence_project": result.evidence_project,
        "expected_cell_count": 1,
        "lock_digest": lock_digest,
        "evidence_topology_identity": topology_identity,
        "source_lock_digest": source_lock,
        "source_evidence_project": "wandb/source",
        "approval_digest": "6" * 64,
    }
    spec = SimpleNamespace(id=result.comparison_id, spec_digest=approved["spec_digest"])
    study = {"id": result.comparison_id, "evidence_project": result.evidence_project}
    monkeypatch.setattr(
        ANALYSIS,
        "_comparison_cohort_lineage",
        lambda *_args, **_kwargs: lineage,
    )
    monkeypatch.setattr(
        ANALYSIS,
        "_validate_attempt_scorer_receipts",
        lambda **_kwargs: "7" * 64,
    )
    bindings = ANALYSIS._validate_bindings(
        result=result,
        rows=[{"attempt_id": attempt_id}],
        approved=approved,
        spec=spec,
        study=study,
        profile={"baseline_commit": "baseline", "candidate_commit": "candidate"},
        repo_root=REPO_ROOT,
    )
    assert bindings["source_pre_run_drift"] == {"status": "matched"}
    assert bindings["scorer_revisions"] == [
        {"id": "scorer", "digest": scorer_digest}
    ]
    assert bindings["attempt_scorer_receipts_digest"] == "7" * 64

    result.task_validity = (
        SimpleNamespace(
            task_id="task-1",
            status="drifted",
            to_dict=lambda: {"task_id": "task-1", "status": "drifted"},
        ),
    )
    with pytest.raises(ValueError, match="task validity blocks"):
        ANALYSIS._validate_bindings(
            result=result,
            rows=[{"attempt_id": attempt_id}],
            approved=approved,
            spec=spec,
            study=study,
            profile={
                "baseline_commit": "baseline",
                "candidate_commit": "candidate",
            },
            repo_root=REPO_ROOT,
        )


def test_attempt_scorer_receipt_rejects_score_only_mutation() -> None:
    scorer_path = (
        REPO_ROOT
        / "examples/comparisons/superpowers-writing-plans-upgrade/"
        "plan_quality_scorer_v3.py"
    )
    evaluator = SimpleNamespace(
        id="plan-quality-confirmatory",
        type="deterministic",
        scorer=scorer_path.relative_to(REPO_ROOT).as_posix(),
        runtime="python312-sandbox-v1",
        dimensions=("artifact_validity",),
    )
    profile = ANALYSIS.load_task_profiles(REPO_ROOT).scorer_runtime(
        evaluator.runtime
    )
    output = "# Exact plan\n\nUse the locked repository evidence."
    encoded = output.encode()
    expected_paths = ["/logs/artifacts/fugue-answer.md"]
    primary = {
        "schema_version": 1,
        "status": "complete",
        "source": "fugue_answer_artifact",
        "source_path": "/logs/artifacts/fugue-answer.md",
        "path": "artifacts/logs/artifacts/fugue-answer.md",
        "contract_version": 1,
        "locked_artifact_paths_digest": ANALYSIS.stable_digest(expected_paths),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "truncated": False,
    }
    revision = "a" * 64
    input_unsigned = {
        "schema_version": 1,
        "status": "bound",
        "source_sha256": "b" * 64,
        "input_bytes": 100,
        "input_sha256": "c" * 64,
        "evidence_digest": "d" * 64,
        "reference_digest": "e" * 64,
        "reference_output_digest": ANALYSIS.stable_digest(output),
        "runtime_profile_id": profile.id,
        "runtime_profile_digest": profile.profile_digest,
        "runtime_image": profile.image,
        "runtime_platform": profile.platform,
    }
    input_receipt = {
        **input_unsigned,
        "receipt_digest": ANALYSIS.stable_digest(input_unsigned),
    }
    cleanup_unsigned = {
        "schema_version": 1,
        "status": "verified_absent",
        "container_name_sha256": "f" * 64,
    }
    cleanup = {
        **cleanup_unsigned,
        "receipt_digest": ANALYSIS.stable_digest(cleanup_unsigned),
    }
    scores = {"artifact_validity": True}
    scorer_unsigned = {
        "schema_version": 1,
        "status": "bound",
        "evaluator_id": evaluator.id,
        "evaluator_revision_digest": revision,
        "approved_scorer_source_sha256": ANALYSIS._sha256(scorer_path),
        "primary_artifact_receipt_digest": ANALYSIS.stable_digest(primary),
        "primary_artifact_sha256": primary["sha256"],
        "primary_artifact_bytes": primary["bytes"],
        "normalized_output_digest": ANALYSIS.stable_digest(output),
        "normalized_scores_digest": ANALYSIS.stable_digest(scores),
        "scorer_input_receipt": input_receipt,
        "runtime_cleanup_receipt": cleanup,
    }
    scorer_receipt = {
        **scorer_unsigned,
        "receipt_digest": ANALYSIS.stable_digest(scorer_unsigned),
    }
    row = {
        "attempt_id": "1" * 64,
        "expected_artifact_paths": expected_paths,
        "final_output": output,
        "comparison_primary_output": primary,
        "comparison_deterministic_scores": {
            f"{evaluator.id}.artifact_validity": True
        },
        "comparison_deterministic_scorer_receipts": {
            evaluator.id: scorer_receipt
        },
    }
    result = SimpleNamespace(
        scorer_revisions=(SimpleNamespace(id=evaluator.id, digest=revision),)
    )
    spec = SimpleNamespace(evaluators=(evaluator,))

    assert len(
        ANALYSIS._validate_attempt_scorer_receipts(
            rows=[row], result=result, spec=spec, repo_root=REPO_ROOT
        )
    ) == 64
    row["comparison_deterministic_scores"][
        f"{evaluator.id}.artifact_validity"
    ] = False
    with pytest.raises(ValueError, match="approved artifact"):
        ANALYSIS._validate_attempt_scorer_receipts(
            rows=[row], result=result, spec=spec, repo_root=REPO_ROOT
        )


def test_direct_evidence_requires_five_resolved_weave_links() -> None:
    def evidence_link(kind: str, status: str = "resolved") -> Any:
        return SimpleNamespace(
            kind=kind,
            status=status,
            ref=f"weave://{kind}",
            url=f"https://wandb.ai/weave/calls/{kind}",
            to_dict=lambda: {
                "kind": kind,
                "status": status,
                "ref": f"weave://{kind}",
                "url": f"https://wandb.ai/weave/calls/{kind}",
            },
        )

    links = tuple(evidence_link(kind) for kind in ANALYSIS.REQUIRED_LINK_KINDS)
    baseline = SimpleNamespace(attempt_id="a" * 64, evidence_links=links)
    candidate = SimpleNamespace(attempt_id="b" * 64, evidence_links=links)
    result = SimpleNamespace(
        paired_cases=(
            SimpleNamespace(
                task_id="task-1",
                baseline=baseline,
                candidate=candidate,
            ),
        )
    )
    assert len(ANALYSIS._direct_links(result)) == 2
    candidate.evidence_links = (*links[:-1], evidence_link(links[-1].kind, "missing"))
    with pytest.raises(ValueError, match="lacks five resolved evidence links"):
        ANALYSIS._direct_links(result)


def test_aligned_case_evidence_keeps_safe_explanations_and_judge_separate() -> None:
    def attempt(attempt_id: str) -> Any:
        return SimpleNamespace(
            attempt_id=attempt_id,
            identity={"candidate_fingerprint": "a" * 64},
            passed=True,
            execution_status="completed",
            evaluation_status="completed",
            evidence_status="reconciled",
            scores={
                "deterministic.correct": True,
                "comparison.judge.community-usefulness": 0.8,
            },
            score_explanations={
                "deterministic.correct": "The host verifier passed.",
                "comparison.judge.community-usefulness": "Advisory only.",
            },
            sanitized_answer_excerpt="A bounded, sanitized excerpt.",
            tools=("Skill",),
            actual_query_scope=(),
            reported_project_identity=None,
            cost_usd=0.25,
            latency_sec=1.5,
            input_tokens=100,
            output_tokens=50,
            tool_calls=1,
            execution_fingerprint="b" * 64,
            runtime_lock_digest="c" * 64,
        )

    change = SimpleNamespace(
        to_dict=lambda: {
            "id": "deterministic.correct",
            "role": "outcome",
            "status": "unchanged",
        }
    )
    result = SimpleNamespace(
        paired_cases=(
            SimpleNamespace(
                pair_id="pair-1",
                task_id="task-1",
                task_label="Task one",
                harness="claude-code",
                attempt=1,
                status="unchanged",
                dimension_changes=(change,),
                baseline=attempt("d" * 64),
                candidate=attempt("e" * 64),
            ),
        )
    )
    row = ANALYSIS._aligned_case_evidence(result)[0]
    assert row["baseline"]["deterministic_scores"] == {
        "deterministic.correct": True
    }
    assert row["baseline"]["score_explanations"] == {
        "deterministic.correct": "The host verifier passed."
    }
    assert row["baseline"]["sanitized_answer_excerpt"] == (
        "A bounded, sanitized excerpt."
    )


def test_mechanism_evidence_requires_skill_assignment_registration_and_use() -> None:
    stages = {
        stage: {
            arm: {"observed": 1, "applicable": 1, "unavailable": 0}
            for arm in ("baseline", "candidate")
        }
        for stage in ("skill_assigned", "skill_registered", "skill_invoked")
    }
    rows = [
        {
            "task_id": "task-1",
            "variant_id": arm,
            "attempt_id": ("a" if arm == "baseline" else "b") * 64,
            "comparison_deterministic_scores": {"skill.opened": True},
            "comparison_dimension_roles": {"skill.opened": "mechanism"},
        }
        for arm in ("baseline", "candidate")
    ]
    result = SimpleNamespace(mechanism_summary=stages)
    summary = ANALYSIS._mechanism_summary(result, rows, ("task-1",))
    assert summary["role"] == "mechanism_only_not_task_outcome"
    assert summary["deterministic_dimensions"] == [
        {"dimension": "skill.opened", "baseline_rate": 1.0, "candidate_rate": 1.0}
    ]

    result.mechanism_summary["skill_invoked"]["candidate"]["observed"] = 0
    with pytest.raises(ValueError, match="incomplete skill_invoked evidence"):
        ANALYSIS._mechanism_summary(result, rows, ("task-1",))


def _trace_audit_inputs() -> tuple[Any, dict[str, Any], dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    pairs = []
    for task_index, partition in enumerate(("dev", "holdout")):
        task_id = f"as-{partition}-audit-{task_index}"
        attempts = []
        for variant_index, variant in enumerate(("baseline", "candidate")):
            attempt_id = f"{task_index}{variant_index}".ljust(64, "a")
            attempts.append(SimpleNamespace(attempt_id=attempt_id))
            cells.append(
                {
                    "applicable": True,
                    "attempt_id": attempt_id,
                    "harness": "claude-code",
                    "task_id": task_id,
                    "trial_index": 1,
                    "variant_id": variant,
                }
            )
        pairs.append(
            SimpleNamespace(
                task_id=task_id,
                harness="claude-code",
                attempt=1,
                status="unchanged",
                dimension_changes=(),
                baseline=attempts[0],
                candidate=attempts[1],
            )
        )
    result = SimpleNamespace(
        comparison_id="anthropic-skill-creator-confirmatory-v1",
        preview_digest="f" * 64,
        result_digest="e" * 64,
        paired_cases=tuple(pairs),
    )
    selection = AUDIT.freeze(
        {"preview_digest": result.preview_digest, "matrix": {"matrix_cells": cells}},
        fraction=0.5,
    )
    required = selection["selected_attempt_ids"]
    checks = {key: True for key in ANALYSIS.REQUIRED_AUDIT_CHECKS}
    review: dict[str, Any] = {
        "schema_version": 1,
        "kind": "completed_blinded_trace_audit",
        "campaign_id": "community-skill-upgrade-confirmatory-campaign-v1",
        "study_id": result.comparison_id,
        "status": "completed",
        "preview_digest": result.preview_digest,
        "result_digest": result.result_digest,
        "selection_digest": selection["selection_digest"],
        "selected_attempt_ids": required,
        "required_attempt_ids": required,
        "reviewed_attempts": [
            {
                "attempt_id": attempt_id,
                "reviews": [
                    {
                        "reviewer": reviewer,
                        "disposition": "verified",
                        "checks": checks,
                        "reason": "Verified against the frozen evidence chain.",
                    }
                    for reviewer in ("reviewer-a", "reviewer-b")
                ],
                "adjudication": None,
            }
            for attempt_id in required
        ],
        "reviewer_attestations": [],
    }
    attested = ANALYSIS._audit_attested_digest(review)
    review["reviewer_attestations"] = [
        {
            "reviewer": reviewer,
            "reviewed_at": "2026-08-03T18:00:00Z",
            "artifact_digest": attested,
        }
        for reviewer in ("reviewer-a", "reviewer-b")
    ]
    return result, selection, review


def test_trace_audit_is_required_complete_blinded_and_bound_to_result(
    tmp_path: Path,
) -> None:
    result, selection, review = _trace_audit_inputs()
    selection_path = tmp_path / "selection.json"
    review_path = tmp_path / "review.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    review_path.write_text(json.dumps(review), encoding="utf-8")
    profile = _study_inputs(result.comparison_id)["profile"]
    summary = ANALYSIS._validate_trace_audit(
        result=result,
        selection_path=selection_path,
        review_path=review_path,
        profile=profile,
        preregistration={},
        campaign_id="community-skill-upgrade-confirmatory-campaign-v1",
    )
    assert summary["status"] == "completed"
    assert summary["reviewers"] == ["reviewer-a", "reviewer-b"]

    incomplete = {**review, "status": "in_progress"}
    review_path.write_text(json.dumps(incomplete), encoding="utf-8")
    with pytest.raises(ValueError, match="completed trace audit is incomplete"):
        ANALYSIS._validate_trace_audit(
            result=result,
            selection_path=selection_path,
            review_path=review_path,
            profile=profile,
            preregistration={},
            campaign_id="community-skill-upgrade-confirmatory-campaign-v1",
        )


def test_sign_test_and_holm_are_exact_and_monotone() -> None:
    assert ANALYSIS._two_sided_sign_p({"a": 1.0, "b": 1.0, "c": 1.0}) == 0.25
    adjusted = ANALYSIS._holm({"a": 0.01, "b": 0.03, "c": 0.2})
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.2}


def test_trace_audit_selection_is_blinded_paired_and_deterministic() -> None:
    cells = []
    for task_index in range(24):
        partition = "dev" if task_index < 8 else "holdout"
        task_id = f"sp-{partition}-task-{task_index:02d}"
        for attempt in range(1, 5):
            for variant in ("baseline", "candidate"):
                cells.append(
                    {
                        "applicable": True,
                        "attempt_id": f"{task_index:02x}{attempt:02x}{'0' if variant == 'baseline' else '1'}".ljust(
                            64, "a"
                        ),
                        "harness": "claude-code",
                        "task_id": task_id,
                        "trial_index": attempt,
                        "variant_id": variant,
                    }
                )
    preview = {"preview_digest": "f" * 64, "matrix": {"matrix_cells": cells}}
    first = AUDIT.freeze(preview, fraction=0.1)
    second = AUDIT.freeze(preview, fraction=0.1)
    assert first == second
    assert first["population_pairs"] == 96
    assert len(first["selected_pairs"]) == 10
    assert len(first["selected_attempt_ids"]) == 20
    assert {item["partition"] for item in first["selected_pairs"]} == {
        "development",
        "holdout",
    }
    assert all(
        set(item)
        == {
            "pair_token",
            "task_id",
            "harness",
            "attempt",
            "partition",
            "artifact_a_attempt_id",
            "artifact_b_attempt_id",
        }
        for item in first["selected_pairs"]
    )
    reviewer_selection = json.dumps(first, sort_keys=True)
    assert "baseline" not in reviewer_selection
    assert "candidate" not in reviewer_selection

    families = {
        "early-development": ["sp-dev-task-00"],
        "late-holdout": ["sp-holdout-task-23"],
    }
    stratified = AUDIT.freeze(
        preview,
        fraction=0.01,
        behavior_families=families,
    )
    selected_tasks = {item["task_id"] for item in stratified["selected_pairs"]}
    assert selected_tasks >= {"sp-dev-task-00", "sp-holdout-task-23"}
