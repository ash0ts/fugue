from __future__ import annotations

import hashlib
import importlib.util
import json
import runpy
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

from fugue.bench.candidates import stable_digest
from fugue.bench.comparison import load_comparison

EXAMPLE = Path("examples/comparisons/anthropic-skill-creator-upgrade")
TASKS = EXAMPLE / "confirmatory-tasks.jsonl"
LABELS = EXAMPLE / "confirmatory-private-labels.jsonl"
SCORER = EXAMPLE / "confirmatory_scorer.py"
PREREG = EXAMPLE / "confirmatory-preregistration.json"
FAMILY_LOCK = EXAMPLE / "confirmatory-task-family-lock.json"
QUALIFICATION = EXAMPLE / "qualification_fixtures.py"
SCORER_VALIDATION_V2 = EXAMPLE / "validate_confirmatory_scorer_v2.py"
SCORER_VALIDATION_RECEIPT_V2 = EXAMPLE / "confirmatory-scorer-validation-v2.json"
PRODUCT_CONTRACT_V2 = EXAMPLE / "compatibility-product-contract-v2.json"
CONFIRMATORY_V1 = EXAMPLE / "confirmatory-v1.yaml"
CONFIRMATORY_V2 = EXAMPLE / "confirmatory-v2.yaml"
CONFIRMATORY_V2_AMENDMENT = EXAMPLE / "confirmatory-v2-amendment.json"
CONFIRMATORY_PREPARER_V1 = EXAMPLE / "prepare_confirmatory.py"
CONFIRMATORY_PREPARER_V2 = EXAMPLE / "prepare_confirmatory_v2.py"
CONFIRMATORY_CONSOLE_V2 = EXAMPLE / "study-console-confirmatory-v2.yaml"
DIMENSIONS = {
    "artifact_validity",
    "frontmatter_semantics",
    "compatibility_selection",
    "name_help_consistency",
    "behavior_preservation",
    "packaging",
    "instruction_quality",
    "dependency_secret_safety",
    "assigned_script_use",
}


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _labels() -> dict[str, dict]:
    return {row["id"]: row for row in _jsonl(LABELS)}


def _score():
    return runpy.run_path(SCORER.as_posix())["score"]


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, path.parent.as_posix())
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(path.parent.as_posix())
    return module


def _evidence(expected: dict) -> dict:
    return {
        "expected": expected,
        "opened_paths": [
            "/opt/skills/skill-creator/SKILL.md",
            "/opt/skills/skill-creator/scripts/init_skill.py",
            "/opt/skills/skill-creator/scripts/quick_validate.py",
        ],
        "tool_calls": [],
    }


def _result(task_id: str, expected: dict, files: dict[str, str]) -> dict:
    return {
        "schema_version": 1,
        "task_id": task_id,
        "disposition": expected["disposition"],
        "skill_name": expected["skill_name"],
        "files": files,
        "findings": expected.get("findings", {}),
        "maintainer_memo": "Completed the bounded request and preserved every declared safety constraint.",
    }


def test_v3_confirmatory_spec_has_frozen_192_cell_design() -> None:
    spec = load_comparison(EXAMPLE / "confirmatory-v1.yaml", repo_root=Path.cwd())
    tasks = _jsonl(TASKS)
    labels = _jsonl(LABELS)

    assert spec.schema_version == 3
    assert spec.id == "anthropic-skill-creator-confirmatory-v1"
    assert spec.baseline.skills == ("anthropic-skill-creator-before-compatibility",)
    assert spec.candidate.skills == ("anthropic-skill-creator-compatibility",)
    assert spec.execution.source_evidence_project == (
        "wandb/fugue-anthropic-skill-creator-source-v1"
    )
    assert spec.execution.source_lock == (
        ".fugue/qualification/community-skill-confirmatory/anthropic/source.lock.json"
    )
    assert spec.execution.evidence_project == (
        "wandb/fugue-anthropic-skill-creator-confirmatory-v1"
    )
    assert spec.execution.attempts == 4
    assert spec.execution.concurrency == 1
    assert (
        spec.execution.scheduling_seed
        == "community-skill-upgrade-confirmatory-campaign-v1"
    )
    assert spec.execution.evidence_checkpoint_cells == 2
    assert spec.execution.max_cost_usd == 1640
    assert len(tasks) == len(labels) == 24
    assert len(tasks) * 2 * spec.execution.attempts == 192
    assert sum(row["partition"] == "discovery" for row in tasks) == 8
    assert sum(row["partition"] == "holdout" for row in tasks) == 16
    assert {row["id"] for row in tasks} == {row["id"] for row in labels}

    deterministic = next(item for item in spec.evaluators if item.type == "deterministic")
    assert set(deterministic.dimensions) == DIMENSIONS
    assert deterministic.dimension_roles["assigned_script_use"] == "mechanism"
    assert deterministic.dimension_roles["artifact_validity"] == "safety_gate"
    judge = next(item for item in spec.evaluators if item.type == "llm_judge")
    assert judge.required is False


def test_v2_is_descriptive_measurement_restart_with_v1_behavioral_inputs() -> None:
    v1_raw = yaml.safe_load(CONFIRMATORY_V1.read_text(encoding="utf-8"))
    v2_raw = yaml.safe_load(CONFIRMATORY_V2.read_text(encoding="utf-8"))
    v2 = load_comparison(CONFIRMATORY_V2, repo_root=Path.cwd())

    for field in ("taskset", "baseline", "candidate", "changed", "evaluators"):
        assert v2_raw[field] == v1_raw[field]
    assert v2.id == "anthropic-skill-creator-confirmatory-v2"
    assert "descriptive measurement development" in v2.question
    assert "not a population or conference claim" in v2.question
    assert v2.execution.source_lock == (
        ".fugue/qualification/community-skill-confirmatory/"
        "anthropic-v2/source.lock.json"
    )
    assert v2.execution.evidence_project == (
        "wandb/fugue-anthropic-skill-creator-confirmatory-v2"
    )
    assert v2.execution.research_id == v2.id
    assert v2.execution.study_console_base_url == "http://127.0.0.1:18105"
    assert v2.execution.attempts == 4
    assert v2.execution.evidence_checkpoint_cells == 2
    assert v2.execution.max_cost_usd == 1640
    assert v2.execution.scheduling_seed == (
        "community-skill-upgrade-measurement-development-v2"
    )
    assert v2.execution.qualification_inputs == {
        "campaign_manifest_sha256": (
            "examples/comparisons/community-skill-upgrades/"
            "conference-campaign-manifest-v2.json"
        ),
        "campaign_preregistration_sha256": (
            "examples/comparisons/community-skill-upgrades/"
            "conference-preregistration-v2.json"
        ),
        "compatibility_product_contract_sha256": (
            "examples/comparisons/anthropic-skill-creator-upgrade/"
            "compatibility-product-contract-v2.json"
        ),
        "confirmatory_analysis_profile_sha256": (
            "examples/comparisons/community-skill-upgrades/"
            "confirmatory-analysis-profiles-v2.json"
        ),
        "confirmatory_analyzer_sha256": (
            "examples/comparisons/community-skill-upgrades/"
            "analyze_confirmatory.py"
        ),
        "confirmatory_budget_policy_sha256": (
            "examples/comparisons/community-skill-upgrades/"
            "confirmatory-budget-policy-v2.json"
        ),
        "scientific_report_generator_sha256": (
            "examples/comparisons/community-skill-upgrades/"
            "generate_scientific_report.py"
        ),
        "scientific_report_template_sha256": (
            "examples/comparisons/community-skill-upgrades/"
            "scientific-report-template-v3.json"
        ),
        "prospective_power_design_sha256": (
            "examples/comparisons/community-skill-upgrades/"
            "prospective-power-design-v1.json"
        ),
        "repository_amendment_sha256": (
            "examples/comparisons/anthropic-skill-creator-upgrade/"
            "confirmatory-v2-amendment.json"
        ),
        "repository_preregistration_sha256": (
            "examples/comparisons/anthropic-skill-creator-upgrade/"
            "confirmatory-preregistration.json"
        ),
        "scorer_mutation_validation_receipt_sha256": (
            "examples/comparisons/anthropic-skill-creator-upgrade/"
            "confirmatory-scorer-validation-v2.json"
        ),
        "scorer_mutation_validator_sha256": (
            "examples/comparisons/anthropic-skill-creator-upgrade/"
            "validate_confirmatory_scorer_v2.py"
        ),
        "host_only_task_family_lock_sha256": (
            "examples/comparisons/anthropic-skill-creator-upgrade/"
            "confirmatory-task-family-lock.json"
        ),
        "qualification_fixture_generator_sha256": (
            "examples/comparisons/anthropic-skill-creator-upgrade/"
            "qualification_fixtures.py"
        ),
        "skill_revision_lock_sha256": (
            "examples/comparisons/anthropic-skill-creator-upgrade/"
            "skill-revisions.lock.json"
        ),
        "upstream_source_preparer_sha256": (
            "examples/comparisons/anthropic-skill-creator-upgrade/"
            "prepare_sources.py"
        ),
        "zero_model_conformance_sha256": (
            "examples/comparisons/anthropic-skill-creator-upgrade/"
            "zero_model_conformance.py"
        ),
        "source_preparer_v1_sha256": (
            "examples/comparisons/anthropic-skill-creator-upgrade/"
            "prepare_confirmatory.py"
        ),
        "source_preparer_v2_sha256": (
            "examples/comparisons/anthropic-skill-creator-upgrade/"
            "prepare_confirmatory_v2.py"
        ),
        "base_preparation_receipt_sha256": (
            ".fugue/comparison-resources/anthropic-skill-creator-upgrade/"
            "confirmatory-preparation.lock.json"
        ),
        "v2_preparation_receipt_sha256": (
            ".fugue/comparison-resources/anthropic-skill-creator-upgrade/"
            "confirmatory-v2-preparation.lock.json"
        ),
    }


def test_v2_amendment_binds_measurement_receipts_and_claim_boundary() -> None:
    amendment = json.loads(CONFIRMATORY_V2_AMENDMENT.read_text(encoding="utf-8"))
    unsigned = dict(amendment)
    amendment_digest = unsigned.pop("amendment_digest")
    receipt = json.loads(
        SCORER_VALIDATION_RECEIPT_V2.read_text(encoding="utf-8")
    )

    assert amendment_digest == stable_digest(unsigned)
    assert amendment["prior_study"]["comparison_spec_sha256"] == _sha256(
        CONFIRMATORY_V1
    )
    assert amendment["prior_study"]["behavioral_result_eligible"] is False
    assert amendment["prior_study"]["run_id"] is None
    revision = amendment["measurement_revision"]
    assert revision["deterministic_scorer_sha256"] == _sha256(SCORER)
    assert revision["mutation_validator_sha256"] == _sha256(
        SCORER_VALIDATION_V2
    )
    assert revision["mutation_validation_receipt_sha256"] == _sha256(
        SCORER_VALIDATION_RECEIPT_V2
    )
    assert revision["mutation_validation_receipt_digest"] == receipt[
        "receipt_digest"
    ]
    assert revision["compatibility_product_contract_sha256"] == _sha256(
        PRODUCT_CONTRACT_V2
    )
    assert amendment["replacement_execution"]["comparison_spec_sha256"] == (
        _sha256(CONFIRMATORY_V2)
    )
    assert amendment["replacement_execution"]["source_preparer_sha256"] == (
        _sha256(CONFIRMATORY_PREPARER_V2)
    )
    assert amendment["claim_scope"]["study_class"] == (
        "measurement_development_descriptive"
    )
    assert amendment["claim_scope"]["conference_claim_eligible"] is False
    assert amendment["claim_scope"]["population_claim_eligible"] is False


def test_v2_preserves_historical_v1_bytes() -> None:
    assert {
        CONFIRMATORY_V1.name: _sha256(CONFIRMATORY_V1),
        PREREG.name: _sha256(PREREG),
        TASKS.name: _sha256(TASKS),
        LABELS.name: _sha256(LABELS),
        SCORER.name: _sha256(SCORER),
        QUALIFICATION.name: _sha256(QUALIFICATION),
        FAMILY_LOCK.name: _sha256(FAMILY_LOCK),
        CONFIRMATORY_PREPARER_V1.name: _sha256(CONFIRMATORY_PREPARER_V1),
    } == {
        "confirmatory-v1.yaml": (
            "63deb6e7c6f3f86236b631f5ff07767b41b66891bf52dd654b5ca389428feb5a"
        ),
        "confirmatory-preregistration.json": (
            "46fb86c8e9d5d4d1500d37a6e2b12eac54a3d7563549e0f77b877f70dd3efc42"
        ),
        "confirmatory-tasks.jsonl": (
            "0cd90f0214108bb0c49aa7b25c279eb980057a9ec1142ab46349ac8b272ec641"
        ),
        "confirmatory-private-labels.jsonl": (
            "2297fe640c51aafa5d7fbd57b90baaab457f56c8f4482b178abc0c0998cfeede"
        ),
        "confirmatory_scorer.py": (
            "c98fe2a020e8d96542a5611966a430d2828257f837bc565573e534b997005977"
        ),
        "qualification_fixtures.py": (
            "953a041ea71175020bfd8465c68a17147f4a382b4601b8041fbeeed85d1056e2"
        ),
        "confirmatory-task-family-lock.json": (
            "0b804d2bc62d8f0435f97ff4ce932d2505315603d8fd85bfcbf10d8d0bfe51d2"
        ),
        "prepare_confirmatory.py": (
            "c6db2e641eccdab69336ddf7c2eaa05eb5b91da99224f69bbaaacc81d2ae7eb0"
        ),
    }


def test_agent_visible_ids_and_prompts_do_not_disclose_boundary_values() -> None:
    tasks = _jsonl(TASKS)
    forbidden_ids = (
        "compatibility-500",
        "compatibility-501",
        "name-40",
        "name-41",
        "name-42",
        "name-63",
        "name-64",
        "name-65",
        "valid-compatibility",
        "invalid-compatibility",
    )
    forbidden_phrases = (
        "compatibility field",
        "compatibility frontmatter",
        "maximum is 64",
        "max 64",
        "500 characters",
        "501 characters",
        "65-character",
    )

    for task in tasks:
        assert not any(value in task["id"] for value in forbidden_ids)
        question = task["input"]["question"].lower()
        assert not any(value in question for value in forbidden_phrases)
        assert "self-reported validation boolean" in question or (
            "do not report a validation boolean" in question
        )

    family_lock = json.loads(FAMILY_LOCK.read_text(encoding="utf-8"))
    assert family_lock["agent_visible"] is False
    assert family_lock["mounted_in_trials"] is False
    assert set(family_lock["families"]) == {row["id"] for row in tasks}


def test_host_scorer_accepts_canonical_created_skill_without_agent_receipt() -> None:
    task_id = "as-dev-create-platform-bound-skill"
    expected = _labels()[task_id]["expected"]
    skill = """---
name: fleet-release-evidence-review
description: Review immutable local release evidence and return a cited bounded decision. Use when a maintainer needs an offline fleet release check.
compatibility: macOS 14+ or Ubuntu 24.04+, Git 2.43+, offline only
---

# Fleet release evidence review

Inspect only local evidence and cite each source path. Verify the locked release data. If evidence is missing or unavailable, stop before making a release claim.
"""
    output = _result(
        task_id,
        expected,
        {"skills/fleet-release-evidence-review/SKILL.md": skill},
    )

    scores = _score()({"id": task_id}, output, _evidence(expected))

    assert set(scores) == DIMENSIONS
    assert all(scores.values()), scores

    forged = dict(output)
    forged["validation"] = {"passed": True}
    forged_scores = _score()({"id": task_id}, forged, _evidence(expected))
    assert forged_scores["artifact_validity"] is False


def test_host_scorer_checks_boundary_noop_and_repair_independently() -> None:
    prepare = _module("anthropic_prepare_confirmatory", EXAMPLE / "prepare_confirmatory.py")
    score = _score()
    labels = _labels()

    noop_id = "as-holdout-metadata-boundary-alpha"
    noop_expected = labels[noop_id]["expected"]
    noop_files = prepare._initial_files(noop_id)
    noop_scores = score(
        {"id": noop_id},
        _result(noop_id, noop_expected, noop_files),
        _evidence(noop_expected),
    )
    assert all(noop_scores.values()), noop_scores

    repair_id = "as-holdout-metadata-boundary-beta"
    repair_expected = labels[repair_id]["expected"]
    source = next(iter(prepare._initial_files(repair_id).values()))
    repaired = source.replace("L" * 501, "Linux with offline local inputs")
    repair_scores = score(
        {"id": repair_id},
        _result(
            repair_id,
            repair_expected,
            {"skills/overbound-metadata-review/SKILL.md": repaired},
        ),
        _evidence(repair_expected),
    )
    assert all(repair_scores.values()), repair_scores

    wrong_type = repaired.replace(
        "compatibility: Linux with offline local inputs",
        "compatibility: [Linux]",
    )
    type_scores = score(
        {"id": repair_id},
        _result(
            repair_id,
            repair_expected,
            {"skills/overbound-metadata-review/SKILL.md": wrong_type},
        ),
        _evidence(repair_expected),
    )
    assert type_scores["frontmatter_semantics"] is False
    assert type_scores["compatibility_selection"] is False


def test_host_scorer_requires_exact_rejection_and_help_repair() -> None:
    score = _score()
    labels = _labels()

    reject_id = "as-holdout-mandated-name-zeta"
    reject_expected = labels[reject_id]["expected"]
    rejected = _result(reject_id, reject_expected, {})
    rejected["maintainer_memo"] = (
        "Rejected the exact 65-character identifier because it exceeds the maximum "
        "64-character package contract; no files were created."
    )
    reject_scores = score(
        {"id": reject_id}, rejected, _evidence(reject_expected)
    )
    assert all(reject_scores.values()), reject_scores

    help_id = "as-dev-init-help-diagnosis"
    help_expected = labels[help_id]["expected"]
    script = (
        "def help_text():\n"
        "    return 'Skill name requirements: lowercase kebab-case; Max 64 characters'\n"
    )
    help_scores = score(
        {"id": help_id},
        _result(help_id, help_expected, {"scripts/init_skill.py": script}),
        _evidence(help_expected),
    )
    assert all(help_scores.values()), help_scores


def test_every_host_only_gold_fixture_passes_and_base_fixture_fails() -> None:
    qualification = _module("anthropic_qualification", QUALIFICATION)
    rows = _jsonl(LABELS)
    score = _score()

    assert qualification.augment(rows) == rows
    for row in rows:
        gold_scores = score(
            {"id": row["id"]},
            row["gold_output"],
            {"expected": row["expected"], **row["gold_evidence"]},
        )
        base_scores = score(
            {"id": row["id"]},
            row["base_output"],
            {"expected": row["expected"], **row["base_evidence"]},
        )
        assert all(gold_scores.values()), (row["id"], gold_scores)
        assert not all(base_scores.values()), (row["id"], base_scores)


def test_v2_mutation_validation_exercises_every_scorer_dimension() -> None:
    validation = _module("anthropic_scorer_validation_v2", SCORER_VALIDATION_V2)

    first = validation.validate()
    second = validation.validate()

    assert first == second
    assert first["status"] == "passed"
    assert first["private_inputs_serialized"] is False
    assert set(first["dimensions"]) == DIMENSIONS
    assert first["receipt_digest"] == validation._stable_digest(
        {key: value for key, value in first.items() if key != "receipt_digest"}
    )
    for dimension, metrics in first["dimensions"].items():
        assert metrics["positive_cases"] == 24, dimension
        assert metrics["positive_failures"] == 0, dimension
        assert metrics["targeted_negative_cases"] >= 8, dimension
        assert metrics["targeted_false_accepts"] == 0, dimension


def test_v2_preparer_recomputes_and_binds_measurement_receipts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepare = _module(
        "anthropic_prepare_confirmatory_v2",
        CONFIRMATORY_PREPARER_V2,
    )
    monkeypatch.setattr(
        prepare,
        "prepare_v1",
        lambda _repo, _output: {"manifest_digest": "a" * 64},
    )
    output = tmp_path / "prepared"
    output.mkdir()

    manifest = prepare.prepare(tmp_path / "anthropic-skills", output)
    written = json.loads(
        (output / "confirmatory-v2-preparation.lock.json").read_text(
            encoding="utf-8"
        )
    )

    assert written == manifest
    assert manifest["study_id"] == "anthropic-skill-creator-confirmatory-v2"
    assert manifest["study_class"] == "measurement_development_descriptive"
    assert manifest["conference_claim_eligible"] is False
    assert manifest["population_claim_eligible"] is False
    assert manifest["private_inputs_serialized"] is False
    assert manifest["compatibility_product_contract"] == {
        "id": "anthropic-skill-creator-compatibility-product-contract-v2",
        "sha256": _sha256(PRODUCT_CONTRACT_V2),
    }
    receipt = json.loads(
        SCORER_VALIDATION_RECEIPT_V2.read_text(encoding="utf-8")
    )
    assert manifest["scorer_mutation_validation"] == {
        "id": "anthropic-skill-creator-confirmatory-scorer-validation-v2",
        "validator_sha256": _sha256(SCORER_VALIDATION_V2),
        "receipt_sha256": _sha256(SCORER_VALIDATION_RECEIPT_V2),
        "receipt_digest": receipt["receipt_digest"],
    }
    assert manifest["preparation_inputs"] == {
        "v1_spec_sha256": _sha256(CONFIRMATORY_V1),
        "v2_spec_sha256": _sha256(CONFIRMATORY_V2),
        "v1_preparer_sha256": _sha256(CONFIRMATORY_PREPARER_V1),
        "v2_preparer_sha256": _sha256(CONFIRMATORY_PREPARER_V2),
        "upstream_source_preparer_sha256": _sha256(EXAMPLE / "prepare_sources.py"),
        "zero_model_conformance_sha256": _sha256(
            EXAMPLE / "zero_model_conformance.py"
        ),
        "qualification_fixture_generator_sha256": _sha256(QUALIFICATION),
        "host_only_task_family_lock_sha256": _sha256(FAMILY_LOCK),
        "skill_revision_lock_sha256": _sha256(EXAMPLE / "skill-revisions.lock.json"),
    }
    assert manifest["manifest_digest"] == prepare._stable_digest(
        {key: value for key, value in manifest.items() if key != "manifest_digest"}
    )


def test_v2_preparer_rejects_a_drifted_mutation_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepare = _module(
        "anthropic_prepare_confirmatory_v2_drift",
        CONFIRMATORY_PREPARER_V2,
    )
    drifted = json.loads(
        SCORER_VALIDATION_RECEIPT_V2.read_text(encoding="utf-8")
    )
    drifted["status"] = "failed"
    path = tmp_path / "drifted-receipt.json"
    path.write_text(json.dumps(drifted), encoding="utf-8")
    monkeypatch.setattr(prepare, "VALIDATION_RECEIPT", path)

    with pytest.raises(RuntimeError, match="disagrees with fresh validation"):
        prepare._verified_validation_receipt()


def test_v2_preparer_rejects_embedded_artifact_hash_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepare = _module(
        "anthropic_prepare_confirmatory_v2_amendment_drift",
        CONFIRMATORY_PREPARER_V2,
    )
    drifted_spec = tmp_path / "confirmatory-v2.yaml"
    drifted_spec.write_bytes(CONFIRMATORY_V2.read_bytes() + b"\n# drift\n")
    monkeypatch.setattr(prepare, "SPEC", drifted_spec)

    with pytest.raises(RuntimeError, match="replacement_spec"):
        prepare._verified_amendment()


def test_v2_amendment_is_verified_before_base_preparation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepare = _module(
        "anthropic_prepare_confirmatory_v2_order",
        CONFIRMATORY_PREPARER_V2,
    )
    base_called = False

    def _base(_repo: Path, _output: Path) -> dict[str, str]:
        nonlocal base_called
        base_called = True
        return {"manifest_digest": "a" * 64}

    monkeypatch.setattr(prepare, "prepare_v1", _base)
    monkeypatch.setattr(prepare, "_verified_validation_receipt", lambda: {})
    monkeypatch.setattr(prepare, "_verified_product_contract", lambda: {})
    monkeypatch.setattr(
        prepare,
        "_verified_amendment",
        lambda: (_ for _ in ()).throw(RuntimeError("amendment drift")),
    )

    with pytest.raises(RuntimeError, match="amendment drift"):
        prepare.prepare(tmp_path / "upstream", tmp_path / "output")
    assert base_called is False


def test_v2_product_contract_separates_parser_acceptance_from_task_success() -> None:
    contract = json.loads(PRODUCT_CONTRACT_V2.read_text(encoding="utf-8"))

    assert contract["status"] == "frozen_before_expanded_holdout"
    observation = contract["upstream_conformance_observation"]
    assert observation["role"] == "infrastructure_behavior_not_task_correctness"
    assert observation["candidate"]["accepted_scalar_length_range"] == [0, 500]
    outcomes = contract["task_outcome_policy"]
    assert outcomes["required"]["must_cover_every_host_locked_requirement_group"]
    assert outcomes["absent"]["field_present"] is False
    assert (
        outcomes["boundary_cases"]["empty_and_whitespace_only"]
        == "upstream_acceptance_observation_only_never_a_task_pass"
    )
    assert contract["evidence_roles"]["assigned_script_use"] == "mechanism"


def test_v2_study_console_keeps_descriptive_identity_visible() -> None:
    console = yaml.safe_load(CONFIRMATORY_CONSOLE_V2.read_text(encoding="utf-8"))

    assert console["research"]["id"] == "anthropic-skill-creator-confirmatory-v2"
    assert "Descriptive" in console["research"]["description"]
    assert "without a population or conference-qualified claim" in (
        console["research"]["objective"]
    )
    assert console["wandb"] == {
        "entity": "wandb",
        "project": "fugue-anthropic-skill-creator-confirmatory-v2",
    }
    assert console["presentation"] == {
        "default_study_id": "anthropic-skill-creator-confirmatory-v2",
        "read_only": True,
    }


def test_confirmatory_archives_are_deterministic_and_contain_no_validator(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.syspath_prepend(EXAMPLE.as_posix())
    prepare = _module("anthropic_prepare_archives", EXAMPLE / "prepare_confirmatory.py")
    task_id = "as-holdout-runtime-metadata-repair"
    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"

    first_record = prepare._archive(first, prepare._initial_files(task_id))
    second_record = prepare._archive(second, prepare._initial_files(task_id))

    assert first.read_bytes() == second.read_bytes()
    assert first_record["sha256"] == second_record["sha256"]
    with tarfile.open(first) as archive:
        paths = archive.getnames()
    assert not any("validate" in path.lower() for path in paths)
    assert paths == sorted(paths)


def test_confirmatory_archive_map_is_exhaustive_and_unknown_ids_fail() -> None:
    prepare = _module("anthropic_prepare_archive_map", EXAMPLE / "prepare_confirmatory.py")
    task_ids = tuple(row["id"] for row in _jsonl(TASKS))

    assert task_ids == prepare._CONFIRMATORY_TASK_IDS
    for task_id in task_ids:
        assert isinstance(prepare._initial_files(task_id), dict)
    with pytest.raises(RuntimeError, match="unknown confirmatory task"):
        prepare._initial_files("as-holdout-unreviewed-new-task")


@pytest.mark.parametrize(
    "relative",
    (
        "../confirmatory-private-labels.jsonl",
        "/tmp/answer-key.json",
        "nested\\oracle.json",
        "README.md",
        "nested/validator.py",
    ),
)
def test_confirmatory_archive_rejects_unsafe_or_private_paths(
    tmp_path: Path,
    relative: str,
) -> None:
    prepare = _module(
        "anthropic_prepare_unsafe_archive", EXAMPLE / "prepare_confirmatory.py"
    )

    with pytest.raises(RuntimeError, match="archive path|evaluator path"):
        prepare._archive(tmp_path / "unsafe.tar", {relative: "private"})


def test_confirmatory_archive_receipts_are_checkout_path_portable(
    tmp_path: Path,
) -> None:
    prepare = _module(
        "anthropic_prepare_portable_archive", EXAMPLE / "prepare_confirmatory.py"
    )
    first_root = tmp_path / "clone-one" / "prepared"
    second_root = tmp_path / "clone-two" / "prepared"
    first_path = first_root / "confirmatory" / "task.tar"
    second_path = second_root / "confirmatory" / "task.tar"
    first_path.parent.mkdir(parents=True)
    second_path.parent.mkdir(parents=True)
    files = prepare._initial_files("as-holdout-runtime-metadata-repair")

    first = prepare._archive(first_path, files, receipt_root=first_root)
    second = prepare._archive(second_path, files, receipt_root=second_root)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first == second
    assert first["archive"] == "confirmatory/task.tar"


def test_confirmatory_archives_do_not_contain_changed_private_gold_bodies(
    tmp_path: Path,
) -> None:
    prepare = _module(
        "anthropic_prepare_private_boundary", EXAMPLE / "prepare_confirmatory.py"
    )
    labels = _labels()
    for task_id in prepare._CONFIRMATORY_TASK_IDS:
        path = tmp_path / f"{task_id}.tar"
        prepare._archive(path, prepare._initial_files(task_id))
        with tarfile.open(path) as archive:
            members = archive.getmembers()
            assert all(member.isfile() for member in members)
            assert all(
                member.name.startswith("workspace/")
                and not Path(member.name).is_absolute()
                and ".." not in Path(member.name).parts
                for member in members
            )
            payload = b"\n".join(
                archive.extractfile(member).read()  # type: ignore[union-attr]
                for member in members
            )
        label = labels[task_id]
        base_files = label["base_output"].get("files", {})
        for relative, gold_body in label["gold_output"].get("files", {}).items():
            if gold_body != base_files.get(relative):
                assert gold_body.encode() not in payload


def test_source_lock_recipe_binds_preparation_and_conformance_code() -> None:
    readme = (EXAMPLE / "README.md").read_text(encoding="utf-8")
    spec = load_comparison(CONFIRMATORY_V2, repo_root=Path.cwd())
    required = {
        "base_preparation_receipt_sha256",
        "compatibility_product_contract_sha256",
        "host_only_task_family_lock_sha256",
        "qualification_fixture_generator_sha256",
        "scorer_mutation_validation_receipt_sha256",
        "scorer_mutation_validator_sha256",
        "skill_revision_lock_sha256",
        "source_preparer_v1_sha256",
        "source_preparer_v2_sha256",
        "upstream_source_preparer_sha256",
        "v2_preparation_receipt_sha256",
        "zero_model_conformance_sha256",
    }
    assert required <= set(spec.execution.qualification_inputs)
    assert "--extra" not in readme
    assert "no README-only supplemental file list is" in readme


def test_zero_model_matrix_and_preregistered_digests_are_frozen() -> None:
    conformance = _module(
        "anthropic_zero_model", EXAMPLE / "zero_model_conformance.py"
    )
    expected = conformance._expected()
    assert expected["baseline"]["frontmatter"] == {
        "absent": True,
        "empty": False,
        "one_character": False,
        "length_500": False,
        "length_501": False,
        "non_string": False,
        "unknown_key": False,
    }
    assert expected["candidate"]["frontmatter"] == {
        "absent": True,
        "empty": True,
        "one_character": True,
        "length_500": True,
        "length_501": False,
        "non_string": False,
        "unknown_key": False,
    }
    assert expected["baseline"]["help_max_characters"] == 40
    assert expected["candidate"]["help_max_characters"] == 64
    assert expected["candidate"]["names"] == {
        "40": True,
        "41": True,
        "64": True,
        "65": False,
    }

    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    locked = prereg["locked_inputs"]
    assert hashlib.sha256(TASKS.read_bytes()).hexdigest() == locked[
        "public_tasks_sha256"
    ]
    assert hashlib.sha256(LABELS.read_bytes()).hexdigest() == locked[
        "private_labels_sha256"
    ]
    assert hashlib.sha256(SCORER.read_bytes()).hexdigest() == locked["scorer_sha256"]
    assert hashlib.sha256(QUALIFICATION.read_bytes()).hexdigest() == locked[
        "qualification_fixture_generator_sha256"
    ]
    assert hashlib.sha256(FAMILY_LOCK.read_bytes()).hexdigest() == locked[
        "host_only_task_family_lock_sha256"
    ]
