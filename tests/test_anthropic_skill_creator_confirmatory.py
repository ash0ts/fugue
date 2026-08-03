from __future__ import annotations

import hashlib
import importlib.util
import json
import runpy
import sys
import tarfile
from pathlib import Path

from fugue.bench.comparison import load_comparison

EXAMPLE = Path("examples/comparisons/anthropic-skill-creator-upgrade")
TASKS = EXAMPLE / "confirmatory-tasks.jsonl"
LABELS = EXAMPLE / "confirmatory-private-labels.jsonl"
SCORER = EXAMPLE / "confirmatory_scorer.py"
PREREG = EXAMPLE / "confirmatory-preregistration.json"
FAMILY_LOCK = EXAMPLE / "confirmatory-task-family-lock.json"
QUALIFICATION = EXAMPLE / "qualification_fixtures.py"
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
    assert spec.execution.evidence_checkpoint_cells == 1
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
