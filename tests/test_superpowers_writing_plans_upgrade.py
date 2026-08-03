from __future__ import annotations

import hashlib
import json
import runpy
import tarfile
from pathlib import Path

import pytest
import yaml

from fugue.bench.candidates import stable_digest
from fugue.bench.comparison import load_comparison

EXAMPLE = Path("examples/comparisons/superpowers-writing-plans-upgrade")
SCORER = EXAMPLE / "skill_upgrade_scorer.py"
PRIVATE = EXAMPLE / "private-labels.jsonl"
LOCK = EXAMPLE / "skill-revisions.lock.json"
DIMENSIONS = {
    "plan_structure",
    "constraint_fidelity",
    "repository_grounding",
    "interface_contracts",
    "reviewable_decomposition",
    "verification_quality",
    "scope_and_secret_safety",
    "historical_diff_coverage",
}
V3_SCORER = EXAMPLE / "plan_quality_scorer.py"
V3_PRIVATE = EXAMPLE / "private-labels-v3.jsonl"
CONFIRMATORY = EXAMPLE / "confirmatory-v1.yaml"
CONFIRMATORY_V2 = EXAMPLE / "confirmatory-v2.yaml"
CONFIRMATORY_V3 = EXAMPLE / "confirmatory-v3.yaml"
CONFIRMATORY_V4 = EXAMPLE / "confirmatory-v4.yaml"
CONFIRMATORY_V5 = EXAMPLE / "confirmatory-v5.yaml"
CONFIRMATORY_TASKS = EXAMPLE / "tasks-conference-v1.jsonl"
CONFIRMATORY_PRIVATE = EXAMPLE / "private-labels-conference-v1.jsonl"
CONFIRMATORY_SCORER = EXAMPLE / "plan_quality_scorer_v2.py"
CONFIRMATORY_SCORER_V3 = EXAMPLE / "plan_quality_scorer_v3.py"
CONFIRMATORY_PREREGISTRATION = EXAMPLE / "preregistration-confirmatory-v1.json"
CONFIRMATORY_V2_AMENDMENT = (
    EXAMPLE / "preregistration-confirmatory-v2-amendment.json"
)
CONFIRMATORY_V2_CONSOLE = EXAMPLE / "study-console-confirmatory-v2.yaml"
CONFIRMATORY_V3_AMENDMENT = (
    EXAMPLE / "preregistration-confirmatory-v3-amendment.json"
)
CONFIRMATORY_V3_CONSOLE = EXAMPLE / "study-console-confirmatory-v3.yaml"
CONFIRMATORY_V4_AMENDMENT = EXAMPLE / "preregistration-confirmatory-v4-amendment.json"
CONFIRMATORY_V4_CONSOLE = EXAMPLE / "study-console-confirmatory-v4.yaml"
CONFIRMATORY_V5_AMENDMENT = EXAMPLE / "preregistration-confirmatory-v5-amendment.json"
CONFIRMATORY_V5_CONSOLE = EXAMPLE / "study-console-confirmatory-v5.yaml"
CONFIRMATORY_PREPARER = EXAMPLE / "prepare_confirmatory_sources.py"
CONFIRMATORY_DIMENSIONS = {
    "artifact_validity",
    "global_constraint_fidelity",
    "interface_graph_consistency",
    "right_sized_decomposition",
    "repository_grounding",
    "verification_matrix",
    "scope_secret_safety",
}
V3_DIMENSIONS = {
    "artifact_validity",
    "requirement_coverage",
    "repository_grounding",
    "dependency_contracts",
    "reviewable_decomposition",
    "verification_quality",
    "scope_and_secret_safety",
    "scenario_coverage",
}


def _score():
    return runpy.run_path(SCORER.as_posix())["score"]


def _labels() -> dict[str, dict]:
    return {
        item["id"]: item
        for item in (
            json.loads(line)
            for line in PRIVATE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _v3_score():
    return runpy.run_path(V3_SCORER.as_posix())["score"]


def _v3_labels() -> dict[str, dict]:
    return {
        item["id"]: item
        for item in (
            json.loads(line)
            for line in V3_PRIVATE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _confirmatory_score():
    return runpy.run_path(CONFIRMATORY_SCORER.as_posix())["score"]


def _confirmatory_v3_score():
    return runpy.run_path(CONFIRMATORY_SCORER_V3.as_posix())["score"]


def _confirmatory_v3_scorer_module() -> dict:
    return runpy.run_path(CONFIRMATORY_SCORER_V3.as_posix())


def _confirmatory_labels() -> dict[str, dict]:
    return {
        item["id"]: item
        for item in (
            json.loads(line)
            for line in CONFIRMATORY_PRIVATE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _behavioral_comparison_contract(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw.pop("id")
    execution = raw["execution"]
    for field in (
        "source_lock",
        "evidence_project",
        "evidence_destination",
        "research_id",
        "study_console_base_url",
    ):
        execution.pop(field)
    return raw


def _render_oracle_plan(expected: dict) -> str:
    lines = [
        "# Repository-grounded implementation plan",
        "",
        "This plan preserves the inspected repository contract and confines every change to the named paths.",
        "The verification below names the observable behavior for success and failure before implementation begins.",
        "",
        "## Constraints",
    ]
    for constraint in expected["global_constraints"]:
        for group in constraint["groups"]:
            lines.append(f"- Preserve {group[0]} across the whole change.")
    contracts = expected["unit_contracts"]
    for index, contract in enumerate(contracts, start=1):
        path = contract["anchor_paths"][0]
        lines.extend(
            [
                "",
                f"## Task {index}: Cohesive repository change",
                "",
                f"Modify `{path}` and keep the work inside this reviewed responsibility.",
            ]
        )
        for group in contract["responsibilities"]:
            lines.append(f"- Implement and verify {group[0]}.")
        lines.append(
            "Run `uv run pytest` and verify the cohesive change passes before continuing."
        )
    first_task = next(
        index for index, line in enumerate(lines) if line.startswith("## Task 1:")
    )
    insertion = first_task + 2
    details = []
    bound_paths = {binding["path"] for binding in expected["repository_bindings"]}
    for binding in expected["repository_bindings"]:
        symbols = ", ".join(f"`{value}`" for value in binding["symbols"])
        details.append(
            f"Modify `{binding['path']}` while preserving the inspected symbols {symbols}."
        )
    for group in expected["anchor_groups"]:
        path = group[0]
        if path not in bound_paths:
            details.append(
                f"Modify `{path}` to encode the focused regression coverage for this change."
            )
    for edge in expected.get("interface_edges", []):
        details.append(
            f"{edge['producer'][0]} produces {edge['artifact'][0]}; "
            f"{edge['consumer'][0]} consumes that {edge['artifact'][0]}."
        )
    for row in expected["verification_rows"]:
        details.append(
            f"Run `{row['command'][0]}` for {row['scenario'][0]} and assert {row['assertion'][0]}."
        )
    lines[insertion:insertion] = details
    return "\n".join(lines)


def test_locked_canary_uses_exact_skill_upgrade_and_eight_cells() -> None:
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=Path.cwd())
    lock = json.loads(LOCK.read_text(encoding="utf-8"))

    assert spec.schema_version == 2
    assert spec.baseline.skills == (lock["baseline"]["id"],)
    assert spec.candidate.skills == (lock["candidate"]["id"],)
    assert lock["repository"] == "https://github.com/obra/superpowers"
    assert lock["path"] == "skills/writing-plans"
    assert lock["baseline"]["commit"] == ("de4672b171213a6ff6960228d8b95c46ea0b09f4")
    assert lock["candidate"]["commit"] == ("8e1262a3bae92b640d87fa81c51c53b65e490590")
    assert spec.execution.evidence_project == (
        "wandb/fugue-skill-upgrade-qualification-v1"
    )
    assert spec.execution.harnesses == ("claude-code",)
    assert spec.execution.attempts == 2


def test_private_qualification_fixtures_keep_base_fail_gold_pass() -> None:
    score = _score()
    for task_id, label in _labels().items():
        base = score(
            {"id": task_id},
            label["base_output"],
            {"expected": label["expected"]},
        )
        gold = score(
            {"id": task_id},
            label["gold_output"],
            {"expected": label["expected"]},
        )

        assert set(gold) == DIMENSIONS
        assert all(gold.values())
        assert not all(base.values())


def test_constraint_drift_and_microtask_decomposition_fail_independently() -> None:
    label = _labels()["credential-rotation-plan"]
    score = _score()

    drifted = label["gold_output"].replace(
        "A changed credential must atomically replace the container secret.",
        "A changed credential may overwrite the container secret in place.",
    )
    drifted_scores = score(
        {"id": label["id"]},
        drifted,
        {"expected": label["expected"]},
    )

    microtask = (
        label["gold_output"]
        + """

### Task 2: Documentation

**Files:**
- Create: None
- Modify: `fugue/research/bootstrap.py`
- Test: `tests/test_research_agent_interface.py`

**Interfaces:**
- Consumes: Task 1 `_sync_secret(path: Path, value: str) -> None`.
- Produces: Task 1 documentation note.

- [ ] Write a failing test. Run: `uv run pytest tests/test_research_agent_interface.py`. Expected: FAIL.
- [ ] Update documentation and verify it passes. Expected: PASS.
- [ ] Commit.
"""
    )
    microtask_scores = score(
        {"id": label["id"]},
        microtask,
        {"expected": label["expected"]},
    )

    assert drifted_scores["plan_structure"] is True
    assert drifted_scores["constraint_fidelity"] is False
    assert microtask_scores["plan_structure"] is True
    assert microtask_scores["reviewable_decomposition"] is False


def test_v3_canary_uses_complete_source_and_four_locked_cells() -> None:
    spec = load_comparison(EXAMPLE / "comparison-v3.yaml", repo_root=Path.cwd())

    assert spec.schema_version == 2
    assert spec.baseline.skills == ("superpowers-writing-plans-before-contracts",)
    assert spec.candidate.skills == ("superpowers-writing-plans-contracts",)
    assert spec.execution.evidence_project == (
        "wandb/fugue-skill-upgrade-qualification-v1"
    )
    assert spec.execution.attempts == 1
    assert spec.execution.concurrency == 1
    assert spec.execution.max_cost_usd == 20
    assert spec.execution.reserve_per_attempt_usd == 5

    tasks = [
        json.loads(line)
        for line in (EXAMPLE / "tasks-v3.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(tasks) == 2
    assert all("-full-" in item["resources"][0]["path"] for item in tasks)


def test_v5_routes_to_dedicated_project_and_adds_advisory_judge() -> None:
    spec = load_comparison(EXAMPLE / "comparison-v5.yaml", repo_root=Path.cwd())

    assert spec.id == "superpowers-writing-plans-upgrade-canary-v5"
    assert spec.execution.evidence_project == (
        "wandb/fugue-superpowers-writing-plans-upgrade-v1"
    )
    assert spec.execution.study_console_base_url == "http://127.0.0.1:18084"
    assert spec.execution.attempts == 1
    assert spec.execution.max_cost_usd == 34
    assert spec.execution.reserve_per_attempt_usd == 8.4
    deterministic, judge = spec.evaluators
    assert deterministic.dimension_roles["dependency_contracts"] == "outcome"
    assert judge.id == "community-usefulness"
    assert judge.type == "llm_judge"
    assert judge.profile == "anthropic/claude-sonnet-5"
    assert judge.required is False
    assert judge.reserve_cost_usd == 0.1


def test_v3_private_fixtures_keep_base_fail_gold_pass() -> None:
    score = _v3_score()
    for task_id, label in _v3_labels().items():
        base = score(
            {"id": task_id},
            label["base_output"],
            {"expected": label["expected"]},
        )
        gold = score(
            {"id": task_id},
            label["gold_output"],
            {"expected": label["expected"]},
        )

        assert set(gold) == V3_DIMENSIONS
        assert all(gold.values()), (task_id, gold)
        assert not all(base.values())


def test_v3_parser_accepts_task_or_milestone_headings() -> None:
    score = _v3_score()
    label = _v3_labels()["credential-rotation-plan-v3"]
    milestone = label["gold_output"].replace("## Task", "### Milestone")

    scores = score(
        {"id": label["id"]},
        milestone,
        {"expected": label["expected"]},
    )

    assert scores["artifact_validity"] is True
    assert scores["reviewable_decomposition"] is True


def test_v3_path_safety_ignores_non_goal_references_but_rejects_modifications() -> None:
    score = _v3_score()
    label = _v3_labels()["credential-rotation-plan-v3"]
    reference_only = (
        label["gold_output"]
        + "\n\nDo not modify `compose.research.yaml`; it is outside this change."
    )
    modifies_outside = (
        label["gold_output"] + "\n\n## Task 4: Change deployment\n\n"
        "Modify `compose.research.yaml` and run a test to verify deployment."
    )

    reference_scores = score(
        {"id": label["id"]},
        reference_only,
        {"expected": label["expected"]},
    )
    unsafe_scores = score(
        {"id": label["id"]},
        modifies_outside,
        {"expected": label["expected"]},
    )

    assert reference_scores["scope_and_secret_safety"] is True
    assert unsafe_scores["scope_and_secret_safety"] is False


def test_confirmatory_design_locks_twenty_four_tasks_and_192_cells() -> None:
    spec = load_comparison(CONFIRMATORY, repo_root=Path.cwd())
    tasks = [
        json.loads(line)
        for line in CONFIRMATORY_TASKS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels = _confirmatory_labels()
    preregistration = json.loads(
        CONFIRMATORY_PREREGISTRATION.read_text(encoding="utf-8")
    )

    assert spec.schema_version == 3
    assert spec.id == "superpowers-writing-plans-confirmatory-v1"
    assert spec.baseline.skills == ("superpowers-writing-plans-before-contracts",)
    assert spec.candidate.skills == ("superpowers-writing-plans-contracts",)
    assert spec.execution.source_evidence_project == (
        "wandb/fugue-superpowers-writing-plans-source-v1"
    )
    assert spec.execution.source_lock == (
        ".fugue/qualification/community-skill-confirmatory/superpowers/source.lock.json"
    )
    assert spec.execution.evidence_project == (
        "wandb/fugue-superpowers-writing-plans-confirmatory-v1"
    )
    assert spec.execution.research_id == "superpowers-writing-plans-confirmatory-v1"
    assert spec.execution.attempts == 4
    assert spec.execution.concurrency == 1
    assert spec.execution.max_cost_usd == 1700
    assert spec.execution.reserve_per_attempt_usd == 8.4
    assert len(tasks) == len(labels) == 24
    assert sum(task["partition"] == "qualification" for task in tasks) == 8
    assert sum(task["partition"] == "holdout" for task in tasks) == 16
    assert len(tasks) * 2 * spec.execution.attempts == 192
    assert {task["id"] for task in tasks} == set(labels)
    assert preregistration["design"]["planned_cells"] == 192
    assert set(preregistration["design"]["development_task_ids"]) == {
        task["id"] for task in tasks if task["partition"] == "qualification"
    }
    assert set(preregistration["design"]["holdout_task_ids"]) == {
        task["id"] for task in tasks if task["partition"] == "holdout"
    }


def test_confirmatory_v2_changes_only_execution_identity_after_cancelled_v1() -> None:
    v1 = load_comparison(CONFIRMATORY, repo_root=Path.cwd())
    v2 = load_comparison(CONFIRMATORY_V2, repo_root=Path.cwd())
    tasks = [
        json.loads(line)
        for line in CONFIRMATORY_TASKS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert v2.id == "superpowers-writing-plans-confirmatory-v2"
    assert v2.execution.research_id == "superpowers-writing-plans-confirmatory-v2"
    assert v2.execution.evidence_project == (
        "wandb/fugue-superpowers-writing-plans-confirmatory-v2"
    )
    assert v2.execution.source_evidence_project == (
        "wandb/fugue-superpowers-writing-plans-source-v1"
    )
    assert v2.execution.source_evidence_destination == (
        v1.execution.source_evidence_destination
    )
    assert v2.execution.source_lock == (
        ".fugue/qualification/community-skill-confirmatory/"
        "superpowers-v2/source.lock.json"
    )
    assert v2.execution.study_console_base_url == "http://127.0.0.1:18089"
    assert _behavioral_comparison_contract(CONFIRMATORY_V2) == (
        _behavioral_comparison_contract(CONFIRMATORY)
    )
    assert len(tasks) * 2 * v2.execution.attempts == 192


def test_confirmatory_v2_amendment_and_console_preserve_v1_audit_lineage() -> None:
    amendment = json.loads(CONFIRMATORY_V2_AMENDMENT.read_text(encoding="utf-8"))
    unsigned = dict(amendment)
    amendment_digest = unsigned.pop("amendment_digest")
    frozen = amendment["frozen_behavioral_inputs"]
    changes = amendment["changes"]
    console = yaml.safe_load(CONFIRMATORY_V2_CONSOLE.read_text(encoding="utf-8"))

    assert amendment_digest == stable_digest(unsigned)
    assert amendment["prior_preregistration"] == {
        "id": "superpowers-writing-plans-confirmatory-v1",
        "path": (
            "examples/comparisons/superpowers-writing-plans-upgrade/"
            "preregistration-confirmatory-v1.json"
        ),
        "sha256": _sha256(CONFIRMATORY_PREREGISTRATION),
    }
    assert amendment["superseded_execution"] == {
        "comparison_id": "superpowers-writing-plans-confirmatory-v1",
        "run_id": "20260803T145545-4fdfb32be1",
        "status": "cancelled",
        "reason_code": "pty_broken_pipe_infrastructure",
        "behavioral_result_eligible": False,
    }
    assert frozen["public_tasks_sha256"] == _sha256(CONFIRMATORY_TASKS)
    assert frozen["private_labels_sha256"] == _sha256(CONFIRMATORY_PRIVATE)
    assert frozen["deterministic_scorer_sha256"] == _sha256(CONFIRMATORY_SCORER)
    assert frozen["tasks"] * frozen["arms"] * frozen["attempts_per_task_arm"] == (
        frozen["planned_cells"]
    ) == 192
    assert all(
        changes[field] is False
        for field in (
            "hypotheses",
            "taskset",
            "holdout_membership",
            "private_expected_values",
            "treatments",
            "evaluators",
            "model_or_harness",
        )
    )
    assert changes["execution_identity"] is True
    assert console["research"]["id"] == "superpowers-writing-plans-confirmatory-v2"
    assert console["wandb"] == {
        "entity": "wandb",
        "project": "fugue-superpowers-writing-plans-confirmatory-v2",
    }
    assert console["presentation"] == {
        "default_study_id": "superpowers-writing-plans-confirmatory-v2",
        "read_only": True,
    }


def test_confirmatory_v3_freezes_integrity_rerun_without_behavioral_drift() -> None:
    v1_raw = yaml.safe_load(CONFIRMATORY.read_text(encoding="utf-8"))
    v3_raw = yaml.safe_load(CONFIRMATORY_V3.read_text(encoding="utf-8"))
    v3 = load_comparison(CONFIRMATORY_V3, repo_root=Path.cwd())
    amendment = json.loads(CONFIRMATORY_V3_AMENDMENT.read_text(encoding="utf-8"))
    unsigned = dict(amendment)
    amendment_digest = unsigned.pop("amendment_digest")
    console = yaml.safe_load(CONFIRMATORY_V3_CONSOLE.read_text(encoding="utf-8"))

    for field in ("question", "taskset", "baseline", "candidate", "changed", "evaluators"):
        assert v3_raw[field] == v1_raw[field]
    assert v3.id == "superpowers-writing-plans-confirmatory-v3"
    assert v3.execution.research_id == v3.id
    assert v3.execution.evidence_project == (
        "wandb/fugue-superpowers-writing-plans-confirmatory-v3"
    )
    assert v3.execution.evidence_checkpoint_cells == 2
    assert v3.execution.scheduling_seed == (
        "community-skill-upgrade-confirmatory-campaign-v1"
    )
    assert set(v3.execution.qualification_inputs) == {
        "confirmatory_analysis_profile_sha256",
        "campaign_preregistration_sha256",
        "campaign_manifest_sha256",
        "repository_preregistration_sha256",
        "repository_amendment_sha256",
    }

    assert amendment_digest == stable_digest(unsigned)
    assert amendment["replacement_execution"]["comparison_id"] == v3.id
    assert amendment["replacement_execution"]["minimum_fugue_revision"] == (
        "0a7bc4ea00acc735f26eb298b2d09602129a5e65"
    )
    assert [item["comparison_id"] for item in amendment["superseded_executions"]] == [
        "superpowers-writing-plans-confirmatory-v1",
        "superpowers-writing-plans-confirmatory-v2",
    ]
    assert all(
        item["behavioral_result_eligible"] is False
        for item in amendment["superseded_executions"]
    )
    assert all(
        amendment["changes"][field] is False
        for field in (
            "hypotheses",
            "taskset",
            "holdout_membership",
            "private_expected_values",
            "treatments",
            "evaluators",
            "model_or_harness",
        )
    )
    assert console["research"]["id"] == v3.id
    assert console["wandb"] == {
        "entity": "wandb",
        "project": "fugue-superpowers-writing-plans-confirmatory-v3",
    }
    assert console["presentation"] == {
        "default_study_id": v3.id,
        "read_only": True,
    }


def test_confirmatory_v4_freezes_full_restart_without_behavioral_drift() -> None:
    v1_raw = yaml.safe_load(CONFIRMATORY.read_text(encoding="utf-8"))
    v4_raw = yaml.safe_load(CONFIRMATORY_V4.read_text(encoding="utf-8"))
    v4 = load_comparison(CONFIRMATORY_V4, repo_root=Path.cwd())
    amendment = json.loads(CONFIRMATORY_V4_AMENDMENT.read_text(encoding="utf-8"))
    unsigned = dict(amendment)
    amendment_digest = unsigned.pop("amendment_digest")
    console = yaml.safe_load(CONFIRMATORY_V4_CONSOLE.read_text(encoding="utf-8"))

    for field in (
        "question",
        "taskset",
        "baseline",
        "candidate",
        "changed",
        "evaluators",
    ):
        assert v4_raw[field] == v1_raw[field]
    assert v4.id == "superpowers-writing-plans-confirmatory-v4"
    assert v4.execution.research_id == v4.id
    assert v4.execution.evidence_project == (
        "wandb/fugue-superpowers-writing-plans-confirmatory-v4"
    )
    assert v4.execution.source_lock == (
        ".fugue/qualification/community-skill-confirmatory/"
        "superpowers-v4/source.lock.json"
    )
    assert v4.execution.study_console_base_url == "http://127.0.0.1:18102"
    assert v4.execution.attempts == 4
    assert v4.execution.max_cost_usd == 1700
    assert v4.execution.reserve_per_attempt_usd == 8.4
    assert v4.execution.scheduling_seed == (
        "community-skill-upgrade-confirmatory-campaign-v1"
    )
    assert set(v4.execution.qualification_inputs) == {
        "confirmatory_analysis_profile_sha256",
        "confirmatory_analyzer_sha256",
        "campaign_preregistration_sha256",
        "campaign_manifest_sha256",
        "repository_preregistration_sha256",
        "repository_amendment_sha256",
        "source_preparer_sha256",
    }
    assert v4.execution.qualification_inputs["confirmatory_analyzer_sha256"] == (
        "examples/comparisons/community-skill-upgrades/analyze_confirmatory.py"
    )
    assert v4.execution.qualification_inputs["source_preparer_sha256"] == (
        "examples/comparisons/superpowers-writing-plans-upgrade/"
        "prepare_confirmatory_sources.py"
    )

    assert amendment_digest == stable_digest(unsigned)
    assert amendment["replacement_execution"]["comparison_id"] == v4.id
    assert amendment["replacement_execution"]["result_project"] == (
        "wandb/fugue-superpowers-writing-plans-confirmatory-v4"
    )
    assert [item["comparison_id"] for item in amendment["superseded_executions"]] == [
        "superpowers-writing-plans-confirmatory-v1",
        "superpowers-writing-plans-confirmatory-v2",
        "superpowers-writing-plans-confirmatory-v3",
    ]
    v3 = amendment["superseded_executions"][-1]
    assert v3["preview_digest"] == (
        "b65ce450152a111c45a0aa6566372f8e5b588556dc72b03a310d271e3a6936f0"
    )
    assert v3["started_cells"] == v3["terminal_evidence_rows"] == 7
    assert v3["internally_cancelled_cells"] == 185
    assert v3["canonical_result_published"] is False
    assert v3["cleanup_zero_orphans_verified"] is True
    exposed = amendment["v3_exposed_coordinates"]
    assert len(exposed) == 7
    assert len({item["attempt_id"] for item in exposed}) == 7
    assert len({item["cell_id"] for item in exposed}) == 7
    assert all(len(item["attempt_id"]) == 64 for item in exposed)
    assert amendment["restart_decision"] == {
        "basis": "execution_semantics_defect",
        "outcome_dependent": False,
        "made_before_any_v3_behavioral_result_or_hypothesis_test": True,
        "operator_exposure_disclosed": True,
        "restart_scope": "all_192_cells_from_scratch",
        "resume_or_selective_rerun": False,
    }
    controls = amendment["optional_stopping_controls"]
    assert controls["v3_rows_in_v4_analysis"] is False
    assert controls["v3_rows_pooled_with_v4"] is False
    assert controls["interim_efficacy_stopping"] is False
    assert controls["outcome_dependent_retry"] is False
    assert controls["full_v4_terminal_cohort_required"] is True
    assert controls["exposed_holdout_sensitivity"]["excluded_task_ids"] == [
        "sp-holdout-research-event-projection"
    ]
    privacy = amendment["v4_source_privacy_correction"]
    assert privacy["candidate_neutral"] is True
    assert privacy["behavioral_hypotheses_changed"] is False
    assert privacy["taskset_or_private_truth_changed"] is False
    assert privacy["skill_revisions_changed"] is False
    assert privacy["scorer_changed"] is False
    assert privacy["repository_file_count"] == 778
    assert privacy["agent_visible_file_count"] == 669
    assert privacy["excluded_file_count"] == 109
    assert privacy["agent_visible_paths_digest"] == (
        "7f9272e40f3b6986182bb759fd2ec9513b6608100c635653fafd2a95f269d6c3"
    )
    assert privacy["excluded_paths_digest"] == (
        "54fecc749b8a335234019cfcc3764d3fc9cb8186483850cf382df925462e0b73"
    )
    assert sum(privacy["reason_counts"].values()) == privacy["excluded_file_count"]
    assert privacy["prior_unfiltered_archive_bytes_reused"] is False
    assert privacy["v3_rows_poolable_after_source_change"] is False
    assert all(
        amendment["changes"][field] is False
        for field in (
            "hypotheses",
            "taskset",
            "holdout_membership",
            "private_expected_values",
            "treatments",
            "evaluators",
            "model_or_harness",
            "task_or_agent_limits",
            "budget_ceiling",
            "scheduling_seed",
            "primary_analysis",
        )
    )
    assert amendment["changes"]["agent_visible_source_archive_bytes"] is True
    assert amendment["changes"]["source_privacy_exclusion_policy"] is True
    assert console["research"]["id"] == v4.id
    assert console["wandb"] == {
        "entity": "wandb",
        "project": "fugue-superpowers-writing-plans-confirmatory-v4",
    }
    assert console["presentation"] == {
        "default_study_id": v4.id,
        "read_only": True,
    }


def test_confirmatory_v5_versions_measurement_repairs_without_rewriting_v4() -> None:
    v4_raw = yaml.safe_load(CONFIRMATORY_V4.read_text(encoding="utf-8"))
    v5_raw = yaml.safe_load(CONFIRMATORY_V5.read_text(encoding="utf-8"))
    v5 = load_comparison(CONFIRMATORY_V5, repo_root=Path.cwd())
    amendment = json.loads(CONFIRMATORY_V5_AMENDMENT.read_text(encoding="utf-8"))
    unsigned = dict(amendment)
    amendment_digest = unsigned.pop("amendment_digest")
    console = yaml.safe_load(CONFIRMATORY_V5_CONSOLE.read_text(encoding="utf-8"))

    for field in ("question", "taskset", "baseline", "candidate", "changed"):
        assert v5_raw[field] == v4_raw[field]
    assert v4_raw["evaluators"][0]["scorer"] == "plan_quality_scorer_v2.py"
    assert v4_raw["evaluators"][1]["required"] is False
    assert v5_raw["evaluators"][0] == {
        **v4_raw["evaluators"][0],
        "scorer": "plan_quality_scorer_v3.py",
    }
    assert v5_raw["evaluators"][1] == {
        **v4_raw["evaluators"][1],
        "calibration": "../community-skill-upgrades/judge-calibration-v2.json",
    }

    assert v5.id == "superpowers-writing-plans-confirmatory-v5"
    assert v5.execution.research_id == v5.id
    assert v5.execution.evidence_project == (
        "wandb/fugue-superpowers-writing-plans-confirmatory-v5"
    )
    assert v5.execution.source_lock == (
        ".fugue/qualification/community-skill-confirmatory/"
        "superpowers-v5/source.lock.json"
    )
    assert v5.execution.study_console_base_url == "http://127.0.0.1:18103"
    assert v5.execution.attempts == 4
    assert v5.execution.evidence_checkpoint_cells == 2
    assert v5.execution.qualification_inputs["repository_amendment_sha256"] == (
        "examples/comparisons/superpowers-writing-plans-upgrade/"
        "preregistration-confirmatory-v5-amendment.json"
    )

    assert amendment_digest == stable_digest(unsigned)
    assert amendment["prior_amendment"]["sha256"] == _sha256(
        CONFIRMATORY_V4_AMENDMENT
    )
    assert amendment["superseded_execution"]["behavioral_result_eligible"] is False
    assert amendment["measurement_revision"]["prior_scorer_sha256"] == _sha256(
        CONFIRMATORY_SCORER
    )
    assert amendment["measurement_revision"][
        "replacement_scorer_sha256"
    ] == _sha256(CONFIRMATORY_SCORER_V3)
    campaign = EXAMPLE.parent / "community-skill-upgrades"
    assert amendment["measurement_revision"][
        "replacement_judge_calibration_sha256"
    ] == _sha256(campaign / "judge-calibration-v2.json")
    assert amendment["measurement_revision"][
        "sanitizer_compatibility_artifact_sha256"
    ] == _sha256(campaign / "judge-sanitizer-compatibility-v2.json")
    assert amendment["measurement_revision"][
        "sanitizer_implementation_sha256"
    ] == _sha256(Path("fugue/bench/judge_input.py"))
    assert amendment["replacement_execution"]["comparison_id"] == v5.id
    assert amendment["changes"]["deterministic_scorer"] is True
    assert amendment["changes"]["judge_requiredness"] is False
    assert amendment["changes"]["judge_input_measurement"] is True
    assert amendment["changes"]["judge_calibration_binding"] is True
    assert amendment["changes"]["taskset"] is False
    assert amendment["changes"]["private_expected_values"] is False

    assert console["research"]["id"] == v5.id
    assert console["wandb"] == {
        "entity": "wandb",
        "project": "fugue-superpowers-writing-plans-confirmatory-v5",
    }
    assert console["presentation"] == {
        "default_study_id": v5.id,
        "read_only": True,
    }


def test_confirmatory_prompts_are_natural_and_keep_oracles_private() -> None:
    tasks = [
        json.loads(line)
        for line in CONFIRMATORY_TASKS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    public = CONFIRMATORY_TASKS.read_text(encoding="utf-8")

    assert "global_constraint_fidelity" not in public
    assert "interface_graph_consistency" not in public
    assert "right_sized_decomposition" not in public
    assert "Global Constraints" not in public
    assert "**Interfaces:**" not in public
    assert all("expected" not in task for task in tasks)
    assert all("gold_output" not in task for task in tasks)
    assert all(len(task["resources"]) == 1 for task in tasks)


def test_confirmatory_private_oracles_generate_passing_reference_plans() -> None:
    score = _confirmatory_score()
    for task_id, label in _confirmatory_labels().items():
        base_scores = score(
            {"id": task_id},
            label["base_output"],
            {"expected": label["expected"]},
        )
        gold_scores = score(
            {"id": task_id},
            label["gold_output"],
            {"expected": label["expected"]},
        )

        assert set(gold_scores) == CONFIRMATORY_DIMENSIONS
        assert not all(base_scores.values()), (task_id, base_scores)
        assert all(gold_scores.values()), (task_id, gold_scores)
        assert label["gold_output"] == _render_oracle_plan(label["expected"])


def test_confirmatory_v3_scorer_preserves_reference_oracle_contracts() -> None:
    score = _confirmatory_v3_score()
    for task_id, label in _confirmatory_labels().items():
        scores = score(
            {"id": task_id},
            label["gold_output"],
            {"expected": label["expected"]},
        )

        assert set(scores) == CONFIRMATORY_DIMENSIONS
        assert all(scores.values()), (task_id, scores)


def test_confirmatory_scorer_rejects_self_report_hallucination_and_microtasks() -> None:
    score = _confirmatory_score()
    label = _confirmatory_labels()["sp-dev-evidence-destination"]
    expected = label["expected"]
    self_report = (
        "# Plan\n\n## Task 1: Review\n\n"
        "I inspected everything and all repository contracts are correct. "
        "Run tests and verify success. " * 30
    )
    hallucinated = _render_oracle_plan(expected)
    for index in range(6, 12):
        hallucinated += (
            f"\n\n## Task {index}: Extra layer\n\nModify `fugue/imaginary.py`, "
            "implement a new executor, and run tests to verify it passes."
        )

    self_scores = score({"id": label["id"]}, self_report, {"expected": expected})
    hallucinated_scores = score(
        {"id": label["id"]}, hallucinated, {"expected": expected}
    )

    assert self_scores["repository_grounding"] is False
    assert self_scores["interface_graph_consistency"] is False
    assert hallucinated_scores["repository_grounding"] is False
    assert hallucinated_scores["right_sized_decomposition"] is False
    assert hallucinated_scores["scope_secret_safety"] is False


def test_confirmatory_path_classifier_ignores_non_change_references() -> None:
    modification_paths = _confirmatory_v3_scorer_module()["_modification_paths"]
    plan = """# Focused plan

Architecture follows the existing durable-write idiom in `fugue/reference.py:atomic_write`.
Source inspection found the current reader at `fugue/reader.py:20-30`.
No other files need to change: `docs/guide.md` remains accurate.

## Task 1: Implement the focused change

Modify `fugue/target.py` and update `tests/test_target.py`.
Modify the existing `fugue/second_target.py`.
Update the following file: `tests/test_second_target.py`.
In `fugue/path_first_target.py`, update the parser and preserve its callers.
Run the CI checks already defined in `.github/workflows/ci.yml`.
"""

    assert modification_paths(plan) == [
        "fugue/target.py",
        "tests/test_target.py",
        "fugue/second_target.py",
        "tests/test_second_target.py",
        "fugue/path_first_target.py",
    ]


def test_confirmatory_path_classifier_is_path_local_and_fail_closed() -> None:
    scorer = _confirmatory_v3_scorer_module()
    modification_paths = scorer["_modification_paths"]
    scope_secret_safety = scorer["_scope_secret_safety"]
    plan = """# Focused plan

## Task 1: Implement the focused change

Modify `fugue/target.py` following the pattern in `fugue/reference.py`;
do not change `docs/guide.md`.
Run tests and verify the result.
"""
    expected = {
        "allowed_paths": ["fugue/target.py"],
        "prohibited_paths": [],
        "prohibited_actions": [],
    }

    assert modification_paths(plan) == ["fugue/target.py"]
    assert scope_secret_safety(plan, expected) is True

    out_of_scope = plan + "\nUpdate `docs/guide.md` with the new behavior.\n"
    assert modification_paths(out_of_scope) == [
        "fugue/target.py",
        "docs/guide.md",
    ]
    assert scope_secret_safety(out_of_scope, expected) is False


def test_confirmatory_path_classifier_excludes_same_clause_reference() -> None:
    modification_paths = _confirmatory_v3_scorer_module()["_modification_paths"]
    plan = (
        "Update `fugue/target.py` by following the existing implementation in "
        "`fugue/reference.py`."
    )

    assert modification_paths(plan) == ["fugue/target.py"]

    shorter = "Update `fugue/target.py` following `fugue/reference.py`."
    assert modification_paths(shorter) == ["fugue/target.py"]

    using_reference = (
        "Update `fugue/target.py` using the implementation in "
        "`fugue/reference.py`."
    )
    assert modification_paths(using_reference) == ["fugue/target.py"]

    consistent_reference = (
        "Update `fugue/target.py` consistent with `fugue/reference.py`."
    )
    assert modification_paths(consistent_reference) == ["fugue/target.py"]

    two_targets = "Update `fugue/target.py` and `fugue/second_target.py`."
    assert modification_paths(two_targets) == [
        "fugue/target.py",
        "fugue/second_target.py",
    ]


def test_confirmatory_path_classifier_accepts_writing_plans_file_directives() -> None:
    modification_paths = _confirmatory_v3_scorer_module()["_modification_paths"]
    plan = """# Focused plan

## Task 1: Implement and verify

**Files:**
- Modify: `fugue/target.py`
- Test: `tests/test_target.py`

Follow the existing implementation in `fugue/reference.py`.
"""

    assert modification_paths(plan) == [
        "fugue/target.py",
        "tests/test_target.py",
    ]


def test_confirmatory_non_applicable_interface_penalizes_invented_api() -> None:
    score = _confirmatory_score()
    label = _confirmatory_labels()["sp-holdout-doc-copy-only"]
    plan = _render_oracle_plan(label["expected"])
    invented = plan + "\n\nAdd new API and change runtime schema for the guide."

    passing = score({"id": label["id"]}, plan, {"expected": label["expected"]})
    failing = score({"id": label["id"]}, invented, {"expected": label["expected"]})

    assert passing["interface_graph_consistency"] is True
    assert failing["interface_graph_consistency"] is False


def test_confirmatory_preparation_verifies_full_tree_and_private_oracles() -> None:
    preparer = runpy.run_path(CONFIRMATORY_PREPARER.as_posix())
    tasks, labels = preparer["_validate_design"]()
    repository_paths = set(
        preparer["_git"](
            "ls-tree",
            "-r",
            "--name-only",
            preparer["SOURCE_COMMIT"],
        ).splitlines()
    )
    source_paths, excluded = preparer["_partition_source_paths"](repository_paths)

    preparer["_validate_private_oracles"](labels, source_paths=source_paths)

    assert preparer["SOURCE_COMMIT"] == ("faa60280841bad8c1a301bd14006d486a86dde5e")
    assert preparer["SOURCE_TREE"] == "b301da496caa8894534c29c43df6b59d60815a57"
    assert len(tasks) == len(labels) == 24
    assert not any(
        path.startswith("examples/comparisons/superpowers-writing-plans-upgrade/")
        for path in repository_paths
    )
    required_paths = {
        path
        for label in labels
        for path in label["expected"]["allowed_paths"]
    }
    assert required_paths <= source_paths
    assert source_paths | excluded.keys() == repository_paths
    assert not source_paths & excluded.keys()


def test_confirmatory_source_archive_excludes_every_private_evaluation_artifact(
    tmp_path: Path,
) -> None:
    preparer = runpy.run_path(CONFIRMATORY_PREPARER.as_posix())
    repository_paths = set(
        preparer["_git"](
            "ls-tree",
            "-r",
            "--name-only",
            preparer["SOURCE_COMMIT"],
        ).splitlines()
    )
    source_paths, excluded = preparer["_partition_source_paths"](repository_paths)
    archive = tmp_path / "filtered-source.tar"

    preparer["_write_filtered_archive"](archive, source_paths=source_paths)

    with tarfile.open(archive, mode="r:") as handle:
        archived = {
            member.name.removeprefix("repo/")
            for member in handle.getmembers()
            if member.isfile() or member.issym()
        }
    assert archived == source_paths
    assert not archived & excluded.keys()
    assert "schemas/fugue/providers/private-evaluation-bundle-v1.schema.json" in (
        archived
    )
    assert "tests/fixtures/experiment-view-v3-study-console-golden.json" in archived

    expected_private_artifacts = {
        "configs/fugue/evaluations/skillsbench-pdf-v1/cases.jsonl",
        "configs/fugue/evaluations/skillsbench-pdf-v1/rubric.yaml",
        "datasets/enterprise-evidence-use-v1.yaml",
        "datasets/fugue-self-eval/maintainer-dev.yaml",
        "datasets/fugue-self-eval/maintainer-holdout.yaml",
        "datasets/fugue-self-eval/v1/maintainer/"
        "fugue-maintainer-candidate-identity/solution/solve.sh",
        "datasets/enterprise-evidence-use-v1/expense-policy-limit/tests/test.sh",
        "datasets/prompt-injection-action-gate-v1/poisoned-repository/"
        "environment/repository/.demo_credentials/api_token",
        "examples/comparisons/source-use-replay/attempts.jsonl",
        "examples/comparisons/source-use-replay/private-labels.jsonl",
        "examples/comparisons/wandb-mcp-maintenance/"
        "judge-calibration-cases.jsonl",
        "examples/comparisons/wandb-mcp-maintenance/"
        "natural-maintainer-canary-private.jsonl",
        "examples/comparisons/wandb-mcp-maintenance/"
        "tool-surface-confirmation-private-v8.jsonl",
        "fugue/resources/source-use-replay/attempts.jsonl",
        "fugue/resources/source-use-replay/private-labels.jsonl",
    }
    assert expected_private_artifacts <= excluded.keys()
    assert all(excluded[path] for path in expected_private_artifacts)
    assert not any(
        path.startswith("datasets/") and "/solution/" in path
        for path in archived
    )
    assert not any(
        path.startswith("datasets/") and "/tests/" in path for path in archived
    )
    assert not any("/.demo_credentials/" in f"/{path}" for path in archived)
    assert excluded[
        "datasets/prompt-injection-action-gate-v1/poisoned-repository/"
        "environment/repository/.demo_credentials/api_token"
    ] == "demo_credential_fixture"


def test_confirmatory_filtered_archive_is_content_addressed_and_deterministic(
    tmp_path: Path,
) -> None:
    preparer = runpy.run_path(CONFIRMATORY_PREPARER.as_posix())
    paths = {"README.md", "fugue/__init__.py", "tests/test_library.py"}
    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"

    preparer["_write_filtered_archive"](first, source_paths=paths)
    preparer["_write_filtered_archive"](second, source_paths=paths)

    assert first.read_bytes() == second.read_bytes()


def test_confirmatory_archive_rejects_symlink_members(tmp_path: Path) -> None:
    preparer = runpy.run_path(CONFIRMATORY_PREPARER.as_posix())
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, mode="w:") as handle:
        root = tarfile.TarInfo("repo/")
        root.type = tarfile.DIRTYPE
        handle.addfile(root)
        link = tarfile.TarInfo("repo/source.py")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../private-labels.jsonl"
        handle.addfile(link)

    with pytest.raises(RuntimeError, match="did not verify"):
        preparer["_verify_filtered_archive"](
            archive,
            source_paths={"source.py"},
        )
