from __future__ import annotations

import json
import runpy
from pathlib import Path

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
CONFIRMATORY_TASKS = EXAMPLE / "tasks-conference-v1.jsonl"
CONFIRMATORY_PRIVATE = EXAMPLE / "private-labels-conference-v1.jsonl"
CONFIRMATORY_SCORER = EXAMPLE / "plan_quality_scorer_v2.py"
CONFIRMATORY_PREREGISTRATION = EXAMPLE / "preregistration-confirmatory-v1.json"
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


def _confirmatory_labels() -> dict[str, dict]:
    return {
        item["id"]: item
        for item in (
            json.loads(line)
            for line in CONFIRMATORY_PRIVATE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


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


def test_v4_routes_to_dedicated_project_and_adds_advisory_judge() -> None:
    spec = load_comparison(EXAMPLE / "comparison-v4.yaml", repo_root=Path.cwd())

    assert spec.id == "superpowers-writing-plans-upgrade-canary-v4"
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
    source_paths = set(
        preparer["_git"](
            "ls-tree",
            "-r",
            "--name-only",
            preparer["SOURCE_COMMIT"],
        ).splitlines()
    )

    preparer["_validate_private_oracles"](labels, source_paths=source_paths)

    assert preparer["SOURCE_COMMIT"] == ("faa60280841bad8c1a301bd14006d486a86dde5e")
    assert preparer["SOURCE_TREE"] == "b301da496caa8894534c29c43df6b59d60815a57"
    assert len(tasks) == len(labels) == 24
    assert not any(
        path.startswith("examples/comparisons/superpowers-writing-plans-upgrade/")
        for path in source_paths
    )
