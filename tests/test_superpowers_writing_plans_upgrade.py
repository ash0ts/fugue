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


def test_locked_canary_uses_exact_skill_upgrade_and_eight_cells() -> None:
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=Path.cwd())
    lock = json.loads(LOCK.read_text(encoding="utf-8"))

    assert spec.schema_version == 2
    assert spec.baseline.skills == (lock["baseline"]["id"],)
    assert spec.candidate.skills == (lock["candidate"]["id"],)
    assert lock["repository"] == "https://github.com/obra/superpowers"
    assert lock["path"] == "skills/writing-plans"
    assert lock["baseline"]["commit"] == (
        "de4672b171213a6ff6960228d8b95c46ea0b09f4"
    )
    assert lock["candidate"]["commit"] == (
        "8e1262a3bae92b640d87fa81c51c53b65e490590"
    )
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

    microtask = label["gold_output"] + """

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
        label["gold_output"]
        + "\n\n## Task 4: Change deployment\n\n"
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
