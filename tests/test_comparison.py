from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from fugue.bench.candidates import stable_digest
from fugue.bench.comparison import (
    COMPARISON_RUNTIME_ROOT,
    ComparisonEvaluatorV1,
    analyze_comparison_rows,
    check_comparison,
    comparison_from_dict,
    compile_comparison,
    load_comparison,
    preview_comparison,
    scaffold_comparison,
    score_comparison_rows,
)

EXAMPLE = Path("examples/comparisons/source-use-replay")
LIVE_SKILL_EXAMPLE = Path("examples/comparisons/source-use-skill")
MCP_MAINTENANCE_EXAMPLE = Path(
    "examples/comparisons/wandb-mcp-maintenance"
)


def test_source_use_comparison_is_ready_and_exact() -> None:
    root = Path.cwd()
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=root)

    readiness = check_comparison(spec, repo_root=root)
    preview = preview_comparison(spec, repo_root=root)

    assert readiness.status == "ready"
    assert readiness.task_count == 4
    assert readiness.base_failures == 4
    assert readiness.gold_passes == 4
    assert readiness.actual_changes == ("skills",)
    assert readiness.estimated_cells == 16
    assert readiness.estimated_cost_usd == 0
    assert preview.matrix["estimated_trials"] == 16
    assert preview.matrix["applicable_cells"] == 16
    assert not (root / COMPARISON_RUNTIME_ROOT / spec.spec_digest).exists()


def test_live_source_use_skill_comparison_has_locked_holdout_resources() -> None:
    root = Path.cwd()
    spec = load_comparison(
        LIVE_SKILL_EXAMPLE / "comparison.yaml", repo_root=root
    )

    readiness = check_comparison(spec, repo_root=root)
    preview = preview_comparison(spec, repo_root=root)

    assert readiness.status == "ready"
    assert readiness.task_count == 8
    assert readiness.base_failures == 8
    assert readiness.gold_passes == 8
    assert readiness.actual_changes == ("skills",)
    assert readiness.estimated_cells == 32
    assert preview.matrix["estimated_trials"] == 32
    assert {cell["harness"] for cell in preview.matrix["matrix_cells"]} == {
        "codex"
    }


@pytest.mark.parametrize(
    ("filename", "tasks", "harnesses", "attempts", "cells"),
    [
        ("discovery.yaml", 4, ("claude-code",), 1, 8),
        ("discovery-wandb.yaml", 4, ("openclaw",), 1, 8),
        ("primary.yaml", 8, ("claude-code",), 2, 32),
        ("wandb-replication.yaml", 8, ("openclaw",), 2, 32),
    ],
)
def test_mcp_maintenance_examples_have_exact_staged_designs(
    filename: str,
    tasks: int,
    harnesses: tuple[str, ...],
    attempts: int,
    cells: int,
) -> None:
    root = Path.cwd()
    spec = load_comparison(
        MCP_MAINTENANCE_EXAMPLE / filename,
        repo_root=root,
    )
    readiness = check_comparison(spec, repo_root=root)

    assert readiness.task_count == tasks
    assert spec.execution.harnesses == harnesses
    assert spec.execution.attempts == attempts
    assert readiness.estimated_cells == cells
    assert readiness.actual_changes == ("integrations",)
    assert readiness.status == "blocked"
    assert any("not adjudicated" in item for item in readiness.blockers)
    assert any("not locked and usable" in item for item in readiness.blockers)


def test_comparison_rejects_unknown_fields_and_undeclared_changes() -> None:
    root = Path.cwd()
    raw = yaml.safe_load((EXAMPLE / "comparison.yaml").read_text())
    raw["surprise"] = True
    with pytest.raises(ValueError, match="unknown comparison field"):
        comparison_from_dict(raw, repo_root=root, source=EXAMPLE)

    raw.pop("surprise")
    raw["changed"] = ["prompt_id"]
    spec = comparison_from_dict(raw, repo_root=root, source=EXAMPLE)
    readiness = check_comparison(spec, repo_root=root)
    assert readiness.status == "blocked"
    assert any("resolved behavior diff" in item for item in readiness.blockers)


def test_comparison_keeps_public_tasks_and_private_labels_separate(
    tmp_path: Path,
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    labels = tmp_path / "labels.jsonl"
    tasks.write_text(
        json.dumps(
            {
                "id": "leak",
                "input": {"question": "question"},
                "expected": {"answer": 1},
            }
        )
        + "\n"
    )
    labels.write_text(
        json.dumps(
            {
                "id": "leak",
                "expected": {"answer": 1},
                "base_output": {"answer": 0},
                "gold_output": {"answer": 1},
            }
        )
        + "\n"
    )
    raw = yaml.safe_load((EXAMPLE / "comparison.yaml").read_text())
    raw["taskset"] = {
        "tasks": tasks.relative_to(Path.cwd()).as_posix()
        if tasks.is_relative_to(Path.cwd())
        else tasks.as_posix(),
        "private_labels": labels.relative_to(Path.cwd()).as_posix()
        if labels.is_relative_to(Path.cwd())
        else labels.as_posix(),
    }
    spec = comparison_from_dict(raw, repo_root=tmp_path, source=tmp_path)
    with pytest.raises(ValueError, match="unknown public task .*field"):
        check_comparison(spec, repo_root=tmp_path)


def test_replay_scores_aligned_improvements_and_regressions() -> None:
    root = Path.cwd()
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=root)
    preview = preview_comparison(spec, repo_root=root)
    rows = [
        json.loads(line)
        for line in (EXAMPLE / "attempts.jsonl").read_text().splitlines()
    ]
    scored = score_comparison_rows(spec, rows, repo_root=root)
    result = analyze_comparison_rows(
        comparison_id=spec.id,
        preview_digest=preview.preview_digest,
        rows=scored,
        source="test-replay",
    )

    assert result.rows == 16
    assert result.baseline_passed == 2
    assert result.candidate_passed == 6
    assert result.improved == 5
    assert result.regressed == 1
    assert result.unchanged == 2
    assert result.incomplete == 0
    assert result.judge_summary == {"status": "not_used"}
    assert result.deterministic_summary["candidate"]["passed"] == 6
    assert result.operational_summary == {
        "execution_states": {"unknown": 16},
        "evidence_states": {"unknown": 16},
        "infrastructure_failures": 0,
        "observed_cost_usd": None,
        "cost_rows": 0,
        "latency_ms": None,
        "latency_rows": 0,
        "input_tokens": None,
        "output_tokens": None,
        "usage_rows": 0,
    }


def test_public_task_resources_are_digest_locked_into_the_task(tmp_path: Path) -> None:
    (tmp_path / "configs/fugue/skills/verify-current-source").mkdir(
        parents=True
    )
    (tmp_path / "configs/fugue/skills/verify-current-source/SKILL.md").write_text(
        "---\nname: verify-current-source\ndescription: Verify sources.\n---\n"
    )
    (tmp_path / "corpus").mkdir()
    resource = tmp_path / "corpus" / "policy.json"
    resource.write_text('{"documents": []}\n')
    (tmp_path / "tasks.jsonl").write_text(
        json.dumps(
            {
                "id": "policy",
                "input": {"question": "Read the corpus and answer."},
                "resources": [
                    {
                        "path": "corpus/policy.json",
                        "target": "/workspace/resources/corpus.json",
                    }
                ],
                "partition": "holdout",
            }
        )
        + "\n"
    )
    (tmp_path / "labels.jsonl").write_text(
        json.dumps(
            {
                "id": "policy",
                "expected": {"answer": "yes"},
                "base_output": {"answer": "no"},
                "gold_output": {"answer": "yes"},
            }
        )
        + "\n"
    )
    raw = yaml.safe_load((EXAMPLE / "comparison.yaml").read_text())
    raw["taskset"] = {"tasks": "tasks.jsonl", "private_labels": "labels.jsonl"}
    spec = comparison_from_dict(raw, repo_root=tmp_path, source=tmp_path)

    experiment, manifest, public = compile_comparison(spec, repo_root=tmp_path)

    assert public[0]["attachments"] == [
        {
            "locked_relative": "corpus/policy.json",
            "sha256": hashlib.sha256(resource.read_bytes()).hexdigest(),
            "target": "/workspace/resources/corpus.json",
        }
    ]
    assert "@sha256:" in public[0]["environment"]["base_image"]
    evaluator_digest = next(
        iter(check_comparison(spec, repo_root=tmp_path).evaluator_digests.values())
    )
    assert manifest["tasks"][0]["metadata"]["task_authoring"]["profile_digests"] == {
        "comparison-evaluator:fact-and-source": evaluator_digest
    }
    assert experiment.research_view is not None
    assert experiment.research_view.scorers[0].revision == evaluator_digest


def test_mechanism_summary_keeps_assignment_registration_and_use_distinct() -> None:
    root = Path.cwd()
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=root)
    rows = [
        {
            "answer": {"amount": 125, "source": "expense-policy-v4.md"},
            "harness": "codex",
            "prediction_id": "candidate-1",
            "task_id": "expense-limit",
            "trial_index": 1,
            "variant_id": "candidate",
            "skills_assigned": ["verify-current-source"],
            "skills_registered": ["verify-current-source"],
            "skill_registration_status": "registered",
            "skill_invocation_evidence": {
                "status": "observed",
                "skills_invoked": ["verify-current-source"],
            },
            "inspected_paths": ["/workspace/resources/expense-policy-v4.md"],
        },
        {
            "answer": {"amount": 100, "source": "expense-policy-v3.md"},
            "harness": "codex",
            "prediction_id": "baseline-1",
            "task_id": "expense-limit",
            "trial_index": 1,
            "variant_id": "baseline",
        },
    ]
    scored = score_comparison_rows(spec, rows, repo_root=root)
    result = analyze_comparison_rows(
        comparison_id=spec.id,
        preview_digest="a" * 64,
        rows=scored,
        source="test",
    )

    assert result.mechanism_summary["skill_assigned"]["candidate"] == {
        "observed": 1,
        "applicable": 1,
        "unavailable": 0,
    }
    assert result.mechanism_summary["relevant_source_used"]["candidate"][
        "observed"
    ] == 1


def test_zero_row_comparison_cannot_succeed() -> None:
    with pytest.raises(ValueError, match="at least one attempt row"):
        analyze_comparison_rows(
            comparison_id="empty",
            preview_digest="0" * 64,
            rows=[],
            source="test",
        )


def test_required_judge_needs_reviewed_calibration(tmp_path: Path) -> None:
    root = Path.cwd()
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=root)
    judge = ComparisonEvaluatorV1(
        id="maintainer-review",
        type="llm_judge",
        required=True,
        profile="wandb/openai/gpt-oss-120b",
        calibration=None,
        rubric="Score evidence grounding and calibration.",
        dimensions=("evidence_grounding", "calibration"),
        evidence=("tool_names",),
        reserve_cost_usd=0.1,
    )
    blocked = replace(spec, evaluators=(*spec.evaluators, judge))
    readiness = check_comparison(blocked, repo_root=root)
    assert readiness.status == "blocked"
    assert any("no reviewed calibration" in item for item in readiness.blockers)

    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
                {
                    "schema_version": 1,
                    "review_status": "adjudicated",
                    "reviewers_per_example": 2,
                    "disagreements_adjudicated": True,
                    "judge_profile": judge.profile,
                    "rubric_digest": stable_digest(
                        {
                            "schema_version": 1,
                            "judge_id": judge.id,
                            "profile": judge.profile,
                            "rubric": judge.rubric,
                            "dimensions": list(judge.dimensions),
                            "evidence": list(judge.evidence),
                        }
                    ),
                    "examples": 48,
                "true_positive_rate": 0.9,
                "true_negative_rate": 0.9,
                "critical_false_passes": 0,
            }
        )
    )
    copied = root / ".fugue" / "test-comparison-calibration.json"
    copied.parent.mkdir(parents=True, exist_ok=True)
    copied.write_text(calibration.read_text())
    try:
        qualified_judge = replace(
            judge, calibration=copied.relative_to(root).as_posix()
        )
        ready = replace(
            spec,
            evaluators=(*spec.evaluators, qualified_judge),
            execution=replace(spec.execution, max_cost_usd=10),
        )
        assert check_comparison(ready, repo_root=root).status == "ready"
    finally:
        copied.unlink(missing_ok=True)


def test_custom_scorer_uses_locked_sandbox_and_private_expected_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path.cwd()
    scorer_path = root / ".fugue" / "test-comparison-scorer.py"
    scorer_path.parent.mkdir(parents=True, exist_ok=True)
    scorer_path.write_text(
        "def score(task, output, evidence):\n"
        "    return {'fact_correct': output == evidence['expected']}\n"
    )
    observed: dict[str, object] = {}

    def fake_runner(*, source, evidence, reference, profile, limits):
        observed.update(
            source=source,
            evidence=evidence,
            reference=reference,
            profile=profile,
            limits=limits,
        )
        passed = reference["output"] == reference["expected"]
        return {
            "score": 1.0 if passed else 0.0,
            "reason": "custom deterministic scorer",
            "details": {"fact_correct": passed},
        }

    monkeypatch.setattr(
        "fugue.bench.task_authoring.run_inline_scorer", fake_runner
    )
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=root)
    evaluator = replace(
        spec.evaluators[0],
        checks=(),
        scorer=scorer_path.relative_to(root).as_posix(),
        runtime="python312-sandbox-v1",
        dimensions=("fact_correct",),
    )
    custom = replace(spec, evaluators=(evaluator,))
    try:
        rows = score_comparison_rows(
            custom,
            [
                {
                    "task_id": "expense-limit",
                    "variant_id": "candidate",
                    "harness": "codex",
                    "trial_index": 1,
                    "answer": {
                        "amount": 125,
                        "source": "expense-policy-v4.md",
                    },
                }
            ],
            repo_root=root,
        )
    finally:
        scorer_path.unlink(missing_ok=True)

    assert rows[0]["pass"] is True
    assert rows[0]["comparison_deterministic_scores"] == {
        "fact-and-source.fact_correct": True
    }
    assert observed["reference"] == {
        "task": {
            "id": "expense-limit",
            "input": {
                "question": (
                    "Return JSON containing the current expense amount "
                    "and its source."
                )
            },
            "resources": [],
            "tags": ["policy"],
            "partition": "holdout",
        },
        "output": {"amount": 125, "source": "expense-policy-v4.md"},
        "expected": {"amount": 125, "source": "expense-policy-v4.md"},
    }
    assert observed["evidence"] == {}
    assert "--network" not in str(observed["source"])


def test_blind_judge_receives_only_public_task_output_and_permitted_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path.cwd()
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=root)
    judge = ComparisonEvaluatorV1(
        id="maintainer-review",
        type="llm_judge",
        required=True,
        profile="wandb/openai/gpt-oss-120b",
        rubric="Score grounding, usefulness, prioritization, and calibration.",
        dimensions=(
            "evidence_grounding",
            "usefulness",
            "prioritization",
            "calibration",
        ),
        evidence=("tool_names", "inspected_paths"),
        reserve_cost_usd=0.1,
    )
    captured: dict[str, str] = {}

    def fake_post(_client, _route, _key, _env, prompt):
        captured["prompt"] = prompt
        return (
            {
                "scores": {
                    "evidence_grounding": 1,
                    "usefulness": 0.75,
                    "prioritization": 0.5,
                    "calibration": 1,
                },
                "overall_assessment": "Grounded and appropriately bounded.",
                "uncertainty": 0.1,
                "rationale": "The response cites the inspected current source.",
            },
            {"input_tokens": 100, "output_tokens": 40},
        )

    monkeypatch.setattr("fugue.bench.evaluations._post_judge", fake_post)
    judged = replace(spec, evaluators=(*spec.evaluators, judge))
    rows = score_comparison_rows(
        judged,
        [
            {
                "answer": {"amount": 125, "source": "expense-policy-v4.md"},
                "harness": "secret-harness-name",
                "prediction_id": "secret-prediction",
                "task_id": "expense-limit",
                "trial_index": 1,
                "variant_id": "secret-candidate-revision",
                "tool_calls": [
                    {
                        "name": "search",
                        "arguments": {"private": "do-not-send"},
                        "output": "private tool body",
                    }
                ],
                "inspected_paths": ["expense-policy-v4.md"],
                "comparison_deterministic_scores": {"private": True},
            }
        ],
        repo_root=root,
        env={"WANDB_API_KEY": "secret", "FUGUE_WANDB_INFERENCE_PROJECT": "wandb/test"},
    )

    prompt = captured["prompt"]
    assert "secret-candidate-revision" not in prompt
    assert "secret-harness-name" not in prompt
    assert "secret-prediction" not in prompt
    assert "do-not-send" not in prompt
    assert "private tool body" not in prompt
    assert '"tool_names": ["search"]' in prompt
    assert rows[0]["comparison_judge_status"] == "scored"
    assert rows[0]["comparison_required_evaluation_complete"] is True


def test_scaffold_refuses_non_empty_destination(tmp_path: Path) -> None:
    destination = tmp_path / "comparison"
    scaffold_comparison(destination)
    assert (destination / "comparison.yaml").is_file()
    assert (
        destination
        / "configs/fugue/skills/verify-current-source/SKILL.md"
    ).is_file()
    spec = load_comparison(
        destination / "comparison.yaml",
        repo_root=destination,
    )
    preview = preview_comparison(spec, repo_root=destination)
    assert preview.readiness["status"] == "ready"
    assert preview.matrix["estimated_trials"] == 4
    assert not (destination / ".fugue").exists()
    with pytest.raises(FileExistsError, match="non-empty"):
        scaffold_comparison(destination)
