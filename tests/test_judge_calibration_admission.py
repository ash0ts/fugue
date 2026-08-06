from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fugue.bench.candidates import stable_digest
from fugue.bench.comparison import (
    ComparisonEvaluatorV1,
    _evaluator_digest,
    _require_execution_judge_calibrations,
    comparison_from_dict,
    load_comparison,
)

EXAMPLE = Path("examples/comparisons/community-skill-selected-v1")
MODALITIES = ("code-change", "implementation-plan", "skill-package")


def _perfect_metrics(*, aggregate: bool = False) -> dict[str, int | float]:
    count = 24 if aggregate else 8
    return {
        "true_positive": count,
        "false_negative": 0,
        "true_negative": count,
        "false_positive": 0,
        "true_positive_rate": 1.0,
        "true_negative_rate": 1.0,
    }


def _receipt(rubric_digest: str) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": 1,
        "kind": "judge-calibration-receipt",
        "cases_digest": "a" * 64,
        "rubric_digest": rubric_digest,
        "model_output_receipt_digest": "b" * 64,
        "reviewer_submission_digests": ["c" * 64, "d" * 64],
        "adjudication_digest": "e" * 64,
        "judge_profile": "anthropic/claude-sonnet-5",
        "judge_model_family": "anthropic",
        "evaluated_agent_model_family": "anthropic",
        "same_model_family": True,
        "claim_role": "advisory",
        "review_status": "adjudicated",
        "case_count": 48,
        "overall": _perfect_metrics(aggregate=True),
        "modalities": {modality: _perfect_metrics() for modality in MODALITIES},
        "critical_false_passes": 0,
        "thresholds": {
            "minimum_tpr_tnr": 0.85,
            "critical_false_passes_max": 0,
        },
        "status": "passed",
    }
    value["receipt_digest"] = stable_digest(value)
    return value


def _judge(tmp_path: Path) -> tuple[ComparisonEvaluatorV1, str]:
    source = EXAMPLE / "judge/rubric.json"
    rubric = tmp_path / "rubric.json"
    rubric.write_bytes(source.read_bytes())
    digest = hashlib.sha256(rubric.read_bytes()).hexdigest()
    return (
        ComparisonEvaluatorV1(
            id="plan-usefulness",
            type="llm_judge",
            required=False,
            profile="anthropic/claude-sonnet-5",
            calibration="private/calibration.json",
            calibration_rubric="rubric.json",
            calibration_modality="implementation-plan",
            calibration_required_for_execution=True,
            rubric="Assess usefulness without overriding deterministic gates.",
            dimensions=(
                "actionability",
                "repository_grounding",
                "reviewability",
                "risk_calibration",
            ),
            dimension_roles={
                "actionability": "outcome",
                "repository_grounding": "outcome",
                "reviewability": "outcome",
                "risk_calibration": "outcome",
            },
            reserve_cost_usd=0.1,
        ),
        digest,
    )


def _spec(judge: ComparisonEvaluatorV1) -> SimpleNamespace:
    return SimpleNamespace(
        evaluators=(judge,),
        execution=SimpleNamespace(model="anthropic/claude-sonnet-5"),
    )


def test_execution_calibration_is_strict_and_changes_the_approval_identity(
    tmp_path: Path,
) -> None:
    judge, rubric_digest = _judge(tmp_path)
    missing_digest = _evaluator_digest(judge, tmp_path)

    with pytest.raises(ValueError, match="calibration prerequisite.*cannot be read"):
        _require_execution_judge_calibrations(_spec(judge), repo_root=tmp_path)

    receipt_path = tmp_path / "private/calibration.json"
    receipt_path.parent.mkdir()
    receipt = _receipt(rubric_digest)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    _require_execution_judge_calibrations(_spec(judge), repo_root=tmp_path)
    first_final_digest = _evaluator_digest(judge, tmp_path)
    assert first_final_digest != missing_digest

    # Even a semantically equivalent, still-passing receipt changes the
    # evaluator digest and therefore the exact preview and stage approvals.
    receipt["model_output_receipt_digest"] = "f" * 64
    receipt.pop("receipt_digest")
    receipt["receipt_digest"] = stable_digest(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    _require_execution_judge_calibrations(_spec(judge), repo_root=tmp_path)
    assert _evaluator_digest(judge, tmp_path) != first_final_digest


def test_execution_calibration_rejects_legacy_and_per_modality_failure(
    tmp_path: Path,
) -> None:
    judge, rubric_digest = _judge(tmp_path)
    receipt_path = tmp_path / "private/calibration.json"
    receipt_path.parent.mkdir()
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "review_status": "adjudicated",
                "reviewers_per_example": 2,
                "disagreements_adjudicated": True,
                "judge_profile": judge.profile,
                "rubric_digest": "a" * 64,
                "cases_digest": "b" * 64,
                "examples": 48,
                "calibration_examples": 36,
                "holdout_examples": 12,
                "true_positive_rate": 1,
                "true_negative_rate": 1,
                "calibration_true_positive_rate": 1,
                "calibration_true_negative_rate": 1,
                "holdout_true_positive_rate": 1,
                "holdout_true_negative_rate": 1,
                "critical_false_passes": 0,
                "passed": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires a strict calibration receipt"):
        _require_execution_judge_calibrations(_spec(judge), repo_root=tmp_path)

    receipt = _receipt(rubric_digest)
    modality = dict(receipt["modalities"])["implementation-plan"]
    assert isinstance(modality, dict)
    modality.update(
        {
            "true_negative": 6,
            "false_positive": 2,
            "true_negative_rate": 0.75,
        }
    )
    receipt.pop("receipt_digest")
    receipt["receipt_digest"] = stable_digest(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="implementation-plan.*below 0.85"):
        _require_execution_judge_calibrations(_spec(judge), repo_root=tmp_path)


def test_campaign_judges_are_advisory_while_human_calibration_is_pending() -> None:
    for lane in (
        "superpowers-writing-plans",
        "anthropic-skill-creator",
        "vercel-react-best-practices",
    ):
        spec = load_comparison(EXAMPLE / lane / "comparison.yaml", repo_root=Path.cwd())
        judge = next(item for item in spec.evaluators if item.type == "llm_judge")
        assert judge.required is False
        assert judge.calibration_required_for_execution is False


def test_execution_calibration_flag_is_strictly_typed_and_judge_only() -> None:
    raw = {
        "schema_version": 3,
        "id": "strict-calibration-parse",
        "question": "Does it work?",
        "taskset": {"tasks": "tasks.jsonl", "private_labels": "labels.jsonl"},
        "baseline": {"label": "base"},
        "candidate": {"label": "candidate", "prompt_id": "candidate-prompt"},
        "changed": ["prompt_id"],
        "evaluators": [
            {
                "id": "facts",
                "type": "deterministic",
                "required": True,
                "checks": ["answer_present"],
                "calibration_required_for_execution": True,
            }
        ],
        "execution": {
            "model": "anthropic/claude-sonnet-5",
            "harnesses": ["claude-code"],
            "attempts": 1,
            "concurrency": 1,
            "max_cost_usd": 1,
            "reserve_per_attempt_usd": 0.5,
            "approval_required": True,
            "trace_content": "full",
            "evidence_project": "wandb/example",
            "evidence_destination": {
                "entity": "wandb",
                "project": "example",
                "api_base_url": "https://api.wandb.ai",
                "trace_base_url": "https://trace.wandb.ai",
                "app_base_url": "https://wandb.ai",
            },
        },
    }
    with pytest.raises(ValueError, match="requires an LLM judge"):
        comparison_from_dict(raw, repo_root=Path.cwd())

    raw["evaluators"][0]["calibration_required_for_execution"] = "true"
    with pytest.raises(ValueError, match="must be a boolean"):
        comparison_from_dict(raw, repo_root=Path.cwd())
