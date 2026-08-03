from __future__ import annotations

import importlib.util
import json
from pathlib import Path
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


def _load_audit_module() -> Any:
    path = CAMPAIGN / "freeze_trace_audit.py"
    spec = importlib.util.spec_from_file_location("freeze_trace_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = _load_audit_module()


def _row(task_index: int, variant: str, attempt: int) -> dict[str, Any]:
    attempt_id = f"attempt-{task_index:02d}-{variant}-{attempt}"
    return {
        "attempt_id": attempt_id,
        "task_id": f"task-{task_index:02d}",
        "variant_id": variant,
        "trial_index": attempt,
        "status": "passed",
        "result_evidence_project": "wandb/fugue-superpowers-writing-plans-confirmatory-v1",
        "trace_link_status": "linked",
        "sandbox_cleanup_verified": True,
        "private_label_boundary_verified": True,
        "weave_evaluation_root_call_id": f"evaluation-{attempt_id}",
        "eval_predict_and_score_call_id": f"score-{attempt_id}",
        "weave_prediction_call_id": f"prediction-{attempt_id}",
        "weave_agent_root_call_id": f"agent-{attempt_id}",
        "weave_dataset_id": "dataset-v1",
        "comparison_deterministic_scores": {
            "plan.outcome": True,
            "plan.safety": True,
        },
        "comparison_dimension_roles": {
            "plan.outcome": "outcome",
            "plan.safety": "safety_gate",
        },
        "cost_usd": 0.25,
        "wall_time_sec": 1.5,
    }


def _write_campaign_inputs(tmp_path: Path) -> tuple[Path, Path]:
    attempts = tmp_path / "attempts.jsonl"
    rows = [
        _row(task_index, variant, attempt)
        for task_index in range(24)
        for variant in ("baseline", "candidate")
        for attempt in range(1, 5)
    ]
    attempts.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "comparison_id": "superpowers-writing-plans-confirmatory-v1",
                "evidence_project": "wandb/fugue-superpowers-writing-plans-confirmatory-v1",
            }
        ),
        encoding="utf-8",
    )
    return attempts, result


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


def test_confirmatory_analysis_treats_attempts_as_task_clusters(tmp_path: Path) -> None:
    attempts, result = _write_campaign_inputs(tmp_path)
    report = ANALYSIS.analyze(
        attempts_path=attempts,
        result_path=result,
        campaign_manifest_path=CAMPAIGN / "conference-campaign-manifest.json",
        preregistration_path=CAMPAIGN / "conference-preregistration.json",
        study_id="superpowers-writing-plans-confirmatory-v1",
    )
    assert report["counts"] == {
        "arms": 2,
        "attempts_per_task_per_arm": 4,
        "rows": 192,
        "tasks": 24,
    }
    assert report["finding"]["status"] == "unchanged"
    assert report["finding"]["equivalence_established"] is True
    assert report["efficiency"]["observed_cost_usd"] == 48.0
    assert len(report["analysis_digest"]) == 64


def test_confirmatory_analysis_rejects_duplicate_attempts(tmp_path: Path) -> None:
    attempts, result = _write_campaign_inputs(tmp_path)
    first = attempts.read_text(encoding="utf-8").splitlines()[0]
    attempts.write_text(attempts.read_text(encoding="utf-8") + first + "\n")
    with pytest.raises(ValueError, match="expected 192 attempt rows"):
        ANALYSIS.analyze(
            attempts_path=attempts,
            result_path=result,
            campaign_manifest_path=CAMPAIGN / "conference-campaign-manifest.json",
            preregistration_path=CAMPAIGN / "conference-preregistration.json",
            study_id="superpowers-writing-plans-confirmatory-v1",
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
