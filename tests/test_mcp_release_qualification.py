from __future__ import annotations

import json
from pathlib import Path

import pytest

from fugue.bench.comparison import check_comparison, compile_comparison, load_comparison
from fugue.bench.mcp_release_qualification import (
    QUALIFICATION_PROJECT,
    _evidence_lock,
    qualification_seed,
    qualification_seed_digest,
    validate_evidence_lock,
)

EXAMPLE = Path("examples/comparisons/wandb-mcp-maintenance")


def _lock() -> dict:
    project = QUALIFICATION_PROJECT
    return _evidence_lock(
        project,
        [
            {
                "id": f"run-{index}",
                "ref": f"wandb-run:///{project}/run-{index}",
            }
            for index in range(6)
        ],
        {
            "dataset": {
                "name": "cases",
                "ref": f"weave:///{project}/object/cases:v1",
                "rows": 8,
            },
            "source_conversations": [
                {
                    "call_id": f"call-{index}",
                    "ref": f"weave:///{project}/call/call-{index}",
                }
                for index in range(24)
            ],
            "evaluations": [
                {
                    "revision": revision,
                    "ref": f"weave:///{project}/object/evaluation-{revision}:v1",
                    "call_ref": f"weave:///{project}/call/evaluation-{revision}",
                    "prediction_rows": 8,
                }
                for revision in ("maintainer-r17", "maintainer-r18")
            ],
        },
    )


def test_seed_has_real_nonzero_evidence_and_actionable_anomaly() -> None:
    seed = qualification_seed()
    facts = seed["facts"]

    assert seed["project"] == QUALIFICATION_PROJECT
    assert facts["run_count"] == 6
    assert facts["source_conversation_count"] == 24
    assert facts["evaluation_prediction_rows"] == 16
    assert facts["latency_anomaly"] == {
        "attempt_label": "broad-history-scan-claude",
        "latency_ms": 4200,
        "cohort_median_ms": 920,
        "ratio": 4.5652,
    }
    assert facts["cost_coverage"] == {
        "attempts": 6,
        "attempts_with_observed_cost": 5,
        "total_observed_usd": 0.96,
        "complete": False,
    }
    assert facts["regressions"] == ["partial-evidence"]
    assert len(qualification_seed_digest()) == 64


def test_evidence_lock_requires_exact_counts_and_immutable_refs() -> None:
    lock = _lock()

    assert validate_evidence_lock(lock) == lock

    wrong_count = _lock()
    wrong_count["counts"]["runs"] = 0
    wrong_count["evidence_lock_digest"] = ""
    from fugue.bench.candidates import stable_digest

    wrong_count["evidence_lock_digest"] = stable_digest(wrong_count)
    with pytest.raises(ValueError, match="runs must equal 6"):
        validate_evidence_lock(wrong_count)

    mutable_ref = _lock()
    mutable_ref["objects"]["dataset"]["ref"] = "wandb/latest"
    mutable_ref["evidence_lock_digest"] = ""
    mutable_ref["evidence_lock_digest"] = stable_digest(mutable_ref)
    with pytest.raises(ValueError, match="Dataset reference is not immutable"):
        validate_evidence_lock(mutable_ref)


def test_existing_lock_must_validate_before_idempotent_reuse(
    tmp_path: Path,
) -> None:
    from fugue.bench.mcp_release_qualification import prepare_hosted_project

    output = tmp_path / "evidence.lock.json"
    output.write_text('{"schema_version": 1}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="digest does not match"):
        prepare_hosted_project(
            project=QUALIFICATION_PROJECT,
            output=output,
            env_file=tmp_path / "missing.env",
        )


def test_checked_in_hosted_lock_is_exact_and_contains_no_credentials() -> None:
    lock = json.loads(
        (EXAMPLE / "evidence.lock.json").read_text(encoding="utf-8")
    )

    assert validate_evidence_lock(lock) == lock
    serialized = json.dumps(lock, sort_keys=True)
    assert "WANDB_API_KEY" not in serialized
    assert "ANTHROPIC_API_KEY" not in serialized
    assert "sk-ant-" not in serialized
    assert all(
        item["ref"].startswith("weave:///")
        for item in lock["objects"]["source_conversations"]
    )


def test_mcp_study_has_exact_80_cell_direct_provider_design() -> None:
    root = Path.cwd()
    designs = (
        ("discovery.yaml", "anthropic/claude-sonnet-5", "claude-code", 8),
        (
            "discovery-wandb.yaml",
            "wandb/deepseek-ai/DeepSeek-V4-Flash",
            "openclaw",
            8,
        ),
        ("primary.yaml", "anthropic/claude-sonnet-5", "claude-code", 32),
        (
            "wandb-replication.yaml",
            "wandb/deepseek-ai/DeepSeek-V4-Flash",
            "openclaw",
            32,
        ),
    )
    total = 0
    for filename, model, harness, cells in designs:
        spec = load_comparison(EXAMPLE / filename, repo_root=root)
        readiness = check_comparison(spec, repo_root=root)
        assert spec.execution.model == model
        assert spec.execution.harnesses == (harness,)
        assert readiness.estimated_cells == cells
        assert {
            evaluator.profile
            for evaluator in spec.evaluators
            if evaluator.type == "llm_judge"
        } == {"anthropic/claude-sonnet-5"}
        total += cells

    assert total == 80


def test_public_tasks_lock_evidence_without_private_label_leakage() -> None:
    root = Path.cwd()
    spec = load_comparison(EXAMPLE / "primary.yaml", repo_root=root)
    _, _, public_tasks = compile_comparison(spec, repo_root=root)
    serialized = json.dumps(public_tasks, sort_keys=True)

    assert len(public_tasks) == 8
    assert all(
        task["attachments"]
        == [
            {
                "locked_relative": (
                    "examples/comparisons/wandb-mcp-maintenance/"
                    "evidence.lock.json"
                ),
                "sha256": task["attachments"][0]["sha256"],
                "target": "/workspace/resources/evidence.lock.json",
            }
        ]
        for task in public_tasks
    )
    assert "current-source-returned-mostly-not-opened" not in serialized
    assert "gold_output" not in serialized
    assert "base_output" not in serialized
    assert "WANDB_API_KEY" not in serialized
    assert "ANTHROPIC_API_KEY" not in serialized


def test_judge_calibration_is_balanced_and_truthfully_pending() -> None:
    cases = [
        json.loads(line)
        for line in (EXAMPLE / "judge-calibration-cases.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    report = json.loads(
        (EXAMPLE / "judge-calibration.json").read_text(encoding="utf-8")
    )

    assert len(cases) == 48
    labels = [item["authored_reference"]["label"] for item in cases]
    assert labels.count("pass") == labels.count("fail") == 24
    assert all(item["reviews"] == [] for item in cases)
    assert report["examples"] == 48
    assert report["judge_profile"] == "anthropic/claude-sonnet-5"
    assert report["review_status"] == "pending_human_review"
    assert report["passed"] is False
