import json
import runpy
from pathlib import Path

import yaml

from fugue.bench.comparison import load_comparison

EXAMPLE = Path("examples/comparisons/wandb-mcp-maintenance")
SCORER = EXAMPLE / "tool_surface_scorer_v7.py"
PRIVATE = EXAMPLE / "tool-surface-confirmation-private-v8.jsonl"
TASKS = EXAMPLE / "tool-surface-confirmation-tasks-v8.jsonl"
SPEC = EXAMPLE / "tool-surface-confirmation-local-v10.yaml"
PROJECT = "wandb/fugue-mcp-release-source-v2"


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


def _evaluation_output(
    summary: int,
    *,
    evidence_status: str = "reconciled",
) -> dict:
    matches = summary == 16
    return {
        "source_project": PROJECT,
        "evaluation_root_count": 2,
        "summary_prediction_count": summary,
        "direct_prediction_count": 16,
        "summary_child_count": 2,
        "other_child_count": 0,
        "summary_matches_direct": matches,
        "recommendation": "advance" if matches else "investigate",
        "bounded": True,
        "evidence_status": evidence_status,
        "maintainer_memo": (
            "Every direct child is accounted for; investigate the summary "
            "definition when its product count differs."
        ),
    }


def _evaluation_evidence(label: dict, prediction_count: int) -> dict:
    calls = json.loads(
        json.dumps(label["gold_evidence"]["mcp_tool_calls"])
    )
    calls[0]["prediction_count"] = prediction_count
    return {"expected": label["expected"], "mcp_tool_calls": calls}


def test_v8_explained_baseline_mismatch_is_honest_but_not_target() -> None:
    label = _labels()["evaluation-summary-accuracy"]

    result = _score()(
        {"id": label["id"]},
        json.dumps(_evaluation_output(18)),
        _evaluation_evidence(label, 18),
    )

    assert result["answer_correct"] is True
    assert result["actual_query_scope"] is True
    assert result["bounded_evidence"] is True
    assert result["evidence_honesty"] is True
    assert result["release_mechanism_used"] is True
    assert result["target_behavior_satisfied"] is False


def test_v8_matching_candidate_is_honest_and_satisfies_target() -> None:
    label = _labels()["evaluation-summary-accuracy"]

    result = _score()(
        {"id": label["id"]},
        json.dumps(_evaluation_output(16)),
        _evaluation_evidence(label, 16),
    )

    assert result["answer_correct"] is True
    assert result["actual_query_scope"] is True
    assert result["bounded_evidence"] is True
    assert result["evidence_honesty"] is True
    assert result["release_mechanism_used"] is True
    assert result["target_behavior_satisfied"] is True


def test_v8_rejects_conflicted_label_for_fully_resolved_mismatch() -> None:
    label = _labels()["evaluation-summary-accuracy"]

    result = _score()(
        {"id": label["id"]},
        json.dumps(_evaluation_output(18, evidence_status="conflicted")),
        _evaluation_evidence(label, 18),
    )

    assert result["bounded_evidence"] is True
    assert result["evidence_honesty"] is False
    assert result["target_behavior_satisfied"] is False


def test_v8_public_task_defines_resolution_separately_from_agreement() -> None:
    tasks = {
        item["id"]: item
        for item in (
            json.loads(line)
            for line in TASKS.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    prompt = tasks["evaluation-summary-accuracy"]["input"]["question"]

    assert "resolution of the evidence, not agreement" in prompt
    assert "use `reconciled`" in prompt
    assert "`summary_matches_direct` and `recommendation`" in prompt
    assert all("expected" not in task for task in tasks.values())


def test_v10_preserves_confirmation_matrix_locks_and_v7_scorer() -> None:
    spec = load_comparison(SPEC, repo_root=Path.cwd())
    evaluator = next(item for item in spec.evaluators if item.id == "tool-surface")
    raw = yaml.safe_load(SPEC.read_text(encoding="utf-8"))

    assert spec.id == "mcp-main-vs-0-4-tool-surface-confirmation-v10"
    assert spec.execution.attempts == 2
    assert spec.execution.concurrency == 1
    assert spec.execution.evidence_checkpoint_cells == 1
    assert spec.execution.max_cost_usd == 10
    assert evaluator.scorer.endswith("/tool_surface_scorer_v7.py")
    assert raw["taskset"] == {
        "tasks": "tool-surface-confirmation-tasks-v8.jsonl",
        "private_labels": "tool-surface-confirmation-private-v8.jsonl",
    }
    assert raw["baseline"]["integrations"] == ["wandb-mcp-main"]
    assert raw["candidate"]["integrations"] == ["wandb-mcp-0-4-current"]
    assert raw["execution"]["source_evidence_project"] == (
        "wandb/fugue-mcp-release-source-v2"
    )
    assert raw["execution"]["evidence_project"] == (
        "wandb/fugue-mcp-release-qualification-v1"
    )
    assert raw["execution"]["release_notes_lock"] == (
        "release-notes.current.lock.json"
    )
    assert raw["execution"]["model"] == "anthropic/claude-sonnet-5"
    assert raw["execution"]["harnesses"] == ["claude-code"]
    assert raw["execution"]["environment"] == {"type": "docker"}
    assert raw["supersedes"] == [
        {
            "result_digest": (
                "cd083539eb06f1253fd663110b551696cc8fe08a7a10dfb164"
                "c3c9e8b1c6de12"
            ),
            "reason": raw["supersedes"][0]["reason"],
        }
    ]
