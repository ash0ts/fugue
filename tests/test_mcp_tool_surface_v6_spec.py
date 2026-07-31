from __future__ import annotations

import json
from pathlib import Path

from fugue.bench.comparison import load_comparison

EXAMPLE = Path("examples/comparisons/wandb-mcp-maintenance")
SPEC = EXAMPLE / "tool-surface-canary-local-v6.yaml"
BASELINE_SHA = "53b199a5f4af29aa82077e2c7f1e2c5e5e0c2ca0"
CANDIDATE_SHA = "5c6cc1c9a1079296daf6613ea6d12daebdd8bcba"


def _ids(path: Path) -> set[str]:
    return {
        json.loads(line)["id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def test_v6_canary_is_exact_source_isolated_and_guards_all_cells() -> None:
    spec = load_comparison(SPEC, repo_root=Path.cwd())

    assert spec.id == "mcp-main-vs-0-4-tool-surface-canary-v6"
    assert Path(spec.taskset.tasks) == (
        EXAMPLE / "tool-surface-canary-tasks-v5.jsonl"
    )
    assert Path(spec.taskset.private_labels) == (
        EXAMPLE / "tool-surface-canary-private-v5.jsonl"
    )
    assert spec.taskset.tasks != spec.taskset.private_labels
    assert _ids(Path(spec.taskset.tasks)) == _ids(
        Path(spec.taskset.private_labels)
    )
    assert len(_ids(Path(spec.taskset.tasks))) == 4

    assert spec.baseline.integrations == ({"id": "wandb-mcp-main"},)
    assert spec.candidate.integrations == ({"id": "wandb-mcp-0-4-current"},)
    assert spec.changed == ("integrations",)
    assert spec.execution.source_evidence_project == (
        "wandb/fugue-mcp-release-source-v2"
    )
    assert spec.execution.evidence_project == (
        "wandb/fugue-mcp-release-qualification-v1"
    )
    assert spec.execution.environment == {"type": "docker"}
    assert spec.execution.harnesses == ("claude-code",)
    assert spec.execution.attempts == 1
    assert spec.execution.concurrency == 1
    assert spec.execution.evidence_checkpoint_cells == 8
    assert spec.execution.max_cost_usd == 5

    deterministic = next(
        evaluator for evaluator in spec.evaluators if evaluator.type == "deterministic"
    )
    assert Path(deterministic.scorer) == EXAMPLE / "tool_surface_scorer_v5.py"
    judge = next(
        evaluator for evaluator in spec.evaluators if evaluator.type == "llm_judge"
    )
    assert judge.id == "maintainer-actionability"
    assert judge.required is False
    assert judge.profile == "wandb/zai-org/GLM-5.2"

    assert spec.decision_policy is not None
    assert spec.decision_policy.candidate_sha == CANDIDATE_SHA
    current = json.loads(
        (EXAMPLE / "current-candidate.json").read_text(encoding="utf-8")
    )
    assert current["baseline"]["commit"] == BASELINE_SHA
    assert current["candidate"]["commit"] == CANDIDATE_SHA
    assert spec.supersedes[0].result_digest == (
        "63ca8185bc6d52509859178d8a0cc5d2fe310a1f80362441f289512d12766c1d"
    )
