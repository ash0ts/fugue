from pathlib import Path

from fugue.bench.comparison import load_comparison

EXAMPLE = Path("examples/comparisons/wandb-mcp-maintenance")
V10 = EXAMPLE / "tool-surface-confirmation-local-v10.yaml"


def test_v10_retains_the_approved_sixteen_cell_contract() -> None:
    spec = load_comparison(V10, repo_root=Path.cwd())
    judge = next(item for item in spec.evaluators if item.type == "llm_judge")
    task_count = sum(
        bool(line.strip())
        for line in Path(spec.taskset.tasks).read_text(
            encoding="utf-8"
        ).splitlines()
    )

    assert spec.id == "mcp-main-vs-0-4-tool-surface-confirmation-v10"
    assert task_count == 4
    assert spec.execution.attempts == 2
    assert task_count * 2 * spec.execution.attempts == 16
    assert spec.execution.max_cost_usd == 10
    assert spec.execution.evidence_checkpoint_cells == 1
    assert judge.required is False
