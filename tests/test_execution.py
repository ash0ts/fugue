from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from fugue.bench.execution import (
    PlannedCell,
    execute_cells,
    latest_cell_records,
    new_run_id,
)
from fugue.bench.export import export_rows


def _cell(run_id: str, name: str, *, applicable: bool = True) -> PlannedCell:
    return PlannedCell(
        id=f"cell-{name}",
        run_id=run_id,
        run_name="test-run",
        workload_id="coding",
        task_id=f"task-{name}",
        harness="codex",
        context_system_id="none",
        variant_id="baseline",
        model_provider="openai",
        model="openai/gpt-5",
        config_path=Path(f"{name}.json"),
        command=(name,),
        env={},
        n_attempts=2,
        applicable=applicable,
        skip_reason=None if applicable else "unsupported",
    )


def test_cells_are_bounded_failure_isolated_and_durable(tmp_path: Path) -> None:
    run_id = new_run_id()
    active = 0
    max_active = 0
    lock = threading.Lock()

    def runner(command, **kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return SimpleNamespace(returncode=1 if command[0] == "fail" else 0)

    cells = [
        _cell(run_id, "one"),
        _cell(run_id, "fail"),
        _cell(run_id, "two"),
        _cell(run_id, "skip", applicable=False),
    ]
    outcomes = execute_cells(
        cells,
        repo_root=tmp_path,
        max_workers=2,
        runner=runner,
    )

    assert 1 < max_active <= 2
    assert {item.cell_id: item.status for item in outcomes} == {
        "cell-one": "passed",
        "cell-fail": "failed",
        "cell-two": "passed",
        "cell-skip": "not_applicable",
    }
    state_path = tmp_path / ".fugue" / "runtime" / run_id / "cells.jsonl"
    latest = {item["cell_id"]: item for item in latest_cell_records(state_path)}
    assert latest["cell-fail"]["status"] == "failed"
    assert latest["cell-skip"]["status"] == "not_applicable"

    rows = export_rows([state_path.parent])
    assert {row["status"] for row in rows} == {
        "passed",
        "failed",
        "not_applicable",
    }
    assert all(row["record_type"] == "cell" for row in rows)


def test_run_ids_are_immutable_and_unique() -> None:
    assert new_run_id() != new_run_id()


def test_execution_rejects_mixed_runs_and_duplicate_cells(tmp_path: Path) -> None:
    first = _cell("run-a", "same")
    second_run = _cell("run-b", "other")
    with pytest.raises(ValueError, match="share a run_id"):
        execute_cells([first, second_run], repo_root=tmp_path, max_workers=1)
    with pytest.raises(ValueError, match="cell ids must be unique"):
        execute_cells([first, first], repo_root=tmp_path, max_workers=1)
