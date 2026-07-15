from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from fugue.bench.execution import (
    PlannedCell,
    execute_cells,
    latest_cell_records,
    new_run_id,
    read_run_manifest,
    update_run_manifest,
    write_run_manifest,
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
        trial_index=1,
        comparison_example_id=f"example-{name}",
        candidate_id="candidate-codex-baseline",
        execution_fingerprint="execution-a",
        config_path=Path(f"{name}.json"),
        result_path=Path("jobs") / name / "result.json",
        command=(name,),
        env={},
        n_attempts=1,
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
    events = [
        json.loads(line)
        for line in (state_path.parent / "events.jsonl").read_text().splitlines()
    ]
    assert any(
        event["event"] == "cell_state"
        and event["cell_id"] == "cell-fail"
        and event["status"] == "failed"
        for event in events
    )


def test_run_ids_are_immutable_and_unique() -> None:
    assert new_run_id() != new_run_id()


def test_real_cell_fails_when_harbor_reports_trial_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = new_run_id()
    cell = _cell(run_id, "errored")
    result_path = tmp_path / cell.result_path
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps({"stats": {"n_errored_trials": 1, "n_cancelled_trials": 0}})
    )
    monkeypatch.setattr("fugue.bench.execution._run_cell_process", lambda *args: 0)

    [outcome] = execute_cells([cell], repo_root=tmp_path, max_workers=1)

    assert outcome.status == "failed"
    assert outcome.error == "1 Harbor trial(s) errored"


def test_execution_rejects_mixed_runs_and_duplicate_cells(tmp_path: Path) -> None:
    first = _cell("run-a", "same")
    second_run = _cell("run-b", "other")
    with pytest.raises(ValueError, match="share a run_id"):
        execute_cells([first, second_run], repo_root=tmp_path, max_workers=1)
    with pytest.raises(ValueError, match="cell ids must be unique"):
        execute_cells([first, first], repo_root=tmp_path, max_workers=1)


def test_cell_lifecycle_overlays_env_without_changing_outcome(tmp_path: Path) -> None:
    cell = _cell("run-live", "live")
    observed = {}
    finished = []

    def runner(command, **kwargs):
        observed.update(kwargs["env"])
        return SimpleNamespace(returncode=0)

    outcomes = execute_cells(
        [cell],
        repo_root=tmp_path,
        max_workers=1,
        runner=runner,
        cell_started=lambda value: {"FUGUE_WEAVE_EVAL_NAME": value.id},
        cell_finished=lambda value, outcome: finished.append(
            (value.id, outcome.status)
        ),
    )

    assert outcomes[0].status == "passed"
    assert observed["FUGUE_WEAVE_EVAL_NAME"] == cell.id
    assert finished == [(cell.id, "passed")]


def test_cancellation_terminates_active_process_and_never_opens_queued_cell(
    tmp_path: Path,
) -> None:
    cancellation = threading.Event()
    opened: list[str] = []
    cells = [
        replace(
            _cell("run-cancel", name),
            command=(sys.executable, "-c", "import time; time.sleep(30)"),
        )
        for name in ("active", "queued")
    ]
    result: list = []

    def execute() -> None:
        result.extend(
            execute_cells(
                cells,
                repo_root=tmp_path,
                max_workers=1,
                cell_started=lambda cell: opened.append(cell.id) or None,
                cancellation_event=cancellation,
            )
        )

    worker = threading.Thread(target=execute)
    worker.start()
    state_path = tmp_path / ".fugue/runtime/run-cancel/cells.jsonl"
    process_group = None
    for _ in range(100):
        latest = {row["cell_id"]: row for row in latest_cell_records(state_path)}
        process_group = latest.get("cell-active", {}).get("harbor_process_group")
        if process_group:
            break
        time.sleep(0.02)
    assert isinstance(process_group, int)

    cancellation.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert {outcome.cell_id: outcome.status for outcome in result} == {
        "cell-active": "cancelled",
        "cell-queued": "cancelled",
    }
    assert opened == ["cell-active"]
    latest = {row["cell_id"]: row for row in latest_cell_records(state_path)}
    assert {row["status"] for row in latest.values()} == {"cancelled"}
    with pytest.raises(ProcessLookupError):
        os.killpg(process_group, 0)


def test_concurrent_run_manifest_updates_are_atomic_and_merged(tmp_path: Path) -> None:
    barrier = threading.Barrier(3)

    def update(values):
        barrier.wait()
        write_run_manifest(tmp_path, "run-atomic", values)

    first = threading.Thread(target=update, args=({"pid": 123},))
    second = threading.Thread(target=update, args=({"trace_project": "team/project"},))
    first.start()
    second.start()
    barrier.wait()
    first.join()
    second.join()

    manifest = read_run_manifest(tmp_path / ".fugue/runtime/run-atomic")
    assert manifest is not None
    assert manifest["pid"] == 123
    assert manifest["trace_project"] == "team/project"


def test_update_run_manifest_merges_partial_updater_result(tmp_path: Path) -> None:
    write_run_manifest(
        tmp_path,
        "run-update",
        {
            "status": "running",
            "pid": 123,
            "evaluation_failures": ["existing failure"],
        },
    )

    update_run_manifest(
        tmp_path,
        "run-update",
        lambda manifest: {
            "evaluation_runs": [
                {"candidate_id": "candidate-a", "name": "evaluation-a"}
            ]
        },
    )

    manifest = read_run_manifest(tmp_path / ".fugue/runtime/run-update")
    assert manifest is not None
    assert manifest["status"] == "running"
    assert manifest["pid"] == 123
    assert manifest["evaluation_failures"] == ["existing failure"]
    assert manifest["evaluation_runs"][0]["candidate_id"] == "candidate-a"
