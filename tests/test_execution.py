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
    schedule_cells,
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
    assert all(
        len(item["attempt_id"]) == 64
        and item["attempt_identity"]["candidate"] == "candidate-codex-baseline"
        for item in latest.values()
    )
    assert latest["cell-fail"]["status"] == "failed"
    assert latest["cell-fail"]["applicable"] is True
    assert latest["cell-skip"]["status"] == "not_applicable"
    assert latest["cell-skip"]["applicable"] is False
    assert latest["cell-skip"]["skip_reason"] == "unsupported"

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


def test_seeded_scheduling_is_reproducible_and_run_independent() -> None:
    first = [_cell("run-a", name) for name in ("one", "two", "three", "four")]
    second = [_cell("run-b", name) for name in ("four", "three", "two", "one")]

    first_order = [cell.task_id for cell in schedule_cells(first, "study-v1")]
    second_order = [cell.task_id for cell in schedule_cells(second, "study-v1")]

    assert first_order == second_order
    assert schedule_cells(first, None) is first


def test_seeded_scheduling_blocks_attempts_and_counterbalances_arms() -> None:
    cells: list[PlannedCell] = []
    for task_index in range(4):
        for attempt in (1, 2):
            task_id = f"task-{task_index}"
            for variant in ("baseline", "candidate"):
                name = f"{task_index}-{attempt}-{variant}"
                cells.append(
                    replace(
                        _cell("run-a", name),
                        id=f"cell-{name}",
                        task_id=task_id,
                        trial_index=attempt,
                        variant_id=variant,
                        candidate_id=f"candidate-{variant}",
                        execution_fingerprint=f"execution-{variant}",
                    )
                )

    scheduled = schedule_cells(cells, "confirmatory-v1")
    reversed_input = schedule_cells(list(reversed(cells)), "confirmatory-v1")
    identities = [
        (cell.task_id, cell.trial_index, cell.variant_id) for cell in scheduled
    ]
    assert identities == [
        (cell.task_id, cell.trial_index, cell.variant_id)
        for cell in reversed_input
    ]

    blocks = [scheduled[index : index + 2] for index in range(0, len(cells), 2)]
    assert all(
        len({(cell.task_id, cell.trial_index) for cell in block}) == 1
        and {cell.variant_id for cell in block} == {"baseline", "candidate"}
        for block in blocks
    )
    first_arms = [block[0].variant_id for block in blocks]
    assert first_arms == ["baseline", "candidate"] * 4


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
    monkeypatch.setattr(
        "fugue.bench.execution._run_cell_process", lambda *args, **kwargs: 0
    )

    [outcome] = execute_cells([cell], repo_root=tmp_path, max_workers=1)

    assert outcome.status == "failed"
    assert outcome.error == "1 Harbor trial(s) errored"
    assert outcome.benchmark_outcome == "unscored"


@pytest.mark.parametrize(
    ("reward", "expected"),
    ((1.0, "passed"), (0.0, "failed"), (0.8, "failed"), (None, "unscored")),
)
def test_harbor_benchmark_outcome_is_separate_from_execution_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reward: float | None,
    expected: str,
) -> None:
    run_id = new_run_id()
    cell = _cell(run_id, "reward")
    result_path = tmp_path / cell.result_path
    result_path.parent.mkdir(parents=True)
    reward_stats = {} if reward is None else {str(reward): ["trial-one"]}
    result_path.write_text(
        json.dumps(
            {
                "stats": {
                    "n_errored_trials": 0,
                    "n_cancelled_trials": 0,
                    "evals": {
                        "agent__model__dataset": {
                            "reward_stats": {"reward": reward_stats}
                        }
                    },
                }
            }
        )
    )
    monkeypatch.setattr(
        "fugue.bench.execution._run_cell_process", lambda *args, **kwargs: 0
    )

    [outcome] = execute_cells([cell], repo_root=tmp_path, max_workers=1)

    assert outcome.status == "passed"
    assert outcome.benchmark_outcome == expected
    assert outcome.reward == reward
    [record] = latest_cell_records(
        tmp_path / ".fugue" / "runtime" / run_id / "cells.jsonl"
    )
    assert record["benchmark_outcome"] == expected
    assert record["reward"] == reward


def test_deterministic_benchmark_failure_remains_evidence_and_allows_next_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "run-benchmark-failure"
    failed_task = _cell(run_id, "deterministic-failure")
    passing_task = _cell(run_id, "passing-task")
    for cell, reward in ((failed_task, 0.0), (passing_task, 1.0)):
        path = tmp_path / cell.result_path
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "stats": {
                        "n_errored_trials": 0,
                        "n_cancelled_trials": 0,
                        "evals": {
                            "agent__model__dataset": {
                                "reward_stats": {"reward": {str(reward): ["trial"]}}
                            }
                        },
                    }
                }
            )
        )
    started: list[str] = []

    def run_cell(cell, *args, **kwargs):
        started.append(cell.id)
        return 0

    monkeypatch.setattr("fugue.bench.execution._run_cell_process", run_cell)
    internal_abort = threading.Event()

    outcomes = execute_cells(
        [failed_task, passing_task],
        repo_root=tmp_path,
        max_workers=1,
        internal_abort_event=internal_abort,
    )

    assert started == [failed_task.id, passing_task.id]
    assert internal_abort.is_set() is False
    assert {
        outcome.cell_id: (outcome.status, outcome.benchmark_outcome)
        for outcome in outcomes
    } == {
        failed_task.id: ("passed", "failed"),
        passing_task.id: ("passed", "passed"),
    }


def test_infrastructure_collection_failure_aborts_queued_paid_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "run-infrastructure-failure"
    failed_cell = _cell(run_id, "broken-log-reader")
    queued_cell = _cell(run_id, "must-not-start")
    started: list[str] = []
    finished: list[tuple[str, str]] = []

    def run_cell(cell, *args, **kwargs):
        started.append(cell.id)
        if cell.id == failed_cell.id:
            raise RuntimeError("cell log reader failed: [Errno 32] Broken pipe")
        return 0

    monkeypatch.setattr("fugue.bench.execution._run_cell_process", run_cell)
    internal_abort = threading.Event()

    outcomes = execute_cells(
        [failed_cell, queued_cell],
        repo_root=tmp_path,
        max_workers=1,
        cell_finished=lambda cell, outcome: finished.append(
            (cell.id, outcome.status)
        ),
        internal_abort_event=internal_abort,
    )

    assert started == [failed_cell.id]
    assert finished == [(failed_cell.id, "failed")]
    assert internal_abort.is_set() is True
    assert {outcome.cell_id: outcome.status for outcome in outcomes} == {
        failed_cell.id: "failed",
        queued_cell.id: "cancelled",
    }
    records = {
        row["cell_id"]: row
        for row in latest_cell_records(
            tmp_path / ".fugue/runtime/run-infrastructure-failure/cells.jsonl"
        )
    }
    assert records[failed_cell.id]["error"] == (
        "RuntimeError: cell log reader failed: [Errno 32] Broken pipe"
    )
    assert records[queued_cell.id]["runtime_outcome"] == "not_started"
    assert records[queued_cell.id]["cancellation_origin"] == "internal"
    events = [
        json.loads(line)
        for line in (
            tmp_path / ".fugue/runtime/run-infrastructure-failure/events.jsonl"
        ).read_text().splitlines()
    ]
    [abort] = [event for event in events if event["event"] == "run_abort_requested"]
    assert abort["cell_id"] == failed_cell.id
    assert abort["cancellation_origin"] == "internal"


def test_provider_diagnostic_does_not_require_a_harbor_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = new_run_id()
    cell = replace(
        _cell(run_id, "diagnostic"),
        execution_kind="provider_diagnostic",
        harness="direct",
    )
    monkeypatch.setattr(
        "fugue.bench.execution._run_cell_process", lambda *args, **kwargs: 0
    )

    [outcome] = execute_cells([cell], repo_root=tmp_path, max_workers=1)

    assert outcome.status == "passed"
    assert outcome.benchmark_outcome == "unscored"
    assert outcome.error is None


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


def test_required_live_evidence_start_failure_prevents_agent_execution(
    tmp_path: Path,
) -> None:
    cell = _cell("run-live-required", "blocked")
    runner_calls: list[list[str]] = []

    def runner(command, **kwargs):
        runner_calls.append(command)
        return SimpleNamespace(returncode=0)

    [outcome] = execute_cells(
        [cell],
        repo_root=tmp_path,
        max_workers=1,
        runner=runner,
        cell_started=lambda _: (_ for _ in ()).throw(
            RuntimeError("bridge unavailable")
        ),
        require_cell_started_success=True,
    )

    assert outcome.status == "failed"
    assert "required live-evidence initialization failed" in str(outcome.error)
    assert runner_calls == []
    state_path = (
        tmp_path
        / ".fugue/runtime/run-live-required/cells.jsonl"
    )
    [latest] = latest_cell_records(state_path)
    assert latest["status"] == "failed"


def test_cancellation_terminates_active_process_and_never_opens_queued_cell(
    tmp_path: Path,
) -> None:
    cancellation = threading.Event()
    opened: list[str] = []
    cells = [
        replace(
            _cell("run-cancel", name),
            command=(sys.executable, "-c", "import time; time.sleep(30)"),
            execution_kind="provider_diagnostic",
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


def test_outer_wall_timeout_is_runtime_failure_not_task_outcome(
    tmp_path: Path,
) -> None:
    cell = replace(
        _cell("run-timeout", "timed"),
        command=(sys.executable, "-c", "import time; time.sleep(30)"),
        execution_kind="provider_diagnostic",
        outer_wall_time_sec=1,
        execution_limits_digest="a" * 64,
    )

    started = time.monotonic()
    [outcome] = execute_cells([cell], repo_root=tmp_path, max_workers=1)

    assert time.monotonic() - started < 5
    assert outcome.status == "failed"
    assert outcome.runtime_outcome == "timed_out"
    assert outcome.benchmark_outcome == "unscored"
    assert outcome.reward is None
    assert outcome.error == "outer cell wall-time limit exceeded after 1 seconds"
    [record] = latest_cell_records(
        tmp_path / ".fugue/runtime/run-timeout/cells.jsonl"
    )
    assert record["runtime_outcome"] == "timed_out"
    assert record["benchmark_outcome"] == "unscored"
    assert record["outer_wall_time_sec"] == 1
    assert record["execution_limits_digest"] == "a" * 64


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
            "evaluation_runs": [{"candidate_id": "candidate-a", "name": "evaluation-a"}]
        },
    )

    manifest = read_run_manifest(tmp_path / ".fugue/runtime/run-update")
    assert manifest is not None
    assert manifest["status"] == "running"
    assert manifest["pid"] == 123
    assert manifest["evaluation_failures"] == ["existing failure"]
    assert manifest["evaluation_runs"][0]["candidate_id"] == "candidate-a"
