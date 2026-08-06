from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from fugue.bench.campaign_accounting import (
    BudgetAdmissionPaused,
    BudgetLeaseLedger,
    BudgetSettlementExceeded,
    measured_row_cost,
    settle_execution_budget_leases,
)
from fugue.bench.comparison import (
    ComparisonPreviewV1,
    _artifact_digest,
    _comparison_execution_schedule,
    _execution_schedule_config,
    _staged_run_gate,
    _StagedFinalizationPending,
    analyze_comparison_rows,
    comparison_stage_receipt,
    execute_comparison_stage,
    load_comparison,
    materialize_comparison,
    preview_comparison,
    scaffold_comparison,
)
from fugue.bench.execution import (
    ExecutionScheduleV1,
    PhysicalExecutionIdentityV1,
    PlannedCell,
    StageSubsetReceiptV1,
    _CellWallTimeExceeded,
    execute_cells,
    latest_cell_records,
)
from fugue.bench.host_capacity import (
    HostCapacityObservationV1,
    HostCapacityReceiptV1,
)
from fugue.bench.operator import (
    PreviewCellSummary,
    PreviewSummary,
    _reconcile_resume_budget_leases,
)
from fugue.research.approvals import ApprovalLedger
from fugue.research.store import StudyStore


def _cell(root: Path, index: int, *, run_id: str = "staged-run") -> PlannedCell:
    digest = f"{index + 1:064x}"
    return PlannedCell(
        id=f"cell-{index:03d}",
        run_id=run_id,
        run_name="staged",
        workload_id="harbor",
        task_id=f"task-{index:03d}",
        harness="claude-code",
        context_system_id="none",
        variant_id="baseline" if index % 2 == 0 else "candidate",
        model_provider="anthropic",
        model="sonnet",
        trial_index=1,
        comparison_example_id=digest,
        candidate_id=digest,
        execution_fingerprint=f"{index + 193:064x}",
        config_path=root / f"config-{index}.yaml",
        result_path=root / f"result-{index}.json",
        command=("fake-runner",),
        env={},
        n_attempts=1,
    )


def test_192_cell_schedule_is_exact_and_digest_bound() -> None:
    logical = [f"{index:064x}" for index in range(1, 193)]
    schedule = ExecutionScheduleV1.create(
        stages={
            "checkpoint": logical[:4],
            "development-attempt-0": logical[4:8],
            "development-repeat": logical[8:96],
            "holdout": logical[96:],
        },
        stage_cost_usd={
            "checkpoint": 4,
            "development-attempt-0": 4,
            "development-repeat": 88,
            "holdout": 96,
        },
        worker_limit=3,
        wave_size=4,
        maximum_physical_executions=196,
        infrastructure_retry_limit=4,
        total_cost_usd=200,
        maximum_in_flight_cost_usd=7.5,
        coordination={
            "group_id": "community-skill-campaign",
            "worker_limit": 3,
            "maximum_physical_executions": 196,
            "total_cost_usd": 200.0,
            "maximum_in_flight_cost_usd": 7.5,
        },
    )

    assert ExecutionScheduleV1.from_dict(schedule.to_dict()) == schedule
    assert len(schedule.logical_attempt_ids) == 192
    assert len(set(schedule.logical_attempt_ids)) == 192
    subset = StageSubsetReceiptV1.create(
        schedule,
        preview_digest="a" * 64,
        stage_id="checkpoint",
        maximum_cost_usd=schedule.stage_cost_usd["checkpoint"],
    )
    assert subset.maximum_cells == 4
    assert subset.maximum_cost_usd == 4

    governed = comparison_stage_receipt(
        SimpleNamespace(
            execution_schedule=schedule.to_dict(),
            preview_digest="b" * 64,
        ),  # type: ignore[arg-type]
        "checkpoint",
    )
    assert governed.maximum_cells == 4
    assert governed.maximum_cost_usd == 5


def test_controller_resume_never_reexecutes_canonical_cells(tmp_path: Path) -> None:
    cells = [_cell(tmp_path, index) for index in range(192)]
    calls: list[str] = []

    def runner(*_args: object, env: dict[str, str], **_kwargs: object) -> object:
        calls.append(env["FUGUE_PHYSICAL_EXECUTION_ID"])
        return SimpleNamespace(returncode=0)

    first = execute_cells(
        cells[:96], repo_root=tmp_path, max_workers=3, runner=runner
    )
    resumed = execute_cells(
        cells,
        repo_root=tmp_path,
        max_workers=3,
        runner=runner,
        resume=True,
    )

    assert len(first) == 96
    assert len(resumed) == 192
    assert len(calls) == 192
    assert len(set(calls)) == 192
    assert len(
        latest_cell_records(tmp_path / ".fugue/runtime/staged-run/cells.jsonl")
    ) == 192


@pytest.mark.parametrize(
    ("failure", "retries", "expected_calls", "terminal_kind", "retry_ordinal"),
    [
        (OSError("runner unavailable"), 1, 2, "success", 1),
        (_CellWallTimeExceeded("bounded timeout"), 3, 1, "agent_timeout", 0),
    ],
)
def test_only_pre_start_infrastructure_failure_is_replaced(
    tmp_path: Path,
    failure: Exception,
    retries: int,
    expected_calls: int,
    terminal_kind: str,
    retry_ordinal: int,
) -> None:
    calls: list[str] = []

    def runner(*_args: object, env: dict[str, str], **_kwargs: object) -> object:
        calls.append(env["FUGUE_PHYSICAL_EXECUTION_ID"])
        if len(calls) == 1:
            raise failure
        return SimpleNamespace(returncode=0)

    outcome = execute_cells(
        [_cell(tmp_path, 0)],
        repo_root=tmp_path,
        max_workers=1,
        runner=runner,
        infrastructure_retries=retries,
    )[0]

    assert len(calls) == expected_calls
    assert len(set(calls)) == expected_calls
    assert outcome.terminal_kind == terminal_kind
    assert outcome.retry_ordinal == retry_ordinal


def test_fatal_gate_stops_later_admissions(tmp_path: Path) -> None:
    cells = [_cell(tmp_path, index) for index in range(3)]
    calls = 0

    def runner(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise OSError("runner unavailable")

    outcomes = execute_cells(
        cells,
        repo_root=tmp_path,
        max_workers=1,
        runner=runner,
        abort_on_terminal_kinds=frozenset({"runner_start_failure"}),
    )

    assert calls == 1
    assert [item.terminal_kind for item in outcomes] == [
        "runner_start_failure",
        "admission_aborted",
        "admission_aborted",
    ]


def test_required_link_failure_is_a_fatal_stage_gate(tmp_path: Path) -> None:
    run_id = "required-link-failure"
    outcomes = execute_cells(
        [_cell(tmp_path, 0, run_id=run_id), _cell(tmp_path, 1, run_id=run_id)],
        repo_root=tmp_path,
        max_workers=1,
        runner=lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
        cell_finished=lambda *_args: (_ for _ in ()).throw(
            RuntimeError("five-link evidence set is missing the Agent root")
        ),
        require_cell_started_success=True,
        abort_on_terminal_kinds=frozenset({"evidence_failure"}),
    )
    fatal, pending = _staged_run_gate(
        SimpleNamespace(run_id=run_id, status="failed", evaluation_failures=()),
        repo_root=tmp_path,
    )

    assert [outcome.terminal_kind for outcome in outcomes] == [
        "evidence_failure",
        "admission_aborted",
    ]
    assert fatal == [
        "cell-000: fatal evidence mismatch",
        "cell-001: admission_aborted",
    ]
    assert pending == []


@pytest.mark.parametrize(
    ("failure", "terminal_kind"),
    [
        (SimpleNamespace(returncode=1), "task_failure"),
        (_CellWallTimeExceeded("bounded timeout"), "agent_timeout"),
    ],
)
def test_behavioral_failure_does_not_block_later_stage(
    tmp_path: Path, failure: object, terminal_kind: str
) -> None:
    run_id = f"behavioral-{terminal_kind}"

    def runner(*_args: object, **_kwargs: object) -> object:
        if isinstance(failure, Exception):
            raise failure
        return failure

    outcome = execute_cells(
        [_cell(tmp_path, 0, run_id=run_id)],
        repo_root=tmp_path,
        max_workers=1,
        runner=runner,
    )[0]
    fatal, pending = _staged_run_gate(
        SimpleNamespace(
            run_id=run_id,
            status="failed",
            evaluation_failures=(),
        ),
        repo_root=tmp_path,
    )

    assert outcome.terminal_kind == terminal_kind
    assert fatal == []
    assert pending == []


def test_missing_cost_pauses_next_budget_lease(tmp_path: Path) -> None:
    ledger = BudgetLeaseLedger(
        tmp_path / "leases.json",
        maximum_total_cost_usd=10,
        maximum_in_flight_cost_usd=5,
        maximum_in_flight_executions=2,
        maximum_physical_executions=4,
    )
    ledger.acquire("physical-a", reserved_cost_usd=2.5)
    ledger.settle("physical-a", actual_cost_usd=None)

    with pytest.raises(RuntimeError, match="missing cost pauses"):
        ledger.acquire("physical-b", reserved_cost_usd=2.5)


def test_budget_settlement_records_and_blocks_ceiling_overspend(
    tmp_path: Path,
) -> None:
    ledgers = tuple(
        BudgetLeaseLedger(
            tmp_path / f"overspend-{index}.json",
            maximum_total_cost_usd=3,
            maximum_in_flight_cost_usd=3,
            maximum_in_flight_executions=2,
            maximum_physical_executions=3,
        )
        for index in range(2)
    )
    for ledger in ledgers:
        ledger.acquire("physical-a", reserved_cost_usd=1.5)
        ledger.settle("physical-a", actual_cost_usd=1.5)
        ledger.acquire("physical-b", reserved_cost_usd=1.5)

    with pytest.raises(
        BudgetSettlementExceeded,
        match="per-execution reserve.*total cost ceiling",
    ):
        settle_execution_budget_leases(
            ledgers,
            physical_execution_id="physical-b",
            terminal_kind="success",
            runtime_outcome="completed",
            actual_cost_usd=1.6,
        )

    for ledger in ledgers:
        leases = ledger.snapshot()["leases"]
        assert leases["physical-b"]["status"] == "overspent"
        assert leases["physical-b"]["actual_cost_usd"] == pytest.approx(1.6)
        with pytest.raises(BudgetSettlementExceeded):
            ledger.acquire("physical-c", reserved_cost_usd=0)


def test_verified_controller_recovery_uses_fresh_physical_identity_and_lease(
    tmp_path: Path,
) -> None:
    run_id = "verified-controller-recovery"
    original = _cell(tmp_path, 0, run_id=run_id)
    original_physical = PhysicalExecutionIdentityV1.create(
        logical_attempt_id=original.attempt_id,
        run_id=run_id,
        retry_ordinal=0,
    )
    interrupted = replace(
        original,
        physical_execution_id=original_physical.physical_execution_id,
        retry_ordinal=0,
    ).record(
        "interrupted",
        runtime_outcome="completed",
        terminal_kind="controller_interrupted",
        physical_execution_terminal_verified=True,
        physical_execution_cleanup_verified=True,
        physical_execution_actual_cost_usd=0.4,
    )
    cells_path = tmp_path / ".fugue/runtime" / run_id / "cells.jsonl"
    cells_path.parent.mkdir(parents=True)
    cells_path.write_text(json.dumps(interrupted, sort_keys=True) + "\n")
    ledger = BudgetLeaseLedger(
        tmp_path / "recovery-leases.json",
        maximum_total_cost_usd=2,
        maximum_in_flight_cost_usd=1,
        maximum_in_flight_executions=1,
        maximum_physical_executions=2,
    )
    ledger.acquire(original_physical.physical_execution_id, reserved_cost_usd=1)
    _reconcile_resume_budget_leases(
        (ledger,),
        records=latest_cell_records(cells_path),
        result_rows=(),
    )

    launched: list[str] = []

    def admit(_cell: PlannedCell, physical: object) -> dict[str, str]:
        physical_id = physical.physical_execution_id  # type: ignore[attr-defined]
        launched.append(physical_id)
        ledger.acquire(physical_id, reserved_cost_usd=1)
        return {}

    def settle(_cell: PlannedCell, physical: object, _outcome: object) -> None:
        ledger.settle(
            physical.physical_execution_id,  # type: ignore[attr-defined]
            actual_cost_usd=0.5,
        )

    outcome = execute_cells(
        [original],
        repo_root=tmp_path,
        max_workers=1,
        runner=lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
        physical_started=admit,
        physical_finished=settle,
        resume=True,
    )[0]

    assert outcome.retry_ordinal == 1
    assert outcome.physical_execution_id == launched[0]
    assert outcome.physical_execution_id != original_physical.physical_execution_id
    leases = ledger.snapshot()["leases"]
    assert leases[original_physical.physical_execution_id]["actual_cost_usd"] == 0.4
    assert leases[outcome.physical_execution_id]["actual_cost_usd"] == 0.5
    history = [json.loads(line) for line in cells_path.read_text().splitlines()]
    assert any(
        row.get("physical_execution_id") == original_physical.physical_execution_id
        and row.get("status") == "interrupted"
        for row in history
    )


def test_ambiguous_controller_loss_blocks_resume_before_admission(
    tmp_path: Path,
) -> None:
    run_id = "ambiguous-controller-loss"
    original = _cell(tmp_path, 0, run_id=run_id)
    physical = PhysicalExecutionIdentityV1.create(
        logical_attempt_id=original.attempt_id,
        run_id=run_id,
        retry_ordinal=0,
    )
    running = replace(
        original,
        physical_execution_id=physical.physical_execution_id,
    ).record("running")
    cells_path = tmp_path / ".fugue/runtime" / run_id / "cells.jsonl"
    cells_path.parent.mkdir(parents=True)
    cells_path.write_text(json.dumps(running, sort_keys=True) + "\n")

    with pytest.raises(RuntimeError, match="not proven terminal and cleaned"):
        execute_cells(
            [original],
            repo_root=tmp_path,
            max_workers=1,
            runner=lambda *_args, **_kwargs: pytest.fail("must not relaunch"),
            resume=True,
        )


def test_prestart_failure_retains_zero_cost_budget_record_for_replacement(
    tmp_path: Path,
) -> None:
    ledger = BudgetLeaseLedger(
        tmp_path / "leases.json",
        maximum_total_cost_usd=5,
        maximum_in_flight_cost_usd=2.5,
        maximum_in_flight_executions=1,
        maximum_physical_executions=2,
    )
    ledger.acquire("physical-0", reserved_cost_usd=2.5)

    settle_execution_budget_leases(
        (ledger,),
        physical_execution_id="physical-0",
        terminal_kind="runner_start_failure",
        runtime_outcome="not_started",
        actual_cost_usd=None,
    )

    replacement = ledger.acquire("physical-1", reserved_cost_usd=2.5)
    assert replacement.status == "active"
    leases = ledger.snapshot()["leases"]
    assert set(leases) == {"physical-0", "physical-1"}
    assert leases["physical-0"]["status"] == "settled"
    assert leases["physical-0"]["actual_cost_usd"] == 0.0


def test_physical_leases_finalize_once_across_retry_evidence_start_and_cancel(
    tmp_path: Path,
) -> None:
    ledger = BudgetLeaseLedger(
        tmp_path / "integrated-leases.json",
        maximum_total_cost_usd=10,
        maximum_in_flight_cost_usd=2.5,
        maximum_in_flight_executions=1,
        maximum_physical_executions=4,
    )
    finalized_ids: list[str] = []

    def admitted(_cell: PlannedCell, physical: object) -> dict[str, str]:
        ledger.acquire(
            physical.physical_execution_id,  # type: ignore[attr-defined]
            reserved_cost_usd=2.5,
        )
        return {}

    def finish_physical(
        _cell: PlannedCell, physical: object, outcome: object
    ) -> None:
        physical_id = physical.physical_execution_id  # type: ignore[attr-defined]
        finalized_ids.append(physical_id)
        settle_execution_budget_leases(
            (ledger,),
            physical_execution_id=physical_id,
            terminal_kind=outcome.terminal_kind,  # type: ignore[attr-defined]
            runtime_outcome=outcome.runtime_outcome,  # type: ignore[attr-defined]
            actual_cost_usd=(
                0.5 if outcome.terminal_kind == "success" else None  # type: ignore[attr-defined]
            ),
        )

    runner_calls = 0

    def retrying_runner(*_args: object, **_kwargs: object) -> object:
        nonlocal runner_calls
        runner_calls += 1
        if runner_calls == 1:
            raise OSError("pre-start failure")
        return SimpleNamespace(returncode=0)

    retried = execute_cells(
        [_cell(tmp_path, 0, run_id="lease-retry")],
        repo_root=tmp_path,
        max_workers=1,
        runner=retrying_runner,
        infrastructure_retries=1,
        physical_started=admitted,
        physical_finished=finish_physical,
    )
    assert retried[0].terminal_kind == "success"

    evidence = execute_cells(
        [_cell(tmp_path, 1, run_id="lease-evidence")],
        repo_root=tmp_path,
        max_workers=1,
        runner=lambda *_args, **_kwargs: pytest.fail("runner must not start"),
        cell_started=lambda _cell: (_ for _ in ()).throw(
            RuntimeError("evidence destination unavailable")
        ),
        require_cell_started_success=True,
        physical_started=admitted,
        physical_finished=finish_physical,
    )
    assert evidence[0].terminal_kind == "evidence_initialization_failure"

    cancellation = threading.Event()

    def cancel_after_admission(_cell: PlannedCell) -> dict[str, str]:
        cancellation.set()
        return {}

    cancelled = execute_cells(
        [_cell(tmp_path, 2, run_id="lease-cancel")],
        repo_root=tmp_path,
        max_workers=1,
        runner=lambda *_args, **_kwargs: pytest.fail("runner must not start"),
        cell_started=cancel_after_admission,
        require_cell_started_success=True,
        cancellation_event=cancellation,
        physical_started=admitted,
        physical_finished=finish_physical,
    )
    assert cancelled[0].terminal_kind == "prestart_cancelled"

    leases = ledger.snapshot()["leases"]
    assert len(leases) == 4
    assert len(finalized_ids) == 4
    assert len(set(finalized_ids)) == 4
    assert all(value["status"] == "settled" for value in leases.values())
    assert sorted(value["actual_cost_usd"] for value in leases.values()) == [
        0.0,
        0.0,
        0.0,
        0.5,
    ]


def test_final_wave_secret_aborts_later_admission_and_cleanup_has_zero_orphans(
    tmp_path: Path,
) -> None:
    cells = [
        replace(
            _cell(tmp_path, index, run_id="final-wave-secret"),
            command=("fake-runner", f"cell-{index:03d}"),
        )
        for index in range(5)
    ]
    live_containers: set[str] = set()
    started: list[str] = []

    def runner(command: list[str], *, env: dict[str, str], **_kwargs: object) -> object:
        del env
        cell_id = command[1]
        live_containers.add(cell_id)
        started.append(cell_id)
        return SimpleNamespace(returncode=0)

    def validate_and_cleanup(cell: PlannedCell, _outcome: object) -> None:
        # Cleanup remains mandatory even when the full post-cell privacy scan
        # discovers protected content in the last admitted batch.
        live_containers.discard(cell.id)
        if cell.id == "cell-002":
            raise RuntimeError("privacy scan found a final-wave secret")

    outcomes = execute_cells(
        cells,
        repo_root=tmp_path,
        max_workers=2,
        wave_size=4,
        runner=runner,
        cell_finished=validate_and_cleanup,
        require_cell_started_success=True,
        abort_on_terminal_kinds=frozenset({"evidence_failure"}),
    )
    by_cell = {outcome.cell_id: outcome for outcome in outcomes}

    assert by_cell["cell-002"].terminal_kind == "evidence_failure"
    assert by_cell["cell-004"].terminal_kind == "admission_aborted"
    assert "cell-004" not in started
    assert live_containers == set()


def test_budget_settlement_adds_reserved_unobserved_judge_cost() -> None:
    assert measured_row_cost(
        {
            "cost_usd": 1.25,
            "comparison_judge_accounted_cost_usd": 0.1,
            "comparison_judge_cost_status": "reserved_unobserved",
        }
    ) == pytest.approx(1.35)
    assert measured_row_cost(
        {
            "cost_usd": None,
            "comparison_judge_accounted_cost_usd": 0.1,
            "comparison_judge_cost_status": "reserved_unobserved",
        }
    ) is None


def test_shared_coordination_ledger_caps_concurrent_studies(tmp_path: Path) -> None:
    path = tmp_path / "community-skill-campaign.json"
    kwargs = {
        "maximum_total_cost_usd": 20,
        "maximum_in_flight_cost_usd": 7.5,
        "maximum_in_flight_executions": 3,
        "maximum_physical_executions": 12,
    }
    study_a = BudgetLeaseLedger(path, **kwargs)
    study_b = BudgetLeaseLedger(path, **kwargs)
    study_a.acquire("study-a-1", reserved_cost_usd=2.5)
    study_a.acquire("study-a-2", reserved_cost_usd=2.5)
    study_b.acquire("study-b-1", reserved_cost_usd=2.5)

    with pytest.raises(BudgetAdmissionPaused, match="in-flight execution"):
        study_b.acquire("study-b-2", reserved_cost_usd=2.5)

    study_a.settle("study-a-1", actual_cost_usd=1.0)
    study_b.acquire("study-b-2", reserved_cost_usd=2.5)


def test_paused_admission_resumes_without_skipping_unstarted_cells(
    tmp_path: Path,
) -> None:
    cells = [_cell(tmp_path, index, run_id="paused-run") for index in range(3)]
    runner_calls: list[str] = []
    admissions = 0

    def runner(*_args: object, env: dict[str, str], **_kwargs: object) -> object:
        runner_calls.append(env["FUGUE_PHYSICAL_EXECUTION_ID"])
        return SimpleNamespace(returncode=0)

    def pause_second(*_args: object) -> dict[str, str]:
        nonlocal admissions
        admissions += 1
        if admissions == 2:
            raise BudgetAdmissionPaused("authoritative cost is pending")
        return {}

    first = execute_cells(
        cells,
        repo_root=tmp_path,
        max_workers=1,
        wave_size=2,
        runner=runner,
        physical_started=pause_second,
        abort_on_terminal_kinds=frozenset({"admission_paused"}),
    )
    resumed = execute_cells(
        cells,
        repo_root=tmp_path,
        max_workers=1,
        wave_size=2,
        runner=runner,
        physical_started=lambda *_args: {},
        abort_on_terminal_kinds=frozenset({"admission_paused"}),
        resume=True,
    )

    assert [item.terminal_kind for item in first] == ["success", "admission_paused"]
    assert len(resumed) == 3
    assert all(item.status == "passed" for item in resumed)
    assert len(runner_calls) == 3


def test_explicit_zero_based_stage_selectors_cover_attempts_once() -> None:
    raw = _execution_schedule_config(
        {
            "stages": [
                {
                    "id": "checkpoint",
                    "task_ids": ["target"],
                    "trial_indexes": [0],
                    "pair_complete": True,
                },
                {"id": "complete-first", "task_ids": ["control"], "trial_indexes": [0]},
                {"id": "repeat", "partitions": ["discovery"], "trial_indexes": [1]},
            ],
            "worker_limit": 2,
            "wave_size": 4,
            "infrastructure_retry_limit": 1,
            "maximum_in_flight_cost_usd": 5,
            "coordination": {
                "group_id": "community-skill-campaign",
                "worker_limit": 3,
                "maximum_physical_executions": 96,
                "total_cost_usd": 270,
                "maximum_in_flight_cost_usd": 7.5,
            },
        },
        concurrency=2,
        maximum_cost_usd=20,
    )
    assert raw is not None
    cells = tuple(
        PreviewCellSummary(
            harness="claude-code",
            variant_id=arm,
            variant_label=arm,
            context_system_id="none",
            workload_id="harbor",
            task_id=task,
            trial_index=attempt + 1,
            trial_count=2,
            applicable=True,
            attempt_id=f"{index:064x}",
        )
        for index, (task, arm, attempt) in enumerate(
            (
                (task, arm, attempt)
                for attempt in range(2)
                for task in ("target", "control")
                for arm in ("baseline", "candidate")
            ),
            start=1,
        )
    )
    matrix = PreviewSummary(
        cells=8,
        applicable_cells=8,
        estimated_trials=8,
        harnesses=("claude-code",),
        variants=("baseline", "candidate"),
        systems=("none",),
        workloads=("harbor",),
        commands=(),
        matrix_cells=cells,
    )
    spec = SimpleNamespace(
        evaluators=(),
        execution=SimpleNamespace(
            schedule=raw,
            reserve_per_attempt_usd=2.5,
            max_cost_usd=20,
        )
    )
    schedule = _comparison_execution_schedule(
        spec,
        matrix,
        [
            {"id": "target", "partition": "discovery"},
            {"id": "control", "partition": "discovery"},
        ],
        host_capacity_receipt=HostCapacityReceiptV1.from_observation(
            HostCapacityObservationV1(
                cpu_count=8,
                available_memory_gib=16,
                free_disk_gib=30,
            )
        ),
    )

    assert schedule is not None
    assert [len(value) for value in schedule.stages.values()] == [2, 2, 4]
    assert schedule.stage_admission["checkpoint"] == {
        "worker_limit": 1,
        "wave_size": 2,
        "pair_complete": True,
    }
    assert schedule.coordination is not None
    assert schedule.coordination["worker_limit"] == 3
    assert len(set(schedule.logical_attempt_ids)) == 8


def test_pair_complete_schedule_counterbalances_arm_order_and_keeps_pairs_adjacent() -> None:
    tasks = ("alpha", "beta", "gamma", "delta")
    arms = ("baseline", "candidate")
    cells = tuple(
        PreviewCellSummary(
            harness="claude-code",
            variant_id=arm,
            variant_label=arm,
            context_system_id="none",
            workload_id="harbor",
            task_id=task,
            trial_index=1,
            trial_count=1,
            applicable=True,
            attempt_id=f"{index:064x}",
        )
        for index, (task, arm) in enumerate(
            ((task, arm) for task in tasks for arm in arms), start=1
        )
    )
    raw = _execution_schedule_config(
        {
            "stages": [
                {
                    "id": "checkpoint",
                    "task_ids": list(tasks),
                    "trial_indexes": [0],
                    "pair_complete": True,
                }
            ],
            "worker_limit": 2,
            "wave_size": 4,
            "infrastructure_retry_limit": 0,
            "maximum_in_flight_cost_usd": 5,
        },
        concurrency=2,
        maximum_cost_usd=20,
    )
    assert raw is not None
    matrix = PreviewSummary(
        cells=8,
        applicable_cells=8,
        estimated_trials=8,
        harnesses=("claude-code",),
        variants=arms,
        systems=("none",),
        workloads=("harbor",),
        commands=(),
        matrix_cells=cells,
    )
    schedule = _comparison_execution_schedule(
        SimpleNamespace(
            evaluators=(),
            execution=SimpleNamespace(
                schedule=raw,
                reserve_per_attempt_usd=2.5,
                max_cost_usd=20,
            ),
        ),
        matrix,
        [{"id": task, "partition": "discovery"} for task in tasks],
    )
    assert schedule is not None
    by_attempt = {cell.attempt_id: cell for cell in cells}
    ordered = [by_attempt[item] for item in schedule.stages["checkpoint"]]
    pairs = [ordered[offset : offset + 2] for offset in range(0, len(ordered), 2)]
    assert all(pair[0].task_id == pair[1].task_id for pair in pairs)
    assert all({item.variant_id for item in pair} == set(arms) for pair in pairs)
    assert {pair[0].variant_id for pair in pairs} == set(arms)
    assert ExecutionScheduleV1.from_dict(schedule.to_dict()) == schedule


def test_real_staged_materialization_and_analysis_share_one_canonical_lock(
    tmp_path: Path,
) -> None:
    comparison_path = scaffold_comparison(tmp_path / "canonical-staged")
    raw = yaml.safe_load(comparison_path.read_text())
    raw["execution"].update(
        {
            "evidence_project": "wandb/canonical-staged-test",
            "schedule": {
                "stages": [
                    {
                        "id": "checkpoint",
                        "trial_indexes": [0],
                        "pair_complete": True,
                    },
                    {
                        "id": "repeat",
                        "trial_indexes": [1],
                        "pair_complete": True,
                    },
                ],
                "worker_limit": 1,
                "wave_size": 2,
                "infrastructure_retry_limit": 0,
                "maximum_physical_executions": 4,
                "maximum_in_flight_cost_usd": 10,
            },
        }
    )
    comparison_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    root = comparison_path.parent
    spec = load_comparison(comparison_path, repo_root=root)
    preview = preview_comparison(spec, repo_root=root)
    assert list(preview.execution_schedule["stages"]) == ["checkpoint", "repeat"]

    _, checkpoint_request = materialize_comparison(
        preview,
        repo_root=root,
        approval_digest="",
    )
    _, repeat_request = materialize_comparison(
        preview,
        repo_root=root,
        approval_digest="",
    )
    approved = checkpoint_request.approved_comparison
    assert approved == repeat_request.approved_comparison
    assert approved["approval_digest"] == ""

    ledger = ApprovalLedger(StudyStore(root).path)
    approvals = []
    for stage_id in ("checkpoint", "repeat"):
        subset = comparison_stage_receipt(preview, stage_id)
        approvals.append(
            ledger.approve(
                subject_kind="experiment",
                preview_digest=subset.subset_digest,
                maximum_cost_usd=subset.maximum_cost_usd,
                maximum_cells=subset.maximum_cells,
                approved_by="canonical-stage-test",
                operation_id=f"approve-canonical-{stage_id}",
            ).approval_digest
        )
    assert len(set(approvals)) == 2

    rows = [
        {
            **{
                key: cell[key]
                for key in (
                    "attempt_id",
                    "attempt_identity",
                    "task_id",
                    "variant_id",
                    "harness",
                    "trial_index",
                    "candidate_id",
                    "execution_fingerprint",
                    "applicable",
                    "skip_reason",
                )
            },
            "integration_provenance": [],
            "run_id": "canonical-staged-controller",
            "trace_project": approved["evidence_project"],
            "trace_receipt": approved["evidence_destination"],
            "approved_comparison": approved,
            "comparison_required_evaluation_complete": True,
        }
        for cell in approved["expected_cells"]
    ]
    result = analyze_comparison_rows(
        comparison_id=spec.id,
        preview_digest=preview.preview_digest,
        rows=rows,
        source="canonical-staged-controller",
        approved_comparison=repeat_request.approved_comparison,
    )

    assert result.rows == 4
    assert result.integrity["approved_manifest_status"] == "reconciled"
    assert result.integrity["approved_manifest_digest"] == approved["lock_digest"]


def test_staged_controller_runs_192_cells_in_order_and_resumes(  # noqa: PLR0915
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fugue.bench import comparison as comparison_module

    templates = [_cell(tmp_path, index) for index in range(192)]
    logical = [cell.attempt_id for cell in templates]
    schedule = ExecutionScheduleV1.create(
        stages={
            "checkpoint": logical[:4],
            "development-attempt-0": logical[4:8],
            "development-repeat": logical[8:96],
            "holdout": logical[96:],
        },
        stage_cost_usd={
            "checkpoint": 4,
            "development-attempt-0": 4,
            "development-repeat": 88,
            "holdout": 96,
        },
        stage_admission={
            "checkpoint": {
                "worker_limit": 1,
                "wave_size": 2,
                "pair_complete": True,
            },
            "development-attempt-0": {
                "worker_limit": 3,
                "wave_size": 4,
                "pair_complete": False,
            },
            "development-repeat": {
                "worker_limit": 3,
                "wave_size": 4,
                "pair_complete": False,
            },
            "holdout": {
                "worker_limit": 3,
                "wave_size": 4,
                "pair_complete": False,
            },
        },
        worker_limit=3,
        wave_size=4,
        maximum_physical_executions=196,
        infrastructure_retry_limit=4,
        total_cost_usd=200,
        maximum_in_flight_cost_usd=7.5,
    )
    draft = ComparisonPreviewV1(
        schema_version=3,
        comparison={"schema_version": 3, "id": "fake-staged-comparison"},
        readiness={"status": "ready"},
        matrix={"estimated_trials": 192},
        experiment={},
        manifest={},
        execution_schedule=schedule.to_dict(),
    )
    preview = replace(
        draft,
        preview_digest=_artifact_digest(draft.to_dict(), "preview_digest"),
    )
    spec = SimpleNamespace(
        id="fake-staged-comparison",
        schema_version=3,
        evaluators=(),
        decision_policy=None,
        supersedes=(),
        execution=SimpleNamespace(
            reserve_per_attempt_usd=1.0,
            max_cost_usd=200.0,
            evidence_project="wandb/fake-staged-comparison",
            source_evidence_project=None,
        ),
    )
    template_by_attempt = {cell.attempt_id: cell for cell in templates}
    statuses: dict[str, str] = {}
    selected_by_run: dict[str, tuple[str, ...]] = {}
    policy_calls: list[tuple[str, int, int, bool, int | None]] = []
    runner_calls: list[str] = []
    interrupted = False

    class FakeOperatorService:
        def __init__(self, repo_root: Path, _env_file: Path | None = None) -> None:
            self.repo_root = repo_root
            self.env = {}

        def run_summary(self, run_id: str) -> SimpleNamespace:
            if run_id not in statuses:
                raise FileNotFoundError(run_id)
            return SimpleNamespace(
                run_id=run_id,
                status=statuses[run_id],
                evaluation_failures=(),
            )

        def execute_run(
            self,
            _request: object,
            *,
            run_id: str,
            selected_attempt_ids: tuple[str, ...],
            resume: bool,
            execution_worker_limit: int,
            execution_wave_size: int,
            evidence_checkpoint_cells_override: int | None,
            **_kwargs: object,
        ) -> SimpleNamespace:
            nonlocal interrupted
            selected_by_run[run_id] = tuple(selected_attempt_ids)
            stage = next(
                key
                for key, values in schedule.stages.items()
                if tuple(values) == tuple(selected_attempt_ids)
            )
            policy_calls.append(
                (
                    stage,
                    execution_worker_limit,
                    execution_wave_size,
                    resume,
                    evidence_checkpoint_cells_override,
                )
            )
            cells = [
                replace(
                    template_by_attempt[attempt],
                    id=f"{stage}-{index:03d}",
                    run_id=run_id,
                    command=("fake-runner", attempt),
                )
                for index, attempt in enumerate(selected_attempt_ids)
            ]

            def runner(command: list[str], **_runner_kwargs: object) -> object:
                runner_calls.append(command[1])
                time.sleep(0.0001)
                return SimpleNamespace(returncode=0)

            if stage == "development-repeat" and not interrupted:
                execute_cells(
                    cells[:17],
                    repo_root=self.repo_root,
                    max_workers=execution_worker_limit,
                    wave_size=execution_wave_size,
                    runner=runner,
                )
                statuses[run_id] = "interrupted"
                interrupted = True
                raise KeyboardInterrupt
            outcomes = execute_cells(
                cells,
                repo_root=self.repo_root,
                max_workers=execution_worker_limit,
                wave_size=execution_wave_size,
                runner=runner,
                resume=resume,
            )
            statuses[run_id] = (
                "passed" if all(item.status == "passed" for item in outcomes) else "failed"
            )
            return self.run_summary(run_id)

        def export_run(
            self,
            run_id: str,
            *,
            out: Path,
            **_kwargs: object,
        ) -> SimpleNamespace:
            records = latest_cell_records(
                self.repo_root / ".fugue/runtime" / run_id / "cells.jsonl"
            )
            by_attempt = {str(item["attempt_id"]): item for item in records}
            rows = [
                {
                    "attempt_id": attempt,
                    "cell_id": by_attempt[attempt]["cell_id"],
                    "trace_project": "wandb/fake-staged-comparison",
                }
                for attempt in selected_by_run[run_id]
            ]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            return SimpleNamespace(path=out)

    finalize_calls = 0

    def fake_finalize(**_kwargs: object) -> tuple[object, Path, Path]:
        nonlocal finalize_calls
        finalize_calls += 1
        if finalize_calls == 1:
            raise _StagedFinalizationPending("Weave is eventually consistent")
        result_path = tmp_path / "canonical-result.json"
        markdown_path = tmp_path / "canonical-result.md"
        result_path.write_text("{}", encoding="utf-8")
        markdown_path.write_text("complete", encoding="utf-8")
        return SimpleNamespace(result_digest="f" * 64), result_path, markdown_path

    monkeypatch.setattr(comparison_module, "OperatorService", FakeOperatorService)
    monkeypatch.setattr(comparison_module, "comparison_from_dict", lambda *_a, **_k: spec)
    monkeypatch.setattr(comparison_module, "preview_comparison", lambda *_a, **_k: preview)
    monkeypatch.setattr(
        comparison_module,
        "materialize_comparison",
        lambda *_a, **_k: (object(), SimpleNamespace(approved_comparison={})),
    )
    monkeypatch.setattr(comparison_module, "_verify_v3_source_drift", lambda *_a, **_k: None)
    monkeypatch.setattr(comparison_module, "_apply_harbor_conformance", lambda *_a, **_k: None)
    monkeypatch.setattr(comparison_module, "_validate_staged_rows", lambda *_a, **_k: None)
    monkeypatch.setattr(comparison_module, "_finalize_staged_comparison", fake_finalize)
    monkeypatch.setattr(
        comparison_module,
        "trace_project_slug",
        lambda _env: "wandb/fake-staged-comparison",
    )
    monkeypatch.setattr(
        comparison_module,
        "_comparison_evidence_environment",
        lambda *_a, **_k: {},
    )

    stages = list(schedule.stages)
    approval_ledger = ApprovalLedger(StudyStore(tmp_path).path)
    approvals = {
        stage: approval_ledger.approve(
            subject_kind="experiment",
            preview_digest=comparison_stage_receipt(preview, stage).subset_digest,
            maximum_cost_usd=comparison_stage_receipt(
                preview, stage
            ).maximum_cost_usd,
            maximum_cells=comparison_stage_receipt(preview, stage).maximum_cells,
            approved_by="staged-test-operator",
            operation_id=f"approve-{stage}",
        ).approval_digest
        for stage in stages
    }
    fatal_controller = "fatal-controller"
    fatal_state = (
        tmp_path
        / ".fugue/runtime/comparison-stages/fake-staged-comparison"
        / f"{fatal_controller}.json"
    )
    fatal_state.parent.mkdir(parents=True, exist_ok=True)
    fatal_state.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "controller_id": fatal_controller,
                "comparison_id": spec.id,
                "preview_digest": preview.preview_digest,
                "schedule_digest": schedule.schedule_digest,
                "stages": {stage: {"status": "pending"} for stage in stages},
                "source_pre_run_drift": None,
                "fatal_blockers": ["privacy evidence failed"],
            },
            sort_keys=True,
        )
    )
    with pytest.raises(RuntimeError, match="blocked by fatal evidence"):
        execute_comparison_stage(
            preview,
            stage_id=stages[0],
            approval_digest=approvals[stages[0]],
            repo_root=tmp_path,
            controller_id=fatal_controller,
            fetch_weave=False,
        )
    with pytest.raises(Exception) as fatal_unclaimed:
        approval_ledger.require_claimed_by(
            approval_digest=approvals[stages[0]],
            subject_kind="experiment",
            preview_digest=comparison_stage_receipt(
                preview, stages[0]
            ).subset_digest,
            subject_id=f"{fatal_controller}-{stages[0]}",
        )
    assert getattr(fatal_unclaimed.value, "code", "") == "approval_not_claimed"
    with pytest.raises(ValueError, match="completed predecessor"):
        execute_comparison_stage(
            preview,
            stage_id=stages[1],
            approval_digest=approvals[stages[1]],
            repo_root=tmp_path,
            fetch_weave=False,
        )
    with pytest.raises(Exception) as unclaimed:
        approval_ledger.require_claimed_by(
            approval_digest=approvals[stages[1]],
            subject_kind="experiment",
            preview_digest=comparison_stage_receipt(
                preview, stages[1]
            ).subset_digest,
            subject_id=f"comparison-{preview.preview_digest[:20]}-{stages[1]}",
        )
    assert getattr(unclaimed.value, "code", "") == "approval_not_claimed"
    checkpoint = execute_comparison_stage(
        preview,
        stage_id=stages[0],
        approval_digest=approvals[stages[0]],
        repo_root=tmp_path,
        fetch_weave=False,
    )
    assert checkpoint["status"] == "stage_complete", checkpoint
    assert execute_comparison_stage(
        preview,
        stage_id=stages[1],
        approval_digest=approvals[stages[1]],
        repo_root=tmp_path,
        fetch_weave=False,
    )["status"] == "stage_complete"
    with pytest.raises(KeyboardInterrupt):
        execute_comparison_stage(
            preview,
            stage_id=stages[2],
            approval_digest=approvals[stages[2]],
            repo_root=tmp_path,
            fetch_weave=False,
        )
    assert execute_comparison_stage(
        preview,
        stage_id=stages[2],
        approval_digest=approvals[stages[2]],
        repo_root=tmp_path,
        fetch_weave=False,
    )["status"] == "stage_complete"
    pending = execute_comparison_stage(
        preview,
        stage_id=stages[3],
        approval_digest=approvals[stages[3]],
        repo_root=tmp_path,
        fetch_weave=False,
    )
    assert pending["status"] == "finalization_pending"
    calls_before_host_resume = len(runner_calls)
    complete = execute_comparison_stage(
        preview,
        stage_id=stages[3],
        approval_digest=approvals[stages[3]],
        repo_root=tmp_path,
        fetch_weave=False,
    )

    assert complete["status"] == "complete"
    assert complete["result_digest"] == "f" * 64
    assert len(runner_calls) == calls_before_host_resume == 192
    assert len(set(runner_calls)) == 192
    assert policy_calls[0][:3] == ("checkpoint", 1, 2)
    assert policy_calls[0][4] == 4
    assert all(
        call[4] == len(schedule.stages[call[0]]) for call in policy_calls
    )
    assert any(call[0] == "development-repeat" and call[3] for call in policy_calls)
    controller = json.loads(
        (
            tmp_path
            / ".fugue/runtime/comparison-stages/fake-staged-comparison"
            / f"comparison-{preview.preview_digest[:20]}.json"
        ).read_text()
    )
    stage_authorizations = [
        controller["stages"][stage]["authorization"] for stage in stages
    ]
    assert len(
        {item["authorization_digest"] for item in stage_authorizations}
    ) == len(stages)
    assert {
        item["approval"]["approval_digest"] for item in stage_authorizations
    } == set(approvals.values())
    all_records = []
    wave_sizes = []
    for run_id in selected_by_run:
        all_records.extend(
            latest_cell_records(tmp_path / ".fugue/runtime" / run_id / "cells.jsonl")
        )
        event_path = tmp_path / ".fugue/runtime" / run_id / "events.jsonl"
        wave_sizes.extend(
            int(event["logical_cell_count"])
            for event in (
                json.loads(line) for line in event_path.read_text().splitlines()
            )
            if event.get("event") == "admission_wave_complete"
        )
    assert len({str(item["attempt_id"]) for item in all_records}) == 192
    assert max(wave_sizes) <= 4
