from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from fugue.bench.candidates import stable_digest
from fugue.bench.execution import (
    CellOutcome,
    PlannedCell,
    TypedInfrastructureFailure,
)
from fugue.bench.execution_recovery import (
    CanonicalizationObservationV1,
    CleanupObservationV1,
    CostObservationV1,
    ExecutionJournalCorrupt,
    ExecutionRecoveryAdapters,
    ExecutionRecoveryController,
    ExecutionRecoveryError,
    ExecutionScheduleV1,
    InterruptedExecutionObservationV1,
    LogicalExecutionPlanV1,
    StageExecutionAuthorizationV1,
    execute_recoverable_cells,
)

_PREVIEW_DIGEST = stable_digest({"preview": "recovery-tests"})
_APPROVAL_DIGEST = stable_digest({"approval": "recovery-tests"})


def _logical_id(index: int) -> str:
    return stable_digest({"logical_attempt": index})


def _plan(
    index: int,
    *,
    stage_ordinal: int = 0,
    block_ordinal: int = 0,
    attempt_ordinal: int | None = None,
    weight: int = 1,
    cost: int = 2_000_000,
) -> LogicalExecutionPlanV1:
    return LogicalExecutionPlanV1.create(
        logical_attempt_id=_logical_id(index),
        stage_id=f"stage-{stage_ordinal}",
        stage_ordinal=stage_ordinal,
        admission_block_id=f"block-{stage_ordinal}-{block_ordinal}",
        block_ordinal=block_ordinal,
        attempt_ordinal=index if attempt_ordinal is None else attempt_ordinal,
        admission_weight=weight,
        maximum_cost_micro_usd=cost,
        planned_harbor_resources=(f"container-{index}",),
    )


def _schedule(
    plans: list[LogicalExecutionPlanV1],
    *,
    capacity: int = 3,
    replacements: int = 2,
    total: int = 30_000_000,
    in_flight: int | None = None,
) -> ExecutionScheduleV1:
    resolved_in_flight = min(total, 6_000_000) if in_flight is None else in_flight
    return ExecutionScheduleV1.create(
        schedule_id="recovery-test",
        logical_attempts=plans,
        capacity_units=capacity,
        maximum_in_flight_executions=3,
        maximum_physical_executions=len(plans) + replacements,
        maximum_infrastructure_replacements=replacements,
        maximum_total_micro_usd=total,
        maximum_in_flight_micro_usd=resolved_in_flight,
    )


def _authorization(
    schedule: ExecutionScheduleV1,
    stage_id: str | None = None,
) -> StageExecutionAuthorizationV1:
    selected = stage_id or schedule.logical_attempts[0].stage_id
    plans = [item for item in schedule.logical_attempts if item.stage_id == selected]
    return StageExecutionAuthorizationV1.create(
        preview_digest=_PREVIEW_DIGEST,
        approval_digest=stable_digest(
            {"approval": _APPROVAL_DIGEST, "stage": selected}
        ),
        schedule_digest=schedule.schedule_digest,
        stage_id=selected,
        maximum_logical_attempts=len(plans),
        maximum_physical_executions=(
            len(plans) + schedule.maximum_infrastructure_replacements
        ),
        maximum_cost_micro_usd=(
            sum(item.maximum_cost_micro_usd for item in plans)
            + schedule.maximum_infrastructure_replacements
            * max(item.maximum_cost_micro_usd for item in plans)
        ),
    )


def _authorize(
    controller: ExecutionRecoveryController,
    stage_id: str | None = None,
) -> StageExecutionAuthorizationV1:
    authorization = _authorization(controller.schedule, stage_id)
    controller.authorize_stage(authorization)
    return authorization


def _admit(
    controller: ExecutionRecoveryController,
    stage_id: str | None = None,
):
    authorization = _authorize(controller, stage_id)
    return controller.admit(stage_id=authorization.stage_id)


def _cleanup(
    physical: object,
    resources: tuple[str, ...] = (),
) -> CleanupObservationV1:
    return CleanupObservationV1(
        verified=True,
        scope_verified=True,
        post_run_inventory=True,
        inspected_resources=resources,
        removed_resources=resources,
        receipt_reference=f"cleanup/{physical.physical_execution_id}.json",
    )


def _cost(physical: object, value: int = 100_000) -> CostObservationV1:
    return CostObservationV1(
        actual_cost_micro_usd=value,
        authoritative=True,
        source="test-ledger",
        receipt_reference=f"cost/{physical.physical_execution_id}.json",
    )


def _canonical(physical: object) -> CanonicalizationObservationV1:
    return CanonicalizationObservationV1(
        kind="test-evidence",
        status="verified",
        reference=f"evidence/{physical.physical_execution_id}.json",
        evidence_digest=stable_digest(
            {"physical_execution_id": physical.physical_execution_id}
        ),
    )


def _adapters(
    *,
    cleanup_verifier=None,
    cost_resolver=None,
    canonicalization_verifier=None,
    resource_resolver=None,
    interrupted_reconciler=None,
    redaction_secrets: tuple[str, ...] = (),
) -> ExecutionRecoveryAdapters:
    return ExecutionRecoveryAdapters(
        contract_digest=_schedule_contract_digest(),
        cleanup_verifier=cleanup_verifier
        or (lambda _cell, physical, _outcome: _cleanup(physical)),
        cost_resolver=cost_resolver
        or (lambda _cell, physical, _outcome: _cost(physical)),
        canonicalization_verifier=canonicalization_verifier
        or (lambda _cell, physical, _outcome: _canonical(physical)),
        resource_resolver=resource_resolver,
        interrupted_reconciler=interrupted_reconciler,
        redaction_secrets=redaction_secrets,
    )


def _schedule_contract_digest() -> str:
    return _schedule([_plan(0)], replacements=0).adapter_contract_digest


def _finish(
    controller: ExecutionRecoveryController,
    physical: object,
    *,
    terminal_kind: str = "success",
    cost: int = 100_000,
) -> None:
    identity = physical
    plan = controller._plan_by_id[identity.logical_attempt_id]
    controller.started(
        identity,
        observed_harbor_resources=plan.planned_harbor_resources,
    )
    controller.finalize(
        identity,
        terminal_kind=terminal_kind,
        result_reference=f"results/{identity.physical_execution_id}.json",
        cleanup=_cleanup(identity, plan.planned_harbor_resources),
        cost=_cost(identity, cost),
        cell_outcome=CellOutcome(
            cell_id=f"cell-{identity.logical_attempt_id[:12]}",
            status=(
                "passed"
                if terminal_kind == "success"
                else "cancelled"
                if terminal_kind == "cancelled"
                else "failed"
            ),
            runtime_outcome=(
                "completed"
                if terminal_kind in {"success", "task_failure"}
                else "timed_out"
                if terminal_kind == "agent_timeout"
                else "cancelled"
                if terminal_kind == "cancelled"
                else "not_started"
            ),
            terminal_kind=terminal_kind,  # type: ignore[arg-type]
        ),
    )
    if terminal_kind in {"success", "task_failure", "agent_timeout", "cancelled"}:
        controller.select_canonical(identity, _canonical(identity))


def test_schedule_is_strict_digest_bound_and_supports_heterogeneous_cells() -> None:
    plans = [
        _plan(0, attempt_ordinal=0, cost=1_500_000),
        _plan(1, attempt_ordinal=1, cost=2_000_000),
        _plan(2, attempt_ordinal=2, weight=3, cost=3_250_000),
    ]
    schedule = _schedule(plans, total=10_000_000, in_flight=4_000_000)

    assert ExecutionScheduleV1.from_dict(schedule.to_dict()) == schedule
    assert [item.maximum_cost_micro_usd for item in schedule.logical_attempts] == [
        1_500_000,
        2_000_000,
        3_250_000,
    ]

    tampered = schedule.to_dict()
    tampered["logical_attempts"][0]["admission_weight"] = 2
    with pytest.raises(ValueError, match="plan digest"):
        ExecutionScheduleV1.from_dict(tampered)

    unknown = schedule.to_dict()
    unknown["display_name"] = "not identity"
    with pytest.raises(ValueError, match="unknown=display_name"):
        ExecutionScheduleV1.from_dict(unknown)


def test_weighted_admission_runs_dynamic_workflow_alone(tmp_path: Path) -> None:
    plans = [_plan(index, attempt_ordinal=index, weight=1) for index in range(3)] + [
        _plan(3, attempt_ordinal=3, weight=3, cost=3_250_000)
    ]
    schedule = _schedule(plans, total=12_000_000, in_flight=6_000_000)
    controller = ExecutionRecoveryController(
        tmp_path / "events.jsonl",
        controller_id="weighted",
        schedule=schedule,
    )

    first = _admit(controller)
    assert first.status == "admitted"
    assert len(first.physical_executions) == 3
    assert (
        sum(
            controller._plan_by_id[item.logical_attempt_id].admission_weight
            for item in first.physical_executions
        )
        == 3
    )
    for physical in first.physical_executions:
        _finish(controller, physical)

    second = _admit(controller)
    assert len(second.physical_executions) == 1
    dynamic = second.physical_executions[0]
    assert controller._plan_by_id[dynamic.logical_attempt_id].admission_weight == 3
    _finish(controller, dynamic)

    assert _admit(controller).status == "complete"
    assert len(controller.snapshot().canonical_results) == 4


def test_only_typed_infrastructure_failure_gets_new_physical_identity(
    tmp_path: Path,
) -> None:
    schedule = _schedule([_plan(0, attempt_ordinal=0)], replacements=1, total=4_000_000)
    controller = ExecutionRecoveryController(
        tmp_path / "events.jsonl",
        controller_id="retry",
        schedule=schedule,
    )
    initial = _admit(controller).physical_executions[0]
    _finish(controller, initial, terminal_kind="runner_start_failure", cost=20_000)

    retry = _admit(controller).physical_executions[0]
    assert retry.logical_attempt_id == initial.logical_attempt_id
    assert retry.retry_ordinal == 1
    assert retry.physical_execution_id != initial.physical_execution_id
    _finish(controller, retry, terminal_kind="agent_timeout")

    assert _admit(controller).status == "complete"
    snapshot = controller.snapshot()
    assert snapshot.canonical_results == {
        retry.logical_attempt_id: retry.physical_execution_id
    }
    assert len(snapshot.physical_executions) == 2


def test_replacement_ceiling_is_atomic_across_one_failed_wave(tmp_path: Path) -> None:
    schedule = _schedule(
        [
            _plan(0, attempt_ordinal=0),
            _plan(1, attempt_ordinal=1),
        ],
        capacity=2,
        replacements=1,
        total=6_000_000,
        in_flight=4_000_000,
    )
    controller = ExecutionRecoveryController(
        tmp_path / "events.jsonl",
        controller_id="replacement-cap",
        schedule=schedule,
    )
    initial = _admit(controller).physical_executions
    assert len(initial) == 2
    for physical in initial:
        _finish(
            controller,
            physical,
            terminal_kind="transport_interrupted",
        )

    only_replacement = _admit(controller)
    assert len(only_replacement.physical_executions) == 1
    _finish(controller, only_replacement.physical_executions[0])
    blocked = _admit(controller)
    assert blocked.status == "blocked"
    assert blocked.reason in {
        "infrastructure replacement ceiling reached",
        "physical execution ceiling reached",
    }


def test_missing_cost_and_ambiguous_active_execution_pause_recovery(
    tmp_path: Path,
) -> None:
    schedule = _schedule(
        [
            _plan(0, attempt_ordinal=0),
            _plan(1, block_ordinal=1, attempt_ordinal=0),
        ],
        replacements=0,
        total=4_000_000,
    )
    path = tmp_path / "events.jsonl"
    controller = ExecutionRecoveryController(
        path, controller_id="pause", schedule=schedule
    )
    first = _admit(controller).physical_executions[0]
    plan = controller._plan_by_id[first.logical_attempt_id]
    controller.started(first, observed_harbor_resources=plan.planned_harbor_resources)

    rehydrated = ExecutionRecoveryController(
        path, controller_id="pause", schedule=schedule
    )
    assert rehydrated.snapshot().active_physical_execution_ids == (
        first.physical_execution_id,
    )
    assert _admit(rehydrated).status == "blocked"

    rehydrated.terminal(
        first,
        terminal_kind="success",
        result_reference="result.json",
        cell_outcome=CellOutcome("cell-first", "passed", terminal_kind="success"),
    )
    rehydrated.cleanup(
        first,
        _cleanup(first, plan.planned_harbor_resources),
    )
    rehydrated.cost(
        first,
        CostObservationV1(
            actual_cost_micro_usd=None,
            authoritative=False,
            source="test-ledger",
            receipt_reference="cost/first-pending.json",
        ),
    )
    missing_cost_state = rehydrated.snapshot().physical_executions[
        first.physical_execution_id
    ]
    assert missing_cost_state.cost_observed is True
    assert missing_cost_state.cost_authoritative is False
    assert _admit(rehydrated).reason == "authoritative cost is missing"

    rehydrated.cost(first, _cost(first, 250_000))
    rehydrated.select_canonical(first, _canonical(first))
    assert _admit(rehydrated).status == "admitted"


def test_rehydrated_active_execution_requires_explicit_typed_reconciliation(
    tmp_path: Path,
) -> None:
    schedule = _schedule([_plan(0, attempt_ordinal=0)], replacements=1, total=4_000_000)
    path = tmp_path / "events.jsonl"
    controller = ExecutionRecoveryController(
        path, controller_id="reconcile", schedule=schedule
    )
    physical = _admit(controller).physical_executions[0]
    plan = controller._plan_by_id[physical.logical_attempt_id]
    controller.started(
        physical, observed_harbor_resources=plan.planned_harbor_resources
    )

    resumed = ExecutionRecoveryController(
        path, controller_id="reconcile", schedule=schedule
    )
    resumed.reconcile_interrupted_physical(
        physical,
        terminal_kind="sandbox_lost",
        cell_outcome=CellOutcome(
            "cell-reconcile",
            "failed",
            error="runner attested sandbox loss",
            terminal_kind="sandbox_lost",
        ),
        result_reference=None,
        cleanup=_cleanup(physical, plan.planned_harbor_resources),
        cost=_cost(physical, 25_000),
    )

    replacement = _admit(resumed).physical_executions[0]
    assert replacement.retry_ordinal == 1
    assert replacement.physical_execution_id != physical.physical_execution_id


def test_cost_over_lease_is_durable_and_fatal(tmp_path: Path) -> None:
    schedule = _schedule(
        [_plan(0, attempt_ordinal=0, cost=100_000)],
        replacements=0,
        total=100_000,
        in_flight=100_000,
    )
    controller = ExecutionRecoveryController(
        tmp_path / "events.jsonl",
        controller_id="cost-overage",
        schedule=schedule,
    )
    physical = _admit(controller).physical_executions[0]
    plan = controller._plan_by_id[physical.logical_attempt_id]
    controller.started(
        physical, observed_harbor_resources=plan.planned_harbor_resources
    )
    controller.finalize(
        physical,
        terminal_kind="success",
        result_reference="result.json",
        cleanup=_cleanup(physical, plan.planned_harbor_resources),
        cost=_cost(physical, 100_001),
        cell_outcome=CellOutcome("cell-cost", "passed", terminal_kind="success"),
    )

    state = controller.snapshot().physical_executions[physical.physical_execution_id]
    assert state.actual_cost_micro_usd == 100_001
    assert state.cost_lease_exceeded is True
    assert _admit(controller).status == "blocked"

    # Admission also fails from the cost event itself if a process died after
    # persisting cost but before appending the redundant fatal-halt marker.
    rows = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert rows[-1]["event_kind"] == "fatal_integrity_halt"
    (tmp_path / "events.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows[:-1]) + "\n",
        encoding="utf-8",
    )
    crash_recovered = ExecutionRecoveryController(
        tmp_path / "events.jsonl",
        controller_id="cost-overage",
        schedule=schedule,
    )
    assert crash_recovered.snapshot().fatal_reasons == ()
    blocked = crash_recovered.admit(stage_id="stage-0")
    assert blocked.status == "blocked"
    assert blocked.reason == "a physical execution exceeded its cost lease"


def test_fatal_integrity_and_cleanup_failure_stop_later_admission(
    tmp_path: Path,
) -> None:
    schedule = _schedule(
        [
            _plan(0, attempt_ordinal=0),
            _plan(1, block_ordinal=1, attempt_ordinal=0),
        ],
        replacements=0,
        total=4_000_000,
    )
    controller = ExecutionRecoveryController(
        tmp_path / "events.jsonl",
        controller_id="fatal",
        schedule=schedule,
    )
    physical = _admit(controller).physical_executions[0]
    plan = controller._plan_by_id[physical.logical_attempt_id]
    controller.started(
        physical, observed_harbor_resources=plan.planned_harbor_resources
    )
    controller.terminal(
        physical,
        terminal_kind="success",
        result_reference="result.json",
        cell_outcome=CellOutcome("cell-fatal", "passed", terminal_kind="success"),
    )
    controller.cleanup(
        physical,
        CleanupObservationV1(
            verified=False,
            inspected_resources=("container-0",),
            remaining_resources=("container-0",),
            receipt_reference="cleanup/fatal.json",
        ),
    )

    blocked = _admit(controller)
    assert blocked.status == "blocked"
    assert "cleanup_failure" in blocked.reason
    snapshot = controller.snapshot()
    assert snapshot.fatal_reasons
    assert snapshot.physical_executions[
        physical.physical_execution_id
    ].remaining_harbor_resources == ("container-0",)


def test_digest_chain_rejects_tampering(tmp_path: Path) -> None:
    schedule = _schedule([_plan(0, attempt_ordinal=0)], replacements=0, total=2_000_000)
    path = tmp_path / "events.jsonl"
    controller = ExecutionRecoveryController(
        path, controller_id="tamper", schedule=schedule
    )
    _admit(controller)
    before = controller.snapshot()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert before.event_count == len(rows)
    assert before.event_chain_digest == rows[-1]["event_digest"]
    rows[1]["payload"]["reserve_micro_usd"] = 1
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")

    with pytest.raises(ExecutionJournalCorrupt):
        controller.snapshot()


def test_terminal_transition_is_atomic_across_controllers(tmp_path: Path) -> None:
    schedule = _schedule([_plan(0, attempt_ordinal=0)], replacements=0, total=2_000_000)
    path = tmp_path / "events.jsonl"
    first = ExecutionRecoveryController(path, controller_id="atomic", schedule=schedule)
    second = ExecutionRecoveryController(
        path, controller_id="atomic", schedule=schedule
    )
    physical = _admit(first).physical_executions[0]
    plan = first._plan_by_id[physical.logical_attempt_id]
    first.started(physical, observed_harbor_resources=plan.planned_harbor_resources)
    outcome = CellOutcome("cell-atomic", "passed", terminal_kind="success")
    barrier = threading.Barrier(2)

    def terminal(controller: ExecutionRecoveryController) -> str:
        barrier.wait()
        try:
            controller.terminal(
                physical,
                terminal_kind="success",
                result_reference="result.json",
                cell_outcome=outcome,
            )
        except ExecutionRecoveryError:
            return "rejected"
        return "recorded"

    with ThreadPoolExecutor(max_workers=2) as pool:
        states = list(pool.map(terminal, (first, second)))

    assert sorted(states) == ["recorded", "rejected"]
    assert (
        first.snapshot()
        .physical_executions[physical.physical_execution_id]
        .terminal_kind
        == "success"
    )


def _cell(tmp_path: Path, index: int, *, run_id: str) -> PlannedCell:
    config = tmp_path / f"cell-{index}.yaml"
    config.write_text("jobs: {}\n", encoding="utf-8")
    return PlannedCell(
        id=f"cell-{index}",
        run_id=run_id,
        run_name="recovery",
        workload_id="harbor",
        task_id=f"task-{index}",
        harness="claude-code",
        context_system_id="none",
        variant_id=f"arm-{index}",
        model_provider="anthropic",
        model="claude-sonnet-5",
        trial_index=1,
        comparison_example_id=stable_digest({"example": index}),
        candidate_id=stable_digest({"candidate": index}),
        execution_fingerprint=stable_digest({"runtime": "harbor"}),
        config_path=config,
        result_path=tmp_path / f"result-{index}.json",
        command=("fake-harbor",),
        env={},
        n_attempts=1,
    )


def test_recovery_wrapper_uses_canonical_cell_executor_and_resumes_retry(
    tmp_path: Path,
) -> None:
    cells = [_cell(tmp_path, index, run_id="canonical-run") for index in range(2)]
    plans = [
        LogicalExecutionPlanV1.create(
            logical_attempt_id=cell.attempt_id,
            stage_id="checkpoint",
            stage_ordinal=0,
            admission_block_id="checkpoint-pair",
            block_ordinal=0,
            attempt_ordinal=index,
            admission_weight=1,
            maximum_cost_micro_usd=500_000,
            planned_harbor_resources=(f"compose-{index}",),
        )
        for index, cell in enumerate(cells)
    ]
    controller = ExecutionRecoveryController(
        tmp_path / "events.jsonl",
        controller_id="canonical-wrapper",
        schedule=_schedule(
            plans,
            capacity=2,
            replacements=1,
            total=1_500_000,
            in_flight=1_000_000,
        ),
    )
    seen: list[str] = []
    failed_once = False

    def runner(*_args: object, env: dict[str, str], **_kwargs: object) -> object:
        nonlocal failed_once
        physical_id = env["FUGUE_PHYSICAL_EXECUTION_ID"]
        seen.append(physical_id)
        if not failed_once and env["FUGUE_RETRY_ORDINAL"] == "0":
            failed_once = True
            raise TypedInfrastructureFailure(
                "runner_start_failure", "Harbor runner was unavailable"
            )
        return SimpleNamespace(returncode=0)

    outcomes = execute_recoverable_cells(
        controller,
        cells,
        stage_authorization=_authorization(controller.schedule),
        adapters=_adapters(
            resource_resolver=lambda cell: (
                f"compose-{0 if cell.id == 'cell-0' else 1}",
            ),
            cleanup_verifier=lambda _cell, physical, _outcome: _cleanup(
                physical,
                controller._plan_by_id[
                    physical.logical_attempt_id
                ].planned_harbor_resources,
            ),
            cost_resolver=lambda _cell, physical, _outcome: _cost(physical, 100_000),
        ),
        repo_root=tmp_path,
        runner=runner,
    )

    snapshot = controller.snapshot()
    assert len(outcomes) == 2
    assert all(item.status == "passed" for item in outcomes)
    assert len(seen) == 3
    assert len(set(seen)) == 3
    assert len(snapshot.physical_executions) == 3
    assert len(snapshot.canonical_results) == 2
    assert _admit(controller).status == "complete"
    run_records = [
        json.loads(line)
        for line in (tmp_path / ".fugue/runtime/canonical-run/cells.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert set(seen).issubset(
        {str(record.get("physical_execution_id")) for record in run_records}
    )


def test_recovery_wrapper_runs_required_local_evaluator_once(tmp_path: Path) -> None:
    cell = _cell(tmp_path, 0, run_id="local-evaluator")
    plan = LogicalExecutionPlanV1.create(
        logical_attempt_id=cell.attempt_id,
        stage_id="checkpoint",
        stage_ordinal=0,
        admission_block_id="checkpoint",
        block_ordinal=0,
        attempt_ordinal=0,
        admission_weight=1,
        maximum_cost_micro_usd=500_000,
    )
    schedule = _schedule([plan], replacements=0, total=500_000, in_flight=500_000)
    controller = ExecutionRecoveryController(
        tmp_path / "events.jsonl",
        controller_id="local-evaluator",
        schedule=schedule,
    )
    evaluated: list[tuple[str, str]] = []

    def evaluate(completed: PlannedCell, outcome: CellOutcome) -> None:
        evaluated.append((completed.physical_execution_id, outcome.status))
        completed.result_path.write_text('{"score": 1}\n', encoding="utf-8")

    outcomes = execute_recoverable_cells(
        controller,
        [cell],
        stage_authorization=_authorization(schedule),
        adapters=_adapters(),
        repo_root=tmp_path,
        runner=lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
        cell_finished=evaluate,
    )

    assert outcomes[0].status == "passed"
    assert len(evaluated) == 1
    assert evaluated[0][1] == "passed"
    state = next(iter(controller.snapshot().physical_executions.values()))
    assert state.result_reference is not None


def test_generic_runner_oserror_is_fatal_and_never_retried(
    tmp_path: Path,
) -> None:
    cell = _cell(tmp_path, 0, run_id="generic-oserror")
    plan = LogicalExecutionPlanV1.create(
        logical_attempt_id=cell.attempt_id,
        stage_id="checkpoint",
        stage_ordinal=0,
        admission_block_id="checkpoint",
        block_ordinal=0,
        attempt_ordinal=0,
        admission_weight=1,
        maximum_cost_micro_usd=500_000,
    )
    controller = ExecutionRecoveryController(
        tmp_path / "events.jsonl",
        controller_id="generic-oserror",
        schedule=_schedule([plan], replacements=1, total=1_000_000, in_flight=500_000),
    )
    calls = 0

    def runner(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise OSError("opaque runner error")

    with pytest.raises(ExecutionRecoveryError, match="execution_failure"):
        execute_recoverable_cells(
            controller,
            [cell],
            stage_authorization=_authorization(controller.schedule),
            adapters=_adapters(),
            repo_root=tmp_path,
            runner=runner,
        )

    assert calls == 1
    assert len(controller.snapshot().physical_executions) == 1


def test_completed_cell_is_durable_before_slow_wave_peer_finishes(
    tmp_path: Path,
) -> None:
    cells = [_cell(tmp_path, index, run_id="mid-wave") for index in range(2)]
    plans = [
        LogicalExecutionPlanV1.create(
            logical_attempt_id=cell.attempt_id,
            stage_id="checkpoint",
            stage_ordinal=0,
            admission_block_id="checkpoint",
            block_ordinal=0,
            attempt_ordinal=index,
            admission_weight=1,
            maximum_cost_micro_usd=500_000,
        )
        for index, cell in enumerate(cells)
    ]
    controller = ExecutionRecoveryController(
        tmp_path / "events.jsonl",
        controller_id="mid-wave",
        schedule=_schedule(
            plans,
            capacity=2,
            replacements=0,
            total=1_000_000,
            in_flight=1_000_000,
        ),
    )
    release_slow = threading.Event()
    runner_lock = threading.Lock()
    runner_calls = 0

    def runner(*_args: object, env: dict[str, str], **_kwargs: object) -> object:
        del env
        nonlocal runner_calls
        with runner_lock:
            runner_calls += 1
            slow = runner_calls == 1
        if slow:
            assert release_slow.wait(timeout=5)
        return SimpleNamespace(returncode=0)

    thread = threading.Thread(
        target=execute_recoverable_cells,
        kwargs={
            "controller": controller,
            "cells": cells,
            "stage_authorization": _authorization(controller.schedule),
            "adapters": _adapters(
                cost_resolver=lambda _cell, physical, _outcome: _cost(physical, 10_000)
            ),
            "repo_root": tmp_path,
            "runner": runner,
        },
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        snapshot = controller.snapshot()
        if len(snapshot.canonical_results) == 1:
            break
        time.sleep(0.01)
    else:
        pytest.fail("fast cell was not durably selected during the active wave")
    assert thread.is_alive()
    assert len(controller.snapshot().active_physical_execution_ids) == 1
    release_slow.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert controller.snapshot().complete


def test_288_logical_cell_matrix_recovers_without_rerunning_valid_work(
    tmp_path: Path,
) -> None:
    plans: list[LogicalExecutionPlanV1] = []
    index = 0
    for stage_ordinal, count, arm_count in ((0, 192, 4), (1, 48, 3), (2, 48, 3)):
        for within_stage in range(count):
            arm = within_stage % arm_count
            dynamic = arm == arm_count - 1
            plans.append(
                _plan(
                    index,
                    stage_ordinal=stage_ordinal,
                    block_ordinal=within_stage // arm_count,
                    attempt_ordinal=arm,
                    weight=3 if dynamic else 1,
                    cost=3_250_000 if dynamic else 2_000_000,
                )
            )
            index += 1
    assert len(plans) == 288
    planned_cost = sum(item.maximum_cost_micro_usd for item in plans)
    schedule = _schedule(
        plans,
        capacity=3,
        replacements=6,
        total=planned_cost + 6 * 3_250_000,
        in_flight=6_000_000,
    )
    path = tmp_path / "events.jsonl"
    controller = ExecutionRecoveryController(
        path, controller_id="matrix-288", schedule=schedule
    )
    retry_logical_ids = {
        plans[index].logical_attempt_id for index in (0, 47, 95, 143, 191, 287)
    }
    waves = 0
    for stage_id in ("stage-0", "stage-1", "stage-2"):
        while True:
            admission = _admit(controller, stage_id)
            if admission.status == "complete":
                break
            assert admission.status == "admitted", admission.reason
            weights = [
                controller._plan_by_id[item.logical_attempt_id].admission_weight
                for item in admission.physical_executions
            ]
            assert sum(weights) <= 3
            if 3 in weights:
                assert weights == [3]
            for physical in admission.physical_executions:
                terminal = (
                    "sandbox_lost"
                    if physical.logical_attempt_id in retry_logical_ids
                    and physical.retry_ordinal == 0
                    else "success"
                )
                _finish(controller, physical, terminal_kind=terminal, cost=100_000)
            waves += 1
            if waves % 11 == 0:
                controller = ExecutionRecoveryController(
                    path, controller_id="matrix-288", schedule=schedule
                )

    snapshot = controller.snapshot()
    assert len(snapshot.canonical_results) == 288
    assert len(snapshot.physical_executions) == 294
    assert len(set(snapshot.canonical_results.values())) == 288
    assert not snapshot.fatal_reasons
    assert not snapshot.active_physical_execution_ids


def test_recovery_wrapper_refuses_ambiguous_controller_state(tmp_path: Path) -> None:
    cell = _cell(tmp_path, 0, run_id="ambiguous-run")
    plan = LogicalExecutionPlanV1.create(
        logical_attempt_id=cell.attempt_id,
        stage_id="checkpoint",
        stage_ordinal=0,
        admission_block_id="checkpoint",
        block_ordinal=0,
        attempt_ordinal=0,
        admission_weight=1,
        maximum_cost_micro_usd=500_000,
    )
    controller = ExecutionRecoveryController(
        tmp_path / "events.jsonl",
        controller_id="ambiguous",
        schedule=_schedule([plan], replacements=0, total=500_000, in_flight=500_000),
    )
    _admit(controller)

    with pytest.raises(ExecutionRecoveryError, match="unresolved active"):
        execute_recoverable_cells(
            controller,
            [cell],
            stage_authorization=_authorization(controller.schedule),
            adapters=_adapters(),
            repo_root=tmp_path,
            runner=lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
        )


def test_recovery_wrapper_rehydrates_canonical_outcomes_without_reexecution(
    tmp_path: Path,
) -> None:
    cells = [_cell(tmp_path, index, run_id="resumed-run") for index in range(4)]
    plans = [
        LogicalExecutionPlanV1.create(
            logical_attempt_id=cell.attempt_id,
            stage_id=f"stage-{index // 2}",
            stage_ordinal=index // 2,
            admission_block_id=f"pair-{index // 2}",
            block_ordinal=0,
            attempt_ordinal=index % 2,
            admission_weight=1,
            maximum_cost_micro_usd=500_000,
        )
        for index, cell in enumerate(cells)
    ]
    schedule = _schedule(
        plans,
        capacity=2,
        replacements=0,
        total=2_000_000,
        in_flight=1_000_000,
    )
    path = tmp_path / "events.jsonl"
    controller = ExecutionRecoveryController(
        path, controller_id="resumed", schedule=schedule
    )
    calls: list[str] = []

    def runner(*_args: object, env: dict[str, str], **_kwargs: object) -> object:
        calls.append(env["FUGUE_PHYSICAL_EXECUTION_ID"])
        return SimpleNamespace(returncode=0)

    checkpoint = execute_recoverable_cells(
        controller,
        cells,
        stage_authorization=_authorization(schedule, "stage-0"),
        adapters=_adapters(),
        repo_root=tmp_path,
        runner=runner,
    )
    assert len(checkpoint) == 2

    resumed = ExecutionRecoveryController(
        path, controller_id="resumed", schedule=schedule
    )
    final = execute_recoverable_cells(
        resumed,
        cells,
        stage_authorization=_authorization(schedule, "stage-1"),
        adapters=_adapters(),
        repo_root=tmp_path,
        runner=runner,
    )

    assert len(final) == 2
    assert len(calls) == 4
    assert len(set(calls)) == 4
    assert {item.cell_id for item in final} == {cell.id for cell in cells[2:]}
    assert len(resumed.snapshot().canonical_results) == 4


def test_stage_authorization_is_exact_durable_and_ordered(tmp_path: Path) -> None:
    plans = [
        _plan(0, stage_ordinal=0, attempt_ordinal=0, cost=100_000),
        _plan(1, stage_ordinal=1, attempt_ordinal=0, cost=100_000),
    ]
    schedule = _schedule(
        plans,
        replacements=1,
        total=300_000,
        in_flight=100_000,
    )
    controller = ExecutionRecoveryController(
        tmp_path / "events.jsonl",
        controller_id="stage-authorization",
        schedule=schedule,
    )

    assert controller.admit(stage_id="stage-0").reason == (
        "stage lacks a durable authorization"
    )
    stage_one = _authorization(schedule, "stage-1")
    controller.authorize_stage(stage_one)
    assert controller.admit(stage_id="stage-1").reason == (
        "an earlier admission block is incomplete"
    )

    wrong_count = StageExecutionAuthorizationV1.create(
        preview_digest=_PREVIEW_DIGEST,
        approval_digest=_APPROVAL_DIGEST,
        schedule_digest=schedule.schedule_digest,
        stage_id="stage-0",
        maximum_logical_attempts=2,
        maximum_physical_executions=2,
        maximum_cost_micro_usd=200_000,
    )
    with pytest.raises(ValueError, match="logical ceiling is not exact"):
        controller.authorize_stage(wrong_count)

    stage_zero = _authorization(schedule, "stage-0")
    controller.authorize_stage(stage_zero)
    rehydrated = ExecutionRecoveryController(
        tmp_path / "events.jsonl",
        controller_id="stage-authorization",
        schedule=schedule,
    )
    assert rehydrated.snapshot().stage_authorizations == {
        "stage-0": stage_zero,
        "stage-1": stage_one,
    }
    changed = StageExecutionAuthorizationV1.create(
        preview_digest=_PREVIEW_DIGEST,
        approval_digest=stable_digest({"approval": "changed"}),
        schedule_digest=schedule.schedule_digest,
        stage_id="stage-0",
        maximum_logical_attempts=1,
        maximum_physical_executions=2,
        maximum_cost_micro_usd=200_000,
    )
    with pytest.raises(ExecutionRecoveryError, match="different durable"):
        rehydrated.authorize_stage(changed)

    cross_preview = ExecutionRecoveryController(
        tmp_path / "cross-preview.jsonl",
        controller_id="cross-preview",
        schedule=schedule,
    )
    cross_preview.authorize_stage(stage_zero)
    different_preview = StageExecutionAuthorizationV1.create(
        preview_digest=stable_digest({"preview": "changed"}),
        approval_digest=stable_digest({"approval": "stage-one"}),
        schedule_digest=schedule.schedule_digest,
        stage_id="stage-1",
        maximum_logical_attempts=1,
        maximum_physical_executions=2,
        maximum_cost_micro_usd=200_000,
    )
    with pytest.raises(ExecutionRecoveryError, match="different full preview"):
        cross_preview.authorize_stage(different_preview)


def test_stage_authorization_caps_replacement_execution(tmp_path: Path) -> None:
    schedule = _schedule(
        [_plan(0, attempt_ordinal=0, cost=100_000)],
        replacements=1,
        total=200_000,
        in_flight=100_000,
    )
    controller = ExecutionRecoveryController(
        tmp_path / "events.jsonl",
        controller_id="stage-physical-cap",
        schedule=schedule,
    )
    authorization = StageExecutionAuthorizationV1.create(
        preview_digest=_PREVIEW_DIGEST,
        approval_digest=_APPROVAL_DIGEST,
        schedule_digest=schedule.schedule_digest,
        stage_id="stage-0",
        maximum_logical_attempts=1,
        maximum_physical_executions=1,
        maximum_cost_micro_usd=100_000,
    )
    controller.authorize_stage(authorization)
    physical = controller.admit(stage_id="stage-0").physical_executions[0]
    _finish(
        controller,
        physical,
        terminal_kind="runner_start_failure",
        cost=10_000,
    )

    blocked = controller.admit(stage_id="stage-0")
    assert blocked.status == "blocked"
    assert blocked.reason == "capacity or budget prevents admission"


def test_partial_terminal_settlements_resume_without_agent_reexecution(
    tmp_path: Path,
) -> None:
    cells = [_cell(tmp_path, index, run_id="partial-settlement") for index in range(2)]
    plans = [
        LogicalExecutionPlanV1.create(
            logical_attempt_id=cell.attempt_id,
            stage_id="checkpoint",
            stage_ordinal=0,
            admission_block_id="checkpoint-pair",
            block_ordinal=0,
            attempt_ordinal=index,
            admission_weight=1,
            maximum_cost_micro_usd=500_000,
            planned_harbor_resources=(f"container-{index}",),
        )
        for index, cell in enumerate(cells)
    ]
    schedule = _schedule(
        plans,
        capacity=2,
        replacements=0,
        total=1_000_000,
        in_flight=1_000_000,
    )
    path = tmp_path / "events.jsonl"
    controller = ExecutionRecoveryController(
        path,
        controller_id="partial-settlement",
        schedule=schedule,
    )
    authorization = _authorization(schedule)
    controller.authorize_stage(authorization)
    physicals = controller.admit(stage_id="checkpoint").physical_executions
    assert len(physicals) == 2
    for index, physical in enumerate(physicals):
        controller.started(
            physical,
            observed_harbor_resources=(f"container-{index}",),
        )
        controller.terminal(
            physical,
            terminal_kind="success",
            result_reference=f"external/{physical.physical_execution_id}.json",
            cell_outcome=CellOutcome(
                cells[index].id,
                "passed",
                runtime_outcome="completed",
                terminal_kind="success",
            ),
        )
    controller.cleanup(physicals[1], _cleanup(physicals[1], ("container-1",)))
    controller.cost(
        physicals[1],
        CostObservationV1(
            actual_cost_micro_usd=None,
            authoritative=False,
            source="pending-ledger",
            receipt_reference="cost/pending.json",
        ),
    )

    runner_calls = 0
    canonical_calls: list[str] = []

    def runner(*_args: object, **_kwargs: object) -> object:
        nonlocal runner_calls
        runner_calls += 1
        return SimpleNamespace(returncode=0)

    outcomes = execute_recoverable_cells(
        ExecutionRecoveryController(
            path,
            controller_id="partial-settlement",
            schedule=schedule,
        ),
        cells,
        stage_authorization=authorization,
        adapters=_adapters(
            cleanup_verifier=lambda _cell, physical, _outcome: _cleanup(
                physical,
                (f"container-{0 if _cell.id == 'cell-0' else 1}",),
            ),
            canonicalization_verifier=lambda _cell, physical, _outcome: (
                canonical_calls.append(physical.physical_execution_id)
                or _canonical(physical)
            ),
        ),
        repo_root=tmp_path,
        runner=runner,
    )

    assert runner_calls == 0
    assert len(canonical_calls) == 2
    assert [item.status for item in outcomes] == ["passed", "passed"]
    repaired = ExecutionRecoveryController(
        path,
        controller_id="partial-settlement",
        schedule=schedule,
    ).snapshot()
    assert len(repaired.canonical_results) == 2
    assert all(
        state.fully_reconciled for state in repaired.physical_executions.values()
    )
    records = (tmp_path / ".fugue/runtime/partial-settlement/cells.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"recovered": true' in records


def test_wrapper_pauses_for_missing_cost_then_resumes_without_agent_spend(
    tmp_path: Path,
) -> None:
    cell = _cell(tmp_path, 0, run_id="pending-cost")
    plan = LogicalExecutionPlanV1.create(
        logical_attempt_id=cell.attempt_id,
        stage_id="checkpoint",
        stage_ordinal=0,
        admission_block_id="checkpoint",
        block_ordinal=0,
        attempt_ordinal=0,
        admission_weight=1,
        maximum_cost_micro_usd=500_000,
    )
    schedule = _schedule([plan], replacements=0, total=500_000, in_flight=500_000)
    path = tmp_path / "events.jsonl"
    controller = ExecutionRecoveryController(
        path, controller_id="pending-cost", schedule=schedule
    )
    authorization = _authorization(schedule)
    runner_calls = 0
    cost_calls = 0

    def runner(*_args: object, **_kwargs: object) -> object:
        nonlocal runner_calls
        runner_calls += 1
        return SimpleNamespace(returncode=0)

    def resolve_cost(
        _cell: PlannedCell, physical: object, _outcome: CellOutcome
    ) -> CostObservationV1:
        nonlocal cost_calls
        cost_calls += 1
        if cost_calls <= 3:
            return CostObservationV1(
                actual_cost_micro_usd=None,
                authoritative=False,
                source="pending-ledger",
                receipt_reference=f"cost/{physical.physical_execution_id}-pending.json",
            )
        return _cost(physical, 100_000)

    adapters = _adapters(cost_resolver=resolve_cost)
    with pytest.raises(ExecutionRecoveryError, match="authoritative cost is missing"):
        execute_recoverable_cells(
            controller,
            [cell],
            stage_authorization=authorization,
            adapters=adapters,
            repo_root=tmp_path,
            runner=runner,
        )

    paused = controller.snapshot()
    assert runner_calls == 1
    assert paused.fatal_reasons == ()
    assert len(paused.physical_executions) == 1
    paused_state = next(iter(paused.physical_executions.values()))
    assert paused_state.cost_observed is True
    assert paused_state.cost_authoritative is False
    assert paused.canonical_results == {}

    outcomes = execute_recoverable_cells(
        ExecutionRecoveryController(
            path, controller_id="pending-cost", schedule=schedule
        ),
        [cell],
        stage_authorization=authorization,
        adapters=adapters,
        repo_root=tmp_path,
        runner=runner,
    )

    assert runner_calls == 1
    assert cost_calls == 4
    assert outcomes[0].status == "passed"
    completed = ExecutionRecoveryController(
        path, controller_id="pending-cost", schedule=schedule
    ).snapshot()
    assert completed.complete is True
    assert len(completed.physical_executions) == 1


def test_active_execution_requires_authoritative_recovery_before_replacement(
    tmp_path: Path,
) -> None:
    cell = _cell(tmp_path, 0, run_id="active-recovery")
    plan = LogicalExecutionPlanV1.create(
        logical_attempt_id=cell.attempt_id,
        stage_id="checkpoint",
        stage_ordinal=0,
        admission_block_id="checkpoint",
        block_ordinal=0,
        attempt_ordinal=0,
        admission_weight=1,
        maximum_cost_micro_usd=500_000,
        planned_harbor_resources=("container-active",),
    )
    schedule = _schedule([plan], replacements=1, total=1_000_000, in_flight=500_000)
    path = tmp_path / "events.jsonl"
    controller = ExecutionRecoveryController(
        path, controller_id="active-recovery", schedule=schedule
    )
    authorization = _authorization(schedule)
    controller.authorize_stage(authorization)
    original = controller.admit(stage_id="checkpoint").physical_executions[0]
    controller.started(original, observed_harbor_resources=("container-active",))
    reconciled: list[str] = []
    runner_calls: list[str] = []
    reconciliation_receipt = tmp_path / "runner/recovery.json"
    reconciliation_receipt.parent.mkdir(parents=True)
    reconciliation_receipt.write_text('{"sandbox": "lost"}\n', encoding="utf-8")
    reconciliation_sha256 = hashlib.sha256(
        reconciliation_receipt.read_bytes()
    ).hexdigest()

    def interrupted(_cell: PlannedCell, physical: object):
        reconciled.append(physical.physical_execution_id)
        return InterruptedExecutionObservationV1(
            terminal_kind="sandbox_lost",
            cell_outcome=CellOutcome(
                _cell.id,
                "failed",
                error="runner attested sandbox loss",
                terminal_kind="sandbox_lost",
            ),
            cleanup=_cleanup(physical, ("container-active",)),
            cost=_cost(physical, 25_000),
            reconciliation_receipt_reference="runner/recovery.json",
            reconciliation_receipt_sha256=reconciliation_sha256,
        )

    def runner(*_args: object, env: dict[str, str], **_kwargs: object) -> object:
        runner_calls.append(env["FUGUE_PHYSICAL_EXECUTION_ID"])
        return SimpleNamespace(returncode=0)

    outcomes = execute_recoverable_cells(
        ExecutionRecoveryController(
            path, controller_id="active-recovery", schedule=schedule
        ),
        [cell],
        stage_authorization=authorization,
        adapters=_adapters(
            interrupted_reconciler=interrupted,
            cleanup_verifier=lambda _cell, physical, _outcome: _cleanup(
                physical, ("container-active",)
            ),
        ),
        repo_root=tmp_path,
        runner=runner,
    )

    assert reconciled == [original.physical_execution_id]
    assert len(runner_calls) == 1
    assert runner_calls[0] != original.physical_execution_id
    assert outcomes[0].status == "passed"
    snapshot = ExecutionRecoveryController(
        path, controller_id="active-recovery", schedule=schedule
    ).snapshot()
    assert len(snapshot.physical_executions) == 2
    assert (
        snapshot.physical_executions[original.physical_execution_id].terminal_kind
        == "sandbox_lost"
    )
    original_state = snapshot.physical_executions[original.physical_execution_id]
    assert original_state.result_reference is not None
    original_receipt = json.loads(
        (tmp_path / original_state.result_reference).read_text(encoding="utf-8")
    )
    assert original_receipt["reconciliation_receipt_sha256"] == reconciliation_sha256


def test_completed_active_agent_work_is_adopted_without_reexecution(
    tmp_path: Path,
) -> None:
    cell = _cell(tmp_path, 0, run_id="completed-active")
    plan = LogicalExecutionPlanV1.create(
        logical_attempt_id=cell.attempt_id,
        stage_id="checkpoint",
        stage_ordinal=0,
        admission_block_id="checkpoint",
        block_ordinal=0,
        attempt_ordinal=0,
        admission_weight=1,
        maximum_cost_micro_usd=500_000,
        planned_harbor_resources=("container-completed",),
    )
    schedule = _schedule([plan], replacements=1, total=1_000_000, in_flight=500_000)
    path = tmp_path / "events.jsonl"
    controller = ExecutionRecoveryController(
        path, controller_id="completed-active", schedule=schedule
    )
    authorization = _authorization(schedule)
    controller.authorize_stage(authorization)
    original = controller.admit(stage_id="checkpoint").physical_executions[0]
    controller.started(original, observed_harbor_resources=("container-completed",))

    cell.result_path.write_text('{"answer": "complete"}\n', encoding="utf-8")
    result_sha256 = hashlib.sha256(cell.result_path.read_bytes()).hexdigest()
    reconciliation_receipt = tmp_path / "runner/completed.json"
    reconciliation_receipt.parent.mkdir(parents=True)
    reconciliation_receipt.write_text(
        '{"status": "completed", "exit_code": 0}\n', encoding="utf-8"
    )
    reconciliation_sha256 = hashlib.sha256(
        reconciliation_receipt.read_bytes()
    ).hexdigest()
    runner_calls = 0

    def interrupted(
        _cell: PlannedCell, physical: object
    ) -> InterruptedExecutionObservationV1:
        assert physical.physical_execution_id == original.physical_execution_id
        return InterruptedExecutionObservationV1(
            terminal_kind="success",
            cell_outcome=CellOutcome(
                _cell.id,
                "passed",
                returncode=0,
                runtime_outcome="completed",
                terminal_kind="success",
            ),
            cleanup=_cleanup(physical, ("container-completed",)),
            cost=_cost(physical, 125_000),
            reconciliation_receipt_reference="runner/completed.json",
            reconciliation_receipt_sha256=reconciliation_sha256,
            result_reference=_cell.result_path.relative_to(tmp_path).as_posix(),
            result_sha256=result_sha256,
        )

    def runner(*_args: object, **_kwargs: object) -> object:
        nonlocal runner_calls
        runner_calls += 1
        return SimpleNamespace(returncode=0)

    outcomes = execute_recoverable_cells(
        ExecutionRecoveryController(
            path, controller_id="completed-active", schedule=schedule
        ),
        [cell],
        stage_authorization=authorization,
        adapters=_adapters(interrupted_reconciler=interrupted),
        repo_root=tmp_path,
        runner=runner,
    )

    assert runner_calls == 0
    assert outcomes[0].status == "passed"
    snapshot = ExecutionRecoveryController(
        path, controller_id="completed-active", schedule=schedule
    ).snapshot()
    assert len(snapshot.physical_executions) == 1
    assert snapshot.canonical_results == {
        original.logical_attempt_id: original.physical_execution_id
    }
    adopted_state = snapshot.physical_executions[original.physical_execution_id]
    assert adopted_state.result_reference is not None
    adopted_receipt = json.loads(
        (tmp_path / adopted_state.result_reference).read_text(encoding="utf-8")
    )
    assert adopted_receipt["source_result_sha256"] == result_sha256
    assert adopted_receipt["reconciliation_receipt_sha256"] == reconciliation_sha256


def test_behavioral_terminal_requires_explicit_canonical_proof(
    tmp_path: Path,
) -> None:
    schedule = _schedule([_plan(0, attempt_ordinal=0)], replacements=0, total=2_000_000)
    controller = ExecutionRecoveryController(
        tmp_path / "events.jsonl",
        controller_id="explicit-canonical",
        schedule=schedule,
    )
    physical = _admit(controller).physical_executions[0]
    plan = controller._plan_by_id[physical.logical_attempt_id]
    controller.started(
        physical, observed_harbor_resources=plan.planned_harbor_resources
    )
    controller.finalize(
        physical,
        terminal_kind="success",
        result_reference="terminal.json",
        cleanup=_cleanup(physical, plan.planned_harbor_resources),
        cost=_cost(physical),
        cell_outcome=CellOutcome("cell-explicit", "passed", terminal_kind="success"),
    )
    assert controller.snapshot().canonical_results == {}
    with pytest.raises(ValueError, match="must be verified"):
        CanonicalizationObservationV1(
            kind="test-evidence",
            status="missing",  # type: ignore[arg-type]
            reference="missing.json",
            evidence_digest=stable_digest({"missing": True}),
        )
    controller.select_canonical(physical, _canonical(physical))
    assert controller.snapshot().canonical_results == {
        physical.logical_attempt_id: physical.physical_execution_id
    }


def test_failed_canonical_reconciliation_halts_without_selecting_result(
    tmp_path: Path,
) -> None:
    cell = _cell(tmp_path, 0, run_id="canonical-failure")
    plan = LogicalExecutionPlanV1.create(
        logical_attempt_id=cell.attempt_id,
        stage_id="checkpoint",
        stage_ordinal=0,
        admission_block_id="checkpoint",
        block_ordinal=0,
        attempt_ordinal=0,
        admission_weight=1,
        maximum_cost_micro_usd=500_000,
    )
    schedule = _schedule([plan], replacements=0, total=500_000, in_flight=500_000)
    controller = ExecutionRecoveryController(
        tmp_path / "events.jsonl",
        controller_id="canonical-failure",
        schedule=schedule,
    )

    def cannot_reconcile(*_args: object) -> CanonicalizationObservationV1:
        raise RuntimeError("prediction publication is not finalized")

    with pytest.raises(
        ExecutionRecoveryError, match="canonical evidence reconciliation failed"
    ):
        execute_recoverable_cells(
            controller,
            [cell],
            stage_authorization=_authorization(schedule),
            adapters=_adapters(canonicalization_verifier=cannot_reconcile),
            repo_root=tmp_path,
            runner=lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
        )

    snapshot = controller.snapshot()
    assert snapshot.canonical_results == {}
    state = next(iter(snapshot.physical_executions.values()))
    assert state.fully_reconciled is True
    assert state.canonical is False


def test_runtime_resources_are_discovered_and_exactly_reconciled(
    tmp_path: Path,
) -> None:
    cell = _cell(tmp_path, 0, run_id="runtime-resources")
    plan = LogicalExecutionPlanV1.create(
        logical_attempt_id=cell.attempt_id,
        stage_id="checkpoint",
        stage_ordinal=0,
        admission_block_id="checkpoint",
        block_ordinal=0,
        attempt_ordinal=0,
        admission_weight=1,
        maximum_cost_micro_usd=500_000,
        planned_harbor_resources=("container-planned",),
    )
    schedule = _schedule([plan], replacements=0, total=500_000, in_flight=500_000)
    controller = ExecutionRecoveryController(
        tmp_path / "events.jsonl",
        controller_id="runtime-resources",
        schedule=schedule,
    )

    execute_recoverable_cells(
        controller,
        [cell],
        stage_authorization=_authorization(schedule),
        adapters=_adapters(
            resource_resolver=lambda _cell: (
                "container-planned",
                "network-runtime",
            ),
            cleanup_verifier=lambda _cell, _physical, _outcome: CleanupObservationV1(
                verified=True,
                scope_verified=True,
                post_run_inventory=True,
                discovered_resources=("volume-runtime",),
                inspected_resources=(
                    "container-planned",
                    "network-runtime",
                    "volume-runtime",
                ),
                removed_resources=(
                    "container-planned",
                    "network-runtime",
                    "volume-runtime",
                ),
                receipt_reference="cleanup/runtime-resources.json",
            ),
        ),
        repo_root=tmp_path,
        runner=lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )

    state = next(iter(controller.snapshot().physical_executions.values()))
    assert state.observed_harbor_resources == (
        "container-planned",
        "network-runtime",
    )
    assert state.discovered_harbor_resources == ("volume-runtime",)
    assert state.cleanup_verified is True


@pytest.mark.parametrize(
    ("returncode", "cancelled", "terminal_kind"),
    [(1, False, "task_failure"), (0, True, "cancelled")],
)
def test_no_result_behavioral_terminal_gets_a_durable_receipt(
    tmp_path: Path,
    returncode: int,
    cancelled: bool,
    terminal_kind: str,
) -> None:
    cell = _cell(tmp_path, 0, run_id=f"receipt-{terminal_kind}")
    plan = LogicalExecutionPlanV1.create(
        logical_attempt_id=cell.attempt_id,
        stage_id="checkpoint",
        stage_ordinal=0,
        admission_block_id="checkpoint",
        block_ordinal=0,
        attempt_ordinal=0,
        admission_weight=1,
        maximum_cost_micro_usd=500_000,
    )
    schedule = _schedule([plan], replacements=0, total=500_000, in_flight=500_000)
    controller = ExecutionRecoveryController(
        tmp_path / "events.jsonl",
        controller_id=f"receipt-{terminal_kind}",
        schedule=schedule,
    )
    cancellation = threading.Event()
    if cancelled:
        cancellation.set()

    outcomes = execute_recoverable_cells(
        controller,
        [cell],
        stage_authorization=_authorization(schedule),
        adapters=_adapters(),
        repo_root=tmp_path,
        runner=lambda *_args, **_kwargs: SimpleNamespace(returncode=returncode),
        cancellation_event=cancellation,
    )

    assert outcomes[0].terminal_kind == terminal_kind
    state = next(iter(controller.snapshot().physical_executions.values()))
    assert state.result_reference is not None
    receipt = tmp_path / state.result_reference
    assert receipt.is_file()
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["cell_outcome"]["terminal_kind"] == terminal_kind
    assert payload["source_result_reference"] is None


def test_secrets_are_redacted_from_recovery_and_run_journals(
    tmp_path: Path,
) -> None:
    secret = "sk-ant-not-a-real-secret-value"
    cell = _cell(tmp_path, 0, run_id="secret-redaction")
    plan = LogicalExecutionPlanV1.create(
        logical_attempt_id=cell.attempt_id,
        stage_id="checkpoint",
        stage_ordinal=0,
        admission_block_id="checkpoint",
        block_ordinal=0,
        attempt_ordinal=0,
        admission_weight=1,
        maximum_cost_micro_usd=500_000,
    )
    schedule = _schedule([plan], replacements=0, total=500_000, in_flight=500_000)
    controller = ExecutionRecoveryController(
        tmp_path / "events.jsonl",
        controller_id="secret-redaction",
        schedule=schedule,
    )

    def runner(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError(f"credential was {secret}")

    with pytest.raises(ExecutionRecoveryError, match="execution_failure"):
        execute_recoverable_cells(
            controller,
            [cell],
            stage_authorization=_authorization(schedule),
            adapters=_adapters(redaction_secrets=(secret,)),
            repo_root=tmp_path,
            runner=runner,
        )

    persisted = "\n".join(
        path.read_text(encoding="utf-8") for path in tmp_path.rglob("*.json*")
    )
    assert secret not in persisted
    assert "[redacted]" in persisted


def test_cell_env_secret_cannot_leak_through_adapter_observation(
    tmp_path: Path,
) -> None:
    secret = "cell-env-secret-value-12345"
    cell = _cell(tmp_path, 0, run_id="cell-env-redaction")
    cell.env["ANTHROPIC_API_KEY"] = secret
    plan = LogicalExecutionPlanV1.create(
        logical_attempt_id=cell.attempt_id,
        stage_id="checkpoint",
        stage_ordinal=0,
        admission_block_id="checkpoint",
        block_ordinal=0,
        attempt_ordinal=0,
        admission_weight=1,
        maximum_cost_micro_usd=500_000,
    )
    schedule = _schedule([plan], replacements=0, total=500_000, in_flight=500_000)
    controller = ExecutionRecoveryController(
        tmp_path / "events.jsonl",
        controller_id="cell-env-redaction",
        schedule=schedule,
    )

    with pytest.raises(ExecutionRecoveryError, match="cleanup verifier failed"):
        execute_recoverable_cells(
            controller,
            [cell],
            stage_authorization=_authorization(schedule),
            adapters=_adapters(
                cleanup_verifier=lambda _cell, _physical, _outcome: (
                    CleanupObservationV1(
                        verified=True,
                        scope_verified=True,
                        post_run_inventory=True,
                        receipt_reference=f"cleanup/{secret}.json",
                    )
                )
            ),
            repo_root=tmp_path,
            runner=lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
        )

    persisted = "\n".join(
        path.read_text(encoding="utf-8") for path in tmp_path.rglob("*.json*")
    )
    assert secret not in persisted


@pytest.mark.parametrize(
    "adapter_field",
    [
        "cleanup_resource",
        "cleanup_receipt",
        "cost_receipt",
        "canonical_reference",
    ],
)
def test_adapter_observation_secrets_are_rejected_before_persistence(
    tmp_path: Path,
    adapter_field: str,
) -> None:
    secret = "adapter-secret-value-12345"
    case_root = tmp_path / adapter_field
    schedule = _schedule(
        [_plan(0, attempt_ordinal=0, cost=500_000)],
        replacements=0,
        total=500_000,
        in_flight=500_000,
    )
    controller = ExecutionRecoveryController(
        case_root / "events.jsonl",
        controller_id=f"adapter-{adapter_field}",
        schedule=schedule,
        redaction_secrets=(secret,),
    )
    physical = _admit(controller).physical_executions[0]
    controller.started(physical, observed_harbor_resources=("container-0",))
    controller.terminal(
        physical,
        terminal_kind="success",
        result_reference="result-safe.json",
        cell_outcome=CellOutcome("cell-adapter", "passed", terminal_kind="success"),
    )

    with pytest.raises(ExecutionRecoveryError, match="sensitive data"):
        if adapter_field == "cleanup_resource":
            resource = f"volume-{secret}"
            controller.cleanup(
                physical,
                CleanupObservationV1(
                    verified=True,
                    scope_verified=True,
                    post_run_inventory=True,
                    discovered_resources=(resource,),
                    inspected_resources=("container-0", resource),
                    removed_resources=("container-0", resource),
                    receipt_reference="cleanup/safe.json",
                ),
            )
        elif adapter_field == "cleanup_receipt":
            controller.cleanup(
                physical,
                CleanupObservationV1(
                    verified=True,
                    scope_verified=True,
                    post_run_inventory=True,
                    inspected_resources=("container-0",),
                    removed_resources=("container-0",),
                    receipt_reference=f"cleanup/{secret}.json",
                ),
            )
        elif adapter_field == "cost_receipt":
            controller.cleanup(physical, _cleanup(physical, ("container-0",)))
            controller.cost(
                physical,
                CostObservationV1(
                    actual_cost_micro_usd=100_000,
                    authoritative=True,
                    source="safe-ledger",
                    receipt_reference=f"cost/{secret}.json",
                ),
            )
        else:
            controller.cleanup(physical, _cleanup(physical, ("container-0",)))
            controller.cost(physical, _cost(physical))
            controller.select_canonical(
                physical,
                CanonicalizationObservationV1(
                    kind="safe-evidence",
                    status="verified",
                    reference=f"evidence/{secret}.json",
                    evidence_digest=stable_digest({"safe": True}),
                ),
            )

    persisted = (case_root / "events.jsonl").read_text(encoding="utf-8")
    assert secret not in persisted


def test_interrupted_reconciliation_secret_reference_is_rejected(
    tmp_path: Path,
) -> None:
    secret = "interrupt-secret-value-12345"
    cell = _cell(tmp_path, 0, run_id="interrupted-secret")
    plan = LogicalExecutionPlanV1.create(
        logical_attempt_id=cell.attempt_id,
        stage_id="checkpoint",
        stage_ordinal=0,
        admission_block_id="checkpoint",
        block_ordinal=0,
        attempt_ordinal=0,
        admission_weight=1,
        maximum_cost_micro_usd=500_000,
    )
    schedule = _schedule([plan], replacements=1, total=1_000_000, in_flight=500_000)
    path = tmp_path / "events.jsonl"
    controller = ExecutionRecoveryController(
        path,
        controller_id="interrupted-secret",
        schedule=schedule,
    )
    authorization = _authorization(schedule)
    controller.authorize_stage(authorization)
    physical = controller.admit(stage_id="checkpoint").physical_executions[0]
    controller.started(physical, observed_harbor_resources=())
    receipt = tmp_path / f"runner/{secret}.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"status": "lost"}\n', encoding="utf-8")
    receipt_sha256 = hashlib.sha256(receipt.read_bytes()).hexdigest()

    def interrupted(
        _cell: PlannedCell, observed: object
    ) -> InterruptedExecutionObservationV1:
        return InterruptedExecutionObservationV1(
            terminal_kind="sandbox_lost",
            cell_outcome=CellOutcome(
                _cell.id,
                "failed",
                terminal_kind="sandbox_lost",
            ),
            cleanup=_cleanup(observed),
            cost=_cost(observed, 10_000),
            reconciliation_receipt_reference=(f"runner/{secret}.json"),
            reconciliation_receipt_sha256=receipt_sha256,
        )

    with pytest.raises(ExecutionRecoveryError, match="sensitive data"):
        execute_recoverable_cells(
            ExecutionRecoveryController(
                path,
                controller_id="interrupted-secret",
                schedule=schedule,
            ),
            [cell],
            stage_authorization=authorization,
            adapters=_adapters(
                interrupted_reconciler=interrupted,
                redaction_secrets=(secret,),
            ),
            repo_root=tmp_path,
            runner=lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
        )

    assert secret not in path.read_text(encoding="utf-8")


def test_interrupted_reconciler_exception_is_redacted_and_halts(
    tmp_path: Path,
) -> None:
    secret = "reconciler-secret-value-12345"
    cell = _cell(tmp_path, 0, run_id="reconciler-error")
    cell.env["ANTHROPIC_API_KEY"] = secret
    plan = LogicalExecutionPlanV1.create(
        logical_attempt_id=cell.attempt_id,
        stage_id="checkpoint",
        stage_ordinal=0,
        admission_block_id="checkpoint",
        block_ordinal=0,
        attempt_ordinal=0,
        admission_weight=1,
        maximum_cost_micro_usd=500_000,
    )
    schedule = _schedule([plan], replacements=1, total=1_000_000, in_flight=500_000)
    path = tmp_path / "events.jsonl"
    controller = ExecutionRecoveryController(
        path, controller_id="reconciler-error", schedule=schedule
    )
    authorization = _authorization(schedule)
    controller.authorize_stage(authorization)
    physical = controller.admit(stage_id="checkpoint").physical_executions[0]
    controller.started(physical, observed_harbor_resources=())

    def interrupted(*_args: object) -> InterruptedExecutionObservationV1:
        raise RuntimeError(f"runner credential was {secret}")

    with pytest.raises(ExecutionRecoveryError) as raised:
        execute_recoverable_cells(
            ExecutionRecoveryController(
                path, controller_id="reconciler-error", schedule=schedule
            ),
            [cell],
            stage_authorization=authorization,
            adapters=_adapters(interrupted_reconciler=interrupted),
            repo_root=tmp_path,
            runner=lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
        )

    assert secret not in str(raised.value)
    persisted = "\n".join(
        item.read_text(encoding="utf-8") for item in tmp_path.rglob("*.json*")
    )
    assert secret not in persisted
    assert "[redacted]" in persisted
