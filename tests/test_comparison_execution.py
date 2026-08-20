from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from fugue.bench.candidates import stable_digest
from fugue.bench.comparison_execution import (
    ComparisonExecutionBindingV1,
    _infrastructure_attempt_cost,
    _local_attempt_cost,
    _physical_runner_terminal_observation,
    compile_comparison_execution_binding,
    execute_durable_comparison_cells,
    execution_stage_authorizations,
    recovery_journal_path,
    verify_comparison_execution_binding,
    verify_resume_stage_authorizations,
)
from fugue.bench.execution import (
    CellOutcome,
    PlannedCell,
    _harbor_job_result,
    _write_physical_runner_terminal_observation,
)
from fugue.bench.execution_recovery import (
    ExecutionRecoveryController,
    PhysicalExecutionIdentityV1,
    _archive_noncanonical_result,
    _materialize_physical_harbor_cell,
)
from fugue.bench.harbor_terminal import DurableHarborTerminalPlugin
from fugue.bench.operator import _physical_harbor_cell_from_journal


def _physical_runner_terminal_outcome(
    *,
    repo_root: Path,
    cell: PlannedCell,
    physical: PhysicalExecutionIdentityV1,
) -> CellOutcome | None:
    observation = _physical_runner_terminal_observation(
        repo_root=repo_root,
        cell=cell,
        physical=physical,
    )
    return observation[0] if observation is not None else None


def _attempt(task: str, arm: str, trial: int = 0) -> dict[str, object]:
    return {
        "attempt_id": stable_digest(
            {"task": task, "arm": arm, "trial": trial}
        ),
        "task_id": task,
        "variant_id": arm,
        "harness": "claude-code",
        "trial_index": trial,
        "applicable": True,
    }


def test_schedule_selects_a_complete_contrast_for_serial_checkpoint() -> None:
    cells = [
        _attempt("task-a", "baseline"),
        _attempt("task-a", "candidate"),
        _attempt("task-b", "baseline"),
        _attempt("task-b", "candidate"),
    ]

    binding = compile_comparison_execution_binding(
        comparison_id="pair-complete",
        expected_cells=cells,
        concurrency=2,
        checkpoint_cells=1,
        maximum_cost_usd=10,
        reserve_per_attempt_usd=2,
        maximum_infrastructure_replacements=1,
    )

    plans = binding.schedule.logical_attempts
    assert binding.checkpoint_cell_count == 2
    assert [item.stage_id for item in plans] == [
        "checkpoint",
        "checkpoint",
        "main",
        "main",
    ]
    assert [item.block_ordinal for item in plans[:2]] == [0, 1]
    assert [item.admission_block_id for item in plans[:2]] == [
        "checkpoint-0001",
        "checkpoint-0002",
    ]
    assert [item.admission_block_id for item in plans[2:]] == [
        "main-wave-0001",
        "main-wave-0001",
    ]
    assert [item.attempt_ordinal for item in plans[2:]] == [0, 1]
    assert binding.schedule.maximum_physical_executions == 5
    authorizations = execution_stage_authorizations(
        binding,
        preview_digest="a" * 64,
        approval_digest="b" * 64,
    )
    assert all(
        item.maximum_cost_micro_usd
        == binding.schedule.maximum_total_micro_usd
        for item in authorizations
    )


def test_execution_binding_fails_closed_after_schedule_tampering() -> None:
    cells = [_attempt("task-a", "baseline"), _attempt("task-a", "candidate")]
    binding = compile_comparison_execution_binding(
        comparison_id="tamper-check",
        expected_cells=cells,
        concurrency=1,
        checkpoint_cells=1,
        maximum_cost_usd=4,
        reserve_per_attempt_usd=2,
        maximum_infrastructure_replacements=0,
    )
    raw = binding.to_dict()
    raw["wave_size"] = 2

    with pytest.raises(ValueError, match="binding digest"):
        ComparisonExecutionBindingV1.from_dict(raw)

    wrong_cells = [*cells, _attempt("task-b", "candidate")]
    with pytest.raises(ValueError, match="does not cover exact cells"):
        verify_comparison_execution_binding(
            binding.to_dict(), expected_cells=wrong_cells
        )


def test_resume_requires_complete_original_stage_authorizations(
    tmp_path: Path,
) -> None:
    cells = [
        _attempt("task-a", "baseline"),
        _attempt("task-a", "candidate"),
        _attempt("task-b", "baseline"),
        _attempt("task-b", "candidate"),
    ]
    binding = compile_comparison_execution_binding(
        comparison_id="renewed-consent",
        expected_cells=cells,
        concurrency=2,
        checkpoint_cells=1,
        maximum_cost_usd=10,
        reserve_per_attempt_usd=2,
        maximum_infrastructure_replacements=1,
    )
    missing_run = "missing-authorization-journal"
    missing_journal = recovery_journal_path(tmp_path, missing_run)
    with pytest.raises(ValueError, match="existing immutable"):
        verify_resume_stage_authorizations(
            repo_root=tmp_path,
            run_id=missing_run,
            binding=binding,
            preview_digest="a" * 64,
            approval_digest="b" * 64,
        )
    assert not missing_journal.exists()

    run_id = "renewed-consent-run"
    controller = ExecutionRecoveryController(
        recovery_journal_path(tmp_path, run_id),
        controller_id=f"comparison-{run_id}",
        schedule=binding.schedule,
    )
    original_approval = "b" * 64
    authorizations = execution_stage_authorizations(
        binding,
        preview_digest="a" * 64,
        approval_digest=original_approval,
    )
    controller.authorize_stage(authorizations[0])

    with pytest.raises(ValueError, match="complete exact stage authorization"):
        verify_resume_stage_authorizations(
            repo_root=tmp_path,
            run_id=run_id,
            binding=binding,
            preview_digest="a" * 64,
            approval_digest=original_approval,
        )

    for authorization in authorizations[1:]:
        controller.authorize_stage(authorization)
    verify_resume_stage_authorizations(
        repo_root=tmp_path,
        run_id=run_id,
        binding=binding,
        preview_digest="a" * 64,
        approval_digest=original_approval,
    )

    with pytest.raises(ValueError, match="complete exact stage authorization"):
        verify_resume_stage_authorizations(
            repo_root=tmp_path,
            run_id=run_id,
            binding=binding,
            preview_digest="a" * 64,
            approval_digest="c" * 64,
        )


def test_actual_cost_requires_authoritative_local_usage() -> None:
    resolved = SimpleNamespace(
        receipts={
            "usage": {"status": "passed", "payload": {"cost_usd": 1.25}}
        }
    )
    unavailable = SimpleNamespace(
        receipts={"usage": {"status": "unavailable", "payload": {}}}
    )
    store = SimpleNamespace(read_attempt=lambda _attempt_id: resolved)
    missing_store = SimpleNamespace(read_attempt=lambda _attempt_id: unavailable)

    assert _local_attempt_cost(store, "a" * 64) == (1_250_000, True)
    assert _local_attempt_cost(missing_store, "a" * 64) == (None, False)

    conservatively_accounted = SimpleNamespace(
        receipts={
            "usage": {
                "status": "passed",
                "payload": {
                    "cost_usd": 1.25,
                    "judge_cost_required": True,
                    "judge_cost_complete": False,
                    "judge_cost_usd": None,
                    "judge_accounted_cost_complete": True,
                    "judge_accounted_cost_usd": 0.1,
                },
            }
        }
    )
    accounted_store = SimpleNamespace(
        read_attempt=lambda _attempt_id: conservatively_accounted
    )
    assert _local_attempt_cost(accounted_store, "a" * 64) == (1_350_000, True)


def _cell(tmp_path: Path, task: str, arm: str) -> PlannedCell:
    return PlannedCell(
        id=f"{task}-{arm}",
        run_id="checkpoint-gate",
        run_name="checkpoint-gate",
        workload_id="harbor",
        task_id=task,
        harness="claude-code",
        context_system_id="none",
        variant_id=arm,
        model_provider="anthropic",
        model="claude-sonnet-5",
        trial_index=1,
        comparison_example_id=stable_digest({"task": task}),
        candidate_id=stable_digest({"arm": arm}),
        execution_fingerprint=stable_digest({"runtime": "harbor"}),
        config_path=tmp_path / f"{task}-{arm}.json",
        result_path=tmp_path / f"{task}-{arm}-result.json",
        command=("fake-harbor",),
        env={},
        n_attempts=1,
    )


def test_checkpoint_post_local_gate_stops_main_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cells = [
        _cell(tmp_path, "task-a", "baseline"),
        _cell(tmp_path, "task-a", "candidate"),
        _cell(tmp_path, "task-b", "baseline"),
        _cell(tmp_path, "task-b", "candidate"),
    ]
    expected = [
        {
            "attempt_id": cell.attempt_id,
            "task_id": cell.task_id,
            "variant_id": cell.variant_id,
            "harness": cell.harness,
            "trial_index": cell.trial_index,
            "applicable": True,
        }
        for cell in cells
    ]
    binding = compile_comparison_execution_binding(
        comparison_id="post-local-gate",
        expected_cells=expected,
        concurrency=2,
        checkpoint_cells=1,
        maximum_cost_usd=10,
        reserve_per_attempt_usd=2,
        maximum_infrastructure_replacements=0,
    )
    approved = {
        "preview_digest": "a" * 64,
        "approval_digest": "b" * 64,
        "expected_cells": expected,
        "execution_schedule": binding.to_dict(),
        "execution_schedule_digest": binding.binding_digest,
    }
    admitted_stages: list[str] = []
    finalized_local: list[str] = []

    def fake_execute(
        _controller: object,
        all_cells: list[PlannedCell],
        *,
        stage_authorization: object,
        wave_lifecycle_factory: object,
        **_kwargs: object,
    ) -> list[CellOutcome]:
        stage_id = stage_authorization.stage_id  # type: ignore[attr-defined]
        admitted_stages.append(stage_id)
        stage_attempts = {
            item.logical_attempt_id
            for item in binding.schedule.logical_attempts
            if item.stage_id == stage_id
        }
        wave = tuple(
            cell for cell in all_cells if cell.attempt_id in stage_attempts
        )
        lifecycle = wave_lifecycle_factory(wave)  # type: ignore[operator]
        outcomes = tuple(
            CellOutcome(cell.id, "passed", terminal_kind="success")
            for cell in wave
        )
        for cell, outcome in zip(wave, outcomes, strict=True):
            lifecycle.finish_cell(cell, outcome)
        lifecycle.finalize(outcomes)
        return list(outcomes)

    monkeypatch.setattr(
        "fugue.bench.comparison_execution.execute_recoverable_cells",
        fake_execute,
    )

    def reject_checkpoint(_outcomes: tuple[CellOutcome, ...]) -> None:
        checkpoint_ids = {
            item.logical_attempt_id
            for item in binding.schedule.logical_attempts
            if item.stage_id == "checkpoint"
        }
        assert set(finalized_local) == checkpoint_ids
        raise RuntimeError("checkpoint source drift")

    with pytest.raises(RuntimeError, match="checkpoint source drift"):
        execute_durable_comparison_cells(
            repo_root=tmp_path,
            run_id="checkpoint-gate",
            cells=cells,
            approved_comparison=approved,
            runner=None,
            begin_cell=lambda _cell: None,
            finish_cell=lambda cell, _outcome: finalized_local.append(
                cell.attempt_id
            ),
            invalidate_cell=None,
            cell_conformance=lambda _cell: {},
            cancellation_event=None,
            secret_values=(),
            finalize_wave=reject_checkpoint,
        )

    assert admitted_stages == ["checkpoint"]


def test_retry_cost_uses_terminal_bound_archived_physical_result(
    tmp_path: Path,
) -> None:
    run_id = "partial-cost"
    base = _cell(tmp_path, "partial", "baseline")
    logical_id = base.attempt_id
    physical = PhysicalExecutionIdentityV1.create(
        logical_attempt_id=logical_id,
        controller_id="comparison-partial-cost",
        retry_ordinal=0,
    )
    root = (
        tmp_path
        / ".fugue"
        / "runtime"
        / run_id
        / "physical-executions"
        / physical.physical_execution_id
    )
    harbor = root / "harbor"
    harbor.mkdir(parents=True)
    original = harbor / "jobs" / "job" / "result.json"
    config = {
        "fugue": {
            "physical_execution": {
                "physical_execution_id": physical.physical_execution_id,
                "logical_attempt_id": logical_id,
                "retry_ordinal": 0,
                "harbor_result_path": original.as_posix(),
            }
        }
    }
    (harbor / "config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    cell = replace(
        base,
        run_id=run_id,
        config_path=harbor / "config.json",
        result_path=original,
        physical_execution_id=physical.physical_execution_id,
        retry_ordinal=0,
        config_sha256=hashlib.sha256(
            (harbor / "config.json").read_bytes()
        ).hexdigest(),
    )
    archived = root / "result.json"
    archived.write_text(
        json.dumps({"stats": {"cost_usd": 0.42}}),
        encoding="utf-8",
    )
    terminal = {
        "physical_execution_id": physical.physical_execution_id,
        "source_result_reference": archived.relative_to(tmp_path).as_posix(),
        "source_result_sha256": hashlib.sha256(archived.read_bytes()).hexdigest(),
    }
    (root / "terminal.json").write_text(
        json.dumps(terminal), encoding="utf-8"
    )

    assert _infrastructure_attempt_cost(
        repo_root=tmp_path,
        run_id=run_id,
        cell=cell,
        physical=physical,
        terminal_kind="sandbox_lost",
    ) == (420_000, True, "physical-harbor-result")

    # Crash after archive but before durable terminal.json: the earlier
    # runner-exit receipt still binds the exact source digest and cost.
    (root / "terminal.json").unlink()
    runner_unsigned = {
        "schema_version": 1,
        "logical_attempt_id": logical_id,
        "physical_execution_id": physical.physical_execution_id,
        "retry_ordinal": 0,
        "config_sha256": hashlib.sha256(
            (harbor / "config.json").read_bytes()
        ).hexdigest(),
        "result_reference": original.relative_to(tmp_path).as_posix(),
        "result_sha256": hashlib.sha256(archived.read_bytes()).hexdigest(),
        "cell_outcome": {
            "cell_id": "partial",
            "status": "failed",
            "returncode": 1,
            "error": "sandbox lost",
            "benchmark_outcome": "unscored",
            "reward": None,
            "runtime_outcome": "not_started",
            "terminal_kind": "sandbox_lost",
        },
    }
    (root / "runner-terminal.json").write_text(
        json.dumps(
            {
                **runner_unsigned,
                "receipt_digest": stable_digest(runner_unsigned),
            }
        ),
        encoding="utf-8",
    )
    assert _infrastructure_attempt_cost(
        repo_root=tmp_path,
        run_id=run_id,
        cell=cell,
        physical=physical,
        terminal_kind="sandbox_lost",
    ) == (420_000, True, "physical-harbor-result")

    runner_unsigned["result_sha256"] = "0" * 64
    (root / "runner-terminal.json").write_text(
        json.dumps(
            {
                **runner_unsigned,
                "receipt_digest": stable_digest(runner_unsigned),
            }
        ),
        encoding="utf-8",
    )
    assert _infrastructure_attempt_cost(
        repo_root=tmp_path,
        run_id=run_id,
        cell=cell,
        physical=physical,
        terminal_kind="sandbox_lost",
    ) == (None, False, "physical-harbor-result")


def test_interrupted_result_requires_bound_runner_exit_observation(
    tmp_path: Path,
) -> None:
    base = _cell(tmp_path, "task-exit", "baseline")
    physical = PhysicalExecutionIdentityV1.create(
        logical_attempt_id=base.attempt_id,
        controller_id="comparison-runner-exit",
        retry_ordinal=0,
    )
    result = tmp_path / "physical-result.json"
    result.write_text(json.dumps({"stats": {"evals": {}}}), encoding="utf-8")
    cell = replace(
        base,
        result_path=result,
        physical_execution_id=physical.physical_execution_id,
        retry_ordinal=0,
        config_sha256="c" * 64,
    )
    # A valid-looking Harbor result is insufficient by itself.
    assert _physical_runner_terminal_outcome(
        repo_root=tmp_path,
        cell=cell,
        physical=physical,
    ) is None

    observed = CellOutcome(
        cell.id,
        "failed",
        returncode=7,
        error=None,
        benchmark_outcome="unscored",
        runtime_outcome="completed",
        terminal_kind="task_failure",
    )
    _write_physical_runner_terminal_observation(
        repo_root=tmp_path,
        cell=cell,
        outcome=observed,
    )
    recovered = _physical_runner_terminal_outcome(
        repo_root=tmp_path,
        cell=cell,
        physical=physical,
    )
    assert recovered is not None
    assert recovered.status == "failed"
    assert recovered.returncode == 7
    assert recovered.terminal_kind == "task_failure"

    # Crash after the runner-bound result is archived but before the durable
    # controller terminal receipt: recovery reuses the exact archive.
    archived = (
        tmp_path
        / ".fugue"
        / "runtime"
        / cell.run_id
        / "physical-executions"
        / physical.physical_execution_id
        / result.name
    )
    archived.parent.mkdir(parents=True, exist_ok=True)
    result.replace(archived)
    expected_reference = archived.relative_to(tmp_path).as_posix()
    assert _archive_noncanonical_result(
        repo_root=tmp_path,
        cell=cell,
        physical=physical,
    ) == expected_reference
    assert _archive_noncanonical_result(
        repo_root=tmp_path,
        cell=cell,
        physical=physical,
    ) == expected_reference


def test_harbor_terminal_hook_recovers_without_parent_process_receipt(
    tmp_path: Path,
) -> None:
    base = _cell(tmp_path, "task-hook", "baseline")
    physical = PhysicalExecutionIdentityV1.create(
        logical_attempt_id=base.attempt_id,
        controller_id="comparison-harbor-hook",
        retry_ordinal=0,
    )
    root = (
        tmp_path
        / ".fugue"
        / "runtime"
        / base.run_id
        / "physical-executions"
        / physical.physical_execution_id
    )
    harbor = root / "harbor"
    result = harbor / "jobs" / "job" / "result.json"
    result.parent.mkdir(parents=True)
    config_path = harbor / "config.json"
    config = {
        "fugue": {
            "physical_execution": {
                "logical_attempt_id": physical.logical_attempt_id,
                "physical_execution_id": physical.physical_execution_id,
                "retry_ordinal": 0,
                "harbor_result_path": result.as_posix(),
            }
        }
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    cell = replace(
        base,
        config_path=config_path,
        result_path=result,
        physical_execution_id=physical.physical_execution_id,
        retry_ordinal=0,
        config_sha256=config_sha,
    )
    receipt = root / "harbor-terminal.json"
    plugin = DurableHarborTerminalPlugin(
        logical_attempt_id=cell.attempt_id,
        physical_execution_id=physical.physical_execution_id,
        retry_ordinal="0",
        cell_id=cell.id,
        config_path=config_path.as_posix(),
        config_sha256=config_sha,
        result_path=result.as_posix(),
        receipt_path=receipt.as_posix(),
    )

    class FakeJob:
        _job_result_path = result

        def __init__(self) -> None:
            self.callback: object | None = None

        def __len__(self) -> int:
            return 1

        def on_trial_ended(self, callback: object) -> None:
            self.callback = callback

    job = FakeJob()
    asyncio.run(plugin.on_job_start(job))
    result.write_text(
        json.dumps(
            {
                "n_total_trials": 1,
                "stats": {
                    "n_completed_trials": 1,
                    "n_running_trials": 0,
                    "n_pending_trials": 0,
                    "n_errored_trials": 0,
                    "n_cancelled_trials": 0,
                    "evals": {
                        "agent__dataset": {
                            "reward_stats": {"reward": {"1": ["trial-1"]}}
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    class FakeTrial:
        id = "00000000-0000-0000-0000-000000000001"
        trial_name = "trial-1"
        finished_at = "2026-01-01T00:00:00Z"

        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            return {
                "id": self.id,
                "trial_name": self.trial_name,
                "finished_at": self.finished_at,
            }

    assert job.callback is not None
    asyncio.run(job.callback(SimpleNamespace(result=FakeTrial())))  # type: ignore[operator]

    recovered = _physical_runner_terminal_outcome(
        repo_root=tmp_path,
        cell=cell,
        physical=physical,
    )
    assert recovered is not None
    assert recovered.status == "passed"
    assert recovered.benchmark_outcome == "passed"
    assert recovered.reward == 1.0
    assert not (root / "runner-terminal.json").exists()
    observation = _physical_runner_terminal_observation(
        repo_root=tmp_path,
        cell=cell,
        physical=physical,
    )
    assert observation is not None
    assert observation[1] == result.resolve()
    parent_result = _harbor_job_result(cell, tmp_path)
    assert parent_result.terminal_kind == "success"
    assert parent_result.benchmark_outcome == "passed"

    # Harbor rewrites the aggregate JobResult after the per-trial END hook.
    # The final bytes may differ while the terminal semantics remain exact.
    final_job = json.loads(result.read_text(encoding="utf-8"))
    final_job["finished_at"] = "2026-01-01T00:00:01Z"
    result.write_text(json.dumps(final_job), encoding="utf-8")
    rewritten = _physical_runner_terminal_observation(
        repo_root=tmp_path,
        cell=cell,
        physical=physical,
    )
    assert rewritten is not None
    assert rewritten[1] == result.resolve()
    assert _harbor_job_result(cell, tmp_path).terminal_kind == "success"

    (root / "harbor-terminal-result.json").write_text(
        json.dumps({"tampered": True}), encoding="utf-8"
    )
    assert (
        _physical_runner_terminal_outcome(
            repo_root=tmp_path,
            cell=cell,
            physical=physical,
        )
        is None
    )
def test_resultless_runner_start_failure_is_authoritatively_recovered(
    tmp_path: Path,
) -> None:
    base = _cell(tmp_path, "task-start", "baseline")
    physical = PhysicalExecutionIdentityV1.create(
        logical_attempt_id=base.attempt_id,
        controller_id="comparison-runner-start",
        retry_ordinal=0,
    )
    cell = replace(
        base,
        physical_execution_id=physical.physical_execution_id,
        retry_ordinal=0,
        config_sha256="d" * 64,
    )
    observed = CellOutcome(
        cell.id,
        "failed",
        returncode=None,
        error="runner did not start",
        benchmark_outcome="unscored",
        runtime_outcome="not_started",
        terminal_kind="runner_start_failure",
    )
    _write_physical_runner_terminal_observation(
        repo_root=tmp_path,
        cell=cell,
        outcome=observed,
    )

    recovered = _physical_runner_terminal_outcome(
        repo_root=tmp_path,
        cell=cell,
        physical=physical,
    )

    assert recovered is not None
    assert recovered.runtime_outcome == "not_started"
    assert recovered.terminal_kind == "runner_start_failure"
    assert not cell.result_path.exists()


def test_started_runner_pre_agent_failure_keeps_cost_unavailable(
    tmp_path: Path,
) -> None:
    base = _cell(tmp_path, "task-plugin-import", "baseline")
    physical = PhysicalExecutionIdentityV1.create(
        logical_attempt_id=base.attempt_id,
        controller_id="comparison-plugin-import",
        retry_ordinal=0,
    )
    cell = replace(
        base,
        physical_execution_id=physical.physical_execution_id,
        retry_ordinal=0,
        config_sha256="a" * 64,
    )
    observed = CellOutcome(
        cell.id,
        "failed",
        returncode=1,
        error="Harbor exited before Agent evidence",
        benchmark_outcome="unscored",
        runtime_outcome="not_started",
        terminal_kind="runner_start_failure",
    )
    _write_physical_runner_terminal_observation(
        repo_root=tmp_path,
        cell=cell,
        outcome=observed,
    )

    recovered = _physical_runner_terminal_observation(
        repo_root=tmp_path,
        cell=cell,
        physical=physical,
    )
    cost, authoritative, source = _infrastructure_attempt_cost(
        repo_root=tmp_path,
        run_id=cell.run_id,
        cell=cell,
        physical=physical,
        terminal_kind="runner_start_failure",
    )

    assert recovered is not None
    assert recovered[0].terminal_kind == "runner_start_failure"
    assert recovered[0].runtime_outcome == "not_started"
    assert (cost, authoritative, source) == (
        None,
        False,
        "physical-harbor-result",
    )
    receipt_path = (
        tmp_path
        / ".fugue"
        / "runtime"
        / cell.run_id
        / "physical-executions"
        / physical.physical_execution_id
        / "runner-terminal.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["runner_start_evidence"][
        "observed_agent_session_or_tool_artifacts"
    ] = ["trial/agent/trajectory.json"]
    receipt["receipt_digest"] = stable_digest(
        {key: value for key, value in receipt.items() if key != "receipt_digest"}
    )
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    assert (
        _physical_runner_terminal_observation(
            repo_root=tmp_path,
            cell=cell,
            physical=physical,
        )
        is None
    )


@pytest.mark.parametrize(
    ("terminal_kind", "status", "runtime_outcome"),
    [
        ("agent_timeout", "failed", "timed_out"),
        ("cancelled", "cancelled", "cancelled"),
    ],
)
def test_resultless_behavioral_terminal_materializes_bound_result(
    tmp_path: Path,
    terminal_kind: str,
    status: str,
    runtime_outcome: str,
) -> None:
    base = _cell(tmp_path, f"task-{terminal_kind}", "baseline")
    physical = PhysicalExecutionIdentityV1.create(
        logical_attempt_id=base.attempt_id,
        controller_id=f"comparison-{terminal_kind}",
        retry_ordinal=0,
    )
    physical_root = (
        tmp_path
        / ".fugue"
        / "runtime"
        / base.run_id
        / "physical-executions"
        / physical.physical_execution_id
    )
    cell = replace(
        base,
        result_path=physical_root / "result.json",
        physical_execution_id=physical.physical_execution_id,
        retry_ordinal=0,
        config_sha256="e" * 64,
    )
    observed = CellOutcome(
        cell.id,
        status,  # type: ignore[arg-type]
        returncode=124 if terminal_kind == "agent_timeout" else 130,
        error=terminal_kind,
        benchmark_outcome="unscored",
        runtime_outcome=runtime_outcome,  # type: ignore[arg-type]
        terminal_kind=terminal_kind,  # type: ignore[arg-type]
    )
    _write_physical_runner_terminal_observation(
        repo_root=tmp_path,
        cell=cell,
        outcome=observed,
    )

    recovered = _physical_runner_terminal_observation(
        repo_root=tmp_path,
        cell=cell,
        physical=physical,
    )

    assert recovered is not None
    assert recovered[0].terminal_kind == terminal_kind
    assert recovered[1] == cell.result_path.resolve()
    payload = json.loads(cell.result_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "fugue_resultless_behavioral_terminal"
    assert payload["terminal_kind"] == terminal_kind
    assert (
        _physical_runner_terminal_observation(
            repo_root=tmp_path,
            cell=cell,
            physical=physical,
        )[1]
        == cell.result_path.resolve()
    )


@pytest.mark.parametrize(
    ("terminal_kind", "status", "runtime_outcome"),
    [
        ("agent_timeout", "failed", "timed_out"),
        ("cancelled", "cancelled", "cancelled"),
    ],
)
def test_crash_after_local_finalization_preserves_behavioral_terminal_and_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_kind: str,
    status: str,
    runtime_outcome: str,
) -> None:
    cells = [
        _cell(tmp_path, "task-crash", "baseline"),
        _cell(tmp_path, "task-crash", "candidate"),
    ]
    expected = [
        {
            "attempt_id": cell.attempt_id,
            "task_id": cell.task_id,
            "variant_id": cell.variant_id,
            "harness": cell.harness,
            "trial_index": cell.trial_index,
            "applicable": True,
        }
        for cell in cells
    ]
    binding = compile_comparison_execution_binding(
        comparison_id=f"crash-{terminal_kind}",
        expected_cells=expected,
        concurrency=1,
        checkpoint_cells=1,
        maximum_cost_usd=10,
        reserve_per_attempt_usd=2,
        maximum_infrastructure_replacements=0,
    )
    approved = {
        "preview_digest": "a" * 64,
        "approval_digest": "b" * 64,
        "execution_authorization_digest": "b" * 64,
        "expected_cells": expected,
        "execution_schedule": binding.to_dict(),
        "execution_schedule_digest": binding.binding_digest,
    }
    record = SimpleNamespace(
        integrity_status="resolved",
        terminal_status=status,
        record_digest="c" * 64,
        receipts={"usage": {"status": "unavailable", "payload": {}}},
    )

    class FakeStore:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def read_attempt(self, _attempt_id: str) -> object:
            return record

    monkeypatch.setattr(
        "fugue.bench.comparison_execution.LocalEvidenceStore", FakeStore
    )

    class Reconciled(RuntimeError):
        pass

    def fake_execute(
        _controller: object,
        all_cells: list[PlannedCell],
        *,
        adapters: object,
        **_kwargs: object,
    ) -> list[CellOutcome]:
        base = all_cells[0]
        physical = PhysicalExecutionIdentityV1.create(
            logical_attempt_id=base.attempt_id,
            controller_id=f"comparison-{base.run_id}",
            retry_ordinal=0,
        )
        physical_root = (
            tmp_path
            / ".fugue"
            / "runtime"
            / base.run_id
            / "physical-executions"
            / physical.physical_execution_id
        )
        config = physical_root / "harbor" / "config.json"
        config.parent.mkdir(parents=True)
        config.write_text("{}", encoding="utf-8")
        cell = replace(
            base,
            config_path=config,
            result_path=physical_root / "result.json",
            physical_execution_id=physical.physical_execution_id,
            retry_ordinal=0,
            config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        )
        outcome = CellOutcome(
            cell.id,
            status,  # type: ignore[arg-type]
            returncode=124 if terminal_kind == "agent_timeout" else 130,
            error=terminal_kind,
            benchmark_outcome="unscored",
            runtime_outcome=runtime_outcome,  # type: ignore[arg-type]
            terminal_kind=terminal_kind,  # type: ignore[arg-type]
        )
        _write_physical_runner_terminal_observation(
            repo_root=tmp_path,
            cell=cell,
            outcome=outcome,
        )
        observation = adapters.interrupted_reconciler(cell, physical)  # type: ignore[attr-defined]
        assert observation is not None
        assert observation.terminal_kind == terminal_kind
        assert observation.cell_outcome.status == status
        assert observation.cost.authoritative is True
        assert observation.cost.actual_cost_micro_usd == 2_000_000
        assert (
            observation.cost.source
            == "preleased-behavioral-reserve-upper-bound"
        )
        assert observation.result_reference == cell.result_path.relative_to(
            tmp_path
        ).as_posix()
        raise Reconciled

    monkeypatch.setattr(
        "fugue.bench.comparison_execution.execute_recoverable_cells",
        fake_execute,
    )
    cleanup = {
        "status": "passed",
        "docker_cleanup": {
            "status": "passed",
            "scope": {
                "kind": "exact_physical_compose_projects",
                "run_id": cells[0].run_id,
                "compose_projects": ["fugue-test"],
            },
            "matched_containers": [],
            "matched_networks": [],
        },
    }

    with pytest.raises(Reconciled):
        execute_durable_comparison_cells(
            repo_root=tmp_path,
            run_id=cells[0].run_id,
            cells=cells,
            approved_comparison=approved,
            runner=None,
            begin_cell=lambda _cell: None,
            finish_cell=lambda _cell, _outcome: None,
            invalidate_cell=None,
            cell_conformance=lambda _cell: cleanup,
            cancellation_event=None,
            secret_values=(),
        )


def test_final_conformance_recomputes_physical_config_from_logical_input(
    tmp_path: Path,
) -> None:
    base = _cell(tmp_path, "task-config", "baseline")
    logical_config = tmp_path / "logical-config.json"
    logical_payload = {
        "job_name": "logical-job",
        "jobs_dir": (tmp_path / "logical-jobs").as_posix(),
        "agents": [{"env": {}}],
        "fugue": {"approved": True},
    }
    logical_config.write_text(json.dumps(logical_payload), encoding="utf-8")
    logical = replace(
        base,
        config_path=logical_config,
        config_sha256=hashlib.sha256(logical_config.read_bytes()).hexdigest(),
        command=("harbor", "jobs", "run", "--config", logical_config.as_posix()),
    )
    physical = PhysicalExecutionIdentityV1.create(
        logical_attempt_id=logical.attempt_id,
        controller_id="comparison-config",
        retry_ordinal=0,
    )
    _materialize_physical_harbor_cell(
        replace(
            logical,
            physical_execution_id=physical.physical_execution_id,
            retry_ordinal=0,
        ),
        physical=physical,
        repo_root=tmp_path,
    )
    restored = _physical_harbor_cell_from_journal(
        repo_root=tmp_path,
        logical_cell=logical,
        physical=physical,
    )
    assert restored.physical_execution_id == physical.physical_execution_id
    assert restored.command.count("--plugin") == 1
    assert (
        "fugue.bench.harbor_terminal:DurableHarborTerminalPlugin"
        in restored.command
    )
    assert "--upload" not in restored.command

    physical_config = restored.config_path
    tampered = json.loads(physical_config.read_text(encoding="utf-8"))
    tampered["unreviewed_runtime_change"] = True
    physical_config.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(Exception, match="physical Harbor config changed"):
        _physical_harbor_cell_from_journal(
            repo_root=tmp_path,
            logical_cell=logical,
            physical=physical,
        )
