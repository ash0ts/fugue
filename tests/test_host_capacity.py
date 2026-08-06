from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from fugue.bench import comparison as comparison_module
from fugue.bench.comparison import (
    ComparisonPreviewV1,
    _artifact_digest,
    _comparison_execution_schedule,
    _execution_schedule_config,
    execute_comparison_stage,
)
from fugue.bench.host_capacity import (
    HostCapacityObservationV1,
    HostCapacityReceiptV1,
    HostCapacityUnavailableError,
    verify_host_capacity_receipt,
)
from fugue.bench.operator import PreviewCellSummary, PreviewSummary


def _observation(
    *, cpu: int, memory_gib: float, disk_gib: float
) -> HostCapacityObservationV1:
    return HostCapacityObservationV1(
        cpu_count=cpu,
        available_memory_gib=memory_gib,
        free_disk_gib=disk_gib,
    )


def _matrix() -> PreviewSummary:
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
            (
                ("target", "baseline"),
                ("target", "candidate"),
                ("control", "baseline"),
                ("control", "candidate"),
            ),
            start=1,
        )
    )
    return PreviewSummary(
        cells=4,
        applicable_cells=4,
        estimated_trials=4,
        harnesses=("claude-code",),
        variants=("baseline", "candidate"),
        systems=("none",),
        workloads=("harbor",),
        commands=(),
        matrix_cells=cells,
    )


def _schedule_spec() -> SimpleNamespace:
    schedule = _execution_schedule_config(
        {
            "stages": [
                {
                    "id": "checkpoint",
                    "task_ids": ["target"],
                    "pair_complete": True,
                },
                {"id": "development", "task_ids": ["control"]},
            ],
            "worker_limit": 2,
            "wave_size": 4,
            "infrastructure_retry_limit": 1,
            "maximum_physical_executions": 5,
            "maximum_in_flight_cost_usd": 5.0,
            "coordination": {
                "group_id": "capacity-test",
                "worker_limit": 3,
                "maximum_physical_executions": 96,
                "total_cost_usd": 270,
                "maximum_in_flight_cost_usd": 7.5,
            },
        },
        concurrency=2,
        maximum_cost_usd=12.5,
    )
    return SimpleNamespace(
        evaluators=(),
        execution=SimpleNamespace(
            schedule=schedule,
            reserve_per_attempt_usd=2.5,
            max_cost_usd=12.5,
        ),
    )


def test_capacity_receipt_is_strict_and_digest_bound() -> None:
    receipt = HostCapacityReceiptV1.from_observation(
        _observation(cpu=12, memory_gib=24, disk_gib=45)
    )

    assert receipt.selected_worker_limit == 3
    assert HostCapacityReceiptV1.from_dict(receipt.to_dict()) == receipt
    with pytest.raises(ValueError, match="unknown host capacity receipt"):
        HostCapacityReceiptV1.from_dict({**receipt.to_dict(), "extra": True})
    with pytest.raises(ValueError, match="digest does not match"):
        HostCapacityReceiptV1.from_dict(
            {**receipt.to_dict(), "receipt_digest": "0" * 64}
        )


def test_low_capacity_falls_back_to_two_workers_and_changes_schedule_digest() -> None:
    high = HostCapacityReceiptV1.from_observation(
        _observation(cpu=12, memory_gib=24, disk_gib=45)
    )
    low = HostCapacityReceiptV1.from_observation(
        _observation(cpu=6, memory_gib=12, disk_gib=20)
    )
    spec = _schedule_spec()
    rows = (
        {"id": "target", "partition": "qualification"},
        {"id": "control", "partition": "qualification"},
    )

    high_schedule = _comparison_execution_schedule(
        spec,
        _matrix(),
        rows,
        host_capacity_receipt=high,
    )
    low_schedule = _comparison_execution_schedule(
        spec,
        _matrix(),
        rows,
        host_capacity_receipt=low,
    )

    assert high_schedule is not None and low_schedule is not None
    assert high_schedule.coordination is not None
    assert low_schedule.coordination is not None
    assert high_schedule.coordination["worker_limit"] == 3
    assert high_schedule.coordination["maximum_in_flight_cost_usd"] == 7.5
    assert low_schedule.coordination["worker_limit"] == 2
    assert low_schedule.coordination["maximum_in_flight_cost_usd"] == 5.0
    assert high_schedule.schedule_digest != low_schedule.schedule_digest
    assert high.receipt_digest != low.receipt_digest
    high_preview = ComparisonPreviewV1(
        schema_version=2,
        comparison={"id": "capacity-test"},
        readiness={"status": "ready"},
        matrix={"estimated_trials": 4},
        experiment={},
        manifest={},
        execution_schedule=high_schedule.to_dict(),
        host_capacity_receipt=high.to_dict(),
    )
    low_preview = replace(
        high_preview,
        execution_schedule=low_schedule.to_dict(),
        host_capacity_receipt=low.to_dict(),
    )
    assert _artifact_digest(
        high_preview.to_dict(), "preview_digest"
    ) != _artifact_digest(low_preview.to_dict(), "preview_digest")


def test_execution_capacity_degradation_stops_before_preview_or_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    approved_capacity = HostCapacityReceiptV1.from_observation(
        _observation(cpu=12, memory_gib=24, disk_gib=45)
    )
    schedule = _comparison_execution_schedule(
        _schedule_spec(),
        _matrix(),
        (
            {"id": "target", "partition": "qualification"},
            {"id": "control", "partition": "qualification"},
        ),
        host_capacity_receipt=approved_capacity,
    )
    assert schedule is not None
    draft = ComparisonPreviewV1(
        schema_version=2,
        comparison={"schema_version": 3, "id": "capacity-test"},
        readiness={"status": "ready"},
        matrix={"estimated_trials": 4},
        experiment={},
        manifest={},
        execution_schedule=schedule.to_dict(),
        host_capacity_receipt=approved_capacity.to_dict(),
    )
    preview = replace(
        draft,
        preview_digest=_artifact_digest(draft.to_dict(), "preview_digest"),
    )
    calls = {"preview": 0, "execute": 0}

    class FakeOperatorService:
        def __init__(self, _root: Path, _env_file: Path | None = None) -> None:
            self.env = {}

        def execute_run(self, *_args: object, **_kwargs: object) -> None:
            calls["execute"] += 1

    def unexpected_preview(*_args: object, **_kwargs: object) -> None:
        calls["preview"] += 1
        raise AssertionError("preview recomputation must follow the capacity gate")

    monkeypatch.setattr(
        comparison_module,
        "comparison_from_dict",
        lambda *_args, **_kwargs: SimpleNamespace(id="capacity-test"),
    )
    monkeypatch.setattr(comparison_module, "OperatorService", FakeOperatorService)
    monkeypatch.setattr(comparison_module, "preview_comparison", unexpected_preview)

    with pytest.raises(
        HostCapacityUnavailableError,
        match="no cells were admitted",
    ):
        execute_comparison_stage(
            preview,
            stage_id="checkpoint",
            approval_digest="a" * 64,
            repo_root=tmp_path,
            host_capacity_probe=lambda _root: _observation(
                cpu=7,
                memory_gib=24,
                disk_gib=45,
            ),
        )

    assert calls == {"preview": 0, "execute": 0}


def test_capacity_recheck_accepts_improvement_without_changing_approved_tier(
    tmp_path: Path,
) -> None:
    approved = HostCapacityReceiptV1.from_observation(
        _observation(cpu=6, memory_gib=12, disk_gib=20)
    )

    observed = verify_host_capacity_receipt(
        approved,
        tmp_path,
        probe=lambda _root: _observation(cpu=12, memory_gib=24, disk_gib=45),
    )

    assert approved.selected_worker_limit == 2
    assert observed.cpu_count == 12
