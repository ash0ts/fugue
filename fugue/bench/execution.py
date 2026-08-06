from __future__ import annotations

import hashlib
import json
import math
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from filelock import FileLock

from fugue.bench.candidates import attempt_id, attempt_identity, stable_digest
from fugue.bench.files import atomic_write_json, latest_jsonl_records
from fugue.bench.files import terminate_process_group as _terminate_process_group
from fugue.bench.sandbox_policy import verify_harbor_job_attestation
from fugue.redaction import redact_text, secrets_from_env

if TYPE_CHECKING:
    from fugue.bench.job_config import RenderedJob

CellStatus = Literal[
    "pending",
    "running",
    "passed",
    "failed",
    "not_applicable",
    "cancelled",
    "interrupted",
]
EventCallback = Callable[[dict[str, Any]], None]
ExecutionKind = Literal["agent", "provider_diagnostic"]
BenchmarkOutcome = Literal["passed", "failed", "unscored", "not_applicable"]
RuntimeOutcome = Literal[
    "completed",
    "timed_out",
    "cancelled",
    "not_started",
    "not_applicable",
]
CellStartedCallback = Callable[["PlannedCell"], Mapping[str, str] | None]
CellFinishedCallback = Callable[["PlannedCell", "CellOutcome"], None]
PhysicalStartedCallback = Callable[
    ["PlannedCell", "PhysicalExecutionIdentityV1"], Mapping[str, str] | None
]
PhysicalFinishedCallback = Callable[
    ["PlannedCell", "PhysicalExecutionIdentityV1", "CellOutcome"], None
]


@dataclass(frozen=True)
class ExecutionScheduleV1:
    """Digest-bound admission order for one immutable logical matrix."""

    schema_version: Literal[1]
    stages: dict[str, tuple[str, ...]]
    stage_cost_usd: dict[str, float]
    per_execution_cost_usd: dict[str, float]
    stage_admission: dict[str, dict[str, Any]]
    worker_limit: int
    wave_size: int
    maximum_physical_executions: int
    infrastructure_retry_limit: int
    total_cost_usd: float
    maximum_in_flight_cost_usd: float
    coordination: dict[str, Any] | None = None
    schedule_digest: str = ""

    def __post_init__(self) -> None:  # noqa: C901 - one strict digest boundary
        if self.schema_version != 1:
            raise ValueError("unsupported execution schedule schema")
        if not self.stages:
            raise ValueError("execution schedule requires at least one stage")
        logical: list[str] = []
        for stage_id, attempt_ids in self.stages.items():
            if not stage_id or not attempt_ids:
                raise ValueError("execution schedule stages must be non-empty")
            logical.extend(attempt_ids)
        if len(set(logical)) != len(logical):
            raise ValueError("logical attempts may appear in only one stage")
        if set(self.stage_cost_usd) != set(self.stages):
            raise ValueError("execution schedule stage costs are incomplete")
        if set(self.per_execution_cost_usd) != {"agent", "judge", "total"}:
            raise ValueError("execution schedule cost components are incomplete")
        if any(
            not math.isfinite(value) or value < 0
            for value in self.per_execution_cost_usd.values()
        ) or not math.isclose(
            self.per_execution_cost_usd["total"],
            self.per_execution_cost_usd["agent"]
            + self.per_execution_cost_usd["judge"],
        ):
            raise ValueError("execution schedule cost components do not reconcile")
        if set(self.stage_admission) != set(self.stages):
            raise ValueError("execution schedule stage admission is incomplete")
        if any(
            not math.isfinite(value) or value < 0
            for value in self.stage_cost_usd.values()
        ):
            raise ValueError("execution schedule stage costs must be non-negative")
        for stage_id, attempt_ids in self.stages.items():
            expected_cost = len(attempt_ids) * self.per_execution_cost_usd["total"]
            if not math.isclose(self.stage_cost_usd[stage_id], expected_cost):
                raise ValueError(
                    f"execution stage {stage_id} cost does not match its cells"
                )
        for stage_id, raw_policy in self.stage_admission.items():
            if set(raw_policy) != {"worker_limit", "wave_size", "pair_complete"}:
                raise ValueError(
                    f"execution stage {stage_id} has an invalid admission policy"
                )
            stage_workers = raw_policy.get("worker_limit")
            stage_wave = raw_policy.get("wave_size")
            pair_complete = raw_policy.get("pair_complete")
            if (
                not isinstance(stage_workers, int)
                or isinstance(stage_workers, bool)
                or stage_workers < 1
                or not isinstance(stage_wave, int)
                or isinstance(stage_wave, bool)
                or stage_wave < stage_workers
                or not isinstance(pair_complete, bool)
            ):
                raise ValueError(
                    f"execution stage {stage_id} has an invalid admission policy"
                )
            if stage_workers > self.worker_limit or stage_wave > self.wave_size:
                raise ValueError(
                    f"execution stage {stage_id} exceeds the schedule admission ceiling"
                )
            if pair_complete and (stage_workers != 1 or stage_wave != 2):
                raise ValueError(
                    "pair-complete stages require one worker and two-cell waves"
                )
        if self.worker_limit < 1 or self.wave_size < 1:
            raise ValueError("execution workers and wave size must be positive")
        if self.maximum_physical_executions < len(logical):
            raise ValueError("physical execution ceiling is below the logical matrix")
        if self.infrastructure_retry_limit < 0:
            raise ValueError("infrastructure retry limit must be non-negative")
        for label, value in (
            ("total cost", self.total_cost_usd),
            ("in-flight cost", self.maximum_in_flight_cost_usd),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"execution {label} must be finite and non-negative")
        if self.maximum_in_flight_cost_usd > self.total_cost_usd:
            raise ValueError("in-flight cost ceiling exceeds total cost ceiling")
        if sum(self.stage_cost_usd.values()) > self.total_cost_usd + 1e-9:
            raise ValueError("execution stage costs exceed the total cost ceiling")
        if self.coordination is not None:
            expected = {
                "group_id",
                "worker_limit",
                "maximum_physical_executions",
                "total_cost_usd",
                "maximum_in_flight_cost_usd",
            }
            if set(self.coordination) != expected:
                raise ValueError("execution coordination policy is invalid")
            group_id = self.coordination.get("group_id")
            global_workers = self.coordination.get("worker_limit")
            global_physical = self.coordination.get("maximum_physical_executions")
            global_total = self.coordination.get("total_cost_usd")
            global_in_flight = self.coordination.get("maximum_in_flight_cost_usd")
            if (
                not isinstance(group_id, str)
                or not group_id
                or not all(char.isalnum() or char in "-_." for char in group_id)
                or not isinstance(global_workers, int)
                or isinstance(global_workers, bool)
                or global_workers < self.worker_limit
                or not isinstance(global_physical, int)
                or isinstance(global_physical, bool)
                or global_physical < len(logical)
            ):
                raise ValueError("execution coordination policy is invalid")
            for label, value in (
                ("total cost", global_total),
                ("in-flight cost", global_in_flight),
            ):
                if (
                    not isinstance(value, int | float)
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    or float(value) < 0
                ):
                    raise ValueError(f"coordination {label} is invalid")
            if (
                float(global_total) < self.total_cost_usd
                or float(global_in_flight) < self.maximum_in_flight_cost_usd
                or float(global_in_flight) > float(global_total)
            ):
                raise ValueError("coordination ceilings do not cover this schedule")
        digest = stable_digest(self._unsigned())
        if self.schedule_digest and self.schedule_digest != digest:
            raise ValueError("execution schedule digest does not match")
        object.__setattr__(self, "schedule_digest", digest)

    @classmethod
    def create(
        cls,
        *,
        stages: Mapping[str, Sequence[str]],
        stage_cost_usd: Mapping[str, float],
        per_execution_cost_usd: Mapping[str, float] | None = None,
        stage_admission: Mapping[str, Mapping[str, Any]] | None = None,
        worker_limit: int,
        wave_size: int,
        maximum_physical_executions: int,
        infrastructure_retry_limit: int,
        total_cost_usd: float,
        maximum_in_flight_cost_usd: float,
        coordination: Mapping[str, Any] | None = None,
    ) -> ExecutionScheduleV1:
        components = per_execution_cost_usd
        if components is None:
            inferred = {
                float(stage_cost_usd[stage_id]) / len(attempt_ids)
                for stage_id, attempt_ids in stages.items()
            }
            if len(inferred) != 1:
                raise ValueError("execution stage costs need explicit cost components")
            total = inferred.pop()
            components = {"agent": total, "judge": 0.0, "total": total}
        return cls(
            schema_version=1,
            stages={key: tuple(value) for key, value in stages.items()},
            stage_cost_usd={key: float(value) for key, value in stage_cost_usd.items()},
            per_execution_cost_usd={
                key: float(value)
                for key, value in (
                    components
                ).items()
            },
            stage_admission={
                key: {
                    "worker_limit": int(policy.get("worker_limit", worker_limit)),
                    "wave_size": int(policy.get("wave_size", wave_size)),
                    "pair_complete": bool(policy.get("pair_complete", False)),
                }
                for key, policy in (
                    stage_admission
                    or {
                        stage_id: {
                            "worker_limit": worker_limit,
                            "wave_size": wave_size,
                            "pair_complete": False,
                        }
                        for stage_id in stages
                    }
                ).items()
            },
            worker_limit=worker_limit,
            wave_size=wave_size,
            maximum_physical_executions=maximum_physical_executions,
            infrastructure_retry_limit=infrastructure_retry_limit,
            total_cost_usd=float(total_cost_usd),
            maximum_in_flight_cost_usd=float(maximum_in_flight_cost_usd),
            coordination=dict(coordination) if coordination is not None else None,
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ExecutionScheduleV1:
        known = {
            "schema_version",
            "stages",
            "stage_cost_usd",
            "per_execution_cost_usd",
            "stage_admission",
            "worker_limit",
            "wave_size",
            "maximum_physical_executions",
            "infrastructure_retry_limit",
            "total_cost_usd",
            "maximum_in_flight_cost_usd",
            "coordination",
            "schedule_digest",
        }
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError("unknown execution schedule field(s): " + ", ".join(unknown))
        raw_stages = raw.get("stages")
        if not isinstance(raw_stages, Mapping):
            raise ValueError("execution schedule stages must be an object")
        stages: dict[str, tuple[str, ...]] = {}
        for key, values in raw_stages.items():
            if not isinstance(values, list | tuple) or not all(
                isinstance(item, str) and item for item in values
            ):
                raise ValueError("execution schedule attempt ids must be strings")
            stages[str(key)] = tuple(values)
        raw_costs = raw.get("stage_cost_usd")
        if not isinstance(raw_costs, Mapping):
            raise ValueError("execution schedule stage costs must be an object")
        raw_components = raw.get("per_execution_cost_usd")
        if not isinstance(raw_components, Mapping):
            raise ValueError("execution schedule cost components must be an object")
        raw_admission = raw.get("stage_admission")
        if not isinstance(raw_admission, Mapping):
            raise ValueError("execution schedule stage admission must be an object")
        stage_admission: dict[str, dict[str, Any]] = {}
        for key, policy in raw_admission.items():
            if not isinstance(policy, Mapping):
                raise ValueError("execution schedule stage admission must be objects")
            stage_admission[str(key)] = dict(policy)
        return cls(
            schema_version=int(raw.get("schema_version") or 0),
            stages=stages,
            stage_cost_usd={
                str(key): float(value)
                for key, value in raw_costs.items()
            },
            per_execution_cost_usd={
                str(key): float(value) for key, value in raw_components.items()
            },
            stage_admission=stage_admission,
            worker_limit=int(raw.get("worker_limit") or 0),
            wave_size=int(raw.get("wave_size") or 0),
            maximum_physical_executions=int(
                raw.get("maximum_physical_executions") or 0
            ),
            infrastructure_retry_limit=int(
                raw.get("infrastructure_retry_limit") or 0
            ),
            total_cost_usd=float(raw.get("total_cost_usd") or 0),
            maximum_in_flight_cost_usd=float(
                raw.get("maximum_in_flight_cost_usd") or 0
            ),
            coordination=(
                dict(raw["coordination"])
                if isinstance(raw.get("coordination"), Mapping)
                else None
            ),
            schedule_digest=str(raw.get("schedule_digest") or ""),
        )

    @property
    def logical_attempt_ids(self) -> tuple[str, ...]:
        return tuple(item for values in self.stages.values() for item in values)

    def _unsigned(self) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "stages": {key: list(value) for key, value in self.stages.items()},
            "stage_cost_usd": dict(self.stage_cost_usd),
            "per_execution_cost_usd": dict(self.per_execution_cost_usd),
            "stage_admission": {
                key: dict(value) for key, value in self.stage_admission.items()
            },
            "worker_limit": self.worker_limit,
            "wave_size": self.wave_size,
            "maximum_physical_executions": self.maximum_physical_executions,
            "infrastructure_retry_limit": self.infrastructure_retry_limit,
            "total_cost_usd": self.total_cost_usd,
            "maximum_in_flight_cost_usd": self.maximum_in_flight_cost_usd,
        }
        if self.coordination is not None:
            value["coordination"] = dict(self.coordination)
        return value

    def to_dict(self) -> dict[str, Any]:
        return {**self._unsigned(), "schedule_digest": self.schedule_digest}


@dataclass(frozen=True)
class PhysicalExecutionIdentityV1:
    """One launch identity; retries never replace the logical attempt identity."""

    schema_version: Literal[1]
    logical_attempt_id: str
    run_id: str
    retry_ordinal: int
    physical_execution_id: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.retry_ordinal < 0:
            raise ValueError("invalid physical execution identity")
        digest = stable_digest(self._unsigned())
        if self.physical_execution_id and self.physical_execution_id != digest:
            raise ValueError("physical execution identity digest does not match")
        object.__setattr__(self, "physical_execution_id", digest)

    @classmethod
    def create(
        cls, *, logical_attempt_id: str, run_id: str, retry_ordinal: int
    ) -> PhysicalExecutionIdentityV1:
        return cls(1, logical_attempt_id, run_id, retry_ordinal)

    def _unsigned(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "logical_attempt_id": self.logical_attempt_id,
            "run_id": self.run_id,
            "retry_ordinal": self.retry_ordinal,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._unsigned(), "physical_execution_id": self.physical_execution_id}


@dataclass(frozen=True)
class StageSubsetReceiptV1:
    """Approval subject for a finite subset of one unchanged full preview."""

    schema_version: Literal[1]
    preview_digest: str
    schedule_digest: str
    stage_id: str
    attempt_ids: tuple[str, ...]
    maximum_cells: int
    maximum_cost_usd: float
    subset_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != 1 or not self.stage_id or not self.attempt_ids:
            raise ValueError("invalid stage subset receipt")
        if self.maximum_cells != len(self.attempt_ids):
            raise ValueError("stage cell ceiling must equal its logical subset")
        if not math.isfinite(self.maximum_cost_usd) or self.maximum_cost_usd < 0:
            raise ValueError("stage cost ceiling must be finite and non-negative")
        digest = stable_digest(self._unsigned())
        if self.subset_digest and self.subset_digest != digest:
            raise ValueError("stage subset digest does not match")
        object.__setattr__(self, "subset_digest", digest)

    @classmethod
    def create(
        cls,
        schedule: ExecutionScheduleV1,
        *,
        preview_digest: str,
        stage_id: str,
        maximum_cost_usd: float,
    ) -> StageSubsetReceiptV1:
        try:
            attempt_ids = schedule.stages[stage_id]
        except KeyError as exc:
            raise ValueError(f"unknown execution stage: {stage_id}") from exc
        return cls(
            1,
            preview_digest,
            schedule.schedule_digest,
            stage_id,
            attempt_ids,
            len(attempt_ids),
            maximum_cost_usd,
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> StageSubsetReceiptV1:
        known = {
            "schema_version",
            "preview_digest",
            "schedule_digest",
            "stage_id",
            "attempt_ids",
            "maximum_cells",
            "maximum_cost_usd",
            "subset_digest",
        }
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(
                "unknown stage subset receipt field(s): " + ", ".join(unknown)
            )
        attempt_ids = raw.get("attempt_ids")
        if not isinstance(attempt_ids, list | tuple) or not all(
            isinstance(item, str) and item for item in attempt_ids
        ):
            raise ValueError("stage subset attempt ids must be strings")
        return cls(
            schema_version=int(raw.get("schema_version") or 0),  # type: ignore[arg-type]
            preview_digest=str(raw.get("preview_digest") or ""),
            schedule_digest=str(raw.get("schedule_digest") or ""),
            stage_id=str(raw.get("stage_id") or ""),
            attempt_ids=tuple(attempt_ids),
            maximum_cells=int(raw.get("maximum_cells") or 0),
            maximum_cost_usd=float(raw.get("maximum_cost_usd") or 0),
            subset_digest=str(raw.get("subset_digest") or ""),
        )

    def _unsigned(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "preview_digest": self.preview_digest,
            "schedule_digest": self.schedule_digest,
            "stage_id": self.stage_id,
            "attempt_ids": list(self.attempt_ids),
            "maximum_cells": self.maximum_cells,
            "maximum_cost_usd": self.maximum_cost_usd,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._unsigned(), "subset_digest": self.subset_digest}


@dataclass(frozen=True)
class PlannedCell:
    id: str
    run_id: str
    run_name: str
    workload_id: str
    task_id: str
    harness: str
    context_system_id: str
    variant_id: str
    model_provider: str
    model: str
    trial_index: int
    comparison_example_id: str
    candidate_id: str
    execution_fingerprint: str
    config_path: Path
    result_path: Path
    command: tuple[str, ...]
    env: dict[str, str]
    n_attempts: int
    execution_kind: ExecutionKind = "agent"
    outer_wall_time_sec: int | None = None
    execution_limits_digest: str = ""
    context_delivery: str = "portable"
    expected_evidence_paths: tuple[str, ...] = ()
    evaluation_asset_lock_sha256: str = ""
    run_snapshot_sha256: str = ""
    source_commit: str = ""
    source_tree: str = ""
    source_dirty_digest: str = ""
    evaluation_case: dict[str, Any] | None = None
    evaluation_rubrics: tuple[dict[str, Any], ...] = ()
    scorer_hashes: dict[str, str] | None = None
    scorer_refs: tuple[str, ...] = ()
    applicable: bool = True
    skip_reason: str | None = None
    config_sha256: str = ""
    runtime_assets: tuple[tuple[str, str], ...] = ()
    approved_comparison: dict[str, Any] = field(default_factory=dict)
    integration_provenance: tuple[dict[str, Any], ...] = ()
    physical_execution_id: str = ""
    retry_ordinal: int = 0

    @property
    def attempt_identity(self) -> dict[str, Any]:
        return attempt_identity(
            task_id=self.task_id,
            arm=self.variant_id,
            harness=self.harness,
            attempt=self.trial_index,
            candidate=self.candidate_id,
            runtime=self.execution_fingerprint,
        )

    @property
    def attempt_id(self) -> str:
        return attempt_id(**self.attempt_identity)

    def record(self, status: CellStatus, **values: Any) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "cell_id": self.id,
            "attempt_id": self.attempt_id,
            "attempt_identity": self.attempt_identity,
            "run_id": self.run_id,
            "run_name": self.run_name,
            "workload_id": self.workload_id,
            "task_id": self.task_id,
            "harness": self.harness,
            "context_system_id": self.context_system_id,
            "context_delivery": self.context_delivery,
            "variant_id": self.variant_id,
            "model_provider": self.model_provider,
            "model": self.model,
            "trial_index": self.trial_index,
            "comparison_example_id": self.comparison_example_id,
            "candidate_id": self.candidate_id,
            "execution_fingerprint": self.execution_fingerprint,
            "execution_kind": self.execution_kind,
            "outer_wall_time_sec": self.outer_wall_time_sec,
            "execution_limits_digest": self.execution_limits_digest or None,
            "applicable": self.applicable,
            "config_path": self.config_path.as_posix(),
            "result_path": self.result_path.as_posix(),
            "command": list(self.command),
            "n_attempts": self.n_attempts,
            "status": status,
            "skip_reason": self.skip_reason,
            "config_sha256": self.config_sha256,
            "runtime_assets": [list(item) for item in self.runtime_assets],
            "run_snapshot_sha256": self.run_snapshot_sha256 or None,
            "source_commit": self.source_commit or None,
            "source_tree": self.source_tree or None,
            "source_dirty_digest": self.source_dirty_digest or None,
            "integration_provenance": list(self.integration_provenance),
            "physical_execution_id": self.physical_execution_id or None,
            "retry_ordinal": self.retry_ordinal,
            **(
                {"approved_comparison": self.approved_comparison}
                if self.approved_comparison
                else {}
            ),
            "recorded_at": datetime.now(UTC).isoformat(),
            **values,
        }


@dataclass(frozen=True)
class CellOutcome:
    cell_id: str
    status: CellStatus
    returncode: int | None = None
    error: str | None = None
    benchmark_outcome: BenchmarkOutcome = "unscored"
    reward: float | None = None
    runtime_outcome: RuntimeOutcome = "not_started"
    terminal_kind: str | None = None
    physical_execution_id: str = ""
    retry_ordinal: int = 0


@dataclass(frozen=True)
class _HarborJobResult:
    error: str | None
    benchmark_outcome: BenchmarkOutcome
    reward: float | None = None


class _CellWallTimeExceeded(RuntimeError):
    pass


class _RunnerStartFailure(RuntimeError):
    """The runner process was never created for this physical execution."""


class _RunnerPostStartFailure(RuntimeError):
    """The runner started, so replay is unsafe even if host bookkeeping failed."""


def new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{timestamp}-{uuid.uuid4().hex[:10]}"


def plan_cells(
    jobs: list[RenderedJob],
    *,
    run_id: str,
    run_name: str,
    scheduling_seed: str | None = None,
    verify_inputs: bool = True,
    approved_comparison: Mapping[str, Any] | None = None,
) -> list[PlannedCell]:
    cells: list[PlannedCell] = []
    for job in jobs:
        stable_attempt_id = attempt_id(
            task_id=job.task_id,
            arm=job.variant_id,
            harness=job.harness,
            attempt=job.trial_index,
            candidate=job.candidate_id,
            runtime=job.resolved_candidate.execution_fingerprint,
        )
        identity = ":".join(
            (
                run_id,
                job.workload_id,
                job.task_id,
                job.harness,
                job.context_system_id,
                job.variant_id,
                str(job.trial_index),
            )
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
        cells.append(
            PlannedCell(
                id=f"cell-{digest}",
                run_id=run_id,
                run_name=run_name,
                workload_id=job.workload_id,
                task_id=job.task_id,
                harness=job.harness,
                context_system_id=job.context_system_id,
                variant_id=job.variant_id,
                model_provider=job.route.provider,
                model=job.route.display_model,
                trial_index=job.trial_index,
                comparison_example_id=job.comparison_example_id,
                candidate_id=job.candidate_id,
                execution_fingerprint=job.resolved_candidate.execution_fingerprint,
                execution_kind=job.execution_kind,
                config_path=job.config_path,
                result_path=job.result_path,
                command=tuple(job.command),
                env={**job.env, "FUGUE_ATTEMPT_ID": stable_attempt_id},
                n_attempts=job.n_attempts,
                outer_wall_time_sec=job.outer_wall_time_sec,
                execution_limits_digest=job.execution_limits_digest,
                context_delivery=job.context_delivery,
                expected_evidence_paths=job.expected_evidence_paths,
                evaluation_case=job.evaluation_case,
                evaluation_rubrics=job.evaluation_rubrics,
                scorer_hashes=job.scorer_hashes,
                scorer_refs=job.scorer_refs,
                applicable=job.applicable,
                skip_reason=job.skip_reason,
                config_sha256=(
                    _path_digest(job.config_path) if verify_inputs else ""
                ),
                runtime_assets=(
                    tuple(
                        (path.as_posix(), _path_digest(path))
                        for path in job.generated_runtime_files
                    )
                    if verify_inputs
                    else ()
                ),
                approved_comparison=dict(approved_comparison or {}),
                integration_provenance=job.integration_provenance,
            )
        )
    return schedule_cells(cells, scheduling_seed)


def schedule_cells(
    cells: list[PlannedCell], scheduling_seed: str | None
) -> list[PlannedCell]:
    if not scheduling_seed:
        return cells
    return sorted(
        cells,
        key=lambda cell: hashlib.sha256(
            ":".join(
                (
                    scheduling_seed,
                    cell.workload_id,
                    cell.task_id,
                    cell.harness,
                    cell.context_system_id,
                    cell.variant_id,
                    str(cell.trial_index),
                )
            ).encode()
        ).hexdigest(),
    )


def execute_cells(  # noqa: C901 - lifecycle guards stay adjacent to execution
    cells: list[PlannedCell],
    *,
    repo_root: Path,
    max_workers: int,
    wave_size: int | None = None,
    runner: Callable[..., Any] | None = None,
    event_callback: EventCallback | None = None,
    cell_started: CellStartedCallback | None = None,
    require_cell_started_success: bool = False,
    cell_finished: CellFinishedCallback | None = None,
    physical_started: PhysicalStartedCallback | None = None,
    physical_finished: PhysicalFinishedCallback | None = None,
    infrastructure_retries: int = 0,
    resume: bool = False,
    abort_on_terminal_kinds: frozenset[str] = frozenset(),
    cancellation_event: threading.Event | None = None,
    cancellation_message: str = "Run cancelled by the operator.",
) -> list[CellOutcome]:
    if max_workers < 1:
        raise ValueError("cell concurrency must be positive")
    admission_wave_size = wave_size or max_workers
    if admission_wave_size < max_workers:
        raise ValueError("cell admission wave must cover cell concurrency")
    if infrastructure_retries < 0:
        raise ValueError("infrastructure retries must be non-negative")
    run_ids = {cell.run_id for cell in cells}
    if len(run_ids) > 1:
        raise ValueError("all cells in one execution must share a run_id")
    cell_ids = [cell.id for cell in cells]
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("cell ids must be unique within an execution")
    store = (
        _RunStore(repo_root / ".fugue" / "runtime" / cells[0].run_id, event_callback)
        if cells
        else None
    )
    runnable: list[PlannedCell] = []
    outcomes: list[CellOutcome] = []
    resume_retry_ordinal: dict[str, int] = {}
    resume_replaced_physical: dict[str, tuple[str, int]] = {}
    prior_by_cell = {
        str(item.get("cell_id")): item
        for item in (
            latest_cell_records(store.cells_path)
            if resume and store is not None and store.cells_path.exists()
            else []
        )
    }
    for cell in cells:
        assert store is not None
        prior = prior_by_cell.get(cell.id)
        if prior and prior.get("status") in {
            "passed",
            "failed",
            "not_applicable",
            "cancelled",
        }:
            outcomes.append(_outcome_from_record(prior))
            continue
        prior_physical_id = str((prior or {}).get("physical_execution_id") or "")
        if resume and prior_physical_id:
            if not _physical_resume_replacement_allowed(prior or {}):
                raise RuntimeError(
                    "resume is blocked because the prior physical execution "
                    f"for {cell.attempt_id} is not proven terminal and cleaned"
                )
            prior_retry_ordinal = int((prior or {}).get("retry_ordinal") or 0)
            resume_retry_ordinal[cell.id] = prior_retry_ordinal + 1
            resume_replaced_physical[cell.id] = (
                prior_physical_id,
                prior_retry_ordinal,
            )
        else:
            resume_retry_ordinal[cell.id] = 0
        store.append_cell(cell.record("pending"))
        store.append_event("cell_state", cell=cell, status="pending")
        if cell.applicable:
            runnable.append(cell)
        else:
            store.append_cell(
                cell.record(
                    "not_applicable",
                    benchmark_outcome="not_applicable",
                    runtime_outcome="not_applicable",
                )
            )
            store.append_event(
                "cell_state",
                cell=cell,
                status="not_applicable",
                message=cell.skip_reason,
            )
            outcomes.append(
                CellOutcome(
                    cell.id,
                    "not_applicable",
                    benchmark_outcome="not_applicable",
                    runtime_outcome="not_applicable",
                )
            )

    def run_once(  # noqa: C901 - one physical lifecycle must settle every exit
        cell: PlannedCell, retry_ordinal: int
    ) -> CellOutcome:
        assert store is not None
        physical = PhysicalExecutionIdentityV1.create(
            logical_attempt_id=cell.attempt_id,
            run_id=cell.run_id,
            retry_ordinal=retry_ordinal,
        )
        cell = replace(
            cell,
            physical_execution_id=physical.physical_execution_id,
            retry_ordinal=retry_ordinal,
        )
        if cancellation_event is not None and cancellation_event.is_set():
            outcome = CellOutcome(
                cell.id,
                "cancelled",
                error=cancellation_message,
                runtime_outcome="cancelled",
                terminal_kind="cancelled",
                physical_execution_id=physical.physical_execution_id,
                retry_ordinal=retry_ordinal,
            )
            ended = datetime.now(UTC)
            store.append_cell(
                cell.record(
                    "cancelled",
                    error=cancellation_message,
                    benchmark_outcome=outcome.benchmark_outcome,
                    runtime_outcome="cancelled",
                    ended_at=ended.isoformat(),
                )
            )
            store.append_event(
                "cell_state",
                cell=cell,
                status="cancelled",
                message=cancellation_message,
            )
            return outcome
        try:
            _verify_cell_inputs(cell, repo_root)
        except Exception as exc:
            error = f"immutable run input verification failed: {exc}"
            outcome = CellOutcome(cell.id, "failed", error=error)
            outcome = replace(
                outcome,
                terminal_kind="configuration_failure",
                physical_execution_id=physical.physical_execution_id,
                retry_ordinal=retry_ordinal,
            )
            store.append_cell(
                cell.record(
                    "failed",
                    error=error,
                    benchmark_outcome="unscored",
                    runtime_outcome="not_started",
                    terminal_kind=outcome.terminal_kind,
                    ended_at=datetime.now(UTC).isoformat(),
                )
            )
            store.append_event("cell_state", cell=cell, status="failed", message=error)
            return outcome
        store.append_cell(cell.record("running"))
        store.append_event("cell_state", cell=cell, status="running")
        started = datetime.now(UTC)
        physical_env: dict[str, str] = {
            "FUGUE_PHYSICAL_EXECUTION_ID": physical.physical_execution_id,
            "FUGUE_RETRY_ORDINAL": str(retry_ordinal),
        }
        physical_admitted = False
        physical_finalized = False

        def finalize_physical(outcome: CellOutcome) -> CellOutcome:
            """Finalize one admitted physical identity exactly once."""

            nonlocal physical_finalized
            if (
                not physical_admitted
                or physical_finalized
                or physical_finished is None
            ):
                return outcome
            # Mark before invoking the callback so a callback failure can never
            # cause a second settlement attempt for the same physical identity.
            physical_finalized = True
            try:
                physical_finished(cell, physical, outcome)
            except Exception as exc:
                return replace(
                    outcome,
                    status="failed",
                    error=(
                        "physical execution accounting failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    terminal_kind="configuration_failure",
                )
            return outcome

        if physical_started is not None:
            try:
                physical_env.update(physical_started(cell, physical) or {})
                physical_admitted = True
            except Exception as exc:
                paused = bool(getattr(exc, "admission_paused", False))
                error = f"physical execution admission failed: {type(exc).__name__}: {exc}"
                outcome = CellOutcome(
                    cell.id,
                    "interrupted" if paused else "failed",
                    error=error,
                    terminal_kind=(
                        "admission_paused" if paused else "configuration_failure"
                    ),
                    physical_execution_id=physical.physical_execution_id,
                    retry_ordinal=retry_ordinal,
                )
                store.append_cell(
                    cell.record(
                        "interrupted" if paused else "failed",
                        error=error,
                        benchmark_outcome="unscored",
                        runtime_outcome="not_started",
                        terminal_kind=outcome.terminal_kind,
                        started_at=started.isoformat(),
                        ended_at=datetime.now(UTC).isoformat(),
                    )
                )
                return outcome
        execution_env, cell_started_called, start_failure = (
            _initialize_cell_evidence(
                cell,
                store=store,
                started=started,
                cell_started=cell_started,
                require_success=require_cell_started_success,
                cancellation_event=cancellation_event,
            )
        )
        execution_env.update(physical_env)
        if start_failure is not None:
            return finalize_physical(
                replace(
                    start_failure,
                    physical_execution_id=physical.physical_execution_id,
                    retry_ordinal=retry_ordinal,
                )
            )
        if cancellation_event is not None and cancellation_event.is_set():
            outcome = CellOutcome(
                cell.id,
                "cancelled",
                error=cancellation_message,
                runtime_outcome="not_started",
                terminal_kind="prestart_cancelled",
                physical_execution_id=physical.physical_execution_id,
                retry_ordinal=retry_ordinal,
            )
            ended = datetime.now(UTC)
            if cell_started_called and cell_finished is not None:
                try:
                    cell_finished(cell, outcome)
                except Exception as exc:
                    store.append_event(
                        "observability_error",
                        cell=cell,
                        message=f"{type(exc).__name__}: {exc}",
                    )
            store.append_cell(
                cell.record(
                    "cancelled",
                    error=cancellation_message,
                    benchmark_outcome=outcome.benchmark_outcome,
                    runtime_outcome="not_started",
                    terminal_kind=outcome.terminal_kind,
                    started_at=started.isoformat(),
                    ended_at=ended.isoformat(),
                    wall_time_sec=(ended - started).total_seconds(),
                )
            )
            store.append_event(
                "cell_state",
                cell=cell,
                status="cancelled",
                message=cancellation_message,
                terminal_kind=outcome.terminal_kind,
                wall_time_sec=(ended - started).total_seconds(),
            )
            return finalize_physical(outcome)
        try:
            if runner is None:
                returncode = _run_cell_process(
                    cell,
                    repo_root,
                    store,
                    execution_env,
                    **(
                        {"cancellation_event": cancellation_event}
                        if cancellation_event is not None
                        else {}
                    ),
                )
            else:
                try:
                    result = runner(
                        list(cell.command),
                        check=False,
                        env=execution_env,
                        cwd=repo_root,
                    )
                except OSError as exc:
                    raise _RunnerStartFailure(str(exc)) from exc
                returncode = int(result.returncode)
            harbor_result = (
                _harbor_job_result(cell, repo_root)
                if runner is None and returncode == 0 and cell.execution_kind == "agent"
                else _HarborJobResult(None, "unscored")
            )
            trial_error = harbor_result.error
            cancellation_requested = bool(
                cancellation_event is not None and cancellation_event.is_set()
            )
            if cancellation_requested and (returncode != 0 or trial_error is not None):
                status: CellStatus = "cancelled"
                trial_error = cancellation_message
            else:
                status = (
                    "passed" if returncode == 0 and trial_error is None else "failed"
                )
            outcome = CellOutcome(
                cell.id,
                status,
                returncode=returncode,
                error=trial_error,
                benchmark_outcome=harbor_result.benchmark_outcome,
                reward=harbor_result.reward,
                runtime_outcome=(
                    "cancelled" if status == "cancelled" else "completed"
                ),
                terminal_kind=(
                    "cancelled"
                    if status == "cancelled"
                    else "success"
                    if status == "passed"
                    else "task_failure"
                ),
                physical_execution_id=physical.physical_execution_id,
                retry_ordinal=retry_ordinal,
            )
        except _CellWallTimeExceeded as exc:
            outcome = CellOutcome(
                cell.id,
                "failed",
                error=str(exc),
                benchmark_outcome="unscored",
                runtime_outcome="timed_out",
                terminal_kind="agent_timeout",
                physical_execution_id=physical.physical_execution_id,
                retry_ordinal=retry_ordinal,
            )
        except _RunnerStartFailure as exc:
            outcome = CellOutcome(
                cell.id,
                "failed",
                error=f"{type(exc).__name__}: {exc}",
                runtime_outcome="not_started",
                terminal_kind="runner_start_failure",
                physical_execution_id=physical.physical_execution_id,
                retry_ordinal=retry_ordinal,
            )
        except Exception as exc:
            outcome = CellOutcome(
                cell.id,
                "failed",
                error=f"{type(exc).__name__}: {exc}",
                runtime_outcome="cancelled"
                if isinstance(exc, _RunnerPostStartFailure)
                else "not_started",
                terminal_kind="configuration_failure",
                physical_execution_id=physical.physical_execution_id,
                retry_ordinal=retry_ordinal,
            )
        ended = datetime.now(UTC)
        if cell_finished is not None:
            try:
                cell_finished(cell, outcome)
            except Exception as exc:
                store.append_event(
                    "observability_error",
                    cell=cell,
                    message=f"{type(exc).__name__}: {exc}",
                )
                if require_cell_started_success:
                    outcome = replace(
                        outcome,
                        status="failed",
                        error=(
                            "required live-evidence finalization failed: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                        terminal_kind="evidence_failure",
                    )
        outcome = finalize_physical(outcome)
        store.append_cell(
            cell.record(
                outcome.status,
                returncode=outcome.returncode,
                error=outcome.error,
                benchmark_outcome=outcome.benchmark_outcome,
                reward=outcome.reward,
                runtime_outcome=outcome.runtime_outcome,
                terminal_kind=outcome.terminal_kind,
                started_at=started.isoformat(),
                ended_at=ended.isoformat(),
                wall_time_sec=(ended - started).total_seconds(),
            )
        )
        store.append_event(
            "cell_state",
            cell=cell,
            status=outcome.status,
            returncode=outcome.returncode,
            message=outcome.error,
            benchmark_outcome=outcome.benchmark_outcome,
            reward=outcome.reward,
            runtime_outcome=outcome.runtime_outcome,
            terminal_kind=outcome.terminal_kind,
            wall_time_sec=(ended - started).total_seconds(),
        )
        return outcome

    def run_one(cell: PlannedCell) -> CellOutcome:
        first_retry_ordinal = resume_retry_ordinal.get(cell.id, 0)
        replaced_physical = resume_replaced_physical.get(cell.id)
        if replaced_physical is not None:
            prior_physical_id, prior_retry_ordinal = replaced_physical
            replacement = PhysicalExecutionIdentityV1.create(
                logical_attempt_id=cell.attempt_id,
                run_id=cell.run_id,
                retry_ordinal=first_retry_ordinal,
            )
            assert store is not None
            store.append_event(
                "physical_execution_replaced",
                cell=cell,
                physical_execution_id=prior_physical_id,
                retry_ordinal=prior_retry_ordinal,
                replacement_physical_execution_id=(
                    replacement.physical_execution_id
                ),
                replacement_retry_ordinal=first_retry_ordinal,
                terminal_kind="controller_interrupted",
            )
        for retry_ordinal in range(
            first_retry_ordinal,
            first_retry_ordinal + infrastructure_retries + 1,
        ):
            outcome = run_once(cell, retry_ordinal)
            if outcome.terminal_kind != "runner_start_failure":
                return outcome
            if retry_ordinal < first_retry_ordinal + infrastructure_retries:
                assert store is not None
                store.append_event(
                    "physical_execution_replaced",
                    cell=cell,
                    physical_execution_id=outcome.physical_execution_id,
                    retry_ordinal=retry_ordinal,
                    terminal_kind=outcome.terminal_kind,
                )
        return outcome

    admitted = 0
    wave_index = 0
    abort_admission = False
    with ThreadPoolExecutor(max_workers=min(max_workers, len(runnable) or 1)) as pool:
        while admitted < len(runnable):
            wave = runnable[admitted : admitted + admission_wave_size]
            wave_index += 1
            admitted += len(wave)
            wave_position = 0
            while wave_position < len(wave):
                batch = wave[wave_position : wave_position + max_workers]
                futures = [pool.submit(run_one, cell) for cell in batch]
                batch_outcomes = [
                    future.result() for future in as_completed(futures)
                ]
                outcomes.extend(batch_outcomes)
                wave_position += len(batch)
                if any(
                    item.terminal_kind in abort_on_terminal_kinds
                    for item in batch_outcomes
                ):
                    abort_admission = True
                    break
            assert store is not None
            store.append_event(
                "admission_wave_complete",
                wave_index=wave_index,
                logical_attempt_ids=[cell.attempt_id for cell in wave[:wave_position]],
                logical_cell_count=wave_position,
                worker_limit=max_workers,
                wave_size=admission_wave_size,
                aborted=abort_admission,
            )
            if abort_admission and any(
                item.terminal_kind == "admission_paused" for item in batch_outcomes
            ):
                store.append_event(
                    "admission_paused",
                    logical_attempt_ids=[
                        cell.attempt_id
                        for cell in [*wave[wave_position:], *runnable[admitted:]]
                    ],
                    reason="authoritative cost evidence is incomplete",
                )
                break
            if abort_admission:
                for cell in [*wave[wave_position:], *runnable[admitted:]]:
                    outcome = CellOutcome(
                        cell.id,
                        "cancelled",
                        error="later admission aborted after a fatal execution gate",
                        runtime_outcome="not_started",
                        terminal_kind="admission_aborted",
                    )
                    store.append_cell(
                        cell.record(
                            "cancelled",
                            error=outcome.error,
                            benchmark_outcome="unscored",
                            runtime_outcome="not_started",
                            terminal_kind=outcome.terminal_kind,
                            ended_at=datetime.now(UTC).isoformat(),
                        )
                    )
                    outcomes.append(outcome)
                break
    return outcomes


def _physical_resume_replacement_allowed(record: Mapping[str, Any]) -> bool:
    """Require structured non-duplication proof before a resumed Agent launch."""

    # Admission paused before a budget lease or process launch is an exact
    # never-started receipt produced by this lifecycle.
    if (
        record.get("terminal_kind") == "admission_paused"
        and record.get("runtime_outcome") == "not_started"
    ):
        return True
    # A recovery implementation may retain the interrupted physical execution
    # and append an explicit terminal/cleanup attestation. Mere controller loss,
    # PID absence, elapsed time, or a stale active lease is never enough.
    actual_cost = record.get("physical_execution_actual_cost_usd")
    cost_valid = bool(
        isinstance(actual_cost, int | float)
        and not isinstance(actual_cost, bool)
        and math.isfinite(float(actual_cost))
        and float(actual_cost) >= 0
    )
    return bool(
        record.get("physical_execution_terminal_verified") is True
        and record.get("physical_execution_cleanup_verified") is True
        and record.get("status") == "interrupted"
        and record.get("runtime_outcome") in {"not_started", "cancelled", "completed"}
        and cost_valid
    )


def physical_resume_replacement_cost(record: Mapping[str, Any]) -> float:
    """Return authoritative prior spend or reject ambiguous replacement."""

    if not _physical_resume_replacement_allowed(record):
        raise ValueError(
            "prior physical execution is not proven terminal, cleaned, and costed"
        )
    if (
        record.get("terminal_kind") == "admission_paused"
        and record.get("runtime_outcome") == "not_started"
    ):
        return 0.0
    return float(record["physical_execution_actual_cost_usd"])


def _initialize_cell_evidence(
    cell: PlannedCell,
    *,
    store: _RunStore,
    started: datetime,
    cell_started: CellStartedCallback | None,
    require_success: bool,
    cancellation_event: threading.Event | None,
) -> tuple[dict[str, str], bool, CellOutcome | None]:
    execution_env = dict(cell.env)
    if cell_started is None or (
        cancellation_event is not None and cancellation_event.is_set()
    ):
        return execution_env, False, None
    try:
        execution_env.update(cell_started(cell) or {})
        return execution_env, True, None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        store.append_event("observability_error", cell=cell, message=error)
        if not require_success:
            return execution_env, True, None
        outcome = CellOutcome(
            cell.id,
            "failed",
            error=f"required live-evidence initialization failed: {error}",
            runtime_outcome="not_started",
            terminal_kind="evidence_initialization_failure",
        )
        ended = datetime.now(UTC)
        wall_time = (ended - started).total_seconds()
        store.append_cell(
            cell.record(
                "failed",
                error=outcome.error,
                benchmark_outcome="unscored",
                runtime_outcome="not_started",
                terminal_kind=outcome.terminal_kind,
                started_at=started.isoformat(),
                ended_at=ended.isoformat(),
                wall_time_sec=wall_time,
            )
        )
        store.append_event(
            "cell_state",
            cell=cell,
            status="failed",
            message=outcome.error,
            terminal_kind=outcome.terminal_kind,
            wall_time_sec=wall_time,
        )
        return execution_env, True, outcome


def _run_cell_process(
    cell: PlannedCell,
    repo_root: Path,
    store: _RunStore,
    env: Mapping[str, str],
    *,
    cancellation_event: threading.Event | None = None,
) -> int:
    if cell.execution_kind == "agent":
        verify_harbor_job_attestation(cell.config_path, repo_root)
    log_path = store.logs_dir / f"{cell.id}.log"
    try:
        process = subprocess.Popen(
            list(cell.command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=dict(env),
            cwd=repo_root,
            start_new_session=True,
        )
    except OSError as exc:
        raise _RunnerStartFailure(str(exc)) from exc
    try:
        return _monitor_started_cell_process(
            cell,
            store=store,
            env=env,
            process=process,
            log_path=log_path,
            cancellation_event=cancellation_event,
        )
    except BaseException as exc:
        _terminate_process_group(process)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:  # pragma: no cover - kill fallback
            process.kill()
            process.wait(timeout=2)
        if process.stdout is not None:
            process.stdout.close()
        if isinstance(exc, _CellWallTimeExceeded) or not isinstance(exc, Exception):
            raise
        raise _RunnerPostStartFailure(str(exc)) from exc


def _monitor_started_cell_process(
    cell: PlannedCell,
    *,
    store: _RunStore,
    env: Mapping[str, str],
    process: subprocess.Popen[str],
    log_path: Path,
    cancellation_event: threading.Event | None,
) -> int:
    store.append_cell(
        cell.record(
            "running",
            harbor_pid=process.pid,
            harbor_process_group=process.pid,
        )
    )
    secrets = secrets_from_env(env)
    reader_error: list[BaseException] = []
    deadline = (
        time.monotonic() + cell.outer_wall_time_sec
        if cell.outer_wall_time_sec is not None
        else None
    )
    timed_out = False

    with log_path.open("a") as log:
        assert process.stdout is not None
        stdout = process.stdout

        def drain_output() -> None:
            try:
                with stdout:
                    for line in stdout:
                        safe_line = redact_text(line, secrets)
                        log.write(safe_line)
                        log.flush()
                        print(safe_line, end="", flush=True)
                        store.append_event("log", cell=cell, chunk=safe_line)
            except BaseException as exc:  # pragma: no cover - defensive I/O guard
                reader_error.append(exc)

        reader = threading.Thread(
            target=drain_output,
            name=f"fugue-cell-log-{cell.id}",
            daemon=True,
        )
        reader.start()
        while process.poll() is None:
            if cancellation_event is not None and cancellation_event.wait(0.1):
                _terminate_process_group(process)
                break
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                _terminate_process_group(process)
                break
            time.sleep(0.05)
        returncode = process.wait()
        reader.join(timeout=2)
        if reader.is_alive():  # pragma: no cover - dead children should close the pipe
            stdout.close()
            reader.join(timeout=2)
    if reader_error:
        raise RuntimeError(f"cell log reader failed: {reader_error[0]}")
    if timed_out:
        raise _CellWallTimeExceeded(
            "outer cell wall-time limit exceeded after "
            f"{cell.outer_wall_time_sec} seconds"
        )
    return returncode


def _verify_cell_inputs(cell: PlannedCell, repo_root: Path) -> None:
    config_path = cell.config_path
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    if cell.config_sha256 and _path_digest(config_path) != cell.config_sha256:
        raise RuntimeError(f"config drift: {config_path}")
    for raw_path, expected in cell.runtime_assets:
        path = Path(raw_path)
        if not path.is_absolute():
            path = repo_root / path
        if _path_digest(path) != expected:
            raise RuntimeError(f"runtime asset drift: {path}")


def _path_digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError(f"cannot read immutable input {path}: {exc}") from exc


def _harbor_job_result(cell: PlannedCell, repo_root: Path) -> _HarborJobResult:
    path = cell.result_path
    if not path.is_absolute():
        path = repo_root / path
    try:
        result = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        return _HarborJobResult(
            f"Harbor did not produce a readable job result: {exc}", "unscored"
        )
    stats = result.get("stats") or {}
    errored = int(stats.get("n_errored_trials") or 0)
    cancelled = int(stats.get("n_cancelled_trials") or 0)
    if errored:
        return _HarborJobResult(f"{errored} Harbor trial(s) errored", "unscored")
    if cancelled:
        return _HarborJobResult(
            f"{cancelled} Harbor trial(s) were cancelled", "unscored"
        )
    rewards: list[float] = []
    for evaluation in (stats.get("evals") or {}).values():
        reward_buckets = ((evaluation or {}).get("reward_stats") or {}).get(
            "reward"
        ) or {}
        for raw_reward, trial_ids in reward_buckets.items():
            try:
                reward = float(raw_reward)
            except (TypeError, ValueError):
                return _HarborJobResult(
                    f"Harbor job result contains an invalid reward: {raw_reward!r}",
                    "unscored",
                )
            if not math.isfinite(reward):
                return _HarborJobResult(
                    f"Harbor job result contains a non-finite reward: {raw_reward!r}",
                    "unscored",
                )
            count = len(trial_ids) if isinstance(trial_ids, list) else 1
            rewards.extend([reward] * count)
    if not rewards:
        return _HarborJobResult(None, "unscored")
    if len(rewards) != 1:
        return _HarborJobResult(
            f"Harbor job result contains {len(rewards)} rewards for one cell",
            "unscored",
        )
    reward = rewards[0]
    return _HarborJobResult(
        None,
        "passed" if reward == 1.0 else "failed",
        reward,
    )


def update_run_manifest(
    repo_root: Path,
    run_id: str,
    updater: Callable[[dict[str, Any]], dict[str, Any]],
) -> Path:
    path = repo_root / ".fugue" / "runtime" / run_id / "run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(path.with_suffix(".lock").as_posix()):
        existing = read_run_manifest(path.parent) or {}
        values = updater(dict(existing))
        created_at = existing.get("created_at") or datetime.now(UTC).isoformat()
        atomic_write_json(
            path,
            {
                **existing,
                "schema_version": 1,
                "run_id": run_id,
                "created_at": created_at,
                "updated_at": datetime.now(UTC).isoformat(),
                **values,
            },
        )
    return path


def write_run_manifest(repo_root: Path, run_id: str, values: dict[str, Any]) -> Path:
    return update_run_manifest(repo_root, run_id, lambda _: values)


def read_run_manifest(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir if run_dir.name == "run.json" else run_dir / "run.json"
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def list_run_manifests(repo_root: Path) -> list[dict[str, Any]]:
    runtime = repo_root / ".fugue" / "runtime"
    if not runtime.exists():
        return []
    values = [
        value
        for path in runtime.glob("*/run.json")
        if (value := read_run_manifest(path)) is not None
    ]
    return sorted(
        values,
        key=lambda item: str(item.get("created_at") or item.get("run_id") or ""),
        reverse=True,
    )


def mark_unfinished_cells(
    run_dir: Path,
    status: Literal["failed", "cancelled", "interrupted"],
    *,
    message: str,
) -> None:
    state_path = run_dir / "cells.jsonl"
    latest = latest_cell_records(state_path)
    store = _RunStore(run_dir)
    for record in latest:
        if record.get("status") not in {"pending", "running"}:
            continue
        updated = {
            **record,
            "status": status,
            "error": message,
            "benchmark_outcome": "failed" if status == "failed" else "unscored",
            "recorded_at": datetime.now(UTC).isoformat(),
            "ended_at": datetime.now(UTC).isoformat(),
        }
        store.append_cell(updated)
        store.append_event(
            "cell_state",
            cell_id=str(record.get("cell_id") or ""),
            status=status,
            message=message,
        )


def latest_cell_records(path: Path) -> list[dict[str, Any]]:
    return latest_jsonl_records(path, "cell_id")


def _outcome_from_record(record: Mapping[str, Any]) -> CellOutcome:
    return CellOutcome(
        cell_id=str(record.get("cell_id") or ""),
        status=str(record.get("status") or "failed"),  # type: ignore[arg-type]
        returncode=(
            int(record["returncode"])
            if record.get("returncode") is not None
            else None
        ),
        error=str(record["error"]) if record.get("error") else None,
        benchmark_outcome=str(
            record.get("benchmark_outcome") or "unscored"
        ),  # type: ignore[arg-type]
        reward=(float(record["reward"]) if record.get("reward") is not None else None),
        runtime_outcome=str(
            record.get("runtime_outcome") or "not_started"
        ),  # type: ignore[arg-type]
        terminal_kind=(
            str(record["terminal_kind"]) if record.get("terminal_kind") else None
        ),
        physical_execution_id=str(record.get("physical_execution_id") or ""),
        retry_ordinal=int(record.get("retry_ordinal") or 0),
    )


class _RunStore:
    def __init__(
        self, run_dir: Path, event_callback: EventCallback | None = None
    ) -> None:
        self.run_dir = run_dir
        self.cells_path = run_dir / "cells.jsonl"
        self.events_path = run_dir / "events.jsonl"
        self.logs_dir = run_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._event_callback = event_callback
        self._lock = threading.Lock()
        self._cells_file_lock = FileLock(f"{self.cells_path}.lock")
        self._events_file_lock = FileLock(f"{self.events_path}.lock")

    def append_cell(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True, default=str) + "\n"
        with self._lock, self._cells_file_lock, self.cells_path.open("a") as handle:
            handle.write(line)
            handle.flush()

    def append_event(
        self,
        event: str,
        *,
        cell: PlannedCell | None = None,
        cell_id: str | None = None,
        **data: Any,
    ) -> None:
        record = {
            "schema_version": 1,
            "event_id": uuid.uuid4().hex,
            "event": event,
            "recorded_at": datetime.now(UTC).isoformat(),
            "run_id": cell.run_id if cell else self.run_dir.name,
            "cell_id": cell.id if cell else cell_id,
            **data,
        }
        line = json.dumps(record, sort_keys=True, default=str) + "\n"
        with self._lock, self._events_file_lock, self.events_path.open("a") as handle:
            handle.write(line)
            handle.flush()
        if self._event_callback is not None:
            self._event_callback(record)
