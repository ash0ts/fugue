from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fugue.bench.candidates import stable_digest
from fugue.bench.execution import CellOutcome, PlannedCell
from fugue.bench.execution_recovery import (
    AdmittedWaveLifecycle,
    CanonicalizationObservationV1,
    CleanupObservationV1,
    CostObservationV1,
    ExecutionRecoveryAdapters,
    ExecutionRecoveryController,
    ExecutionScheduleV1,
    InterruptedExecutionObservationV1,
    LogicalExecutionPlanV1,
    PhysicalExecutionIdentityV1,
    StageExecutionAuthorizationV1,
    execute_recoverable_cells,
)
from fugue.bench.files import atomic_write_json, latest_jsonl_records
from fugue.bench.harbor_outcome import (
    HARBOR_TERMINAL_CLASSIFIER_DIGEST,
    classify_harbor_terminal,
)
from fugue.bench.local_evidence import LocalEvidenceStore

_MICRO_USD = 1_000_000
_SCHEDULE_KIND = "comparison_execution_schedule"
_JOURNAL_NAME = "comparison-execution.jsonl"
_RECEIPT_ROOT = "comparison-execution-receipts"


@dataclass(frozen=True)
class ComparisonExecutionBindingV1:
    """The generic execution schedule frozen into one comparison preview."""

    schedule: ExecutionScheduleV1
    logical_cell_count: int
    non_applicable_attempt_ids: tuple[str, ...]
    checkpoint_cell_count: int
    wave_size: int
    binding_digest: str = ""

    def __post_init__(self) -> None:
        if self.logical_cell_count < 1:
            raise ValueError("comparison execution binding requires cells")
        if self.checkpoint_cell_count < 1:
            raise ValueError("comparison execution checkpoint must be nonempty")
        if self.wave_size < 1:
            raise ValueError("comparison execution wave size must be positive")
        if tuple(sorted(set(self.non_applicable_attempt_ids))) != (
            self.non_applicable_attempt_ids
        ):
            raise ValueError("non-applicable attempt ids must be sorted and unique")
        if self.logical_cell_count != (
            len(self.schedule.logical_attempts)
            + len(self.non_applicable_attempt_ids)
        ):
            raise ValueError("comparison execution binding cell count disagrees")
        unsigned = self._unsigned()
        digest = stable_digest(unsigned)
        if self.binding_digest and self.binding_digest != digest:
            raise ValueError("comparison execution binding digest does not match")
        object.__setattr__(self, "binding_digest", digest)

    def _unsigned(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": _SCHEDULE_KIND,
            "schedule": self.schedule.to_dict(),
            "schedule_digest": self.schedule.schedule_digest,
            "logical_cell_count": self.logical_cell_count,
            "maximum_physical_executions": (
                self.schedule.maximum_physical_executions
            ),
            "non_applicable_attempt_ids": list(
                self.non_applicable_attempt_ids
            ),
            "checkpoint_cell_count": self.checkpoint_cell_count,
            "wave_size": self.wave_size,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._unsigned(), "binding_digest": self.binding_digest}

    @classmethod
    def from_dict(
        cls, raw: Mapping[str, Any]
    ) -> ComparisonExecutionBindingV1:
        expected = {
            "schema_version",
            "kind",
            "schedule",
            "schedule_digest",
            "logical_cell_count",
            "maximum_physical_executions",
            "non_applicable_attempt_ids",
            "checkpoint_cell_count",
            "wave_size",
            "binding_digest",
        }
        if set(raw) != expected:
            raise ValueError("comparison execution binding fields are invalid")
        if raw.get("schema_version") != 1 or raw.get("kind") != _SCHEDULE_KIND:
            raise ValueError("unsupported comparison execution binding")
        schedule_raw = raw.get("schedule")
        if not isinstance(schedule_raw, Mapping):
            raise ValueError("comparison execution schedule must be an object")
        schedule = ExecutionScheduleV1.from_dict(schedule_raw)
        if raw.get("schedule_digest") != schedule.schedule_digest:
            raise ValueError("comparison execution schedule digest disagrees")
        if raw.get("maximum_physical_executions") != (
            schedule.maximum_physical_executions
        ):
            raise ValueError("comparison physical-execution ceiling disagrees")
        non_applicable = raw.get("non_applicable_attempt_ids")
        if not isinstance(non_applicable, list) or any(
            not isinstance(item, str) for item in non_applicable
        ):
            raise ValueError("non-applicable attempt ids must be a list")
        return cls(
            schedule=schedule,
            logical_cell_count=_strict_int(
                raw.get("logical_cell_count"), "logical cell count"
            ),
            non_applicable_attempt_ids=tuple(non_applicable),
            checkpoint_cell_count=_strict_int(
                raw.get("checkpoint_cell_count"), "checkpoint cell count"
            ),
            wave_size=_strict_int(raw.get("wave_size"), "wave size"),
            binding_digest=str(raw.get("binding_digest") or ""),
        )


def compile_comparison_execution_binding(
    *,
    comparison_id: str,
    expected_cells: Sequence[Mapping[str, Any]],
    concurrency: int,
    checkpoint_cells: int,
    maximum_cost_usd: float,
    reserve_per_attempt_usd: float,
    maximum_infrastructure_replacements: int,
) -> ComparisonExecutionBindingV1:
    """Compile checkpoint-first admission from the canonical preview cells."""

    if not expected_cells:
        raise ValueError("comparison execution schedule requires expected cells")
    if concurrency < 1:
        raise ValueError("comparison concurrency must be positive")
    if maximum_infrastructure_replacements < 0:
        raise ValueError("infrastructure replacement ceiling cannot be negative")
    applicable = [item for item in expected_cells if bool(item.get("applicable"))]
    if not applicable:
        raise ValueError("comparison execution schedule requires applicable cells")
    attempt_ids = [str(item.get("attempt_id") or "") for item in applicable]
    if len(set(attempt_ids)) != len(attempt_ids):
        raise ValueError("comparison execution schedule has duplicate attempts")
    contrast_groups: list[list[Mapping[str, Any]]] = []
    group_index: dict[tuple[str, str, int], int] = {}
    for cell in applicable:
        contrast = (
            str(cell.get("task_id") or ""),
            str(cell.get("harness") or ""),
            _strict_int(cell.get("trial_index"), "trial index"),
        )
        index = group_index.get(contrast)
        if index is None:
            index = len(contrast_groups)
            group_index[contrast] = index
            contrast_groups.append([])
        contrast_groups[index].append(cell)
    requested_checkpoint = max(1, checkpoint_cells)
    checkpoint_groups: list[list[Mapping[str, Any]]] = []
    checkpoint_count = 0
    for group in contrast_groups:
        checkpoint_groups.append(group)
        checkpoint_count += len(group)
        if checkpoint_count >= requested_checkpoint:
            break
    checkpoint_group_count = len(checkpoint_groups)
    ordered_groups = [
        *checkpoint_groups,
        *contrast_groups[checkpoint_group_count:],
    ]
    ordered_applicable = [cell for group in ordered_groups for cell in group]
    total_micro = max(1, _usd_to_micro(maximum_cost_usd))
    reserve_micro = _usd_to_micro(reserve_per_attempt_usd)
    if reserve_micro:
        costs = [reserve_micro] * len(ordered_applicable)
    else:
        quotient, remainder = divmod(total_micro, len(ordered_applicable))
        costs = [
            quotient + (1 if ordinal < remainder else 0)
            for ordinal in range(len(ordered_applicable))
        ]
    planned = sum(costs)
    if planned > total_micro:
        raise ValueError(
            "per-attempt reserves exceed the comparison maximum cost"
        )
    cost_by_attempt = {
        str(cell["attempt_id"]): cost_micro
        for cell, cost_micro in zip(ordered_applicable, costs, strict=True)
    }
    attempts: list[LogicalExecutionPlanV1] = []
    checkpoint_cells_ordered = [
        cell for group in checkpoint_groups for cell in group
    ]
    # Checkpoint comparisons are selected as complete task/attempt contrasts,
    # then admitted one physical execution at a time.  The next task cannot
    # begin until both arms of the preceding contrast have settled.
    for block_ordinal, cell in enumerate(checkpoint_cells_ordered):
        attempt = str(cell["attempt_id"])
        attempts.append(
            LogicalExecutionPlanV1.create(
                logical_attempt_id=attempt,
                stage_id="checkpoint",
                stage_ordinal=0,
                admission_block_id=f"checkpoint-{block_ordinal + 1:04d}",
                block_ordinal=block_ordinal,
                attempt_ordinal=0,
                admission_weight=1,
                maximum_cost_micro_usd=cost_by_attempt[attempt],
            )
        )
    # Main admission preserves each contrast as one bounded wave.  A contrast
    # larger than the host capacity is split into consecutive blocks, but is
    # never interleaved with another task.
    main_block_ordinal = 0
    for group in contrast_groups[checkpoint_group_count:]:
        for start in range(0, len(group), concurrency):
            block = group[start : start + concurrency]
            for attempt_ordinal, cell in enumerate(block):
                attempt = str(cell["attempt_id"])
                attempts.append(
                    LogicalExecutionPlanV1.create(
                        logical_attempt_id=attempt,
                        stage_id="main",
                        stage_ordinal=1,
                        admission_block_id=(
                            f"main-wave-{main_block_ordinal + 1:04d}"
                        ),
                        block_ordinal=main_block_ordinal,
                        attempt_ordinal=attempt_ordinal,
                        admission_weight=1,
                        maximum_cost_micro_usd=cost_by_attempt[attempt],
                    )
                )
            main_block_ordinal += 1
    # When every cell is a checkpoint there is one stage. Otherwise the
    # checkpoint and main stage ordinals above are contiguous by construction.
    schedule = ExecutionScheduleV1.create(
        schedule_id=f"comparison-{comparison_id}"[:200],
        logical_attempts=attempts,
        capacity_units=concurrency,
        maximum_in_flight_executions=concurrency,
        maximum_physical_executions=(
            len(attempts) + maximum_infrastructure_replacements
        ),
        maximum_infrastructure_replacements=(
            maximum_infrastructure_replacements
        ),
        maximum_total_micro_usd=total_micro,
        maximum_in_flight_micro_usd=max(
            1,
            min(
                total_micro,
                max(
                    max(costs[:checkpoint_count]),
                    sum(sorted(costs, reverse=True)[:concurrency]),
                ),
            ),
        ),
    )
    return ComparisonExecutionBindingV1(
        schedule=schedule,
        logical_cell_count=len(expected_cells),
        non_applicable_attempt_ids=tuple(
            sorted(
                str(item.get("attempt_id") or "")
                for item in expected_cells
                if not bool(item.get("applicable"))
            )
        ),
        checkpoint_cell_count=checkpoint_count,
        wave_size=concurrency,
    )


def verify_comparison_execution_binding(
    binding: Mapping[str, Any],
    *,
    expected_cells: Sequence[Mapping[str, Any]],
) -> ComparisonExecutionBindingV1:
    parsed = ComparisonExecutionBindingV1.from_dict(binding)
    expected_ids = {
        str(item.get("attempt_id") or "")
        for item in expected_cells
        if bool(item.get("applicable"))
    }
    scheduled_ids = {
        item.logical_attempt_id for item in parsed.schedule.logical_attempts
    }
    if scheduled_ids != expected_ids:
        raise ValueError("comparison execution schedule does not cover exact cells")
    non_applicable = tuple(
        sorted(
            str(item.get("attempt_id") or "")
            for item in expected_cells
            if not bool(item.get("applicable"))
        )
    )
    if parsed.non_applicable_attempt_ids != non_applicable:
        raise ValueError("comparison execution exclusions disagree with exact cells")
    if parsed.logical_cell_count != len(expected_cells):
        raise ValueError("comparison execution logical cell count drifted")
    return parsed


def execution_binding_from_approved(
    approved: Mapping[str, Any],
) -> ComparisonExecutionBindingV1:
    raw = approved.get("execution_schedule")
    if not isinstance(raw, Mapping):
        raise ValueError("approved comparison has no execution schedule")
    expected = approved.get("expected_cells")
    if not isinstance(expected, list) or any(
        not isinstance(item, Mapping) for item in expected
    ):
        raise ValueError("approved comparison expected cells are invalid")
    parsed = verify_comparison_execution_binding(raw, expected_cells=expected)
    if approved.get("execution_schedule_digest") != parsed.binding_digest:
        raise ValueError("approved comparison schedule binding digest disagrees")
    return parsed


def execution_stage_authorizations(
    binding: ComparisonExecutionBindingV1,
    *,
    preview_digest: str,
    approval_digest: str,
) -> tuple[StageExecutionAuthorizationV1, ...]:
    schedule = binding.schedule
    replacements = schedule.maximum_infrastructure_replacements
    stages: list[StageExecutionAuthorizationV1] = []
    for stage_id in dict.fromkeys(
        item.stage_id for item in schedule.logical_attempts
    ):
        plans = [
            item for item in schedule.logical_attempts if item.stage_id == stage_id
        ]
        stages.append(
            StageExecutionAuthorizationV1.create(
                preview_digest=preview_digest,
                approval_digest=approval_digest,
                schedule_digest=schedule.schedule_digest,
                stage_id=stage_id,
                maximum_logical_attempts=len(plans),
                maximum_physical_executions=len(plans) + replacements,
                # The global immutable schedule remains the hard spend cap.
                # A stage must be able to consume unused contingency for one
                # predeclared infrastructure replacement without minting a new
                # approval or silently expanding total spend.
                maximum_cost_micro_usd=schedule.maximum_total_micro_usd,
            )
        )
    return tuple(stages)


def recovery_journal_path(repo_root: Path, run_id: str) -> Path:
    return repo_root / ".fugue" / "runtime" / run_id / _JOURNAL_NAME


def authorize_comparison_execution(
    *,
    controller: ExecutionRecoveryController,
    binding: ComparisonExecutionBindingV1,
    preview_digest: str,
    approval_digest: str,
) -> tuple[StageExecutionAuthorizationV1, ...]:
    authorizations = execution_stage_authorizations(
        binding,
        preview_digest=preview_digest,
        approval_digest=approval_digest,
    )
    # Authorize the complete frozen schedule before the first paid admission.
    # The CLI still requires a current approval claim on every resume; journal
    # authorization prevents expansion but never bypasses ledger validity.
    for authorization in authorizations:
        controller.authorize_stage(authorization)
    return authorizations


def verify_resume_stage_authorizations(
    *,
    repo_root: Path,
    run_id: str,
    binding: ComparisonExecutionBindingV1,
    preview_digest: str,
    approval_digest: str,
) -> None:
    """Require a resume to inherit the exact pre-spend authorization set.

    A fresh, live approval is still claimed by the CLI before this check.  The
    immutable run continues to carry the receipt that originally authorized
    its input lock, however, so renewing operator consent cannot rewrite the
    accepted run or its journal.  A run that crashed before all stages were
    durably authorized has no paid work to recover and must be started again.
    """

    journal = recovery_journal_path(repo_root, run_id)
    if not journal.is_file() or journal.is_symlink():
        raise ValueError(
            "resume requires the existing immutable comparison execution journal"
        )
    controller = ExecutionRecoveryController(
        journal,
        controller_id=_controller_id(run_id),
        schedule=binding.schedule,
    )
    observed = controller.snapshot().stage_authorizations
    expected = {
        item.stage_id: item
        for item in execution_stage_authorizations(
            binding,
            preview_digest=preview_digest,
            approval_digest=approval_digest,
        )
    }
    if observed != expected:
        raise ValueError(
            "resume requires the complete exact stage authorization set "
            "written before the first paid admission"
        )


def execute_durable_comparison_cells(  # noqa: C901 - one recovery boundary
    *,
    repo_root: Path,
    run_id: str,
    cells: Sequence[PlannedCell],
    approved_comparison: Mapping[str, Any],
    runner: Callable[..., Any] | None,
    begin_cell: Callable[[PlannedCell], Mapping[str, str] | None],
    finish_cell: Callable[[PlannedCell, CellOutcome], None],
    invalidate_cell: Callable[[PlannedCell, CellOutcome], None] | None,
    cell_conformance: Callable[[PlannedCell], Mapping[str, Any]],
    cancellation_event: Any,
    secret_values: Sequence[str],
    finalize_wave: Callable[[tuple[CellOutcome, ...]], None] | None = None,
) -> tuple[list[CellOutcome], dict[str, Any]]:
    binding = execution_binding_from_approved(approved_comparison)
    applicable = [cell for cell in cells if cell.applicable]
    if {cell.attempt_id for cell in applicable} != {
        item.logical_attempt_id for item in binding.schedule.logical_attempts
    }:
        raise ValueError("materialized comparison cells disagree with schedule")
    controller = ExecutionRecoveryController(
        recovery_journal_path(repo_root, run_id),
        controller_id=_controller_id(run_id),
        schedule=binding.schedule,
        redaction_secrets=secret_values,
    )
    authorizations = authorize_comparison_execution(
        controller=controller,
        binding=binding,
        preview_digest=str(approved_comparison.get("preview_digest") or ""),
        approval_digest=str(
            approved_comparison.get("execution_authorization_digest")
            or approved_comparison.get("approval_digest")
            or ""
        ),
    )
    def cleanup(
        cell: PlannedCell,
        physical: PhysicalExecutionIdentityV1,
        outcome: CellOutcome,
    ) -> CleanupObservationV1:
        if outcome.terminal_kind == "runner_start_failure":
            config_path = _absolute(cell.config_path, repo_root)
            if not config_path.is_file() or config_path.is_symlink():
                raise ValueError("no-start physical config is unavailable")
            payload = {
                "schema_version": 1,
                "status": "passed",
                "run_id": run_id,
                "logical_attempt_id": cell.attempt_id,
                "physical_execution_id": physical.physical_execution_id,
                "scope": {
                    "kind": "verified_no_runner_start",
                    "config_sha256": hashlib.sha256(
                        config_path.read_bytes()
                    ).hexdigest(),
                },
                "docker_cleanup": {
                    "status": "not_applicable",
                    "reason": "Harbor process creation failed before resource launch",
                },
            }
            path = _receipt_path(repo_root, run_id, physical, "cleanup")
            atomic_write_json(path, payload)
            return CleanupObservationV1(
                verified=True,
                scope_verified=True,
                post_run_inventory=True,
                discovered_resources=(),
                inspected_resources=(),
                removed_resources=(),
                remaining_resources=(),
                receipt_reference=path.relative_to(repo_root).as_posix(),
            )
        receipt = dict(cell_conformance(cell))
        path = _receipt_path(repo_root, run_id, physical, "cleanup")
        atomic_write_json(path, receipt)
        cleanup_payload = receipt.get("docker_cleanup")
        cleanup_values = (
            dict(cleanup_payload)
            if isinstance(cleanup_payload, Mapping)
            else {}
        )
        resources = _cleanup_resources(cleanup_values)
        scope = cleanup_values.get("scope")
        scope_verified = bool(
            isinstance(scope, Mapping)
            and scope.get("kind") == "exact_physical_compose_projects"
            and scope.get("run_id") == run_id
            and isinstance(scope.get("compose_projects"), list)
            and scope.get("compose_projects")
        )
        passed = receipt.get("status") == "passed" and (
            cleanup_values.get("status") == "passed"
            and scope_verified
            and not resources
        )
        return CleanupObservationV1(
            verified=passed,
            scope_verified=scope_verified,
            post_run_inventory=cleanup_values.get("status") == "passed",
            discovered_resources=resources,
            inspected_resources=resources,
            removed_resources=() if resources else (),
            remaining_resources=resources if not passed else (),
            receipt_reference=path.relative_to(repo_root).as_posix(),
        )

    def cost(
        cell: PlannedCell,
        physical: PhysicalExecutionIdentityV1,
        outcome: CellOutcome,
    ) -> CostObservationV1:
        actual_micro, authoritative = _local_attempt_cost(
            LocalEvidenceStore(repo_root, run_id), cell.attempt_id
        )
        source = "local-attempt-usage"
        actual_cost_available = authoritative
        if not authoritative and outcome.terminal_kind in {
            "agent_timeout",
            "cancelled",
        }:
            logical = next(
                item
                for item in binding.schedule.logical_attempts
                if item.logical_attempt_id == cell.attempt_id
            )
            # A post-start behavioral terminal is never retried. When its
            # provider usage did not finalize, settle the durable lease at the
            # full pre-approved reserve so later admission cannot undercount
            # spend. This is an accounted upper bound, not an observed cost.
            actual_micro = logical.maximum_cost_micro_usd
            authoritative = True
            source = "preleased-behavioral-reserve-upper-bound"
        if not authoritative and outcome.terminal_kind in {
            "runner_start_failure",
            "sandbox_lost",
            "transport_interrupted",
        }:
            actual_micro, authoritative, source = _infrastructure_attempt_cost(
                repo_root=repo_root,
                run_id=run_id,
                cell=cell,
                physical=physical,
                terminal_kind=outcome.terminal_kind,
            )
            actual_cost_available = authoritative
        if not authoritative:
            source = f"{source}-unavailable"
        payload = {
            "schema_version": 1,
            "logical_attempt_id": cell.attempt_id,
            "physical_execution_id": physical.physical_execution_id,
            "retry_ordinal": physical.retry_ordinal,
            "actual_cost_micro_usd": actual_micro,
            "authoritative": authoritative,
            "actual_cost_available": actual_cost_available,
            "source": source,
        }
        path = _receipt_path(
            repo_root,
            run_id,
            physical,
            "cost-authoritative" if authoritative else "cost-unavailable",
        )
        atomic_write_json(path, payload)
        return CostObservationV1(
            actual_cost_micro_usd=actual_micro,
            authoritative=authoritative,
            source=str(payload["source"]),
            receipt_reference=path.relative_to(repo_root).as_posix(),
        )

    def canonical(
        cell: PlannedCell,
        physical: PhysicalExecutionIdentityV1,
        _outcome: CellOutcome,
    ) -> CanonicalizationObservationV1:
        record = LocalEvidenceStore(repo_root, run_id).read_attempt(cell.attempt_id)
        if record.integrity_status != "resolved":
            raise ValueError("local attempt evidence is not reconciled")
        path = _receipt_path(repo_root, run_id, physical, "canonical")
        payload = {
            "schema_version": 1,
            "attempt_id": cell.attempt_id,
            "physical_execution_id": physical.physical_execution_id,
            "record_digest": record.record_digest,
            "prediction_row_sha256": record.prediction_row_sha256,
        }
        atomic_write_json(path, payload)
        return CanonicalizationObservationV1(
            kind="local-evidence",
            status="verified",
            reference=path.relative_to(repo_root).as_posix(),
            evidence_digest=stable_digest(payload),
        )

    def interrupted(
        cell: PlannedCell,
        physical: PhysicalExecutionIdentityV1,
    ) -> InterruptedExecutionObservationV1 | None:
        terminal = _terminal_cell_record(repo_root, run_id, physical)
        store = LocalEvidenceStore(repo_root, run_id)
        observation = _physical_runner_terminal_observation(
            repo_root=repo_root,
            cell=cell,
            physical=physical,
        )
        recovered_outcome = observation[0] if observation is not None else None
        bound_result_path = observation[1] if observation is not None else None
        record = None
        try:
            record = store.read_attempt(cell.attempt_id)
        except FileNotFoundError:
            if observation is None:
                return None
            assert recovered_outcome is not None
            if recovered_outcome.terminal_kind in {
                "runner_start_failure",
                "sandbox_lost",
                "transport_interrupted",
            }:
                if invalidate_cell is not None:
                    invalidate_cell(cell, recovered_outcome)
            else:
                # The exact runner result and exit observation are immutable;
                # resume only host-side evaluation/evidence finalization.
                finish_cell(cell, recovered_outcome)
                try:
                    record = store.read_attempt(cell.attempt_id)
                except FileNotFoundError as exc:
                    raise ValueError(
                        "host-only finalization did not create local attempt evidence"
                    ) from exc
        else:
            if (
                recovered_outcome is not None
                and recovered_outcome.terminal_kind
                in {
                    "runner_start_failure",
                    "sandbox_lost",
                    "transport_interrupted",
                }
            ):
                raise ValueError(
                    "infrastructure terminal receipt conflicts with finalized "
                    "local attempt evidence"
                )
        if record is not None and record.integrity_status != "resolved":
            return None
        terminal_kind = str((terminal or {}).get("terminal_kind") or "")
        terminal_outcome: CellOutcome | None = None
        if terminal_kind in {
            "success",
            "task_failure",
            "agent_timeout",
            "cancelled",
        }:
            terminal_outcome = CellOutcome(
                cell_id=cell.id,
                status=str((terminal or {}).get("status") or "failed"),  # type: ignore[arg-type]
                returncode=(
                    int(terminal["returncode"])
                    if terminal and terminal.get("returncode") is not None
                    else None
                ),
                error=(
                    str(terminal["error"])
                    if terminal and terminal.get("error")
                    else None
                ),
                benchmark_outcome=str(
                    (terminal or {}).get("benchmark_outcome") or "unscored"
                ),  # type: ignore[arg-type]
                reward=(
                    float(terminal["reward"])
                    if terminal and terminal.get("reward") is not None
                    else None
                ),
                runtime_outcome=str(
                    (terminal or {}).get("runtime_outcome") or "completed"
                ),  # type: ignore[arg-type]
                terminal_kind=terminal_kind,  # type: ignore[arg-type]
            )
        if recovered_outcome is not None:
            if terminal_outcome is not None and _outcome_identity(
                terminal_outcome
            ) != _outcome_identity(recovered_outcome):
                raise ValueError(
                    "runner terminal receipt conflicts with the terminal cell record"
                )
            outcome = recovered_outcome
            terminal_kind = recovered_outcome.terminal_kind
        elif terminal_outcome is not None:
            outcome = terminal_outcome
        elif record is not None:
            prediction = next(
                (
                    node
                    for node in record.nodes
                    if node.kind == "prediction" and node.artifact is not None
                ),
                None,
            )
            if prediction is None or prediction.artifact is None:
                return None
            prediction_path = (
                LocalEvidenceStore(repo_root, run_id).root
                / prediction.artifact.path
            )
            try:
                row = json.loads(prediction_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
            if (
                not isinstance(row, Mapping)
                or stable_digest(row) != record.prediction_row_sha256
            ):
                return None
            terminal_kind = (
                "success" if record.terminal_status == "passed" else "task_failure"
            )
            outcome = CellOutcome(
                cell_id=cell.id,
                status=record.terminal_status,  # type: ignore[arg-type]
                returncode=0 if record.terminal_status == "passed" else 1,
                benchmark_outcome=str(
                    row.get("benchmark_outcome") or "unscored"
                ),  # type: ignore[arg-type]
                reward=(
                    float(row["reward"]) if row.get("reward") is not None else None
                ),
                runtime_outcome="completed",
                terminal_kind=terminal_kind,
            )
        else:
            return None
        if record is not None and record.terminal_status != outcome.status:
            raise ValueError(
                "runner terminal receipt conflicts with finalized local attempt status"
            )
        infrastructure = terminal_kind in {
            "runner_start_failure",
            "sandbox_lost",
            "transport_interrupted",
        }
        if not infrastructure and bound_result_path is None:
            bound_result_path = _absolute(cell.result_path, repo_root)
        if (
            not infrastructure
            and (
                bound_result_path is None
                or not bound_result_path.is_file()
                or bound_result_path.is_symlink()
            )
        ):
            return None
        bound_result_sha256 = (
            hashlib.sha256(bound_result_path.read_bytes()).hexdigest()
            if bound_result_path is not None and bound_result_path.is_file()
            else None
        )
        receipt_payload = {
            "schema_version": 1,
            "logical_attempt_id": cell.attempt_id,
            "physical_execution_id": physical.physical_execution_id,
            "terminal_record_sha256": (
                stable_digest(terminal) if terminal is not None else None
            ),
            "local_record_digest": record.record_digest if record else None,
            "bound_runner_result_reference": (
                bound_result_path.relative_to(repo_root).as_posix()
                if bound_result_path is not None
                else None
            ),
            "bound_runner_result_sha256": bound_result_sha256,
        }
        receipt_path = _receipt_path(
            repo_root, run_id, physical, "interrupted-reconciliation"
        )
        atomic_write_json(receipt_path, receipt_payload)
        return InterruptedExecutionObservationV1(
            terminal_kind=terminal_kind,  # type: ignore[arg-type]
            cell_outcome=outcome,
            cleanup=cleanup(cell, physical, outcome),
            cost=cost(cell, physical, outcome),
            reconciliation_receipt_reference=(
                receipt_path.relative_to(repo_root).as_posix()
            ),
            reconciliation_receipt_sha256=hashlib.sha256(
                receipt_path.read_bytes()
            ).hexdigest(),
            result_reference=(
                None
                if infrastructure or bound_result_path is None
                else bound_result_path.relative_to(repo_root).as_posix()
            ),
            result_sha256=(None if infrastructure else bound_result_sha256),
        )

    adapters = ExecutionRecoveryAdapters(
        contract_digest=binding.schedule.adapter_contract_digest,
        cleanup_verifier=cleanup,
        cost_resolver=cost,
        canonicalization_verifier=canonical,
        interrupted_reconciler=interrupted,
        redaction_secrets=tuple(secret_values),
    )

    def lifecycle(_wave: tuple[PlannedCell, ...]) -> AdmittedWaveLifecycle:
        return AdmittedWaveLifecycle(
            begin_cell=begin_cell,
            finish_cell=finish_cell,
            invalidate_cell=invalidate_cell,
            # The post-cell boundary runs only after canonical local evidence
            # exists for every completed member. Comparison checkpoint gates
            # belong here, never inside the best-effort host evaluator.
            finalize=finalize_wave or (lambda _outcomes: None),
        )

    outcomes: list[CellOutcome] = []
    for authorization in authorizations:
        outcomes.extend(
            execute_recoverable_cells(
                controller,
                applicable,
                stage_authorization=authorization,
                adapters=adapters,
                repo_root=repo_root,
                runner=runner,
                wave_lifecycle_factory=lifecycle,
                cancellation_event=cancellation_event,
            )
        )
    snapshot = controller.snapshot()
    return outcomes, {
        "schema_version": 1,
        "schedule_digest": binding.schedule.schedule_digest,
        "binding_digest": binding.binding_digest,
        "controller_id": _controller_id(run_id),
        "journal": recovery_journal_path(repo_root, run_id)
        .relative_to(repo_root)
        .as_posix(),
        "event_chain_digest": snapshot.event_chain_digest,
        "logical_attempt_count": snapshot.logical_attempt_count,
        "physical_execution_count": len(snapshot.physical_executions),
        "canonical_result_count": len(snapshot.canonical_results),
        "complete": snapshot.complete,
    }


def _terminal_cell_record(
    repo_root: Path,
    run_id: str,
    physical: PhysicalExecutionIdentityV1,
) -> dict[str, Any] | None:
    path = repo_root / ".fugue" / "runtime" / run_id / "cells.jsonl"
    if not path.is_file():
        return None
    records = latest_jsonl_records(path, "cell_id")
    matches = [
        value
        for value in records.values()
        if value.get("attempt_id") == physical.logical_attempt_id
        and value.get("physical_execution_id") == physical.physical_execution_id
        and value.get("status")
        in {"passed", "failed", "cancelled", "interrupted"}
    ]
    return matches[-1] if matches else None


def _outcome_identity(outcome: CellOutcome) -> tuple[object, ...]:
    return (
        outcome.status,
        outcome.benchmark_outcome,
        outcome.reward,
        outcome.runtime_outcome,
        outcome.terminal_kind,
    )


def _physical_runner_terminal_observation(
    *,
    repo_root: Path,
    cell: PlannedCell,
    physical: PhysicalExecutionIdentityV1,
) -> tuple[CellOutcome, Path | None] | None:
    parent = _physical_parent_runner_terminal_observation(
        repo_root=repo_root,
        cell=cell,
        physical=physical,
    )
    harbor = _physical_harbor_terminal_observation(
        repo_root=repo_root,
        cell=cell,
        physical=physical,
    )
    if parent is not None and harbor is not None:
        if _outcome_identity(parent[0]) != _outcome_identity(harbor[0]):
            raise ValueError(
                "parent and Harbor terminal receipts disagree on the cell outcome"
            )
    return parent or harbor


def _physical_parent_runner_terminal_observation(
    *,
    repo_root: Path,
    cell: PlannedCell,
    physical: PhysicalExecutionIdentityV1,
) -> tuple[CellOutcome, Path | None] | None:
    path = (
        repo_root
        / ".fugue"
        / "runtime"
        / cell.run_id
        / "physical-executions"
        / physical.physical_execution_id
        / "runner-terminal.json"
    )
    if not path.is_file() or path.is_symlink():
        return None
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(receipt, Mapping):
        return None
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    if (
        receipt.get("receipt_digest") != stable_digest(unsigned)
        or receipt.get("logical_attempt_id") != cell.attempt_id
        or receipt.get("physical_execution_id")
        != physical.physical_execution_id
        or receipt.get("retry_ordinal") != physical.retry_ordinal
        or receipt.get("config_sha256") != cell.config_sha256
    ):
        return None
    raw = receipt.get("cell_outcome")
    if not isinstance(raw, Mapping):
        return None
    try:
        outcome = CellOutcome(
            cell_id=str(raw["cell_id"]),
            status=str(raw["status"]),  # type: ignore[arg-type]
            returncode=(
                int(raw["returncode"])
                if raw.get("returncode") is not None
                else None
            ),
            error=str(raw["error"]) if raw.get("error") is not None else None,
            benchmark_outcome=str(raw["benchmark_outcome"]),  # type: ignore[arg-type]
            reward=float(raw["reward"]) if raw.get("reward") is not None else None,
            runtime_outcome=str(raw["runtime_outcome"]),  # type: ignore[arg-type]
            terminal_kind=str(raw["terminal_kind"]),  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError):
        return None
    result_reference = receipt.get("result_reference")
    result_sha256 = receipt.get("result_sha256")
    result_path = _absolute(cell.result_path, repo_root)
    if result_reference is None and result_sha256 is None:
        if outcome.terminal_kind == "runner_start_failure":
            return outcome, None
        if outcome.terminal_kind in {"agent_timeout", "cancelled"}:
            terminal_result = _resultless_behavioral_terminal(
                repo_root=repo_root,
                cell=cell,
                physical=physical,
                outcome=outcome,
                runner_receipt_path=path,
            )
            return outcome, terminal_result
        return None
    if not isinstance(result_reference, str) or not isinstance(result_sha256, str):
        return None
    try:
        expected_reference = result_path.resolve().relative_to(
            repo_root.resolve()
        ).as_posix()
    except ValueError:
        return None
    if result_reference != expected_reference:
        return None
    bound_path = result_path
    if not bound_path.is_file():
        bound_path = (
            repo_root
            / ".fugue"
            / "runtime"
            / cell.run_id
            / "physical-executions"
            / physical.physical_execution_id
            / result_path.name
        )
    if (
        not bound_path.is_file()
        or bound_path.is_symlink()
        or hashlib.sha256(bound_path.read_bytes()).hexdigest() != result_sha256
    ):
        return None
    return outcome, bound_path


def _resultless_behavioral_terminal(
    *,
    repo_root: Path,
    cell: PlannedCell,
    physical: PhysicalExecutionIdentityV1,
    outcome: CellOutcome,
    runner_receipt_path: Path,
) -> Path:
    """Materialize the exact frozen result for a resultless terminal.

    Agent timeouts and cancellations are behavioral observations and must not
    be retried as infrastructure. The parent runner receipt proves their typed
    terminal state even when Harbor has no JobResult, so recovery binds a
    small immutable result envelope at the already-frozen cell result path.
    """

    if outcome.terminal_kind not in {"agent_timeout", "cancelled"}:
        raise ValueError("resultless behavioral terminal kind is unsupported")
    physical_root = runner_receipt_path.parent.resolve()
    result_path = _absolute(cell.result_path, repo_root).resolve()
    try:
        result_path.relative_to(physical_root)
    except ValueError as exc:
        raise ValueError(
            "resultless behavioral result is outside its physical namespace"
        ) from exc
    unsigned = {
        "schema_version": 1,
        "kind": "fugue_resultless_behavioral_terminal",
        "logical_attempt_id": cell.attempt_id,
        "physical_execution_id": physical.physical_execution_id,
        "retry_ordinal": physical.retry_ordinal,
        "cell_id": cell.id,
        "terminal_kind": outcome.terminal_kind,
        "cell_outcome": {
            "cell_id": outcome.cell_id,
            "status": outcome.status,
            "returncode": outcome.returncode,
            "error": outcome.error,
            "benchmark_outcome": outcome.benchmark_outcome,
            "reward": outcome.reward,
            "runtime_outcome": outcome.runtime_outcome,
            "terminal_kind": outcome.terminal_kind,
        },
        "runner_terminal_receipt_sha256": hashlib.sha256(
            runner_receipt_path.read_bytes()
        ).hexdigest(),
    }
    payload = {**unsigned, "result_digest": stable_digest(unsigned)}
    if result_path.exists():
        if result_path.is_symlink():
            raise ValueError("resultless behavioral result cannot be a symlink")
        try:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                "resultless behavioral result is unreadable"
            ) from exc
        if existing != payload:
            raise ValueError("resultless behavioral result changed")
    else:
        atomic_write_json(result_path, payload, mode=0o600)
    return result_path


def _physical_harbor_terminal_observation(
    *,
    repo_root: Path,
    cell: PlannedCell,
    physical: PhysicalExecutionIdentityV1,
) -> tuple[CellOutcome, Path] | None:
    physical_root = (
        repo_root
        / ".fugue"
        / "runtime"
        / cell.run_id
        / "physical-executions"
        / physical.physical_execution_id
    )
    path = physical_root / "harbor-terminal.json"
    if not path.is_file() or path.is_symlink():
        return None
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(receipt, Mapping):
        return None
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    if (
        receipt.get("receipt_digest") != stable_digest(unsigned)
        or receipt.get("schema_version") != 1
        or receipt.get("kind") != "harbor_single_trial_terminal"
        or receipt.get("classifier_digest")
        != HARBOR_TERMINAL_CLASSIFIER_DIGEST
        or receipt.get("logical_attempt_id") != cell.attempt_id
        or receipt.get("physical_execution_id")
        != physical.physical_execution_id
        or receipt.get("retry_ordinal") != physical.retry_ordinal
        or receipt.get("config_sha256") != cell.config_sha256
    ):
        return None
    expected_config = _absolute(cell.config_path, repo_root).resolve()
    if receipt.get("config_path") != expected_config.as_posix():
        return None
    raw_snapshot = receipt.get("terminal_result_path")
    raw_trial = receipt.get("terminal_trial_path")
    if not isinstance(raw_snapshot, str) or not isinstance(raw_trial, str):
        return None
    snapshot = Path(raw_snapshot).resolve()
    trial_snapshot = Path(raw_trial).resolve()
    source_result = _absolute(cell.result_path, repo_root).resolve()
    try:
        snapshot.relative_to(physical_root.resolve())
        trial_snapshot.relative_to(physical_root.resolve())
        source_result.relative_to(physical_root.resolve())
    except ValueError:
        return None
    if receipt.get("source_result_path") != source_result.as_posix():
        return None
    if any(
        not candidate.is_file() or candidate.is_symlink()
        for candidate in (snapshot, trial_snapshot, source_result)
    ):
        return None
    terminal_result_sha256 = receipt.get("terminal_result_sha256")
    if (
        hashlib.sha256(snapshot.read_bytes()).hexdigest()
        != terminal_result_sha256
        or hashlib.sha256(trial_snapshot.read_bytes()).hexdigest()
        != receipt.get("terminal_trial_sha256")
    ):
        return None
    try:
        terminal_job = json.loads(snapshot.read_text(encoding="utf-8"))
        terminal_trial = json.loads(trial_snapshot.read_text(encoding="utf-8"))
        final_job = json.loads(source_result.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(terminal_job, Mapping)
        or not isinstance(terminal_trial, Mapping)
        or not isinstance(final_job, Mapping)
    ):
        return None
    stats = terminal_job.get("stats")
    if (
        int(terminal_job.get("n_total_trials") or 0) != 1
        or not isinstance(stats, Mapping)
        or int(stats.get("n_completed_trials") or 0) != 1
        or int(stats.get("n_running_trials") or 0) != 0
        or int(stats.get("n_pending_trials") or 0) != 0
        or not terminal_trial.get("finished_at")
        or str(terminal_trial.get("id") or "") != str(receipt.get("trial_id") or "")
        or str(terminal_trial.get("trial_name") or "")
        != str(receipt.get("trial_name") or "")
    ):
        return None
    raw = receipt.get("cell_outcome")
    if not isinstance(raw, Mapping):
        return None
    try:
        snapshot_outcome = classify_harbor_terminal(
            terminal_job,
            cell_id=cell.id,
            trial=terminal_trial,
        )
        final_outcome = classify_harbor_terminal(
            final_job,
            cell_id=cell.id,
            trial=terminal_trial,
        )
    except (TypeError, ValueError):
        return None
    if dict(raw) != snapshot_outcome or final_outcome != snapshot_outcome:
        return None
    try:
        outcome = CellOutcome(
            cell_id=str(snapshot_outcome["cell_id"]),
            status=str(snapshot_outcome["status"]),  # type: ignore[arg-type]
            returncode=(
                int(snapshot_outcome["returncode"])
                if snapshot_outcome.get("returncode") is not None
                else None
            ),
            error=(
                str(snapshot_outcome["error"])
                if snapshot_outcome.get("error") is not None
                else None
            ),
            benchmark_outcome=str(
                snapshot_outcome["benchmark_outcome"]
            ),  # type: ignore[arg-type]
            reward=(
                float(snapshot_outcome["reward"])
                if snapshot_outcome.get("reward") is not None
                else None
            ),
            runtime_outcome=str(
                snapshot_outcome["runtime_outcome"]
            ),  # type: ignore[arg-type]
            terminal_kind=str(
                snapshot_outcome["terminal_kind"]
            ),  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError):
        return None
    if outcome.cell_id != cell.id:
        return None
    # The snapshot proves Harbor observed terminal bytes before controller
    # loss. Recovery must nevertheless bind the frozen canonical result path;
    # returning the snapshot would fail the controller's exact-path check.
    return outcome, source_result


def _cleanup_resources(cleanup: Mapping[str, Any]) -> tuple[str, ...]:
    values: set[str] = set()
    for item in cleanup.get("matched_containers") or ():
        if isinstance(item, Mapping) and item.get("container_id"):
            values.add(f"container-{str(item['container_id'])}")
    for item in cleanup.get("matched_networks") or ():
        if isinstance(item, Mapping) and item.get("network_id"):
            values.add(f"network-{str(item['network_id'])}")
    return tuple(sorted(values))


def _local_attempt_cost(
    store: LocalEvidenceStore,
    attempt_id: str,
) -> tuple[int | None, bool]:
    """Resolve actual spend only from the canonical per-attempt receipt."""

    try:
        record = store.read_attempt(attempt_id)
    except FileNotFoundError:
        return None, False
    usage = record.receipts.get("usage")
    if not isinstance(usage, Mapping) or usage.get("status") != "passed":
        return None, False
    payload = usage.get("payload")
    if not isinstance(payload, Mapping):
        return None, False
    if payload.get("judge_cost_required") is True and (
        payload.get("judge_cost_complete") is not True
    ):
        if payload.get("judge_accounted_cost_complete") is not True:
            return None, False
    value = payload.get("cost_usd")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None, False
    if not math.isfinite(float(value)) or float(value) < 0:
        return None, False
    judge_cost = (
        0
        if payload.get("judge_cost_required") is not True
        else payload.get("judge_cost_usd", 0)
        if payload.get("judge_cost_complete") is True
        else payload.get("judge_accounted_cost_usd")
    )
    if isinstance(judge_cost, bool) or not isinstance(
        judge_cost, int | float
    ):
        return None, False
    total = float(value) + float(judge_cost)
    if not math.isfinite(total) or total < 0:
        return None, False
    return _usd_to_micro(total), True


def _infrastructure_attempt_cost(
    *,
    repo_root: Path,
    run_id: str,
    cell: PlannedCell,
    physical: PhysicalExecutionIdentityV1,
    terminal_kind: str,
) -> tuple[int | None, bool, str]:
    """Resolve typed infrastructure spend from that physical execution only."""

    if terminal_kind == "runner_start_failure":
        return 0, True, "typed-runner-start-receipt"
    if terminal_kind == "sandbox_lost":
        terminal = _physical_harbor_terminal_observation(
            repo_root=repo_root,
            cell=cell,
            physical=physical,
        )
        if terminal is not None and terminal[0].terminal_kind == "sandbox_lost":
            return 0, True, "typed-pre-agent-harbor-terminal"
    physical_root = (
        repo_root
        / ".fugue"
        / "runtime"
        / run_id
        / "physical-executions"
        / physical.physical_execution_id
        / "harbor"
    )
    config_path = physical_root / "config.json"
    if not config_path.is_file() or config_path.is_symlink():
        return None, False, "physical-harbor-result"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, False, "physical-harbor-result"
    namespace = (
        ((config.get("fugue") or {}).get("physical_execution") or {})
        if isinstance(config, Mapping)
        else {}
    )
    if (
        not isinstance(namespace, Mapping)
        or namespace.get("physical_execution_id")
        != physical.physical_execution_id
        or namespace.get("logical_attempt_id") != physical.logical_attempt_id
        or namespace.get("retry_ordinal") != physical.retry_ordinal
    ):
        return None, False, "physical-harbor-result"
    result_path = Path(str(namespace.get("harbor_result_path") or ""))
    try:
        result_path.resolve().relative_to(physical_root.resolve())
    except ValueError:
        return None, False, "physical-harbor-result"
    archived_path = physical_root.parent / result_path.name
    candidates = (result_path, archived_path)
    for path in candidates:
        if not path.is_file() or path.is_symlink():
            continue
        if path == archived_path:
            terminal_path = physical_root.parent / "terminal.json"
            expected_reference = path.relative_to(repo_root).as_posix()
            archive_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            bound = False
            if terminal_path.is_file() and not terminal_path.is_symlink():
                try:
                    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    terminal = None
                bound = bool(
                    isinstance(terminal, Mapping)
                    and terminal.get("physical_execution_id")
                    == physical.physical_execution_id
                    and terminal.get("source_result_reference") == expected_reference
                    and terminal.get("source_result_sha256") == archive_digest
                )
            if not bound:
                runner_path = physical_root.parent / "runner-terminal.json"
                try:
                    runner = json.loads(runner_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    runner = None
                runner_unsigned = (
                    {
                        key: value
                        for key, value in runner.items()
                        if key != "receipt_digest"
                    }
                    if isinstance(runner, Mapping)
                    else {}
                )
                try:
                    original_reference = result_path.relative_to(repo_root).as_posix()
                except ValueError:
                    original_reference = ""
                bound = bool(
                    isinstance(runner, Mapping)
                    and runner.get("receipt_digest")
                    == stable_digest(runner_unsigned)
                    and runner.get("logical_attempt_id")
                    == physical.logical_attempt_id
                    and runner.get("physical_execution_id")
                    == physical.physical_execution_id
                    and runner.get("retry_ordinal") == physical.retry_ordinal
                    and runner.get("config_sha256")
                    == hashlib.sha256(config_path.read_bytes()).hexdigest()
                    and runner.get("result_reference") == original_reference
                    and runner.get("result_sha256") == archive_digest
                )
            if not bound:
                continue
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stats = result.get("stats") if isinstance(result, Mapping) else None
        value = stats.get("cost_usd") if isinstance(stats, Mapping) else None
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        if not math.isfinite(float(value)) or float(value) < 0:
            continue
        return _usd_to_micro(float(value)), True, "physical-harbor-result"
    return None, False, "physical-harbor-result"


def _receipt_path(
    repo_root: Path,
    run_id: str,
    physical: PhysicalExecutionIdentityV1,
    kind: str,
) -> Path:
    return (
        repo_root
        / ".fugue"
        / "runtime"
        / run_id
        / _RECEIPT_ROOT
        / f"{physical.physical_execution_id}-{kind}.json"
    )


def _controller_id(run_id: str) -> str:
    return f"comparison-{run_id}"[:200]


def _absolute(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _usd_to_micro(value: float) -> int:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise ValueError("comparison cost must be finite")
    if float(value) < 0:
        raise ValueError("comparison cost cannot be negative")
    return int(round(float(value) * _MICRO_USD))


def _strict_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value
